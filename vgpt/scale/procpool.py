"""Assistant workers as operating-system processes: the way the service uses many cores.

Measured on the development laptop (docs/SCALING.md): with several model contexts inside ONE process, answers per second stop rising and fall under load - the part of an answer that is not the model (reading the
question's data, the typed engines, the answer text) is Python code, and threads of one process share one interpreter lock; spinning model threads then fight for the same cores.  The fix is the ordinary one: one
process per worker.  Each worker holds its own model context and its own sessions; the parent only moves requests and answers.

    pool = ProcessPool(cfg, workers=1, min_workers=1, max_workers=4)
    answer = pool.ask("alice", "how is my tyre pressure?", persona="driver")          # any number of threads may call this

Dispatch.  Jobs wait in one priority queue (driver, then service, then fleet / validation, then batch).  A person is kept on the same worker (their dialogue - "and the week before?" - lives there); a new person
goes to the idle worker with the fewest people.  A worker serves one question at a time.
Failure.  A worker that dies takes only its own questions with it: the job in flight is retried once on another worker, a replacement is started, and the other workers never notice (this is also the safety
argument of docs/SAFETY_CASE.md: the assistant's failures stay inside a process).
Scaling.  The controller in autoscale.py adds a worker when the wait grows and removes an idle one when the load is gone; the numbers it needs are `metrics()`.
The same methods as `ElasticModelPool` (`contexts`, `add_context`, `remove_context`, `metrics`) mean the controller does not know which of the two it is steering."""
import collections
import heapq
import itertools
import json
import multiprocessing as mp
import os
import sys
import threading
import time

from .autoscale import PRIORITY, Refused


class _Duplex:
    """A two-way message channel built from two operating-system pipes.  multiprocessing.Pipe() on Linux is a Unix-domain socket pair, and the product's offline guard (vgpt/offline.py) refuses every socket that is not a CAN socket - correctly: a worker process does not need one."""

    def __init__(self, rd, wr): self.rd, self.wr = rd, wr
    def send(self, obj): self.wr.send(obj)
    def recv(self): return self.rd.recv()
    def poll(self, timeout=0.0): return self.rd.poll(timeout)
    def fileno(self): return self.rd.fileno()
    def close(self):
        for c in (self.rd, self.wr):
            try: c.close()
            except OSError: pass


def make_channel(ctx):
    """-> (parent end, child end) of a two-way channel: two one-way operating-system pipes (Pipe(duplex=False) is os.pipe on Linux and a named pipe on Windows, never a socket)."""
    c_r, p_w = ctx.Pipe(duplex=False); p_r, c_w = ctx.Pipe(duplex=False)
    return _Duplex(p_r, p_w), _Duplex(c_r, c_w)


def _fake_main(conn, cfg):
    """A worker without a model, for testing the dispatcher: sleeps `fake` seconds per question, remembers each person's turns, and dies on the text 'die'."""
    conn.send(("ready", os.getpid())); turns = collections.Counter()
    while True:
        try: msg = conn.recv()
        except (EOFError, OSError): return
        if msg[0] == "stop": return
        if msg[0] != "ask": continue
        _, rid, session, text, persona, now = msg
        if text == "die": os._exit(3)
        time.sleep(float(cfg["fake"])); turns[session] += 1; conn.send(("ok", rid, {"answer": f"{text}", "session": session, "turn": turns[session], "pid": os.getpid(), "persona": persona}, float(cfg["fake"])))


def worker_main(conn, cfg):
    """Child process: build the model and the brains once, then answer requests one at a time."""
    if cfg.get("fake") is not None: return _fake_main(conn, cfg)
    os.environ.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    sys.path.insert(0, cfg["root"])
    try:
        from vgpt.product import make_meaning_brain
        from vgpt.vehicle import lake
        if cfg.get("db"): lake.adopt_path(cfg["db"])
        if cfg["backend"] in ("ort-int8", "ort-gpu"):
            from vgpt.meaning.ort_backend import OrtFrameModel
            model = OrtFrameModel(cfg["ort_dir"], threads=cfg["threads"], provider="gpu" if cfg["backend"] == "ort-gpu" else "cpu")
        else:
            from vgpt.meaning.interface import FrameModel
            model = FrameModel(cfg["adapter"], threads=cfg["threads"])
        brains = {}
        conn.send(("ready", os.getpid()))
    except Exception as e:
        conn.send(("failed", repr(e))); return
    while True:
        try: msg = conn.recv()
        except (EOFError, OSError): return
        kind = msg[0]
        if kind == "stop": return
        if kind == "ping": conn.send(("pong", msg[1])); continue
        if kind != "ask": continue
        _, rid, session, text, persona, now = msg; t0 = time.perf_counter()
        try:
            b = brains.get(session)
            if b is None: b = brains[session] = make_meaning_brain(cfg["db"], slm=model, threads=cfg["threads"])
            if now is not None: b.now = int(now)
            b.persona = persona; r = b.ask(text); r = json.loads(json.dumps(r, default=str)); conn.send(("ok", rid, r, time.perf_counter() - t0))
        except Exception as e: conn.send(("err", rid, f"{type(e).__name__}: {e}", time.perf_counter() - t0))


class _W:
    def __init__(self, idx, proc, conn): self.idx, self.proc, self.conn = idx, proc, conn; self.ready = False; self.busy = None; self.sessions = set(); self.draining = False; self.dead = False; self.n = 0; self.busy_s = 0.0; self.pid = None; self.started = time.monotonic()


class _Job:
    def __init__(self, prio, seq, session, text, persona, now): self.prio, self.seq, self.session, self.text, self.persona, self.now = prio, seq, session, text, persona, now; self.ev = threading.Event(); self.box = None; self.t0 = time.perf_counter(); self.tries = 0

    def __lt__(self, o): return (self.prio, self.seq) < (o.prio, o.seq)


class ProcessPool:
    def __init__(self, cfg, workers=1, min_workers=1, max_workers=4):
        self.cfg = cfg; self.ctx = mp.get_context("spawn"); self.cond = threading.Condition(); self.pending = []; self.seq = itertools.count(); self.workers = []; self.min_workers = max(1, int(min_workers)); self.max_workers = max(int(max_workers), int(workers))
        self.recent = collections.deque(maxlen=3000); self.shed_level = 0; self.reject_below = None; self.affinity = {}; self.restarts = 0; self.closed = False; self.n_done = 0; self.target = max(1, int(workers))
        first = [self._spawn() for _ in range(max(1, int(workers)))]
        for w in first:
            t0 = time.monotonic()
            while not w.ready and not w.dead and time.monotonic() - t0 < cfg.get("start_timeout", 300): time.sleep(0.05)
        if not any(w.ready for w in first): raise RuntimeError("no assistant worker could start: " + str(getattr(first[0], "error", "")))
        self._disp = threading.Thread(target=self._dispatch, daemon=True, name="pool-dispatch"); self._disp.start()

    # ---------------------------------------------------------------- workers
    def _spawn(self):
        parent, child = make_channel(self.ctx); p = self.ctx.Process(target=worker_main, args=(child, self.cfg), daemon=True, name=f"assistant-worker-{len(self.workers)}"); p.start(); child.close()
        w = _W(next(self.seq), p, parent)
        with self.cond: self.workers.append(w)
        threading.Thread(target=self._reader, args=(w,), daemon=True, name=f"worker-reader-{w.idx}").start(); return w

    def _reader(self, w):
        while True:
            try: m = w.conn.recv()
            except (EOFError, OSError): break
            k = m[0]
            if k == "ready":
                with self.cond: w.ready = True; w.pid = m[1]; self.cond.notify_all()
            elif k == "failed":
                w.error = m[1]; break
            elif k in ("ok", "err"):
                _, rid, payload, dt = m; job = w.busy
                with self.cond:
                    w.busy = None; w.n += 1; w.busy_s += dt
                    if job is not None:
                        job.box = ("ok", payload) if k == "ok" else ("err", payload); self.recent.append((time.monotonic(), time.perf_counter() - job.t0, job.prio)); self.n_done += 1; job.ev.set()
                    retire = w.draining; self.cond.notify_all()
                if retire:                                                                              # it was told to go while it was answering: it goes now
                    try: w.conn.send(("stop",))
                    except (OSError, ValueError): pass
        self._died(w)

    def _died(self, w):
        with self.cond:
            if w.dead: return
            w.dead = True; job = w.busy; w.busy = None
            if w in self.workers: self.workers.remove(w)
            for s in list(w.sessions): self.affinity.pop(s, None)
            if job is not None:                                                                         # the question in flight is tried once more on another worker
                if job.tries < 1 and not self.closed: job.tries += 1; heapq.heappush(self.pending, job)
                else: job.box = ("err", "the assistant worker ended while answering"); job.ev.set()
            need = (not self.closed) and not w.draining and len([x for x in self.workers if not x.dead and not x.draining]) < self.target
            self.cond.notify_all()
        if need:
            self.restarts += 1; self._spawn()

    def _dispatch(self):
        with self.cond:
            while not self.closed:
                sent = False
                for job in sorted(self.pending):
                    w = self._pick(job)
                    if w is None: continue
                    self.pending.remove(job); heapq.heapify(self.pending); w.busy = job; w.sessions.add(job.session); self.affinity[job.session] = w
                    try: w.conn.send(("ask", id(job), job.session, job.text, job.persona, job.now))
                    except (OSError, ValueError): w.busy = None; heapq.heappush(self.pending, job); continue
                    sent = True; break
                if not sent: self.cond.wait(0.2)

    def _pick(self, job):
        a = self.affinity.get(job.session)
        if a is not None and not a.dead and not a.draining:
            return a if (a.ready and a.busy is None) else None                                            # the person's dialogue lives there: wait for it
        idle = [w for w in self.workers if w.ready and w.busy is None and not w.draining and not w.dead]
        return min(idle, key=lambda w: (len(w.sessions), w.n)) if idle else None

    # ---------------------------------------------------------------- the questions
    def ask(self, session, text, persona="driver", timeout=300.0, now=None):
        prio = PRIORITY.get(persona, 1)
        if self.reject_below is not None and prio >= self.reject_below: raise Refused(f"the assistant is at its limit: only the driver's questions are answered for now ({persona} request refused)")
        job = _Job(prio, next(self.seq), session, text, persona, now)
        with self.cond: heapq.heappush(self.pending, job); self.cond.notify_all()
        if not job.ev.wait(timeout): raise TimeoutError("the assistant did not answer in time")
        kind, payload = job.box
        if kind == "err": raise RuntimeError(payload)
        return payload

    # ---------------------------------------------------------------- what the controller reads and does
    def contexts(self):
        with self.cond: return len([w for w in self.workers if not w.dead and not w.draining])

    def add_context(self):
        with self.cond:
            live = [w for w in self.workers if not w.dead and not w.draining]
            if len(live) >= self.max_workers or any(not w.ready for w in live): return False
            self.target = len(live) + 1
        self._spawn(); return True

    def remove_context(self):
        with self.cond:
            live = [w for w in self.workers if not w.dead and not w.draining]
            if len(live) <= self.min_workers: return False
            idle = [w for w in live if w.ready and w.busy is None]; ready = [w for w in live if w.ready]
            if not ready: return False
            w = min(idle or ready, key=lambda w: (w.busy is not None, len(w.sessions), w.n)); w.draining = True; self.target = len(live) - 1       # an idle one at once; under saturation a busy one finishes the question in hand first
            for s in list(w.sessions): self.affinity.pop(s, None)
            stop_now = w.busy is None
        if stop_now:
            try: w.conn.send(("stop",))
            except (OSError, ValueError): pass
        return True

    def metrics(self, window_s=10.0):
        now = time.monotonic()
        with self.cond:
            rec = [(t, w, p) for t, w, p in self.recent if now - t <= window_s]; live = [w for w in self.workers if not w.dead and not w.draining]
            ready = sum(1 for w in live if w.ready); fl = sum(1 for w in live if w.busy is not None); q = len(self.pending)
        v = sorted(x[1] for x in rec); k = len(v); d = sorted(x[1] for x in rec if x[2] == 0)
        return {"contexts": len(live), "ready": ready, "loading": len(live) - ready, "in_flight": fl, "queue": q, "done": self.n_done, "answers": k, "p50": v[k // 2] if k else 0.0, "p95": v[min(k - 1, int(k * 0.95))] if k else 0.0, "driver_p95": d[min(len(d) - 1, int(len(d) * 0.95))] if d else 0.0}

    def stats(self):
        with self.cond: return {"workers": len(self.workers), "answers": self.n_done, "restarts": self.restarts, "pids": [w.pid for w in self.workers], "busy_s": [round(w.busy_s, 2) for w in self.workers], "shed_level": self.shed_level}

    def kill_worker(self, i=0):
        """For the fault test: end a worker the hard way."""
        with self.cond: ws = [w for w in self.workers if w.ready and not w.dead]
        if ws: ws[min(i, len(ws) - 1)].proc.kill()

    def close(self):
        with self.cond: self.closed = True; ws = list(self.workers); self.cond.notify_all()
        for w in ws:
            try: w.conn.send(("stop",))
            except (OSError, ValueError): pass
        for w in ws:
            try: w.proc.join(5)
            except Exception: pass
            if w.proc.is_alive(): w.proc.kill()
