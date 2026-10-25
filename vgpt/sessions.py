"""Concurrent assistants over one vehicle database: a pool of model contexts and one session per person, with the same question-and-answer interface at every resource tier.

    pool = ModelPool(lambda: FrameModel(adapter, threads=t.model["threads"]), workers=t.model["workers"])      # tier: 1 context on the 2 GB board, 3 on a VM, N on a vehicle HPC
    hub  = SessionHub(db, pool, brain_factory, max_sessions=t.sessions)
    answer = hub.ask("driver-1", "how is my tyre pressure?")                                                   # any number of threads may call this

What runs in parallel and what does not:
  * the language model: each `ModelPool` worker owns one model context (llama.cpp contexts are not thread-safe); frame-parsing requests are served by the least busy worker, so `workers` questions can
    be in the model at once.  With one worker (the 2 GB board) they queue - the latency a person sees is then queue time + one parse.
  * everything after the frame (typed resolution, SQL, statistics, investigations) runs in the session's own thread on its own read-only database handle, so sessions overlap there (SQLite readers do not block
    each other, and the writer is another process or thread).
  * a session keeps its own dialogue state: "and the week before that" refers to that person's previous question, never to someone else's.
The assistant side never writes the database (read-only handles) and never touches the buses: the safety-critical functions of the vehicle stay isolated from it."""
import collections
import queue
import threading
import time


CURRENT = threading.local()                                  # the person's role while a session thread asks: the model pool reads it to put a driver's question before a batch job's


class Busy(Exception):
    """Every session slot is in use and none is idle long enough to be reclaimed."""


class ModelWorker(threading.Thread):
    def __init__(self, factory, name):
        super().__init__(daemon=True, name=name); self.factory = factory; self.q = queue.Queue(); self.model = None; self.ready = threading.Event(); self.error = None; self.n = 0; self.busy_s = 0.0; self.waiting = 0

    def run(self):
        try: self.model = self.factory()
        except Exception as e: self.error = e; self.ready.set(); return
        self.ready.set()
        while True:
            job = self.q.get()
            if job is None: return
            text, hint, prev, box, ev = job; t0 = time.perf_counter()
            try: box.append(self.model.parse(text, hint) if prev is None else self.model.parse(text, hint, prev=prev))
            except Exception as e: box.append(e)
            self.busy_s += time.perf_counter() - t0; self.n += 1; self.waiting -= 1; ev.set()


class ModelPool:
    """`parse(text, hint)` and `glossary` like a single model, backed by `workers` model contexts."""
    def __init__(self, factory, workers=1):
        self.workers = [ModelWorker(factory, f"model-{i}") for i in range(max(1, int(workers)))]
        for w in self.workers: w.start()
        for w in self.workers: w.ready.wait()
        bad = [w.error for w in self.workers if w.error]
        if bad: raise bad[0]
        self.glossary = getattr(self.workers[0].model, "glossary", None); self.onto = getattr(self.workers[0].model, "onto", None); self.lock = threading.Lock(); self.wait_s = collections.deque(maxlen=500)

    def parse(self, text, hint=None, prev=None):
        with self.lock:
            w = min(self.workers, key=lambda x: x.waiting); w.waiting += 1
        box, ev = [], threading.Event(); t0 = time.perf_counter(); w.q.put((text, hint, prev, box, ev)); ev.wait(); self.wait_s.append(time.perf_counter() - t0)
        if isinstance(box[0], Exception): raise box[0]
        return box[0]

    def stats(self):
        w = sorted(self.wait_s); n = len(w)
        return {"workers": len(self.workers), "parses": sum(x.n for x in self.workers), "busy_s": [round(x.busy_s, 2) for x in self.workers], "wait_p50_ms": round(1000 * w[n // 2], 1) if n else None, "wait_p95_ms": round(1000 * w[int(n * 0.95)], 1) if n else None}

    def close(self):
        for w in self.workers: w.q.put(None)


class _Session(threading.Thread):
    def __init__(self, sid, db, pool, brain_factory, now_fn):
        super().__init__(daemon=True, name=f"session-{sid}"); self.sid, self.db, self.pool, self.factory, self.now_fn = sid, db, pool, brain_factory, now_fn
        self.q = queue.Queue(); self.last = time.monotonic(); self.busy = False; self.ready = threading.Event(); self.error = None; self.n = 0

    def run(self):
        try: brain = self.factory(self.db, slm=self.pool)                                                         # the SQLite handle belongs to this thread
        except Exception as e: self.error = e; self.ready.set(); return
        self.ready.set()
        while True:
            job = self.q.get()
            if job is None:
                try: brain.c.close()
                except Exception: pass
                return
            text, persona, box, ev = job; self.busy = True
            try:
                if self.now_fn: brain.now = int(self.now_fn())
                brain.persona = persona; CURRENT.persona = persona; box.append(brain.ask(text))
            except Exception as e: box.append(e)
            self.busy = False; self.last = time.monotonic(); self.n += 1; ev.set()


class SessionHub:
    def __init__(self, db, pool, brain_factory, max_sessions=4, idle_s=900.0, now_fn=None):
        self.db, self.pool, self.factory, self.max, self.idle, self.now_fn = db, pool, brain_factory, int(max_sessions), float(idle_s), now_fn; self.sessions = {}; self.lock = threading.Lock(); self.lat = collections.deque(maxlen=2000)

    def _session(self, sid):
        with self.lock:
            s = self.sessions.get(sid)
            if s is not None: return s
            if len(self.sessions) >= self.max:
                idle = [x for x in self.sessions.values() if not x.busy and x.q.empty()]                              # the least recently used session that is not in the middle of a question gives up its slot
                if not idle: raise Busy(f"{self.max} sessions are active")
                old = min(idle, key=lambda x: x.last); old.q.put(None); del self.sessions[old.sid]
            s = self.sessions[sid] = _Session(sid, self.db, self.pool, self.factory, self.now_fn); s.start()
        s.ready.wait()
        if s.error:
            with self.lock: self.sessions.pop(sid, None)
            raise s.error
        return s

    def ask(self, sid, text, persona="driver", timeout=180.0):
        s = self._session(sid); box, ev = [], threading.Event(); t0 = time.perf_counter(); s.q.put((text, persona, box, ev))
        if not ev.wait(timeout): raise TimeoutError("the assistant did not answer in time")
        self.lat.append(time.perf_counter() - t0)
        if isinstance(box[0], Exception): raise box[0]
        return box[0]

    def stats(self):
        w = sorted(self.lat); n = len(w)
        return {"sessions": len(self.sessions), "max_sessions": self.max, "answers": n, "p50_ms": round(1000 * w[n // 2], 1) if n else None, "p95_ms": round(1000 * w[int(n * 0.95)], 1) if n else None, "model": self.pool.stats()}

    def close(self):
        with self.lock:
            for s in self.sessions.values(): s.q.put(None)
            self.sessions.clear()
