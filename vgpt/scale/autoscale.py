"""The model pool that grows and shrinks, and the controller that decides when.

    pool = ElasticModelPool(factory, workers=plan.contexts, max_workers=plan.max_contexts)
    ctl  = Autoscaler(pool, plan, on_shed=...)           # starts a thread that samples the pool once a second
    hub  = SessionHub(db, pool, brain_factory, max_sessions=plan.sessions)

Why an elastic pool.  One model context answers one question at a time.  On a 2-core board that is all there is; on a bigger machine the same service must use the extra cores, and give them back when the load goes.
All workers read one priority queue (a driver's question goes before a service engineer's, which goes before fleet and validation batches), so adding a worker needs no routing: it starts taking jobs the moment its model
has loaded.  Removing one sends it a stop token.

Why a controller.  The target is an answer time (2 s).  What threatens it is the queue in front of the model, which the controller reads as the recent 95th percentile of the wait plus the queue length.  Order of response:
  1. add a model context (bounded by cores and memory, and not while the previous one is still loading);
  2. when no context can be added, give things up in a fixed order (`plan.SHED_LADDER`): background analytics, camera analysis rate, queue service/fleet questions behind the driver's, refuse non-driver questions,
     shorten answers - the driver's question is the last thing ever touched;
  3. when the load is gone for a while: restore in the reverse order, then release contexts (their memory goes back to the vehicle).
`Autoscaler.step(metrics, now)` is a pure decision function (tested with invented loads); `run()` is the loop that feeds it real ones."""
import collections
import itertools
import queue
import threading
import time

from .plan import SHED_LADDER

PRIORITY = {"driver": 0, "service": 1, "engineer": 1, "fleet": 2, "validation": 2, "batch": 3}


class Refused(Exception):
    """The load is too high for this kind of question right now (the shed ladder reached 'reject_non_driver_questions')."""


class _Worker(threading.Thread):
    def __init__(self, pool, name):
        super().__init__(daemon=True, name=name); self.pool = pool; self.model = None; self.ready = threading.Event(); self.error = None; self.n = 0; self.busy_s = 0.0; self.stop_f = False

    def run(self):
        p = self.pool
        try: self.model = p.factory()
        except Exception as e: self.error = e; p._loaded(); self.ready.set(); p._gone(self); return
        p._loaded(); self.ready.set()
        while True:
            prio, _, job = p.q.get()
            if job is None: p._gone(self); return                                                        # a stop token
            text, hint, prev, box, ev, t_enq = job; t0 = time.perf_counter()
            with p.lock: p.in_flight += 1
            try: box.append(self.model.parse(text, hint) if prev is None else self.model.parse(text, hint, prev=prev))
            except Exception as e: box.append(e)
            now = time.perf_counter(); self.busy_s += now - t0; self.n += 1
            with p.lock: p.in_flight -= 1; p.done += 1; p.recent.append((time.monotonic(), now - t_enq, prio))
            ev.set()


class ElasticModelPool:
    """`parse(text, hint, prev)` and `glossary` like a single model, backed by a changing number of model contexts."""
    def __init__(self, factory, workers=1, min_workers=1, max_workers=None):
        self.factory = factory; self.q = queue.PriorityQueue(); self.seq = itertools.count(); self.lock = threading.Lock(); self.workers = []; self.in_flight = 0
        self.min_workers = max(1, int(min_workers)); self.max_workers = int(max_workers or max(workers, 1)); self.recent = collections.deque(maxlen=2000); self.done = 0; self.wait_s = collections.deque(maxlen=500); self.loading = 0
        first_workers = [self._spawn() for _ in range(max(1, int(workers)))]
        for w in first_workers: w.ready.wait()
        bad = [w.error for w in first_workers if w.error]
        if bad: raise bad[0]
        first = first_workers[0].model; self.glossary = getattr(first, "glossary", None); self.onto = getattr(first, "onto", None); self.shed_level = 0; self.reject_below = None

    def _spawn(self):
        w = _Worker(self, f"model-{next(self.seq)}");
        with self.lock: self.workers.append(w); self.loading += 1
        w.start(); return w

    def _loaded(self):
        with self.lock: self.loading = max(0, self.loading - 1)

    def _gone(self, w):
        with self.lock:
            if w in self.workers: self.workers.remove(w)

    def parse(self, text, hint=None, prev=None, persona=None):
        if persona is None:
            from .. import sessions as S
            persona = getattr(getattr(S, "CURRENT", None), "persona", "driver")
        prio = PRIORITY.get(persona, 1)
        if self.reject_below is not None and prio >= self.reject_below: raise Refused(f"the assistant is at its limit: only the driver's questions are answered for now ({persona} request refused)")
        box, ev = [], threading.Event(); t0 = time.perf_counter(); self.q.put((prio, next(self.seq), (text, hint, prev, box, ev, t0))); ev.wait(); self.wait_s.append(time.perf_counter() - t0)
        if isinstance(box[0], Exception): raise box[0]
        return box[0]

    def __getattr__(self, name):
        """The pool looks like one model: anything else a brain asks of its model (version, onto, mask ...) is answered by the first context's."""
        if name.startswith("_") or name in ("workers", "lock", "q"): raise AttributeError(name)
        ws = self.__dict__.get("workers") or []
        for w in list(ws):
            if w.model is not None and hasattr(w.model, name): return getattr(w.model, name)
        raise AttributeError(name)

    def refresh(self):
        """Another vehicle: every context rebuilds its decoding words."""
        for w in list(self.workers):
            if w.model is not None and hasattr(w.model, "refresh"): w.model.refresh()

    # ------------------------------------------------------------------------------------------ the controller's levers
    def contexts(self):
        with self.lock: return len(self.workers)

    def add_context(self):
        with self.lock:
            if len(self.workers) >= self.max_workers or self.loading: return False
        self._spawn(); return True

    def remove_context(self):
        with self.lock:
            if len(self.workers) <= self.min_workers: return False
        self.q.put((-1, next(self.seq), None)); return True                                          # the first idle worker takes the token and ends

    def metrics(self, window_s=10.0):
        now = time.monotonic()
        with self.lock:
            rec = [(t, w, p) for t, w, p in self.recent if now - t <= window_s]; n = len(self.workers); ready = n - self.loading; fl = self.in_flight
        w = sorted(x[1] for x in rec); k = len(w)
        return {"contexts": n, "ready": ready, "loading": self.loading, "in_flight": fl, "queue": self.q.qsize(), "done": self.done, "answers": k, "p50": w[k // 2] if k else 0.0, "p95": w[min(k - 1, int(k * 0.95))] if k else 0.0,
                "driver_p95": self._pct([x[1] for x in rec if x[2] == 0])}

    @staticmethod
    def _pct(v, q=0.95):
        v = sorted(v); return v[min(len(v) - 1, int(len(v) * q))] if v else 0.0

    def stats(self):
        w = sorted(self.wait_s); n = len(w)
        return {"workers": len(self.workers), "parses": sum(x.n for x in self.workers), "busy_s": [round(x.busy_s, 2) for x in self.workers], "wait_p50_ms": round(1000 * w[n // 2], 1) if n else None, "wait_p95_ms": round(1000 * w[int(n * 0.95)], 1) if n else None, "shed_level": self.shed_level}

    def close(self):
        for _ in range(len(self.workers)): self.q.put((-2, next(self.seq), None))


class Autoscaler:
    def __init__(self, pool, plan, on_shed=None, on_restore=None, target_s=None, interval_s=1.0, calm_steps=8, cooldown_s=3.0, clock=time.monotonic, mem_used_mb=None, mem_budget_mb=None, min_gain=0.2, verify_s=None):
        self.pool, self.plan = pool, plan; self.target = float(target_s or plan.latency_target_s); self.level = 0; self.calm = 0; self.last_add = -1e9; self.cooldown = float(cooldown_s); self.calm_steps = int(calm_steps)
        self.on_shed = on_shed or (lambda step: None); self.on_restore = on_restore or (lambda step: None); self.interval = interval_s; self.clock = clock; self.log = []; self.thread = None; self.stop_f = threading.Event()
        self.mem_used_mb = mem_used_mb; self.mem_budget_mb = mem_budget_mb; self.window = max(2.0, 10 * interval_s)
        self.min_gain = float(min_gain); self.verify_s = float(verify_s or self.window); self.verifs = []; self.proven = False; self.revert_to = None; self.scale_out_cap = None; self.samples = collections.deque(maxlen=4000); self.min_answers = 12; self.verify_max_s = 120.0        # an added context must raise the answers per second by min_gain or it is taken away again

    def _rate(self, now, span, fallback):
        """Answers per second over the last `span` seconds from the cumulative count (the fallback when no count was seen)."""
        if len(self.samples) < 2: return fallback
        t1, d1 = self.samples[-1]; old = [x for x in self.samples if t1 - x[0] >= span]; t0, d0 = old[-1] if old else self.samples[0]
        return (d1 - d0) / (t1 - t0) if t1 - t0 >= 5.0 else fallback

    # ---- the decision, pure: metrics in, actions out ----
    def step(self, m, now=None):
        now = self.clock() if now is None else now; acts = []; tgt = self.target
        if m.get("done") is not None: self.samples.append((now, m["done"]))
        over = m["p95"] > 0.75 * tgt or m["queue"] >= max(2, 2 * max(1, m["ready"]))
        under = m["p95"] < 0.25 * tgt and m["queue"] == 0 and m["in_flight"] < max(1, m["ready"])
        mem_hot = bool(self.mem_used_mb and self.mem_budget_mb and self.mem_used_mb() > 0.92 * self.mem_budget_mb)
        if self.revert_to is not None:                                                                                        # a context that did not pay for itself is being taken away (it can only leave when idle)
            if m["contexts"] > self.revert_to: acts.append(("remove_context", "the extra context did not raise throughput"))
            else: self.revert_to = None
            for a in acts: self.log.append((round(now, 2), a[0], a[1]))
            return acts
        self.verifs = [v for v in self.verifs if now - v["t"] <= 4 * self.verify_max_s]                                           # a context that never became ready is not waited for
        for v in list(self.verifs):
            if m["contexts"] < v["n"] + 1 or m["ready"] < v["n"] + 1: continue
            if v["ready_at"] is None: v["ready_at"] = now; v["done0"] = m.get("done"); continue
            waited = now - v["ready_at"]; got = (m["done"] - v["done0"]) if (m.get("done") is not None and v["done0"] is not None) else None
            enough = waited >= self.verify_s if got is None else ((waited >= self.verify_s and got >= self.min_answers) or waited >= self.verify_max_s)        # a slow machine needs time for enough answers: 4 answers in 10 s is +-25 % noise
            if not enough: continue
            busy = m["queue"] > 0 or m["in_flight"] >= m["ready"]; after = (got / max(waited, 1e-9)) if got is not None else m["answers"] / self.window; self.verifs.remove(v)
            if busy and after < v["before"] * (1.0 + self.min_gain):
                self.scale_out_cap = v["n"]; self.pool.max_workers = v["n"]; self.revert_to = v["n"]; self.verifs = []
                why = f"{v['n']} -> {v['n'] + 1} contexts moved {v['before']:.2f} to {after:.2f} answers/s (< +{int(100 * self.min_gain)} %): scale-out is capped at {v['n']} on this machine"
                acts.append(("remove_context", why)); self.log.append((round(now, 2), "scale_out_capped", why))
                for a_ in acts: self.log.append((round(now, 2), a_[0], a_[1]))
                return acts
            if busy: self.proven = True                                                                                         # the machine does scale out: later steps no longer wait for each other
        if mem_hot and m["contexts"] > self.pool.min_workers: acts.append(("remove_context", "memory pressure"))
        elif over:
            self.calm = 0
            if m["contexts"] < self.pool.max_workers and not m["loading"] and now - self.last_add >= self.cooldown and not mem_hot and (self.proven or not self.verifs): acts.append(("add_context", f"p95 {m['p95']:.2f}s, queue {m['queue']}")); self.last_add = now; self.verifs.append({"n": m["contexts"], "before": self._rate(now, 30.0, m["answers"] / self.window), "ready_at": None, "t": now, "done0": None})
            elif m["contexts"] >= self.pool.max_workers and self.level < len(SHED_LADDER) and m["p95"] > 0.9 * tgt: self.level += 1; acts.append(("shed", SHED_LADDER[self.level - 1]))
        elif under:
            self.calm += 1
            if self.calm >= self.calm_steps:
                self.calm = 0
                if self.level > 0: acts.append(("restore", SHED_LADDER[self.level - 1])); self.level -= 1
                elif m["contexts"] > self.pool.min_workers: acts.append(("remove_context", "idle"))
        else: self.calm = 0
        for a in acts: self.log.append((round(now, 2), a[0], a[1]))
        return acts

    def apply(self, acts):
        for kind, why in acts:
            if kind == "add_context": self.pool.add_context()
            elif kind == "remove_context": self.pool.remove_context()
            elif kind == "shed":
                self.on_shed(why); self.pool.shed_level = self.level
                if why == "queue_service_and_fleet_questions": pass                                    # the shared queue already serves the driver first
                if why == "reject_non_driver_questions": self.pool.reject_below = 1
            elif kind == "restore":
                self.on_restore(why); self.pool.shed_level = self.level
                if why == "reject_non_driver_questions": self.pool.reject_below = None

    def run(self):
        while not self.stop_f.wait(self.interval): self.apply(self.step(self.pool.metrics(self.window)))

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name="autoscaler"); self.thread.start(); return self

    def stop(self): self.stop_f.set()
