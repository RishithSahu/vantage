"""A deterministic stand-in for the model planner, for tests of everything AROUND the model (HTTP server, concurrency, vehicle switching, request isolation). It has the Planner's interface (plan / reg / dialect / reload_schema / expand_ids), loads no language model and maps a question to one of a few valid plans by a stable hash, so the same question always gives the same answer and different questions can give different ones. It interprets nothing: it is a test double, never part of the product. A real-model integration check lives in tests/test_model_route.py (opt-in, needs the checkpoint)."""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpt import dialects, toolspec as T
from vgpt.modelplan import Planner, PlanResult


class StubPlanner(Planner):
    def __init__(self, conn, script=None):
        super().__init__(conn, load_model=False, dialect="xlam"); self.script = {k.strip().lower(): v for k, v in (script or {}).items()}      # optional: exact question -> (function, arguments) for a test that checks what the executor answers

    def _signals(self): return sorted(c.id for c in self.reg.items if c.kind == "signal" and c.available and c.dtype == "numeric")
    def _events(self): return sorted(c.id for c in self.reg.items if c.kind == "event" and c.available)

    def plan(self, question, state=None, now=None, tz=0, max_tokens=0, feedback=None, extra_ids=None, result=None):
        h = int(hashlib.sha256(question.strip().lower().encode()).hexdigest(), 16); sig = self._signals(); ev = self._events(); out = PlanResult(message=question)
        kind = h % 5; scripted = self.script.get(question.strip().lower())
        if scripted: fn, args = scripted
        elif "joke" in question.lower() or "hello" in question.lower(): fn, args = "decline", {"reason": "offtopic"}
        elif kind == 0: fn, args = "current_value", {"turn": "new", "signal": sig[h % len(sig)]}
        elif kind == 1: fn, args = "statistic", {"turn": "new", "signal": sig[(h >> 8) % len(sig)], "stat": "mean", "period": "last 7 days"}
        elif kind == 2 and ev: fn, args = "count_events", {"turn": "new", "event": ev[(h >> 8) % len(ev)], "period": "last 7 days"}
        elif kind == 3: fn, args = "extreme", {"turn": "new", "signal": sig[(h >> 16) % len(sig)], "which": "max", "period": "last 7 days"}
        else: fn, args = "trend", {"turn": "new", "signal": sig[(h >> 24) % len(sig)], "period": "last 7 days"}
        out.fn, out.args = fn, args; out.plan = T.call_to_plan(fn, args); out.raw = self.dialect.assistant_text(fn, args); out.timing = {"plan_ms": 0.0}
        return out
