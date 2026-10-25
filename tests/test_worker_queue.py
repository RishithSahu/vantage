"""The assistant's single worker: a bounded queue, deadlines, conversations per session, no answer cache, and de-duplication of the SAME request only while it is in flight.

The brain here is a small stand-in (the worker does not care what answers), so every case is about what the worker and `App.ask` do around it:

  * a full queue is answered "busy" at once (no waiting, no unbounded memory);
  * a job whose deadline passed before the worker reached it is thrown away - it is never run;
  * a job that was already running when its waiter gave up has its effect on the conversation undone;
  * each session has its own conversation: a follow-up is read against that session's last question only;
  * there is no answer cache: the same words asked twice are two runs; the same REQUEST (session + request id) submitted twice while it is running is one run;
  * even a history that cannot be opened leaves a worker that answers, with the reason."""
import importlib.util
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def server():
    if "uiserver_wq" not in sys.modules:
        spec = importlib.util.spec_from_file_location("uiserver_wq", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); sys.modules["uiserver_wq"] = m; spec.loader.exec_module(m)
    return sys.modules["uiserver_wq"]


class Dialogue:
    def __init__(self): self.turns = []


class FakeBrain:
    """Answers "ans:<text>" after `delay`; remembers what it was asked (the conversation: `turns`), and the text of the previous question of ITS session as `saw`."""
    runs = []
    gate = None
    delay = 0.0

    def __init__(self, *a, **k):
        self.units = {"p": "kpa", "d": "km", "t": "c"}; self.dialogue = Dialogue(); self._last_data = None; self.turn = 0; self.now = 0; self.persona = "driver"; self.c = type("C", (), {"close": lambda s: None})()
        self.qcs = type("Q", (), {"choices": [], "pending": None, "options": [], "last": None})()

    def ask(self, text):
        FakeBrain.runs.append(text)
        if FakeBrain.gate is not None: FakeBrain.gate.wait(5)
        if FakeBrain.delay: time.sleep(FakeBrain.delay)
        saw = self.dialogue.turns[-1] if self.dialogue.turns else None
        self.dialogue.turns.append(text); self._last_data = text; self.turn += 1
        return {"answer": "ans:" + text, "intent": "OK", "path": "fake", "ms": {"total": 1}, "params": {}, "saw": saw, "facts": {"route": "fake"}}


def worker(**kw):
    m = server(); FakeBrain.runs = []; FakeBrain.gate = None; FakeBrain.delay = 0.0
    clock = type("Clock", (), {"now": lambda s: 1_790_406_000})()
    w = m.BrainWorker("unused.db", None, clock, brain_factory=lambda *a, **k: FakeBrain(), planner_factory=lambda conn: None); w.start(); assert w.ready.wait(5); return w


class Queue(unittest.TestCase):
    def test_a_full_queue_is_answered_busy_at_once(self):
        w = worker(); FakeBrain.gate = threading.Event(); results = []
        threads = [threading.Thread(target=lambda i=i: results.append(w.ask(f"q{i}", "driver"))) for i in range(w.q.maxsize + 3)]
        for t in threads[:1]: t.start()
        time.sleep(0.2)                                                                                              # the first is running (blocked on the gate)
        for t in threads[1:]: t.start(); time.sleep(0.01)
        time.sleep(0.3); t0 = time.monotonic(); extra = w.ask("one more", "driver"); took = time.monotonic() - t0
        self.assertTrue(extra.get("busy")); self.assertLess(took, 0.5, "a busy answer must not wait"); self.assertEqual(extra["path"], "busy")
        FakeBrain.gate.set()
        for t in threads: t.join(10)
        self.assertEqual(sum(1 for r in results if r.get("busy")), 2)                                              # 1 running + queue size queued; the other two refused
        self.assertEqual(len(FakeBrain.runs), w.q.maxsize + 1)

    def test_a_job_past_its_deadline_is_never_run(self):
        w = worker(); FakeBrain.gate = threading.Event(); first = []
        t = threading.Thread(target=lambda: first.append(w.ask("slow one", "driver"))); t.start(); time.sleep(0.2)
        late = w.ask("too late", "driver", deadline_s=0.2)                                                           # waits 0.2 s behind the slow one and gives up
        self.assertEqual(late["path"], "error"); self.assertEqual(late["answer"], "Timed out.")
        FakeBrain.gate.set(); t.join(5); time.sleep(0.3)
        self.assertEqual(FakeBrain.runs, ["slow one"], "the abandoned question must not run later"); self.assertEqual(w.expired, 1)

    def test_a_running_job_whose_waiter_gave_up_leaves_the_conversation_as_it_was(self):
        w = worker(); self.assertEqual(w.ask("first", "driver")["saw"], None)
        FakeBrain.delay = 0.6; r = w.ask("abandoned", "driver", deadline_s=0.2); self.assertEqual(r["answer"], "Timed out."); FakeBrain.delay = 0.0
        time.sleep(0.8)                                                                                              # it finished running after its waiter left
        self.assertEqual(w.ask("retry", "driver")["saw"], "first", "the retry is read against what came before the abandoned question, not against it")
        self.assertEqual(w.discarded_effects, 1)

    def test_each_session_has_its_own_conversation(self):
        w = worker()
        w.ask("driver one", "driver", "driver"); w.ask("engineer one", "engineer", "tok-eng")
        self.assertEqual(w.ask("driver two", "driver", "driver")["saw"], "driver one")
        self.assertEqual(w.ask("engineer two", "engineer", "tok-eng")["saw"], "engineer one")
        self.assertIsNone(w.ask("other first", "driver", "tok-other")["saw"])


class AppAsk(unittest.TestCase):
    def app(self, **kw):
        import collections
        m = server(); w = worker(); a = object.__new__(m.App)
        a.brain = w; a.flight_lock = threading.Lock(); a.flight = {}; a.ask_lock = threading.Lock(); a.post_lock = threading.Lock(); a.latency = collections.deque(maxlen=300); a.answers = collections.OrderedDict()
        return a

    def test_the_same_words_asked_twice_are_two_runs_not_a_cached_answer(self):
        a = self.app(); r1 = a.ask("what is my speed", "driver"); r2 = a.ask("what is my speed", "driver")
        self.assertEqual(FakeBrain.runs, ["what is my speed", "what is my speed"]); self.assertNotEqual(r1["rid"], r2["rid"])

    def test_the_same_request_in_flight_is_one_run_and_both_callers_get_the_answer(self):
        a = self.app(); FakeBrain.delay = 0.5; out = []
        ts = [threading.Thread(target=lambda: out.append(a.ask("what is my speed", "driver", "driver", "req-1"))) for _ in range(3)]
        for t in ts: t.start()
        for t in ts: t.join(10)
        self.assertEqual(len(FakeBrain.runs), 1); self.assertEqual(len(out), 3); self.assertEqual({r["answer"] for r in out}, {"ans:what is my speed"}); self.assertEqual(len({r["rid"] for r in out}), 1)
        self.assertEqual(a.flight, {}, "a finished request leaves nothing behind to answer the next one")
        a.ask("what is my speed", "driver", "driver", "req-1"); self.assertEqual(len(FakeBrain.runs), 2, "once finished, the same id is a new question")

    def test_a_busy_queue_gives_the_busy_answer_through_app_ask(self):
        a = self.app(); FakeBrain.gate = threading.Event(); ts = [threading.Thread(target=lambda i=i: a.ask(f"q{i}", "driver")) for i in range(a.brain.q.maxsize + 1)]
        for t in ts: t.start(); time.sleep(0.02)
        time.sleep(0.2); self.assertTrue(a.ask("overflow", "driver").get("busy")); FakeBrain.gate.set()
        for t in ts: t.join(10)


class NoHistory(unittest.TestCase):
    def test_a_history_that_cannot_be_opened_still_gives_a_worker_that_says_so(self):
        m = server(); clock = type("Clock", (), {"now": lambda s: 1})()

        def broken(*a, **k): raise FileNotFoundError("model")
        w = m.BrainWorker(os.path.join(ROOT, "data", "no_such_history.db"), None, clock, brain_factory=broken); w.start()
        self.assertTrue(w.ready.wait(10)); r = w.ask("what is my speed", "driver")
        self.assertEqual(r["path"], "error"); self.assertIn("could not be opened", r["answer"]); self.assertIn("FileNotFoundError", str(w.load_error))


if __name__ == "__main__": unittest.main()
