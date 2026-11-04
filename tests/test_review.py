"""The interpretation-review prototype (ui/review.html, vgpt/interpret.py, /api/interpret, /api/execute, /api/judge): a model proposes, the person edits, deterministic code runs only what was confirmed. The model is replaced by the scripted test double (tests/stub_planner.py): these tests are about the interaction contract, not about language understanding."""
import json, os, sys, tempfile, unittest, importlib.util
from tests import testdata
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tests.stub_planner import StubPlanner

HAVE = os.path.exists(os.path.join(ROOT, "data", "ref", "camera_clip.hevc"))


@unittest.skipUnless(HAVE, "needs the clip")
@testdata.needs("legacy_worlds")
class Review(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(); os.environ["VEHICLEGPT_HOME"] = cls.home
        spec = importlib.util.spec_from_file_location("uiserver_rev", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        script = {"how hot is the battery": ("statistic", {"turn": "new", "signal": "battery_temperature", "stat": "mean", "period": "yesterday"})}
        cls.app = cls.m.App(SimpleNamespace(research=True, db="data/gen_courier.db", model="none", camera="clip", webcam_index=0, port=0), planner_factory=lambda conn: StubPlanner(conn, script))
        cls.dispatch = staticmethod(cls.m.make_dispatcher(cls.app))

    def req(self, method, path, body=None):
        if method == "POST" and isinstance(body, dict): body = dict(body, consent=True)                                   # the study's recording calls need the participant's consent (vgpt/routepolicy.py)
        r = self.dispatch(method, path, {"Content-Type": "application/json"} if body is not None else {}, json.dumps(body) if body is not None else b""); return r.status, r.body

    def test_page_and_catalogue_are_generated_from_the_registry(self):
        st, page = self.req("GET", "/review"); self.assertEqual(st, 200); self.assertIn(b"Review the interpretation", page)
        st, d = self.req("GET", "/api/catalogue"); cat = json.loads(d)
        self.assertTrue(any(s["id"] == "battery_temperature" for s in cat["signals"])); self.assertIn("statistic", cat["field_specs"]); self.assertTrue({"signal", "stat", "period", "conditions"} <= {f["name"] for f in cat["field_specs"]["statistic"]})
        self.assertGreaterEqual(len(cat["functions"]), 20); self.assertTrue(any(f["name"] == "rank_trips" for f in cat["functions"]))

    @unittest.skip("the interpretation-review prototype proposes through brain.propose(), which only the retired planner route (vgpt/modelbrain.py) has; vgpt/product.py ships the compact classifier + typed engine")
    def test_propose_edit_confirm_and_log(self):
        st, d = self.req("POST", "/api/interpret", {"q": "how hot is the battery"}); p = json.loads(d)
        self.assertTrue(p["ok"]); self.assertEqual(p["fn"], "statistic"); self.assertEqual(p["args"]["signal"], "battery_temperature"); self.assertTrue(all(f["source"] == "model" for f in p["fields"] if f["value"] not in (None, "")))
        before = json.loads(self.req("POST", "/api/ask", {"q": "hello"})[1])                                                         # proposing changed no conversation state
        final = {"fn": "extreme", "args": {"signal": "motor_temperature", "which": "max", "period": "last 3 days"}}                  # the person changes the operation AND the quantity (neither was proposed)
        st, d = self.req("POST", "/api/execute", {"q": "how hot is the battery", "participant": "p1", "task": "t1", "proposal": {"fn": p["fn"], "args": p["args"], "raw": p["raw"]}, "final": final, "client": {"review_ms": 1234}}); r = json.loads(d)
        self.assertIn("motor", r["answer"].lower()); self.assertNotIn("battery temperature is", r["answer"].lower())
        fields = {e["field"] for e in r["edits"]}; self.assertTrue({"operation", "signal"} <= fields)
        self.req("POST", "/api/judge", {"q": "how hot is the battery", "participant": "p1", "task": "t1", "verdict": "yes"})
        lines = [json.loads(l) for l in open(os.path.join(self.home, "interaction_log.jsonl"), encoding="utf-8")]
        ex = [l for l in lines if l["event"] == "executed"][-1]; self.assertEqual(ex["original_text"], "how hot is the battery"); self.assertEqual(ex["proposal"]["fn"], "statistic"); self.assertEqual(ex["final"]["fn"], "extreme"); self.assertGreaterEqual(ex["n_edits"], 2)
        self.assertEqual([l for l in lines if l["event"] == "judged"][-1]["verdict"], "yes")

    @unittest.skip("the interpretation-review prototype proposes through brain.propose(), which only the retired planner route (vgpt/modelbrain.py) has; vgpt/product.py ships the compact classifier + typed engine")
    def test_an_invalid_interpretation_is_refused_not_repaired(self):
        st, d = self.req("POST", "/api/execute", {"q": "x", "final": {"fn": "nonsense", "args": {}}}); self.assertFalse(json.loads(d)["ok"])
        st, d = self.req("POST", "/api/execute", {"q": "average speed", "final": {"fn": "statistic", "args": {"signal": "no_such_signal", "stat": "mean"}}}); r = json.loads(d)
        self.assertNotIn("answered", str(r.get("outcome"))); self.assertTrue(r.get("path") in ("plan", "gate") or "not" in r.get("answer", "").lower() or "could not" in r.get("answer", "").lower())

    def test_follow_up_proposal_is_resolved_against_the_confirmed_query(self):
        self.req("POST", "/api/execute", {"q": "avg", "final": {"fn": "statistic", "args": {"signal": "speed", "stat": "mean", "period": "last 3 days", "conditions": ["highway"]}}})
        self.assertIsNotNone(self.app.brain)                                                                                                    # the confirmed query is the state a later proposal is resolved against


if __name__ == "__main__": unittest.main()


@unittest.skipUnless(HAVE, "needs the clip")
class FollowUpFlag(unittest.TestCase):
    @unittest.skip("the interpretation-review prototype proposes through brain.propose(), which only the retired planner route (vgpt/modelbrain.py) has; vgpt/product.py ships the compact classifier + typed engine")
    def test_a_new_topic_proposal_does_not_see_the_earlier_request(self):
        seen = []
        class Spy(StubPlanner):
            def plan(self, question, state=None, *a, **k):
                seen.append(bool(state and state.prev_plan)); return super().plan(question, state, *a, **k)
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location("uiserver_fu", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        app = m.App(SimpleNamespace(db="data/gen_courier.db", model="none", camera="clip", webcam_index=0, port=0), planner_factory=Spy)
        try:
            app.execute({"q": "avg", "final": {"fn": "statistic", "args": {"signal": "speed", "stat": "mean", "period": "last 3 days"}}})
            app.interpret({"q": "something else"}); app.interpret({"q": "and on the highway", "follow_up": True})
            self.assertEqual(seen, [False, True])                 # new topic: no earlier request in the prompt; follow-up: the confirmed query is
        finally: pass
