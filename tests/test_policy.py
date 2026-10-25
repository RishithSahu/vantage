"""Clarification-first policy: three outcomes, concrete readings, answers only for supported interpretations, accepted state untouched by a clarification."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import nl2ir, policy
from vgpt.brain import Brain
from vgpt.query_ir import Query

NOW = 1790406000; DBF = os.path.join(ROOT, "data", "gen_courier_ext.db")


def decide(text, prev=None):
    st = nl2ir.ConvState()
    if prev: st.last = Query.from_text(prev)
    pr = nl2ir.parse(text, now=NOW, tz=3600, state=st); return policy.decide(text, pr, None, Query.from_text(prev) if prev else None, None, NOW, 3600)


class Decisions(unittest.TestCase):
    def test_a_fully_supported_question_is_answered_and_discloses_its_default(self):
        d = decide("average battery temperature"); self.assertEqual(d.kind, "answer"); self.assertIn("No period was given, so I used the last 7 days.", d.disclosures)

    def test_an_ambiguous_sensor_is_a_clarification_with_each_sensor_as_a_reading(self):
        d = decide("how often was it colder than 5 degrees last month"); self.assertEqual(d.kind, "clarify"); self.assertGreaterEqual(len(d.candidates), 3); self.assertRegex(d.question, r"\(1\)")
        sig = {c.scope.conditions[0].signal for c in d.candidates}; self.assertTrue({"ambient_temperature", "battery_temperature"} <= sig)                      # a unit gives a dimension: each temperature sensor is offered, none is guessed

    def test_an_unsupported_reading_is_never_answered_silently(self):
        d = decide("how many of those were there"); self.assertEqual(d.kind, "clarify"); self.assertRegex(d.question, r"(?i)previous|refer")


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class Dialogue(unittest.TestCase):
    def setUp(self): os.environ["VGPT_QC"] = "all"; self.addCleanup(os.environ.pop, "VGPT_QC", None); self.b = Brain(DBF, slm_ir=None, now=NOW, policy="clarify")

    def test_a_reply_chooses_a_reading_and_only_then_the_state_changes(self):
        r = self.b.ask("temperature last night", use_slm=False); self.assertIn("Which temperature", r["answer"]); self.assertIsNone(self.b.qcs.last); self.assertEqual(self.b.trace.records[-1].outcome, "clarify")
        r = self.b.ask("2", use_slm=False); self.assertEqual(self.b.qcs.last.target, "motor_temperature"); self.assertEqual(self.b.trace.records[-1].outcome, "answered"); self.assertEqual(self.b.qcs.choices, [])

    def test_an_unrelated_reply_clears_the_pending_choices(self):
        self.b.ask("temperature last night", use_slm=False); self.assertTrue(self.b.qcs.choices)
        self.b.ask("average speed on the highway last week", use_slm=False); self.assertEqual(self.b.qcs.choices, []); self.assertEqual(self.b.qcs.last.op, "aggregate")

    def test_an_unparseable_data_question_is_not_answered_by_a_fuzzy_route(self):
        r = self.b.ask("how often was it colder than 5 degrees last month", use_slm=False); self.assertIn("Do you mean", r["answer"]); self.assertEqual(r["path"], "qc")

    def test_charging_rate_and_duration_are_different_capabilities(self):
        rate = self.b.ask("how is charging speed", use_slm=False)
        self.assertEqual(rate["path"], "qc"); self.assertIn("charging power", rate["answer"].lower()); self.assertNotIn("vehicle speed", rate["answer"]); self.assertNotIn("km/h", rate["answer"])
        duration = self.b.ask("how long does the car take to charge", use_slm=False)
        self.assertEqual(duration["path"], "qa"); self.assertRegex(duration["answer"], r"needs \d+ kWh"); self.assertRegex(duration["answer"], r"\d+ min|\d+ h")
        correction = self.b.ask("no, I mean how long does the car take to charge", use_slm=False)
        self.assertEqual(correction["path"], "qa"); self.assertRegex(correction["answer"], r"needs \d+ kWh")

    def test_the_default_policy_keeps_the_earlier_behaviour(self):
        b = Brain(DBF, slm_ir=None, now=NOW); self.assertEqual(b.policy, "legacy"); self.assertEqual(b.qcs.choices, [])


if __name__ == "__main__": unittest.main()
