"""Semantic integrity: valid-but-wrong measures are rejected because the REQUEST does not support them (not because they are absent from the schema);
'not recorded' needs positive evidence from the schema; a decision-under-hazard question is a safety request, never a telemetry statistic."""
import os
import sqlite3
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import measure_grounding as MG
from vgpt import safety_policy as SP
from vgpt import lexicon as LX
from vgpt import signal_registry as R
from vgpt import learned_front as LF
from vgpt.query_ir import Query


def status(plan, text, ctx=None):
    return [t.status for t in MG.trace_plan(Query.from_text(plan), text, ctx)]


class MeasureGrounding(unittest.TestCase):
    def test_cabin_co2_is_not_coolant_flow(self):
        for t in ("What was the cabin CO2 level this week?", "average carbon dioxide in the cabin yesterday"):
            self.assertEqual(status("op=aggregate tgt=coolant_flow agg=mean t=cal:this_week", t), ["ungrounded"])
            self.assertTrue(MG.plan_issues(Query.from_text("op=aggregate tgt=coolant_flow agg=mean t=cal:this_week"), t))

    def test_brake_fluid_is_not_brake_pad_wear(self):
        self.assertEqual(status("op=current_value tgt=brake_pad_life t=now", "is the brake fluid quality okay"), ["contradicted"])
        self.assertEqual(status("op=current_value tgt=brake_fluid_condition t=now", "is the brake fluid quality okay"), ["explicit"])

    def test_rated_speed_is_not_recorded_speed(self):
        self.assertEqual(status("op=extreme tgt=speed agg=max t=last:7:day", "what is the rated maximum speed of this car"), ["contradicted"])
        self.assertEqual(status("op=current_value tgt=rated_max_speed t=now", "what is the rated maximum speed of this car"), ["explicit"])

    def test_tyre_pressure_is_not_a_temperature(self):
        self.assertEqual(status("op=aggregate tgt=ambient_temperature agg=mean t=last:7:day", "average tyre pressure last week"), ["contradicted"])
        self.assertEqual(status("op=aggregate tgt=tyre_pressure agg=mean t=last:7:day", "average tyre pressure last week"), ["explicit"])

    def test_a_different_component_of_the_same_quantity_is_rejected(self):
        self.assertEqual(status("op=extreme tgt=battery_temperature agg=max t=last:7:day", "highest motor temperature this week"), ["contradicted"])
        self.assertEqual(status("op=extreme tgt=motor_temperature agg=max t=last:7:day", "highest motor temperature this week"), ["explicit"])

    def test_correct_paraphrases_stay_accepted(self):
        for plan, text in (("op=aggregate tgt=battery_temperature agg=mean t=last:7:day", "mean temperature of the traction battery last week"),
                           ("op=exists tgt=speed t=last:7:day p.thr=gt:120 unit=km_per_h", "did I ever pass 120 km/h in the last week"),
                           ("op=exists tgt=motor_temperature t=last:7:day p.thr=gt:90 unit=C", "did the motor go above 90 degrees last week"),
                           ("op=current_value tgt=hvac_power t=now", "how much power is the heater pulling"),
                           ("op=aggregate tgt=wh_per_km agg=mean t=cal:this_month", "typical energy per kilometre so far this month")):
            self.assertEqual(MG.plan_issues(Query.from_text(plan), text), [], text)

    def test_a_followup_inherits_the_measure_and_quantity_kind_from_trusted_context(self):
        ctx = "op=extreme tgt=battery_temperature agg=max t=last:7:day"
        self.assertEqual(status("op=extreme tgt=battery_temperature agg=max t=cal:yesterday", "and yesterday?", ctx), ["context"])
        self.assertEqual(status("op=extreme tgt=motor_temperature agg=max t=last:7:day", "what about the motor", ctx), ["explicit"])

    def test_without_context_a_bare_followup_is_ungrounded(self):
        self.assertEqual(status("op=extreme tgt=battery_temperature agg=max t=cal:yesterday", "and yesterday?"), ["ungrounded"])


class _Q: last = None
class _B: qcs = _Q()


class Stub:
    lora_path = None
    def __init__(self, out): self.out = out
    def compile(self, text, context=None): return Query.from_text(self.out), {"raw": self.out, "ms": 0.0, "min_p": 0.99}


class FrontIntegration(unittest.TestCase):
    def front(self, plan):
        f = LF.LearnedFront(_B(), Stub(plan)); f.contract = 2; return f

    def test_valid_but_wrong_measure_is_never_executed(self):
        f = self.front("op=aggregate tgt=coolant_flow agg=mean t=cal:this_week")
        r, it = f.answer("What was the cabin CO2 level this week?")
        self.assertEqual(r["facts"]["why"], "unrecorded"); self.assertEqual(r["facts"]["grounding"], "absent quantity"); self.assertNotIn("L/min", r["answer"])

    def test_unrelated_wrong_measure_without_absent_evidence_abstains(self):
        f = self.front("op=aggregate tgt=coolant_flow agg=mean t=last:7:day")
        r, it = f.answer("what was the average tyre pressure in the last 7 days")
        self.assertIsNone(r); self.assertTrue(it.issues)

    def test_safety_topic_is_not_gated_by_confidence(self):
        class Low(Stub):
            def compile(self, text, context=None): return Query.from_text(self.out), {"raw": self.out, "ms": 0.0, "min_p": 0.3}
        f = LF.LearnedFront(_B(), Low("op=topic t=all p.topic=safety_assessment")); f.contract = 2
        r, it = f.answer("Should I keep driving if a red brake warning appears?")
        self.assertEqual(r["intent"], "SAFETY"); self.assertIn("stop", r["answer"].lower())


class Outcomes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = R.Profile(sqlite3.connect(os.path.join(HERE, "data", "gen_demo.db"))) if os.path.exists(os.path.join(HERE, "data", "gen_demo.db")) else None

    @unittest.skipUnless(os.path.exists(os.path.join(HERE, "data", "gen_demo.db")), "needs data/gen_demo.db")
    def test_unrecorded_is_contradicted_by_a_recorded_signal(self):
        iss, why = MG.outcome_issues("unrecorded", "did my tyre pressure get worse over time", profile=self.p)
        self.assertTrue(iss); self.assertEqual(why, "contradicted")
        iss, why = MG.outcome_issues("unrecorded", "has my 12 V battery been weak", profile=self.p)
        self.assertTrue(iss)

    @unittest.skipUnless(os.path.exists(os.path.join(HERE, "data", "gen_demo.db")), "needs data/gen_demo.db")
    def test_unrecorded_needs_positive_evidence(self):
        self.assertEqual(MG.outcome_issues("unrecorded", "what was the cabin CO2 level", profile=self.p), ([], "unrecorded"))
        self.assertEqual(MG.outcome_issues("unrecorded", "how was it", profile=self.p), ([], "ambiguous"))
        self.assertEqual(MG.outcome_issues("unrecorded", "what was the cabin temperature", profile=self.p)[1], "unrecorded")    # known concept this database lacks

    def test_other_outcomes_are_untouched(self):
        self.assertEqual(MG.outcome_issues("unsafe", "anything"), ([], "unsafe"))


class Safety(unittest.TestCase):
    POS = ["Should I keep driving if a red brake warning appears?", "is it safe to drive with this warning", "Can I continue driving if the battery is overheating?",
           "my tyre pressure warning came on, can I keep driving", "Is it okay to drive with a low 12 V battery warning?", "the brake warning is red what should I do",
           "should I stop if the motor temperature warning shows"]
    NEG = ["how many red brake warnings did I get this week", "when did the last tyre pressure warning appear", "show me all warnings from yesterday", "what was the average motor temperature last week",
           "list the faults in the last 30 days", "did I get a battery overheating warning yesterday", "hello", "can I see my energy use today"]

    def test_decisions_under_hazard_are_safety_requests(self):
        for t in self.POS: self.assertIsNotNone(SP.classify(t), t)

    def test_retrospective_and_analytic_questions_are_not(self):
        for t in self.NEG: self.assertIsNone(SP.classify(t), t)

    def test_response_is_conservative_and_not_a_count(self):
        r = SP.respond(SP.classify(self.POS[0]))
        a = r["answer"].lower()
        self.assertEqual(r["intent"], "SAFETY"); self.assertIn("stop", a); self.assertIn("workshop", a)
        self.assertNotRegex(a, r"\b\d+ warnings?\b"); self.assertNotIn("you can keep driving", a); self.assertNotIn("is fine", a)

    def test_it_does_not_name_a_failed_component(self):
        a = SP.respond(SP.classify("Should I keep driving if a red brake warning appears?"))["answer"].lower()
        self.assertIn("cannot tell from the data which part", a)

    @unittest.skipUnless(os.path.exists(os.path.join(HERE, "data", "gen_demo.db")), "needs data/gen_demo.db")
    def test_brain_answers_safety_before_any_statistic_in_every_frontend(self):
        from vgpt.product import make_legacy_brain
        b = make_legacy_brain(os.path.join(HERE, "data", "gen_demo.db"), slm=None, now=1787625900)
        for fe in ("legacy", "learned_only"):
            os.environ["VGPT_FRONTEND"] = fe
            try:
                r = b.ask(self.POS[0], use_slm=False)
            finally: os.environ.pop("VGPT_FRONTEND", None)
            self.assertEqual(r["intent"], "SAFETY"); self.assertEqual(r["facts"]["route"], "safety_policy")


class Negation(unittest.TestCase):
    def test_inverted_negation_is_rejected(self):
        from vgpt import condition_grounding as CG
        q = Query.from_text("op=duration tgt=speed t=last:7:day c=highway p.thr=gt:100 unit=km_per_h")
        self.assertTrue(CG.plan_issues(q, "Time spent above 100 km/h but not on the highway in the last 7 days"))

    def test_correct_polarity_and_plain_positive_are_accepted(self):
        from vgpt import condition_grounding as CG
        ok = Query.from_text("op=duration tgt=speed t=last:7:day c=not:highway p.thr=gt:100 unit=km_per_h")
        self.assertEqual(CG.plan_issues(ok, "Time spent above 100 km/h but not on the highway in the last 7 days"), [])
        pos = Query.from_text("op=aggregate tgt=speed agg=mean t=last:7:day c=highway")
        self.assertEqual(CG.plan_issues(pos, "average speed on the highway in the last 7 days"), [])


class Lexicon(unittest.TestCase):
    def test_paraphrases_and_canonical_units(self):
        self.assertIn("battery_temperature", LX.signal_candidates("cell block temperature"))
        self.assertTrue(any(m.kind == "unit" and m.ids == ("km/h",) for m in LX.mentions("above 100 km/h")))


if __name__ == "__main__":
    unittest.main()
