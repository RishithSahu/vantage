"""Learned semantic front end.

Part 1 (always runs): structural behaviour with a stub model - numeric grounding, abstention, context handling, the ablation tripwire.
Part 2 (needs models/qp-lora.gguf + llama_cpp): the plan the Qwen2.5-0.5B adapter writes for spelling / semantic regressions that the old rule layer used to patch
with word lists.  The assertions are on the PLAN (signal, window, state, operation), never on text.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.query_ir import Query
from vgpt import learned_front as LF

ADAPTER = os.environ.get("VGPT_QP_LORA") or os.path.join(HERE, "models", "qp-lora.gguf"); BASE = os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")


class _Qcs: last = None


class _Brain: qcs = _Qcs()


class Stub:
    def __init__(self, out): self.out = out; self.ctx = []

    def compile(self, text, context=None):
        self.ctx.append(context)
        return Query.from_text(self.out), {"raw": self.out, "ms": 0.0}


class Structure(unittest.TestCase):
    def front(self, plan): return LF.LearnedFront(_Brain(), Stub(plan))

    def test_numbers_must_come_from_the_question(self):
        f = self.front("op=exists tgt=speed t=last:7:day p.thr=gt:120 unit=km_per_h")
        self.assertEqual(f.interpret("did my speed ever go over 120 last week").issues, [])
        self.assertTrue(f.interpret("did my speed ever go fast last week").issues)          # 120 was invented

    def test_misspelt_number_words_are_still_grounded(self):
        f = self.front("op=exists tgt=motor_temperature t=last:7:day p.thr=gt:50 unit=C")
        self.assertEqual(f.interpret("did the motor temp cross fitfy over the last 7 days").issues, [])

    def test_median_needs_no_number_in_the_question(self):
        f = self.front("op=percentile tgt=speed t=last:7:day p.pct=50")
        self.assertEqual(f.interpret("what is the median speed last week").issues, [])

    def test_none_plan_abstains_with_its_reason(self):
        f = self.front("op=none t=all p.why=unrecorded")
        it = f.interpret("how many passengers are in the car")
        self.assertTrue(it.abstain)

    def test_context_is_only_sent_while_a_conversation_is_open(self):
        b = _Brain(); b.qcs = _Qcs(); st = Stub("op=aggregate tgt=speed agg=mean t=cal:yesterday"); f = LF.LearnedFront(b, st); f.last_plan = "op=aggregate tgt=speed agg=mean t=last:7:day"
        f.interpret("and yesterday?"); self.assertIsNone(st.ctx[-1])
        b.qcs.last = object(); f.interpret("and yesterday?"); self.assertEqual(st.ctx[-1], "op=aggregate tgt=speed agg=mean t=last:7:day")

    def test_tripwire_blocks_legacy_understanding(self):
        import importlib
        from vgpt import nlu
        try:
            armed = LF.arm_tripwire(); self.assertIn("vgpt.nlu.normalize", armed)
            with self.assertRaises(LF.LegacyRuleCalled): nlu.normalize("what is the speed")
        finally:
            LF.disarm_tripwire()


class ValidatorRepairs(unittest.TestCase):
    """Repairs approved after the v12/v13 gate diagnosis (H18 inherited numbers; corrupted period-word regex)."""
    CTX = "op=duration tgt=speed t=last:7:day p.thr=gt:100 unit=km_per_h"
    PLAN = "op=duration tgt=speed t=last:7:day c=not:charging p.thr=gt:100 unit=km_per_h"

    def check(self, plan, text, context=None):
        return LF.LearnedFront(_Brain(), None).check(Query.from_text(plan), text, context=context)

    def test_h18_inherited_threshold_is_accepted_with_context(self):
        self.assertEqual(self.check(self.PLAN, "not counting the charging time", self.CTX), [])

    def test_h18_without_context_still_rejects(self):                       # the pre-change behaviour is still correct when no trusted context exists
        self.assertTrue(self.check(self.PLAN, "not counting the charging time"))

    def test_fabricated_number_is_rejected_even_with_context(self):
        self.assertTrue(self.check(self.PLAN.replace("gt:100", "gt:120"), "not counting the charging time", self.CTX))

    def test_contradicting_the_users_own_number_is_rejected(self):
        self.assertTrue(self.check(self.PLAN, "what about 120?", self.CTX))                       # user said 120, plan silently keeps 100
        self.assertEqual(self.check(self.PLAN.replace("gt:100", "gt:120"), "what about 120?", self.CTX), [])

    def test_context_does_not_license_unrelated_numbers(self):
        self.assertTrue(self.check("op=percentile tgt=speed t=last:7:day p.pct=90", "and the percentile?", self.CTX))

    def test_interpret_passes_the_open_conversation_context(self):
        b = _Brain(); b.qcs = _Qcs(); b.qcs.last = object()
        f = LF.LearnedFront(b, Stub(self.PLAN)); f.last_plan = self.CTX
        self.assertEqual(f.interpret("not counting the charging time").issues, [])

    def test_period_word_regex_matches_whole_words(self):
        for t in ("this week", "over the last 2 weeks", "yesterday", "per day", "a fortnight ago", "this month"):
            self.assertTrue(LF._has_period_word(t), t)
        for t in ("monday", "someday", "daylight", "weekend", "what is my speed"):
            self.assertFalse(LF._has_period_word(t), t)

    def test_no_backspace_character_in_the_module(self):
        import inspect
        self.assertNotIn(chr(8), inspect.getsource(LF))


class PeriodGrounding(unittest.TestCase):
    """Window-length grounding: explicit, equivalent, inherited, replaced, invented and contradictory periods (rolling windows only; calendar windows have no length to ground)."""
    A = "op=aggregate tgt=speed agg=mean t=%s"

    def issues(self, t, question, context=None):
        out = LF.LearnedFront(_Brain(), None).check(Query.from_text(self.A % t), question, context=context)
        return [i for i in out if "period" in i or "window" in i]

    def test_explicit_rolling_windows(self):
        self.assertEqual(self.issues("last:12:day", "average speed over the last 12 days"), [])
        self.assertEqual(self.issues("last:3:hour", "average speed in the last 3 hours"), [])

    def test_policy_implied_windows(self):
        self.assertEqual(self.issues("last:7:day", "what was my average speed last week"), [])
        self.assertEqual(self.issues("last:30:day", "average speed last month"), [])
        self.assertEqual(self.issues("last:14:day", "average speed over the past fortnight"), [])
        self.assertEqual(self.issues("last:24:hour", "average speed over the past day"), [])

    def test_equivalent_expressions_of_one_duration(self):
        self.assertEqual(self.issues("last:14:day", "average speed over the last two weeks"), [])
        self.assertEqual(self.issues("last:2:week", "average speed over the last 14 days"), [])
        self.assertEqual(self.issues("last:1:month", "average speed over the last 30 days"), [])

    def test_calendar_windows_have_no_length_to_ground(self):
        self.assertEqual(self.issues("cal:this_week", "average speed this week"), [])
        self.assertEqual(self.issues("cal:last_week", "average speed in the previous calendar week"), [])

    def test_no_period_stated_allows_only_the_documented_default(self):
        self.assertEqual(self.issues("last:7:day", "average speed"), [])
        self.assertTrue(self.issues("last:9:day", "average speed"))
        self.assertTrue(self.issues("last:45:day", "average speed"))

    def test_follow_up_inherits_the_window(self):
        ctx = self.A % "last:30:day"
        self.assertEqual(self.issues("last:30:day", "and for the motor temperature", ctx), [])

    def test_follow_up_replaces_the_window(self):
        ctx = self.A % "last:30:day"
        self.assertEqual(self.issues("last:7:day", "what about the last 7 days", ctx), [])
        self.assertTrue(self.issues("last:30:day", "what about the last 7 days", ctx))      # kept the old window although the user replaced it

    def test_invented_duration_is_rejected_even_with_context(self):
        ctx = self.A % "last:30:day"
        self.assertTrue(self.issues("last:12:day", "and for the motor temperature", ctx))

    def test_contradictory_period_is_rejected(self):
        self.assertTrue(self.issues("last:12:day", "average speed last week"))
        self.assertTrue(self.issues("last:5:day", "average speed over the last 12 days"))

    def test_second_window_of_a_comparison_is_grounded(self):
        plan = "op=window_change tgt=speed agg=mean t=last:7:day t2=prev:7:day"
        ok = LF.LearnedFront(_Brain(), None).check(Query.from_text(plan), "compare with the previous 7 days")
        self.assertEqual([i for i in ok if "period" in i or "window" in i], [])
        bad = LF.LearnedFront(_Brain(), None).check(Query.from_text(plan.replace("prev:7", "prev:21")), "compare with the previous 7 days")
        self.assertTrue([i for i in bad if "period" in i or "window" in i])

    def test_whole_month_is_the_policy_month(self):                                   # found by re-checking H20 ("how about over the whole month")
        self.assertEqual(self.issues("last:30:day", "how about over the whole month", self.A % "last:7:day"), [])
        self.assertTrue(self.issues("last:12:day", "how about over the whole month", self.A % "last:7:day"))

    def test_small_words_are_not_periods(self):
        self.assertEqual(self.issues("last:7:day", "and for the motor temperature"), [])        # "and" must not read as 1 day


@unittest.skipUnless(os.path.exists(ADAPTER) and os.path.exists(BASE) and __import__("importlib").util.find_spec("llama_cpp") is not None, "needs the query-plan adapter")
class ModelPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt.ir_slm import IrSlm
        cls.m = IrSlm(BASE, ADAPTER, threads=2, n_ctx=1536, mode="tuned")

    def plan(self, text, context=None):
        q, info = self.m.compile(text, context=context)
        self.assertIsNotNone(q, info)
        return q

    def conds(self, q): return {c.to_text() for c in q.scope.conditions}

    def test_torque_is_not_true(self):
        q = self.plan("what was the motor torque yesterday"); self.assertEqual(q.target, "motor_torque"); self.assertEqual(q.scope.time.to_text(), "cal:yesterday")

    def test_headwind_is_not_heading(self):
        q = self.plan("what was the headwind on my last trip"); self.assertNotEqual(q.target, "heading")

    def test_run_together_signal_name(self):
        q = self.plan("what is the coolantflow right now"); self.assertEqual(q.target, "coolant_flow")

    def test_slang_and_abbreviation(self):
        q = self.plan("wuz da battery temp yday"); self.assertEqual(q.target, "battery_temperature"); self.assertEqual(q.scope.time.to_text(), "cal:yesterday")

    def test_nite_is_night(self):
        q = self.plan("average speed at nite last week"); self.assertIn("night", self.conds(q))

    def test_can_bus_is_not_a_signal(self):
        q = self.plan("what is the CAN bus"); self.assertIn(q.op, ("none", "capability"))
        q = self.plan("can you show me my speed yesterday"); self.assertEqual(q.target, "speed")

    def test_traffic_lights_are_unsupported(self):
        q = self.plan("how many traffic lights did I stop at on my last trip"); self.assertEqual(q.op, "none")

    def test_follow_up_inherits_the_previous_plan(self):
        q = self.plan("and yesterday?", context="op=duration tgt=speed t=last:7:day p.thr=gt:90 unit=km_per_h")
        self.assertEqual((q.op, q.target, q.scope.time.to_text()), ("duration", "speed", "cal:yesterday"))

    def test_self_contained_question_ignores_unrelated_context(self):
        q = self.plan("how many minutes was the motor temperature above 50 C last week", context="op=exists tgt=speed t=cal:yesterday p.thr=gt:120 unit=km_per_h")
        self.assertEqual((q.op, q.target), ("duration", "motor_temperature"))

    # ---- independent variants: topics / wordings that are NOT in the training vocabulary (training/qp/families_extra.py)
    def test_unseen_unrecorded_topics_abstain(self):
        for text in ("how many stop signs did I drive past on my last trip", "what was the ferry crossing time yesterday", "how many car washes have I visited this month"):
            q, info = self.m.compile(text)                                   # abstain = the plan says none OR the model's own confidence is below the gate
            self.assertTrue(q is not None and (q.op == "none" or (info.get("min_p") or 1.0) < 0.8), (text, info.get("raw"), info.get("min_p")))

    def test_unknown_acronym_is_not_defined_by_guessing(self):
        q = self.plan("what is NVH"); self.assertEqual(q.op, "none")
        from vgpt.learned_front import definition
        self.assertIsNone(definition((q.params or {}).get("kind")))

    def test_definition_of_known_term_is_a_definition_not_a_signal(self):
        q = self.plan("could you tell me what SoH means"); self.assertEqual((q.op, q.params.get("why"), q.params.get("kind")), ("none", "definition", "soh"))
        q = self.plan("what is my SoH right now"); self.assertNotEqual(q.op, "none")        # the same word as a data question

    def test_percentile_is_one_operation(self):
        q = self.plan("what is the 90th percentile of the battery temperature last week"); self.assertEqual((q.op, q.params.get("pct")), ("percentile", "90"))
        q = self.plan("median coolant flow yesterday"); self.assertEqual((q.op, q.params.get("pct")), ("percentile", "50"))

    def test_negation_scope(self):
        q = self.plan("how many minutes was the speed above 90 km/h last week not counting the time it was charging")
        self.assertEqual(q.op, "duration"); self.assertIn("not:charging", self.conds(q))


if __name__ == "__main__": unittest.main()
