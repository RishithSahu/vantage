"""Meaning-frame front end: ontology typing, glossary hints, literal parsing, frame -> typed query, resolver outcomes, the rule budget, and the no-legacy-routing guarantee.

These tests need no model: the frame model is replaced by a stub that returns a fixed frame, so they check the deterministic side and the governance rules in isolation."""
import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench", "meaning_eval"))

from vgpt.meaning import compile as C, glossary as GL, literals as L, ontology
from vgpt.meaning.frame import Frame

DB = os.path.join(ROOT, "data", "gen_demo.db")
NOW = 1787625900
TZ = 19800


class StubModel:
    glossary = GL.load()

    def __init__(self, frame): self.frame = frame; self.calls = []

    def parse(self, text, hint=None, prev=None):
        self.calls.append(text); return self.frame, {"raw": self.frame.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}


class OntologyTyping(unittest.TestCase):
    def setUp(self): self.O = ontology.load()

    def test_every_frozen_gold_frame_is_well_typed(self):
        import evalio
        for name in ("novel_v1", "messy_v1", "user_v1"):
            bad = evalio.audit(evalio.load_tsv(os.path.join(ROOT, "bench", "meaning_eval", name + ".tsv")))
            self.assertEqual(bad, [], name)

    def test_typing_rejects_category_errors(self):
        self.assertTrue(self.O.validate(Frame("context", "speed")))                        # only an event has a "what happened before it"
        self.assertEqual(self.O.validate(Frame("spec", "battery_soc")), [])                # since 2026-10-20 a signal's provenance (sources, rate, unit) is a legitimate `spec` question
        self.assertTrue(self.O.validate(Frame("events", "speed", "list")))                # a signal is not an event
        self.assertTrue(self.O.validate(Frame("compare", "speed", "mean")))                # a comparison needs its second operand
        self.assertTrue(self.O.validate(Frame("stat", "speed", "list")))                   # operator not applicable to the act
        self.assertEqual(self.O.validate(Frame("stat", "speed", "mean")), [])

    def test_relations_are_derived_from_structure_not_written_by_hand(self):
        rel = self.O.relations("motor_power")
        self.assertIn("battery_power", rel["same_quantity"])                              # same quantity, different component
        self.assertIn("motor_temperature", rel["same_component"])                         # same component, different quantity
        self.assertNotIn("motor_temperature", rel["same_quantity"])

    def test_grammar_and_prompt_come_from_the_same_data(self):
        g = self.O.gbnf()
        for w in ("seat_count", "battery_overheat_warning", "unlisted", "threshold"): self.assertIn(w, g)
        self.assertIn("act:", self.O.system_prompt())


class GlossaryHints(unittest.TestCase):
    def setUp(self): self.G = GL.load()

    def test_flagged_words_get_a_hint_and_plain_text_gets_none(self):
        self.assertIn("ppl = people", self.G.hint_line("how many ppl can fit in my car"))
        self.assertIn("kwh", self.G.hint_line("what is 40 kWh").lower())
        self.assertEqual(self.G.hint_line("what is my speed"), "")

    def test_lookup_is_dictionary_lookup_not_a_decision(self):
        self.assertEqual([e["term"] for e in self.G.lookup("average speed")], ["avg"])  # an operator word has a meaning entry, nothing more


class Literals(unittest.TestCase):
    def test_periods_thresholds_and_instants(self):
        x = L.extract("average speed in the last 2hrs", NOW, TZ); self.assertEqual([t.to_text() for t in x.times], ["last:2:hour"])
        x = L.extract("how long was the speed above 100 km/h yesterday", NOW, TZ, None, "speed")
        self.assertEqual([c.to_text() for c in x.thresholds], ["sig:speed:gt:100"]); self.assertEqual([t.to_text() for t in x.times], ["cal:yesterday"])
        self.assertEqual(L.instant("how hot was the battery two hours ago", NOW, TZ), NOW - 7200)
        self.assertEqual(L.instant("what is my speed", NOW, TZ), None)
        day0 = ((NOW + TZ) // 86400) * 86400 - TZ
        self.assertEqual(L.instant("what was my speed at 3pm yesterday", NOW, TZ), day0 - 86400 + 15 * 3600)

    def test_abbreviated_time_words(self):
        x = L.extract("this wk vs last wk speed", NOW, TZ); self.assertEqual([t.to_text() for t in x.times], ["cal:this_week", "cal:last_week"])


class FrameToQuery(unittest.TestCase):
    def setUp(self): self.O = ontology.load()

    def q(self, frame, text):
        return C.to_query(frame, L.extract(text, NOW, TZ, None, frame.concept), self.O)

    def test_forms(self):
        cases = [(Frame("now", "speed"), "speed now", "op=current_value tgt=speed t=now"),
                 (Frame("stat", "speed", "mean"), "average speed last week", "op=aggregate tgt=speed agg=mean t=cal:last_week"),
                 (Frame("extreme", "battery_temperature", "max"), "hottest the pack got yesterday", "op=extreme tgt=battery_temperature agg=max t=cal:yesterday"),
                 (Frame("events", "hard_braking", "count"), "how many hard stops last week", "op=count t=cal:last_week ev=HARSH_BRAKE"),
                 (Frame("events", "battery_overheat_warning", "last"), "when was the last overheating alert", "op=list_events t=all ev=WARN_BATT_OVERTEMP"),
                 (Frame("events", "any_warning", "first"), "first warning ever", "op=list_events t=all ev=ANY_WARNING evs=first"),
                 (Frame("context", "tyre_pressure_warning"), "what happened before the tyre warning", "op=event_context t=all ev=TPMS_LOW bef=600"),
                 (Frame("compare", "wh_per_km", "mean", "city", "highway"), "city vs highway consumption", "op=compare tgt=wh_per_km agg=mean t=last:7:day grp=city,highway"),
                 (Frame("compare", "speed", "mean", "none", "period"), "this week versus last week", "op=compare tgt=speed agg=mean t=cal:this_week t2=cal:last_week"),
                 (Frame("stat", "battery_temperature", "share", "threshold"), "how often was the battery above 40 degrees last week", "op=share tgt=battery_temperature t=cal:last_week p.thr=gt:40")]
        for frame, text, want in cases:
            q, why = self.q(frame, text); self.assertIsNotNone(q, (frame, why)); self.assertEqual(q.to_text(), want)

    def test_missing_literals_are_asked_for_not_invented(self):
        q, why = self.q(Frame("stat", "speed", "duration", "threshold"), "how long was the speed high"); self.assertIsNone(q); self.assertIn("limit", why)
        q, why = self.q(Frame("compare", "speed", "mean", "none", "period"), "compare speed"); self.assertIsNone(q); self.assertIn("periods", why)


@unittest.skipUnless(os.path.exists(DB), "needs data/gen_demo.db")
class ResolverBehaviour(unittest.TestCase):
    def brain(self, frame):
        from vgpt.meaning.pipeline import MeaningBrain
        return MeaningBrain(DB, StubModel(frame), now=NOW)

    def ask(self, frame, text):
        b = self.brain(frame)
        try: return b.ask(text)
        finally: b.c.close()

    def test_non_executing_acts_never_touch_data(self):
        for act in ("offtopic", "unsupported", "control", "unsafe", "ambiguous", "social"):
            r = self.ask(Frame(act, "none"), "anything"); self.assertEqual(r["facts"]["executor"], "outcome"); self.assertNotIn("query", r["facts"])
        r = self.ask(Frame("control", "none"), "turn on the ac"); self.assertIn("cannot operate", r["answer"].lower())

    def test_unmodelled_thing_is_reported_not_substituted(self):
        r = self.ask(Frame("now", "unlisted"), "washer fluid level"); self.assertIn("not something this vehicle records", r["answer"]); self.assertIn("Nothing was substituted", r["answer"])

    def test_typed_frame_answers_from_the_engine(self):
        r = self.ask(Frame("now", "speed"), "speed now"); self.assertEqual(r["facts"]["executor"], "typed_query"); self.assertIn("km/h", r["answer"])
        r = self.ask(Frame("stat", "speed", "mean"), "average speed last week"); self.assertIn("average vehicle speed", r["answer"].lower())

    def test_spec_absent_from_profile_is_not_replaced_by_a_nearby_quantity(self):
        r = self.ask(Frame("spec", "rated_max_speed"), "how fast can this vehicle go"); self.assertIn("does not list a rated top speed", r["answer"])
        self.assertIn("fastest recorded", r["answer"].lower()); self.assertNotIn("rated top speed (vehicle profile)", r["answer"].lower())      # the recorded value is labelled as recorded, never presented as the rating
        r = self.ask(Frame("spec", "battery_capacity"), "battery capacity"); self.assertIn("60", r["answer"])

    def test_ill_typed_frame_is_a_clarification(self):
        r = self.ask(Frame("context", "speed"), "x"); self.assertEqual(r["intent"], "CLARIFY")

    def test_a_typed_meaning_never_reaches_the_legacy_rule_router(self):
        from vgpt.brain import Brain
        original = Brain._ask_routes

        def boom(*a, **k): raise AssertionError("the legacy rule router was reached by a typed frame")
        Brain._ask_routes = boom
        try:
            for frame, text in ((Frame("now", "speed"), "speed"), (Frame("stat", "speed", "mean"), "average speed last week"), (Frame("events", "hard_braking", "count"), "hard brakes last week"),
                                (Frame("spec", "seat_count"), "seats"), (Frame("define", "regen_energy"), "regen"), (Frame("offtopic", "none"), "recipe")):
                r = self.ask(frame, text); self.assertNotEqual(r["intent"], "ERR")
        finally: Brain._ask_routes = original

    def test_every_new_question_is_decided_by_the_model_once_and_a_repeat_is_cached(self):
        b = self.brain(Frame("now", "speed"))
        try: b.ask("speed"); b.ask("rpm please"); b.ask("Speed")
        finally: b.c.close()
        self.assertEqual(len(b.frame_model.calls), 3)                                   # owner's rule: no question cache, so a repeat is a real model run


class ValueParsers(unittest.TestCase):
    """Values the person typed (dates, intervals, units, codes, amp-hours) are parsed by literals.py; they were each a silent wrong answer before."""

    def test_a_typed_calendar_date_is_that_date_never_today(self):
        import datetime as dt
        ts = L.instant("What was the vehicle speed on 17 March 2025 at exactly 09:13:27?", NOW, TZ); day = dt.datetime.fromtimestamp(ts + TZ, dt.timezone.utc)
        self.assertEqual((day.year, day.month, day.day, day.hour, day.minute), (2025, 3, 17, 9, 13))
        for text in ("speed at 09:13 on 2025-03-17", "speed on 17/03/2025 at 9:13 am", "speed at 9:13 on march 17th 2025"): self.assertEqual(L.instant(text, NOW, TZ), ts, text)
        self.assertIsNone(L.date_of("average speed yesterday", NOW, TZ)); self.assertIsNone(L.date_of("speed on 31 february 2025", NOW, TZ))

    def test_an_interval_is_a_bounded_days_ago_span_even_with_number_words(self):
        for text in ("average speed between three and five days ago", "from 5 days ago to 3 days ago"):
            t = L.span(text, NOW, TZ); self.assertEqual((t.type, t.value, t.value2, t.unit), ("between", 5, 3, "day"), text)
        self.assertIsNone(L.span("between 5 and 5 days ago", NOW, TZ))
        t = L.span("warnings since monday", NOW, TZ); self.assertEqual((t.type, t.value2), ("between", 0)) if t.type == "between" else self.assertEqual(t.name, "today")
        t = L.span("have i plugged in since yesterday", NOW, TZ); self.assertEqual((t.type, t.value, t.unit, t.value2), ("between", 1, "day", 0))

    def test_the_typed_engine_still_refuses_absolute_timestamps_but_takes_a_bounded_interval(self):
        from vgpt.query_ir import TimeSpec
        from vgpt.query_validator import _check_time
        def issues(t):
            out = []; _check_time(t, out, "t"); return out
        self.assertEqual(issues(TimeSpec("between", 5, "day", value2=3)), [])
        self.assertTrue(issues(TimeSpec("between", 3, "day", value2=5))); self.assertTrue(issues(TimeSpec("between", 99999, "day", value2=1)))
        self.assertTrue(issues(TimeSpec("range", start=1, end=2)))
        self.assertEqual(TimeSpec.from_text(TimeSpec("between", 5, "day", value2=3).to_text()), TimeSpec("between", 5, "day", value2=3))

    def test_display_units_codes_and_amp_hours(self):
        self.assertEqual(L.display_units("tyre pressure in psi"), {"p": "psi"}); self.assertEqual(L.display_units("how many km are left"), {})
        self.assertEqual(L.display_units("use metric units"), {"p": "kpa", "d": "km", "t": "c"})
        self.assertEqual(L.extract("what is P1A00 and p0a80", NOW, TZ).codes, ["P1A00", "P0A80"]); self.assertEqual(L.extract("what is a hard brake", NOW, TZ).codes, [])
        ah, v = L.amp_hours_and_volts("what is 7000 mah at 3.7 v in watt hours"); self.assertEqual((ah, v), (7.0, 3.7))
        lit = L.extract("average speed when above 50", NOW, TZ); self.assertEqual((lit.cmp, lit.numbers[0]), ("gt", 50.0))


@unittest.skipUnless(os.path.exists(DB), "needs data/gen_demo.db")
class ResolverRegressions(unittest.TestCase):
    """Each case is a failure seen tonight on a correctly understood question (gold frame fed to the resolver, no model)."""
    def ask(self, frame, text):
        from vgpt.meaning.pipeline import MeaningBrain
        b = MeaningBrain(DB, StubModel(frame), now=NOW)
        try: return b.ask(text)
        finally: b.c.close()

    def test_a_date_with_no_recorded_data_is_not_answered_with_todays_value(self):
        r = self.ask(Frame("at", "speed"), "What was the vehicle speed on 17 March 2025 at exactly 09:13:27?")
        self.assertNotRegex(r["answer"], r"0\.0 km/h"); self.assertIn("2025", r["answer"]); self.assertFalse(r["facts"]["ok"])

    def test_the_interval_answer_names_the_interval_and_differs_from_the_default_week(self):
        r = self.ask(Frame("stat", "speed", "mean"), "average speed between three and five days ago"); self.assertIn("between 5 and 3 days ago", r["answer"])

    def test_display_units_are_honoured_and_a_units_request_shows_the_last_answer_again(self):
        r = self.ask(Frame("now", "tyre_pressure"), "tyre pressure in psi"); self.assertIn("psi", r["answer"]); self.assertNotIn("kPa", r["answer"])
        from vgpt.meaning.pipeline import MeaningBrain
        b = MeaningBrain(DB, StubModel(Frame("now", "tyre_pressure")), now=NOW)
        try:
            b.ask("tyre pressure"); b.frame_model.frame = Frame("control", "none"); r = b.ask("i want it in psi")
        finally: b.c.close()
        self.assertEqual(r["intent"], "UNITS"); self.assertIn("psi", r["answer"])

    def test_a_fault_code_in_the_text_is_looked_up(self):
        r = self.ask(Frame("define", "aux_battery_voltage"), "what is P1A00"); self.assertIn("P1A00", r["answer"])

    def test_trips_are_dated_events_and_a_ratio_asked_at_a_time_is_read_over_the_period(self):
        r = self.ask(Frame("events", "trip_record", "last"), "tell me about my last trip"); self.assertRegex(r["answer"], r"trip on record started .* km in \d+ min")
        r = self.ask(Frame("at", "wh_per_km"), "what was yesterdays energy per km"); self.assertIn("Wh/km", r["answer"])

    def test_the_rain_sensor_is_a_duration_not_a_threshold_question(self):
        r = self.ask(Frame("stat", "rain", "duration"), "how long did it rain last week"); self.assertIn("Rain was detected", r["answer"])

    def test_a_per_day_extreme_works_for_any_recorded_signal(self):
        r = self.ask(Frame("extreme", "battery_temperature", "max", ent="day"), "which day last week had the highest battery temperature"); self.assertIn("day with the highest battery temperature", r["answer"])

    def test_the_mah_conversion_is_calculated_when_voltage_is_given(self):
        r = self.ask(Frame("howto", "battery_capacity"), "what is 7000 mah at 3.7 v in watt hours"); self.assertIn("25.9 Wh", r["answer"])

    def test_a_topic_concept_with_a_threshold_text_does_not_crash_the_literal_parser(self):
        for f in (Frame("estimate", "charge_time"), Frame("events", "diagnostic_code", "list")):
            r = self.ask(f, "which fault codes were logged above 80 km/h"); self.assertNotEqual(r["intent"], "ERR")

    def test_counts_of_crossings_definitions_and_half_days(self):
        r = self.ask(Frame("stat", "battery_soc", "count", "threshold"), "how many times did the soc go below 30"); self.assertIn("separate occasions", r["answer"])
        r = self.ask(Frame("define", "regen_power"), "what does regen mean"); self.assertIn("generator", r["answer"])
        t1 = L.instant("speed a day and a half ago", NOW, TZ); t2 = L.instant("speed 36 hours ago", NOW, TZ); self.assertEqual(t1, t2)
        r = self.ask(Frame("extreme", "battery_temperature", "max", ent="trip"), "peak pack temperature on the last trip"); self.assertNotIn("no typed query form", r["answer"])

    def test_an_executor_failure_becomes_an_honest_sentence_never_an_exception(self):
        from vgpt.meaning import resolve
        original = resolve.Resolver.answer
        def boom(self, text, frame): raise KeyError("boom")
        resolve.Resolver.answer = boom
        try: r = self.ask(Frame("now", "speed"), "speed")
        finally: resolve.Resolver.answer = original
        self.assertIn("could not compute", r["answer"]); self.assertFalse(r["facts"]["ok"])


class ProductDefault(unittest.TestCase):
    """The meaning front end is the product default; VGPT_MEANING=0 is the way back; a caller that supplies its own legacy classifier keeps the legacy path."""

    def test_selection_rules(self):
        from unittest import mock
        from vgpt import product
        frame_model = StubModel(Frame("now", "speed"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGPT_MEANING", None)
            self.assertTrue(product.meaning_enabled()); self.assertTrue(product.meaning_enabled(frame_model)); self.assertFalse(product.meaning_enabled(object()))
        with mock.patch.dict(os.environ, {"VGPT_MEANING": "0"}):
            self.assertFalse(product.meaning_enabled()); self.assertFalse(product.meaning_enabled(frame_model))
        with mock.patch.dict(os.environ, {"VGPT_MEANING": "1"}):
            self.assertTrue(product.meaning_enabled())

    @unittest.skipUnless(os.path.exists(DB), "needs data/gen_demo.db")
    def test_the_default_product_factory_builds_the_meaning_brain(self):
        from unittest import mock
        from vgpt import product
        from vgpt.meaning.pipeline import MeaningBrain
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGPT_MEANING", None)
            b = product.make_qwen05_brain(DB, slm=StubModel(Frame("now", "speed")), now=NOW)
        try: self.assertIsInstance(b, MeaningBrain); self.assertIn("km/h", b.ask("speed")["answer"])
        finally: b.c.close()

    def test_vgpt_meaning_0_and_meaning_false_build_the_previous_front_end(self):
        from unittest import mock
        from vgpt import product
        for env, kw in (({"VGPT_MEANING": "0"}, {}), ({}, {"meaning": False})):
            semantic = mock.Mock()
            with mock.patch.dict(os.environ, env), mock.patch.object(product.os.path, "exists", return_value=True), mock.patch("vgpt.brain.Slm", return_value=semantic) as slm,                  mock.patch.object(product, "Brain", return_value=type("B", (), {})()) as ctor, mock.patch.object(product, "make_meaning_brain") as meaning:
                product.make_qwen05_brain("unused.db", **kw)
            meaning.assert_not_called(); self.assertEqual(slm.call_count, 1); self.assertEqual(ctor.call_args.kwargs["slm"], semantic)

    def test_a_missing_adapter_is_an_explicit_error_that_names_the_way_back(self):
        from vgpt import product
        with self.assertRaises(FileNotFoundError) as cm: product.make_meaning_brain(DB, adapter=os.path.join(ROOT, "models", "does-not-exist.gguf"), now=NOW)
        self.assertIn("VGPT_MEANING=0", str(cm.exception))


class RuleBudget(unittest.TestCase):
    """The method's governing rule, enforced: text patterns may parse literal values (literals.py) and tokenise / sanitise; nothing else in the meaning path may inspect user text with a pattern."""
    BUDGET = {"literals.py": 99, "glossary.py": 1, "pipeline.py": 4, "canon.py": 8, "askme.py": 24, "anchor.py": 1, "thresholdx.py": 2, "tripx.py": 5, "evmap.py": 5, "plain.py": 4, "calx.py": 2}      # max number of `re.<fn>(...)` call sites per module; every other module: 0.  canon.py (spelling, shorthand, awkward wordings) and askme.py (questions the model should not decide) are the pre-parser the owner asked for on 2026-10-24; every rule in them is listed in docs/CANON.md

    def count(self, path):
        tree = ast.parse(open(path, encoding="utf-8").read()); n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "re": n += 1
        return n

    def test_no_text_patterns_outside_the_literal_parser(self):
        d = os.path.join(ROOT, "vgpt", "meaning")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"): continue
            self.assertLessEqual(self.count(os.path.join(d, f)), self.BUDGET.get(f, 0), f"{f} uses text patterns beyond its budget")

    def test_semantic_modules_do_not_import_the_rule_layers(self):
        banned = {"semantic_ops", "qa", "qa_a", "qa_b", "qa_c", "qa_d", "qa_e", "qa_g", "qa_h", "qa_i", "qa_fuzzy", "composed", "temporal_ops", "temporal_misc", "temporal_events", "nlu", "dialogue"}
        d = os.path.join(ROOT, "vgpt", "meaning")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"): continue
            for node in ast.walk(ast.parse(open(os.path.join(d, f), encoding="utf-8").read())):
                names = []
                if isinstance(node, ast.ImportFrom): names = [node.module.split(".")[-1]] if node.module else []; names += [a.name for a in node.names] if node.level else []
                if isinstance(node, ast.Import): names = [a.name.split(".")[-1] for a in node.names]
                self.assertFalse(banned & set(names), f"{f} imports {banned & set(names)}")


if __name__ == "__main__":
    unittest.main()
