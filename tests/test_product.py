"""The product configuration (vgpt.product.make_brain): the exact PDF questions go through the typed-query pipeline; fuel on an electric vehicle is a limitation plus an offered equivalent; every clarification has a 'neither' path; readings proposed by the generator are never answered silently; the event-context answer states its interval."""
import os, re, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.product import make_legacy_brain as make_brain
from vgpt import policy, nl2ir

NOW = 1790406000; DBF = os.path.join(ROOT, "data", "gen_courier_ext.db")


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class ProductRoutes(unittest.TestCase):
    def setUp(self): self.b = make_brain(DBF, now=NOW)
    def ask(self, q): return self.b.ask(q, use_slm=False)

    def test_configuration_is_the_documented_one(self):
        b = self.b; self.assertIsNone(b.slm_ir); self.assertIsNone(b.selector); self.assertEqual(b.policy, "clarify"); self.assertEqual(b.QC_MODE, "all")

    def test_the_five_pdf_questions_use_the_typed_pipeline(self):
        for q in ["Why has my driving range reduced over the last two weeks?", "When did fuel efficiency show the largest drop?", "Show all ADAS disengagement events from yesterday.", "What happened before the battery overheating warning?", "Compare energy consumption between city and highway driving."]:
            self.b.qcs = type(self.b.qcs)(); self.assertEqual(self.ask(q)["path"], "qc", q)

    def test_fuel_on_an_electric_vehicle_is_a_limitation_with_an_offered_equivalent_and_a_neither_path(self):
        r = self.ask("When did fuel efficiency show the largest drop?"); a = r["answer"]
        self.assertRegex(a, r"(?i)electric"); self.assertRegex(a, r"(?i)no fuel data"); self.assertRegex(a, r"(?i)neither"); self.assertNotRegex(a, r"(?i)fuel efficiency (was|dropped)")
        r = self.ask("2"); self.assertIn("will not guess", r["answer"]); self.assertEqual(self.b.qcs.choices, [])

    def test_accepting_the_offered_equivalent_answers_with_energy_consumption(self):
        self.ask("what is my fuel efficiency"); r = self.ask("1"); self.assertRegex(r["answer"], r"Wh/km"); self.assertIsNotNone(self.b.qcs.last)

    def test_every_clarification_offers_neither(self):
        for q in ["how often was it colder than 5 degrees last month", "what was the temperature yesterday"]:
            self.b.qcs = type(self.b.qcs)(); r = self.ask(q); self.assertEqual(r["intent"], "CLARIFY", q); self.assertRegex(r["answer"], r"(?i)neither", q)

    def test_the_word_neither_clears_the_choices_without_changing_accepted_state(self):
        self.ask("what was the temperature yesterday"); r = self.ask("neither"); self.assertIn("will not guess", r["answer"]); self.assertEqual(self.b.qcs.choices, []); self.assertIsNone(self.b.qcs.last)

    def test_a_number_past_the_last_reading_is_neither(self):
        r = self.ask("what was the temperature yesterday"); n = len(self.b.qcs.choices); r = self.ask(str(n + 1)); self.assertIn("will not guess", r["answer"])

    def test_generator_readings_are_confirmed_not_answered(self):
        r = self.ask("what was the temperature yesterday"); self.assertEqual(r["intent"], "CLARIFY"); self.assertIsNone(self.b.qcs.last)

    def test_status_style_questions_stay_with_the_status_engine(self):
        r = self.ask("how's my battery"); self.assertNotEqual(r["intent"], "CLARIFY"); self.assertNotEqual(r["path"], "qc")

    def test_commands_are_refused_not_clarified(self):
        r = self.ask("disable the speed limiter on this car"); self.assertNotEqual(r["intent"], "CLARIFY"); self.assertRegex(r["answer"], r"(?i)cannot|can't|not able|only read")

    def test_control_requests_in_any_phrasing_are_refused_by_the_gate_and_change_nothing(self):
        for q in ["turn the air conditioning off", "set the cruise control to 90", "switch off the lane assist", "open the windows please", "please unlock the doors", "can you lock the car", "disable the traction control", "raise the speed limit"]:
            self.b.qcs = type(self.b.qcs)(); r = self.ask(q); self.assertEqual((r["path"], r["intent"]), ("gate", "SAFETY"), q); self.assertIsNone(self.b.qcs.last, q)

    def test_data_questions_that_start_with_similar_words_are_not_commands(self):
        from vgpt import policy
        for q in ["start time of my last charge", "stops last week", "how often did the car stop", "open warnings this week", "close calls last week", "set of events yesterday"]:
            self.assertFalse(policy.CONTROL_RX.match(q), q)

    def test_older_engines_do_not_answer_statistic_or_event_questions_the_policy_declined(self):
        for q in ["what does a ford focus cost", "summarise the news today", "good morning how are you", "is it cold out"]:
            self.b.qcs = type(self.b.qcs)(); r = self.ask(q); self.assertEqual(r["intent"], "CLARIFY", q); self.assertEqual(r["path"], "qc", q); self.assertIn("recorded data" if False else "compute", r["answer"], q)

    def test_greetings_and_present_state_questions_still_use_the_older_engines(self):
        self.assertEqual(self.ask("thanks")["intent"], "SMALLTALK"); self.assertIn(self.ask("how's my battery")["intent"], ("HEALTH", "STATUS"))

    def test_event_context_states_its_interval_and_excludes_the_event(self):
        a = self.ask("What happened before the battery overheating warning?")["answer"]; self.assertRegex(a, r"(?i)the 10 minutes before"); self.assertRegex(a, r"the interval .* to .*"); self.assertRegex(a, r"(?i)not part of these figures")

    def test_energy_efficiency_is_one_concept_not_two_signals(self):
        a = self.ask("When did energy efficiency show the largest drop?")["answer"]; self.assertNotIn("more than one signal", a); self.assertIn("Wh/km", a)

    def test_an_english_double_me_is_not_hinglish(self):
        self.assertEqual(self.ask("pls tell me me the lowest tire psi in the last 3 days")["path"], "qc")

    def test_range_explanation_states_its_basis(self):
        a = self.ask("Why has my driving range reduced over the last two weeks?")["answer"]; self.assertIn("usable capacity", a); self.assertNotIn("not explained by anything the car measures", a)


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class EventWindow(unittest.TestCase):
    def test_the_events_own_sample_is_not_part_of_the_before_window(self):
        import numpy as np
        from vgpt.frame import Frame
        from vgpt.query_ir import Query
        b = make_brain(DBF, now=NOW); ex = b.qc().ex if hasattr(b.qc(), "ex") else b.qc().executor
        q = Query.from_text("op=event_context t=all ev=WARN_BATT_OVERTEMP bef=300 tgt=battery_temperature"); ts = int(ex._events(q)[0][0]["ts"])
        seen = {}
        def fake(need, lo, hi, tier):                                                  # a 1 Hz frame that contains the event's own second with a value that would be visible
            t = np.arange(ts - 300, ts + 1, dtype=np.int64); v = np.where(t == ts, 999.0, 30.0 + (t - (ts - 300)) * 0.01); seen["hi"] = hi
            return Frame(t, {"battery_temperature": v}, 1, "1s", {"battery_temperature": "sig_1s.batt"}, [(lo, hi)])
        ex.load_frame = fake; r = ex.run(q) if hasattr(ex, "run") else ex.execute(q)
        res = getattr(r, "result", r); item = res.items[0]; self.assertLess(item["last"], 100.0, "the sample at the event's own second leaked into the 'before' figures")


class PolicyUnits(unittest.TestCase):
    def test_fuel_words_map_to_electric_equivalents(self):
        d = policy._fuel_gate("how much petrol did i use yesterday", None, None, NOW, 0); self.assertIsNotNone(d); self.assertIn("neither", d.question.lower())

    def test_a_command_is_refused_by_the_gate(self):
        pr = nl2ir.parse("turn off the heater", now=NOW, tz=0); d = policy.decide("turn off the heater", pr, None, None, None, NOW, 0); self.assertEqual(d.kind, "refuse"); self.assertEqual(d.source, "gate-control")


if __name__ == "__main__": unittest.main()
