"""Dialogue layer, definition policy and glossary arbitration of the meaning front end (no model: a stub returns the frame the learned model returned for each wording).

Regression source: the owner's own test session of 2026-10-19 (follow-ups lost, 'what is tyre pressure' answered with 'the tyres on the four wheels', 'break pad temperature' answered as
brake-pad wear, 'how did you get that' answered with a greeting)."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import glossary as GL, literals as L, ontology
from vgpt.meaning.arbitrate import arbitrate
from vgpt.meaning.frame import Frame
from vgpt.meaning.pipeline import MeaningBrain

DB = os.path.join(ROOT, "data", "gen_courier.db")
NOW = 1790352000                                    # 2026-09-25 16:00 UTC, the end of the generated history
F = Frame


class Scripted:
    """Returns the frame the real model returned for a wording; anything else is `ambiguous` (what the model says about a fragment)."""
    glossary = GL.load()

    def __init__(self, table): self.table = {k.lower(): v for k, v in table.items()}

    def parse(self, text, hint=None, prev=None):
        f = self.table.get(text.lower().strip(), F("ambiguous", "none"))
        return f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}


TABLE = {
    "how many hard brakes last week": F("events", "hard_braking", "count"),
    "how many adas disengagements last week": F("events", "adas_disengagement", "count"),
    "and the week before that": F("ambiguous", "none"), "what about yesterday": F("ambiguous", "none"), "i want to know for just yesterday": F("ambiguous", "none"),
    "which one was the hardest": F("extreme", "acceleration", "max"), "which one was the longest": F("extreme", "distance_driven", "max"),
    "why did it happen": F("why", "any_warning"), "how did you get that": F("social", "none"), "how did you answer the questions give me explanation for it": F("social", "none"),
    "what was the average speed last week": F("stat", "speed", "mean"), "what about the motor temperature": F("why", "motor_temperature"),
    "how fast was i going then": F("at", "speed"), "what was the battery temperature then": F("context", "battery_temperature"),
    "what is my tyre pressure": F("now", "tyre_pressure"), "what about the front left": F("now", "unlisted"), "and the rear right": F("ambiguous", "none"),
    "what is my front left tyre pressure": F("now", "tyre_pressure"),
    "what is tyre pressure": F("define", "tyre_pressure"), "what is coolant flow": F("define", "coolant_flow"), "what is turning radius": F("define", "turn_radius"),
    "what is break pad temps": F("define", "brake_pad_life"), "what is my brake pad temperature": F("now", "brake_pad_life"), "how worn are my brake pads": F("now", "brake_pad_life"),
    "how much battery is left": F("now", "battery_soc"),
}


def brain():
    return MeaningBrain(DB, Scripted(TABLE), now=NOW)


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class FollowUps(unittest.TestCase):
    def test_a_period_only_follow_up_keeps_the_previous_question(self):
        b = brain(); b.ask("how many hard brakes last week"); r = b.ask("i want to know for just yesterday")
        self.assertEqual(r["facts"]["frame"]["concept"], "hard_braking"); self.assertEqual(r["facts"]["dialogue"]["kind"], "period")
        self.assertIn("yesterday", r["answer"]); self.assertNotIn("not sure which signal", r["answer"])

    def test_a_period_relative_to_the_previous_one_is_computed_from_the_previous_window(self):
        b = brain(); first = b.ask("how many adas disengagements last week"); r = b.ask("and the week before that")
        lo, hi = first["facts"]["evidence"]["period_ts"]; self.assertEqual(r["facts"]["dialogue"]["kind"], "period")
        got = r["facts"]["evidence"]["period_ts"]; self.assertEqual(got[1], lo); self.assertAlmostEqual(got[1] - got[0], 7 * 86400, delta=3600)    # the 7 days that END where the last window began

    def test_a_chain_of_periods_keeps_the_topic(self):
        b = brain(); b.ask("how many adas disengagements last week"); b.ask("what about yesterday"); r = b.ask("and the week before that")
        self.assertEqual(r["facts"]["frame"]["concept"], "adas_disengagement")

    def test_which_one_picks_from_the_events_just_listed(self):
        b = brain(); b.ask("how many hard brakes last week"); r = b.ask("which one was the hardest")
        self.assertEqual(r["facts"]["route"], "dialogue:ordinal"); self.assertIn("0.", r["answer"]); self.assertTrue(r["facts"]["ok"])

    def test_last_week_is_a_period_not_an_ordinal(self):
        b = brain(); b.ask("how many adas disengagements last week"); r = b.ask("what was the average speed last week")
        self.assertNotEqual(r["facts"].get("route"), "dialogue:ordinal"); self.assertEqual(r["facts"]["frame"]["concept"], "speed")

    def test_it_refers_to_the_event_the_last_answer_picked(self):
        b = brain(); picked = b.ask("how many hard brakes last week"); one = b.ask("which one was the hardest"); r = b.ask("why did it happen")
        self.assertEqual(r["facts"]["frame"]["concept"], "hard_braking"); self.assertEqual(r["facts"]["dialogue"]["kind"], "referent")
        stamp = one["answer"].split(": ", 1)[1].split(" - ")[0]; self.assertIn(stamp.replace(" ", " ").split(",")[0], r["answer"])

    def test_then_is_the_instant_of_the_previous_answer(self):
        b = brain(); b.ask("how many hard brakes last week"); b.ask("which one was the hardest"); r = b.ask("how fast was i going then")
        self.assertEqual(r["facts"]["dialogue"]["kind"], "deixis"); self.assertIn("km/h", r["answer"])
        r2 = b.ask("what was the battery temperature then"); self.assertEqual(r2["facts"]["frame"]["act"], "at"); self.assertIn("°C", r2["answer"])

    def test_a_concept_swap_keeps_the_period_and_statistic(self):
        b = brain(); b.ask("what was the average speed last week"); r = b.ask("what about the motor temperature")
        self.assertEqual(r["facts"]["frame"]["act"], "stat"); self.assertEqual(r["facts"]["frame"]["op"], "mean"); self.assertIn("motor temperature", r["answer"].lower())

    def test_how_did_you_get_that_explains_the_last_answer(self):
        b = brain(); b.ask("how much battery is left"); r = b.ask("how did you get that")
        self.assertEqual(r["intent"], "EXPLAIN"); self.assertIn("typed query", r["answer"]); self.assertIn("state of charge", r["answer"].lower()); self.assertNotIn("Hello", r["answer"])
        r = b.ask("how did you answer the questions give me explanation for it"); self.assertEqual(r["intent"], "EXPLAIN")

    def test_provenance_with_nothing_to_explain_says_so(self):
        r = brain().ask("how did you get that"); self.assertIn("no earlier answer", r["answer"])

    def test_a_fresh_question_is_not_treated_as_a_follow_up(self):
        b = brain(); b.ask("how many hard brakes last week"); r = b.ask("how much battery is left")
        self.assertNotIn("dialogue", r["facts"]); self.assertEqual(r["facts"]["frame"]["concept"], "battery_soc")


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class WheelPosition(unittest.TestCase):
    def test_the_wheel_is_a_typed_value(self):
        for t, pos in (("front left tyre", "fl"), ("the rear-right one", "rr"), ("rl pressure", "rl"), ("my fr tyre", "fr"), ("coolant flow", None), ("fleet average", None)):
            self.assertEqual(L.wheel_position(t), pos, t)

    def test_a_named_wheel_reads_that_sensor(self):
        r = brain().ask("what is my front left tyre pressure"); self.assertIn("Front-left", r["answer"])

    def test_a_wheel_only_follow_up_keeps_the_tyre_question(self):
        b = brain(); b.ask("what is my tyre pressure"); r = b.ask("what about the front left"); self.assertIn("Front-left", r["answer"]); r = b.ask("and the rear right"); self.assertIn("Rear-right", r["answer"])


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class DefinitionPolicy(unittest.TestCase):
    def test_a_measured_quantity_is_defined_and_read(self):
        r = brain().ask("what is tyre pressure"); self.assertIn("kilopascals", r["answer"]); self.assertIn("kPa", r["answer"].split("\n", 1)[1]); self.assertNotIn("four wheels", r["answer"])

    def test_the_reading_is_the_present_value_not_a_second_definition(self):
        r = brain().ask("what is coolant flow"); self.assertIn("litres per minute", r["answer"]); self.assertIn("L/min", r["answer"])

    def test_an_unrecorded_quantity_is_defined_and_says_it_is_not_recorded(self):
        r = brain().ask("what is turning radius"); self.assertIn("radius of the circle", r["answer"]); self.assertIn("does not record", r["answer"])


class Arbitration(unittest.TestCase):
    def setUp(self): self.O = ontology.load(); self.G = GL.load()

    def test_the_longer_specific_term_beats_the_term_it_contains(self):
        for q in ("what is break pad temps", "what is my brake pad temperature", "how hot are the brake pads"):
            f, note = arbitrate(q, F("define" if q.startswith("what is break") else "now", "brake_pad_life"), self.O, self.G)
            self.assertEqual(f.concept, "brake_temperature", q)

    def test_the_models_concept_stands_when_the_text_supports_it_elsewhere(self):
        f, note = arbitrate("how worn are my brake pads", F("now", "brake_pad_life"), self.O, self.G); self.assertEqual(f.concept, "brake_pad_life")
        f, note = arbitrate("how worn are the brake pads and what is the brake temperature", F("now", "brake_pad_life"), self.O, self.G); self.assertEqual(f.concept, "brake_pad_life")

    def test_rare_words_the_model_misreads_are_taken_from_the_glossary(self):
        for q, wrong, want, act, op in (("how many ECU resets were there", "diagnostic_code", "ecu_reset", "events", "count"), ("were there any sensor conflicts", "diagnostic_code", "sensor_conflict", "events", "exists"),
                                        ("how many voltage dips happened this month", "low_12v_warning", "voltage_dip", "events", "count"), ("when was the last unexpected shutdown", "any_warning", "unexpected_shutdown", "events", "last"),
                                        ("did the power derate happen", "battery_power", "power_derate", "events", "drop")):
            f, note = arbitrate(q, F(act, wrong, op), self.O, self.G); self.assertEqual(f.concept, want, q); self.assertEqual(f.act, "events"); self.assertNotIn(f.op, ("none", "drop"), q)

    def test_no_match_leaves_the_frame_alone(self):
        f, note = arbitrate("average speed last week", F("stat", "speed", "mean"), self.O, self.G); self.assertEqual((f.concept, note), ("speed", None))


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class BrakeTemperatureHonesty(unittest.TestCase):
    def test_brake_temperature_is_known_but_not_recorded(self):
        """The scripted frame is what an early model returned (pad life for 'pad temperature').  full_v11 reads this wording correctly by itself; the optional rare-word backstop (VGPT_ARBITRATE=1) must still overrule that misreading."""
        with mock.patch.dict(os.environ, {"VGPT_ARBITRATE": "1"}): r = brain().ask("what is my brake pad temperature")
        self.assertIn("isn't recorded", r["answer"]); self.assertNotIn("48", r["answer"])                      # never the pad-wear percentage under another name
        self.assertEqual(r["facts"]["frame"]["concept"], "brake_temperature")


if __name__ == "__main__":
    unittest.main()
