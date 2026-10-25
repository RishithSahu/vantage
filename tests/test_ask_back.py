"""Asking back (docs/ASK_BACK.md): a reading below the confidence threshold with a second reading of the same slot becomes "did you mean 1 or 2"; the answer "1" / "2" runs that frame for the original question."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.meaning import glossary as GL
from vgpt.meaning.frame import Frame
from vgpt.meaning.pipeline import MeaningBrain

DB = os.path.join(ROOT, "data", "gen_courier.db")
NOW = 1790352000
F = Frame


class Scripted:
    glossary = GL.load()

    def __init__(self, table): self.table = {k.lower(): v for k, v in table.items()}; self.calls = []

    def parse(self, text, hint=None, prev=None):
        self.calls.append(text); f, p, alt = self.table.get(text.lower().strip(), (F("ambiguous", "none"), 0.99, None))
        info = {"raw": f.to_text(), "pmin": p, "ms": 0.0, "hint": ""}
        if alt: info["alt"] = alt
        return f, info


TABLE = {
    "batt condition": (F("health", "battery_soh"), 0.62, {"slot": "concept", "index": 1, "chosen": "battery_soh", "other": "battery_condition", "p_other": 0.38}),
    "what is my speed": (F("now", "speed"), 0.99, None),
    "murky speed question": (F("now", "speed"), 0.55, None),
    "when should i service": (F("advice", "maintenance"), 0.7, {"slot": "act", "index": 0, "chosen": "advice", "other": "estimate", "p_other": 0.24}),
}


@unittest.skipUnless(os.path.exists(DB), "generated history not present")
class AskBack(unittest.TestCase):
    def brain(self, tau=0.8): return MeaningBrain(DB, Scripted(TABLE), now=NOW, tau=tau)

    def test_two_readings_are_offered_and_one_is_chosen(self):
        b = self.brain(); r = b.ask("batt condition")
        self.assertEqual(r["facts"]["route"], "clarify:choose"); self.assertIn("1.", r["answer"]); self.assertIn("2.", r["answer"]); self.assertEqual(r["facts"]["other"]["concept"], "battery_condition")
        r2 = b.ask("2"); self.assertEqual(r2["facts"]["frame"]["concept"], "battery_condition"); self.assertEqual(r2["facts"]["chosen"], 2); self.assertNotEqual(r2["facts"].get("route"), "clarify:choose")
        b = self.brain(); b.ask("batt condition"); r1 = b.ask("first one"); self.assertEqual(r1["facts"]["frame"]["concept"], "battery_soh"); self.assertEqual(r1["facts"]["chosen"], 1)

    def test_the_act_can_be_the_slot_that_differs(self):
        b = self.brain(); r = b.ask("when should i service"); self.assertEqual(r["facts"]["route"], "clarify:choose"); self.assertEqual(r["facts"]["other"]["act"], "estimate")

    def test_anything_else_after_the_question_is_a_new_question(self):
        b = self.brain(); b.ask("batt condition"); r = b.ask("what is my speed")
        self.assertEqual(r["facts"]["frame"]["concept"], "speed"); self.assertNotIn("chosen", r["facts"])
        r = b.ask("2"); self.assertNotEqual(r["facts"].get("route"), "clarify:choose")                         # nothing is pending any more: "2" is just a word

    def test_a_confident_reading_is_answered_at_once_and_an_unsure_one_without_a_second_reading_keeps_the_old_message(self):
        b = self.brain(); self.assertNotEqual(b.ask("what is my speed")["facts"].get("route"), "clarify:choose")
        r = b.ask("murky speed question"); self.assertEqual(r["facts"]["route"], "clarify:low_confidence")

    def test_a_threshold_of_zero_turns_it_off(self):
        r = self.brain(tau=0.0).ask("batt condition"); self.assertNotEqual(r["facts"].get("route"), "clarify:choose")


if __name__ == "__main__":
    unittest.main()
