"""The previous-turn gate: the previous frame is shown to the frame model only for a question that points back or whose bare reading is incomplete."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.meaning import prevgate as PG
from vgpt.meaning.frame import Frame


class Wants(unittest.TestCase):
    def test_follow_ups_get_the_previous_turn(self):
        for q in ("and yesterday", "what about the motor", "which one was the longest", "the latest one", "now the coolant flow", "same thing for regen energy", "do the ambient temperature too", "why did it happen", "how did you get that",
                  "and the week before that", "speed"):
            self.assertTrue(PG.wants_prev(q), q)

    def test_complete_questions_do_not(self):
        for q in ("how fast am I going", "what was the average battery temperature last week", "how many hard brakes did I have in the last month", "why is the range dropping", "is the tyre pressure slowly going down"):
            self.assertFalse(PG.wants_prev(q), q)


class Incomplete(unittest.TestCase):
    def test_an_ambiguous_or_empty_reading_asks_for_the_previous_turn(self):
        self.assertTrue(PG.frame_needs_prev(Frame("ambiguous", "none"), 0.99))
        self.assertTrue(PG.frame_needs_prev(Frame("now", "unlisted"), 0.99))
        self.assertTrue(PG.frame_needs_prev(Frame("now", "speed"), 0.3))

    def test_a_complete_confident_reading_does_not(self):
        self.assertFalse(PG.frame_needs_prev(Frame("now", "speed"), 0.97))
        self.assertFalse(PG.frame_needs_prev(Frame("summary", "none"), 0.9))


class OnTheModel(unittest.TestCase):
    def test_the_line_is_added_exactly_when_the_gate_says(self):
        try:
            sys.path.insert(0, os.path.join(ROOT, "tests")); import _envguard; restore = _envguard.adopt_product_model()
            try:
                from vgpt.meaning.ort_backend import OrtFrameModel
                d = os.path.join(ROOT, "models", "ort_frame2w")
                if not os.path.isdir(d): raise FileNotFoundError(d)
                m = OrtFrameModel(d, threads=2)
            finally: restore()
        except Exception as e:
            self.skipTest("int8 frame model not available: %s" % e)
        prev = "stat battery_temperature mean last_week none none none none"
        _, a = m.parse("how fast am I going", prev=prev); self.assertFalse(a["prev_used"])
        _, b = m.parse("and yesterday", prev=prev); self.assertTrue(b["prev_used"])
        os.environ["VGPT_PREV"] = "always"
        try: _, c = m.parse("how fast am I going", prev=prev)
        finally: os.environ.pop("VGPT_PREV", None)
        self.assertNotIn("prev_used", c)


if __name__ == "__main__":
    unittest.main()
