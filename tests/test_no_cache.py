"""The owner's rule: no question / answer cache of any kind.  A repeated question is read by the model again and answered from the data as it is now.  The only de-duplication allowed is of the SAME request while it is running
(tests/test_worker_queue.py)."""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WORLD = os.path.join(ROOT, "data", "world_aster.db")


class Source(unittest.TestCase):
    def files(self):
        for base in ("vgpt", "ui"):
            for dp, _, fs in os.walk(os.path.join(ROOT, base)):
                if "__pycache__" in dp or os.sep + "modern" in dp: continue
                for f in fs:
                    if f.endswith(".py"): yield os.path.join(dp, f)

    def test_no_switch_to_turn_a_question_cache_on_exists(self):
        for p in self.files():
            s = open(p, encoding="utf-8").read()
            self.assertNotIn("VGPT_FRAME_CACHE", s, p); self.assertNotIn("ask_cache", s, p)

    def test_the_brain_keeps_no_table_of_frames_by_question(self):
        s = open(os.path.join(ROOT, "vgpt", "meaning", "pipeline.py"), encoding="utf-8").read(); self.assertNotIn("_frames", s); self.assertNotIn("cached=True", s)


@unittest.skipUnless(os.path.exists(WORLD), "needs data/world_aster.db")
class Behaviour(unittest.TestCase):
    def test_a_repeated_question_is_read_by_the_model_every_time(self):
        from vgpt.meaning import glossary as GL
        from vgpt.meaning.frame import Frame
        from vgpt.meaning.pipeline import MeaningBrain
        from vgpt.vehicle import lake as LK

        class Counting:
            glossary = GL.load(); calls = 0
            def parse(self, text, hint=None, prev=None):
                Counting.calls += 1; f = Frame("now", "speed"); return f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
        LK.adopt_path(WORLD)
        try:
            b = MeaningBrain(WORLD, Counting(), now=1792360800)
            for _ in range(3): b.dialogue.reset(); b.ask("what is my speed right now")
            self.assertEqual(Counting.calls, 3)
        finally: LK.release()


if __name__ == "__main__": unittest.main()
