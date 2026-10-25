"""docs/BOUNDARY.md says what is learned and what is code: it must list every layer and every module that holds text patterns, and the switches it names must really switch the stages off."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DOC = open(os.path.join(ROOT, "docs", "BOUNDARY.md"), encoding="utf-8").read()
WORLD = os.path.join(ROOT, "data", "world_aster.db")


class Document(unittest.TestCase):
    def test_every_layer_and_every_module_with_text_patterns_is_listed(self):
        from vgpt.meaning import layers
        for name in layers.ALL: self.assertIn(name, DOC, f"layer {name} is not in docs/BOUNDARY.md")
        src = open(os.path.join(ROOT, "tests", "test_meaning.py"), encoding="utf-8").read()
        import re
        budget = re.search(r"BUDGET = (\{[^}]*\})", src).group(1)
        for mod in re.findall(r'"([a-z_]+\.py)"', budget):
            if mod in ("glossary.py", "pipeline.py", "plain.py", "calx.py", "tripx.py", "evmap.py", "thresholdx.py"): continue                       # helpers named in the stage rows by their executor family
            self.assertIn(mod, DOC, f"{mod} has a pattern budget but is not described in docs/BOUNDARY.md")
        for e in ("calx", "tripx", "evmap", "thresholdx", "plain"): self.assertIn(e, DOC)

    def test_nothing_in_the_product_calls_itself_model_only_any_more(self):
        for rel in ("vgpt/product.py", "vgpt/meaning/pipeline.py"):
            s = open(os.path.join(ROOT, rel), encoding="utf-8").read().lower(); self.assertNotIn("no rule decides", s, rel); self.assertIn("boundary.md", s, rel)


class Switches(unittest.TestCase):
    def test_the_preparser_switch_follows_the_documented_variables(self):
        from vgpt.meaning import layers as LY
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("VGPT_PREPARSER", "VGPT_MODEL_ONLY", "VGPT_LAYERS_OFF"): os.environ.pop(k, None)
            self.assertTrue(LY.preparser_on())
            for k, v in (("VGPT_PREPARSER", "0"), ("VGPT_MODEL_ONLY", "1"), ("VGPT_LAYERS_OFF", "all")):
                with mock.patch.dict(os.environ, {k: v}): self.assertFalse(LY.preparser_on(), k)

    @unittest.skipUnless(os.path.exists(WORLD), "needs data/world_aster.db")
    def test_with_the_preparser_off_a_question_it_would_answer_goes_to_the_model(self):
        from vgpt.meaning import glossary as GL
        from vgpt.meaning.frame import Frame
        from vgpt.meaning.pipeline import MeaningBrain
        from vgpt.vehicle import lake as LK

        class Counting:
            glossary = GL.load(); calls = 0
            def parse(self, text, hint=None, prev=None):
                Counting.calls += 1; f = Frame("social", "none"); return f, {"raw": f.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
        LK.adopt_path(WORLD)
        try:
            b = MeaningBrain(WORLD, Counting(), now=1792360800); b.ask("who are you"); on = Counting.calls
            with mock.patch.dict(os.environ, {"VGPT_PREPARSER": "0"}): b.dialogue.reset(); b.ask("who are you")
            self.assertEqual(on, 0, "the pre-parser answers who the assistant is without the model"); self.assertEqual(Counting.calls, 1, "switched off, the model reads it")
        finally: LK.release()


if __name__ == "__main__": unittest.main()
