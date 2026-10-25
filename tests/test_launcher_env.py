"""The launcher adopts the product's language model (`models/product.json`) when it RUNS.  Importing it - as tests and tools do - must change nothing in the environment of the importing process; otherwise every test that comes after
such an import silently runs with the new model's settings and measures something else (this happened: seven answer tests failed in a long run only because an earlier test imported the launcher)."""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
PRODUCT = os.path.join(ROOT, "models", "product.json")


def load():
    spec = importlib.util.spec_from_file_location("vehiclegpt_env", os.path.join(ROOT, "vehiclegpt.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class ImportingTheLauncher(unittest.TestCase):
    def test_changes_no_environment_variable(self):
        before = dict(os.environ); load(); self.assertEqual(dict(os.environ), before)

    def test_in_a_fresh_process_too(self):
        code = "import os, sys; sys.path.insert(0, %r); before = dict(os.environ); import vehiclegpt; print('SAME' if dict(os.environ) == before else 'CHANGED ' + str(sorted(set(os.environ) ^ set(before))))" % ROOT
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT); self.assertIn("SAME", p.stdout, p.stdout + p.stderr)


@unittest.skipUnless(os.path.exists(PRODUCT), "no models/product.json: nothing to adopt")
class RunningTheLauncher(unittest.TestCase):
    def setUp(self):
        self.cfg = json.load(open(PRODUCT, encoding="utf-8")); self.adapter = os.path.join(ROOT, "models", self.cfg["adapter"])

    def test_adoption_sets_the_frame_and_the_adapter_the_file_names_and_only_then(self):
        m = load(); orig = (os.environ.get("VGPT_FRAME"), os.environ.get("VGPT_MEANING_ADAPTER"))
        with mock.patch.dict(os.environ):
            os.environ.pop("VGPT_FRAME", None); os.environ.pop("VGPT_MEANING_ADAPTER", None); m._adopt_product_model()
            self.assertEqual(os.environ["VGPT_FRAME"], str(int(self.cfg.get("frame", 1)))); self.assertEqual(os.environ["VGPT_MEANING_ADAPTER"], self.adapter)
        self.assertEqual((os.environ.get("VGPT_FRAME"), os.environ.get("VGPT_MEANING_ADAPTER")), orig)                                              # and the patch restored the process

    def test_an_explicit_setting_of_the_person_wins(self):
        m = load()
        with mock.patch.dict(os.environ, {"VGPT_FRAME": "1"}): os.environ.pop("VGPT_MEANING_ADAPTER", None); m._adopt_product_model(); self.assertEqual(os.environ["VGPT_FRAME"], "1"); self.assertNotIn("VGPT_MEANING_ADAPTER", os.environ)

    def test_main_adopts_before_anything_is_started(self):
        m = load()
        with mock.patch.dict(os.environ):
            os.environ.pop("VGPT_FRAME", None); os.environ.pop("VGPT_MEANING_ADAPTER", None)
            with mock.patch("builtins.print"): self.assertEqual(m.main(["version"]), 0)
            self.assertEqual(os.environ.get("VGPT_FRAME"), str(int(self.cfg.get("frame", 1))))


if __name__ == "__main__":
    unittest.main()
