"""The release gate for the model: the model that models/product.json selects is the model that is exercised.  Both backends (llama.cpp with the adapter, ONNX Runtime int8) read the same questions; they must agree on the frame
for nearly all of them, and each must report the same kind of confidence.  If the selected files are absent this FAILS under VGPT_REQUIRE_TESTDATA=1 (a release is not built on a model nobody ran) and skips otherwise."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests import testdata  # noqa: E402

PRODUCT = os.path.join(ROOT, "models", "product.json")
QS = ["what is my speed right now", "how many trips last week", "what was my highest speed yesterday", "is my tyre pressure low", "how much battery is left", "what is the average battery temperature this week",
      "when did I last charge", "any warnings today", "why is my range dropping", "how long to charge to 80%", "compare my energy use this week with last week", "what was my fuel level at 6 pm yesterday"]


def selected():
    try: j = json.load(open(PRODUCT, encoding="utf-8"))
    except (OSError, ValueError): return None
    return j


class Selected(unittest.TestCase):
    def test_the_manifest_names_files_that_exist_and_both_backends_read_the_same_frames(self):
        j = selected()
        if j is None: testdata.need_files("models/product.json", group="models")
        need = ["models/" + j["adapter"], "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"] + (["models/%s/model.onnx" % j["ort"]] if j.get("ort") else [])
        testdata.need_files(*need, group="models")
        import vehiclegpt; vehiclegpt._adopt_product_model()
        from vgpt.meaning.interface import FrameModel
        from vgpt.meaning.ort_backend import OrtFrameModel
        from vgpt.vehicle import lake as LK
        world = os.path.join(ROOT, "data", "world_aster.db")
        if os.path.exists(world): LK.adopt_path(world)
        llama = FrameModel(os.environ["VGPT_MEANING_ADAPTER"], threads=4, version=int(j.get("frame", 2))); self.assertEqual(llama.decoder, "mask")
        if j.get("ort"):
            ort = OrtFrameModel(os.path.join(ROOT, "models", j["ort"]), threads=4, version=int(j.get("frame", 2)))
            same = 0
            for q in QS:
                a, ia = llama.parse(q); b, ib = ort.parse(q); self.assertIsNotNone(ia["pmin"]); self.assertIsNotNone(ib["pmin"]); self.assertNotIn("decode_error", ia); self.assertNotIn("decode_error", ib)
                same += int(a is not None and b is not None and a.to_text() == b.to_text())
                self.assertLess(abs(ia["pmin"] - ib["pmin"]), 0.35, q)                                                    # the same kind of number (the int8 build moves close calls, not the scale)
            self.assertGreaterEqual(same, len(QS) - 1, "the two backends read the same frames")
        LK.release()


if __name__ == "__main__": unittest.main()
