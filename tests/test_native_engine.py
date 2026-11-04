"""The native inference engine (engine/vg_engine.c, vgpt/meaning/native.py): it builds, several cars in one call give exactly what each gives alone, and it agrees with the ONNX build on the next-token choice.
Skipped when the weight file (models/vg_frame2r12.vgw, tools/export_vgw.py) or a C compiler is missing."""
import os, shutil, sys, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
VGW = os.path.join(ROOT, "models", "vg_frame2r12.vgw"); ORT = os.path.join(ROOT, "models", "ort_frame2r12")


@unittest.skipUnless(os.path.exists(VGW) and shutil.which("gcc"), "needs models/vg_frame2r12.vgw and gcc")
class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt.meaning import native
        cls.eng = native.Engine(VGW, threads=2); cls.rng = np.random.default_rng(7); cls.cand = np.sort(cls.rng.choice(150000, 500, replace=False)).astype(np.int64)
        cls.empty = [np.zeros((1, 2, 0, 64), np.float32) for _ in range(48)]

    def test_cars_in_one_call_get_exactly_what_each_gets_alone(self):
        lens = [5, 17, 1, 9]; seqs = [(self.rng.integers(1000, 100000, n), 0, self.empty, 1) for n in lens]
        alone = [self.eng.forward1(*s[:3], s[3], self.cand) for s in seqs]; batch = self.eng.forward(seqs, self.cand)
        for (la, ka), (lb, kb) in zip(alone, batch):
            self.assertTrue(np.array_equal(la, lb)); self.assertTrue(all(np.array_equal(p, q) for p, q in zip(ka, kb)))

    def test_a_decode_pass_extends_the_memory_it_was_given(self):
        ids = self.rng.integers(1000, 100000, 12); lg, kv = self.eng.forward1(ids, 0, self.empty, 1, self.cand); self.assertEqual(kv[0].shape, (1, 2, 12, 64))
        lg2, kv2 = self.eng.forward1(self.rng.integers(1000, 100000, 3), 12, kv, 3, self.cand); self.assertEqual(lg2.shape, (3, len(self.cand))); self.assertEqual(kv2[0].shape, (1, 2, 15, 64))
        self.assertTrue(np.array_equal(kv2[0][:, :, :12], kv[0]))                       # the old rows are untouched

    @unittest.skipUnless(os.path.exists(os.path.join(ORT, "model.onnx")), "needs the ONNX build")
    def test_the_next_token_choice_agrees_with_onnx_runtime(self):
        os.environ.setdefault("VGPT_FRAME", "2"); import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(ORT, "model.onnx"), providers=["CPUExecutionProvider"]); ids = self.rng.integers(1000, 100000, 20)
        feed = {"ids": ids[None].astype(np.int64), "pos": np.arange(20, dtype=np.int64)[None], "nkeep": np.array([1], np.int64), "cand": self.cand}
        for l in range(24): feed[f"pk{l}"] = self.empty[0]; feed[f"pv{l}"] = self.empty[0]
        ref = s.run(None, feed)[-1][0]; lg, _ = self.eng.forward1(ids, 0, self.empty, 1, self.cand)
        self.assertEqual(int(ref[0].argmax()), int(lg[0].argmax())); self.assertLess(float(np.abs(ref - lg).max()), 2.0)      # logits are within int8 rounding noise


if __name__ == "__main__": unittest.main()
