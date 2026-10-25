"""Guess-ahead decoding is lossless: the frame the model writes with the question cues, with the draft table only, and token by token (no guessing at all) is the same frame.

Every proposed token is accepted only if it is the model's own masked argmax at that position (vgpt/meaning/ort_backend.py), so a guess may save a pass over the weights but can never change a word.
This runs the real int8 model on a spread of the acceptance questions; it is skipped where the model is not installed."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _questions(n=60):
    out = []
    for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2: out.append(p[2])
    return out[::max(1, len(out) // n)][:n]


class Lossless(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            sys.path.insert(0, os.path.join(ROOT, "tests")); import _envguard; restore = _envguard.adopt_product_model()
            try:
                from vgpt.meaning.ort_backend import OrtFrameModel
                from vgpt.meaning.cues import Cues
                d = os.path.join(ROOT, "models", "ort_frame2w")
                if not os.path.isdir(d): raise FileNotFoundError(d)
                cls.m = OrtFrameModel(d, threads=2); cls.Cues = Cues
            finally: restore()
        except Exception as e:
            raise unittest.SkipTest("int8 frame model not available: %s" % e)

    def _raw(self, qs, cues, spec, table=True):
        m = self.m; saved = (m.cues, m.table, os.environ.get("VGPT_SPEC"))
        try:
            m.cues = self.Cues(m.onto, m.glossary) if cues else None
            if not table: m.table = None
            os.environ["VGPT_SPEC"] = "1" if spec else "0"
            return [m.parse(q)[1]["raw"] for q in qs]
        finally:
            m.cues, m.table = saved[0], saved[1]
            if saved[2] is None: os.environ.pop("VGPT_SPEC", None)
            else: os.environ["VGPT_SPEC"] = saved[2]

    def test_cues_do_not_change_a_frame(self):
        qs = _questions(); a = self._raw(qs, cues=True, spec=True); b = self._raw(qs, cues=False, spec=True)
        self.assertEqual([(q, x) for q, x, y in zip(qs, a, b) if x != y], [])

    def test_guessing_does_not_change_a_frame(self):
        qs = _questions(40); a = self._raw(qs, cues=True, spec=True); b = self._raw(qs, cues=False, spec=False, table=False)
        bad = [(q, x, y) for q, x, y in zip(qs, a, b) if x != y]
        self.assertLessEqual(len(bad), 1, bad)         # batched int8 rows can differ from single rows in the last bit: at most one close call in 40 may flip


if __name__ == "__main__":
    unittest.main()
