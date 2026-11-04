"""change_ops._split with the permutation tests done as one batch must give exactly what shuffling one at a time gave (the old loop is kept here as the reference)."""
import math, os, sys, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import change_ops as C


def old_split(ys, B=400):
    n = len(ys)
    if n < 6: return None
    ys = np.asarray(ys, float); ks = list(range(3, n - 2)); stats = C._tstats_all(ys, ks); k = ks[int(np.argmax(stats))]; obs = float(np.max(stats))
    rng = np.random.default_rng(7); ge = 0
    for _ in range(B):
        p = rng.permutation(ys); ge += float(np.max(C._tstats_all(p, ks))) >= obs - 1e-12
    a, b = ys[:k], ys[k:]; return k, (1 + ge) / (1 + B), float(np.mean(a)), float(np.mean(b))


class Split(unittest.TestCase):
    def test_same_answer(self):
        rng = np.random.default_rng(1)
        for n, shift in ((8, 0.0), (30, 3.0), (200, 0.0), (500, 1.0), (1461, 0.4), (1461, 0.0)):
            y = rng.normal(150, 12, n); y[n // 2:] += shift * 12
            a = C._split(list(y)); k, p, ma, mb = old_split(list(y))
            self.assertEqual((a["k"], a["p_adj"], a["mean_before"], a["mean_after"]), (k, p, ma, mb), (n, shift))
        self.assertIsNone(C._split([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
