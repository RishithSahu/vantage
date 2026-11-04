"""sensor_quality.stuck and spikes, vectorised, must flag exactly what the loops flagged (the loops are kept here as the reference)."""
import os, sys, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import sensor_quality as SQ


def old_best(v, moving, min_run=20):
    same = np.concatenate([[False], v[1:] == v[:-1]]); run = best = 0
    for i, s in enumerate(same):
        if s and (moving is None or moving[i]) and not (moving is None and v[i] == 0):
            run += 1
            if run > best: best = run
        else: run = 0
    return best


def old_spikes(values, k=8.0):
    v = np.asarray(values, dtype=float)
    if len(v) < 20: return 0
    ok = ~np.isnan(v)
    if ok.sum() < 20: return 0
    d = np.abs(np.diff(v[ok])); med = np.nanmedian(d) or 1e-9; idx = np.where(d > k * max(med, 1e-6) * 10)[0]; sd3 = 6 * np.nanstd(v[ok]) / 3
    return len([i for i in idx if i + 2 < len(d) and d[i + 1] > 0.8 * d[i] and d[i] > sd3])


class Same(unittest.TestCase):
    def test_stuck_run(self):
        rng = np.random.default_rng(5)
        for n in (0, 1, 30, 3000):
            for flat in (0.0, 0.5, 0.97):
                v = np.round(rng.normal(20, 5, n), 0); keep = rng.random(n) < flat; v[keep] = 20.0
                v[rng.random(n) < 0.02] = np.nan; v[:5] = 0
                for moving in (None, rng.random(n) < 0.7):
                    ref = old_best(v, moving); got = SQ._longest_true(np.concatenate([[False], v[1:] == v[:-1]]) & (np.asarray(moving, dtype=bool) if moving is not None else ~(v == 0)) if n else np.array([], bool))
                    self.assertEqual(got, ref, (n, flat))
                    flags = SQ.stuck(v, moving, min_run=20, signal="speed"); want = (n >= 20) and (ref >= 20) and bool(np.nanstd(v) > 0); self.assertEqual(bool(flags), want, (n, flat))

    def test_spikes(self):
        rng = np.random.default_rng(6)
        for n in (10, 100, 5000):
            v = rng.normal(40, 1, n); v[rng.random(n) < 0.03] += rng.choice([-30, 30], 1)[0]; v[rng.random(n) < 0.01] = np.nan
            ref = old_spikes(v); flags = SQ.spikes(v, "battery_temperature")
            want = (n >= 20) and ((~np.isnan(v)).sum() >= 20) and ref >= 1 and ref / max(n, 1) < 0.02; self.assertEqual(bool(flags), want, n)
            if flags: self.assertIn(f"{ref} isolated", str(flags[0]))


if __name__ == "__main__":
    unittest.main()
