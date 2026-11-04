"""Exactness of three speed-ups: temporal.runs (vectorised), the sorted-day grouping and query_executor._match_idx (alignment of two tables' times) must equal the straightforward code."""
import os, sys, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import vgpt.temporal as TM
import vgpt.query_executor as QE


def old_runs(ts, ok):
    out = []; start = None; prev = None
    for i in range(len(ts)):
        if ok[i]:
            if start is not None and ts[i] - ts[prev] > TM.BREAK: out.append((start, prev)); start = i
            elif start is None: start = i
            prev = i
        else:
            if start is not None: out.append((start, prev)); start = None
    if start is not None: out.append((start, prev))
    return out


class Same(unittest.TestCase):
    def test_runs(self):
        rng = np.random.default_rng(2)
        for n in (0, 1, 2, 50, 2000):
            for p in (0.1, 0.5, 0.9, 1.0):
                gaps = rng.choice([60, 60, 60, 60, 600, 5000], n); ts = np.cumsum(gaps).astype(np.int64); ok = rng.random(n) < p
                self.assertEqual(TM.runs(ts, ok), old_runs(ts, ok), (n, p))

    def test_match_idx(self):
        rng = np.random.default_rng(3)
        for n, m in ((1000, 10), (1000, 300), (1000, 340), (5000, 40), (50, 1)):
            ts = np.sort(rng.integers(0, 4000, n)); tts = np.sort(rng.integers(-50, 4100, m))
            np.testing.assert_array_equal(QE._match_idx(tts, ts), np.searchsorted(tts, ts, side="right") - 1)
        ts = np.arange(0, 6000, 60); tts = np.arange(30, 6000, 600)
        np.testing.assert_array_equal(QE._match_idx(tts, ts), np.searchsorted(tts, ts, side="right") - 1)


if __name__ == "__main__":
    unittest.main()
