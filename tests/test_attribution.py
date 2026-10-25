"""Attribution against counterfactual twins (bench/counterfactual.py builds them; the generator's roof box is an unrecorded cause). A regression guard on the explanation engine, not a proof of correctness:
the unexplained part must keep most of the injected cause, and the recorded factors must not absorb most of it."""
import glob, os, statistics, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from bench.eval_attribution import score


@unittest.skipUnless(len(glob.glob(os.path.join(ROOT, "data", "cf_*_with.db"))) >= 4, "counterfactual twins not generated (python bench/counterfactual.py --preset roadwarrior --seeds 11,12,13,14)")
class Attribution(unittest.TestCase):
    def test_an_unrecorded_cause_stays_in_the_unexplained_part(self):
        rows = [r for r in score() if len(r) > 3 and r[5] >= 5.0]; self.assertGreaterEqual(len(rows), 4)
        cover = statistics.median(r[7] / r[5] for r in rows); leak = statistics.median(r[8] / r[5] for r in rows)
        self.assertGreater(cover, 0.5, f"median share of the injected cause left unexplained: {cover:.2f}"); self.assertLess(leak, 0.5, f"median share absorbed by recorded factors: {leak:.2f}")


if __name__ == "__main__": unittest.main()
