"""Two rewrites that must not change a number, checked on the real 4-year database when it is present:
  * per-day accumulation by slices of one prepared frame  == the old loop that rebuilt everything for every day over the whole window;
  * tyre deficits for all trips from the columnar copy of the tyre minutes == one query per trip."""
import os, shutil, sqlite3, sys, tempfile, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "data", "aster_ev_4yr.db")


@unittest.skipUnless(os.path.exists(DB), "needs data/aster_ev_4yr.db")
class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vgpt import colstore
        from vgpt.brain import Brain
        d = tempfile.mkdtemp(); cls.db = os.path.join(d, "w.db"); shutil.copy(DB, cls.db)
        cls.b = Brain(cls.db, slm=None, slm_ir=None, policy="clarify", qc_mode="all"); cls.b.now = int(sqlite3.connect(cls.db).execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) + 60
        cls.ex = cls.b.qc().executor; cs = colstore.of(cls.ex.c); cs.build("sig_1m", []); cs.build("ext_1m", [])

    def test_accumulation_per_day(self):
        from vgpt import derived as D
        ex = self.ex; hi = self.b.now
        for target in ("energy_used", "distance_driven", "regen_energy", "wh_per_km"):
            for span, tier in ((60 * 86400, "1m"), (400 * 86400, "1m")):
                fr = ex.load_frame({"speed", "battery_power", "drive_mode"} | ({"regen_power"} if ex.p.concept_available("regen_power") else set()), hi - span, hi, tier)
                if len(fr) == 0: continue
                mask = np.ones(len(fr), bool); mask[::7] = False; days = (fr.ts + ex.ctx.tz) // 86400; pre = D.prepare(fr, ex.p)
                lo_i = np.searchsorted(days, np.unique(days[mask]), side="left"); hi_i = np.searchsorted(days, np.unique(days[mask]), side="right")
                for d, i0, i1 in list(zip(np.unique(days[mask]), lo_i, hi_i))[::9]:
                    m = mask & (days == d); old = D.accumulate(target, fr, ex.p, m); new = D.accumulate(target, fr, ex.p, mask, pre=pre, part=slice(int(i0), int(i1)))
                    self.assertEqual(old is None, new is None, (target, d))
                    if old is not None: self.assertEqual(old[0], new[0], (target, int(d))); self.assertEqual(old[5], new[5]); self.assertEqual(old[4], new[4])

    def test_tyre_deficits_from_the_copy(self):
        from vgpt import qa_core as Q
        T = Q.trips_between(self.ex.c, 0, 2 ** 40); a = [dict(t) for t in T]; b = [dict(t) for t in T]
        for t in a: t["tyre"] = Q.tyre_deficit(self.ex.c, t["start"])
        Q.attach_tyre(self.ex.c, b); self.assertEqual([t["tyre"] for t in a], [t["tyre"] for t in b])


if __name__ == "__main__": unittest.main()
