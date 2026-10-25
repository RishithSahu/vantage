"""Instant load, for testing: the car tells Vantage where a finished history of it is, Vantage copies it into the vehicle's database at once and the car carries on from the end of it.  And what is refused."""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_pipe import Bench, model_dirs
from vgpt.car import writer as CW
from vgpt.car.car import Car


class Instant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="instant_"); cls.db = os.path.join(cls.dir, "kestrel.db")
        r = subprocess.run([sys.executable, "-m", "vgpt.world.build", "--profile", "kestrel_petrol", "--days", "40", "--seed", "4", "--end", "2026-05-01", "--out", cls.db], capture_output=True, text=True, cwd=ROOT)
        if r.returncode: raise unittest.SkipTest("could not build a world: " + r.stderr[-200:])
        c = sqlite3.connect(cls.db); cls.lo, cls.hi = c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1m").fetchone(); cls.n = c.execute("SELECT COUNT(*) FROM sig_1m").fetchone()[0]; c.close()

    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.dir, True)

    def run_car(self, model, dataset, instant=True, accept=None, seconds=20.0, path=None):
        b = Bench(self)
        if accept is not None: b.svc.accept_history = accept
        meta = model_dirs()[model]; r, w = os.pipe(); b.svc.attach(r); c = Car(meta["dir"], CW.Writer(w), dataset=dataset, speed=60.0, seed=7, instant=instant); c.start()
        t0 = c.drive.t; end = time.monotonic() + 120
        while time.monotonic() < end and c.is_alive():
            st = b.svc.status(); imp = st.get("import")
            if (imp and imp["phase"] in ("done", "refused", "failed") and c.drive.t - t0 >= seconds) or (imp is None and c.drive.t - t0 >= 5 * seconds and st.get("vehicle")): break
            time.sleep(0.2)
        c.stop(); b.svc.join(20); b.svc.drain(30); return b, c

    def test_the_whole_history_is_there_at_once_and_the_car_carries_on_from_its_end(self):
        b, c = self.run_car("kestrel_petrol", self.db)
        st = b.svc.status(); imp = st["import"]; self.assertTrue(imp and imp["ok"], imp); self.assertEqual(imp["phase"], "done")
        d = b.db()
        try:
            n, lo, hi = d.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM sig_1m").fetchone()
            self.assertGreaterEqual(n, self.n); self.assertEqual(lo, self.lo); self.assertGreaterEqual(hi, self.hi)             # all of it, in the history's own calendar, plus what the car sent after
            self.assertEqual(d.execute("SELECT COUNT(*) FROM agg_1d").fetchone()[0] > 0, sqlite3.connect(self.db).execute("SELECT COUNT(*) FROM agg_1d").fetchone()[0] > 0)
        finally: d.close()
        self.assertTrue(any("instant load" in e["text"] for e in st["events"]))
        self.assertTrue(c.status()["instant"])

    def test_it_is_a_copy_not_a_replay(self):
        t = time.time(); b, c = self.run_car("kestrel_petrol", self.db, seconds=5.0); self.assertTrue(b.svc.status()["import"]["ok"])
        self.assertLess(b.svc.status()["import"]["seconds"], 30)

    def test_a_switch_that_is_off_a_wrong_car_and_a_bad_path_are_refused_and_the_car_still_works(self):
        b, c = self.run_car("kestrel_petrol", self.db, accept=False, seconds=5.0); imp = b.svc.status()["import"]; self.assertEqual(imp["phase"], "refused"); self.assertIn("does not accept", imp["reason"])
        self.assertIsNotNone(b.svc.status().get("vehicle"))
        other = "aster_ev" if "aster_ev" in model_dirs() else "meridian_ev"
        b, c = self.run_car(other, self.db, seconds=5.0); imp = b.svc.status()["import"]; self.assertEqual(imp["phase"], "refused"); self.assertIn("does not bind", imp["reason"])
        b = Bench(self); r, w = os.pipe(); b.svc.attach(r); wr = CW.Writer(w); wr.control(proto=1, kind="vehicle data link", clock="calendar"); wr.control(history={"path": "relative/path.db"})
        c2 = Car(model_dirs()["kestrel_petrol"]["dir"], wr, speed=60.0); c2.start(); time.sleep(3); c2.stop(); b.svc.join(20)
        imp = b.svc.status().get("import"); self.assertTrue(imp is None or imp["phase"] == "refused")

    def test_a_vehicle_that_already_has_a_history_is_not_overwritten(self):
        b, c = self.run_car("kestrel_petrol", self.db, seconds=5.0); self.assertTrue(b.svc.status()["import"]["ok"])
        r, w = os.pipe(); b.svc.attach(r); c2 = Car(model_dirs()["kestrel_petrol"]["dir"], CW.Writer(w), dataset=self.db, speed=60.0, seed=7, instant=True); c2.start(); time.sleep(8); c2.stop(); b.svc.join(20)
        imp = b.svc.status()["import"]; self.assertEqual(imp["phase"], "refused"); self.assertIn("already has", imp["reason"])
        d = b.db()
        try: self.assertGreaterEqual(d.execute("SELECT COUNT(*) FROM sig_1m").fetchone()[0], self.n)                              # and the first history is still whole
        finally: d.close()


if __name__ == "__main__":
    unittest.main()
