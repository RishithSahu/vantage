"""The application reading a car through the local pipe: it waits, adopts the vehicle the receiver recognises, follows the car's clock, switches when another car connects, and keeps what is stored when the car goes."""
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

from stub_planner import StubPlanner  # noqa: E402
from vgpt.car import models, transport as CT, writer as CW  # noqa: E402
from vgpt.car.car import Car  # noqa: E402
from vgpt.pipe import transport as PT  # noqa: E402
import secrets  # noqa: E402


def wait(cond, seconds=60, step=0.2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(step)
    return False


class Linked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp(prefix="vgpt_linked_home_")
        from ui import server as S
        cls.S = S; cls.home = tempfile.mkdtemp(prefix="vgpt_linked_"); cls.endpoint = PT.endpoint("t" + secrets.token_hex(4))
        a = SimpleNamespace(db="data/vehicle.db", model="none", camera="clip", webcam_index=0, live=None, linked=True, link_home=cls.home, link_endpoint=cls.endpoint)
        cls.app = S.App(a, planner_factory=lambda conn: StubPlanner(conn)); cls.d = staticmethod(S.make_dispatcher(cls.app)); cls.cars = []

    @classmethod
    def tearDownClass(cls):
        for c in cls.cars: c.stop()
        cls.app._close_linked(); shutil.rmtree(cls.home, True)

    def state(self): return self.d("GET", "/api/state").json()

    def drive(self, model_id, seconds=40.0, seed=7):
        fd = CT.connect(self.endpoint, timeout=20); self.assertIsNotNone(fd); car = Car(models.find(model_id)["dir"], CW.Writer(fd), speed=60.0, seed=seed); car.start(); self.cars.append(car)
        wait(lambda: car.drive.t - car.drive.t0 >= seconds or not car.is_alive(), 120); return car

    def test_1_it_waits_for_a_car_and_says_so_without_showing_anyone_elses_data(self):
        link = self.state()["link"]; self.assertTrue(link.get("waiting")); self.assertEqual(self.app.dataset, "waiting for a car"); self.assertIsNone(self.app.livecar); self.assertIsNone(self.app.pipe.rt)

    def test_2_a_car_that_connects_becomes_the_history_it_reads_and_its_clock_is_the_clock(self):
        car = self.drive("tesla_model_3"); self.assertTrue(wait(lambda: "tesla_model_3" in (self.app.db or "")), self.state()["link"])
        st = self.state(); self.assertFalse(st["link"].get("waiting")); self.assertEqual(st["link"]["model"], "Tesla Model 3 Long Range"); self.assertGreater(st["sim_t"], 1_000_000_000)
        self.assertLess(abs(st["sim_t"] - car.drive.t), 120, "the application's time is the car's time"); self.assertIsNone(self.app.plan)
        self.assertTrue(self.app.linked.runtime() is not None or not car.is_alive())

    def test_3_the_stored_data_is_what_the_application_reads(self):
        import sqlite3
        c = sqlite3.connect("file:" + self.app.db.replace("\\", "/") + "?mode=ro", uri=True)
        try: self.assertGreater(c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0], 10); self.assertEqual(dict(c.execute("SELECT k, v FROM meta WHERE k='vehicle.id'").fetchall()), {"vehicle.id": "tesla_model_3"})
        finally: c.close()
        t = self.app.telemetry(self.app.clock.now()); self.assertIn("speed", t)

    def test_4_another_car_replaces_it_and_gets_a_history_of_its_own(self):
        first = self.app.db
        for c in self.cars: c.stop()                                                                                     # a car goes before the next one connects (one pipe, one car at a time)
        self.drive("kestrel_petrol", seed=9)
        self.assertTrue(wait(lambda: "kestrel_petrol" in (self.app.db or "")), self.state()["link"]); self.assertNotEqual(first, self.app.db); self.assertTrue(os.path.exists(first), "the first car's history stays")

    def test_5_when_the_car_goes_what_is_stored_stays_and_the_link_says_so(self):
        for c in self.cars: c.stop()
        self.assertTrue(wait(lambda: self.state()["link"].get("offline")), self.state()["link"]); self.assertIn("kestrel_petrol", self.app.db)
        a = self.app.ask("hello", "driver"); self.assertTrue(a.get("answer"))

    def test_6_a_new_start_reads_the_last_stored_vehicle_before_any_car_connects(self):
        a = SimpleNamespace(db="data/vehicle.db", model="none", camera="clip", webcam_index=0, live=None, linked=True, link_home=self.home, link_endpoint=PT.endpoint("t" + secrets.token_hex(4)))
        app2 = self.S.App(a, planner_factory=lambda conn: StubPlanner(conn))
        try: self.assertIn("vehicles", app2.db); self.assertNotEqual(app2.dataset, "waiting for a car")
        finally: app2._close_linked()


if __name__ == "__main__":
    unittest.main()
