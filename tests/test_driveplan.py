"""The car drives by itself: the head-unit clock follows a plan made from the recorded trips of the loaded history (real time while driving, parked time skipped), and the page offers nothing to drive it with."""
import importlib.util
import os
import sqlite3
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODERN = os.path.join(ROOT, "ui", "modern")
DAY = 86400


def server():
    spec = importlib.util.spec_from_file_location("uiserver_driveplan", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = server(); cls.t0 = 1_780_000_000
        cls.trips = [(cls.t0 + d * DAY + h * 3600, cls.t0 + d * DAY + h * 3600 + 1500) for d in range(100) for h in (8, 18)]          # two 25-minute drives a day for 100 days
        cls.plan = cls.m.DrivePlan(cls.trips)

    def test_it_starts_far_enough_into_the_history_for_weeks_of_questions(self):
        self.assertGreaterEqual(self.plan.begin - self.t0, 55 * DAY); self.assertLess(self.plan.begin, self.plan.end)

    def test_everything_runs_in_real_time_drives_and_the_parked_time_between_them(self):
        a, b = self.plan.starts[0], self.plan.starts[1]; self.assertEqual(self.plan.at(a + 5)[0], 1.0)
        self.assertEqual(self.plan.at(b + 5)[0], 1.0, "nothing is sped up on its own: the app reads the car calmly, as it would installed in it")
        self.assertTrue(all(rate in (0.0, 1.0) for rate, _ in self.plan.segs))

    def test_the_time_speed_slider_is_the_only_thing_that_speeds_time_up(self):
        clk = self.m.Clock(0); clk.use(self.plan, scenario="x"); self.assertEqual(clk.mult, 1.0); self.assertEqual(clk.rate(), 1.0)
        for bad in (0, -1, 3, 7, 1000, "fast", None): self.assertEqual(clk.set_mult(bad), 1.0, bad)                          # only the slider's steps are accepted
        self.assertEqual(clk.set_mult(60), 60.0); a = clk.now(); time.sleep(0.3); self.assertAlmostEqual((clk.now() - a) / 60.0, 0.3, delta=0.15); self.assertEqual(clk.rate(), 60.0)
        clk.use(self.plan, scenario="x"); self.assertEqual(clk.mult, 1.0, "loading a dataset starts in real time again")

    def test_a_sped_up_clock_still_stops_where_the_next_drive_begins_and_loops_at_the_end(self):
        clk = self.m.Clock(0); clk.use(self.plan, scenario="x"); clk.set_mult(600); clk.set(t=self.plan.end - 30); time.sleep(0.3); t = clk.now(); self.assertGreaterEqual(clk.laps, 1); self.assertLess(t - self.plan.begin, 600)
        self.assertEqual(self.m.Clock.MULTS[0], 1)

    def test_it_runs_at_one_second_per_second_while_driving(self):
        clk = self.m.Clock(0); clk.use(self.plan, scenario="x"); a = clk.now(); time.sleep(0.3); self.assertAlmostEqual(clk.now() - a, 0.3, delta=0.15)

    def test_when_the_last_drive_is_over_the_history_plays_again(self):
        clk = self.m.Clock(0); clk.use(self.plan, scenario="x"); clk.set(t=self.plan.end - 0.2); time.sleep(0.6); t = clk.now(); self.assertEqual(clk.laps, 1); self.assertGreaterEqual(t, self.plan.begin); self.assertLess(t - self.plan.begin, 1.0)

    def test_without_looping_the_car_stays_where_it_is(self):
        plan = self.m.DrivePlan(self.trips, loop=False); clk = self.m.Clock(0); clk.use(plan, scenario="x"); clk.set(t=plan.end - 0.2); time.sleep(0.6); self.assertEqual(clk.now(), plan.end); self.assertEqual(clk.rate(), 0.0); self.assertEqual(clk.laps, 0)

    def test_the_part_of_the_history_stored_once_per_second_is_the_part_that_plays(self):
        hires = self.t0 + 93 * DAY; plan = self.m.DrivePlan(self.trips, hires_from=hires); self.assertGreaterEqual(plan.begin + 25, hires); self.assertLess(plan.begin, hires + DAY)

    def test_a_history_without_trips_has_no_plan(self):
        c = sqlite3.connect(":memory:"); self.assertIsNone(self.m.make_plan(c)); c.execute("CREATE TABLE trips(start INT, end INT)"); self.assertIsNone(self.m.make_plan(c))

    def test_the_shipped_histories_get_a_plan(self):
        for name in ("world_aster.db", "world_meridian.db"):
            p = os.path.join(ROOT, "data", name)
            if not os.path.exists(p): continue
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True); plan = self.m.make_plan(c); c.close(); self.assertIsNotNone(plan, name); self.assertGreater(plan.end - plan.begin, 3 * DAY, name)


class Sparse(unittest.TestCase):
    """A recording with one sample a minute (Case Study 5's diesel file) gets a plan from its single trip and starts when the car first moves - still in real time."""
    @classmethod
    def setUpClass(cls):
        cls.m = server(); cls.t0 = 1_788_220_800; cls.c = sqlite3.connect(":memory:")
        cls.c.execute("CREATE TABLE sig_1s(ts INT, speed INT)"); cls.c.execute("CREATE TABLE trips(start INT, end INT)")
        cls.c.executemany("INSERT INTO sig_1s VALUES(?,?)", [(cls.t0 + 60 * i, 0 if i < 500 else 500 + 10 * (i % 40)) for i in range(1500)]); cls.c.execute("INSERT INTO trips VALUES(?,?)", (cls.t0, cls.t0 + 60 * 1499))

    def test_one_long_trip_of_minute_samples_gets_a_real_time_plan_that_starts_when_the_car_moves(self):
        plan = self.m.make_plan(self.c); self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan.begin, self.t0 + 60 * 495); self.assertLess(plan.begin, self.t0 + 60 * 500, "the standing start is not played")
        self.assertEqual(plan.at(plan.begin + 10)[0], 1.0, "no automatic speed-up, even for a recording that changes once a minute")

    def test_the_clock_runs_one_second_per_second_and_says_nothing_about_speed(self):
        plan = self.m.make_plan(self.c); clk = self.m.Clock(0); clk.use(plan, scenario="x"); a = clk.now(); time.sleep(0.4); self.assertAlmostEqual(clk.now() - a, 0.4, delta=0.2)
        js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read(); self.assertNotIn("recording shown", js); self.assertNotIn("skipping ahead", js); self.assertIn("', time ×'", js, "only a speed the person chose is shown")

    def test_one_second_data_with_one_trip_still_gets_no_plan_and_a_few_trips_get_one(self):
        c = sqlite3.connect(":memory:"); c.execute("CREATE TABLE sig_1s(ts INT, speed INT)"); c.execute("CREATE TABLE trips(start INT, end INT)")
        c.executemany("INSERT INTO sig_1s VALUES(?,?)", [(self.t0 + i, 100) for i in range(600)]); c.execute("INSERT INTO trips VALUES(?,?)", (self.t0, self.t0 + 599)); self.assertIsNone(self.m.make_plan(c))
        for k in range(3): c.execute("INSERT INTO trips VALUES(?,?)", (self.t0 + 1000 * k, self.t0 + 1000 * k + 500))
        self.assertIsNotNone(self.m.make_plan(c))

    def test_the_hero_tag_follows_the_drive_state_even_without_a_fuel_or_battery_gauge(self):
        js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read(); i = js.index("function renderHeroCar"); body = js[i:js.index("\nfunction lightArc", i)]
        self.assertNotIn("if (!el) return;", body, "a speed-only vehicle has no #hcSoc, but its state tag must still update")


class Page(unittest.TestCase):
    def test_the_page_has_no_live_controls_and_the_test_track_has_only_the_dataset(self):
        html = open(os.path.join(MODERN, "index.html"), encoding="utf-8").read(); js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read(); css = open(os.path.join(MODERN, "app.css"), encoding="utf-8").read()
        for gone in ("pg-live", "railLive", "lvctl", 'type="range"'): self.assertNotIn(gone, html, gone)
        for gone in ("Playback speed", "Inject a camera fault", "Camera source", "Replay a moment", "liveTick", "/api/live", "/api/fault", "/api/playback", "/api/camera"): self.assertNotIn(gone, js, gone)
        self.assertIn('id="gDs"', js); self.assertNotIn(".lvctl", css); self.assertNotIn(".scn", css)
        self.assertIn('id="gTs"', js); self.assertIn("/api/timescale", js); self.assertIn("Real time", js)                     # the one speed control the owner asked for: a Time speed slider in the Test track section, real time by default

    def test_what_the_page_hides_is_really_hidden(self):
        css = open(os.path.join(MODERN, "app.css"), encoding="utf-8").read(); self.assertIn("[hidden]{display:none!important}", css, "the rail entries have their own display rule: without this the Energy and Map entries stay on screen for a vehicle that has neither")
        chk = open(os.path.join(MODERN, "layout_check.js"), encoding="utf-8").read(); self.assertIn("MARKED HIDDEN BUT STILL DRAWN", chk)

    def test_the_server_does_not_start_the_virtual_car_unless_asked(self):
        src = open(os.path.join(ROOT, "ui", "server.py"), encoding="utf-8").read(); self.assertIn('"--live-car", default="off"', src)


if __name__ == "__main__":
    unittest.main()
