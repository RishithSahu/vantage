"""Calendar periods and long histories: a year or month typed in a question, counting past 200 trips, a parked car's last known fuel level, the camera not crying wolf in the dark, and (when the 3-year world
is present) the assistant's answers against the numbers its own database gives."""
import datetime as dt
import os
import sqlite3
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import literals as L  # noqa: E402
from vgpt.query_ir import Query, TimeSpec  # noqa: E402
from vgpt.query_planner import resolve_time  # noqa: E402
from vgpt.query_validator import validate  # noqa: E402

NOW = 1792450800   # 2026-10-19 23:00 UTC; the history's tz is +1 h
TZ = 3600


def names(text): return [t.to_text() for t in L.calendar_periods(text, NOW, TZ)]


class Periods(unittest.TestCase):
    def test_years_and_months_typed_in_full(self):
        self.assertEqual(names("how many trips in 2024"), ["cal:y2024"])
        self.assertEqual(names("how far did I drive in March 2025"), ["cal:m2025_03"])
        self.assertEqual(names("compare 2024 versus 2025"), ["cal:y2024", "cal:y2025"])
        self.assertEqual(names("trips last year"), ["cal:last_year"])
        self.assertEqual(names("any shutdown ever"), ["all"])

    def test_a_number_that_is_not_a_year_is_left_alone(self):
        self.assertEqual(names("is 2024 rpm too high"), [])
        self.assertEqual(names("how many trips on 17 march 2025"), [])                         # a full date is an instant

    def test_the_model_never_sees_a_bare_year(self):
        self.assertEqual(L.mask_calendar("how far did I drive in March 2025"), "how far did I drive in last month")
        self.assertEqual(L.mask_calendar("top speed 2024 km/h"), "top speed 2024 km/h")

    def test_perfect_tense_with_no_period_is_the_whole_history_but_a_typed_period_wins(self):
        self.assertEqual([t.to_text() for t in L.extract("how many shutdowns have there been", NOW, TZ).times], ["all"])
        self.assertEqual([t.to_text() for t in L.extract("how many hard brakes have i had this week", NOW, TZ).times], ["cal:this_week"])

    def test_last_n_years(self):
        self.assertEqual([t.to_text() for t in L.extract("trips in the last 2 years", NOW, TZ).times], ["last:730:day"])

    def test_resolution_to_local_seconds(self):
        ctx = types.SimpleNamespace(now=NOW, tz=TZ, lo=0, hi=NOW)
        lo, hi, label = resolve_time(TimeSpec("calendar", name="y2024"), ctx)
        self.assertEqual((dt.datetime.fromtimestamp(lo + TZ, dt.timezone.utc).isoformat()[:10], dt.datetime.fromtimestamp(hi + TZ, dt.timezone.utc).isoformat()[:10]), ("2024-01-01", "2025-01-01"))
        lo, hi, _ = resolve_time(TimeSpec("calendar", name="m2025_02"), ctx); self.assertEqual((hi - lo) // 86400, 28)
        lo, hi, _ = resolve_time(TimeSpec("calendar", name="this_year"), ctx); self.assertEqual(hi, NOW + 1)
        lo, hi, _ = resolve_time(TimeSpec("relative", 12, "month"), ctx)
        self.assertEqual(dt.datetime.fromtimestamp(lo + TZ, dt.timezone.utc).strftime("%Y-%m"), "2025-10")

    def test_the_validator_accepts_these_names_and_nothing_else(self):
        from vgpt.query_ir import is_calendar_name
        for ok in ("y2025", "m2025_03", "last_year", "this_year", "today"): self.assertTrue(is_calendar_name(ok), ok)
        for bad in ("y25", "m2025-03", "m2025_13", "year2025", ""): self.assertFalse(is_calendar_name(bad), bad)


class Camera(unittest.TestCase):
    def test_a_dark_scene_is_low_light_not_a_blurred_lens(self):
        import numpy as np
        from vgpt.camera import CameraMonitor
        rng = np.random.default_rng(1); m = CameraMonitor()
        day = (np.clip(110 + 70 * np.sin(np.add.outer(np.arange(120) / 9.0, np.arange(160) / 7.0)), 0, 255)).astype(np.uint8)
        for i in range(10): m.process(i * 0.2, day)
        night = np.clip(14 + rng.normal(0, 2.0, (120, 160)) + 4 * np.sin(np.add.outer(np.arange(120) / 9.0, np.arange(160) / 7.0)), 0, 255).astype(np.uint8)
        events = []
        for i in range(30): events += [r for r in m.process(3 + i * 0.2, night)[1] if r[0] == "event"]
        self.assertEqual([e for e in events if e[1] == "CAMERA_DEGRADED"], [])


class LastKnownFuel(unittest.TestCase):
    def test_a_parked_cars_fuel_level_is_still_what_it_was(self):
        from vgpt.vehicle import caps
        c = sqlite3.connect(":memory:")
        c.executescript("CREATE TABLE sig_catalog(sid INTEGER, concept TEXT); CREATE TABLE lk_1s(sid INT, ts INT, v REAL); CREATE TABLE lk_1m(sid INT, ts INT, v REAL);")
        c.execute("INSERT INTO sig_catalog VALUES(1,'fuel_level'),(2,'engine_rpm')"); c.execute("INSERT INTO lk_1m VALUES(1,1000,66.0)"); c.execute("INSERT INTO lk_1m VALUES(2,1000,0.0)")
        got = caps.Reader(c).at(["fuel_level", "engine_rpm"], 1000 + 3600)
        self.assertEqual(got, {"fuel_level": 66.0})                                                 # the tank, not the revs: a stale engine speed is not shown as current


THREE_YEARS = os.path.join(ROOT, "data", "world_3yr.db")


@unittest.skipUnless(os.path.exists(THREE_YEARS), "data/world_3yr.db is not present (python -m vgpt.world.build x3, then tools_merge_years.py)")
class ThreeYearCounting(unittest.TestCase):
    def test_every_trip_of_three_years_is_counted(self):
        from vgpt.product import make_brain
        c = sqlite3.connect(f"file:{THREE_YEARS}?mode=ro", uri=True); end = int(c.execute("SELECT v FROM meta WHERE k='end'").fetchone()[0]); n = c.execute("SELECT COUNT(*) FROM trips").fetchone()[0]; c.close()
        b = make_brain(THREE_YEARS, now=end)
        r = b.ask("how many trips in total")
        self.assertIn(f"{n}", r["answer"].replace(",", ""))


if __name__ == "__main__":
    unittest.main()
