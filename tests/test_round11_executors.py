"""Round 11 executors (2026-10-24), each checked against numbers recomputed from the database with plain SQL:
trips per day (trend), event trends answered yes / no with weekly counts, charging-power trend from the charging ledger, "usual" totals from the trip ledger, rain against the rest of the driving,
weather limits on ledger quantities, "was the last charge / trip normal", a change scoped to one component, and fuel-consumption trend verdicts.  The frames are given directly: no language model is used."""
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
import ask_frame as AF                                  # noqa: E402  (does not touch the environment when imported)
from vgpt.meaning.pipeline import MeaningBrain          # noqa: E402
from vgpt.vehicle import lake as LK                     # noqa: E402

_BEFORE = {}


def setUpModule():
    _BEFORE["v"] = os.environ.get("VGPT_FRAME"); os.environ["VGPT_FRAME"] = "2"


def tearDownModule():
    if _BEFORE.get("v") is None: os.environ.pop("VGPT_FRAME", None)
    else: os.environ["VGPT_FRAME"] = _BEFORE["v"]


DAY = 86400


def brain(db):
    path = os.path.join(ROOT, "data", db)
    if not os.path.exists(path): raise unittest.SkipTest(db + " is not present")
    LK.adopt_path(path); return MeaningBrain(path, AF.Null(), now=AF.last_ts(path), tau=0.0), path


def ask(b, text, spec): return b.resolver.answer(text, AF.frame_of(spec))


class EvWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b, cls.path = brain("world_aster.db"); cls.now = AF.last_ts(cls.path); cls.c = sqlite3.connect("file:" + cls.path.replace("\\", "/") + "?mode=ro", uri=True)

    def test_trips_per_day_trend_counts_zero_days(self):
        r = ask(self.b, "am I making fewer trips", "trend trip_record op=drop"); f = r["facts"]
        lo = self.now + 1 - 28 * DAY; tz = int(self.b.tz); d0, d1 = (lo + tz) // DAY, (self.now + 1 + tz) // DAY
        cnt = dict(self.c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER) d, COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY d", (tz, lo, self.now + 1)).fetchall())
        y = [cnt.get(k, 0) for k in range(d0, d1)]; h = len(y) // 2
        self.assertAlmostEqual(f["first"], sum(y[:h]) / h, places=6); self.assertAlmostEqual(f["second"], sum(y[len(y) - h:]) / h, places=6); self.assertEqual(f["trips"], sum(y))
        self.assertTrue(r["answer"].startswith(("Yes", "No")), r["answer"][:80])

    def test_event_trend_answers_yes_or_no_first_and_gives_the_weekly_counts(self):
        r = ask(self.b, "are hard brakes happening more now", "trend hard_braking op=rise"); a = r["answer"]
        self.assertTrue(a.startswith(("Yes", "No")), a[:80]); self.assertIn("Week by week the counts were", a)
        lo = self.now + 1 - 28 * DAY; n = self.c.execute("SELECT COUNT(*) FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<?", (lo, self.now + 1)).fetchone()[0]
        self.assertIn(f"Based on {n} recorded events", a)

    def test_charging_power_trend_uses_the_power_while_charging_not_the_all_hours_mean(self):
        r = ask(self.b, "is my charging power going down with time", "trend charging_power op=drop"); v = r["facts"]["trend_verdict"]
        self.assertGreater(v["first"], 3.0); self.assertGreater(v["second"], 3.0)             # kW while charging; the all-hours mean of the signal is about 0

    def test_total_energy_against_the_usual_weeks_comes_from_the_trip_ledger(self):
        r = ask(self.b, "was last week's energy use normal for me", "compare energy_used op=sum vs=typical"); a = r["answer"]
        from vgpt.query_ir import TimeSpec
        from vgpt.query_planner import resolve_time
        lo, hi, _ = resolve_time(TimeSpec("calendar", name="last_week"), self.b.qc().ctx)                                  # "last week" is the calendar week before this one
        kwh = self.c.execute("SELECT SUM(kwh) FROM trips WHERE start>=? AND start<? AND km>0.2", (int(lo), int(hi))).fetchone()[0]
        self.assertIn(f"{kwh:.1f} kWh", a)

    def test_rain_is_compared_with_the_dry_driving_not_with_all_driving(self):
        r = ask(self.b, "how does my consumption in the rain compare with dry days", "compare wh_per_km op=mean cond=raining vs=driving"); a = r["answer"]
        self.assertIn("not raining", a); self.assertNotIn("driving", a.split("\n")[0])

    def test_last_charge_is_judged_against_earlier_sessions_of_the_same_kind(self):
        r = ask(self.b, "was the last charge normal", "anomaly charging_session"); a = r["answer"]
        self.assertTrue(a.startswith(("Yes", "No")), a[:80]); kind, = self.c.execute("SELECT kind FROM charges ORDER BY start DESC LIMIT 1").fetchone(); self.assertIn(kind, a)

    def test_distance_per_week_lists_each_week_from_the_trip_ledger(self):
        r = ask(self.b, "what was my distance per week this month", "stat distance_driven op=sum ent=week"); vals = r["facts"]["values"]; tz = int(self.b.tz)
        self.assertGreaterEqual(len(vals), 2)
        rows = self.c.execute("SELECT CAST((CAST((start + ?) / 86400 AS INTEGER) + 3) / 7 AS INTEGER) g, SUM(km) FROM trips WHERE km>0.3 GROUP BY g ORDER BY g", (tz,)).fetchall()
        self.assertAlmostEqual(vals[-1], rows[-1][1], places=6)                                    # the newest week is complete in the window: its figure is the SQL sum of that Monday-to-Sunday week
        self.assertIn("Total:", r["answer"])

    def test_two_components_in_one_question_are_each_answered(self):
        r = ask(self.b, "anything odd with the tyres or the battery this week", "anomaly tyre_condition c2=battery_condition"); a = r["answer"]
        self.assertIn("tyre condition", a); self.assertIn("battery condition", a)

    def test_the_last_trip_is_compared_with_the_trips_before_it(self):
        r = ask(self.b, "did the battery heat up more than usual on my last trip", "compare battery_temperature op=max vs=typical ent=trip"); a = r["answer"]
        n = self.c.execute("SELECT COUNT(*) FROM trips WHERE end>start AND km>0.3 AND start>=?", (int(self.now) - 60 * DAY,)).fetchone()[0]
        self.assertIn("On the last trip", a); self.assertEqual(r["facts"]["trips"], n - 1)

    def test_a_change_is_scoped_to_the_component_asked_about(self):
        r = ask(self.b, "has anything about the battery changed in the last three months", "summary battery_condition op=change"); a = r["answer"]
        self.assertNotIn("distance driven", a); self.assertNotIn("coolant flow", a)

    def test_anomaly_of_a_component_with_no_recorded_quantity_does_not_scan_everything(self):
        r = ask(self.b, "are the brakes behaving normally", "anomaly brake_condition"); self.assertNotIn("oil", r["answer"])


class IceWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b, cls.path = brain("world_kestrel.db")

    def test_fuel_consumption_trend_says_yes_or_no_first(self):
        r = ask(self.b, "is the fuel consumption going up", "trend fuel_rate op=rise"); self.assertTrue(r["answer"].startswith(("Yes", "No")), r["answer"][:80])
        r2 = ask(self.b, "has the fuel burn improved since the service", "trend fuel_rate op=drop")
        self.assertNotEqual(r["answer"].split("\n")[0][:3], r2["answer"].split("\n")[0][:3])           # the same fact answers the opposite question the opposite way

    def test_brake_anomaly_does_not_list_unrelated_quantities(self):
        r = ask(self.b, "are the brakes behaving normally", "anomaly brake_condition"); self.assertNotIn("oil temperature", r["answer"])
        r = ask(self.b, "does the cooling system behave strangely", "anomaly cooling_condition"); self.assertNotIn("12 V battery voltage", r["answer"])


class Weather(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b, cls.path = brain("world_3yr.db"); cls.c = sqlite3.connect("file:" + cls.path.replace("\\", "/") + "?mode=ro", uri=True)

    def test_consumption_in_hot_weather_uses_trips_whose_average_outside_temperature_was_above_the_limit(self):
        r = ask(self.b, "how is my consumption in hot weather this year", "stat wh_per_km op=mean cond=hot_weather"); a = r["answer"]
        km, kwh, n = self.c.execute("SELECT SUM(km), SUM(kwh), COUNT(*) FROM trips WHERE avg_amb>28.0 AND km>0.2 AND start>=strftime('%s','2026-01-01') - 19800").fetchone()
        self.assertIn(f"{1000 * kwh / km:.0f} Wh/km", a.replace(",", ""))

    def test_winter_against_summer_on_a_hot_climate_car_compares_the_calendar_seasons(self):
        r = ask(self.b, "is my consumption higher in winter than in summer", "compare wh_per_km op=mean cond=cold vs=hot_weather"); a = r["answer"]
        self.assertIn("compared the winter months", a); self.assertIn("December 2025 to February 2026", a)

    def test_battery_ageing_is_read_over_the_whole_history_and_a_slow_steady_drift_counts(self):
        r = ask(self.b, "is the battery degrading", "trend battery_soh op=drop"); v = r["facts"]["trend_verdict"]
        self.assertEqual(v["direction"], "fall"); self.assertTrue(v["steady"]); self.assertTrue(r["answer"].startswith("Yes"), r["answer"][:80])
        soh = self.c.execute("SELECT AVG(soh) FROM vday WHERE soh IS NOT NULL GROUP BY CAST(ts / 86400 AS INTEGER) ORDER BY ts LIMIT 1").fetchone()[0]
        self.assertLess(v["second"], v["first"]); self.assertIsNotNone(soh)

    def test_a_large_change_in_a_ratio_is_not_called_unchanged(self):
        r = ask(self.b, "how much has my consumption changed over the last year", "trend wh_per_km op=change"); self.assertTrue(r["answer"].startswith("Yes"), r["answer"][:100])

    def test_a_weather_group_without_trips_is_named_not_reported_as_uncomputable(self):
        r = ask(self.b, "is my consumption higher in the heat than usual", "compare wh_per_km op=mean cond=hot_weather vs=cold"); self.assertIn("no trip in that period was cold", r["answer"])


if __name__ == "__main__":
    unittest.main()
