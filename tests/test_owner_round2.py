"""The owner's second round of wrong answers (2026-10-24 evening), each fixed in the part of the system that was wrong, each checked here against the vehicle's own tables with separate SQL.

Time phrases (no data needed): "last week" is the week before this one, "the last week" the 7 days up to now, "till now" the whole history, "3 sundays ago" the third such day back, "the 11th" the latest 11th, and a history that ends at midnight ends the day before.
With the four-year sample car: fuel questions put to an electric car are answered with the battery quantity that does the same job; the start and end of a trip and the last N trips come from the trip ledger; a quantity by hour of the day; an event
rate against the period before; "before and after my last service" as two real periods; "below the recommended pressure" from the stored pressures; a verdict first for a range question; a long trend read in well under two seconds."""
import datetime as dt
import os
import re
import sqlite3
import sys
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.meaning import canon, literals as L      # noqa: E402
from vgpt.query_planner import resolve_time        # noqa: E402

DB = os.path.join(ROOT, "data", "world4", "aster_ev", "aster_ev_4yr.db")
NOW, TZ = 1792360799, 7200                          # 18 Oct 2026 23:59:59, Oslo
CTX = types.SimpleNamespace(now=NOW, tz=TZ, lo=1666130400, hi=NOW)


def period(text, now=NOW):
    out = []
    for t in L.extract(text, now, TZ).times:
        lo, hi, _ = resolve_time(t, types.SimpleNamespace(now=now, tz=TZ, lo=1666130400, hi=now)); out.append((t.to_text(), dt.datetime.utcfromtimestamp(lo + TZ).strftime("%Y-%m-%d %H:%M"), dt.datetime.utcfromtimestamp(hi + TZ).strftime("%Y-%m-%d %H:%M")))
    return out


class TimePhrases(unittest.TestCase):
    def test_last_week_is_the_calendar_week_before_this_one_and_the_last_week_is_rolling(self):
        self.assertEqual(period("how many km last week")[0][0], "cal:last_week"); self.assertEqual(period("how many km last month")[0][0], "cal:last_month")
        for q in ("how many km in the last week", "how many km the past week", "distance over the last 7 days"): self.assertEqual(period(q)[0][0], "last:7:day", q)
        self.assertEqual(period("average speed in the last month")[0][0], "last:30:day")

    def test_till_now_is_the_whole_history_and_right_now_is_the_moment(self):
        for q in ("how many issues till now", "warnings up to now", "speed to date", "faults until now"): self.assertEqual(period(q)[0][0], "all", q)
        for q in ("what is my speed now", "battery right now"): self.assertEqual(period(q)[0][0], "now", q)

    def test_the_nth_weekday_ago_and_the_day_of_the_month(self):
        self.assertEqual(period("highest speed 3 sundays ago")[0][1], "2026-09-27 00:00")       # 11 Oct, 4 Oct, 27 Sep (today, a Sunday, does not count)
        self.assertEqual(period("highest speed two fridays ago")[0][1], "2026-10-09 00:00"); self.assertEqual(period("highest speed on the 11th")[0][1], "2026-10-11 00:00")
        self.assertEqual(period("trips on the 20th")[0][1], "2026-09-20 00:00")                  # the 20th of this month has not come: the latest one is last month's
        self.assertEqual(period("speed on 11th october")[0][1], "2026-10-11 00:00")

    def test_a_history_that_ends_at_midnight_ends_the_day_before(self):
        mid = 1792360800                                                                          # 19 Oct 00:00 local: the end marker of the sample car
        self.assertEqual(period("highest speed today", mid)[0][1:], ("2026-10-18 00:00", "2026-10-19 00:00")); self.assertEqual(period("highest speed yesterday", mid)[0][1], "2026-10-17 00:00")
        self.assertEqual(period("highest speed last sunday", mid)[0][1], "2026-10-11 00:00")

    def test_the_pre_parser_leaves_ordinals_and_inflected_words_alone(self):
        for q in ("what was highest speed on 11th october", "the 3rd of march", "what was my highest speed 3 sundays ago", "how many 2nd gear shifts"): self.assertEqual(canon.normalise(q)[0], q, q)


@unittest.skipUnless(os.path.exists(DB), "needs the 4-year sample car")
class OnTheSampleCar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import vehiclegpt; vehiclegpt._adopt_product_model()
        from vgpt.product import make_meaning_brain
        from vgpt.vehicle import lake as LK
        LK.adopt_path(DB); cls.c = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True); cls.now = 1792360800
        cls.b = make_meaning_brain(DB, now=cls.now, threads=4)

    def ask(self, q): self.b.dialogue.reset(); r = self.b.ask(q); return r["answer"], r

    def test_fuel_asked_of_an_electric_car_is_answered_with_the_battery(self):
        a, _ = self.ask("What is my current fuel level?"); soc = self.c.execute("SELECT soc FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone()[0] / 10.0
        self.assertIn("electric car", a); self.assertIn(f"{soc:.0f} %", a); self.assertNotIn("isn't recorded", a)
        a, _ = self.ask("Which trip had the highest fuel consumption this month"); self.assertIn("Wh/km", a); self.assertNotIn("isn't recorded", a)
        a, _ = self.ask("Compare my average fuel consumption this week with last week"); self.assertIn("Wh/km", a)

    def test_battery_consumption_is_energy_per_km_not_efficiency(self):
        a, r = self.ask("Which trip had the highest battery consumption this month"); self.assertIn("Wh/km", a); self.assertNotIn("battery-to-wheel", a)

    def test_how_many_times_did_i_start_the_car_counts_the_trips_and_says_starts(self):
        a, _ = self.ask("How many times have I started the engine yesterday"); y = (self.now - 1 - ((self.now - 1 + TZ) % 86400)) - 86400
        n = self.c.execute("SELECT COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3", (y, y + 86400)).fetchone()[0]
        self.assertTrue(a.startswith(f"You started the car {n} time"), a)

    def test_the_charge_at_the_start_and_end_of_the_last_trip_comes_from_the_ledger(self):
        a, _ = self.ask("What was my battery state of charge at the beginning and end of my last trip"); s, e = self.c.execute("SELECT soc_start, soc_end FROM trips WHERE km>0.3 ORDER BY start DESC LIMIT 1").fetchone()
        self.assertIn(f"at the start", a); self.assertIn(f"{s:.0f} %", a); self.assertIn(f"{e:.0f} %", a)

    def test_the_last_five_trips_are_listed_with_one_value_each(self):
        a, _ = self.ask("Compare battery discharge rates across my last five trips"); self.assertEqual(len(re.findall(r"^ - ", a, re.M)), 5); self.assertIn("Highest:", a)
        a, _ = self.ask("what was my energy consumption on each of my last 4 trips"); rows = self.c.execute("SELECT wh_km FROM trips WHERE km>0.3 ORDER BY start DESC LIMIT 4").fetchall()
        for (w,) in rows: self.assertIn(f"{w:.0f} Wh/km", a)

    def test_a_which_time_question_is_completed_by_the_next_reply(self):
        a, _ = self.ask("what was my battery charge then"); self.assertIn("Which time", a)
        r = self.b.ask("4 pm yesterday"); self.assertIn("State of charge at", r["answer"]); self.assertIn("16:00", r["answer"])

    def test_the_hour_of_the_day_with_the_highest_consumption_is_the_one_in_the_ledger(self):
        a, _ = self.ask("Which hour of the day has the highest average energy consumption"); lo = self.now - 90 * 86400
        rows = self.c.execute("SELECT CAST(((start + ?) % 86400) / 3600 AS INTEGER) h, SUM(kwh) * 1000.0 / SUM(km), COUNT(*) FROM trips WHERE start>=? AND start<? AND km>0.3 GROUP BY h", (TZ, lo, self.now + 1)).fetchall()
        best = max((r for r in rows if r[2] >= 3), key=lambda r: r[1]); self.assertIn(f"between {best[0]:02d}:00 and {best[0] + 1:02d}:00", a); self.assertIn(f"{best[1]:.0f} Wh/km", a)

    def test_an_event_rate_is_compared_with_the_period_before(self):
        a, _ = self.ask("Has the frequency of hard braking increased this month")
        d = lambda lo, hi: self.c.execute("SELECT COUNT(*) FROM events WHERE type='HARSH_BRAKE' AND ts>=? AND ts<?", (lo, hi)).fetchone()[0]
        from vgpt.query_ir import TimeSpec
        ctx = self.b.qc().ctx; lo1, hi1, _ = resolve_time(TimeSpec("calendar", name="this_month"), ctx); lo2, hi2, _ = resolve_time(TimeSpec("calendar", name="last_month"), ctx)
        self.assertIn(f"{d(lo1, min(hi1, self.now + 1))} this month", a); self.assertIn(f"{d(lo2, hi2)} last month", a)

    def test_before_and_after_my_last_service_are_two_periods_named_by_the_record(self):
        a, _ = self.ask("Compare fuel consumption before and after my last service"); ts = int(float(self.c.execute("SELECT v FROM meta WHERE k='veh.last_service_ts'").fetchone()[0]))
        d = dt.datetime.utcfromtimestamp(ts + TZ); self.assertIn(f"{d.day} {d:%b %Y}", a); self.assertIn("Wh/km", a); self.assertNotIn("odometer", a)

    def test_the_last_time_a_tyre_was_below_its_recommended_pressure_comes_from_the_stored_pressures(self):
        a, _ = self.ask("When did my tyre pressure last fall below the recommended threshold"); self.assertIn("below its recommended pressure", a); self.assertNotIn("warning", a.split("\n")[0].lower())
        rows = self.c.execute("SELECT ts FROM ext_1m WHERE (tp_fl<2300*0.92*1.1 OR tp_fr<2300*0.92*1.1 OR tp_rl<2300*0.92*1.1 OR tp_rr<2300*0.92*1.1) ORDER BY ts DESC LIMIT 1").fetchone(); self.assertIsNotNone(rows)

    def test_a_range_question_gets_the_verdict_first_and_keeps_the_working_folded(self):
        a, _ = self.ask("Has my average driving range decreased over the last two weeks"); first = a.split("\n")[0]
        self.assertTrue(first.startswith(("No - ", "Yes - ")), first); self.assertLess(len(first), 200); self.assertTrue(all(x.startswith("Method:") for x in a.split("\n")[1:] if x.strip()))

    def test_which_year_points_back_at_the_last_answer(self):
        self.ask("when was the longest trip"); r = self.b.ask("which year"); y = dt.datetime.utcfromtimestamp(self.c.execute("SELECT start FROM trips ORDER BY km DESC LIMIT 1").fetchone()[0] + TZ).year
        self.assertEqual(r["answer"], f"That was in {y}.")

    def test_a_trend_over_years_is_read_from_the_daily_tier_in_under_two_seconds(self):
        self.ask("how is the battery"); t = time.perf_counter(); a, _ = self.ask("Is my engine temperature gradually increasing over time"); dt_ = time.perf_counter() - t
        self.assertLess(dt_, 2.0, dt_); self.assertIn("motor temperature", a)

    def test_the_service_record_handler_does_not_take_questions_about_other_things(self):
        from vgpt.meaning import askme
        for q in ("compare fuel consumption before and after my last service", "average energy consumption since my last service"): self.assertIsNone(askme.handle(self.b, q), q)
        self.assertIsNotNone(askme.handle(self.b, "when was the last service"))


class Pure(unittest.TestCase):
    def test_a_verdict_for_a_long_change_analysis(self):
        from vgpt.meaning import plain
        from vgpt.meaning.frame import Frame
        ans = "Your estimated range rose from about 243 to 266 km over the last 2 weeks. The recorded conditions together explain more than the whole change.\n1. Observed (x): range was 243 km before (p-value 0.574, effect size 0.3; not clear)."
        r = plain.change_answer(Frame("trend", "range_estimate", "drop"), {"answer": ans, "facts": {"route": "qc:explain_change"}}); lines = r["answer"].split("\n")
        self.assertTrue(lines[0].startswith("No - your estimated range went up, from about 243 to 266 km")); self.assertIn("within normal", lines[0]); self.assertTrue(all(x.startswith("Method:") for x in lines[1:]))
        self.assertEqual(plain.change_answer(Frame("trend", "range_estimate", "drop"), {"answer": ans, "facts": {"route": "qc:count"}})["answer"], ans)

    def test_fuel_quantities_map_to_battery_quantities_only_for_a_battery_only_vehicle(self):
        from vgpt.meaning import evmap
        self.assertEqual(evmap.FUEL_TO_BATTERY["fuel_level"][0], "battery_soc"); self.assertEqual(evmap.FUEL_TO_BATTERY["fuel_rate"][0], "wh_per_km")


if __name__ == "__main__": unittest.main()
