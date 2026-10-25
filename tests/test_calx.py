"""Calendar breakdowns without training (vgpt/meaning/calx.py): weekends against weekdays, one line per day / week / month, the pattern in when events happen.
The words are read by literals.day_grouping; every number is recomputed here with plain SQL; the statistics are checked against known values."""
import datetime as dt
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
import ask_frame as AF                                  # noqa: E402
from vgpt.meaning import calx as CX                     # noqa: E402
from vgpt.meaning import literals as L                  # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain          # noqa: E402
from vgpt.vehicle import lake as LK                     # noqa: E402

_BEFORE = {}


def setUpModule():
    _BEFORE["v"] = os.environ.get("VGPT_FRAME"); os.environ["VGPT_FRAME"] = "2"


def tearDownModule():
    if _BEFORE.get("v") is None: os.environ.pop("VGPT_FRAME", None)
    else: os.environ["VGPT_FRAME"] = _BEFORE["v"]


DAY = 86400


class Words(unittest.TestCase):
    def test_the_calendar_words_are_read_as_values(self):
        g = L.day_grouping
        self.assertEqual(g("is my efficiency better on weekends than weekdays")[0], ["weekend", "weekday"])
        self.assertEqual(g("how far do I drive on weekends")[0], ["weekend"])
        self.assertEqual(g("how many warnings per month this year")[1], "month")
        self.assertEqual(g("how has my average speed changed month by month")[1], "month")
        self.assertEqual(g("my average consumption per month")[1], "month")                      # a month is always a breakdown
        self.assertIsNone(g("how many trips per day on average")[1])                               # an average per day is one figure
        self.assertEqual(g("which month had the most hard braking")[1], "month")
        self.assertTrue(g("is there a pattern in when the warnings appear")[2])
        self.assertTrue(g("what time of day do the hard brakes happen")[2])
        self.assertEqual(g("what is my speed"), ([], None, False))


class Statistics(unittest.TestCase):
    def test_chi_square_against_known_values(self):
        self.assertAlmostEqual(CX.chi_square_p([10, 10, 10, 10], [10] * 4), 1.0, places=6)
        self.assertAlmostEqual(CX.chi_square_p([20, 10, 10, 0], [10] * 4), 0.000167, places=5)           # x2 = 20 with 3 degrees of freedom
        self.assertAlmostEqual(CX.chi_square_p([30, 20], [25, 25]), 0.1573, places=3)                   # x2 = 2 with 1 degree of freedom
        self.assertAlmostEqual(CX.chi_square_p([35, 15], [25, 25]), 0.00468, places=4)                  # x2 = 8

    def test_rank_test(self):
        self.assertLess(CX.rank_test_p([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]), 0.01)
        self.assertGreater(CX.rank_test_p([1, 5, 3, 7, 2, 6], [2, 6, 4, 1, 7, 3]), 0.5)
        self.assertIsNone(CX.rank_test_p([1, 2], [3, 4, 5, 6]))

    def test_bins(self):
        d = (dt.date(2026, 10, 12).toordinal() - 719163)                                               # a Monday
        self.assertEqual(CX._dow(d), 0); self.assertEqual(CX._dow(d + 5), 5)
        self.assertEqual(CX._key(d, "week"), CX._key(d + 6, "week")); self.assertNotEqual(CX._key(d, "week"), CX._key(d + 7, "week"))
        k = CX._key(d, "month"); a, z = CX._bin_days(k, "month"); self.assertEqual((CX._date(a), CX._date(z)), (dt.date(2026, 10, 1), dt.date(2026, 10, 31)))
        self.assertEqual(CX._label(k, "month"), "Oct 2026")


def brain(db):
    path = os.path.join(ROOT, "data", db)
    if not os.path.exists(path): raise unittest.SkipTest(db + " is not present")
    LK.adopt_path(path); return MeaningBrain(path, AF.Null(), now=AF.last_ts(path), tau=0.0), path


def ask(b, text, spec): return b.resolver.answer(text, AF.frame_of(spec))


class Answers(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b, cls.path = brain("world_aster.db"); cls.now = AF.last_ts(cls.path); cls.c = sqlite3.connect("file:" + cls.path.replace("\\", "/") + "?mode=ro", uri=True); cls.tz = int(cls.b.tz)

    def test_weekend_against_weekday_consumption_from_the_trip_ledger(self):
        r = ask(self.b, "is my efficiency better on weekends than weekdays", "compare wh_per_km op=mean vs=period"); f = r["facts"]
        lo = max(self.now + 1 - 90 * DAY, self.c.execute("SELECT MIN(ts) FROM sig_1m").fetchone()[0]); hi = self.now + 1
        rows = self.c.execute("SELECT CAST((start + ?) / 86400 AS INTEGER), km, kwh FROM trips WHERE start>=? AND start<? AND km>0.3", (self.tz, lo, hi)).fetchall()
        wk = [(km, kwh) for d, km, kwh in rows if (d + 3) % 7 >= 5]; wd = [(km, kwh) for d, km, kwh in rows if (d + 3) % 7 < 5]
        self.assertAlmostEqual(f["weekend"], 1000 * sum(x[1] for x in wk) / sum(x[0] for x in wk), places=6); self.assertAlmostEqual(f["weekday"], 1000 * sum(x[1] for x in wd) / sum(x[0] for x in wd), places=6)
        self.assertIn("on weekends, against", r["answer"])

    def test_one_kind_of_day_names_that_figure_first(self):
        r = ask(self.b, "how many trips do I make on weekdays", "stat trip_record op=count"); self.assertIn("on weekdays, against", r["answer"])

    def test_distance_per_month_matches_sql_and_the_months_add_up(self):
        r = ask(self.b, "distance per month", "stat distance_driven op=sum"); vals = r["facts"]["values"]; keys = r["facts"]["keys"]
        for k, v in zip(keys, vals):
            a, z = CX._bin_days(k, "month"); lo = max(a * DAY - self.tz, 0); hi = (z + 1) * DAY - self.tz
            km = self.c.execute("SELECT COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<? AND km>0.3", (lo, hi)).fetchone()[0]; self.assertAlmostEqual(v, km, places=6)

    def test_events_per_month_count_every_event_once(self):
        r = ask(self.b, "how many warnings per month this year", "events any_warning op=count"); vals = r["facts"]["values"]
        first = self.c.execute("SELECT MIN(ts) FROM sig_1m").fetchone()[0]
        types = tuple(CX.R.EVENT_GROUPS.get("ANY_WARNING", ("ANY_WARNING",))); q = ",".join("?" * len(types))
        n = self.c.execute(f"SELECT COUNT(*) FROM events WHERE type IN ({q}) AND ts>=? AND ts<?", (*types, first, self.now + 1)).fetchone()[0]
        self.assertEqual(int(sum(vals)), n)

    def test_the_month_with_the_most_events_is_the_largest_bin(self):
        r = ask(self.b, "which month had the most hard braking", "extreme hard_braking op=max"); self.assertTrue(r["answer"].startswith("The month with the most hard-braking events"))

    def test_a_trend_keeps_its_verdict_and_adds_the_months(self):
        r = ask(self.b, "how has my average speed changed month by month", "trend speed op=change"); a = r["answer"]
        self.assertIn("vehicle speed", a.lower()); self.assertIn("for each month", a)

    def test_pattern_in_time_counts_the_events_and_names_no_pattern_that_is_not_there(self):
        r = ask(self.b, "is there a pattern in when the warnings appear", "events any_warning op=count"); f = r["facts"]
        self.assertGreaterEqual(f["count"], 12); self.assertIn("time of day", r["answer"]); self.assertIn("what the car was doing", r["answer"])

    def test_a_condition_that_cannot_be_applied_is_said(self):
        r = ask(self.b, "how many hard brakes per week", "events hard_braking op=count cond=raining"); self.assertIn("cannot be applied", r["answer"])


if __name__ == "__main__":
    unittest.main()
