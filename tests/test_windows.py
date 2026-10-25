"""Time-window correctness against plain SQL: the requested interval is enforced exactly (partial first and last days, a sample exactly on a boundary, trips that cross midnight, two different scopes in one comparison).
The reference shares no code with the planner/executor: windows are computed here from `now` and the database's time-zone metadata."""
import os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from vgpt import signal_registry as R
from vgpt.brain import Brain
from vgpt.query_ir import Query, TimeSpec

DAY = 86400; NOW = 1790406000; DBF = os.path.join(ROOT, "data", "gen_courier_ext.db")


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class Windows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = Brain(DBF, slm_ir=None, now=NOW); cls.qc = cls.brain.qc(); cls.c = sqlite3.connect(f"file:{DBF}?mode=ro", uri=True)
        cls.tz = int(float(dict(cls.c.execute("SELECT k, v FROM meta").fetchall())["env.tz_h"]) * 3600); cls.today0 = (NOW + cls.tz) // DAY * DAY - cls.tz

    def col(self, sig, lo, hi):
        t, col, sc = R.STORE[sig]["1m"][:3]
        return np.array([r[0] for r in self.c.execute(f"SELECT {col} FROM {t} WHERE ts >= ? AND ts < ? AND {col} IS NOT NULL", (lo, hi))], float) / sc

    def ex(self): return self.qc.ex if hasattr(self.qc, "ex") else self.qc.executor

    def test_the_series_covers_exactly_the_interval_including_partial_days(self):
        from vgpt import change_ops
        q = Query.from_text("op=detect_change tgt=battery_temperature t=all"); lo = self.today0 - 3 * DAY + 11 * 3600 + 17 * 60; hi = self.today0 - 1 * DAY + 5 * 3600 + 41 * 60               # starts mid-morning, ends early morning
        ser, err = change_ops._series(self.ex(), q, "battery_temperature", lo, hi); self.assertTrue(ser, err)
        total = sum(n for _, _, n in ser); ref = self.col("battery_temperature", lo, hi); self.assertEqual(total, len(ref), "every in-interval sample, and only those, is aggregated")
        first_day_start = self.today0 - 3 * DAY; first = [s for s in ser if s[0] == first_day_start]
        if first:
            ref_first = self.col("battery_temperature", lo, first_day_start + DAY); self.assertEqual(first[0][2], len(ref_first)); self.assertAlmostEqual(first[0][1], float(ref_first.mean()), delta=1e-6)
        last_day_start = self.today0 - 1 * DAY; last = [s for s in ser if s[0] == last_day_start]
        if last: self.assertEqual(last[0][2], len(self.col("battery_temperature", last_day_start, hi)))

    def test_a_sample_exactly_on_the_end_boundary_is_excluded_and_on_the_start_boundary_included(self):
        from vgpt import change_ops
        q = Query.from_text("op=detect_change tgt=battery_temperature t=all"); ts = [r[0] for r in self.c.execute("SELECT ts FROM sig_1m WHERE ts >= ? ORDER BY ts LIMIT 400", (self.today0 - 5 * DAY,))]
        lo, hi = ts[10], ts[200]; ser, _ = change_ops._series(self.ex(), q, "battery_temperature", lo, hi); self.assertEqual(sum(n for _, _, n in ser), len(self.col("battery_temperature", lo, hi)))
        self.assertGreater(len(self.col("battery_temperature", lo, hi + 1)), len(self.col("battery_temperature", lo, hi)), "the sample at `hi` exists in the data, so the test is meaningful")

    def test_yesterday_is_the_local_calendar_day_midnight_to_midnight(self):
        r = self.qc.execute(Query.from_text("op=aggregate tgt=battery_temperature agg=mean t=cal:yesterday")); self.assertEqual(r.status, "answered")
        ref = self.col("battery_temperature", self.today0 - DAY, self.today0); self.assertAlmostEqual(float(r.result.value), float(ref.mean()), delta=1e-6)
        self.assertEqual((int(r.result.period[0]), int(r.result.period[1])), (self.today0 - DAY, self.today0))

    def test_a_comparison_uses_its_own_two_scopes(self):
        r = self.qc.execute(Query.from_text("op=compare tgt=battery_temperature agg=mean t=cal:yesterday t2=cal:day_before_yesterday")); self.assertEqual(r.status, "answered")
        a = self.col("battery_temperature", self.today0 - DAY, self.today0); b = self.col("battery_temperature", self.today0 - 2 * DAY, self.today0 - DAY)
        got = {i.get("group", i.get("label")): i["value"] for i in r.result.items}; vals = list(got.values())
        self.assertTrue(len(a) and len(b)); self.assertAlmostEqual(vals[0], float(a.mean()), delta=1e-6); self.assertAlmostEqual(vals[1], float(b.mean()), delta=1e-6)


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class MidnightCrossing(unittest.TestCase):
    """A trip is inserted into a COPY of the history so that it starts 20 minutes before local midnight and ends 40 minutes after it (the generated histories contain no such trip). Distance is owned by the day the trip STARTS."""
    @classmethod
    def setUpClass(cls):
        import datetime as dt, tempfile
        cls.tmpdir = tempfile.mkdtemp(); cls.path = os.path.join(cls.tmpdir, "crossing.db")
        src = sqlite3.connect(f"file:{DBF}?mode=ro", uri=True); dst = sqlite3.connect(cls.path); src.backup(dst); src.close()
        cls.tz = int(float(dict(dst.execute("SELECT k, v FROM meta").fetchall())["env.tz_h"]) * 3600); today0 = (NOW + cls.tz) // DAY * DAY - cls.tz
        cls.d0 = today0 - 5 * DAY; cls.start = cls.d0 + DAY - 1200; cls.end = cls.start + 3600
        cls.day = lambda t: dt.datetime.fromtimestamp(t + cls.tz, dt.timezone.utc).strftime("%Y-%m-%d")
        cls.before = (cls.dist(cls.path, cls.day(cls.d0)), cls.dist(cls.path, cls.day(cls.d0 + DAY)))
        dst.execute("INSERT INTO trips(start, end, km, kwh, wh_km, avg_speed, hwy_share, avg_amb, avg_hvac_w, soc_start, soc_end) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (cls.start, cls.end, 25.0, 4.0, 160.0, 25.0, 0.2, 15.0, 300.0, 80.0, 75.0)); dst.commit(); dst.close()
        cls.after = (cls.dist(cls.path, cls.day(cls.d0)), cls.dist(cls.path, cls.day(cls.d0 + DAY)))

    @classmethod
    def dist(cls, db, d):
        r = Brain(db, slm_ir=None, now=NOW).qc().execute(Query.from_text(f"op=aggregate tgt=distance_driven agg=sum t=date:{d}")); assert r.status == "answered", (d, r.status); return float(r.result.value)

    def test_the_fixture_trip_really_crosses_local_midnight(self):
        self.assertNotEqual((self.start + self.tz) // DAY, (self.end + self.tz) // DAY)

    def test_the_trip_is_counted_on_the_day_it_starts(self):
        self.assertAlmostEqual(self.after[0] - self.before[0], 25.0, delta=1e-6)

    def test_the_following_day_does_not_count_any_of_it(self):
        self.assertAlmostEqual(self.after[1], self.before[1], delta=1e-6)

    def test_the_engine_agrees_with_plain_sql_on_the_start_day(self):
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True); ref = c.execute("SELECT SUM(km) FROM trips WHERE start >= ? AND start < ?", (self.d0, self.d0 + DAY)).fetchone()[0]; c.close()
        self.assertAlmostEqual(self.after[0], ref, delta=1e-6)


if __name__ == "__main__": unittest.main()
