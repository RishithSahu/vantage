"""Retention: when stored detail is removed (the 1 Hz tier ages out, a day of readings is lost, everything in the window is gone) the answer must narrow its claim, state the limit, or decline - never keep the finer wording and never invent a number.
Each test starts from a copy of a real generated database and deletes rows, so the 'before' and 'after' answers come from the same history."""
import os, shutil, sqlite3, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.qc import QueryCompiler
from vgpt.query_ir import Query

SRC = os.path.join(ROOT, "data", "gen_courier_ext.db"); END = 1790341893 + 300; DAY = 86400; NOW = 1790406000


@unittest.skipUnless(os.path.exists(SRC), "vehicle database not present")
class Retention(unittest.TestCase):
    def setUp(self): self.dir = tempfile.mkdtemp(); self.db = os.path.join(self.dir, "x.db"); shutil.copy(SRC, self.db)
    def tearDown(self): shutil.rmtree(self.dir, ignore_errors=True)

    def run_q(self, text, now=END):
        c = sqlite3.connect(self.db)
        try: return QueryCompiler(c, now=now).execute(Query.from_text(text))
        finally: c.close()

    def delete(self, sql, *args):
        c = sqlite3.connect(self.db); c.execute(sql, args); c.commit(); c.close()
        # rows removed from the MIDDLE of the history by another process are not noticed by the columnar copy (it re-checks 14 spread-out rows, not every day): start it afresh, as a restart of the assistant would
        from vgpt import colstore
        colstore._STORES.pop(self.db, None); shutil.rmtree(self.db + ".cols", ignore_errors=True)

    def test_losing_the_one_second_tier_narrows_the_claim_about_peaks(self):
        q = "op=extreme tgt=speed agg=max t=last:1:hour"; before = self.run_q(q); self.assertEqual(before.result.tier, "1 Hz"); self.assertNotIn("one-minute averages", before.answer.text)
        self.delete("DELETE FROM sig_1s"); after = self.run_q(q)
        self.assertEqual(after.status, "answered"); self.assertEqual(after.result.tier, "per-minute")
        self.assertIn("one-minute averages", after.answer.text); self.assertIn("brief peak", after.answer.text)
        self.assertLessEqual(after.result.value, before.result.value + 1e-9)               # a maximum of averages cannot exceed the maximum of the samples

    def test_a_driven_day_without_readings_is_reported_not_averaged_over_silently(self):
        q = "op=aggregate tgt=battery_temperature agg=mean t=last:14:day"; before = self.run_q(q, NOW); self.assertEqual(before.status, "answered"); self.assertFalse(before.result.extra["contract"]["signal_coverage"])
        c = sqlite3.connect(self.db); tz = int(float(dict(c.execute("SELECT k, v FROM meta").fetchall())["env.tz_h"]) * 3600); start = c.execute("SELECT start FROM trips WHERE start >= ? ORDER BY start LIMIT 1 OFFSET 5", (NOW - 14 * DAY,)).fetchone()[0]; c.close()
        day0 = (start + tz) // DAY * DAY - tz; self.delete("DELETE FROM sig_1m WHERE ts >= ? AND ts < ?", day0, day0 + DAY)
        after = self.run_q(q, NOW); self.assertEqual(after.status, "answered"); ct = after.result.extra["contract"]
        self.assertEqual(ct["completeness"], "partial"); self.assertTrue(ct["signal_coverage"], ct); self.assertRegex(after.answer.text, r"readings covering \d+ of \d+ active days")

    def test_a_window_whose_samples_have_aged_out_is_declined_with_a_reason(self):
        now = NOW - 2 * DAY; q = "op=aggregate tgt=battery_temperature agg=mean t=last:3:day"; self.assertEqual(self.run_q(q, now).status, "answered")
        for t in ("sig_1m", "sig_1s"): self.delete(f"DELETE FROM {t} WHERE ts >= ? AND ts < ?", now - 3 * DAY, now + 1)            # earlier history stays, so the signal is still a recorded one
        after = self.run_q(q, now); self.assertIn(after.status, ("no_data", "missing_data")); self.assertNotRegex(after.answer.text, r"\d+(\.\d+)?\s*\S?C\b"); self.assertIn("no recorded data", after.answer.text.lower())

    def test_a_trip_ledger_total_does_not_change_when_recent_raw_samples_age_out(self):
        q = "op=aggregate tgt=distance_driven agg=sum t=last:7:day"; before = self.run_q(q, NOW)
        for t in ("sig_1m", "sig_1s"): self.delete(f"DELETE FROM {t} WHERE ts >= ?", NOW - 7 * DAY)
        after = self.run_q(q, NOW); self.assertEqual(after.status, "answered"); self.assertAlmostEqual(before.result.value, after.result.value, delta=1e-6)               # totals come from the trip ledger, which outlives the raw rows

    # ---------------------------------------------------------------------------------------------------------- coverage semantics
    def _tz(self):
        c = sqlite3.connect(self.db); tz = int(float(dict(c.execute("SELECT k, v FROM meta").fetchall())["env.tz_h"]) * 3600); c.close(); return tz

    def test_one_reading_on_a_day_is_not_complete_coverage_of_that_day(self):
        tz = self._tz(); q = "op=aggregate tgt=battery_temperature agg=mean t=last:14:day"
        c = sqlite3.connect(self.db); c.execute("DELETE FROM sig_1m WHERE batt IS NOT NULL AND ts NOT IN (SELECT MIN(ts) FROM sig_1m GROUP BY (ts + ?) / 86400)", (tz,)); c.commit(); c.close()
        r = self.run_q(q, NOW); ct = r.result.extra["contract"]; d = ct["signal_days"]["battery_temperature"]; self.assertEqual(d["covered"], 0); self.assertGreater(d["partial"], 0); self.assertEqual(ct["completeness"], "partial")

    def test_coverage_is_unknown_not_complete_when_no_ledger_exists(self):
        c = sqlite3.connect(self.db); c.execute("DROP TABLE trips"); c.execute("DROP TABLE charges"); c.commit(); c.close()
        r = self.run_q("op=aggregate tgt=battery_temperature agg=mean t=last:14:day", NOW)
        if r.status == "answered": ct = r.result.extra["contract"]; self.assertIn("battery_temperature", ct["signal_coverage"]); self.assertIsNone(ct["signal_coverage"]["battery_temperature"]); self.assertRegex(r.answer.text, r"could not be checked")

    def test_an_empty_ledger_makes_no_day_level_claim(self):
        c = sqlite3.connect(self.db); c.execute("DELETE FROM trips"); c.execute("DELETE FROM charges"); c.commit(); c.close()
        r = self.run_q("op=aggregate tgt=battery_temperature agg=mean t=last:14:day", NOW)
        if r.status == "answered": self.assertEqual(r.result.extra["contract"].get("signal_days", {}), {})

    def test_a_derived_quantity_follows_its_dependencies(self):
        tz = self._tz(); c = sqlite3.connect(self.db); day = c.execute("SELECT start FROM trips WHERE start >= ? ORDER BY start LIMIT 1 OFFSET 4", (NOW - 14 * DAY,)).fetchone()[0]; d0 = (day + tz) // DAY * DAY - tz
        c.execute("UPDATE sig_1m SET power = NULL WHERE ts >= ? AND ts < ?", (d0, d0 + DAY)); c.commit(); c.close()
        r = self.run_q("op=aggregate tgt=wh_per_km agg=mean t=last:14:day", NOW)
        if r.status == "answered": self.assertIn("battery_power", r.result.extra["contract"]["signal_days"]); self.assertLess(r.result.extra["contract"]["signal_days"]["battery_power"]["covered"], r.result.extra["contract"]["signal_days"]["battery_power"]["expected"])

    def test_both_periods_of_a_comparison_are_inspected(self):
        tz = self._tz(); today0 = (NOW + tz) // DAY * DAY - tz; c = sqlite3.connect(self.db); c.execute("DELETE FROM sig_1m WHERE ts >= ? AND ts < ?", (today0 - 2 * DAY, today0 - DAY)); c.commit(); c.close()
        r = self.run_q("op=compare tgt=battery_temperature agg=mean t=cal:yesterday t2=cal:day_before_yesterday", NOW)
        if r.status == "answered": self.assertTrue(any("second period" in k for k in r.result.extra["contract"]["signal_coverage"]), r.result.extra["contract"])


if __name__ == "__main__": unittest.main()
