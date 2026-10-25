"""Independent reference check of the executor: the same questions answered with plain SQL / numpy straight from the stored tables, sharing no code with query_executor / query_planner (only the table encoding in the signal registry: which column holds
which signal and its integer scale). A disagreement is a bug in one of the two. Windows are computed here from `now` and the database's time-zone metadata, not by the planner's resolve_time.
Strictness: whenever the reference has data, the executor MUST answer (a refusal fails the test); event counts compare the executor's number with the reference including zero; rankings compare length and item identity (trip start) before values.
A missing fixture database is reported as a skip with the path, never silently passed."""
import os, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from vgpt import signal_registry as R
from vgpt.brain import Brain
from vgpt.query_ir import Query

DAY = 86400; NOW = 1790406000; DBF = os.path.join(ROOT, "data", "gen_courier_ext.db")


@unittest.skipUnless(os.path.exists(DBF), f"fixture database missing: {DBF}")
class Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qc = Brain(DBF, slm_ir=None, now=NOW).qc(); cls.c = sqlite3.connect(f"file:{DBF}?mode=ro", uri=True)
        cls.tz = int(float(dict(cls.c.execute("SELECT k, v FROM meta").fetchall())["env.tz_h"]) * 3600); cls.today0 = (NOW + cls.tz) // DAY * DAY - cls.tz

    def window(self, name):
        if name.startswith("last:"): n = int(name.split(":")[1]); return NOW - n * DAY, NOW + 1
        return {"yesterday": (self.today0 - DAY, self.today0), "today": (self.today0, NOW + 1)}[name]

    def column(self, sig, lo, hi):
        table, col, scale = R.STORE[sig]["1m"]
        rows = self.c.execute(f"SELECT {col} FROM {table} WHERE ts >= ? AND ts < ? AND {col} IS NOT NULL", (lo, hi)).fetchall()
        return np.array([r[0] for r in rows], float) / scale

    def run_q(self, text): return self.qc.execute(Query.from_text(text.replace("t=yesterday", "t=cal:yesterday").replace("t=today", "t=cal:today")))

    def test_statistics_of_measured_signals_match_plain_sql(self):
        stats = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median, "std": lambda x: np.std(x, ddof=1)}
        checked = 0; windows_without_data = []
        for sig in ("speed", "battery_temperature", "battery_soc", "coolant_flow", "motor_temperature"):
            for win in ("last:7:day", "last:14:day", "yesterday", "last:3:day"):
                lo, hi = self.window(win); x = self.column(sig, lo, hi)
                if len(x) < 30: windows_without_data.append((sig, win, len(x))); continue
                for agg, fn in stats.items():
                    op = "extreme" if agg in ("max", "min") else "aggregate"; r = self.run_q(f"op={op} tgt={sig} agg={agg} t={win}")
                    self.assertEqual(r.status, "answered", (sig, win, agg, "the reference has", len(x), "samples but the executor did not answer"))
                    ref = float(fn(x)); got = float(r.result.value); self.assertAlmostEqual(got, ref, delta=1e-6 + 1e-6 * abs(ref), msg=(sig, win, agg)); checked += 1
        self.assertGreater(checked, 40, f"too few reference cases ran; windows without data: {windows_without_data}")

    def test_event_counts_match_plain_sql_including_zero(self):
        zero = nonzero = 0
        for ev in ("HARSH_BRAKE", "HARSH_ACCEL", "CHARGE_START", "DTC_SET", "TPMS_LOW"):
            for win in ("last:14:day", "last:7:day", "yesterday"):
                lo, hi = self.window(win); ref = self.c.execute("SELECT COUNT(*) FROM events WHERE type = ? AND ts >= ? AND ts < ?", (ev, lo, hi)).fetchone()[0]
                r = self.run_q(f"op=count t={win} ev={ev}"); self.assertEqual(r.status, "answered", (ev, win, "a count, including a count of zero, must be answered"))
                self.assertEqual(int(r.result.value), ref, (ev, win)); zero += ref == 0; nonzero += ref > 0
        self.assertGreater(zero, 0); self.assertGreater(nonzero, 0)                                  # the comparison covered both empty and non-empty windows

    def test_trip_totals_and_rankings_match_the_ledger(self):
        lo, hi = self.window("last:14:day")
        ref = self.c.execute("SELECT SUM(kwh), SUM(km) FROM trips WHERE start >= ? AND start < ?", (lo, hi)).fetchone()
        r = self.run_q("op=aggregate tgt=energy_used agg=sum t=last:14:day"); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), ref[0], delta=0.01 + 0.002 * ref[0])
        r = self.run_q("op=aggregate tgt=distance_driven agg=sum t=last:14:day"); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), ref[1], delta=0.05 + 0.002 * ref[1])
        top = self.c.execute("SELECT start, km FROM trips WHERE start >= ? AND start < ? ORDER BY km DESC LIMIT 3", (lo, hi)).fetchall()
        r = self.run_q("op=rank t=last:14:day sel=trip:distance_driven:desc:3:-"); self.assertEqual(r.status, "answered"); items = r.result.items
        self.assertEqual(len(items), len(top), "the ranking must return as many trips as the ledger has up to k")
        if len({k for _, k in top}) == len(top): self.assertEqual([i["start"] for i in items], [s for s, _ in top], "the same trips in the same order")
        for (s, k), i in zip(top, items): self.assertAlmostEqual(k, i["value"], delta=0.05)

    def test_consumption_of_a_period_is_total_energy_over_total_distance(self):
        lo, hi = self.window("last:14:day"); kwh, km = self.c.execute("SELECT SUM(kwh), SUM(km) FROM trips WHERE start >= ? AND start < ?", (lo, hi)).fetchone()
        r = self.run_q("op=aggregate tgt=wh_per_km agg=mean t=last:14:day"); self.assertEqual(r.status, "answered"); self.assertAlmostEqual(float(r.result.value), 1000 * kwh / km, delta=0.5)


if __name__ == "__main__": unittest.main()
