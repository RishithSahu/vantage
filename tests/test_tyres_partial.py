"""One wheel missing from the newest minute must not make the answer list three tyres: each wheel's own latest reading of the last 15 minutes is used, and a wheel that really has none is said to be not recorded."""
import os
import sqlite3
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt import analytics_x as AX      # noqa: E402

DB = os.path.join(ROOT, "data", "world_aster.db")


@unittest.skipUnless(os.path.exists(DB), "needs data/world_aster.db")
class Partial(unittest.TestCase):
    def memory_copy(self):
        src = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True); m = sqlite3.connect(":memory:")
        for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name IN ('meta','ext_1m','events','agg_1d') AND sql IS NOT NULL"): m.execute(sql)
        m.executemany("INSERT INTO meta VALUES (?,?)", src.execute("SELECT k, v FROM meta").fetchall())
        cols = [r[1] for r in src.execute("PRAGMA table_info(ext_1m)")]; rows = src.execute("SELECT * FROM ext_1m ORDER BY ts DESC LIMIT 30").fetchall()
        m.executemany("INSERT INTO ext_1m VALUES (%s)" % ",".join("?" * len(cols)), rows); m.executemany("INSERT INTO events VALUES (?,?,?,?)", src.execute("SELECT * FROM events ORDER BY ts DESC LIMIT 200").fetchall())
        return m, rows[0][cols.index("ts")]

    def test_a_wheel_missing_from_the_newest_minute_is_read_from_the_minutes_before(self):
        m, now = self.memory_copy(); full = AX.tyres(m, now); self.assertEqual(len(full["wheels"]), 4)
        m.execute("UPDATE ext_1m SET tp_fl=NULL WHERE ts=?", (now,)); part = AX.tyres(m, now)
        self.assertEqual([w["name"] for w in part["wheels"]], [w["name"] for w in full["wheels"]]); self.assertEqual(part.get("not_recorded", []), [])

    def test_a_wheel_with_no_reading_for_a_long_time_is_said_to_be_not_recorded(self):
        m, now = self.memory_copy(); m.execute("UPDATE ext_1m SET tp_rr=NULL"); f = AX.tyres(m, now)
        self.assertEqual(len(f["wheels"]), 3); self.assertIn("rear right", f.get("not_recorded", []))


if __name__ == "__main__": unittest.main()
