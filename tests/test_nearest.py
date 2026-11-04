"""vgpt/nearest.py must pick the very row that `ORDER BY ABS(ts - ?) LIMIT 1` picks: before the first row, after the last, exactly between two (the earlier wins), with NULL readings skipped, on a small generated table and on the real four-year database."""
import os, random, sqlite3, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.nearest import nearest

REAL = os.path.join(ROOT, "data", "aster_ev_4yr.db")


def reference(c, table, what, t, notnull):
    w = f" WHERE {notnull} IS NOT NULL" if notnull else ""
    return c.execute(f"SELECT {what} FROM {table}{w} ORDER BY ABS(ts - ?) LIMIT 1", (t,)).fetchone()


class Nearest(unittest.TestCase):
    def test_generated_table(self):
        c = sqlite3.connect(":memory:"); c.execute("CREATE TABLE s(ts INTEGER PRIMARY KEY, v INT) WITHOUT ROWID"); rng = random.Random(3)
        c.executemany("INSERT INTO s VALUES (?,?)", [(60 * i + (7 if i % 11 == 0 else 0), None if rng.random() < .4 else rng.randint(0, 99)) for i in range(400)])
        for t in [-500, 0, 1, 30, 31, 59, 60, 90, 91, 23000, 23999, 24500] + [rng.randint(-100, 24200) for _ in range(300)]:
            for nn in (None, "v"): self.assertEqual(nearest(c, "s", "ts, v", t, nn), reference(c, "s", "ts, v", t, nn), (t, nn))
        self.assertIsNone(nearest(sqlite3.connect(":memory:").execute("CREATE TABLE s(ts INTEGER PRIMARY KEY, v INT)").connection, "s", "ts, v", 5, "v"))

    @unittest.skipUnless(os.path.exists(REAL), "the four-year database is not here")
    def test_real_history(self):
        c = sqlite3.connect(f"file:{REAL}?mode=ro", uri=True); rng = random.Random(5)
        for table, what, nn, n in (("sig_1m", "ts, speed", "speed", 12), ("sig_1m", "ts, motor / 10.0", "motor", 6), ("ext_1m", "ts, tp_fl, tp_fr, tp_rl, tp_rr", None, 12), ("sig_1s", "ts, speed", "speed", 12), ("gps_1s", "ts, lat, lon", None, 12)):
            try: lo, hi = c.execute(f"SELECT MIN(ts), MAX(ts) FROM {table}").fetchone()
            except sqlite3.OperationalError: continue
            if lo is None: continue
            for t in [lo - 1000, hi + 1000, lo, hi] + [rng.randint(lo, hi) for _ in range(n)] + [(lo + hi) // 2 // 60 * 60 + 30]:
                with self.subTest(table=table, t=t): self.assertEqual(nearest(c, table, what, t, nn), reference(c, table, what, t, nn))


if __name__ == "__main__":
    unittest.main()
