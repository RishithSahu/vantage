"""The lossless archive of expired seconds: written before the database deletes them, loaded for one question, never altering the database, and checked by hash when it is read."""
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import archive, qa_core as QC, store  # noqa: E402

DAY = 86400
T0 = 1_790_000_000 // DAY * DAY            # a UTC midnight


def make_db(days=10, fine_days=3, seed=5):
    """A vehicle database with `days` days of drives (1.5 hours a day at 1 Hz) and the retention of a vehicle that keeps `fine_days` days of seconds.  -> (path, the rows it was given)."""
    rnd = random.Random(seed); d = tempfile.mkdtemp(prefix="arch_"); path = os.path.join(d, "vantage.db"); c = store.open_rw(path); rows = []
    for k in range(days):
        base = T0 + k * DAY + 8 * 3600; sp = 0.0
        for i in range(5400):
            sp = max(0.0, min(1300.0, sp + rnd.uniform(-8, 8))); ts = base + i
            rows.append((ts, int(sp), 900 - k * 3 - i // 600, int(sp * 9) - 300, 250 + i // 100, 120, 180, 400, 500 + int(sp) // 10, 1 if sp > 0 else None))
    c.executemany("INSERT INTO sig_1s VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    c.execute("INSERT OR REPLACE INTO meta VALUES('ret.fine_days',?)", (str(fine_days),)); c.execute("INSERT OR REPLACE INTO meta VALUES('ret.archive_days','365')")
    store.rollup(c, T0, T0 + days * DAY); c.commit(); return path, rows, c


class Codec(unittest.TestCase):
    def test_every_value_comes_back_exactly_nulls_negatives_and_floats_included(self):
        rows = [(1, -5, None, 2 ** 40), (2, None, 7, -2 ** 40), (3, 9, 7, 0), (10, -9, None, 1)]
        self.assertEqual(archive.decode(archive.encode(rows, "iiii")), rows)
        fl = [(1, 2, 0.1), (1, 3, None), (2, 3, 1e-300), (2, 4, float("1e300"))]
        self.assertEqual(archive.decode(archive.encode(fl, "iif")), fl)
        self.assertEqual(archive.decode(archive.encode([], "iii")), [])

    def test_a_day_of_one_second_rows_is_small(self):
        rnd = random.Random(1); sp = 0.0; rows = []
        for i in range(86400):
            sp = max(0.0, min(1300.0, sp + rnd.uniform(-8, 8))); rows.append((T0 + i, int(sp), 900 - i // 600, int(sp * 9) - 300, 250 + i // 100, 120, 180, 400, 500 + int(sp) // 10, 1))
        blob = archive.encode(rows, "iiiiiiiiii"); raw = 86400 * 10 * 8
        self.assertLess(len(blob), 0.2 * raw, f"{len(blob)} bytes for {raw} raw bytes"); self.assertEqual(archive.decode(blob), rows)


class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path, cls.rows, cls.c = make_db(); cls.now = T0 + 10 * DAY; cls.before = len(cls.rows)
        store.compact(cls.c, cls.now); cls.c.close(); cls.live_lo = None

    @classmethod
    def tearDownClass(cls): shutil.rmtree(os.path.dirname(cls.path), True)

    def test_the_expired_seconds_left_the_database_and_are_in_the_archive(self):
        c = sqlite3.connect(self.path); n_live = c.execute("SELECT COUNT(*) FROM sig_1s").fetchone()[0]; idx = c.execute("SELECT SUM(rows), MIN(lo), MAX(hi) FROM archive_idx WHERE tbl='sig_1s'").fetchone(); c.close()
        self.assertEqual(n_live + idx[0], self.before, "every second is in exactly one place"); self.assertLess(n_live, self.before)
        d = os.path.join(os.path.dirname(self.path), "archive"); self.assertTrue(os.listdir(d)); self.assertEqual(idx[1], T0 + 8 * 3600)

    def test_a_question_over_expired_seconds_gets_every_one_of_them_exactly(self):
        c = store.open_ro(self.path); lo, hi = T0 + 2 * DAY, T0 + 4 * DAY; self.assertIsNone(c.execute("SELECT MIN(ts) FROM sig_1s WHERE ts<?", (T0 + 5 * DAY,)).fetchone()[0], "the live table no longer has them")
        r = archive.hydrate(c, lo, hi); want = [x for x in self.rows if lo <= x[0] < hi]
        self.assertEqual(r["rows"], len(want)); self.assertEqual(r["damaged"], [])
        got = c.execute("SELECT ts, speed, soc, power, batt, flow, amb, hvac, motor, mode FROM sig_1s WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall()
        self.assertEqual(got, want, "lossless"); self.assertEqual(archive.hydrate(c, lo, hi)["rows"], len(want), "asking again loads nothing twice")
        self.assertEqual(c.execute("SELECT COUNT(*) FROM temp.arch_sig_1s").fetchone()[0], len(want)); c.close()

    def test_the_live_rows_and_the_archive_read_as_one_table(self):
        c = store.open_ro(self.path); live_lo = c.execute("SELECT MIN(ts) FROM main.sig_1s").fetchone()[0]; lo, hi = live_lo - DAY, live_lo + 2 * 3600
        archive.hydrate(c, lo, hi); got = c.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM sig_1s WHERE ts>=? AND ts<?", (lo, hi)).fetchone(); want = [x for x in self.rows if lo <= x[0] < hi]
        self.assertEqual(got, (len(want), want[0][0], want[-1][0])); c.close()

    def test_the_assistants_handle_stays_read_only(self):
        c = store.open_ro(self.path); archive.hydrate(c, T0 + 8 * 3600, T0 + DAY)
        for sql in ("INSERT INTO main.sig_1s VALUES(1,1,1,1,1,1,1,1,1,1)", "DELETE FROM main.meta", "CREATE TABLE x(a)", "ATTACH DATABASE ':memory:' AS z"):
            with self.assertRaises(sqlite3.Error, msg=sql): c.execute(sql)
        c.close()

    def test_the_analytics_load_reads_the_archive_for_an_exact_window(self):
        c = store.open_ro(self.path); lo = T0 + 8 * 3600 + 600; hi = lo + 3 * 3600
        sig = QC.load(c, lo, hi, prefer="1s"); want = [x for x in self.rows if lo <= x[0] < hi]
        self.assertEqual(sig.tier, "1 Hz"); self.assertEqual(len(sig.ts), len(want)); self.assertEqual(max(sig.cols["speed"]), max(x[1] for x in want) / 10.0); c.close()

    def test_the_archive_is_found_by_the_planner_and_the_coverage(self):
        c = store.open_ro(self.path); lo, hi = archive.coverage(c); self.assertEqual(lo, T0 + 8 * 3600); self.assertGreater(hi, lo)
        from vgpt import query_executor as QE
        ctx = QE.context_for(c, self.now); self.assertEqual(ctx.arch_lo, lo); c.close()

    def test_compacting_twice_archives_nothing_twice(self):
        c = store.open_rw(self.path); n0 = c.execute("SELECT COUNT(*), SUM(rows) FROM archive_idx").fetchone(); store.compact(c, self.now); n1 = c.execute("SELECT COUNT(*), SUM(rows) FROM archive_idx").fetchone(); c.close()
        self.assertEqual(n0, n1)

    def test_the_chain_and_the_hashes_hold(self):
        c = sqlite3.connect(self.path); ok, bad = archive.verify(c); c.close(); self.assertTrue(ok, bad)


class Tampering(unittest.TestCase):
    def test_a_changed_file_is_refused_and_the_question_is_told_the_seconds_are_not_there(self):
        path, rows, c = make_db(days=6, fine_days=2); store.compact(c, T0 + 6 * DAY); c.close()
        d = os.path.join(os.path.dirname(path), "archive"); victim = sorted(os.listdir(d))[0]; p = os.path.join(d, victim); b = bytearray(open(p, "rb").read()); b[len(b) // 2] ^= 0xFF; open(p, "wb").write(bytes(b))
        r = store.open_ro(path); out = archive.hydrate(r, T0, T0 + 6 * DAY); self.assertIn(victim, out["damaged"]); r.close()
        c = sqlite3.connect(path); ok, bad = archive.verify(c); c.close(); self.assertFalse(ok); self.assertTrue(any("changed" in x for x in bad)); shutil.rmtree(os.path.dirname(path), True)

    def test_a_removed_file_is_noticed(self):
        path, rows, c = make_db(days=6, fine_days=2); store.compact(c, T0 + 6 * DAY); c.close()
        d = os.path.join(os.path.dirname(path), "archive"); os.remove(os.path.join(d, sorted(os.listdir(d))[0])); c = sqlite3.connect(path); ok, bad = archive.verify(c); c.close()
        self.assertFalse(ok); self.assertTrue(any("missing" in x for x in bad)); shutil.rmtree(os.path.dirname(path), True)


class Retention(unittest.TestCase):
    def test_files_older_than_the_archive_retention_are_removed_with_their_index(self):
        path, rows, c = make_db(days=6, fine_days=2); c.execute("INSERT OR REPLACE INTO meta VALUES('ret.archive_days','3')"); store.compact(c, T0 + 6 * DAY)
        left = c.execute("SELECT MIN(lo) FROM archive_idx").fetchone()[0]; c.close(); d = os.path.join(os.path.dirname(path), "archive")
        self.assertTrue(left is None or left >= T0 + 6 * DAY - 3 * DAY, "nothing older than three days is kept"); self.assertEqual(len(os.listdir(d)), sqlite3.connect(path).execute("SELECT COUNT(*) FROM archive_idx").fetchone()[0]); shutil.rmtree(os.path.dirname(path), True)

    def test_the_lake_seconds_are_archived_and_read_back_too(self):
        path, rows, c = make_db(days=5, fine_days=2); c.execute("CREATE TABLE IF NOT EXISTS lk_1s(sid INTEGER, ts INTEGER, v REAL, PRIMARY KEY(sid, ts)) WITHOUT ROWID")
        lk = [(s, T0 + 8 * 3600 + i, math.sin(i / 50.0) * 100 + s) for s in (1, 2, 3) for i in range(2 * 3600)]; c.executemany("INSERT INTO lk_1s VALUES(?,?,?)", lk)
        c.execute("CREATE TABLE IF NOT EXISTS lk_1m(sid INTEGER, ts INTEGER, v REAL, vmin REAL, vmax REAL, PRIMARY KEY(sid, ts)) WITHOUT ROWID")
        store.compact(c, T0 + 5 * DAY); c.close(); r = store.open_ro(path); lo, hi = T0 + 8 * 3600, T0 + 8 * 3600 + 7200
        self.assertEqual(r.execute("SELECT COUNT(*) FROM main.lk_1s").fetchone()[0], 0); archive.hydrate(r, lo, hi)
        got = r.execute("SELECT sid, ts, v FROM lk_1s WHERE ts>=? AND ts<? ORDER BY sid, ts", (lo, hi)).fetchall(); self.assertEqual(got, sorted(lk)); r.close(); shutil.rmtree(os.path.dirname(path), True)


if __name__ == "__main__":
    unittest.main()
