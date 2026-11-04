"""The columnar copy of a per-minute table (vgpt/colstore.py) must give exactly what the database gives, while the history is still being written one minute at a time:
built early and read later, extended as rows arrive, rebuilt when the database is replaced, NULL read as NaN, and ignored when it cannot be trusted."""
import os, sqlite3, sys, tempfile, time, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import colstore
colstore.REFRESH_MIN_S = 0.0                                                                     # tests change the data between calls and expect the very next call to see it

DAY = 86400; T0 = 1_700_000_000 - 1_700_000_000 % 60


def make(path, t_from, t_to, seed=1, nulls=True):
    c = sqlite3.connect(path); c.execute("PRAGMA journal_mode=WAL"); c.execute("CREATE TABLE IF NOT EXISTS t_1m(ts INTEGER PRIMARY KEY, a INT, b INT) WITHOUT ROWID")
    rng = np.random.default_rng(seed); ts = np.arange(t_from, t_to, 60)
    c.executemany("INSERT INTO t_1m VALUES (?,?,?)", [(int(t), int(rng.integers(0, 900)), None if nulls and i % 97 == 0 else int(rng.integers(-50, 50))) for i, t in enumerate(ts)]); c.commit(); return c


def sql(c, lo, hi): return np.array(c.execute("SELECT ts, a, b FROM t_1m WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall(), dtype=float).reshape(-1, 3)


def wait_idle():
    for _ in range(300):
        with colstore._LOCK:
            if not colstore._BUILDING: return
        time.sleep(0.02)


class Copy(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.db = os.path.join(self.d, "w.db"); colstore._STORES.clear(); colstore._BUILDING.clear()

    def same(self, c, cs, lo, hi):
        got = cs.read(c, "t_1m", ["a", "b"], lo, hi); ref = sql(c, lo, hi)
        self.assertIsNotNone(got); self.assertEqual(got.shape, ref.shape); np.testing.assert_array_equal(got, ref)      # NaN equals NaN here

    def test_a_copy_built_early_still_matches_while_rows_keep_arriving(self):
        c = make(self.db, T0, T0 + 10 * DAY); cs = colstore.ColStore(self.db)
        os.environ["VGPT_COLSTORE_SYNC"] = "0"
        try: self.assertIsNone(cs.read(c, "t_1m", ["a", "b"], T0, T0 + 10 * DAY)); wait_idle()                                # not built yet and building on demand switched off: the database answers (and the copy is built in the background)
        finally: os.environ.pop("VGPT_COLSTORE_SYNC", None)
        self.same(c, cs, T0, T0 + 10 * DAY)                                                                                 # on demand (the default): the copy is made in one pass and this very read uses it
        cs.build("t_1m", ["a", "b"]); self.same(c, cs, T0, T0 + 10 * DAY); self.same(c, cs, T0 + 2 * DAY, T0 + 9 * DAY)
        nxt = T0 + 10 * DAY
        for step in range(5):                                                                                                  # the live writer: a few minutes, then hours, then a day, between reads
            n = (3, 120, 600, 1500, 3000)[step]; make(self.db, nxt, nxt + n * 60, seed=step + 5); nxt += n * 60
            self.same(c, cs, T0, nxt + 60); self.same(c, cs, nxt - 4 * DAY, nxt + 60)
        wait_idle(); self.same(c, cs, T0, nxt + 60)                                                                        # extended in the background, still the same
        self.assertGreater(cs._load_meta("t_1m")["n"], 10 * DAY // 60)

    def test_extension_makes_the_next_read_need_no_database_tail(self):
        c = make(self.db, T0, T0 + 5 * DAY); cs = colstore.ColStore(self.db); cs.build("t_1m", ["a", "b"])
        make(self.db, T0 + 5 * DAY, T0 + 5 * DAY + 3000 * 60, seed=9)
        self.same(c, cs, T0, T0 + 6 * DAY); wait_idle(); m = cs._load_meta("t_1m"); self.assertEqual(m["n"], 5 * DAY // 60 + 3000)
        self.same(c, cs, T0, T0 + 7 * DAY)

    def test_a_replaced_database_is_not_read_through_the_old_copy(self):
        c = make(self.db, T0, T0 + 6 * DAY, seed=1); cs = colstore.ColStore(self.db); cs.build("t_1m", ["a", "b"]); self.same(c, cs, T0, T0 + 6 * DAY)
        c.execute("DELETE FROM t_1m"); c.commit(); make(self.db, T0, T0 + 6 * DAY, seed=2)
        self.same(c, cs, T0, T0 + 6 * DAY)                                                                                 # the copy is of the other data: it is not read through; it is made again from this one first
        wait_idle(); self.same(c, cs, T0, T0 + 6 * DAY)

    def test_rewritten_newest_rows_and_dropped_oldest_rows_do_not_force_a_rebuild(self):
        """The live store rewrites a minute row when the minute is finished and drops the oldest rows (retention); neither may make the copy rebuild (27 s on the Pi) or give a stale number."""
        c = make(self.db, T0, T0 + 8 * DAY); cs = colstore.ColStore(self.db); cs.build("t_1m", ["a", "b"]); builds = []; real = cs._build
        cs._build = lambda tb, cols: (builds.append(tb), real(tb, cols))[1]
        self.same(c, cs, T0, T0 + 8 * DAY)
        last = T0 + 8 * DAY - 60
        for k in range(4):                                                                                              # the newest minutes get new values, and a new minute arrives
            for j in range(3): c.execute("UPDATE t_1m SET a=a+7 WHERE ts=?", (last - 60 * j,))
            c.execute("INSERT INTO t_1m VALUES (?,?,?)", (last + 60, 11 + k, 3)); c.commit(); last += 60
            self.same(c, cs, T0, last + 60)
        c.execute("DELETE FROM t_1m WHERE ts < ?", (T0 + 2 * DAY,)); c.commit()                                      # retention drops the first two days
        self.same(c, cs, T0, last + 60); self.same(c, cs, T0 + 2 * DAY, last + 60); self.assertEqual(builds, [])

    def test_short_windows_and_unknown_columns_are_left_to_the_database(self):
        c = make(self.db, T0, T0 + 6 * DAY); cs = colstore.ColStore(self.db); cs.build("t_1m", ["a", "b"])
        self.assertIsNone(cs.read(c, "t_1m", ["a"], T0, T0 + 3600)); self.assertIsNone(cs.read(c, "t_1m", ["zzz"], T0, T0 + 6 * DAY)); self.assertIsNone(cs.read(c, "t", ["a"], T0, T0 + 6 * DAY))

    def test_a_folder_that_cannot_be_written_disables_it(self):
        os.environ["VGPT_COLSTORE"] = "0"
        try: cs = colstore.ColStore(self.db); self.assertFalse(cs.ok)
        finally: os.environ.pop("VGPT_COLSTORE")


if __name__ == "__main__": unittest.main()
