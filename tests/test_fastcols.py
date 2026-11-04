"""The per-minute questions answered from the columnar copy (vgpt/fastcols.py) must give exactly what the SQL they replace gave: on a generated history (with NULLs, with rows that arrived after the copy was made) and on the real four-year database."""
import os, sqlite3, sys, tempfile, time, unittest
from types import SimpleNamespace
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt import colstore, fastcols as FC, qa_core as Q, qa_b, analytics as A
colstore.REFRESH_MIN_S = 0.0                                                                     # tests change the data between calls and expect the very next call to see it, fastcols as FC, qa_core as Q, qa_b, analytics as A

DAY = 86400; T0 = 1_700_000_000 - 1_700_000_000 % DAY
SCHEMA = "CREATE TABLE sig_1m(ts INTEGER PRIMARY KEY, speed INT, soc INT, power INT, batt INT, batt_max INT, flow INT, flow_min INT, amb INT, hvac INT, motor INT, mode INT) WITHOUT ROWID"


def rows(t_from, t_to, seed):
    rng = np.random.default_rng(seed); out = []
    for t in range(t_from, t_to, 60):
        mode = int(rng.choice([0, 1, 2, 3], p=[.4, .25, .25, .1])); run = mode in (1, 2, 3)
        flow = int(rng.integers(300, 900)) if run and rng.random() > .1 else int(rng.integers(0, 60))
        if t > t_to - 3 * DAY and run and rng.random() < .3: flow = int(rng.integers(100, 250))                    # a low-flow spell at the end, so 'abnormal' has something to find
        r = [t, int(rng.integers(0, 1300)), int(rng.integers(50, 1000)), int(rng.integers(-300, 900)), int(rng.integers(150, 450)), int(rng.integers(200, 500)), flow, max(flow - 20, 0), int(rng.integers(-50, 380)), int(rng.integers(0, 500)), int(rng.integers(200, 800)), mode]
        for j in (1, 6, 8):
            if rng.random() < .01: r[j] = None
        out.append(r)
    return out


def wait_idle():
    for _ in range(600):
        with colstore._LOCK:
            if not colstore._BUILDING: return
        time.sleep(0.02)


def slow(fn, *a, **k):
    os.environ["VGPT_FASTCOLS"] = "0"
    try: return fn(*a, **k)
    finally: os.environ.pop("VGPT_FASTCOLS", None)


class Fast(unittest.TestCase):
    def setUp(self):
        A.TZ = 19800                                                                                                  # a local day that does not start at midnight UTC
        self.d = tempfile.mkdtemp(); self.db = os.path.join(self.d, "w.db"); colstore._STORES.clear(); colstore._BUILDING.clear()
        self.c = sqlite3.connect(self.db); self.c.execute("PRAGMA journal_mode=WAL"); self.c.execute(SCHEMA); self.c.execute("CREATE TABLE sig_1s(ts INTEGER PRIMARY KEY, speed INT) WITHOUT ROWID")
        self.hi = T0 + 40 * DAY; self.c.executemany("INSERT INTO sig_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows(T0, self.hi, 1)); self.c.commit()
        colstore.of(self.c).build("sig_1m", ["flow"]); self.assertTrue(os.path.exists(os.path.join(self.db + ".cols", "sig_1m.json")))

    def tearDown(self): A.TZ = 0

    def all(self):
        o = SimpleNamespace(c=self.c); now = self.c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]
        lo = T0
        for modes, minv, col, minn in (((1, 2), None, "motor", 20), ((1, 2, 3), 3.0, "flow", 20), ((1, 2, 3), None, "batt_max", 5), ((1, 2, 3), None, "batt", 20), ((0, 1), None, "soc", 1), ((1, 2), 3.0, "speed", 10)):
            for a, b in ((lo, now + 1), (lo + 5 * DAY, lo + 20 * DAY + 77), (now - 4 * DAY, now + 1)):
                with self.subTest(daily=(col, modes, a, b)): self.assertEqual(qa_b.MixB.daily(o, col, a, b, modes=modes, minv=minv, minn=minn), slow(qa_b.MixB.daily, o, col, a, b, modes=modes, minv=minv, minn=minn))
        for a, b in ((lo, now + 1), (lo + 3 * DAY, lo + 30 * DAY)):
            with self.subTest(flow_days=(a, b)): self.assertEqual(Q.flow_days(self.c, a, b), slow(Q.flow_days, self.c, a, b))
        for col, modes, minv in (("(batt-amb)", (1, 2, 3), None), ("(motor-amb)", (1, 2, 3), None), ("flow", (1, 2, 3), 3.0), ("speed", (1, 2), None)):
            with self.subTest(baseline=col): self.assertEqual(Q.baseline_stats(self.c, col, lo, lo + 14 * DAY, modes=modes, minv=minv), slow(Q.baseline_stats, self.c, col, lo, lo + 14 * DAY, modes=modes, minv=minv))
        for col in ("flow", "speed", "soc"):
            for a, b in ((lo, now + 1), (lo + 3 * DAY, lo + 30 * DAY + 11)):
                with self.subTest(per_day=(col, a, b)):
                    ref = dict(self.c.execute(f"SELECT (ts + ?) / 86400, COUNT(*) FROM sig_1m WHERE ts >= ? AND ts < ? AND {col} IS NOT NULL GROUP BY 1", (A.TZ, a, b)).fetchall())
                    self.assertEqual(FC.count_per_day(self.c, "sig_1m", col, a, b, A.TZ), ref)
                    self.assertEqual(FC.series(self.c, "sig_1m", col, 10, a, b), {int(t): v / 10 for t, v in self.c.execute(f"SELECT ts, {col} FROM sig_1m WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (a, b))})
                    mn, k = self.c.execute(f"SELECT MIN({col}), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (a, b)).fetchone(); last = self.c.execute(f"SELECT {col} FROM sig_1m WHERE ts>=? AND ts<? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (a, b)).fetchone()
                    self.assertEqual(FC.min_count_last(self.c, "sig_1m", col, a, b), (mn, k, last[0] if last else None))
        for a, b in ((lo, now + 1), (now - 10 * DAY, now + 1), (lo + 4 * DAY, lo + 9 * DAY + 5)):
            with self.subTest(load=(a, b)):
                x, y = Q.load(self.c, a, b), slow(Q.load, self.c, a, b)
                self.assertEqual((x.ts, x.res, x.tier), (y.ts, y.res, y.tier)); self.assertEqual(list(x.cols), list(y.cols))
                for k in x.cols: np.testing.assert_array_equal(np.array(x.cols[k]), np.array(y.cols[k]))
            with self.subTest(rows_per_day=(a, b)):
                self.assertEqual(FC.rows_per_day(self.c, "sig_1m", a, b, A.TZ, DAY), dict(self.c.execute("SELECT CAST((ts + ?) / ? AS INTEGER), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? GROUP BY 1", (A.TZ, DAY, a, b))))
        for n in (now, now - 3 * DAY, now - 25 * DAY):
            with self.subTest(last_abnormal=n): self.assertEqual(Q.last_abnormal(self.c, n), slow(Q.last_abnormal, self.c, n))
            with self.subTest(flow_onset=n): self.assertEqual(Q.flow_onset(self.c, n), slow(Q.flow_onset, self.c, n))

    def test_the_copy_gives_what_the_database_gives(self):
        self.all()
        self.assertTrue(FC.daily(self.c, "flow", T0, self.hi, (1, 2, 3), 3.0, 10.0, "AVG", 20))                    # and it really is the copy answering, not the fallback

    def test_rows_that_arrived_after_the_copy_was_made_are_included(self):
        nxt = self.hi
        for step in (30, 600, 3000):                                                                                  # the live writer: half an hour, then ten hours, then two days
            self.c.executemany("INSERT INTO sig_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows(nxt, nxt + step * 60, step)); self.c.commit(); nxt += step * 60
            self.all()
        wait_idle()
        self.all()

    def test_while_the_writer_keeps_appending(self):
        """The app's real situation: a writer adds rows (a burst every few milliseconds here, one a minute in life) while questions are asked and the copy is extended in the background.  A question sees one consistent snapshot, so the fast and the SQL answer must agree each time."""
        import threading
        stop = threading.Event(); nxt = [self.hi]; old = colstore.EXTEND_AT; colstore.EXTEND_AT = 40; errors = []

        def write():
            w = sqlite3.connect(self.db, timeout=30)
            try:
                k = 0
                while not stop.is_set():
                    w.executemany("INSERT INTO sig_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows(nxt[0], nxt[0] + 25 * 60, k)); w.commit(); nxt[0] += 25 * 60; k += 1; time.sleep(0.004)
            except Exception as e: errors.append(e)
            finally: w.close()
        th = threading.Thread(target=write); th.start()
        try:
            o = SimpleNamespace(c=self.c); checked = 0; t_end = time.time() + 6
            while time.time() < t_end:
                self.c.execute("BEGIN")                                                                               # one snapshot for both answers
                try:
                    now = self.c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]
                    self.assertEqual(qa_b.MixB.daily(o, "flow", T0, now + 1, modes=(1, 2, 3), minv=3.0), slow(qa_b.MixB.daily, o, "flow", T0, now + 1, modes=(1, 2, 3), minv=3.0))
                    self.assertEqual(Q.flow_days(self.c, T0, now + 1), slow(Q.flow_days, self.c, T0, now + 1))
                    self.assertEqual(Q.last_abnormal(self.c, now), slow(Q.last_abnormal, self.c, now))
                    x, y = Q.load(self.c, now - 6 * DAY, now + 1), slow(Q.load, self.c, now - 6 * DAY, now + 1); self.assertEqual(x.ts, y.ts); np.testing.assert_array_equal(np.array(x.cols["flow"]), np.array(y.cols["flow"]))
                    checked += 1
                finally: self.c.execute("COMMIT")
        finally:
            stop.set(); th.join(); colstore.EXTEND_AT = old
        self.assertFalse(errors); self.assertGreater(checked, 0); wait_idle(); self.all()

    def test_a_window_the_copy_cannot_serve_falls_back_to_the_database(self):
        o = SimpleNamespace(c=self.c)
        self.assertIsNone(FC.daily(self.c, "flow", T0, T0 + DAY, (1,), None, 10.0, "AVG", 1))                       # shorter than the copy's smallest window
        self.assertEqual(qa_b.MixB.daily(o, "flow", T0, T0 + DAY, modes=(1,), minn=1), slow(qa_b.MixB.daily, o, "flow", T0, T0 + DAY, modes=(1,), minn=1))
        self.assertIsNone(FC.daily(self.c, "flow", T0, self.hi, (1,), None, 10.0, "MAX", 1))                         # not the mean: SQL as before


REAL = os.path.join(ROOT, "data", "aster_ev_4yr.db")


@unittest.skipUnless(os.path.exists(os.path.join(REAL + ".cols", "sig_1m.json")), "the four-year database and its copy are not here")
class Real(unittest.TestCase):
    def test_the_four_year_history(self):
        c = sqlite3.connect(f"file:{REAL}?mode=ro", uri=True); o = SimpleNamespace(c=c); now = c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]; lo = c.execute("SELECT MIN(ts) FROM sig_1m").fetchone()[0]; A.TZ = 0
        for col, modes, minv in (("flow", (1, 2, 3), 3.0), ("batt", (1, 2, 3), None), ("motor", (1, 2), None), ("batt_max", (1, 2, 3), None)):
            with self.subTest(daily=col): self.assertEqual(qa_b.MixB.daily(o, col, lo, now + 1, modes=modes, minv=minv), slow(qa_b.MixB.daily, o, col, lo, now + 1, modes=modes, minv=minv))
        self.assertEqual(Q.flow_days(c, lo, now + 1), slow(Q.flow_days, c, lo, now + 1))
        for n in (now, now - 400 * DAY, now - 900 * DAY):
            with self.subTest(last_abnormal=n): self.assertEqual(Q.last_abnormal(c, n), slow(Q.last_abnormal, c, n))
            with self.subTest(flow_onset=n): self.assertEqual(Q.flow_onset(c, n), slow(Q.flow_onset, c, n))


if __name__ == "__main__":
    unittest.main()
