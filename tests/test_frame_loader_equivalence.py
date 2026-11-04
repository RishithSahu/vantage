"""The frame loader reads arrays (and, for long windows, the columnar copy) where it used to read rows one by one.  This holds a copy of the old row-by-row algorithm and checks, on the real 4-year database when it is present,
that the loader returns the very same frames for windows of every length, both tiers and several signal sets, with the columnar copy on and off."""
import os, sqlite3, sys, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "data", "aster_ev_4yr.db")


def old_load_frame(ex, concepts, lo, hi, tier):
    """The loader as it was before the arrays: rows from SQLite, np.array(rows), searchsorted for every table."""
    from vgpt.frame import Frame
    from vgpt.query_planner import compile_select
    MAX_ROWS = 1_800_000; by_table = {}
    for c in sorted(set(concepts)):
        tiers = ex.p.tiers_of(c) or (ex.p.resolve(c).tiers if ex.p.resolve(c).status == "measured" else {})
        if not tiers: continue
        use = tier if tier in tiers else ("1m" if "1m" in tiers else next(iter(tiers)))
        tb, col, sc, *rest = tiers[use]; by_table.setdefault((tb, use), []).append((c, col, sc, rest[0] if rest else 0.0))
    if not by_table: return Frame(np.array([], dtype=np.int64), {}, 60 if tier == "1m" else 1, tier)
    order = sorted(by_table, key=lambda k: (0 if k[0].startswith("sig_") else 1 if k[0].startswith("xsig") else 2, k[1] != tier)); loaded = []
    for (tb, use) in order:
        items = by_table[(tb, use)]; cols = []
        for _, col, _, _ in items:
            for x in (["tp_fl", "tp_fr", "tp_rl", "tp_rr"] if col == "tp_*" else [col]):
                if x not in cols: cols.append(x)
        sql, args = compile_select(tb, cols, lo, hi); loaded.append((tb, use, items, cols, ex.c.execute(sql + f" LIMIT {MAX_ROWS}", args).fetchall()))
    base_tb, base_use, _, _, base_rows = loaded[0]
    res = 1 if base_use == "1s" else 60 if base_use == "1m" else 86400
    ts = np.array([r[0] for r in base_rows], dtype=np.int64); frame = Frame(ts, {}, res, base_use if base_use in ("1s", "1m") else "day", {}, [(lo, hi)])
    for (tb, use, items, cols, rows) in loaded:
        if not rows:
            for c, col, sc, off in items: frame.cols[c] = np.full(len(ts), np.nan); frame.source[c] = f"{tb}.{col}"
            continue
        arr = np.array(rows, dtype=float); tts = arr[:, 0].astype(np.int64)
        idx = np.searchsorted(tts, ts, side="right") - 1; ok = (idx >= 0)
        tol = 90 if use == "1m" else 5 if use == "1s" else 86400
        ok &= (np.abs(ts - tts[np.clip(idx, 0, len(tts) - 1)]) <= tol) if tb != base_tb else True
        idx = np.clip(idx, 0, len(tts) - 1)
        for c, col, sc, off in items:
            if col == "tp_*":
                v = arr[idx][:, [1 + cols.index(x) for x in ("tp_fl", "tp_fr", "tp_rl", "tp_rr")]] / sc + off; vals = np.where(ok, np.nanmin(v, axis=1), np.nan)
            else:
                j = 1 + cols.index(col); vals = np.where(ok, arr[idx][:, j] / sc + off, np.nan)
            if c == "gps_speed": vals = np.where(vals < 0, np.nan, vals)
            frame.cols[c] = vals; frame.source[c] = f"{tb}.{col}"
    return frame


@unittest.skipUnless(os.path.exists(DB), "needs data/aster_ev_4yr.db")
class Loader(unittest.TestCase):
    def test_same_frames_as_the_row_by_row_loader(self):
        import shutil, tempfile
        d = tempfile.mkdtemp(); db = os.path.join(d, "w.db"); shutil.copy(DB, db)
        from vgpt import colstore
        from vgpt.brain import Brain
        b = Brain(db, slm=None, slm_ir=None, policy="clarify", qc_mode="all"); b.now = int(sqlite3.connect(db).execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) + 60; ex = b.qc().executor
        cs = colstore.of(ex.c); cs.build("sig_1m", []); cs.build("ext_1m", []); hi = b.now; rng = np.random.default_rng(4)
        sets = [["speed", "drive_mode"], ["battery_power", "speed", "drive_mode"], ["motor_temperature", "ambient_temperature", "drive_mode"], ["tyre_pressure", "drive_mode"], ["battery_soc", "coolant_flow"], ["tyre_pressure_fl", "speed"]]
        spans = [3600, 6 * 3600, 2 * 86400, 5 * 86400, 30 * 86400, 200 * 86400, 1500 * 86400]
        for n in range(24):
            span = spans[n % len(spans)]; lo = int(hi - span - int(rng.integers(0, 400 * 86400))); cons = sets[n % len(sets)]; tier = "1m"
            for on in ("1", "0"):
                os.environ["VGPT_COLSTORE"] = on; cs.ok = on == "1"; ex.__dict__.pop("_frame_memo", None)
                new = ex._load_frame(cons, lo, lo + span, tier); old = old_load_frame(ex, cons, lo, lo + span, tier)
                self.assertEqual(len(new), len(old), (cons, span, on)); np.testing.assert_array_equal(new.ts, old.ts)
                for k in old.cols: np.testing.assert_array_equal(new.cols[k], old.cols[k], err_msg=f"{cons} {span}s copy={on} {k}")
        os.environ.pop("VGPT_COLSTORE", None)


if __name__ == "__main__": unittest.main()
