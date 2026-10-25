"""Join consecutive one-year world databases (vgpt.world.build, one profile) into one multi-year history.

    python tools_merge_years.py data/_y1.db data/_y2.db data/_y3.db --out data/world_3yr.db

The newest database is the base (its catalogue, profile, 1 Hz window and places are kept); every earlier database contributes the rows that lie before it: minute tiers, events, logs, trips, charges, event
context and the daily ledger.  What is continuous in a real car is made continuous: the odometer carries on from the end of the earlier year, the battery's state of health continues to fall instead of
starting again at 100 %, and each year's ground-truth file is merged (the faults injected in every year, with their exact timestamps).  Nothing is invented: every row comes from one of the inputs."""
import argparse
import json
import os
import shutil
import sqlite3

TIMED = (("sig_1m", "ts"), ("lk_1m", "ts"), ("cam_1m", "ts"), ("ext_1m", "ts"), ("vday", "ts"), ("events", "ts"), ("logs", "ts"), ("event_ctx", "ts"), ("event_ctx_v2", "ts"), ("event_ctx_x", "ts"),
         ("event_thumb", "ts"), ("event_thumb_v2", "ts"), ("source_log", "ts"), ("trips", "start"), ("trip_ext", "start"), ("charges", "start"))


def cols(c, t): return [r[1] for r in c.execute(f"PRAGMA table_info({t})")]


def _span(d, sql):
    r = d.execute(sql).fetchone(); return r if r and r[0] is not None else None


def merge(inputs, out):
    inputs = sorted(inputs, key=lambda p: int(sqlite3.connect(p).execute("SELECT v FROM meta WHERE k='t0'").fetchone()[0]))
    for e in ("", "-wal", "-shm"):
        if os.path.exists(out + e): os.remove(out + e)
    # how far each year has to be shifted so that it starts where the year before it ended (odometer in km, state of health in %)
    shifts = []; prev_end_odo = prev_end_soh = None
    for k, p in enumerate(inputs):
        d = sqlite3.connect(p)
        odo = _span(d, "SELECT MIN(odo), MAX(odo) FROM vday"); soh = d.execute("SELECT (SELECT soh FROM vday WHERE soh IS NOT NULL ORDER BY ts LIMIT 1), (SELECT soh FROM vday WHERE soh IS NOT NULL ORDER BY ts DESC LIMIT 1)").fetchone(); d.close()
        so = (prev_end_odo - odo[0]) if (prev_end_odo is not None and odo) else 0.0
        sh = (prev_end_soh - soh[0]) if (prev_end_soh is not None and soh and soh[0] is not None) else 0.0
        shifts.append((so, sh))
        if odo: prev_end_odo = odo[1] + so
        if soh and soh[1] is not None: prev_end_soh = soh[1] + sh
    shutil.copyfile(inputs[-1], out)
    c = sqlite3.connect(out); c.execute("PRAGMA journal_mode=DELETE")
    sid = dict(c.execute("SELECT concept, sid FROM sig_catalog").fetchall())

    def shift_rows(conn, so, sh):
        if so or sh:
            conn.execute("UPDATE vday SET odo = odo + ?, soh = soh + ?", (so, sh))
            conn.execute("UPDATE ext_1m SET odo = odo + ?", (int(round(so * 10)),))
            for tb in ("lk_1m", "lk_1s"):
                for concept, delta in (("odometer", so), ("battery_soh", sh)):
                    if concept in sid and delta:
                        if tb == "lk_1m": conn.execute("UPDATE lk_1m SET v = v + ?, vmin = vmin + ?, vmax = vmax + ? WHERE sid = ?", (delta, delta, delta, sid[concept]))
                        else: conn.execute("UPDATE lk_1s SET v = v + ? WHERE sid = ?", (delta, sid[concept]))
    shift_rows(c, *shifts[-1]); c.commit()
    k_last = len(inputs) - 1; t0_first = None
    for k, p in reversed(list(enumerate(inputs[:-1]))):       # newest first: each year is added in front of the ones already merged
        tmp = out + f".part{k}"; shutil.copyfile(p, tmp); w = sqlite3.connect(tmp); shift_rows(w, *shifts[k]); w.commit()      # the earlier year, shifted, in a scratch copy
        if k == 0: t0_first = w.execute("SELECT v FROM meta WHERE k='t0'").fetchone()[0]
        for t, tc in TIMED:
            try: cn = cols(w, t)
            except sqlite3.Error: continue
            if not cn: continue
            lo = c.execute(f"SELECT MIN({tc}) FROM {t}").fetchone()[0]; lo = lo if lo is not None else 2 ** 62
            rows = w.execute(f"SELECT {','.join(cn)} FROM {t} WHERE {tc} < ?", (lo,)).fetchall()
            if not rows: continue
            if t == "trips":
                i = cn.index("id"); rows = [r[:i] + (None,) + r[i + 1:] for r in rows]
            ph = ",".join("?" * len(cn)); c.executemany(f"INSERT OR IGNORE INTO {t}({','.join(cn)}) VALUES({ph})", rows)
        w.close(); os.remove(tmp)
    if t0_first: c.execute("INSERT OR REPLACE INTO meta VALUES('t0',?)", (t0_first,))
    last = c.execute("SELECT MAX(odo) FROM vday").fetchone()[0]
    if last: c.execute("INSERT OR REPLACE INTO meta VALUES('veh.odo_end',?)", (str(last),))
    c.commit()
    try: c.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')"); c.commit()
    except sqlite3.Error: pass
    c.execute("VACUUM"); c.close()
    return shifts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
    print(merge(a.inputs, a.out))
    truth = {"years": []}
    for p in sorted(a.inputs):
        tp = p[:-3] + "_truth.json"
        if os.path.exists(tp): truth["years"].append(json.load(open(tp)))
    json.dump(truth, open(a.out[:-3] + "_truth.json", "w"))


if __name__ == "__main__":
    main()
