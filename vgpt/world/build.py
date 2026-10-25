"""Build a long, multi-source vehicle history for any profile pack, with ground truth.

    python -m vgpt.world.build --profile aster_ev --days 105 --seed 5 --out data/world_aster.db

105 days (15 weeks) of one vehicle's life: real driving traces and accelerometer data (VIF-AEGIS, EPA cycles), synthetic or real weather, driver routine, charging, physics, the life generator's stories
(cooling pump, tyre leak, weak 12 V battery, dirty windscreen, roadworks, roof box, harsh week, alignment drift) and the fault layer (vgpt/world/faults.py: ECU resets, brown-outs, an unexpected shutdown,
CAN bursts, sensor conflicts, an intermittent DC/DC fault, thermal soak, DTC cycles, ADAS flapping, AEB, range-estimate glitches).  The result is written as a PLATFORM database:

  legacy tables of the on-device schema (sig_1s 7 days at 1 Hz, sig_1m, trips, trip_ext, charges, places, events, event_ctx, logs, cam_1m, ext_1m, vday, gps_1s)   - what every answer route reads
  the signal lake of the profile (sig_catalog, lk_1s, lk_1m, per-signal views)                                                                                  - every other signal, any vehicle
  event_ctx_x (the non-core channels around each important event), profile + retention in meta
and <db>_truth.json with everything the generators decided (stories, injected faults with exact timestamps, daily aggregates).
Why a bulk writer next to the live collectors: 105 days at 10 Hz of raw bus traffic is 90 million frames; the history is written through the same tiers directly, and tests/test_world.py proves
on an hour of raw frames that the live path (emitter -> router -> ingestor -> lake) produces the same rows."""
import argparse
import json
import os
import sqlite3
import time
import zlib

import numpy as np

from .. import datagen as DG, datagen_events as DE, datagen_life as DL, store
from ..vehicle import lake as LK
from ..vehicle.lake import CTX_TYPES_X, CTX_X_CONCEPTS
from ..vehicle.profile import Profile
from .faults import ALL as ALL_FAULTS, Faults
from .signals import Signals
from . import ice as ICE

DEFAULT_STORIES = ["cooling_pump", "tyre_leak", "weak_12v", "dirty_windscreen", "roadworks", "roof_box", "harsh_week", "alignment_drift"]
def vehicle_from_profile(profile, base="family"):
    V = dict(DG.VEHICLES[base]); sp = profile.specs
    if "veh.cap_kwh" in sp: V["cap"] = float(sp["veh.cap_kwh"])
    if "veh.tyre_spec_front" in sp: V["tyre"] = (float(sp["veh.tyre_spec_front"]), float(sp.get("veh.tyre_spec_rear", sp["veh.tyre_spec_front"])))
    V["label"] = str(sp.get("veh.model", V["label"]))
    if "veh.rated_max_speed" in sp: V["top"] = float(sp["veh.rated_max_speed"])
    return V


def build_world(profile="aster_ev", days=105, seed=5, city="london", driver="commuter", vehicle="family", end="2026-10-18", stories=None, faults=ALL_FAULTS, work_km=27, verbose=True):
    p = profile if isinstance(profile, Profile) else Profile.load(profile); p.check()
    DG.VEHICLES["pack"] = vehicle_from_profile(p, vehicle)
    gv = DL.build(seed=seed, days=days, city=city, driver=driver, vehicle="pack", end=end, stories=DEFAULT_STORIES if stories is None else stories, offline=True, verbose=verbose, work_km=work_km)
    gv.profile = p; gv.faults = Faults(gv, p, seed, which=faults); gv.fault_seed = seed
    if verbose: print(f"  faults injected: { {k: len(v) for k, v in gv.faults.truth.items()} }")
    return gv


def _stats(a):
    m = ~np.isnan(a); cnt = m.sum(1); a0 = np.where(m, a, 0.0)
    return a0.sum(1) / np.maximum(cnt, 1), np.where(m, a, np.inf).min(1), np.where(m, a, -np.inf).max(1), cnt


def write_world(gv, path, verbose=True):
    p, F = gv.profile, gv.faults; t0 = time.time()
    DE.seed(path, gv)                                                                             # the legacy tables (and the meta the generator knows)
    conn = store.open_rw(path); sids = LK.provision(conn, p)
    conn.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", list(gv.meta.items())); conn.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(gv.t0 + gv.n),))
    lake = [c for c in LK.lake_concepts(p) if c in sids]; S = Signals(gv, p, gv.fault_seed, faults=F); days = gv.n // 86400; fine = p.retention["fine_days"]; S.fine_days = fine; ctx_c = [c for c in CTX_X_CONCEPTS if c in lake or c in ("speed", "battery_temperature", "coolant_flow", "motor_temperature")]
    by_day = {}
    for (ts, ty, sv, de) in gv.events:
        if ty in CTX_TYPES_X: by_day.setdefault(int(ts - gv.t0) // 86400, []).append((int(ts), ty))
    conn.execute("BEGIN"); n1m = n1s = 0
    for d in range(days):
        out = S.day(d); awake = out["_awake"]; base = gv.t0 + d * 86400; amin = awake.reshape(1440, 60).any(1); keep = amin | (((np.arange(1440) + d * 1440) % 10) == 0); mts = base + np.arange(1440) * 60
        rows = []
        for c in lake:
            mean, lo, hi, cnt = _stats(out[c].astype(np.float64).reshape(1440, 60)); idx = np.where(keep & (cnt > 0))[0]; sid = sids[c]
            rows += [(sid, int(mts[i]), float(mean[i]), float(lo[i]), float(hi[i]), int(cnt[i])) for i in idx]
        conn.executemany("INSERT OR REPLACE INTO lk_1m VALUES(?,?,?,?,?,?)", rows); n1m += len(rows)
        if d >= days - fine:
            sec = np.arange(86400); sel = awake | (sec % 60 == 0); rows = []
            for c in lake:
                a = out[c]; idx = np.where(sel & ~np.isnan(a))[0]; sid = sids[c]; vals = np.round(a[idx].astype(np.float64), 3)
                rows += list(zip([sid] * len(idx), (base + idx).tolist(), vals.tolist()))
            conn.executemany("INSERT OR REPLACE INTO lk_1s VALUES(?,?,?)", rows); n1s += len(rows)
        blobs = []
        for ts, ty in by_day.get(d, ()):
            i = int(ts - base); a0, a1 = max(0, i - 120), min(86400, i + 30)
            sig = {c: [None if np.isnan(x) else round(float(x), 3) for x in out[c][a0:a1]] for c in ctx_c if c in out}
            blobs.append((int(ts), ty, zlib.compress(json.dumps({"t0": int(base + a0), "hz": 1, "sig": sig}, separators=(",", ":")).encode(), 6)))
        conn.executemany("INSERT OR REPLACE INTO event_ctx_x VALUES(?,?,?)", blobs)
        if verbose and d % 15 == 0: print(f"    day {d + 1}/{days}: {n1m} lake minutes, {n1s} lake seconds so far ({time.time() - t0:.0f}s)")
    for (a, b) in F.gaps: conn.execute("DELETE FROM sig_1s WHERE ts>=? AND ts<?", (gv.t0 + a, gv.t0 + b)); conn.execute("DELETE FROM gps_1s WHERE ts>=? AND ts<?", (gv.t0 + a, gv.t0 + b))
    for src, ivs in F.dropouts.items():
        if any(c in LK.EXT_COLS for c in p.concept_names() if any(b.source == src and b.concept == c for b in p.bindings)):
            for (a, b) in ivs: conn.execute("DELETE FROM ext_1m WHERE ts>=? AND ts<?", (gv.t0 + a - 59, gv.t0 + b))
    for (ts, src, msg) in F.logs:
        cur = conn.execute("INSERT INTO logs(ts,src,msg) VALUES(?,?,?)", (int(ts), src, msg)); conn.execute("INSERT INTO logs_fts(rowid,msg,src) VALUES(?,?,?)", (cur.lastrowid, msg, src))
    for (ts, src, status, detail) in getattr(F, "source_log", []): conn.execute("INSERT INTO source_log VALUES(?,?,?,?)", (int(ts), src, status, detail))
    conn.execute("COMMIT")
    if S.ice is not None: conn.execute("BEGIN"); ICE.finish(conn, gv, S.ice); conn.execute("UPDATE vday SET soh=NULL"); conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    tables = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("sig_1s", "sig_1m", "lk_1s", "lk_1m", "events", "logs", "trips", "charges", "event_ctx", "event_ctx_x")}
    conn.execute("VACUUM"); conn.close()
    truth = dict(gv.truth); truth["profile"] = p.id; truth["tables"] = tables; truth["lake_concepts"] = lake
    tp = path[:-3] + "_truth.json" if path.endswith(".db") else path + "_truth.json"
    def js(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, (set, tuple)): return list(o)
        raise TypeError(type(o))
    json.dump(truth, open(tp, "w"), indent=1, default=js)
    if verbose: print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB) in {time.time() - t0:.0f}s: {tables}")
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="aster_ev"); ap.add_argument("--days", type=int, default=105); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--city", default="london", choices=sorted(DG.CITIES))
    ap.add_argument("--driver", default="commuter", choices=sorted(DG.DRIVERS)); ap.add_argument("--vehicle", default="family", choices=sorted(DG.VEHICLES)); ap.add_argument("--end", default="2026-10-18")
    ap.add_argument("--work-km", type=float, default=27.0); ap.add_argument("--stories", default=",".join(DEFAULT_STORIES)); ap.add_argument("--faults", default=",".join(ALL_FAULTS)); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t = time.time()
    gv = build_world(a.profile, a.days, a.seed, a.city, a.driver, a.vehicle, a.end, [s for s in a.stories.split(",") if s], tuple(f for f in a.faults.split(",") if f), a.work_km)
    write_world(gv, a.out); print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
