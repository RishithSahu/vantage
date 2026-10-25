"""Builds tests/corpus/: the small, versioned test corpus (a few megabytes per vehicle) that covers what the case study names, so that the answer tests do not depend on the multi-gigabyte generated histories.

    vehicles        ev, petrol, diesel      the last days of the generated sample cars (5 days of one-minute rows, 6 hours of one-second rows, 30 days of trips, events and daily rows)
                    hybrid                  the petrol car's engine and fuel with a traction-battery series (the electric car's, folded onto the same minutes): a synthetic hybrid, said so in its meta and its manifest entry
    protocols       bus, obd, prop          slices of the cars whose data arrives over J1939, OBD-II / DBC and a proprietary CAN (protocol variation)
    situations      missing_signals         the electric car without coolant flow or pack temperature
                    faults                  the electric car with exactly known injected events: 3 ECU resets, 1 trouble code, 1 tyre-pressure warning in its last 4 hours
                    stale                   the electric car whose record ends two days before "now" (the manifest's `now`)
                    no_data                 the electric car with every measurement and event removed (schema and vehicle profile stay)

`MANIFEST.json` says, for each file: what it is, the expected vehicle kind, the `now` to ask at, its SHA-256 and size, and the facts a test may rely on.  Everything is derived from the project's own generated cars: nothing here is third-party data."""
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
CORPUS = os.path.join(ROOT, "tests", "corpus")
SRC = {"ev": "data/world4/aster_ev/aster_ev_4yr.db", "petrol": "data/world4/kestrel_petrol/kestrel_petrol_4yr.db", "diesel": "data/world4/rhino_diesel/rhino_diesel_4yr.db",
       "bus": "data/world4/volta_bus/volta_bus_4yr.db", "obd": "data/world4/acura_rdx_2018/acura_rdx_2018_4yr.db", "prop": "data/world4/proprietary_demo/proprietary_demo_4yr.db"}
DAY = 86400
KEEP_1M_DAYS, KEEP_1S_HOURS, KEEP_LONG_DAYS = 5, 6, 30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def tables(c): return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


def cols(c, t): return [r[1] for r in c.execute(f"PRAGMA table_info([{t}])")]


def slice_car(src, dst, tmp):
    """The last days of one generated car, as a compact file at `dst`.  -> (end ts of its one-minute rows, row counts)."""
    work = os.path.join(tmp, os.path.basename(dst) + ".work"); s = sqlite3.connect("file:" + src.replace("\\", "/") + "?mode=ro", uri=True); d = sqlite3.connect(work); s.backup(d); s.close()
    end = int(d.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0])
    cut = {"1s": end - KEEP_1S_HOURS * 3600, "1m": end - KEEP_1M_DAYS * DAY, "long": end - KEEP_LONG_DAYS * DAY}
    which = {"sig_1s": "1s", "lk_1s": "1s", "sig_1m": "1m", "lk_1m": "1m", "ext_1m": "1m", "cam_1m": "1m", "gps_1s": "1m"}
    d.execute("PRAGMA foreign_keys=OFF")
    for t in tables(d):
        if t.startswith("logs_fts") or t in ("agg_1d_state", "meta", "sig_catalog", "places"): continue
        cs = cols(d, t)
        if t == "agg_1d": d.execute("DELETE FROM agg_1d WHERE d < ?", ((cut["long"] + 3600) // DAY,)); continue
        col = "ts" if "ts" in cs else "start" if "start" in cs else None
        if col is None: continue
        d.execute(f"DELETE FROM [{t}] WHERE {col} < ?", (cut[which.get(t, "long")],))
    try: d.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")                                          # the full-text index follows the rows that are left
    except sqlite3.Error: pass
    d.commit(); d.execute("VACUUM"); d.close()
    if os.path.exists(dst): os.remove(dst)
    d = sqlite3.connect(work); d.execute("VACUUM INTO ?", (dst,)); d.close(); os.remove(work)
    c = sqlite3.connect(dst); counts = {t: c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0] for t in ("sig_1m", "sig_1s", "trips", "events", "lk_1m")}; c.close()
    return end, counts


def edit(base, dst, tmp, fn):
    work = os.path.join(tmp, os.path.basename(dst) + ".work"); shutil.copy(base, work); c = sqlite3.connect(work); out = fn(c) or {}; c.commit(); c.execute("VACUUM"); c.close()
    if os.path.exists(dst): os.remove(dst)
    c = sqlite3.connect(work); c.execute("VACUUM INTO ?", (dst,)); c.close(); os.remove(work); return out


# ------------------------------------------------------------------------------------------------ the situations
def hybrid(ev_path, petrol_path, dst, tmp):
    e = sqlite3.connect(ev_path); soc_m = [r[0] for r in e.execute("SELECT soc FROM sig_1m WHERE soc IS NOT NULL ORDER BY ts")]; soc_s = [r[0] for r in e.execute("SELECT soc FROM sig_1s WHERE soc IS NOT NULL ORDER BY ts")]; e.close()

    def fn(c):
        for t, series in (("sig_1m", soc_m), ("sig_1s", soc_s)):
            rows = [r[0] for r in c.execute(f"SELECT ts FROM {t} ORDER BY ts")]
            c.executemany(f"UPDATE {t} SET soc=? WHERE ts=?", ((series[i % len(series)], ts) for i, ts in enumerate(rows)))
        prof = json.loads(c.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()[0]); prof["vehicle"]["powertrain"] = "phev"; prof["vehicle"].setdefault("specs", {})["veh.cap_kwh"] = 14.0              # the vehicle's own profile says plug-in hybrid: its vocabulary keeps the battery
        c.execute("UPDATE meta SET v=? WHERE k='vehicle.profile'", (json.dumps(prof),))
        c.execute("INSERT OR REPLACE INTO meta VALUES('veh.cap_kwh', '14.0')"); c.execute("INSERT OR REPLACE INTO meta VALUES('veh.model', 'Synthetic hybrid (petrol engine and fuel, folded traction-battery series)')")
        c.execute("INSERT OR REPLACE INTO meta VALUES('corpus.note', 'synthetic: the traction-battery state of charge is the electric sample car''s, folded onto this car''s minutes')")
        return {"soc_last": c.execute("SELECT soc FROM sig_1m WHERE soc IS NOT NULL ORDER BY ts DESC LIMIT 1").fetchone()[0] / 10.0}
    return edit(petrol_path, dst, tmp, fn)


def missing_signals(base, dst, tmp):
    def fn(c):
        for t in ("sig_1m", "sig_1s"): c.execute(f"UPDATE {t} SET flow=NULL, batt=NULL")
        try: c.execute("DELETE FROM sig_catalog WHERE concept IN ('coolant_flow','battery_temperature')")
        except sqlite3.Error: pass
        return {"missing": ["coolant flow", "battery temperature"]}
    return edit(base, dst, tmp, fn)


def faults(base, dst, tmp):
    def fn(c):
        end = int(c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]); ts = {"ECU_RESET": [end - 3 * 3600, end - 2 * 3600, end - 3600], "DTC_SET": [end - 1800], "TPMS_LOW": [end - 1200]}
        for typ, lst in ts.items():
            for t in lst: c.execute("INSERT INTO events(ts, type, sev, detail) VALUES(?,?,?,?)", (t, typ, "warn", {"ECU_RESET": "controller reset (injected for the corpus)", "DTC_SET": "P0A94 DC/DC converter performance (injected for the corpus)", "TPMS_LOW": "front left tyre pressure low (injected for the corpus)"}[typ]))
        return {"injected": ts, "ecu_resets_last_24h": 3 + int(c.execute("SELECT COUNT(*) FROM events WHERE type='ECU_RESET' AND ts>=? AND ts<?", (end - DAY, end - 3 * 3600)).fetchone()[0])}
    return edit(base, dst, tmp, fn)


def stale(base, dst, tmp, gap_days=2):
    def fn(c):
        end = int(c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]); cut = end - gap_days * DAY
        for t in tables(c):
            if t.startswith("logs_fts") or t in ("meta", "sig_catalog", "places", "agg_1d", "agg_1d_state"): continue
            cs = cols(c, t); col = "ts" if "ts" in cs else "start" if "start" in cs else None
            if col: c.execute(f"DELETE FROM [{t}] WHERE {col} > ?", (cut,))
        try: c.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")
        except sqlite3.Error: pass
        return {"original_end": end, "record_ends": cut, "gap_days": gap_days}
    return edit(base, dst, tmp, fn)


def no_data(base, dst, tmp):
    def fn(c):
        for t in tables(c):
            if t.startswith("logs_fts") or t in ("meta", "sig_catalog", "places"): continue
            c.execute(f"DELETE FROM [{t}]")
        try: c.execute("INSERT INTO logs_fts(logs_fts) VALUES('rebuild')")
        except sqlite3.Error: pass
        return {}
    return edit(base, dst, tmp, fn)


def build():
    os.makedirs(CORPUS, exist_ok=True); tmp = tempfile.mkdtemp(prefix="corpus_"); entries = []; ends = {}; t0 = time.time()

    def add(name, kind, scenario, path, now, note, facts=None):
        c = sqlite3.connect(path); rows = {t: c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0] for t in ("sig_1m", "sig_1s", "trips", "events")}; c.close()
        entries.append({"name": name, "file": os.path.basename(path), "kind": kind, "scenario": scenario, "now": int(now), "note": note, "rows": rows, "bytes": os.path.getsize(path), "sha256": sha256(path), "facts": facts or {}})
        print(f"{name:16} {kind:7} {os.path.getsize(path) / 1e6:6.1f} MB  now={int(now)}", flush=True)
    for key in ("ev", "petrol", "diesel", "bus", "obd", "prop"):
        src = os.path.join(ROOT, SRC[key])
        if not os.path.exists(src): print("missing source", src); continue
        dst = os.path.join(CORPUS, f"{key}.db"); end, counts = slice_car(src, dst, tmp); ends[key] = end; c = sqlite3.connect(dst)
        from vgpt.vehicle import caps
        kind = caps.vehicle_kind(c) or "none"; c.close()
        add(key, kind, {"ev": "electric", "petrol": "petrol", "diesel": "diesel", "bus": "protocol: J1939 bus", "obd": "protocol: OBD-II / DBC", "prop": "protocol: proprietary CAN"}[key], dst, end + 1, f"the last days of the generated sample car '{key}'")
    if "ev" in ends:
        ev = os.path.join(CORPUS, "ev.db"); end = ends["ev"]
        for name, fn, scen, note in (("missing_signals", missing_signals, "missing signals", "the electric car without coolant flow or pack temperature"), ("faults", faults, "faults", "the electric car with exactly known injected events"),
                                      ("stale", stale, "stale data", "the electric car whose record ends two days before now"), ("no_data", no_data, "no data", "the electric car with every measurement and event removed")):
            dst = os.path.join(CORPUS, name + ".db"); facts = fn(ev, dst, tmp); c = sqlite3.connect(dst)
            from vgpt.vehicle import caps
            kind = caps.vehicle_kind(c) or "none"; c.close(); add(name, kind, scen, dst, end + 1, note, facts)
    if "ev" in ends and "petrol" in ends:
        dst = os.path.join(CORPUS, "hybrid.db"); facts = hybrid(os.path.join(CORPUS, "ev.db"), os.path.join(CORPUS, "petrol.db"), dst, tmp); c = sqlite3.connect(dst)
        from vgpt.vehicle import caps
        kind = caps.vehicle_kind(c) or "none"; c.close(); add("hybrid", kind, "hybrid (synthetic)", dst, ends["petrol"] + 1, "a synthetic hybrid: the petrol car's engine and fuel with a folded traction-battery series", facts)
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"version": 1, "made": time.strftime("%Y-%m-%d"), "window": {"minute_days": KEEP_1M_DAYS, "second_hours": KEEP_1S_HOURS, "long_days": KEEP_LONG_DAYS}, "files": entries}, open(os.path.join(CORPUS, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(entries)} files, {sum(e['bytes'] for e in entries) / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__": sys.exit(build())
