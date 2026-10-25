"""The test-data bundle: what the tests need that is not source code, how it is made, and how it is checked.

    python tools_testdata.py check               which groups of test data are present, and whether their files still match the recorded SHA-256 (exit code 1 when a required group is missing or changed)
    python tools_testdata.py make-driving        write tests/fixtures/driving_segments.npz from the 1 Hz tail of the generated sample cars (the small derived stand-in for the real driving traces)
    python tools_testdata.py corpus              build tests/corpus/: a few-megabyte slice of every kind of vehicle the case study names (electric, petrol, diesel, hybrid) and of the situations that matter (missing signals,
                                                 protocol variation, faults, stale data, no data), with tests/corpus/MANIFEST.json
    python tools_testdata.py manifest            (re)write tests/testdata_manifest.json: the groups, their files and their SHA-256

`tests/testdata.py` reads the manifest.  A test that needs a group asks `testdata.require(group)`: absent data skips the test with the reason and the way to get it, and `VGPT_REQUIRE_TESTDATA=1` (the release gate) turns every such skip into a failure."""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
MANIFEST = os.path.join(ROOT, "tests", "testdata_manifest.json")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "driving_segments.npz")
CORPUS = os.path.join(ROOT, "tests", "corpus")
CARS = {"ev": "data/world4/aster_ev/aster_ev_4yr.db", "petrol": "data/world4/kestrel_petrol/kestrel_petrol_4yr.db", "diesel": "data/world4/rhino_diesel/rhino_diesel_4yr.db",
        "bus": "data/world4/volta_bus/volta_bus_4yr.db", "obd": "data/world4/acura_rdx_2018/acura_rdx_2018_4yr.db", "prop": "data/world4/proprietary_demo/proprietary_demo_4yr.db"}
GROUPS = {
    "driving_fixture": {"what": "3-minute driving windows (speed, acceleration) derived from the generated sample cars; lets the world generator run without the real VIF-AEGIS / EPA traces", "files": ["tests/fixtures/driving_segments.npz"], "make": "python tools_testdata.py make-driving"},
    "corpus": {"what": "small slices of an electric, a petrol, a diesel and a hybrid car and of the situations that matter (missing signals, protocol variation, faults, stale data, no data)", "files": ["tests/corpus/MANIFEST.json"], "make": "python tools_testdata.py corpus"},
    "sample_cars": {"what": "the four-year generated sample cars (electric, petrol, diesel): the ground the answer tests are checked against", "files": [v for k, v in CARS.items() if k in ("ev", "petrol", "diesel")], "make": "python tools_build_worlds.py (needs the driving traces or the driving fixture)"},
    "legacy_worlds": {"what": "the first generated vehicles and their ground truth (gen_*.db, truth.json), which older tests quote number for number; they cannot be regenerated bit-for-bit without the real driving traces", "files": ["data/gen_courier.db", "data/gen_courier_ext.db", "data/gen_demo.db", "data/gen_cold.db", "data/gen_roadwarrior.db", "data/truth.json"], "make": "restore them from the release data bundle"},
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def load_manifest():
    try: return json.load(open(MANIFEST, encoding="utf-8"))
    except (OSError, ValueError): return {"version": 0, "groups": {}}


def check(strict_groups=("driving_fixture", "corpus")):
    m = load_manifest(); bad = 0
    for g, spec in GROUPS.items():
        rec = (m.get("groups") or {}).get(g, {}).get("files", {}); missing, changed = [], []
        for f in spec["files"]:
            p = os.path.join(ROOT, f)
            if not os.path.exists(p): missing.append(f); continue
            if f in rec and rec[f] and sha256(p) != rec[f]: changed.append(f)
        state = "ok" if not missing and not changed else "MISSING " + ", ".join(missing) if missing else "CHANGED " + ", ".join(changed)
        print(f"{g:16} {state}\n   {spec['what']}" + ("" if state == "ok" else f"\n   make: {spec['make']}"))
        if (missing or changed) and g in strict_groups: bad += 1
    return 1 if bad else 0


def write_manifest(hash_big=True):
    m = {"version": 1, "made": time.strftime("%Y-%m-%d"), "groups": {}}
    for g, spec in GROUPS.items():
        files = {}
        for f in spec["files"]:
            p = os.path.join(ROOT, f)
            if os.path.exists(p): files[f] = sha256(p) if (hash_big or os.path.getsize(p) < 50e6) else None
            else: files[f] = None
        m["groups"][g] = {"what": spec["what"], "make": spec["make"], "files": files}
    json.dump(m, open(MANIFEST, "w", encoding="utf-8"), indent=1); print("wrote", MANIFEST)


# ------------------------------------------------------------------------------------------------ the driving fixture
def make_driving():
    import numpy as np
    G = 9.80665; W = 180; pools = {"city": [], "hwy": [], "fast": [], "sport": []}
    for key in ("ev", "petrol", "diesel"):
        p = os.path.join(ROOT, CARS[key])
        if not os.path.exists(p): continue
        c = sqlite3.connect("file:" + p.replace("\\", "/") + "?mode=ro", uri=True)
        rows = c.execute("SELECT ts, speed FROM sig_1s WHERE speed IS NOT NULL ORDER BY ts").fetchall(); c.close()
        if not rows: continue
        ts = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows], np.float32) / 10.0                  # stored in 0.1 km/h
        cut = np.where(np.diff(ts) != 1)[0] + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(ts)]):
            run = v[a:b]
            if len(run) < W: continue
            lon = (np.gradient(run / 3.6) / G).astype(np.float32)
            for i in range(0, len(run) - W, 120):
                w = run[i:i + W]; m = float(w.mean()); seg = (w.copy(), lon[i:i + W].copy(), np.zeros(W, np.float32), float(np.percentile(np.abs(lon[i:i + W]), 95)))
                if m >= 108: pools["fast"].append(seg)
                if m >= 85: pools["hwy"].append(seg)
                elif 10 <= m <= 62: pools["city"].append(seg)
                if m >= 50 and seg[3] > 0.15: pools["sport"].append(seg)
    keep = {"city": 40, "hwy": 20, "fast": 8, "sport": 8}; out = {}
    for k, n in keep.items():
        pool = pools[k] or pools["hwy"] or pools["city"]
        idx = np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)                                       # spread over the whole pool, deterministic
        out[k + "_v"] = np.stack([pool[i][0] for i in idx]); out[k + "_lon"] = np.stack([pool[i][1] for i in idx]); out[k + "_lat"] = np.stack([pool[i][2] for i in idx])
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True); np.savez_compressed(FIXTURE, **out)
    print("wrote", FIXTURE, {k: len(out[k + "_v"]) for k in keep}, round(os.path.getsize(FIXTURE) / 1024), "KB")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    for n in ("check", "make-driving", "corpus", "manifest"): sub.add_parser(n)
    a = ap.parse_args()
    if a.cmd == "check": return check()
    if a.cmd == "make-driving": return make_driving()
    if a.cmd == "manifest": return write_manifest()
    if a.cmd == "corpus":
        from tools_testdata_corpus import build
        return build()
    ap.print_help(); return 2


if __name__ == "__main__": sys.exit(main() or 0)
