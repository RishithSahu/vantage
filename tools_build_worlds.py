"""Build a four-year, dense history for every car model (or the ones named): four one-year worlds with their own seeds, joined into one continuous database, with the daily tier rolled.

    python tools_build_worlds.py [--only id,id] [--jobs 6] [--out data/world4] [--keep-years]

Each car gets its own city, driver and seed (from its id), so the histories differ the way real cars' lives do.  Every year is built by vgpt.world.build (physics, weather, routine, stories, faults); the years are joined by
tools_merge_years (odometer and state of health continue across the joins, nothing is invented); the daily tier (agg_1d) is rolled so that a question about four years reads ~1,500 rows per quantity.
Dense: every quantity the car's decoder file binds (30-45 per car) at one-minute resolution for four years, one-second resolution for the last 14 days, and the event / trip / charge / refuel / log records of the
whole period.  One car takes about three minutes; the file is 100-600 MB."""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
ENDS = [("2023-10-18", 365), ("2024-10-18", 366), ("2025-10-18", 365), ("2026-10-18", 365)]
ENDS5 = [("2022-10-18", 365)] + ENDS
CITIES = ["london", "oslo", "graz", "chicago", "phoenix", "sydney", "singapore", "bengaluru"]
DRIVERS = ["commuter", "commuter", "delivery", "long_distance", "retiree", "rideshare"]


def pick(pid, spec):
    h = int(hashlib.sha256(pid.encode()).hexdigest(), 16); sp = spec or {}
    wb = float(sp.get("veh.wheelbase_m", 2.7)); cap = float(sp.get("veh.cap_kwh", 0))
    base = "suv" if wb > 3.0 or cap > 90 else "city" if wb < 2.4 else "compact" if wb < 2.65 else "family"
    return dict(city=CITIES[h % len(CITIES)], driver=DRIVERS[(h >> 8) % len(DRIVERS)], vehicle=base, seed=1000 + (h >> 16) % 8000, work_km=float(18 + (h >> 24) % 28))


def settle(path):
    """Roll the daily tier and compact the file.  The minute tier stays in the database: a question about any part of the history is answered by the same SQL as a question about last week, with no step in between."""
    from vgpt import store
    conn = store.open_rw(path); store.ensure_daily_schema(conn); n = store.roll_daily(conn); conn.commit(); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"); conn.execute("VACUUM"); conn.close()
    return n


def build_car(pid, out_dir, keep_years=False, ends=None):
    from vgpt.vehicle.profile import Profile
    t0 = time.time(); p = Profile.load(pid); c = pick(pid, p.specs); work = os.path.join(out_dir, "_" + pid); os.makedirs(work, exist_ok=True); years = []
    ends = ends or ENDS
    for k, (end, days) in enumerate(ends):
        y = os.path.join(work, f"y{k + 1}.db"); years.append(y)
        r = subprocess.run([sys.executable, "-m", "vgpt.world.build", "--profile", pid, "--days", str(days), "--seed", str(c["seed"] + k), "--end", end, "--city", c["city"], "--driver", c["driver"], "--vehicle", c["vehicle"],
                            "--work-km", str(c["work_km"]), "--out", y], capture_output=True, text=True, cwd=ROOT)
        if r.returncode: return pid, False, f"year {k + 1}: " + (r.stderr.strip().splitlines() or ["?"])[-1][:160], 0
    car_dir = os.path.join(out_dir, pid); os.makedirs(car_dir, exist_ok=True); out = os.path.join(car_dir, f"{pid}_{len(ends)}yr.db")          # the archive folder sits next to the database, so each car has a folder of its own
    import tools_merge_years as M
    M.merge(years, out)
    n = settle(out)
    truth = {}
    for k, y in enumerate(years):
        tp = y[:-3] + "_truth.json"
        if os.path.exists(tp): truth[f"year{k + 1}"] = json.load(open(tp))
    truth["build"] = dict(c, profile=pid, ends=[e for e, _ in ends], days=sum(d for _, d in ends))
    json.dump(truth, open(out[:-3] + "_truth.json", "w"), default=str)
    if not keep_years: shutil.rmtree(work, ignore_errors=True)
    return pid, True, f"{os.path.getsize(out) / 1e6:.0f} MB, tier {n} rows, {time.time() - t0:.0f}s", os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--out", default=os.path.join(ROOT, "data", "world4")); ap.add_argument("--keep-years", action="store_true"); ap.add_argument("--years", type=int, default=4, choices=(4, 5)); ap.add_argument("--min-free-gb", type=float, default=0.0, help="stop starting new cars when the disk has less free space than this"); a = ap.parse_args()
    ids = [x for x in a.only.split(",") if x] or [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(ROOT, "vgpt", "vehicle", "packs", "*", "profile.json")))]
    os.makedirs(a.out, exist_ok=True); todo = [i for i in ids if not os.path.exists(os.path.join(a.out, i, f"{i}_{a.years}yr.db"))]; total = 0; bad = 0
    print(f"{len(todo)} cars to build, {len(ids) - len(todo)} already there; {a.jobs} at a time", flush=True)
    def run(i):
        if a.min_free_gb and shutil.disk_usage(a.out).free / 1e9 < a.min_free_gb: return i, False, "skipped: the disk is below --min-free-gb", 0
        return build_car(i, a.out, a.keep_years, ENDS5 if a.years == 5 else ENDS)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for pid, ok, msg, size in ex.map(run, todo):
            print(f"{pid:26s} {'ok ' if ok else 'FAILED'} {msg}", flush=True); total += size; bad += (not ok)
    print(f"done: {len(todo) - bad} built, {bad} failed, {total / 1e9:.1f} GB")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
