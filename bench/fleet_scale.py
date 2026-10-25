"""Many vehicles into one machine: N simulated cars, each driving its own model and streaming its buses in real time over its own pipe, into N receiver processes started by the FleetHub.

    python bench/fleet_scale.py [--cars 1,2,4,8,16] [--seconds 60] [--out results/scaling/laptop_fleet.json]

Per step: how many receivers recognised their car (with no per-car setup), records stored per second in total, drops, resident memory of all receivers together, the CPU the receivers used (the cars' own simulation is not counted:
on a real fleet that work is on the cars), and then one receiver is killed to show that the others do not notice and the hub restarts it.
The car models rotate through the catalogue, so a step with 16 cars has 16 different wirings (CAN, CAN-FD, J1939, OBD-II, SOME/IP, Android VHAL, logs, GNSS)."""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")


def proc_tree_stats(pids):
    import psutil
    rss = cpu = 0.0
    for pid in pids:
        try:
            p = psutil.Process(pid); procs = [p] + p.children(recursive=True)
            for q in procs: rss += q.memory_info().rss; cpu += sum(q.cpu_times()[:2])
        except psutil.Error: pass
    return rss / 2 ** 20, cpu


def step(n, seconds, car_ids):
    from vgpt.car import models, transport as CT, writer as CW
    from vgpt.car.car import Car
    from vgpt.scale.fleet import FleetHub
    home = tempfile.mkdtemp(prefix="vgpt_fleet_"); hub = FleetHub(home, slots=n, name="fb%d_%d" % (n, os.getpid())); cars = []; t0 = time.time()
    try:
        for i in range(n):
            m = models.find(car_ids[i % len(car_ids)]); fd = CT.connect(hub.endpoint(i), timeout=60); c = Car(m["dir"], CW.Writer(fd), speed=1.0, seed=100 + i); c.start(); cars.append(c)
        time.sleep(seconds)
        rows = hub.status(); pids = [r["pid"] for r in rows if r["pid"]]; rss, cpu1 = proc_tree_stats(pids); time.sleep(5.0); rows2 = hub.status(); rss2, cpu2 = proc_tree_stats(pids)
        rec1 = sum(r["records"] for r in rows); rec2 = sum(r["records"] for r in rows2)
        out = {"cars": n, "models": sorted({c.profile.id for c in cars}), "recognised": sum(1 for r in rows2 if r["state"] == "running"), "records_per_s": round((rec2 - rec1) / 5.0, 1), "records": rec2,
               "dropped": sum(r["dropped"] for r in rows2), "receiver_rss_mb": round(rss2), "rss_per_receiver_mb": round(rss2 / max(n, 1)), "receiver_cpu_pct_of_one_core": round(100.0 * (cpu2 - cpu1) / 5.0, 1), "seconds": round(time.time() - t0)}
        victim = n - 1; before = {r["slot"]: r["records"] for r in rows2}; hub.kill(victim); time.sleep(6.0); rows3 = hub.status()
        out["after_kill"] = {"victim_restarts": rows3[victim]["restarts"], "others_still_running": sum(1 for r in rows3 if r["slot"] != victim and r["state"] == "running"), "others_kept_growing": sum(1 for r in rows3 if r["slot"] != victim and r["records"] > before[r["slot"]])}
        return out
    finally:
        for c in cars:
            try: c.stop(join=False)
            except Exception: pass
        hub.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cars", default="1,2,4,8,16"); ap.add_argument("--seconds", type=float, default=60.0); ap.add_argument("--out"); a = ap.parse_args()
    ids = ["aster_ev", "rhino_diesel", "meridian_ev", "tesla_model_3", "heavy_truck_j1939", "kestrel_petrol", "aaos_reference_ev", "byd_atto_3", "mg_zs_ev", "toyota_prius_hybrid", "volvo_v60_2015", "chevrolet_bolt_ev", "hongqi_hs5", "hyundai_elantra_ccan", "mercedes_e350_w212", "bmw_3_series_e9x"]
    res = []
    for n in [int(x) for x in a.cars.split(",")]:
        r = step(n, a.seconds, ids); res.append(r); print(json.dumps(r), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump({"mode": "fleet", "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"), "cpus": os.cpu_count(), "result": res, "time": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()
