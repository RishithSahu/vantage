"""The live side of the platform, in one process, from the command line (the product is fully offline: this is the only mode left).

  python tools_live.py run --db data/live_demo.db --from-db data/world_aster.db --minutes 40 [--fault shutdown@600 ...]
        the virtual car and the runtime together, as fast as the machine allows (no sleeping): a scripted drive with faults, then a summary of what the database holds.

The version that also listened on UDP / TCP / MQTT / VISS and sent the virtual car's traffic over UDP is in vehiclegpt2_retired_networking_2026-10-21/."""
import argparse
import os
import shutil
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)


def cmd_run(a):
    from vgpt.live.runtime import Runtime
    from vgpt.live.sim import Simulation, car_from_db
    if a.from_db:
        if os.path.exists(a.db): os.remove(a.db)
        shutil.copyfile(a.from_db, a.db)
    rt = Runtime(a.db); car = car_from_db(a.db, seed=a.seed); sim = Simulation(rt, car)
    plan = sorted((int(f.partition("@")[2] or 0), f.partition("@")[0]) for f in (a.fault or []))
    if a.skip_to_hour is not None:
        day = int((car.t + car.tz) // 86400); sim.skip_to(day * 86400 - car.tz + a.skip_to_hour * 3600 + (86400 if a.skip_to_hour * 3600 < (car.t + car.tz) % 86400 else 0))
    car.start_trip(); t0 = time.time(); n = int(a.minutes * 60); start = car.t; fired = set()
    for i in range(n):
        sim.step()
        for k, (at, name) in enumerate(plan):
            if k not in fired and i >= at: sim.inject(name); fired.add(k); print(f"  +{at}s inject {name}")
        if car.arrived and not a.keep_going: break
    rt.publish(); s = rt.stats(); dt = time.time() - t0; rt.close()
    c = sqlite3.connect(a.db)
    print(f"{int(car.t - start)} simulated seconds in {dt:.1f} s ({(car.t - start) / dt:.0f}x), {s['msgs']} messages ({s['msgs'] / dt:.0f}/s), decode {s['decode_us']} us/message")
    for q in ("SELECT COUNT(*), MIN(ts), MAX(ts) FROM sig_1s WHERE ts>?", "SELECT COUNT(*) FROM events WHERE ts>?", "SELECT COUNT(*) FROM trips WHERE start>?"): print("  ", q.split('FROM')[1].split('WHERE')[0].strip(), c.execute(q, (start,)).fetchone())
    print("   events:", c.execute("SELECT type, COUNT(*) FROM events WHERE ts>? GROUP BY type ORDER BY 2 DESC", (start,)).fetchall())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--db", required=True); p.add_argument("--from-db"); p.add_argument("--minutes", type=float, default=40); p.add_argument("--seed", type=int, default=7); p.add_argument("--fault", action="append", help="name or name@seconds-into-the-run")
    p.add_argument("--skip-to-hour", type=float); p.add_argument("--keep-going", action="store_true")
    a = ap.parse_args(); {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
