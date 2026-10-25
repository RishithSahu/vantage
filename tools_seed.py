"""DEV tool: build the simulated 28-day vehicle, seed the on-device DB, write ground truth. Usage: python tools_seed.py"""
import os, sys, time, json
from vgpt import sim, store
t = time.time(); veh = sim.build(); print("sim %.1fs" % (time.time() - t))
store.seed_from_sim("data/vehicle.db", veh); sim.save_truth(veh, "data/truth.json")
print("seeded %.1fs, db %.1f MB" % (time.time() - t, os.path.getsize("data/vehicle.db") / 1e6))
c = store.open_ro("data/vehicle.db")
for tb in ("sig_1s", "sig_1m", "trips", "events", "event_ctx", "logs"): print(f"  {tb:10s} {c.execute(f'select count(*) from {tb}').fetchone()[0]:>9,}")
print("  ctx bytes", c.execute("select sum(length(blob)) from event_ctx").fetchone()[0])
