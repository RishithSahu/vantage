"""DEV tool: record a high-rate replay of the vehicle buses (recorded-CAN-replay route allowed by the brief). Usage: python tools_record.py [can_hz]
Writes data/replay.pkl: one pickled (ts, bus, payload) per message, streamable. 10 min around the day-26 fast-charge/overheat story."""
import pickle, sys, time
from vgpt import sim
hz = int(sys.argv[1]) if len(sys.argv) > 1 else 100
veh = sim.build(); sim.save_truth(veh, "data/truth.json")
start = 26 * 86400 + 13 * 3600 + 6 * 60; secs = 600; n = 0
with open("data/replay.pkl", "wb") as f:
    for m in sim.live_messages(veh, start, secs, can_hz=hz): pickle.dump(m, f, protocol=4); n += 1
print(f"recorded {n:,} messages ({n/secs:,.0f} msg/s over {secs}s of vehicle time; CAN {hz} Hz x 4 ids)")
