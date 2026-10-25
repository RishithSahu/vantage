"""Does the assistant name the real cause of a recorded event?  Scored on every fault the generator planted in the four 105-day vehicle histories (ground truth in data/world_*_truth.json, never read by the assistant).

    python bench/event_cause_bench.py [--worlds aster,kestrel,meridian,rhino]

Given only the time, the engine finds the event there, its causal chain, and ranks causes.  The top cause is compared with the planted one.  Also scored: a time with no recorded event (must answer that there is none), and the
chain test (asking about the assistance system's disengagement two seconds after a controller reset must lead back to the reset)."""
import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vgpt.cause.events import explain_event          # noqa: E402

# planted fault family -> (key of its time, cause the engine must name)
EXPECT = {"ecu_resets_adas": ("ts", "ecu_watchdog_reset"), "unexpected_shutdown": ("ts", "supply_voltage_loss"), "speed_conflict": ("ts", "speed_scaling_mismatch"), "ambient_glitch": ("ts", "invalid_sensor_value"),
          "can_burst": ("ts", "bus_disturbance"), "voltage_dip": ("ts", "supply_dip"), "aeb": ("ts", "obstacle_ahead")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worlds", default="aster,kestrel,meridian,rhino"); ap.add_argument("--show", action="store_true"); ap.add_argument("--no-logs", action="store_true", help="ablation: the controllers' log lines are not used (events and signals only)"); a = ap.parse_args(); tot = {}; rows = []
    for w in a.worlds.split(","):
        db = os.path.join(ROOT, "data", f"world_{w}.db"); truth = json.load(open(os.path.join(ROOT, "data", f"world_{w}_truth.json")))["faults"]; c = sqlite3.connect("file:" + db.replace("\\", "/") + "?mode=ro", uri=True)
        for fam, (key, want) in EXPECT.items():
            for f in truth.get(fam, []):
                r = explain_event(c, f[key], use_logs=not a.no_logs); top = r["causes"][0]["id"] if r and r["causes"] else None; ok = top == want
                if fam == "voltage_dip" and top == "supply_voltage_loss": ok = True
                t = tot.setdefault(fam, [0, 0]); t[0] += ok; t[1] += 1; rows.append((w, fam, ok, top, want, r["event"]["type"] if r else None))
                if a.show and not ok: print("MISS", w, fam, "got", top, "wanted", want, "event", r["event"]["type"] if r else None)
        # the chain: the disengagement that follows a controller reset leads back to it
        for f in truth.get("ecu_resets_adas", []):
            r = explain_event(c, f["adas_disengage_ts"], "ADAS_DISENGAGE", use_logs=not a.no_logs); top = r["causes"][0]["id"] if r and r["causes"] else None; t = tot.setdefault("adas_disengage_chain", [0, 0]); t[0] += top == "ecu_watchdog_reset"; t[1] += 1
        # a time with nothing recorded
        import random
        rnd = random.Random(7); lo, hi = c.execute("SELECT MIN(ts), MAX(ts) FROM events").fetchone(); none_ok = n = 0
        while n < 30:
            ts = rnd.randint(lo, hi)
            if c.execute("SELECT COUNT(*) FROM events WHERE ts BETWEEN ? AND ?", (ts - 60, ts + 60)).fetchone()[0]: continue
            n += 1; none_ok += explain_event(c, ts, use_logs=not a.no_logs) is None
        t = tot.setdefault("no_event_at_that_time", [0, 0]); t[0] += none_ok; t[1] += n
        c.close()
    print("%-26s %s" % ("fault family", "named the planted cause"))
    for k, (ok, n) in tot.items(): print("%-26s %3d / %3d  (%.0f %%)" % (k, ok, n, 100.0 * ok / max(n, 1)))
    allok = sum(v[0] for v in tot.values()); alln = sum(v[1] for v in tot.values()); print("ALL: %d / %d (%.1f %%)" % (allok, alln, 100.0 * allok / alln))


if __name__ == "__main__": main()
