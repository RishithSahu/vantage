"""Does the assistant notice a fault that was put into a simulated car?

    python bench/anomaly_battery.py [--cars aster_ev kestrel_petrol rhino_diesel] [--minutes 25] [--out results/anomaly_battery.json]

For each fault of vgpt/live/car.py FAULTS and each car: the live simulator drives, the fault is injected, the runtime writes the database, and the assistant is asked (by frame, no language model) the questions a person asks
with the warning lamp on: "is anything unusual", "is anything wrong with the car", "which codes are active", and the component question that belongs to the fault.  The answer must contain the words that name the fault
(KEYWORDS: any of them) - the test is whether the finding reaches the person, not whether the wording is pretty.  A fault the car cannot show (camera on a car without one) is skipped."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, HERE)
os.environ["VGPT_FRAME"] = "2"
import test_live as TL                                  # noqa: E402
import ask_frame as AF                                  # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain          # noqa: E402
from vgpt.vehicle import lake as LK                     # noqa: E402

KEYWORDS = {"coolant_pump": ["coolant", "pump", "flow", "overheat", "over-temperature", "overtemperature"], "tyre_leak": ["tyre", "tire", "rear left"], "camera_blocked": ["camera"], "weak_12v": ["12 v", "12v", "voltage", "p0562"],
            "speed_conflict": ["speed", "sensor", "conflict", "source"], "ecu_reset": ["reset", "ecu", "controller"], "brownout": ["12 v", "voltage", "dip", "brown"], "shutdown": ["shutdown", "shut down", "power", "voltage", "gap"],
            "can_burst": ["bus", "can", "dropout", "silent", "communication"], "ambient_glitch": ["outside", "ambient", "temperature"], "aeb": ["emergency", "brak", "aeb"], "range_glitch": ["range"], "overheat": ["temperature", "heater", "overheat", "pack", "hot"]}
QUESTIONS = [("is anything unusual", "anomaly none"), ("is anything wrong with the car", "health vehicle_health"), ("which diagnostic codes are active", "health diagnostic_status"), ("are there any warnings", "events any_warning op=exists")]
NEEDS = {"camera_blocked": "camera", "tyre_leak": "tyres"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cars", nargs="+", default=["aster_ev", "kestrel_petrol", "rhino_diesel"]); ap.add_argument("--minutes", type=int, default=25); ap.add_argument("--out", default="results/anomaly_battery.json"); a = ap.parse_args()
    rows = []
    for car_id in a.cars:
        for fault in KEYWORDS:
            db, rt, car, sim = TL.rig(car_id, tag=f"_ab_{fault}"); car.start_trip(); TL.run(sim, 240)
            try:
                if fault == "tyre_leak": sim.inject(fault, wheel=2, kpa_min=8.0)
                else: sim.inject(fault)
            except Exception as e:
                rt.close(); rows.append({"car": car_id, "fault": fault, "skipped": f"{type(e).__name__}"}); continue
            TL.run(sim, a.minutes * 60); rt.publish(); rt.close()
            LK.adopt_path(db); b = MeaningBrain(db, AF.Null(), now=AF.last_ts(db), tau=0.0); found = {}; answers = {}
            for text, spec in QUESTIONS:
                b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b._last_data = None
                try: ans = str(b.resolver.answer(text, AF.frame_of(spec)).get("answer", ""))
                except Exception as e: ans = "EXC " + type(e).__name__
                answers[text] = ans[:300]; found[text] = any(k in ans.lower() for k in KEYWORDS[fault])
            ev = sorted({e[1] for e in TL.events(db)})
            rows.append({"car": car_id, "fault": fault, "events": ev, "found": found, "any_found": any(found.values()), "answers": answers})
            print(f"{car_id:16s} {fault:15s} found by {sum(found.values())}/{len(found)} questions   events: {','.join(ev)[:70]}", flush=True)
    done = [r for r in rows if "found" in r]; print(f"\n{sum(r['any_found'] for r in done)}/{len(done)} faults reached the person by at least one question; {sum(sum(r['found'].values()) for r in done)}/{sum(len(r['found']) for r in done)} answers")
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
