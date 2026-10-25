"""Drive a simulated car, inject a fault, and keep the database the live runtime wrote, so the assistant can be asked about it.

    python bench/live_scenario.py CAR FAULT OUT.db [--minutes 25] [--wheel 2]

CAR: aster_ev | kestrel_petrol | rhino_diesel ...; FAULT: tyre_leak | coolant_pump | weak_12v | ecu_reset | shutdown | none ..."""
import argparse, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_live as TL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("car"); ap.add_argument("fault"); ap.add_argument("out"); ap.add_argument("--minutes", type=int, default=25); ap.add_argument("--wheel", type=int, default=2); a = ap.parse_args()
    db, rt, car, sim = TL.rig(a.car, tag="_scn_" + a.fault); car.start_trip(); TL.run(sim, 240)
    if a.fault == "tyre_leak": sim.inject("tyre_leak", wheel=a.wheel, kpa_min=8.0)
    elif a.fault != "none": sim.inject(a.fault)
    TL.run(sim, a.minutes * 60); rt.publish(); rt.close(); shutil.copy(db, a.out)
    ev = [e[1] for e in TL.events(a.out)]; import collections; print(a.out, dict(collections.Counter(ev)))


if __name__ == "__main__":
    main()
