"""DEV tool: generate a random but realistic vehicle history from REAL data (weather, driving traces, accelerometer) - see vgpt/datagen.py.
  python tools_datagen.py --list                          show cities / drivers / vehicles / stories
  python tools_datagen.py --preset demo                   the default demo vehicle -> data/gen_demo.db
  python tools_datagen.py --seed 7                        random (but reproducible) vehicle, driver, city and faults from the seed
  python tools_datagen.py --seed 7 --city oslo --driver delivery --vehicle city --days 28 --end 2026-09-20 --stories tyre_leak,cooling_pump
Output: data/gen_<name>.db (the on-device schema + extension tables) and data/gen_<name>_truth.json (ground truth computed from the generator's arrays)."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from vgpt import datagen as DG, datagen_life as DL, datagen_events as DE

PRESETS = {"demo": dict(seed=7, city="bengaluru", driver="commuter", vehicle="family", end="2026-08-25", work_km=24, stories=["tyre_leak", "cooling_pump", "dirty_windscreen", "roadworks", "harsh_week"]),
           "cold": dict(seed=21, city="oslo", driver="commuter", vehicle="compact", stories=["weak_12v", "roof_box"], end="2026-01-31"),
           "courier": dict(seed=33, city="london", driver="delivery", vehicle="city", stories=["cooling_pump", "tyre_leak"]),
           "roadwarrior": dict(seed=5, city="graz", driver="long_distance", vehicle="suv", stories=["roadworks", "roof_box"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--preset", choices=sorted(PRESETS)); ap.add_argument("--seed", type=int); ap.add_argument("--city", choices=sorted(DG.CITIES))
    ap.add_argument("--driver", choices=sorted(DG.DRIVERS)); ap.add_argument("--vehicle", choices=sorted(DG.VEHICLES)); ap.add_argument("--days", type=int, default=28); ap.add_argument("--end", help="last day, YYYY-MM-DD (default: 8 days ago)")
    ap.add_argument("--stories", help="comma list or 'none' (default: random from the seed)"); ap.add_argument("--offline", action="store_true", help="synthetic weather instead of downloading Open-Meteo data"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.list:
        print("cities:   ", ", ".join(f"{k} ({v['label']})" for k, v in DG.CITIES.items())); print("drivers:  ", "; ".join(f"{k}: {v['desc']}" for k, v in DG.DRIVERS.items()))
        print("vehicles: ", ", ".join(f"{k} ({v['label']})" for k, v in DG.VEHICLES.items())); print("stories:  ", ", ".join(DG.STORIES)); print("presets:  ", ", ".join(PRESETS)); return
    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    if a.seed is not None: cfg["seed"] = a.seed
    cfg.setdefault("seed", int(time.time()) % 100000); r = np.random.default_rng(cfg["seed"] + 991)                       # random profile choices come from the seed
    cfg.setdefault("city", a.city or str(r.choice(sorted(DG.CITIES)))); cfg.setdefault("driver", a.driver or str(r.choice(sorted(DG.DRIVERS)))); cfg.setdefault("vehicle", a.vehicle or str(r.choice(sorted(DG.VEHICLES))))
    for k, v in (("city", a.city), ("driver", a.driver), ("vehicle", a.vehicle)):
        if v: cfg[k] = v
    if a.stories is not None: cfg["stories"] = [] if a.stories == "none" else a.stories.split(",")
    if a.end: cfg["end"] = a.end
    name = a.preset or f"{cfg['city']}_{cfg['driver']}_{cfg['seed']}"; out = a.out or os.path.join(HERE, "data", f"gen_{name}.db")
    print(f"generating '{name}' (seed {cfg['seed']}) ..."); t = time.time()
    gv = DL.build(days=a.days, offline=a.offline, **{k: v for k, v in cfg.items()})
    DE.seed(out, gv); tj = out[:-3] + "_truth.json"
    def js(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (set,)): return sorted(o)
        raise TypeError(type(o))
    json.dump(gv.truth, open(tj, "w"), indent=1, default=js)
    tr = gv.truth; print(f"done in {time.time() - t:.1f}s -> {out} ({os.path.getsize(out) / 1e6:.1f} MB) + {os.path.basename(tj)}")
    print(f"  {tr['trips']} trips, {tr['total_km']} km, {tr['total_kwh']} kWh ({tr['wh_km']} Wh/km), odometer {tr['odo_start']:.0f} -> {tr['odo_end']:.0f} km, stories: {', '.join(gv.st) or 'none'}")


if __name__ == "__main__": main()
