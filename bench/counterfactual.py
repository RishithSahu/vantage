"""Counterfactual vehicle pairs: ground truth for 'why did it change' questions.

The generator injects causes ('stories': roof box, cooling pump, ...). For one configuration this builds the vehicle twice with the same seed, the same routes and the same (offline, synthetic) weather: once as generated, once with ONLY the roof-box effect switched
off after the simulation object is constructed (every random draw is identical, so the two histories differ only by that cause). The difference between the two histories is the true size of the injected cause; what remains of a change in the twin is the
effect of everything else (weather, speed mix, tyre pressure, ...). A causal explanation can then be scored against a decomposition it never saw.

    python bench/counterfactual.py --preset roadwarrior            # writes data/cf_roadwarrior_with.db / _without.db (+ truth json)
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from vgpt import datagen as DG, datagen_events as DE, datagen_life as DL
import tools_datagen as TD


def js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, set): return sorted(o)
    raise TypeError(type(o))


def build(preset, without, days=28, seed=None):
    cfg = dict(TD.PRESETS[preset]); cfg["seed"] = cfg["seed"] if seed is None else seed; orig = DL.Life.__init__
    def init(self, *a, **k):
        orig(self, *a, **k)
        if without: self.roof_t = 10 ** 12                                   # same draws as the original run, but the roof box never goes on
    DL.Life.__init__ = init
    try: return DL.build(days=days, offline=True, verbose=False, **cfg)
    finally: DL.Life.__init__ = orig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preset", required=True, choices=["cold", "roadwarrior"]); ap.add_argument("--days", type=int, default=28); ap.add_argument("--seeds", default="", help="comma list of seeds (default: the preset's own)"); a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [None]
    for seed in seeds:
        for without in (False, True):
            t = time.time(); gv = build(a.preset, without, a.days, seed); name = f"cf_{a.preset}{'' if seed is None else '_s%d' % seed}_{'without' if without else 'with'}"; out = os.path.join(HERE, "data", name + ".db")
            for ext in ("", "-wal", "-shm"):
                if os.path.exists(out + ext): os.remove(out + ext)
            DE.seed(out, gv); json.dump(gv.truth, open(out[:-3] + "_truth.json", "w"), default=js)
            print(f"{name}: {gv.truth['trips']} trips, {gv.truth['total_km']} km, {gv.truth['wh_km']} Wh/km, stories {dict((k, v) for k, v in gv.st.items())}, {time.time() - t:.0f}s")

if __name__ == "__main__": main()
