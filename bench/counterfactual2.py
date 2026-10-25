"""Counterfactual families with TWO injected causes: a tyre leak (gradual; its effect on consumption goes through low pressure, which IS a recorded driver) and a roof box (a step; NOT recorded).
Variants of the same vehicle (same seed, routes and offline weather; random draws are identical because each cause is switched off only AFTER the simulation object is constructed):
    all          both causes on
    no_tyre      tyre leak off (every wheel stays at its spec pressure)
    no_roof      roof box off
    none         neither
The truth for each cause on a window is a difference of the observed change between variants: tyre = gap(all) - gap(no_tyre); roof = gap(all) - gap(no_roof).
    python bench/counterfactual2.py --seeds 41,42,43,44,45,46"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from vgpt import datagen_events as DE, datagen_life as DL
import tools_datagen as TD
from bench.counterfactual import js

PRESET = dict(seed=0, city="london", driver="delivery", vehicle="city", stories=["tyre_leak", "roof_box"])


def build(variant, seed, days=28):
    cfg = dict(PRESET); cfg["seed"] = seed; orig = DL.Life.__init__
    def init(self, *a, **k):
        orig(self, *a, **k)
        if variant in ("no_tyre", "none"): self.cold = np.broadcast_to(self.spec[:, None], self.cold.shape).astype(self.cold.dtype).copy()      # `cold` is the actual cold pressure of each wheel per hour: held at spec = no leak
        if variant in ("no_roof", "none"): self.roof_t = 10 ** 12
    DL.Life.__init__ = init
    try: return DL.build(days=days, offline=True, verbose=False, **cfg)
    finally: DL.Life.__init__ = orig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--days", type=int, default=28); a = ap.parse_args()
    for seed in [int(x) for x in a.seeds.split(",")]:
        for variant in ("all", "no_tyre", "no_roof", "none"):
            t = time.time(); gv = build(variant, seed, a.days); out = os.path.join(HERE, "data", f"cf2_s{seed}_{variant}.db")
            for ext in ("", "-wal", "-shm"):
                if os.path.exists(out + ext): os.remove(out + ext)
            DE.seed(out, gv); json.dump(gv.truth, open(out[:-3] + "_truth.json", "w"), default=js)
            print(f"cf2_s{seed}_{variant}: {gv.truth['trips']} trips, {gv.truth['wh_km']} Wh/km, stories {dict(gv.st)}, {time.time() - t:.0f}s")


if __name__ == "__main__": main()
