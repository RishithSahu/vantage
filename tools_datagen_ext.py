"""DEV tool: add the extended signals and fault scenarios (vgpt/datagen_ext.py) to a generated database - additively.
  python tools_datagen_ext.py data/gen_courier.db --out data/gen_courier_ext.db      copy, then extend the copy (the original is never touched)
  python tools_datagen_ext.py data/gen_courier.db --in-place --seed 3                extend in place (adds xsig_1s / xsig_1m / meta rows only)
  python tools_datagen_ext.py data/gen_courier.db --out x.db --no-scenarios          physical signals only, no injected faults
Ground truth of the injected faults: <out without .db>_faults.json"""
import argparse, os, shutil, sqlite3, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vgpt import datagen_ext as X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db"); ap.add_argument("--out"); ap.add_argument("--in-place", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-scenarios", action="store_true")
    a = ap.parse_args()
    if not a.out and not a.in_place: ap.error("give --out FILE or --in-place")
    target = a.db
    if a.out:
        src = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); dst = sqlite3.connect(a.out); src.backup(dst); dst.close(); src.close(); target = a.out    # consistent copy even with a live WAL
    t = time.time(); truth = X.extend(target, seed=a.seed, scenarios=not a.no_scenarios)
    print(f"extended {target} in {time.time() - t:.1f}s; {len(truth['scenarios'])} fault scenario(s): {', '.join(s['name'] for s in truth['scenarios']) or 'none'}")


if __name__ == "__main__": main()
