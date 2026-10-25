"""A short patch round: every row of the newest hand-written families plus a random share of the rest (a replay, so what was learned is not forgotten).

    python training/meaning/make_subset.py --src training/meaning/data/frame2_r3 --out training/meaning/data/frame2_r3s --new v4_natural,v5_natural,v6_natural,v7_natural --replay 0.35 --seed 3

The validation and composition files are copied unchanged.  Nothing frozen is touched."""
import argparse
import io
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--new", required=True); ap.add_argument("--replay", type=float, default=0.35); ap.add_argument("--seed", type=int, default=3); a = ap.parse_args()
    src = a.src if os.path.isabs(a.src) else os.path.join(ROOT, a.src); out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out); new = set(a.new.split(",")); r = random.Random(a.seed); os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in io.open(os.path.join(src, "train.jsonl"), encoding="utf-8")]; keep = [x for x in rows if x["src"] in new or r.random() < a.replay]; r.shuffle(keep)
    with io.open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for x in keep: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for name in ("val.jsonl", "test_comp.jsonl", "manifest.json"):
        if os.path.exists(os.path.join(src, name)): shutil.copy(os.path.join(src, name), os.path.join(out, name))
    print(f"{len(keep)} of {len(rows)} rows ({sum(1 for x in rows if x['src'] in new)} new) -> {out}")


if __name__ == "__main__":
    main()
