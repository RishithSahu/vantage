"""Re-score stored predictions (results/meaning/<name>_r10/predictions.json) with the current gold_review.json, so adapters are compared under the same rulings.

    python bench/meaning_eval/rescore.py frame2w frame2r10_unknown frame2r10_hint [--sets a b ...]
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
pad = lambda t: " ".join((str(t).split() + ["none"] * 8)[:8])
FROZEN = ["novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5", "platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "slots_v2", "followup_v2", "digest_v1", "catalog_v1",
          "followup_v3", "basic_v2", "hist_v2", "icevocab_petrol_v1", "icevocab_diesel_v1", "followup_fresh_v4", "longwrap_v1"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--sets", nargs="*"); a = ap.parse_args()
    review = {}
    for e in json.load(open(os.path.join(HERE, "gold_review.json"), encoding="utf-8")):
        for key in ((e["set"], e["text"].strip().lower()), ("*", e["text"].strip().lower())):
            review.setdefault(key, []).extend(x for x in (pad(y) for y in e["accepted"]) if x not in review.get(key, []))
    tab = {}
    for nm in a.names:
        P = json.load(open(os.path.join(ROOT, "results", "meaning", nm + "_r10", "predictions.json"), encoding="utf-8"))["predictions"]
        c = collections.defaultdict(lambda: [0, 0, 0])
        for r in P:
            if a.sets and r["set"] not in a.sets: continue
            acc = review.get((r["set"], r["text"].strip().lower())) or review.get(("*", r["text"].strip().lower()))
            ok = r["strict"] or (acc is not None and pad(r["pred"]) in acc)
            for k in (r["set"], "~FROZEN" if r["set"] in FROZEN else "~NEW"):
                c[k][0] += 1; c[k][1] += r["strict"]; c[k][2] += ok
        tab[nm] = c
    sets = sorted({k for c in tab.values() for k in c}, key=lambda k: (k.startswith("~"), k))
    print(f"{'set':22s}{'n':>6s}  " + "  ".join(f"{nm[:16]:>16s}" for nm in a.names))
    for k in sets:
        n = max(tab[nm][k][0] for nm in a.names if k in tab[nm])
        print(f"{k:22s}{n:6d}  " + "  ".join(f"{100 * tab[nm][k][2] / tab[nm][k][0]:7.1f}/{100 * tab[nm][k][1] / tab[nm][k][0]:5.1f}  " if k in tab[nm] else " " * 16 for nm in a.names))
    print("(reviewed/strict, percent)")


if __name__ == "__main__":
    main()
