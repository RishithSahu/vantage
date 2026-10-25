"""Of the 283 frozen questions the shipped adapter (frame2w) answered differently from the gold, which does a newer adapter now get right?

    python bench/meaning_eval/fixed_283.py frame2r10_unknown [frame2r11 ...] [--show 40]

The 283 are the first-round entries of gold_review.json (round-1 rulings, docs/FAILURE_ANALYSIS_2026-10-24.md): G = the gold is right and the model was wrong (a genuine error), P = the model's reading was right and the
gold wrong, B / Q / CT = both readings give an equivalent answer.  For each ruling class: how many the newer adapter answers acceptably (strict, or one of the accepted frames) and which genuine errors remain."""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
pad = lambda t: " ".join((str(t).split() + ["none"] * 8)[:8])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--show", type=int, default=30); a = ap.parse_args()
    rev = json.load(open(os.path.join(HERE, "gold_review.json"), encoding="utf-8"))
    first = [e for e in rev if "round 2" not in str(e.get("note", ""))]
    print(len(first), "first-round entries;", dict(collections.Counter(e["ruling"] for e in first)))
    for nm in a.names:
        P = {(r["set"], r["text"].strip().lower()): r for r in json.load(open(os.path.join(ROOT, "results", "meaning", nm + "_r10", "predictions.json"), encoding="utf-8"))["predictions"]}
        cnt = collections.defaultdict(lambda: [0, 0]); left = []
        for e in first:
            r = P.get((e["set"], e["text"].strip().lower()))
            if r is None: continue
            ok = r["strict"] or pad(r["pred"]) in [pad(x) for x in e["accepted"]]
            cnt[e["ruling"]][0] += 1; cnt[e["ruling"]][1] += ok
            if e["ruling"] == "G" and not ok: left.append((e["set"], e["text"], e["gold"], r["pred"]))
        print(f"\n== {nm}: " + ", ".join(f"{k}: {v[1]}/{v[0]}" for k, v in sorted(cnt.items())))
        g = cnt["G"]
        if g[0]: print(f"   genuine errors of frame2w now answered acceptably: {g[1]} of {g[0]} ({100 * g[1] / g[0]:.0f} %)")
        for s, t, gd, pr in left[:a.show]: print(f"   still wrong [{s}] {t[:70]} | gold {gd.replace(' none', '')[:40]} | got {pr.replace(' none', '')[:40]}")


if __name__ == "__main__":
    main()
