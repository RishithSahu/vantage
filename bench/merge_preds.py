"""Merge the predictions of several evaluation runs of ONE adapter (different sets) into results/meaning/<out>_r10/predictions.json.

    python bench/merge_preds.py frame2w_all frame2w frame2w_v5 frame2w_v6 frame2w_v2sets"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, parts = sys.argv[1], sys.argv[2:]; preds = []; summary = {}; seen = set()
for nm in parts:
    d = json.load(open(os.path.join(ROOT, "results", "meaning", nm + "_r10", "predictions.json"), encoding="utf-8"))
    for r in d["predictions"]:
        k = (r["set"], r["id"])
        if k not in seen: seen.add(k); preds.append(r)
    summary.update(d["summary"])
os.makedirs(os.path.join(ROOT, "results", "meaning", out + "_r10"), exist_ok=True)
json.dump({"adapter": "merged:" + ",".join(parts), "summary": summary, "predictions": preds}, open(os.path.join(ROOT, "results", "meaning", out + "_r10", "predictions.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(preds), "predictions ->", out)
