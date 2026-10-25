"""Evaluate a meaning-frame adapter on every frozen set and the sets written for round 10, with the car type told to the model and the reviewed labels applied.

    python bench/meaning_eval/eval_r10.py --adapter models/meaning-lora-frame2w.gguf --name frame2w [--vehicle-hint] [--default-vehicle ev|none] [--sets ...]

Scoring (stated here, applied identically to every adapter):
  strict     the predicted words start with the gold words (6-word gold of the older sets: the first six words)
  reviewed   strict OR the prediction (padded to 8 words) is one of the `accepted` frames of bench/meaning_eval/gold_review.json for that question - the hand rulings of docs/FAILURE_ANALYSIS_2026-10-24.md
             (a gold that was wrong, or two readings that print an equivalent answer). Rows without a ruling are scored strictly.
Car type: a set whose 4th column is a car type (ice_v1 and the new sets) uses it; icevocab_petrol_v1 / icevocab_diesel_v1 use petrol / diesel; everything else uses --default-vehicle (ev).
Output: results/meaning/<name>_r10/predictions.json and a table on stdout."""
import argparse
import collections
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")

FROZEN = ["novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5", "platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "slots_v2", "followup_v2", "digest_v1", "catalog_v1",
          "followup_v3", "basic_v2", "hist_v2", "icevocab_petrol_v1", "icevocab_diesel_v1", "followup_fresh_v4", "longwrap_v1"]
NEW = ["sealed_v4", "sealed_v5", "ice_v1", "trend_v1", "anomaly_v1", "anomaly_v2", "trend_v2", "sealed_v6", "sealed_v7", "wording_v1", "holdout_v10"]
VEHICLES = {"ev", "petrol", "diesel", "hybrid", "ice"}
pad = lambda t: " ".join((str(t).split() + ["none"] * 8)[:8])
ok_prefix = lambda p, g: p.split()[:len(g.split())] == g.split()


def load(name):
    path = os.path.join(HERE, name + ".tsv")
    if not os.path.exists(path): return []
    rows = []
    for ln in io.open(path, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t")
        if name.startswith("followup") and len(p) >= 4: rows.append({"id": p[0], "prev": p[1], "text": p[2], "gold": p[3], "tag": p[4] if len(p) > 4 else ""})
        elif not name.startswith("followup") and len(p) >= 3: rows.append({"id": p[0], "prev": None, "text": p[1], "gold": p[2], "tag": p[3] if len(p) > 3 else ""})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--name", required=True); ap.add_argument("--sets", nargs="*", default=FROZEN + NEW); ap.add_argument("--vehicle-hint", action="store_true")
    ap.add_argument("--default-vehicle", default="ev"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-vehicle-tag", action="store_true", help="ignore the car type of the sets (the ablation of the car-type line)"); a = ap.parse_args()
    if a.vehicle_hint: os.environ["VGPT_VEHICLE_HINT"] = "1"
    from vgpt.meaning import interface
    ad = a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)
    fm = interface.FrameModel(None if a.adapter == "none" else ad, threads=a.threads)
    review = {}
    rp = os.path.join(HERE, "gold_review.json")
    if os.path.exists(rp):
        for e in json.load(open(rp, encoding="utf-8")):
            for key in ((e["set"], e["text"].strip().lower()), ("*", e["text"].strip().lower())): review.setdefault(key, []).extend(x for x in (pad(y) for y in e["accepted"]) if x not in review.get(key, []))                    # every ruling of a question adds its acceptable frames
    out, summary, t0 = [], {}, time.time()
    for name in a.sets:
        rows = load(name)
        if not rows: continue
        if a.limit: rows = rows[:a.limit]
        n = s = rv = 0; by = collections.Counter(); byn = collections.Counter()
        for r in rows:
            veh = r["tag"].split(":")[0] if r["tag"].split(":")[0] in VEHICLES else ("petrol" if name == "icevocab_petrol_v1" else "diesel" if name == "icevocab_diesel_v1" else (None if a.default_vehicle == "none" else a.default_vehicle))
            if a.no_vehicle_tag: veh = None
            fm.set_vehicle(veh)
            try: f, info = fm.parse(r["text"], prev=r["prev"])
            except Exception as e: f, info = None, {"raw": "EXC " + type(e).__name__, "pmin": 0.0, "ms": 0}
            raw = info["raw"]; strict = ok_prefix(raw, r["gold"]); acc = review.get((name, r["text"].strip().lower())) or review.get(("*", r["text"].strip().lower()))
            reviewed = strict or (acc is not None and pad(raw) in acc)
            n += 1; s += strict; rv += reviewed; fam = r["tag"].split(":")[-1] if r["tag"] else ""; by[fam] += reviewed; byn[fam] += 1
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"], "pred": raw, "strict": strict, "reviewed": reviewed, "pmin": info.get("pmin"), "vehicle": veh, "family": fam})
        summary[name] = {"n": n, "strict": round(s / n, 4), "reviewed": round(rv / n, 4), "by_family": {k: f"{by[k]}/{byn[k]}" for k in sorted(byn) if k}}
        print(f"{name:22s} n={n:4d} strict {100 * s / n:5.1f} %  reviewed {100 * rv / n:5.1f} %   [{time.time() - t0:.0f}s]", flush=True)
    fz = [v for k, v in summary.items() if k in FROZEN]; N = sum(v["n"] for v in fz)
    if fz: print(f"\nFROZEN pooled ({len(fz)} sets, {N} questions): strict {100 * sum(v['strict'] * v['n'] for v in fz) / N:.2f} %   reviewed {100 * sum(v['reviewed'] * v['n'] for v in fz) / N:.2f} %")
    nw = [v for k, v in summary.items() if k in NEW]; M = sum(v["n"] for v in nw)
    if nw: print(f"NEW pooled ({len(nw)} sets, {M} questions): strict {100 * sum(v['strict'] * v['n'] for v in nw) / M:.2f} %   reviewed {100 * sum(v['reviewed'] * v['n'] for v in nw) / M:.2f} %")
    d = os.path.join(ROOT, "results", "meaning", a.name + "_r10"); os.makedirs(d, exist_ok=True)
    json.dump({"adapter": a.adapter, "vehicle_hint": a.vehicle_hint, "default_vehicle": a.default_vehicle, "summary": summary, "predictions": out}, io.open(os.path.join(d, "predictions.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", d)


if __name__ == "__main__":
    main()
