"""Evaluate a meaning-frame adapter on the frozen evaluation sets.

    python bench/meaning_eval/eval_frames.py --adapter models/meaning-lora-X.gguf --name X --sets novel_v1 messy_v1 user_v1 cs5 [--no-hints]

Reports (all with explicit denominators): exact six-slot frame, act, concept, act+concept, op (where the gold op is not 'none'), false execution (an answerable frame for a request that must not
be executed), false abstention (a non-executing frame for an answerable request), calibration (AUROC of the lowest token probability for exact-correct vs wrong), and latency.
Writes every prediction to results/meaning/<name>/predictions.json so errors can be read."""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import evalio                                                      # noqa: E402
from vgpt.meaning.frame import Frame                               # noqa: E402

NONEXEC = {"social", "offtopic", "unsupported", "control", "unsafe", "ambiguous"}


def cs5_rows():
    """CS5 end-to-end questions with their own semantic gold (a different author); scored on act-set + concept (+ condition pair for comparisons)."""
    ent = {"driving_range": "range_estimate", "fuel_efficiency": "wh_per_km", "energy_consumption": "wh_per_km", "adas_disengagement": "adas_disengagement", "battery_overtemperature_warning": "battery_overheat_warning",
           "vehicle_speed": "speed", "diagnostic_record": "diagnostic_code", "camera_blockage": "camera_blocked", "charging_session": "charging_session"}
    ops = {"explain_trend": {"why", "trend"}, "locate_largest_drop": {"trend"}, "list_events": {"events"}, "event_context": {"context"}, "compare_aggregates": {"compare"}, "event_existence": {"events"},
           "latest_event": {"events"}, "current_value": {"now", "estimate"}, "aggregate": {"stat", "extreme"}, "series": {"trend"}, "insufficient_data": {"compare", "stat", "trend", "events"},
           "clarify": {"ambiguous"}, "refuse_control": {"control", "unsafe"}, "current_values": {"now", "health"}}
    rows = []
    for line in open(os.path.join(ROOT, "data", "case_study5_e2e", "questions.jsonl"), encoding="utf-8"):
        q = json.loads(line); s = q["expected_semantics"]; o = s.get("operation"); e = s.get("entity")
        acts = ops.get(o)
        if acts is None: continue
        concept = ent.get(e) if o not in ("clarify", "refuse_control") else "none"
        if concept is None and o not in ("clarify", "refuse_control"): continue
        rows.append({"id": q["id"], "text": q["question"], "acts": acts, "concept": concept, "conds": set(s.get("conditions") or []), "tag": q["family"]})
    return rows


def auroc(pos, neg):
    if not pos or not neg: return None
    n = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg); return round(n / (len(pos) * len(neg)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--name", required=True); ap.add_argument("--sets", nargs="+", default=["novel_v1", "messy_v1", "user_v1", "cs5"])
    ap.add_argument("--no-hints", action="store_true"); ap.add_argument("--base", default=None); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    from vgpt.meaning import interface
    fm = interface.FrameModel(None if a.adapter == "none" else os.path.join(ROOT, a.adapter) if not os.path.isabs(a.adapter) else a.adapter, base=a.base or interface.BASE, threads=a.threads, use_hints=not a.no_hints)
    preds, summary = [], {}
    for name in a.sets:
        if name == "cs5":
            rows = cs5_rows(); ok = 0; n = 0; per = collections.Counter(); per_n = collections.Counter()
            for r in rows:
                f, info = fm.parse(r["text"]); good = bool(f) and f.act in r["acts"] and (r["concept"] == "none" or f.concept == r["concept"])
                if good and r["conds"] and f.act == "compare": good = {f.cond, f.vs} == r["conds"] or r["conds"] <= {f.cond, f.vs}
                ok += good; n += 1; per[r["tag"]] += good; per_n[r["tag"]] += 1
                preds.append({"set": "cs5", "id": r["id"], "text": r["text"], "pred": f.to_text() if f else None, "raw": info["raw"], "ok": bool(good), "pmin": info["pmin"], "ms": info["ms"]})
            summary["cs5"] = {"n": n, "act_and_concept": round(ok / n, 3), "by_family": {k: f"{per[k]}/{per_n[k]}" for k in sorted(per_n)}}
            continue
        rows = evalio.load_tsv(os.path.join(HERE, name + ".tsv")); m = collections.Counter(); by_act = collections.defaultdict(lambda: [0, 0]); by_tag = collections.defaultdict(lambda: [0, 0])
        good_p, bad_p, lat = [], [], []
        for r in rows:
            f, info = fm.parse(r["text"]); g = r["gold"]; lat.append(info["ms"])
            exact = bool(f) and f == g
            m["n"] += 1; m["parsed"] += bool(f); m["exact"] += exact
            if f:
                m["act"] += f.act == g.act; m["concept"] += f.concept == g.concept; m["act_concept"] += (f.act == g.act and f.concept == g.concept)
                if g.op != "none": m["op_n"] += 1; m["op"] += f.op == g.op
                if g.act in NONEXEC: m["nonexec_n"] += 1; m["false_exec"] += f.act not in NONEXEC; m["nonexec_act"] += f.act == g.act
                else: m["exec_n"] += 1; m["false_abstain"] += f.act in NONEXEC
            elif g.act in NONEXEC: m["nonexec_n"] += 1
            else: m["exec_n"] += 1
            (good_p if exact else bad_p).append(info["pmin"] or 0.0)
            by_act[g.act][0] += exact; by_act[g.act][1] += 1; by_tag[r["tag"]][0] += exact; by_tag[r["tag"]][1] += 1
            preds.append({"set": name, "id": r["id"], "text": r["text"], "gold": g.to_text(), "pred": f.to_text() if f else None, "raw": info["raw"], "ok": exact, "pmin": info["pmin"], "ms": info["ms"], "hint": info["hint"]})
        n = m["n"]
        summary[name] = {"n": n, "exact": round(m["exact"] / n, 3), "act": round(m["act"] / n, 3), "concept": round(m["concept"] / n, 3), "act_concept": round(m["act_concept"] / n, 3),
                         "op": f"{m['op']}/{m['op_n']}", "false_exec": f"{m['false_exec']}/{m['nonexec_n']}", "false_abstain": f"{m['false_abstain']}/{m['exec_n']}", "unparsed": n - m["parsed"],
                         "auroc_pmin": auroc(good_p, bad_p), "ms_mean": round(sum(lat) / len(lat)), "by_act": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_act.items())}, "by_tag": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_tag.items())}}
    out = os.path.join(ROOT, "results", "meaning", a.name); os.makedirs(out, exist_ok=True)
    json.dump({"adapter": a.adapter, "hints": not a.no_hints, "summary": summary, "predictions": preds}, open(os.path.join(out, "predictions.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in summary.items(): print(k, json.dumps({x: y for x, y in v.items() if x not in ("by_act", "by_tag", "by_family")}))
    print("wrote", out)


if __name__ == "__main__":
    main()
