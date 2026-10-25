"""Bounded feasibility probe of a DECISION model (LiquidAI/d1-omni-600M, frozen, no training, text only): does it choose the requested result, and decide whether the message states a period / restriction / number / selection, on the 55 stand-alone questions of the frozen slice? Strict target = the independent annotation (bench/slice_v1_annotations.tsv). Raw top-1 only (no top-3). Reports the time of the complete input processing and the model's own input-token count.
The checkpoint ships its own Python modules (loaded with trust_remote_code from a folder that was READ before use); it is a float32 reference implementation - NOT a deployment configuration (memory is reported).
    python -I tools_pin.py 8,9 bench/diag_d1.py --model PATH_TO_D1_FOLDER [--limit N] [--out results/d1_slice.json]"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from bench.diag_gliner import annotations, has
from bench.indep_io import load
from vgpt import toolspec as T

PRESENCE = {
    "period": "Does the message itself state a time period or interval for the data (for example yesterday, last week, over the past 5 days, this month)?",
    "restriction": "Does the message restrict the data to a driving situation or condition (for example on the highway, in the city, while charging, at night, in the rain)?",
    "number": "Does the message state a numeric threshold or limit with a value (for example above 40 km/h, below 20 percent)?",
    "selection": "Does the message pick particular trips or charging sessions (for example the longest trip, the latest charging session, my five most efficient trips)?",
}
ANN_KEY = {"period": "interval", "restriction": "restriction", "number": "number", "selection": "selection"}


def rss_mb():
    try:
        import psutil; return psutil.Process().memory_info().rss / 1e6
    except Exception: return -1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out"); a = ap.parse_args()
    torch.set_num_threads(2)
    from transformers import AutoModel
    t = time.time(); model = AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).eval(); print(f"load {time.time() - t:.1f}s, rss {rss_mb():.0f} MB, params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; ann = annotations()
    rows = [rows[i] for i in ids if not rows[i]["prev_text"] and ann[i]["result"] in T.RESULT_NL]
    if a.limit: rows = rows[: a.limit]
    results = list(T.RESULT_NL); stats = list(T.STAT_NL) + ["none"]
    qs = {"result": {"type": "choice", "instructions": "What kind of result does the person want back from the vehicle's recorded data?", "criteria": {k: T.RESULT_NL[k] for k in results}},
          "statistic": {"type": "choice", "instructions": "Which summary statistic does the message ask for?", "criteria": {**{k: T.STAT_NL[k] for k in T.STAT_NL}, "none": "No particular statistic is asked for"}}}
    for k, text in PRESENCE.items(): qs[k] = {"type": "noul", "instructions": text}
    ok = 0; st_ok = st_n = 0; pres = {k: [0, 0, 0, 0] for k in PRESENCE}; per = []; ms = []; toks = []
    for r in rows:
        t0 = time.perf_counter(); out = model.system_one(r["text"], qs); dt = (time.perf_counter() - t0) * 1000; ms.append(dt); toks.append(out["usage"]["input_tokens"]); an = ann[r["id"]]; ans = out["answers"]
        got = ans["result"]["choice"]; ok += got == an["result"]; rec = {"id": r["id"], "text": r["text"], "gold": an["result"], "got": got, "conf": round(ans["result"]["confidence"], 2)}
        want_st = an["statistic"] if has(an["statistic"]) else "none"
        if want_st in stats: st_n += 1; st_ok += ans["statistic"]["choice"] == want_st
        for k in PRESENCE:
            p = ans[k]["noul"]; said = has(an[ANN_KEY[k]]); pred = p >= 0.5; c = pres[k]; c[0] += pred == said; c[1] += 1
            if said: c[2] += 1; c[3] += 0 if pred else 0
            rec[k] = round(p, 2)
        per.append(rec)
    n = len(rows)
    print(f"{n} stand-alone slice questions: requested result raw top-1 {ok}/{n} = {100 * ok / n:.0f}%;  statistic (incl. 'none') {st_ok}/{st_n};  complete input processing {sum(ms) / n:.0f} ms/question for {len(qs)} questions (model input {sum(toks) / n:.0f} tokens);  rss {rss_mb():.0f} MB")
    for k in PRESENCE:
        tp = sum(1 for r in per if has(ann[r["id"]][ANN_KEY[k]]) and r[k] >= .5); pos = sum(1 for r in per if has(ann[r["id"]][ANN_KEY[k]])); fp = sum(1 for r in per if not has(ann[r["id"]][ANN_KEY[k]]) and r[k] >= .5); neg = n - pos
        print(f"  {k:12s} presence correct {pres[k][0]}/{n};  stated and found {tp}/{pos};  not stated but claimed {fp}/{neg}")
    import collections
    print("result errors:", collections.Counter((r["gold"], r["got"]) for r in per if r["gold"] != r["got"]).most_common(12))
    if a.out: json.dump({"summary": {"n": n, "result_top1": ok}, "rows": per}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
