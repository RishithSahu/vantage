"""Routing accuracy for the extended domains (tyres, weather, service, health, driving, places) + the questions from a real tester's chat transcript ("field").
NOTE: written by the developer AFTER the rules - not an independent set; 'field' items are the tester's own words.
Usage: python bench/eval_ext.py <model.gguf|none> [db]   -> prints per-split accuracy and every miss; also answers each question on the db and flags errors"""
import os, sys, json, statistics as st
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
sys.path.insert(0, ".")
from vgpt.brain import Brain, Slm
Q = json.load(open("bench/questions_ext.json", encoding="utf-8"))
model = sys.argv[1]; db = sys.argv[2] if len(sys.argv) > 2 else "data/gen_demo.db"
slm = None if model == "none" else Slm("models/" + model, threads=2); slm.warmup() if slm else None
b = Brain(db, slm=slm); res = {}; miss = []; errs = []
for q in Q:
    b.units = {"p": "kpa", "d": "km", "t": "c"}
    r = b.ask(q["q"]); good = (r["intent"] == q["intent"]) if q["intent"] != "NONE" else (r["intent"] == "NONE" and r["path"] in ("gate", "unsupported", "clarify"))
    d = res.setdefault(q["split"], [0, 0, 0]); d[1] += 1; d[0] += good
    if r["path"] == "slm" or r["path"] == "clarify": d[2] += 1
    if not good: miss.append((q["split"], q["q"], q["intent"], r["intent"], r["path"]))
    if r["path"] == "error" or "went wrong" in r["answer"]: errs.append((q["q"], r["answer"]))
for k, (ok, n, via_slm) in res.items(): print(f"{(model[:20] if slm else 'rules only'):22s} {k:6s}: {ok}/{n} correct ({100 * ok // n}%), {via_slm} needed the language model")
tot = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values()); print(f"TOTAL {tot}/{n} ({100 * tot // n}%)")
for m in miss: print("   MISS", m)
for e in errs: print("   ERROR", e)
