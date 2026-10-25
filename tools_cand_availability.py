"""Candidate availability, measured on EVERY answerable question of a set (not only on the questions the rules could not answer): is the intended query (it executes to the gold result on all databases) among the deterministic generator's candidates?
Availability is a property of the generator; it is NOT user success (the person must also recognise the reading) and NOT model success.
    python tools_pin.py 8,9 tools_cand_availability.py --set dev [--out results/cand_availability_dev.json]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench.eval_v2 import Engines
from bench.evalcore import score
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import candidates as CD
from vgpt.query_ir import IRError, Query
ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--out"); a = ap.parse_args()
path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set); rows = load(path); eng = Engines(); out = []; ans = [r for r in rows if r["ir"] != "op=none"]
for r in ans:
    prev = None
    if r["prev_text"] and r["prev_ir"]:
        try: prev = Query.from_text(r["prev_ir"])
        except IRError: prev = None
    cands = CD.generate(r["text"], prev, None, DEFAULT_NOW, 3600, rules_draft=None); texts = [q.to_text() for q, _ in cands]
    out.append({"id": r["id"], "family": r["family"], "n": len(texts), "available": any(score(eng, r["ir"], c)["correct"] for c in texts)})
k = sum(1 for o in out if o["available"]); print(f"{a.set}: intended query among the generator's candidates for {k} of {len(out)} answerable questions ({100 * k / len(out):.1f}%); mean candidates {sum(o['n'] for o in out) / len(out):.1f}")
if a.out: json.dump({"set": a.set, "k": k, "n": len(out), "rows": out}, open(a.out, "w"), indent=1)
