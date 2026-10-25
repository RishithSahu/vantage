"""Run every Round-52 question through the assistant and print the typed reading and the answer for review.
    python bench/round52_run.py [--db data/gen_demo.db] [--now 1787625900] [--out results/round52/answers.json]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db")); ap.add_argument("--now", type=int, default=1787625900)
ap.add_argument("--out", default=os.path.join(HERE, "results", "round52", "answers.json")); ap.add_argument("--only", default="")
a = ap.parse_args()
from vgpt.product import make_legacy_brain
b = make_legacy_brain(a.db, slm=None, now=a.now)
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(HERE, "bench", "round52_questions.tsv"), encoding="utf8") if l.strip()]
out = []
for qid, q in rows:
    if a.only and not qid.startswith(a.only): continue
    b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
    r = b.ask(q, use_slm=False); f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
    out.append({"id": qid, "q": q, "path": r.get("path"), "intent": r.get("intent"), "route": f.get("route"), "answer": r.get("answer"), "query": (f.get("query") or f.get("canonical_query") or f.get("queries"))})
    print(f"{qid:4s} [{r.get('path')}/{f.get('route')}] {q}\n      -> {(r.get('answer') or '')[:420].replace(chr(10), ' / ')}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
