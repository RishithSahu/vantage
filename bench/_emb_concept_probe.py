import json, glob, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from vgpt.retrieval import Embedder
from vgpt.meaning import ontology, glossary
onto = ontology.load(2); gl = glossary.load()
defs = json.load(open(os.path.join(ROOT, "vgpt", "meaning", "concept_docs.json")))
defs = defs.get("docs", defs) if isinstance(defs, dict) else defs
ids = [c for c in onto.words()["concept"] if c in onto.concepts]
docs = []
for cid in ids:
    c = onto.concepts[cid]; extra = [e.get("term") or "" for e in gl.index.values() if e.get("concept") == cid][:12]
    docs.append(" ; ".join([c.label] + list(c.aliases)[:10] + extra + [str(defs.get(cid, ""))[:200]]))
E = Embedder(os.path.join(ROOT, "models", "bge-small-en-v1.5"), int8=True)
D = E.encode(docs)
rows = {}
for f in glob.glob(os.path.join(ROOT, "results", "meaning", "*", "predictions.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if "frame2w" not in str(d.get("adapter")): continue
    for r in d.get("predictions", []):
        if "gold" in r: rows[(r["set"], r["text"])] = r
rows = list(rows.values()); Q = E.encode([r["text"] for r in rows], query=True); S = Q @ D.T
idx = {c: i for i, c in enumerate(ids)}
def rank(r, i, c):
    return int((S[i] > S[i][idx[c]]).sum()) + 1 if c in idx else None
real = [(i, r) for i, r in enumerate(rows) if r["gold"].split()[1] in idx and r["pred"].split()[1] in idx]
top1 = np.mean([rank(r, i, r["gold"].split()[1]) == 1 for i, r in real]); top3 = np.mean([rank(r, i, r["gold"].split()[1]) <= 3 for i, r in real])
print("embedding alone, concept top1 %.1f%% top3 %.1f%% over %d rows with a real gold concept" % (100 * top1, 100 * top3, len(real)))
wrong = [(i, r) for i, r in real if r["gold"].split()[1] != r["pred"].split()[1]]
gold_better = sum(S[i][idx[r["gold"].split()[1]]] > S[i][idx[r["pred"].split()[1]]] for i, r in wrong)
print("model-wrong concept rows:", len(wrong), "- embedding prefers gold over the model's pick in", gold_better)
right = [(i, r) for i, r in real if r["gold"].split()[1] == r["pred"].split()[1]]
emb_disagree = sum(rank(r, i, r["pred"].split()[1]) > 1 for i, r in right)
print("model-right rows:", len(right), "- embedding would not rank the model's (correct) pick first in", emb_disagree)
