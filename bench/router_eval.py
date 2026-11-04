"""Could a tiny nearest-example router answer the easy questions instead of the language model, without losing a single correct answer?

For every audit question (the 3,842 of 2026-10-25, follow-ups excluded because they need the previous frame) the nearest TRAINING questions of the same car type are found (TF-IDF over characters and words: no model, no download, microseconds on the Pi).  The router
answers only when the nearest question is very close AND the neighbours agree on one frame.  What matters is the NET effect on the strict-correct count against the language model's own answers (results/meaning/audit1025_r10):
    net = (router right) - (model right), summed over the questions the router would take.
The acceptance rule written before the run (2026-10-26): reject if the whole system loses 15 or more strict-correct questions; prefer none lost.

    python bench/router_eval.py [--audit audit1025]"""
import argparse, io, json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
import eval_r10 as E  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from scipy.sparse import hstack  # noqa: E402


def norm(t): return " ".join(str(t).lower().split())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", default="audit1025"); ap.add_argument("--train", default="training/meaning/data/frame2_r12/train.jsonl"); a = ap.parse_args()
    tr = [json.loads(l) for l in io.open(os.path.join(ROOT, a.train), encoding="utf-8")]
    tr = [t for t in tr if t.get("kind") == "frame"]
    base = json.load(open(os.path.join(ROOT, "results", "meaning", a.audit + "_r10", "predictions.json"), encoding="utf-8"))["predictions"]
    key = lambda p: (p["set"], str(p["id"]))
    bm = {key(p): p for p in base}
    rows = []
    for s in E.FROZEN + E.NEW:
        for r in E.load(s):
            p = bm.get((s, str(r["id"])))
            if p is not None: rows.append((s, r, p))
    print("train rows", len(tr), "eval rows", len(rows))
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2); vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    tt = [norm(t["text"]) for t in tr]; vc.fit(tt); vw.fit(tt)
    feat = lambda texts: hstack([vc.transform(texts), vw.transform(texts)]).tocsr()
    Xt = feat(tt);
    from sklearn.preprocessing import normalize
    Xt = normalize(Xt)
    tveh = np.array([t.get("vehicle", "ev") for t in tr]); tans = [E.pad(t["answer"]) for t in tr]
    veh_of = lambda s, r: r["tag"].split(":")[0] if r["tag"].split(":")[0] in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev")
    res = []
    for s, r, p in rows:
        if r["prev"] is not None: res.append(None); continue
        x = normalize(feat([norm(r["text"])])); sims = (Xt @ x.T).toarray().ravel(); v = veh_of(s, r)
        m = np.where(tveh == v)[0] if v in ("ev", "petrol", "diesel") else np.arange(len(tr))
        top = m[np.argsort(-sims[m])[:5]]; votes = collections.Counter()
        for i in top: votes[tans[i]] += sims[i]
        fr, w = votes.most_common(1)[0]; agree = w / max(1e-9, sum(votes.values())); res.append((float(sims[top[0]]), agree, fr, E.pad(r["gold"])))
    mc = [str(p["strict"]) == "True" for _, _, p in rows]; total_model = sum(mc)
    print("model strict-correct", total_model, "of", len(rows))
    print(f"{'min sim':>8} {'min agree':>9} {'taken':>6} {'share':>6} {'router right':>13} {'model right':>12} {'net':>5}")
    best = None
    for ts in (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        for ta in (1.0, 0.8):
            taken = right_r = right_m = 0
            for (s, r, p), m_ok, q in zip(rows, mc, res):
                if q is None: continue
                sim, agree, fr, gold = q
                if sim >= ts and agree >= ta:
                    taken += 1; right_r += int(fr.split()[:len(gold.split())] == gold.split()[:len(fr.split())] and E.ok_prefix(fr, r["gold"])); right_m += int(m_ok)
            print(f"{ts:8.2f} {ta:9.1f} {taken:6d} {100 * taken / len(rows):5.1f}% {right_r:13d} {right_m:12d} {right_r - right_m:+5d}")


if __name__ == "__main__": main()
