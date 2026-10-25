"""ORACLE-CANDIDATE SELECTION diagnostic (not product accuracy): can a frozen compact model choose the COMPLETE correct interpretation of a question among the correct one and valid nearby alternatives (bench/oracle_candidates.py: operation, statistic/polarity, target, time, scope, event attachment)? Also: does it decline when NONE of the candidates is correct (the same sets with the gold removed)?
Selectors (one per run, one model resident at a time; identical candidate sets and descriptions):
  nli     MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary: P(entailment) of  premise = 'The user asks: "<question>"'  hypothesis = 'The user asks for <candidate>.'  (class 0 = entailment, class 1 = not_entailment, read from the config)
  small   GLiNER2.5-small  classify_text, the candidate descriptions as the labels (multi-label scores, no threshold)
  decide  GLiNER2.5-Decide classify_text, same
Order sensitivity: every set is scored in 4 candidate orders (original, reversed, 2 shuffles); reported per order, with the order-averaged score, and as the fraction of questions whose top candidate changes with the order.
    PYTHONPATH=pylib_gliner2_v2 python tools_pin.py 8,9 bench/diag_selector.py --selector nli|small|decide [--out results/sel_x.json]"""
import argparse, collections, json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from bench.oracle_candidates import build

HYP = "The user asks for {}."; PREM = 'The user asks: "{}"'


class NLI:
    name = "DeBERTa-v3-xsmall NLI"

    def __init__(self):
        import torch; from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(2); d = os.path.join(HERE, "models", "deberta_xsmall_nli"); self.tok = AutoTokenizer.from_pretrained(d); self.m = AutoModelForSequenceClassification.from_pretrained(d).eval(); self.torch = torch
        self.ent = [i for i, l in self.m.config.id2label.items() if l.lower() == "entailment"][0]; self.max_tokens = 0

    def score(self, question, cands):
        enc = self.tok([PREM.format(question)] * len(cands), [HYP.format(c) for c in cands], return_tensors="pt", padding=True, truncation=True, max_length=512); self.max_tokens = max(self.max_tokens, int(enc["attention_mask"].sum(1).max()))
        with self.torch.no_grad(): p = self.torch.softmax(self.m(**enc).logits, -1)[:, int(self.ent)]
        return p.tolist()


class Gliner:
    def __init__(self, path, name):
        from gliner2 import AutoExtractor
        import torch; torch.set_num_threads(2); self.m = AutoExtractor.from_pretrained(os.path.join(HERE, "models", path)); self.m.eval(); self.name = name; self.max_tokens = 0

    def score(self, question, cands):
        r = self.m.classify_text(question, {"meaning": {"labels": list(cands), "multi_label": True, "cls_threshold": 0.0}}, include_confidence=True)["meaning"]; d = {e["label"]: float(e["confidence"]) for e in r}
        return [d.get(c, 0.0) for c in cands]


def auroc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selector", required=True, choices=["nli", "small", "decide"]); ap.add_argument("--n-alt", type=int, default=6); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out"); a = ap.parse_args()
    t = time.time(); sel = {"nli": NLI, "small": lambda: Gliner("gliner25_small", "GLiNER2.5-small"), "decide": lambda: Gliner("gliner25_decide", "GLiNER2.5-Decide")}[a.selector](); load_s = time.time() - t
    sets = build(n_alt=a.n_alt)[: a.limit or None]; rng = random.Random(3); n = len(sets)
    orders = ["original", "reversed", "shuffle1", "shuffle2"]; hit = {o: 0 for o in orders}; avg_hit = 0; flips = 0; best_neg = []; best_pos = []; wrong_kind = collections.Counter(); ms = []; recs = []
    for s in sets:
        cands = [d for _, d in s["candidates"]]; kinds = {d: k for k, d in s["candidates"]}; gold = cands[0]; idx = list(range(len(cands))); per = {}
        perms = {"original": idx, "reversed": idx[::-1], "shuffle1": rng.sample(idx, len(idx)), "shuffle2": rng.sample(idx, len(idx))}; tops = []
        for o, perm in perms.items():
            t0 = time.perf_counter(); sc = sel.score(s["text"], [cands[i] for i in perm]); ms.append((time.perf_counter() - t0) * 1000)
            for i, v in zip(perm, sc): per.setdefault(i, []).append(v)
            top = perm[int(np.argmax(sc))]; tops.append(top); hit[o] += top == 0
        mean = {i: float(np.mean(v)) for i, v in per.items()}; top = max(mean, key=mean.get); avg_hit += top == 0; flips += len(set(tops)) > 1; best_pos.append(mean[top])
        if top != 0: wrong_kind[kinds[cands[top]]] += 1
        # none-correct: the same alternatives without the gold
        nper = {}
        for o, perm in perms.items():
            perm2 = [i for i in perm if i != 0]; sc = sel.score(s["text"], [cands[i] for i in perm2])
            for i, v in zip(perm2, sc): nper.setdefault(i, []).append(v)
        best_neg.append(max(float(np.mean(v)) for v in nper.values())); recs.append({"id": s["id"], "text": s["text"], "top": cands[top], "gold": gold, "kind_of_top": kinds[cands[top]], "scores": {cands[i]: round(v, 3) for i, v in mean.items()}, "best_without_gold": round(best_neg[-1], 3)})
    print(f"{sel.name} ({a.selector}): {n} questions, {len(sets[0]['candidates'])} candidates each (max {max(len(s['candidates']) for s in sets)}); load {load_s:.1f}s; scoring {np.mean(ms):.0f} ms per candidate set (a set of ~7 candidates)")
    print("  top-1 with the gold among the candidates: " + ", ".join(f"{o} {hit[o]}/{n} ({100 * hit[o] / n:.0f}%)" for o in orders) + f";  order-averaged scores {avg_hit}/{n} ({100 * avg_hit / n:.0f}%);  top candidate changes with the order for {flips}/{n} questions")
    print(f"  wrong picks by kind of the chosen (order-averaged): {dict(wrong_kind)}")
    thr = 0.5; print(f"  NO correct candidate: best score >= {thr} (would be accepted) {sum(b >= thr for b in best_neg)}/{n};  with the gold present the top score >= {thr}: {sum(b >= thr for b in best_pos)}/{n};  separation (AUROC of the best score, gold present vs absent) {auroc(best_pos, best_neg):.2f}")
    if hasattr(sel, "max_tokens") and sel.max_tokens: print(f"  longest premise+hypothesis: {sel.max_tokens} tokens (limit 512: no truncation)")
    if a.out: json.dump({"selector": a.selector, "name": sel.name, "n": n, "top1": hit, "avg_top1": avg_hit, "flips": flips, "rows": recs}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
