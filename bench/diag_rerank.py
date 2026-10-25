"""Diagnostic: can the SLM resolve a SPECIFIC distinction among a few plausible results? Candidates (the top 3 of the fused Granite + extractor scores of bench/diag_label_scoring.py and bench/diag_gliner.py) are shown with their descriptions in every ORDER (all permutations), the SLM scores each complete candidate label, and the scores are averaged over the permutations (this cancels position bias without calibration). Compared with the candidate it started from. The original question is always shown; nothing is excluded permanently (the SLM may also choose the candidate the earlier stage ranked lower).
It reads results/dls_scores.json (label scoring) and results/gl_base_slice.json (extractor). Strict target: the independent annotation of the requested result. A DIAGNOSTIC, not a product path: it adds model calls.
    python tools_pin.py 8,9 bench/diag_rerank.py [--model PATH] [--k 3]"""
import argparse, itertools, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_label_scoring import Scorer
from bench.indep_io import load
from vgpt import toolspec as T
from vgpt.slm_backend import PrefixLlama

SYSTEM = "You decide what kind of result a person wants back from one vehicle's recorded data. Several kinds are offered; answer with the label of the ONE that fits the message best."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "granite-4.0-1b-Q4_K_M.gguf")); ap.add_argument("--k", type=int, default=3); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    sc = json.load(open(os.path.join(HERE, "results", "dls_scores.json"))); gl = {r["id"]: r["result_scores"] for r in json.load(open(os.path.join(HERE, "results", "gl_base_slice.json")))}; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}
    labels = list(next(iter(sc["scores"].values()))[0]); ids = list(sc["gold"])[: a.limit or None]
    def fused(i):
        v = np.array([[o[l] for l in labels] for o in sc["scores"][i]]).mean(0); v = v - v.max(); g = np.exp(v) / np.exp(v).sum(); c = np.clip(np.array([gl[i][l] for l in labels]), 1e-6, 1 - 1e-6); c = c / c.sum(); return np.log(g + 1e-9) + np.log(c + 1e-9)
    lm = PrefixLlama(a.model, n_ctx=2048, threads=2, mmap=True); assert lm.truncatable; lm.set_prefix(SYSTEM, [], name="p"); t0 = time.time(); start = after = both_top3 = n = 0; ch = []
    for i in ids:
        s = fused(i); cand = [labels[j] for j in np.argsort(-s)[: a.k]]; gold = sc["gold"][i]; n += 1; start += cand[0] == gold; both_top3 += gold in cand
        tot = {c: 0.0 for c in cand}; perms = list(itertools.permutations(cand))
        for perm in perms:
            msg = "Options:\n" + "\n".join(f"- {c}: {T.RESULT_NL[c]}" for c in perm) + f"\nMessage: {rows[i]['text']}"; sc_ = Scorer(lm, list(perm)); lm.llm.n_tokens = lm.prefixes["p"]["tokens"]; lm._resident = "p"
            out = sc_.score(lm.suffix_ids(msg, "", "p", None))
            for c in perm: tot[c] += out[c] / len(perms)
        pick = max(tot, key=tot.get); after += pick == gold; ch.append((i, cand, pick, gold))
    print(f"{n} stand-alone questions: gold within the candidates {both_top3}/{n};  the stage before picks it first {start}/{n} = {100 * start / n:.0f}%;  after the SLM resolves the {a.k} candidates (all {len(perms)} orders averaged) {after}/{n} = {100 * after / n:.0f}%   ({(time.time() - t0) / n:.0f} s/question)")
    for x in ch:
        if x[2] != x[3]: print("  ", x)


if __name__ == "__main__": main()
