"""Diagnostic: does retrieving the nearest demonstrations from the INDEPENDENT generated bank (data/demo_bank.json) let the frozen SLM choose the requested result? Same 55 stand-alone slice questions, same label descriptions, strict target = independent annotation. Demonstrations are retrieved per question by bge-small similarity (k nearest, shown as earlier chat turns, nearest last); every complete label is scored (no calibration, label order fixed as in RESULT_NL). Saves per-question scores for fusion with the extractor (results/bank_scores_<k>.json).
    python tools_pin.py 8,9 bench/diag_bank_result.py [--k 8] [--model PATH]"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_label_scoring import Scorer, LABELS
from bench.diag_gliner import annotations
from bench.indep_io import load
from vgpt import toolspec as T
from vgpt.retrieval import Embedder
from vgpt.slm_backend import PrefixLlama

SYSTEM = ("You classify what a person asks about one vehicle's recorded data. Choose the ONE kind of result they want back:\n" + "\n".join(f"- {l}: {T.RESULT_NL[l]}" for l in LABELS) + "\nAnswer with the label only.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "granite-4.0-1b-Q4_K_M.gguf")); ap.add_argument("--k", type=int, default=8); ap.add_argument("--per-result", type=int, default=2, help="at most this many demonstrations of one result among the k")
    ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); a = ap.parse_args()
    bank = json.load(open(os.path.join(HERE, "data", "demo_bank.json")))["entries"]; emb = Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True); bv = emb.encode([b["q"] for b in bank])
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; ann = annotations()
    rows = [rows[i] for i in ids if not rows[i]["prev_text"] and ann[i]["result"] in LABELS]
    lm = PrefixLlama(a.model, n_ctx=3072, threads=2, mmap=True); assert lm.truncatable; lm.set_prefix(SYSTEM, [], name="p"); sc = Scorer(lm, LABELS)
    out = {}; t0 = time.time(); top1 = top3 = 0; errs = []
    for r in rows:
        sims = bv @ emb.encode([r["text"]], query=True)[0]; pick = []; cnt = {}
        for i in np.argsort(-sims):
            res = bank[int(i)]["result"]
            if cnt.get(res, 0) >= a.per_result: continue
            cnt[res] = cnt.get(res, 0) + 1; pick.append(int(i))
            if len(pick) >= a.k: break
        hist = [("Message: " + bank[i]["q"], bank[i]["result"]) for i in pick[::-1]] if a.k else None
        lm.llm.n_tokens = lm.prefixes["p"]["tokens"]; lm._resident = "p"
        s = sc.score(lm.suffix_ids("Message: " + r["text"], "", "p", hist)); out[r["id"]] = s; rank = sorted(s, key=s.get, reverse=True); g = ann[r["id"]]["result"]
        top1 += rank[0] == g; top3 += g in rank[:3]
        if rank[0] != g: errs.append((r["id"], g, rank[:3], r["text"]))
    n = len(rows); print(f"k={a.k} (<= {a.per_result} per result): top-1 {top1}/{n} = {100 * top1 / n:.0f}%  top-3 {top3}/{n} = {100 * top3 / n:.0f}%  ({(time.time() - t0) / n:.1f} s/question)")
    for e in errs: print("  ", e)
    json.dump({"gold": {r["id"]: ann[r["id"]]["result"] for r in rows}, "scores": out}, open(os.path.join(HERE, "results", f"bank_scores_{a.k}.json"), "w"))


if __name__ == "__main__": main()
