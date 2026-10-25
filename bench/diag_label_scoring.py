"""Diagnostic of the DECISION PROCEDURE for the requested result: greedy constrained generation versus scoring every COMPLETE label, with and without contextual calibration, and sensitivity to label order. Same model, same prompt text, same questions (stand-alone questions of the frozen slice), same label descriptions; the target is the independent annotation of the requested result (bench/slice_v1_annotations.tsv), judged STRICTLY (no equivalence classes; refusal-type results are only right when the label is refuse).
  greedy      the label produced by constrained greedy decoding (what the production interface does)
  scored      argmax of log P(complete label + end of turn | prompt) over all labels (a trie over the label tokens, so shared prefixes are scored once)
  calibrated  scored minus the label's log-prior estimated from content-free inputs ("N/A", "[MASK]", empty message) under the SAME prompt: contextual calibration, an inference change only
Order sensitivity: the labels are listed in the original and in the reversed order; the table reports each and how often the decision changes. Tokenization of every label is printed once.
    python tools_pin.py 8,9 bench/diag_label_scoring.py --model models/granite-4.0-1b-Q4_K_M.gguf [--stride 1]
Needs a plain-transformer model (the live cache is rolled back after each continuation); hybrid models would need saved states."""
import argparse, collections, ctypes, math, os, sqlite3, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import llama_cpp
from bench.diag_gliner import annotations
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import toolspec as T
from vgpt.modelplan import Planner, State
from vgpt.slm_backend import PrefixLlama

LABELS = list(T.RESULT_NL)
NEUTRAL = ["N/A", "[MASK]", "", "the", "?"]


def logsoftmax(x):
    x = x.astype(np.float64); m = x.max(); return x - m - math.log(np.exp(x - m).sum())


class Scorer:
    def __init__(self, lm, labels):
        self.lm = lm; self.llm = lm.llm; self.labels = labels; self.n = self.llm.n_vocab(); eos = lm.eos
        self.seqs = [self.llm.tokenize(l.encode(), add_bos=False, special=False) + [eos] for l in labels]

    def logits(self): return np.ctypeslib.as_array(llama_cpp.llama_get_logits(self.llm.ctx), shape=(self.n,)).copy()

    def score(self, base_ids):
        """Evaluate the prompt once, then every label sequence through a trie (edges are evaluated one token at a time; the cache is rolled back between branches). -> {label: log P(label + end of turn | prompt)}"""
        llm = self.llm; llm.eval(base_ids); base = llm.n_tokens; out = {}
        trie = {}
        for i, s in enumerate(self.seqs):
            node = trie
            for t in s: node = node.setdefault(t, {})
            node["_leaf"] = i
        def walk(node, depth, acc, lp):
            for t, child in node.items():
                if t == "_leaf": continue
                if "_leaf" in child and len(child) == 1 and False: pass
                nacc = acc + lp[t]
                llm.n_tokens = base + depth; llm.eval([t]); lp2 = logsoftmax(self.logits()) if any(k != "_leaf" for k in child) else None
                if "_leaf" in child: out[self.labels[child["_leaf"]]] = nacc
                if lp2 is not None: walk(child, depth + 1, nacc, lp2)
        llm.n_tokens = base
        # logits after the prompt: the last eval left them for the final prompt token
        walk(trie, 0, 0.0, logsoftmax(self.logits()))
        llm.n_tokens = base
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "granite-4.0-1b-Q4_K_M.gguf")); ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shuffles", type=int, default=2, help="extra random label orders; the decision is also reported averaged over all orders"); a = ap.parse_args()
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; ann = annotations()
    rows = [rows[i] for i in ids if not rows[i]["prev_text"] and ann[i]["result"] in LABELS]
    if a.limit: rows = rows[: a.limit]
    P = Planner(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True), load_model=False)
    lm = PrefixLlama(a.model, n_ctx=4096, threads=2, mmap=True)
    assert lm.truncatable, "needs a plain-transformer model"
    results = {}; import random
    orders = [("original order", LABELS), ("reversed order", LABELS[::-1])] + [(f"shuffle {k}", random.Random(k).sample(LABELS, len(LABELS))) for k in range(a.shuffles)]
    allscores = {}
    for order_name, labels in orders:
        system = ("You classify what a person asks about one vehicle's recorded data. Choose the ONE kind of result they want back:\n" + "\n".join(f"- {l}: {T.RESULT_NL[l]}" for l in labels) + "\nAnswer with the label only.")
        lm.set_prefix(system, [], name=order_name); sc = Scorer(lm, LABELS)
        if order_name == "original order": print("label tokenization (tokens incl. end of turn):", {l: len(s) for l, s in zip(LABELS, sc.seqs)})
        g = "root ::= " + " | ".join(f'"{l}"' for l in LABELS)
        def prompt_ids(message): return lm.suffix_ids("Message: " + message, "", order_name, None)
        def evaluate(message):
            lm.llm.load_state(lm._state_of(order_name)) if lm._resident != order_name else None
            lm._resident = order_name; lm.llm.n_tokens = lm.prefixes[order_name]["tokens"]; ids = prompt_ids(message)
            return sc.score(ids)
        prior = collections.defaultdict(list)
        for nt in NEUTRAL:
            for l, v in evaluate(nt).items(): prior[l].append(v)
        logprior = {l: float(np.mean(v)) for l, v in prior.items()}; hits = collections.Counter(); per = []; t0 = time.time()
        for r in rows:
            gold = ann[r["id"]]["result"]; s = evaluate(r["text"]); lm.llm.n_tokens = lm.prefixes[order_name]["tokens"]
            greedy, _ = lm.generate("Message: " + r["text"], grammar=g, max_tokens=8, stop_text=None, prefix=order_name); greedy = greedy.strip()
            raw = max(s, key=s.get); cal = max(s, key=lambda l: s[l] - logprior[l]); rank_raw = sorted(s, key=s.get, reverse=True); rank_cal = sorted(s, key=lambda l: s[l] - logprior[l], reverse=True)
            hits["greedy"] += greedy == gold; hits["scored"] += raw == gold; hits["calibrated"] += cal == gold; hits["scored top-3"] += gold in rank_raw[:3]; hits["calibrated top-3"] += gold in rank_cal[:3]
            allscores.setdefault(r["id"], []).append({l: s[l] - logprior[l] for l in s}); per.append({"id": r["id"], "gold": gold, "greedy": greedy, "scored": raw, "calibrated": cal, "top3_cal": rank_cal[:3]})
        n = len(rows); print(f"[{order_name}] n={n}  greedy {hits['greedy']}/{n} = {100 * hits['greedy'] / n:.0f}%   scored {hits['scored']}/{n} = {100 * hits['scored'] / n:.0f}%   calibrated {hits['calibrated']}/{n} = {100 * hits['calibrated'] / n:.0f}%   top-3: scored {100 * hits['scored top-3'] / n:.0f}%, calibrated {100 * hits['calibrated top-3'] / n:.0f}%   ({(time.time() - t0) / n:.1f} s/question)")
        results[order_name] = per
    n = len(rows); gold = {r["id"]: ann[r["id"]]["result"] for r in rows}; avg = {i: {l: float(np.mean([sc_[l] for sc_ in v])) for l in v[0]} for i, v in allscores.items()}
    top1 = sum(max(avg[i], key=avg[i].get) == gold[i] for i in avg); top3 = sum(gold[i] in sorted(avg[i], key=avg[i].get, reverse=True)[:3] for i in avg)
    print(f"AVERAGED over {len(orders)} label orders (calibrated scores): top-1 {top1}/{n} = {100 * top1 / n:.0f}%, top-3 {top3}/{n} = {100 * top3 / n:.0f}%")
    import json; json.dump({"orders": [o for o, _ in orders], "gold": gold, "scores": allscores}, open(os.path.join(HERE, "results", "dls_scores.json"), "w"))
    a1, a2 = results["original order"], results["reversed order"]
    for k in ("greedy", "scored", "calibrated"): print(f"decision changes when the label order is reversed ({k}): {sum(1 for x, y in zip(a1, a2) if x[k] != y[k])}/{len(a1)}")
    conf = collections.Counter((x["gold"], x["calibrated"]) for x in a1 if x["gold"] != x["calibrated"]); print("calibrated errors:", conf.most_common(10))


if __name__ == "__main__": main()
