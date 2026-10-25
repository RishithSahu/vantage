"""Validity checks for the frozen-model planner before any accuracy number is read:
  1. CAPACITY - for every dev + final question, under the configuration given, prefix + question (+ examples, history, draft) + output allowance fits the context window; the worst case is reported.
  2. EQUIVALENCE - the restored-prefix path produces the same greedy output as ordinary full-prompt inference (same grammar) on a sample of questions.
    python tools_pin.py 8,9 bench/check_prompt.py --model models/LFM2.5-350M-Q4_K_M.gguf --bank 4 --draft"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.indep_io import load
from vgpt.schema_planner import SchemaPlanner
from vgpt import nl2ir
from training.build_dataset import DEFAULT_NOW

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--bank", type=int, default=0); ap.add_argument("--draft", action="store_true"); ap.add_argument("--mode", default="full"); ap.add_argument("--n_ctx", type=int, default=4608); ap.add_argument("--sample", type=int, default=30); a = ap.parse_args()
p = SchemaPlanner(a.model, mode=a.mode, bank_k=a.bank, n_ctx=a.n_ctx, cache_dir=os.path.join(HERE, "results", "_cache")); be = p.be; rows = load(os.path.join(HERE, "bench", "indep_dev.tsv")) + load(os.path.join(HERE, "bench", "indep_final.tsv"))
worst = 0; over = 0; longest = None
for r in rows:
    prev = (r["prev_text"], r["prev_ir"]) if r["prev_text"] else None; draft = None
    if a.draft: pr = nl2ir.parse(r["text"], now=DEFAULT_NOW, tz=3600); draft = pr.query.to_text() if (pr.query is not None and pr.query.op) else None
    sl = p.schema_slice(r["text"], prev[1] if prev else None) if a.mode == "slice" else (None, None, None)
    user = p._user(r["text"], prev, sl, draft); ids = be.llm.tokenize(user.encode(), add_bos=False, special=True); total = be.prefix_tokens + len(ids) + 80
    if total > worst: worst, longest = total, r["id"]
    over += total > a.n_ctx
print(f"CAPACITY: prefix {be.prefix_tokens} tokens; worst case {worst} of {a.n_ctx} (question {longest}); {over} of {len(rows)} questions would overflow; max question length {max(len(r['text']) for r in rows)} chars (no truncation is applied)")
import random
random.Random(0).shuffle(rows); same = 0; diff = []
for r in rows[: a.sample]:
    prev = (r["prev_text"], r["prev_ir"]) if r["prev_text"] else None; user = p._user(r["text"], prev, (None, None, None) if a.mode != "slice" else p.schema_slice(r["text"], prev[1] if prev else None), None)
    g = p.view.grammar() if a.mode != "slice" else p.view.grammar(*p.schema_slice(r["text"], prev[1] if prev else None)); out1, _ = be.generate(user, grammar=g, max_tokens=80)
    gr = be.grammar(g); res = be.llm.create_chat_completion(messages=[{"role": "system", "content": p.system_text()}, {"role": "user", "content": user}], grammar=gr, max_tokens=80, temperature=0.0); out2 = res["choices"][0]["message"]["content"].strip()
    if out1 == out2: same += 1
    else: diff.append((r["text"], out1, out2))
print(f"EQUIVALENCE: restored-prefix output == full-prompt output on {same} of {min(a.sample, len(rows))} sampled questions")
for d in diff[:5]: print("  DIFF", d)
