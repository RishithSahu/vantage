"""Function-selection confusion from a results/m*_*.json of bench/eval_model.py (gold function derived from the gold query, as in bench/diag_function.py).   python bench/fn_confusion.py results/x.json"""
import collections, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_function import EQUIV, gold_function
d = json.load(open(sys.argv[1], encoding="utf-8")); rec = {(r["text"], r["prev_text"]): r for r in d["records"]}; ok = n = 0; conf = collections.Counter(); per = collections.defaultdict(lambda: [0, 0])
for r in d["rows"]:
    c = rec.get((r["text"], r["prev_text"]), {}); g = gold_function(r["gold"])
    if g is None or r["prev_text"]: continue
    call = c.get("call") or ""; fn = (call.split("(")[0] if "(" in call and not call.startswith("{") else (call.split('"name": "')[-1].split('"')[0] if call.startswith("{") else "")).strip(); n += 1
    good = fn == g or fn in EQUIV.get(g, ()) or (g == "decline" and fn in ("decline", "ask_user", "unresolved", "say")); ok += good; per[g][1] += 1; per[g][0] += good
    if not good: conf[(g, fn)] += 1
print(f"function accuracy on stand-alone questions: {ok}/{n} = {ok / max(n, 1):.2f}"); print(" per gold function:", {k: f"{a}/{b}" for k, (a, b) in sorted(per.items())})
for (g, f), c in conf.most_common(12): print(f"   gold {g:18s} -> chose {f:18s} x{c}")
