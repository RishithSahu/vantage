"""Diagnostic: how often does the frozen model pick the intended FUNCTION (operation) on the development questions that stand alone? Reads only the first few generated tokens (the function name), so a full pass is quick.
Used to compare prompt layouts (example order / balance, definitions) and models on the same footing; it is not an accuracy claim about the product (arguments are not looked at).
    python bench/diag_function.py --set dev [--model PATH] [--variant current|shuffled]"""
import argparse, collections, os, random, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import plancompile as PC, prompt
from vgpt.modelplan import Planner, State
from vgpt.query_ir import Query
from vgpt.toolspec import FUNCS

EQUIV = {"count_events": {"count_events", "did_event_happen"}, "did_event_happen": {"did_event_happen", "count_events"}}


def gold_function(ir):
    try: q = Query.from_text(ir)
    except Exception: return None
    if ir.startswith("op=none"): return "decline"
    p = PC.query_to_plan(q)
    return p["fn"] if p else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--model"); ap.add_argument("--variant", default="current"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n-ctx", type=int, default=6144); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set)
    if os.path.basename(path).startswith("acceptance"): raise SystemExit("sealed")
    rows = [r for r in load(path) if not r["prev_text"]]
    if a.limit: rows = rows[: a.limit]
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    P = Planner(conn, model_path=a.model, n_ctx=a.n_ctx) if a.model else Planner(conn, n_ctx=a.n_ctx)                              # loads the model, detects its tool-call dialect, builds the default prefix "plan"
    lm = P.lm; shots = list(P.shots); tools = list(P.tools); name = "plan"
    if a.variant != "current":
        if a.variant == "shuffled": random.Random(7).shuffle(shots)
        elif a.variant == "noshots": shots = []
        elif a.variant == "format6": shots = [shots[i] for i in (0, 1, 2, 6, 4, 17) if i < len(shots)]
        elif a.variant == "toolsshuf": random.Random(3).shuffle(shots); random.Random(5).shuffle(tools)
        name = "p"; lm.set_prefix(P.system, shots, name=name, tools=tools)
    js = P.dialect.style == "json"; stop = ('", "arguments"',) if js else ("(",)
    ok = 0; conf = collections.Counter(); n = 0; t0 = time.time()
    for r in rows:
        g = gold_function(r["ir"]); msg = P.build_message(r["text"], State(), DEFAULT_NOW)[0]
        out, _ = lm.generate(msg, grammar=P.grammar_new, max_tokens=12, stop_text=stop, prefix=name, assistant_prefix=P.dialect.open_text); fn = (out.split('"name": "')[-1].split('"')[0] if js else out.split("(")[0]).strip(); n += 1
        good = fn == g or fn in EQUIV.get(g, ()) or (g == "decline" and fn in ("decline", "ask_user", "unresolved", "say"))
        ok += good
        if not good: conf[(g, fn)] += 1
    print(f"variant={a.variant} model={os.path.basename(P.model_path)} function accuracy {ok}/{n} = {ok / n:.2f}  ({time.time() - t0:.0f}s)")
    for (g, f), c in conf.most_common(14): print(f"   gold {g:18s} -> chose {f:18s} x{c}")


if __name__ == "__main__": main()
