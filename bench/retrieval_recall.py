"""Retrieval recall that matters for generation: is EVERY capability the gold reading needs (target, second target, event, signals in threshold conditions) among the retrieved ones, and among the cards actually shown to the model?
The weaker 'at least one relevant capability in the top 3' is reported too, only to show how much it overstates. Retrieval never sees the gold reading; follow-ups get the previous gold reading's identifiers as conversation state (what a correct earlier turn would have carried).
    python bench/retrieval_recall.py --set dev|final [--k 8]"""
import argparse, os, sqlite3, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.eval_model import required_ids
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import plancompile as PC, prompt
from vgpt.modelplan import Planner, State
from vgpt.query_ir import Query


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--no-windows", action="store_true"); ap.add_argument("--floor", type=float, default=0.0); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set)
    if os.path.basename(path).startswith("acceptance"): raise SystemExit("sealed")
    P = Planner(sqlite3.connect(f"file:{a.db}?mode=ro", uri=True), load_model=False)
    if a.no_windows: P.retriever.search = (lambda f: (lambda q, k=6: f(q, k, windows=False)))(P.retriever.search)
    if a.floor: P.retriever.WIN_FLOOR = a.floor
    n = al_hit = al_card = one_hit = top3 = 0; misses = []
    for r in load(path):
        if r["ir"].startswith("op=none"): continue
        try: need = required_ids(Query.from_text(r["ir"]))
        except Exception: continue
        if not need: continue
        st = State()
        if r["prev_ir"]:
            try: st.prev_plan = PC.query_to_plan(Query.from_text(r["prev_ir"]))
            except Exception: pass
        msg, hits, cards, *_ = P.build_message(r["text"], st, DEFAULT_NOW, k=a.k); ids = [h.id for h in hits]; carried = set(prompt.plan_ids(st.prev_plan)); n += 1
        have = set(ids) | carried
        al_hit += all(i in have for i in need); al_card += all(i in cards for i in need); one_hit += any(i in have for i in need); top3 += any(i in set(ids[:3]) | carried for i in need)
        if not all(i in cards for i in need): misses.append((r["id"], r["text"], need, [i for i in need if i not in cards]))
    pct = lambda k: f"{k}/{n} = {100 * k / max(n, 1):.0f}%"
    print(f"{os.path.basename(path)}: questions with required capabilities: {n}\n  all required retrieved        {pct(al_hit)}\n  all required shown as cards   {pct(al_card)}\n  at least one retrieved        {pct(one_hit)}\n  at least one in top 3         {pct(top3)}")
    for m in misses[:25]: print("  MISSING", m[0], m[1][:60], "| needs", m[2], "| absent", m[3])


if __name__ == "__main__": main()
