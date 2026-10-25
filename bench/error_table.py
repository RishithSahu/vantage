"""Per-question ERROR TABLE for a development run (results/m*_*.json written by bench/eval_model.py), used to choose the next intervention from the dominant failure rather than from a headline score.
For every answerable question it records, in cascade order:
    caps      all required capabilities reached the ACTUAL prompt (the cards the model saw, plus the state)
    op        the operation is the intended one (judged on the plan the model wrote, with the state it was given; equivalent operations such as count / did-it-happen are accepted)
    target    signal(s) and event(s) correct
    period    period(s) correct, after documented defaults
    restr     conditions / groups / selection / statistic / unit / event window correct
    state     (follow-ups) the resolved plan equals the intended one after the state edit
    compiled  compilation preserved the plan (the outcome is not an invalid / unsupported plan)
    answer    the final answer is scored correct by the evaluator (clarifications, refusals and wrong answers are NOT correct)
The plan is judged WITHOUT the grounding check (message=None), so understanding errors and provenance failures are not mixed; a run's grounding failures are counted separately.
    python bench/error_table.py results/x.json [--csv results/x_errors.csv] [--misses]"""
import argparse, collections, csv, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import sqlite3
from bench.diag_format import diff_fields, normalise
from bench.eval_model import required_ids
from training.build_dataset import DEFAULT_NOW
from vgpt import capreg, plancompile as PC, signal_registry as R
from vgpt.query_ir import Query

EQUIV = {"count_events": {"did_event_happen"}, "did_event_happen": {"count_events"}}
REFUSE_FNS = ("decline", "ask_user", "unresolved", "say")


def analyse(path):
    d = json.load(open(path, encoding="utf-8")); recs = {(r["text"], r["prev_text"]): r for r in d["records"]}
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); out = []
    for row in d["rows"]:
        rec = recs.get((row["text"], row["prev_text"])) or {}; gold = row["gold"]; e = {"id": row["id"], "family": row["family"], "text": row["text"], "gold": gold, "call": rec.get("call"), "outcome": rec.get("outcome"), "answer_ok": bool(row["correct"]), "followup": bool(row["prev_text"])}
        if gold.startswith("op=none"):
            e["kind"] = "refusal"; e["refused_ok"] = row["correct"]; out.append(e); continue
        e["kind"] = "answerable"; gq = Query.from_text(gold); need = required_ids(gq); cards = set(rec.get("card_ids") or []); e["caps"] = all(i in cards for i in need) if need else True
        mp = rec.get("model_plan"); fn = rec.get("fn")
        if not mp or fn in REFUSE_FNS or not fn:
            e.update(op=False, target=False, period=False, restr=False, state=False, compiled=False, cats=["no plan" if not mp else fn]); out.append(e); continue
        gp = PC.query_to_plan(gq); gfn = gp["fn"] if gp else None; e["op"] = fn == gfn or fn in EQUIV.get(gfn, ())
        o = PC.compile_plan(mp, rec.get("state_plan"), None, reg, DEFAULT_NOW, 0)
        if o.kind != "query": e.update(target=False, period=False, restr=False, state=False, compiled=False, cats=["not executable: " + o.kind + (" " + (o.reason or "")[:60] if o.reason else "")]); out.append(e); continue
        cq = o.query; g2 = Query.from_text(gold); normalise(g2, cq); cats = diff_fields(g2, cq); e["cats"] = cats
        e["target"] = not any(c in cats for c in ("target",)) and ("event" not in cats or True) and not (g2.event and cq.event and g2.event.type != cq.event.type)
        e["period"] = "period" not in cats; e["restr"] = not any(c in cats for c in ("conditions_added", "conditions_missing", "conditions_wrong", "groups", "statistic", "selection", "unit")) and not (g2.event and cq.event and (g2.event.selector, g2.event.before, g2.event.after) != (cq.event.selector, cq.event.before, cq.event.after) and g2.op not in ("count",))
        e["state"] = (not cats) if e["followup"] else None; e["compiled"] = rec.get("outcome") not in ("invalid_plan", "unsupported_calculation", "execution_error")
        e["plan_exact"] = not cats; out.append(e)
    return d["summary"], out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--csv"); ap.add_argument("--misses", action="store_true"); a = ap.parse_args()
    s, rows = analyse(a.path); ans = [r for r in rows if r["kind"] == "answerable"]; ref = [r for r in rows if r["kind"] == "refusal"]; n = len(ans)
    stages = [("caps", "all required capabilities in the actual prompt"), ("op", "operation correct"), ("target", "target(s) / event correct"), ("period", "period correct"), ("restr", "restrictions / selection / statistic correct")]
    print(f"{os.path.basename(a.path)}: {s['system']}  answerable {n}  refusal-type {len(ref)}  (total {len(rows)})")
    print("| stage | questions passing | of answerable |\n|---|---|---|")
    for k, label in stages: print(f"| {label} | {sum(1 for r in ans if r.get(k))} | {n} |")
    both = [r for r in ans if r.get("op") and r.get("target") and r.get("period") and r.get("restr")]
    fu = [r for r in ans if r["followup"]]; print(f"| operation AND target AND period AND restrictions all correct (plan exact) | {len(both)} | {n} |")
    print(f"| follow-ups whose resolved plan is exact | {sum(1 for r in fu if r.get('state'))} | {len(fu)} |")
    print(f"| compilation preserved the plan (not invalid / unsupported / error) | {sum(1 for r in ans if r.get('compiled'))} | {n} |")
    print(f"| final answer scored CORRECT (clarifications, refusals, wrong answers excluded) | {sum(1 for r in ans if r['answer_ok'])} | {n} |")
    print(f"| refusal-type questions handled correctly | {sum(1 for r in ref if r.get('refused_ok'))} | {len(ref)} |")
    first = collections.Counter()
    for r in ans:
        if r["answer_ok"]: continue
        first["capability not in prompt" if not r.get("caps") else "operation" if not r.get("op") else "target / event" if not r.get("target") else "period" if not r.get("period") else "restrictions / selection / statistic" if not r.get("restr") else "plan exact but answer wrong or not executed"] += 1
    print("\nEarliest failing stage among answerable questions that were not answered correctly:", dict(first.most_common()))
    cats = collections.Counter(c for r in ans if not r["answer_ok"] for c in r.get("cats", [])); print("All differences (a question can have several):", dict(cats.most_common()))
    if a.misses:
        for r in ans:
            if not r["answer_ok"]: print(f"-- {r['id']} {r['text'][:60]} | gold {r['gold'][:60]} | call {(r['call'] or '')[:110]} | op={r.get('op')} tgt={r.get('target')} per={r.get('period')} restr={r.get('restr')} | {r.get('cats')}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["id", "kind", "family", "followup", "caps", "op", "target", "period", "restr", "state", "compiled", "answer_ok", "outcome", "differences", "text", "gold", "call"])
            for r in rows: w.writerow([r["id"], r["kind"], r["family"], r["followup"], r.get("caps"), r.get("op"), r.get("target"), r.get("period"), r.get("restr"), r.get("state"), r.get("compiled"), r["answer_ok"], r.get("outcome"), ";".join(r.get("cats", [])), r["text"], r["gold"], r.get("call")])


if __name__ == "__main__": main()
