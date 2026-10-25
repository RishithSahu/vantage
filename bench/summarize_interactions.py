"""Summary of the interpretation-review study log (interaction_log.jsonl written by the prototype, one line per proposal / execution / verdict) against bench/user_study/tasks.tsv.
Per participant and task, from what people actually did:
  unassisted  the FIRST proposal was run without any edit and is correct (automatic: the confirmed query equals the gold; and/or the participant's own verdict)
  assisted    the LAST executed interpretation is correct after the participant's edits
  edits       number of changed fields over the task;  time: proposal wait + review time (client clocks), and the participant's total when judged
  abandoned   a proposal was shown but nothing was ever run for that task
Both views are printed: by the gold (automatic) and by the participant's verdict - they can disagree, and that is information. NOTHING here is simulated: no oracle clarification, no author-run tasks counted as users.
    python bench/summarize_interactions.py LOG.jsonl [--tasks bench/user_study/tasks.tsv] [--db data/gen_courier_ext.db]"""
import argparse, collections, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_format import diff_fields, normalise
from training.build_dataset import DEFAULT_NOW
from vgpt import capreg, plancompile as PC, signal_registry as R, toolspec as T
from vgpt.query_ir import Query


def load_tasks(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        t = l.rstrip("\n").split("\t"); out[t[0]] = {"category": t[1], "goal": t[2], "gold": t[3]}
    return out


def correct_by_gold(final, gold, reg):
    """The confirmed typed call compiles to the gold query (the evaluator's field comparison)."""
    if gold == "op=none": return final["fn"] == "decline"
    if gold == "op=unresolved": return final["fn"] in ("unresolved", "ask_user") or False
    try:
        o = PC.compile_plan(T.call_to_plan(final["fn"], final["args"]), None, None, reg, DEFAULT_NOW, 0)
        if o.kind != "query": return False
        gq = Query.from_text(gold); normalise(gq, o.query); return not diff_fields(gq, o.query)
    except Exception: return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--tasks", default=os.path.join(HERE, "bench", "user_study", "tasks.tsv")); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); a = ap.parse_args()
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{a.db}?mode=ro", uri=True))); tasks = load_tasks(a.tasks); ev = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    by = collections.defaultdict(list)
    for e in ev: by[(e.get("participant", ""), e.get("task", ""))].append(e)
    rows = []
    for (who, task), es in sorted(by.items()):
        ex = [e for e in es if e["event"] == "executed"]; ju = [e for e in es if e["event"] == "judged"]; gold = (tasks.get(task) or ({} if task.endswith("a") else tasks.get(task[:-1], {}) if task.endswith("b") else {})).get("gold") or None; gave_up = any(e["event"] == "abandoned" for e in es)
        if not ex: rows.append({"who": who, "task": task, "abandoned": True, "gave_up": gave_up}); continue
        first, last = ex[0], ex[-1]; auto = lambda e: correct_by_gold(e["final"], gold, reg) if gold else None
        verdict = ju[-1]["verdict"] if ju else None
        rows.append({"who": who, "task": task, "category": (tasks.get(task) or tasks.get(task[:-1]) or {}).get("category"), "abandoned": False, "runs": len(ex), "edits": sum(e["n_edits"] for e in ex), "first_unedited": first["n_edits"] == 0, "first_auto": auto(first), "last_auto": auto(last), "verdict": verdict, "rebuilt": bool(last.get("rebuilt")), "sat": next((j["satisfaction"] for j in reversed(ju) if j.get("satisfaction")), None), "own": task.startswith("own"), "gave_up": gave_up,
                     "wait_s": round(sum((e.get("client") or {}).get("propose_ms", 0) for e in ex) / 1000, 1), "edit_s": round(sum((e.get("client") or {}).get("review_ms", 0) for e in ex) / 1000, 1),
                     "time_s": round(sum((e.get("client") or {}).get("review_ms", 0) + (e.get("proposal", {}).get("timing") or {}).get("plan_ms", 0) for e in ex) / 1000, 1), "total_s": round(((ju[-1].get("client") or {}).get("total_ms", 0)) / 1000, 1) if ju else None})
    n = [r for r in rows if not r.get("abandoned")]; ab = [r for r in rows if r.get("abandoned")]; N = len(rows)
    if not N: print("no interactions"); return
    pct = lambda k: f"{k}/{N} = {100 * k / N:.0f}%"
    print(f"{N} participant-tasks from {len({r['who'] for r in rows})} participant(s); abandoned (proposal shown, nothing run): {len(ab)}")
    print(f"UNASSISTED (first proposal, no edit) correct by gold: {pct(sum(1 for r in n if r['first_unedited'] and r['first_auto']))};  by the participant's verdict: {pct(sum(1 for r in n if r['first_unedited'] and r['verdict'] == 'yes'))}")
    print(f"ASSISTED (final run after edits) correct by gold: {pct(sum(1 for r in n if r['last_auto']))};  by the participant's verdict: {pct(sum(1 for r in n if r['verdict'] == 'yes'))}")
    if n:
        corr = [r for r in n if not r["rebuilt"]]; reb = [r for r in n if r["rebuilt"]]; sats = [r["sat"] for r in n if r["sat"]]
        print(f"SEPARATELY: first proposal right (no edit, participant says yes) {sum(1 for r in n if r['first_unedited'] and r['verdict'] == 'yes')}/{N};  met the goal after CORRECTING (not rebuilt) {sum(1 for r in corr if r['verdict'] == 'yes' and not r['first_unedited'])};  met the goal but the person REBUILT the query {sum(1 for r in reb if r['verdict'] == 'yes')} (not counted as the model's help);  gave up {sum(1 for r in rows if r.get('gave_up'))};  satisfaction mean {sum(sats) / len(sats):.1f} over {len(sats)} ratings" if sats else f"SEPARATELY: first proposal right {sum(1 for r in n if r['first_unedited'] and r['verdict'] == 'yes')}/{N}; corrected {sum(1 for r in corr if r['verdict'] == 'yes' and not r['first_unedited'])}; rebuilt {sum(1 for r in reb if r['verdict'] == 'yes')}; gave up {sum(1 for r in rows if r.get('gave_up'))}; no satisfaction ratings yet")
        print(f"TIME per task: waiting for the model {sum(r['wait_s'] for r in n) / len(n):.1f} s, reviewing/editing {sum(r['edit_s'] for r in n) / len(n):.1f} s  (the wait is {100 * sum(r['wait_s'] for r in n) / max(sum(r['wait_s'] + r['edit_s'] for r in n), 1e-9):.0f}% of the two);  own-goal tasks: {sum(1 for r in n if r['own'])}")
    if n: print(f"edits per task: mean {sum(r['edits'] for r in n) / len(n):.1f}; runs per task {sum(r['runs'] for r in n) / len(n):.1f}; review+proposal time per task {sum(r['time_s'] for r in n) / len(n):.0f} s; participants disagreeing with the gold: {sum(1 for r in n if r['verdict'] and (r['verdict'] == 'yes') != bool(r['last_auto']))}")
    cat = collections.defaultdict(list)
    for r in n: cat[r["category"]].append(r)
    for c, rs in sorted(cat.items(), key=lambda x: str(x[0])): print(f"  {str(c):28s} n={len(rs)}  unassisted {sum(1 for r in rs if r['first_unedited'] and r['first_auto'])}  assisted {sum(1 for r in rs if r['last_auto'])}  mean edits {sum(r['edits'] for r in rs) / len(rs):.1f}")


if __name__ == "__main__": main()
