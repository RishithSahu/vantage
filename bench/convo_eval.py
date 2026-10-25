"""Runs scripted conversations through the real system (Brain with the query-compiler path) so that every turn sees the system's OWN accepted state, and prints the update trace of each turn (what changed, what was inherited).
    python tools_pin.py 4,5 bench/convo_eval.py [--out results/convo_dev.json] [--db data/gen_courier_ext.db]
Scoring per turn: the planned query equals the expected IR field by field (bench/evalcore.parts), or the expected CLARIFY / REFUSE outcome occurred. A conversation succeeds only if every turn does. A wrong turn does not receive gold history: later turns inherit from what the system accepted."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import importlib
from bench.convos_dev import CONVOS
from bench.evalcore import parts
from vgpt.brain import Brain
from vgpt.query_ir import Query

NOW = 1790406000
os.environ["VGPT_QC"] = "all"                                                                   # the query-compiler path answers every turn (the older engines are not used for this evaluation)


def run(db, convos=CONVOS, model=None, policy="legacy", selector=None):
    out = []
    for cid, turns in convos.items():
        b = Brain(db, slm_ir=model, now=NOW, policy=policy, selector=selector); recs = []; ok_all = True            # the query-compiler path answers every turn (the older engines are not used for this evaluation)
        for i, (utt, want) in enumerate(turns, 1):
            n0 = len(b.trace.records); res = b.ask(utt, use_slm=model is not None)
            if len(b.trace.records) > n0: rec = b.trace.records[-1]; outcome = rec.outcome; ir = rec.ir; updates = rec.updates; inherited = rec.inherited
            else: outcome = "refused"; ir = ""; updates = []; inherited = []; b.trace.record(utt, None, "refused", (res.get("answer") or "")[:80])
            if want == "CLARIFY": ok = outcome == "clarify"
            elif want == "REFUSE": ok = outcome == "refused"
            else:
                try: ok = bool(ir) and outcome in ("answered", "no_data", "limitation") and parts(Query.from_text(ir)) == parts(Query.from_text(want))
                except Exception: ok = False
            ok_all &= ok; recs.append({"turn": i, "utterance": utt, "expected": want, "outcome": outcome, "ir": ir, "ok": ok, "updates": updates, "inherited": inherited})
        out.append({"conversation": cid, "success": ok_all, "turns": recs, "trace": b.trace.describe()})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); ap.add_argument("--verbose", action="store_true"); ap.add_argument("--set", default="dev", choices=["dev", "check"]); ap.add_argument("--policy", default="legacy", choices=["legacy", "clarify"]); ap.add_argument("--selector", default=""); a = ap.parse_args()
    sel = None
    if a.selector:
        from vgpt.candidate_planner import CandidateSelector
        sel = CandidateSelector(a.selector, cache_dir=os.path.join(HERE, "results", "_cache"))
    res = run(a.db, importlib.import_module("bench.convos_" + a.set).CONVOS, policy=a.policy, selector=sel); turns = [t for c in res for t in c["turns"]]
    for c in res:
        print(("PASS " if c["success"] else "FAIL ") + c["conversation"])
        for t in c["turns"]:
            if a.verbose or not t["ok"]: print(f"   turn {t['turn']} {'ok ' if t['ok'] else 'BAD'} {t['utterance']!r}: got {t['outcome']} {t['ir'] or '-'}  | expected {t['expected']}")
    print(f"\nconversations {sum(c['success'] for c in res)}/{len(res)}; turns {sum(t['ok'] for t in turns)}/{len(turns)}")
    if a.out: os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
