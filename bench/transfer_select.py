"""Transfer criterion of the pre-registered model-role experiment: on the new-signal fixture (washer fluid level, unknown to the built-in registry, known only through the overlay's label and synonyms) the frozen selector must choose the right reading on at least 8 of 10 questions.
Questions are written to use the registered synonyms in wordings the rule parser does not handle directly where possible, so that the candidate generator + selector path is what is exercised. Run once; the result is reported whatever it is.
    python tools_pin.py 8,9 bench/transfer_select.py --model models/LFM2.5-350M-Q4_K_M.gguf"""
import argparse, json, os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ["VGPT_QC"] = "all"
from bench import schema_fixtures
from bench.evalcore import parts
from vgpt import nl2ir, policy, schema_overlay as SO
from vgpt.brain import Brain
from vgpt.candidate_planner import CandidateSelector
from vgpt.query_ir import Query

NOW = 1790406000
W = "washer_fluid_level"
CASES = [("typical windscreen washer level over the last 10 days", f"op=aggregate tgt={W} agg=mean t=last:10:day"), ("has the washer fluid been dropping lately", f"op=trend tgt={W} t=last:14:day"), ("what is the screen wash level right now", f"op=current_value tgt={W} t=now"),
         ("lowest washer fluid since the start", f"op=extreme tgt={W} agg=min t=all"), ("when did the washer fluid level change the most", f"op=detect_change tgt={W} t=all"), ("how long was the screen wash level below 30 percent this week", f"op=percent_time t=cal:this_week c=sig:{W}:lt:30:%"),
         ("do you track the windscreen washer level", f"op=capability tgt={W} t=all"), ("average washer fluid yesterday", f"op=aggregate tgt={W} agg=mean t=cal:yesterday"), ("peak screen wash level last week", f"op=extreme tgt={W} agg=max t=last:7:day"),
         ("why has the washer fluid level changed", f"op=explain_change tgt={W} t=last:14:day")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-350M-Q4_K_M.gguf")); ap.add_argument("--out", default=os.path.join(HERE, "results", "transfer_select.json")); a = ap.parse_args()
    d = tempfile.mkdtemp(); made = schema_fixtures.build(d); spec = json.load(open(made["new_signal"][:-3] + ".overlay.json")); out = []
    with SO.overlay(spec):
        b = Brain(made["new_signal"], slm_ir=None, now=NOW); prof = b.qc().profile; sel = CandidateSelector(a.model, cache_dir=os.path.join(HERE, "results", "_cache"))
        for text, gold in CASES:
            pr = nl2ir.parse(text, now=NOW, tz=3600, profile=prof); d_, inf = policy.decide_by_selection(text, pr, sel, None, prof, NOW, 3600)
            g = parts(Query.from_text(gold)); chosen = d_.query if d_.kind == "answer" else None
            avail = any(parts(Query.from_text(c)) == g for c in (inf["candidates"] or [])) if inf["candidates"] is not None else (pr.ok and parts(pr.query) == g)
            hit_clar = d_.kind == "clarify" and any(parts(c) == g for c in d_.candidates)
            ok = (chosen is not None and parts(chosen) == g) or hit_clar
            out.append({"question": text, "gold": gold, "rules_ok": bool(pr.ok), "rules_correct": bool(pr.ok and parts(pr.query) == g), "decision": d_.kind, "chosen": chosen.to_text() if chosen else None, "clarify_offered": [c.to_text() for c in d_.candidates], "candidate_available": bool(avail), "correct_or_offered": bool(ok), "answered_correct": bool(chosen is not None and parts(chosen) == g)})
    n_ok = sum(r["correct_or_offered"] for r in out); n_ans = sum(r["answered_correct"] for r in out); n_av = sum(r["candidate_available"] for r in out)
    for r in out: print(("OK  " if r["correct_or_offered"] else "MISS"), r["decision"].ljust(8), "rules" if r["rules_correct"] else "     ", r["question"], "|", r["chosen"] or r["clarify_offered"])
    print(f"\ncandidate available {n_av}/10; right reading answered {n_ans}/10; right reading answered or offered {n_ok}/10 (criterion: at least 8)")
    json.dump({"summary": {"available": n_av, "answered_correct": n_ans, "correct_or_offered": n_ok}, "rows": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
