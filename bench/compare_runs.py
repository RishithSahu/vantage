"""Side-by-side table of eval_v2 result files: accuracy, answer coverage, wrong-among-answered and latency reported together, with 95% intervals.
    python bench/compare_runs.py results/abl_lfm_static.json results/abl_lfm_bank4.json ..."""
import json, os, sys


def row(path):
    d = json.load(open(path)); s = d["summary"]; rows = d["rows"]; ptok = [r.get("ptok") for r in rows if r.get("ptok")]
    ca, cl = s["correct_all"], s["correct_answerable"]; w = s["wrong_among_answered"]; lat = s["latency_ms"]
    return [os.path.basename(path).replace("abl_", "").replace(".json", ""), f"{100 * ca['rate']:.1f} ({100 * ca['ci95'][0]:.0f}-{100 * ca['ci95'][1]:.0f})", f"{100 * cl['rate']:.1f}", f"{100 * s['answered_coverage']['rate']:.1f}",
            f"{100 * w['rate']:.1f} ({w['k']}/{w['n']})", f"{100 * s['false_action_on_refusal_set']['rate']:.0f}", f"{100 * s['false_refusal_on_answerable']['rate']:.0f}", f"{lat['mean']:.0f}", f"{lat['p50']:.0f}", f"{lat['p95']:.0f}",
            f"{sum(ptok) / len(ptok):.0f}" if ptok else "-"]


hdr = ["run", "correct/all % (95% CI)", "correct/answerable %", "answered %", "wrong among answered", "false action on refusals %", "false refusal %", "ms mean", "p50", "p95", "prompt tokens (suffix)"]
table = [hdr] + [row(p) for p in sys.argv[1:]]; w = [max(len(r[i]) for r in table) for i in range(len(hdr))]
for r in table: print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
