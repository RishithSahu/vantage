"""Builds the three result tables of the final baseline from results/f1_*.json (nothing typed by hand):
  results/table_final.txt        one line per system and set (correct of all questions; answers / wrong; clarifications / containing the intended reading; oracle after one turn; false actions; latency)
  results/table_errors.txt       wrong fields, family breakdown and the answerable-question outcome categories (clarified WITH vs WITHOUT the intended reading are separate)
  results/table_paired_final.txt paired comparisons (bench/paired.py, exploratory) between systems on the same questions
    python bench/make_final_tables.py
"Correct after one extra turn" is an ORACLE upper bound: it assumes the person always recognises and picks the intended reading when one is offered."""
import collections, json, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HERE, "results")
SYSTEMS = ["rules", "policyrules", "product", "select", "policyqwen", "policylfm"]; SETS = ["dev", "diag", "final"]
LABEL = {"rules": "rules alone", "policyrules": "rules + generator + clarification policy (no model) [product configuration, interpretation + typed pipeline only]", "product": "the product configuration END TO END through vgpt.product.make_brain (gates, policy, typed pipeline, older-engine routing)", "select": "+ frozen LFM candidate selection [experimental, disabled]", "policyqwen": "+ Qwen-0.5B + LoRA whole query [experimental comparator]", "policylfm": "+ frozen LFM whole-query generation with the rules' draft [experimental]"}


def load(sys_, s):
    p = os.path.join(R, f"f1_{sys_}_{s}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def line(name, r):
    m = r["summary"]; o = m["outcomes"]; c = m["clarification"]; fa = m["false_action_on_refusal_set"]; lat = m["latency_ms"]; dfr = sum(1 for x in r["rows"] if x.get("source") == "status")
    return (f"{name:<22} n={m['n']:>3} correct {m['correct_all']['k']:>3} | answers {o['answer']:>3} (wrong {m['wrong_among_answered']['k']:>2}) explained {o['explained']:>2} refusals {o['refusal']:>3} (of which deferred to the older status engine {dfr}) clarify {o['clarify']:>3} (hit {c['contains_intended_reading']:>2}, unneeded {c['unneeded']}) "
            f"| false-action {fa['k']}/{fa['n']} | plan-exact {m['plan_exact']['k']:>3} | after-1-turn(oracle) {c['correct_after_one_extra_turn']:>3} | ms mean {lat['mean']:>7.1f} p95 {lat['p95']:>7.1f}")


def categories(rows):
    c = collections.Counter()
    for x in rows:
        if x["gold_kind"] == "refusal": continue
        if x["correct"]: c["correct"] += 1
        elif x["kind"] == "clarify": c["clarified, intended reading offered" if x.get("clarify_hit") else "clarified, NO intended reading"] += 1
        elif x["wrong_answer"]: c["wrong answer"] += 1
        elif x.get("wrong_non_answer"): c["wrong explained limitation"] += 1
        else: c["refused / no reading"] += 1
    return dict(c)


def main():
    out = []; err = []
    for s in SETS:
        out.append(f"== {s}")
        for sy in SYSTEMS:
            r = load(sy, s)
            if r: out.append(line(f"{sy}_{s}", r))
    for s in ("own_dev", "own_final"):
        r = load("select", s)
        if r: out.append(f"== select with the system's OWN previous interpretation as history ({s})\n" + line(f"select_{s}", r))
    open(os.path.join(R, "table_final.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    for s in ("dev", "final"):
        for sy in SYSTEMS:
            r = load(sy, s)
            if not r: continue
            m = r["summary"]; fam = collections.defaultdict(lambda: [0, 0, 0, 0])
            for x in r["rows"]:
                f = fam[x["family"]]; f[0] += 1; f[1] += bool(x["correct"]); f[2] += bool(x["wrong_answer"]); f[3] += x["kind"] == "clarify"
            err.append(f"== {sy}_{s}: wrong fields {m['wrong_fields']}")
            err.append("   family n/correct/wrong/clarify: " + "  ".join(f"{k} {v[0]}/{v[1]}/{v[2]}/{v[3]}" for k, v in sorted(fam.items())))
            err.append(f"   answerable: {categories(r['rows'])}")
            fr = [x for x in r["rows"] if x["false_refusal"]]
            if fr: err.append("   false refusals (answerable question refused): " + "; ".join(f"{x['id']} '{x['text'][:60]}' [{x.get('source')}]" for x in fr[:12]))
    open(os.path.join(R, "table_errors.txt"), "w", encoding="utf-8").write("\n".join(err) + "\n")
    pr = []
    for s in ("dev", "final"):
        for a, b in (("rules", "policyrules"), ("policyrules", "select"), ("policyrules", "policyqwen"), ("policyrules", "product")):
            fa, fb = os.path.join(R, f"f1_{a}_{s}.json"), os.path.join(R, f"f1_{b}_{s}.json")
            if not (os.path.exists(fa) and os.path.exists(fb)): continue
            pr.append(f"=== {s}: {a} (A) vs {b} (B)")
            p = subprocess.run([sys.executable, os.path.join(HERE, "bench", "paired.py"), fa, fb], capture_output=True, text=True, cwd=HERE); pr.append((p.stdout or p.stderr).strip())
    dv = []
    for s in SETS:
        a, b = load("policyrules", s), load("product", s)
        if not (a and b): continue
        A = {x["id"]: x for x in a["rows"]}; dv.append(f"== {s}: questions where the end-to-end product differs from the interpretation-only harness (kind / correct / clarify-hit)")
        for x in b["rows"]:
            y = A[x["id"]]
            if (y["kind"], y["correct"], y.get("clarify_hit")) != (x["kind"], x["correct"], x.get("clarify_hit")): dv.append(f"  {x['id']} '{x['text'][:70]}' harness: {y['kind']}/{y['correct']}  product: {x['kind']}/{x['correct']} [{x.get('source')}]")
        sm = b["summary"]; dv.append(f"  routes {sm.get('routes')}; typed-pipeline status {sm.get('typed_pipeline_status')}; left the typed pipeline: " + "; ".join(f"'{z['text'][:40]}' ({z['route']}/{z['intent']})" for z in sm.get("left_typed_pipeline", [])))
    open(os.path.join(R, "product_vs_harness.txt"), "w", encoding="utf-8").write("\n".join(dv) + "\n")
    open(os.path.join(R, "table_paired_final.txt"), "w", encoding="utf-8").write("\n".join(pr) + "\n"); print("tables written")


if __name__ == "__main__": main()
