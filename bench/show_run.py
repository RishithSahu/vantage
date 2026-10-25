"""Compact view of a results/m*_*.json written by bench/eval_model.py (headline numbers, stage attribution, timings; optionally the misses).   python bench/show_run.py results/x.json [--misses]"""
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8")); s = d["summary"]; g = lambda k: s[k]
print(f'{s["system"]} on {s["set"]}  (n={s["n"]}, dialect {s.get("dialect")}, ctx {s.get("n_ctx")})')
print(" outcomes (scored):", s["outcomes"], "| route outcomes:", s["outcomes_model_route"])
ca, cv, an = s["correct_all"], s["correct_answerable"], s["answered_coverage"]
print(f' correct overall {ca["k"]}/{ca["n"]} ({100 * ca["rate"]:.0f}%)  answerable {cv["k"]}/{cv["n"]}  answered {an["k"]}/{an["n"]}  wrong among answered {s["wrong_among_answered"]["k"]}/{s["wrong_among_answered"]["n"]}  false action on refusal set {s["false_action_on_refusal_set"]["k"]}/{s["false_action_on_refusal_set"]["n"]}  false refusal {s["false_refusal_on_answerable"]["k"]}/{s["false_refusal_on_answerable"]["n"]}')
print(" clarification:", {k: s["clarification"][k] for k in ("asked", "on_answerable", "unneeded")})
print(" earliest failing stage:", s["earliest_failing_stage"]); print(" retrieval recall:", s["retrieval_recall"])
print(" wrong fields:", s["wrong_fields"]); print(" by family:", {f: v["correct"] for f, v in s["by_family"].items()})
t = s["timing_ms"]; print(" timing ms (mean/p95):", {k: (t[k]["mean"], t[k]["p95"]) for k in ("retrieve_ms", "state_ms", "prefill_ms", "decode_ms", "generate_ms", "execute_ms", "total_ms")}, "tokens", s["tokens"], "prefix", s["prefix_tokens"], "wall", s["wall_s"], "s")
if "--misses" in sys.argv:
    rec = {(r["text"], r["prev_text"]): r for r in d["records"]}
    for r in d["rows"]:
        if r["correct"]: continue
        c = rec.get((r["text"], r["prev_text"]), {}); print(f'-- {r["id"]} [{r.get("stage")}] {r["text"][:70]} | prev: {(r["prev_text"] or "")[:30]}\n     gold: {r["gold"][:100]}\n     call: {(c.get("call") or "")[:200]} -> {c.get("outcome")}')
