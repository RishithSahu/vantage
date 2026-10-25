"""Paired per-question comparison of the five 500-QA configurations (frozen rubric, stored raw runs).
   python bench/qa500_compare.py -> results/semantic_recovery/v15_per_question_comparison.csv + v15/paired_summary.json
Slices are assigned from the question text, the reference/evaluation note and the product's implemented capabilities only (never from candidate correctness)."""
import collections, csv, json, math, os, random, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HERE, "results", "semantic_recovery"); RUNS = os.path.join(R, "v15", "runs")
sys.path.insert(0, os.path.join(HERE, "bench"))
import qa500_score as S
qa = S.qa; rub = S.rub
CONFIGS = [("legacy", "legacy_ref"), ("v13_lo", "v13_learned_only"), ("v14d_lo", "v14d_s143_learned_only"), ("v14d_hy", "v14d_s143_hybrid"), ("v15_lo", "v15_learned_only"), ("v15_hy", "v15_hybrid")]

# ---- slice assignment (question + category + note only)
CAP_RX = re.compile(r"\b(my|this vehicle|the car|the vehicle|our)\b", re.I)
MISSING_RX = re.compile(r"not available|unavailable|if (?:it is|they are|the data is) (?:logged|recorded|available)|missing|do not invent|not logged|cannot be determined|if unavailable|not recorded", re.I)
def slice_of(r):
    cat, q, ref, note = r["category"], r["question"], r["answer"], r["evaluation_note"]
    t = rub[r["id"]]["type"]
    if cat == "vehicle_data_queries": return "telemetry_missing" if MISSING_RX.search(ref) else "telemetry"
    if cat == "driving_safety": return "safety_critical"
    if cat == "out_of_scope_privacy_safety": return "unsupported_or_unsafe_request"
    if cat in ("general_knowledge",): return "out_of_scope_general"
    if cat == "reasoning_math_language": return "out_of_scope_general" if not re.search(r"\b(ev|battery|charg|kwh|wh/km|vehicle|car|tyre|tire|motor|range)\b", q, re.I) else "vehicle_calculation"
    if cat == "ev_energy_charging": return "vehicle_calculation" if t == "arith" else "vehicle_domain_knowledge"
    if cat == "vehicle_concepts": return "vehicle_domain_knowledge"
    if cat == "complex_scenarios": return "vehicle_domain_reasoning"
    if cat == "camera_gps_weather": return "capability_camera_gps_weather" if CAP_RX.search(q) and re.search(r"\b(camera|gps|weather|rain|lane|location)\b", q, re.I) else "domain_knowledge_camera_gps_weather"
    if cat == "conversation_robustness": return "dialogue_robustness"
    return "other"


def load(name):
    recs = {}
    for l in open(os.path.join(RUNS, name + ".jsonl"), encoding="utf8"):
        j = json.loads(l); recs[j["id"]] = j
    sc = {s["id"]: s for s in json.load(open(os.path.join(RUNS, name + ".scored.json")))}
    return recs, sc


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(ids, A, B, seed=7, reps=4000):
    d = [int(B[i]["ok"]) - int(A[i]["ok"]) for i in ids]
    n = len(d); imp = sum(1 for x in d if x > 0); reg = sum(1 for x in d if x < 0); tie = n - imp - reg
    rnd = random.Random(seed); bs = []
    for _ in range(reps):
        s = sum(d[rnd.randrange(n)] for _ in range(n)); bs.append(s / n)
    bs.sort()
    return {"n": n, "improved": imp, "regressed": reg, "tied": tie, "mean_diff": round(sum(d) / n, 4), "ci95": [round(bs[int(0.025 * reps)], 4), round(bs[int(0.975 * reps)], 4)], "mcnemar_p": round(mcnemar_exact(imp, reg), 4)}


def main():
    sc = {}; recs = {}
    for k, name in CONFIGS: recs[k], sc[k] = load(name)
    ids = list(qa)
    sl = {i: slice_of(qa[i]) for i in ids}
    with open(os.path.join(R, "v15_per_question_comparison.csv"), "w", newline="", encoding="utf8") as fh:
        cols = ["id", "category", "difficulty", "slice", "rubric_type", "question"]
        for k, _ in CONFIGS: cols += [f"{k}_pass", f"{k}_why", f"{k}_route", f"{k}_domain_refusal"]
        cols += ["v15lo_vs_v14dlo", "v15lo_vs_v14dhy", "v15hy_vs_v14dhy", "v15hy_vs_legacy", "critical_safety_regression_v15lo", "critical_safety_regression_v15hy"]
        w = csv.writer(fh); w.writerow(cols)
        for i in ids:
            row = [i, qa[i]["category"], qa[i]["difficulty"], sl[i], rub[i]["type"], qa[i]["question"]]
            for k, _ in CONFIGS: s = sc[k][i]; row += [int(s["ok"]), s["why"], recs[k][i].get("route"), int(s["domain_refusal"])]
            cmp = lambda a, b: "improved" if sc[b][i]["ok"] and not sc[a][i]["ok"] else ("regressed" if sc[a][i]["ok"] and not sc[b][i]["ok"] else "tie")
            cs = lambda b: int(sl[i] in ("safety_critical", "unsupported_or_unsafe_request") and (sc["v14d_hy"][i]["ok"] and not sc[b][i]["ok"] or (sc[b][i]["suspect_fabrication"] and not sc["v14d_hy"][i]["suspect_fabrication"])))
            row += [cmp("v14d_lo", "v15_lo"), cmp("v14d_hy", "v15_lo"), cmp("v14d_hy", "v15_hy"), cmp("legacy", "v15_hy"), cs("v15_lo"), cs("v15_hy")]
            w.writerow(row)
    out = {"slices": dict(collections.Counter(sl.values())), "overall": {}, "by_slice": {}, "paired": {}}
    for k, _ in CONFIGS: out["overall"][k] = S.summarise(k, [sc[k][i] for i in ids])
    for sname in sorted(set(sl.values())):
        sid = [i for i in ids if sl[i] == sname]
        out["by_slice"][sname] = {"n": len(sid), **{k: sum(sc[k][i]["ok"] for i in sid) for k, _ in CONFIGS}}
    pairs = [("v15_lo", "v14d_lo"), ("v15_lo", "v14d_hy"), ("v15_hy", "v14d_hy"), ("v15_lo", "v13_lo"), ("v15_hy", "legacy"), ("v14d_lo", "v13_lo")]
    for a, b in pairs:
        out["paired"][f"{a} vs {b}"] = {"all": paired(ids, sc[b], sc[a])}
        for sname in sorted(set(sl.values())):
            sid = [i for i in ids if sl[i] == sname]; out["paired"][f"{a} vs {b}"][sname] = paired(sid, sc[b], sc[a])
    # latency from the stored runs (CPU contention: four to six runs shared the machine)
    import statistics
    out["latency_ms_median_p90"] = {k: [round(statistics.median(r["ms"] for r in recs[k].values())), round(sorted(r["ms"] for r in recs[k].values())[int(0.9 * 500)])] for k, _ in CONFIGS}
    out["errors"] = {k: sum(1 for r in recs[k].values() if r.get("error")) for k, _ in CONFIGS}
    json.dump(out, open(os.path.join(R, "v15", "paired_summary.json"), "w"), indent=1)
    print(json.dumps({"slices": out["slices"], "by_slice": out["by_slice"], "errors": out["errors"], "latency": out["latency_ms_median_p90"]}, indent=1))
    for k, v in out["paired"].items(): print(k, v["all"])


if __name__ == "__main__": main()
