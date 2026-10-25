"""Paired 500-QA comparison of the v16 candidate against v14d (reference), v15 and legacy, with the FROZEN rubric and scorer (unchanged).
   python bench/qa500_score.py results/semantic_recovery/v16/runs/v16_learned_only.jsonl results/semantic_recovery/v16/runs/v16_hybrid.jsonl   # writes .scored.json
   python bench/qa500_compare_v16.py   -> results/semantic_recovery/v16/paired_summary.json, results/semantic_recovery/v16_per_question_comparison.csv"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "bench"))
import qa500_compare as C, qa500_score as S

R = C.R; V15 = os.path.join(R, "v15", "runs"); V16 = os.path.join(R, "v16", "runs")
CONFIGS = [("legacy", "legacy_ref", V15), ("v14d_lo", "v14d_s143_learned_only", V15), ("v14d_hy", "v14d_s143_hybrid", V15), ("v15_lo", "v15_learned_only", V15), ("v15_hy", "v15_hybrid", V15),
           ("v16_lo", "v16_learned_only", V16), ("v16_hy", "v16_hybrid", V16)]


def load(name, d):
    recs = {}
    for l in open(os.path.join(d, name + ".jsonl"), encoding="utf8"):
        j = json.loads(l); recs[j["id"]] = j
    return recs, {s["id"]: s for s in json.load(open(os.path.join(d, name + ".scored.json")))}


def main():
    recs, sc = {}, {}
    for k, n, d in CONFIGS: recs[k], sc[k] = load(n, d)
    ids = list(S.qa); sl = {i: C.slice_of(S.qa[i]) for i in ids}
    cols = ["id", "category", "slice", "rubric_type", "question"]
    for k, _, _ in CONFIGS: cols += [f"{k}_pass", f"{k}_route", f"{k}_greeting_text"]
    cols += ["v16lo_vs_v14dlo", "v16hy_vs_v14dhy", "v16lo_vs_v14dhy"]
    cmp = lambda a, b, i: "improved" if sc[b][i]["ok"] and not sc[a][i]["ok"] else ("regressed" if sc[a][i]["ok"] and not sc[b][i]["ok"] else "tie")
    with open(os.path.join(R, "v16_per_question_comparison.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for i in ids:
            row = [i, S.qa[i]["category"], sl[i], S.rub[i]["type"], S.qa[i]["question"]]
            for k, _, _ in CONFIGS: row += [int(sc[k][i]["ok"]), recs[k][i].get("route"), int((recs[k][i].get("answer") or "").startswith("Hello - I answer questions"))]
            row += [cmp("v14d_lo", "v16_lo", i), cmp("v14d_hy", "v16_hy", i), cmp("v14d_hy", "v16_lo", i)]
            w.writerow(row)
    out = {"overall": {k: S.summarise(k, [sc[k][i] for i in ids]) for k, _, _ in CONFIGS}, "by_slice": {}, "paired": {}}
    for s in sorted(set(sl.values())):
        sid = [i for i in ids if sl[i] == s]; out["by_slice"][s] = {"n": len(sid), **{k: sum(sc[k][i]["ok"] for i in sid) for k, _, _ in CONFIGS}}
    for a, b in (("v16_lo", "v14d_lo"), ("v16_hy", "v14d_hy"), ("v16_lo", "v14d_hy"), ("v16_hy", "v15_hy"), ("v16_hy", "legacy")):
        out["paired"][f"{a} vs {b}"] = {"all": C.paired(ids, sc[b], sc[a])}
        for s in sorted(set(sl.values())):
            sid = [i for i in ids if sl[i] == s]; out["paired"][f"{a} vs {b}"][s] = C.paired(sid, sc[b], sc[a])
    out["greeting_text_on_unsupported_unsafe"] = {k: sum(1 for i in ids if sl[i] == "unsupported_or_unsafe_request" and (recs[k][i].get("answer") or "").startswith("Hello - I answer questions")) for k, _, _ in CONFIGS}
    json.dump(out, open(os.path.join(R, "v16", "paired_summary.json"), "w"), indent=1)
    for k, _, _ in CONFIGS: s = out["overall"][k]; print(f"{k:8s} {s['correct']}/500 {s['rate']:.3f} CI {s['ci95']} fabricated {s['suspect_fabrication']} grounded {s['grounded_executed']}")
    for k, v in out["paired"].items(): print(k, v["all"])
    print("greeting text on unsupported/unsafe:", out["greeting_text_on_unsupported_unsafe"])
    for s, v in out["by_slice"].items(): print(f"  {s:34s} n={v['n']:3d} " + " ".join(f"{k}={v[k]}" for k, _, _ in CONFIGS))


if __name__ == "__main__": main()
