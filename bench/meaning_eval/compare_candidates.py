"""One table for several trained adapters: everything measured about each, side by side.

    python bench/meaning_eval/compare_candidates.py full_v11:v11 frame2 frame2b frame2m frame2c [--out results/meaning/candidates.md]

Each name is the prefix of the result folders written by post_training.py / run_frame2.py (`frame2` -> results/meaning/frame2_*, v2_frame2*.json, answerability_frame2.json, variety_frame2.json).  `full_v11:v11` is the previous
product, read from its own older result folders.  Numbers: exact frame in per cent (strict, and with the catalogue-equivalence rule for the older sets), answerability per vehicle class, variety.  Missing results are shown as '-'."""
import argparse
import collections
import io
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); RES = os.path.join(ROOT, "results", "meaning")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "bench"))
import rescore_v3 as R  # noqa: E402


def older(prefix):
    n = s = a = 0
    for name, (folder, v11, key) in R.SETS.items():
        rows = R.load_preds(prefix + "_" + folder, key) if prefix != "v11" else R.load_preds(v11, key)
        if rows is None: return None
        if prefix == "v11": n += len(rows); s += sum(1 for r in rows if r["ok"]); a += sum(1 for r in rows if r["ok"]); continue
        st, mv, _ = R.score(rows); n += len(rows); s += st; a += st + mv
    return (100.0 * s / n, 100.0 * a / n, n) if n else None


def ext(prefix):
    files = [os.path.join(RES, f"v2_{prefix}_ext.json"), os.path.join(RES, f"v2_{prefix}.json"), os.path.join(RES, f"v2_{prefix}_fresh.json")]; rows = []; seen = set()
    for p in files:
        if os.path.exists(p):
            for r in json.load(io.open(p, encoding="utf-8"))["rows"]:
                if (r["set"], r["id"]) not in seen: seen.add((r["set"], r["id"])); rows.append(r)
    by = collections.defaultdict(list)
    for r in rows: by[r["set"]].append(r)
    out = {k: 100.0 * sum(1 for r in v if r.get("exact")) / len(v) for k, v in by.items()}
    fresh = [r for r in rows if r.get("tag") == "fresh"]
    if fresh: out["fresh"] = 100.0 * sum(1 for r in fresh if r.get("exact")) / len(fresh)
    return out


def ice(prefix):
    p = os.path.join(RES, prefix + "_ice", "predictions.json")
    if not os.path.exists(p): return {}
    d = json.load(io.open(p, encoding="utf-8")); return {k: 100.0 * v["exact"] for k, v in d["summary"].items()}


def answ(prefix):
    p = os.path.join(RES, f"answerability_{prefix}.json")
    if not os.path.exists(p): return None
    sys.path.insert(0, HERE); import answerability as A
    rows = json.load(io.open(p, encoding="utf-8")); truths = {}; by = collections.defaultdict(list)
    for x in rows:
        if x["dataset"] not in truths: truths[x["dataset"]] = A.truth(os.path.join(ROOT, "data", x["dataset"]))
        have = truths[x["dataset"]]; got = A.classify(x["answer"]); want = "answered" if x["concept"] in have else ("either" if A.EITHER.get(x["concept"]) in have else "declined")
        about = A.names_the_quantity(x["concept"], x["answer"]) if got == "answered" else None
        ok = want == "either" or (got == want and not (got == "answered" and want == "answered" and not about)); by[x["class"].split(" (")[0]].append((ok, want == "declined" and got == "answered"))
    return {k: (100.0 * sum(o for o, _ in v) / len(v), sum(f for _, f in v)) for k, v in by.items()}


def variety(prefix):
    p = os.path.join(RES, f"variety_{prefix}.json")
    if not os.path.exists(p): return None
    import answer_variety as V
    rows = json.load(io.open(p, encoding="utf-8")); g = collections.defaultdict(list)
    for r in rows: g[(r["dataset"], r["group"])].append(r)
    rep = pairs = 0
    for v in g.values():
        for a, b in itertools.combinations(v, 2):
            wa, wb = V.words(a["answer"]), V.words(b["answer"]); pairs += 1; rep += (len(wa & wb) / max(1, min(len(wa), len(wb))) >= 0.8)
    return 100.0 * sum(1 for r in rows if r["grounded"]) / len(rows), rep, pairs


def fmt(x, nd=1): return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--out", default=os.path.join(RES, "candidates.md")); a = ap.parse_args()
    cols = []
    for spec in a.names:
        name, _, alias = spec.partition(":"); alias = alias or name; pre = alias if alias == "v11" else name
        o = older(pre); e = ext(name) if alias != "v11" else {}; ic = ice(name) if alias != "v11" else {}; an = answ(name) if alias != "v11" else None; va = variety(name if alias != "v11" else "v11_baseline")
        cols.append((alias, o, e, ic, an, va))
    head = "| measure | " + " | ".join(c[0] for c in cols) + " |"; sep = "|---|" + "---|" * len(cols); lines = [head, sep]
    def row(label, fn): lines.append(f"| {label} | " + " | ".join(fn(c) for c in cols) + " |")
    row("15 older sets, pooled, strict", lambda c: fmt(c[1][0]) if c[1] else "-")
    row("15 older sets, pooled, catalogue rule", lambda c: fmt(c[1][1]) if c[1] else "-")
    for k in ("slots_v2", "followup_v2", "digest_v1", "catalog_v1", "followup_v3", "followup_fresh_v4", "fresh"): row(k + " exact", lambda c, k=k: fmt(c[2].get(k)) if c[2] else "-")
    for k in ("icevocab_diesel_v1", "icevocab_petrol_v1", "longwrap_v1"): row(k, lambda c, k=k: fmt(c[3].get(k)) if c[3] else "-")
    classes = sorted({k for c in cols if c[4] for k in c[4]})
    for k in classes: row(f"answerability {k} (false answers)", lambda c, k=k: (f"{c[4][k][0]:.1f} ({c[4][k][1]})" if c[4] and k in c[4] else "-"))
    row("variety: grounded %", lambda c: fmt(c[5][0]) if c[5] else "-"); row("variety: repeated pairs", lambda c: f"{c[5][1]} of {c[5][2]}" if c[5] else "-")
    text = "\n".join(lines); print(text); io.open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
