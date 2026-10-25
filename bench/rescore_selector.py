"""Corrected scoring of the saved oracle-candidate selector results (results/sel_<name>.json) WITHOUT re-running any model, using bench/oracle_review.tsv (a review written from the candidate sets alone): DROP candidates are removed (top-1 is re-taken among the remaining candidates from the saved order-averaged scores - exact for the independent NLI scores, an approximation for the GLiNER classifiers whose scores depend on the label set), AMBIG questions are excluded from the denominator. The original numbers stay in the ledger.
    python bench/rescore_selector.py"""
import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def review():
    out = {}
    for l in open(os.path.join(HERE, "bench", "oracle_review.tsv"), encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        i, kind, cand, why = (l.rstrip("\n").split("\t") + ["", "", "", ""])[:4]; out.setdefault(i, []).append((kind, cand, why))
    return out


def main():
    rv = review()
    for k in ("nli", "small", "decide"):
        rows = json.load(open(os.path.join(HERE, "results", f"sel_{k}.json")))["rows"]; ok = n = amb = 0; orig = sum(r["top"] == r["gold"] for r in rows)
        for r in rows:
            ent = rv.get(r["id"], [])
            if any(e[0] == "AMBIG" for e in ent): amb += 1; continue
            drop = {e[1] for e in ent if e[0] == "DROP"}; sc = {c: v for c, v in r["scores"].items() if c not in drop and not c.startswith("op=")}; top = max(sc, key=sc.get); n += 1; ok += top == r["gold"]
        print(f"{k:7s} original {orig}/{len(rows)} = {100 * orig / len(rows):.0f}%   corrected (drop unfair candidates, exclude {amb} ambiguous questions) {ok}/{n} = {100 * ok / n:.0f}%")


if __name__ == "__main__": main()
