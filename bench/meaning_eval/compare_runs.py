"""Row-level comparison of two evaluated runs on the same set (what v2 fixed, what it broke).

    python bench/meaning_eval/compare_runs.py full full_v2 novel_v1 messy_v1 user_v1 cs5        # runs under results/meaning/<name>/predictions.json
    python bench/meaning_eval/compare_runs.py full_sealed full_v2_sealed sealed_v2 --list       # --list prints every flipped row

Counts rows right in A only (A+), in B only (B+), in both, in neither, with an exact two-sided sign test on the flips (b = rows that changed).  A one-run difference below
the flip noise is not evidence; the sign test says how surprising the split of flips is if both runs were equally good."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name):
    return json.load(open(os.path.join(ROOT, "results", "meaning", name, "predictions.json"), encoding="utf-8"))["predictions"]


def sign_p(a, b):
    n = a + b
    if not n: return 1.0
    k = min(a, b); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    args = [x for x in sys.argv[1:] if x != "--list"]; show = "--list" in sys.argv
    A, B, sets = args[0], args[1], args[2:]; pa, pb = load(A), load(B)
    print(f"{A} vs {B}")
    for s in sets:
        ra = {p["id"]: p for p in pa if p["set"] == s}; rb = {p["id"]: p for p in pb if p["set"] == s}
        ids = [i for i in ra if i in rb]
        if not ids: print(f"  {s}: no common rows"); continue
        a_only = [i for i in ids if ra[i]["ok"] and not rb[i]["ok"]]; b_only = [i for i in ids if rb[i]["ok"] and not ra[i]["ok"]]
        both = sum(1 for i in ids if ra[i]["ok"] and rb[i]["ok"]); neither = sum(1 for i in ids if not ra[i]["ok"] and not rb[i]["ok"])
        print(f"  {s}: n={len(ids)}  {A} {sum(ra[i]['ok'] for i in ids)}  {B} {sum(rb[i]['ok'] for i in ids)}  | both {both}, neither {neither}, only {A} {len(a_only)}, only {B} {len(b_only)}  sign-test p={sign_p(len(a_only), len(b_only)):.3f}")
        if show:
            for tag, lst in ((f"only {A}", a_only), (f"only {B}", b_only)):
                for i in lst:
                    r = ra[i]; print(f"    [{tag}] {i} | {r['text']} | gold {r.get('gold')} | {A}: {ra[i]['pred']} | {B}: {rb[i]['pred']}")


if __name__ == "__main__":
    main()
