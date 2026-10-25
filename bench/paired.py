"""Paired comparison of two eval_v2 result files on the SAME questions (matched by id).
For each outcome (correct, wrong answer on an answerable question, false action on a refusal case, answered) it reports the counts, the paired difference B - A with a 95% bootstrap interval over questions, and an exact McNemar test on the discordant pairs.
Overlap of the two systems' separate intervals is not used. Questions are one sample from one author, so the interval describes sampling noise on these questions only.
Both files must contain exactly the same question ids with the same text and gold (otherwise the comparison aborts: no silent intersection). Resampling is by CLUSTER (questions that share a previous turn are one conversation), so dependent questions are not counted as independent.
Every p-value here is EXPLORATORY: the development questions have been inspected repeatedly while the system was changed, which selection-biases them; the tests do not undo that contamination.
    python bench/paired.py results/abl_A.json results/abl_B.json"""
import json, math, random, sys


def load(p): return {r["id"]: r for r in json.load(open(p))["rows"]}


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n; return min(1.0, 2 * p)


class Mismatch(Exception): pass


def check_same_questions(A, B):
    if set(A) != set(B): raise Mismatch(f"question ids differ: only in A {sorted(set(A) - set(B))[:5]}, only in B {sorted(set(B) - set(A))[:5]}")
    bad = [i for i in A if A[i]["text"] != B[i]["text"] or A[i]["gold"] != B[i]["gold"]]
    if bad: raise Mismatch(f"text or gold differs for {bad[:5]} (different question-file versions)")


def compare(A, B, seed=0, reps=10000):
    check_same_questions(A, B)
    ids = sorted(A); rng = random.Random(seed); out = {}
    cluster = {i: (A[i].get("prev_text") or i) for i in ids}
    ans = [i for i in ids if A[i]["gold"] != "op=none"]; ref = [i for i in ids if A[i]["gold"] == "op=none"]
    defs = {"correct (all questions)": (ids, lambda r: bool(r["correct"])), "wrong answer (answerable)": (ans, lambda r: bool(r["wrong_answer"])), "false action (refusal cases)": (ref, lambda r: bool(r["false_action"])), "answered (answerable)": (ans, lambda r: bool(r["answered"]))}
    for name, (sub, f) in defs.items():
        if not sub: continue
        a = [f(A[i]) for i in sub]; b = [f(B[i]) for i in sub]; d = [int(y) - int(x) for x, y in zip(a, b)]; n = len(d)
        groups = {}
        for i, x in zip(sub, d): groups.setdefault(cluster[i], []).append(x)
        gl = list(groups.values()); boots = []
        for _ in range(reps):
            tot = 0; cnt = 0
            for _g in range(len(gl)): g = gl[rng.randrange(len(gl))]; tot += sum(g); cnt += len(g)
            boots.append(tot / max(cnt, 1))
        boots.sort()
        only_a = sum(1 for x, y in zip(a, b) if x and not y); only_b = sum(1 for x, y in zip(a, b) if y and not x)
        out[name] = {"n": n, "A": sum(a), "B": sum(b), "diff_points": round(100 * sum(d) / n, 1), "ci95_points": [round(100 * boots[int(0.025 * reps)], 1), round(100 * boots[int(0.975 * reps)], 1)], "only_A": only_a, "only_B": only_b, "mcnemar_p": round(mcnemar(only_a, only_b), 4)}
    return out


if __name__ == "__main__":
    A, B = load(sys.argv[1]), load(sys.argv[2]); print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}")
    for k, v in compare(A, B).items(): print(f"  {k:30s} A {v['A']:3d}  B {v['B']:3d}  (n={v['n']})  B-A {v['diff_points']:+5.1f} pts  95% CI [{v['ci95_points'][0]:+.1f}, {v['ci95_points'][1]:+.1f}]  discordant A-only {v['only_A']} / B-only {v['only_B']}  McNemar p={v['mcnemar_p']} (exploratory)")
