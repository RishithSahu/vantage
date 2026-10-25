"""Side-by-side table of every evaluated arm (results/meaning/<arm>/predictions.json, plus <arm>_sealed for sealed_v2 and <arm>_sealed3 for sealed_v3 when they exist).

    python bench/meaning_eval/summarize_arms.py full words_only natural_only template_only no_aux no_hints full_v2

Numbers are exact six-slot frame accuracy in percent ("a+c" = act and concept right); false_exec / false_abst are summed over novel, messy and user."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def summary(name):
    p = os.path.join(ROOT, "results", "meaning", name, "predictions.json")
    return json.load(open(p, encoding="utf-8"))["summary"] if os.path.exists(p) else None


def main():
    arms = sys.argv[1:]; cols = [("novel_v1", "exact"), ("novel_v1", "act_concept"), ("messy_v1", "exact"), ("messy_v1", "act_concept"), ("user_v1", "exact"), ("user_v1", "act_concept"), ("cs5", "act_and_concept")]
    head = ["arm"] + [f"{s.split('_')[0]}:{m.replace('act_concept', 'a+c').replace('act_and_concept', 'a+c')}" for s, m in cols] + ["sealed_v2:exact", "sealed_v3:exact", "false_exec", "false_abst", "auroc"]
    print(" | ".join(head)); print(" | ".join(["---"] * len(head)))
    for a in arms:
        S = summary(a)
        if S is None: print(a, "| (not evaluated)"); continue
        row = [a]
        for s, m in cols: row.append(f"{S[s][m]*100:.1f}" if s in S and m in S[s] else "-")
        for suffix, key in (("_sealed", "sealed_v2"), ("_sealed3", "sealed_v3")):
            T = summary(a + suffix); row.append(f"{T[key]['exact']*100:.1f}" if T and key in T else "-")
        fe = [S[s]["false_exec"] for s in ("novel_v1", "messy_v1", "user_v1") if s in S]; fa = [S[s]["false_abstain"] for s in ("novel_v1", "messy_v1", "user_v1") if s in S]
        n = lambda xs, i: sum(int(x.split("/")[i]) for x in xs)
        row += [f"{n(fe, 0)}/{n(fe, 1)}", f"{n(fa, 0)}/{n(fa, 1)}", str(S.get("novel_v1", {}).get("auroc_pmin"))]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
