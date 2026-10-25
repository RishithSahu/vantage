"""Second review round (2026-10-24): the new adapter's mistakes on the frozen sets, ruled the same way as the first round (docs/FAILURE_ANALYSIS_2026-10-24.md).
Merges results/r10_new_to_rule.json + rulings2_*.txt (+ the pairs whose printed answers are identical on all three vehicles) into bench/meaning_eval/gold_review.json: each entry keeps its `accepted` list; an entry for the same
question with a new predicted frame ADDS that frame to the accepted frames when the ruling is B / P / Q / CT."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
pad = lambda t: " ".join((t.split() + ["none"] * 8)[:8])
new = json.load(open(os.path.join(R, "results", "r10_new_to_rule.json"), encoding="utf-8")); rul = {}
for f in ("rulings2_a.txt", "rulings2_b.txt"):
    for tok in open(os.path.join(H, f)).read().split():
        m = re.match(r"(\d+)([A-Z]+)$", tok); rul[int(m.group(1))] = m.group(2)
assert sorted(rul) == list(range(len(new))), (len(rul), len(new))
rp = os.path.join(R, "bench", "meaning_eval", "gold_review.json"); rev = json.load(open(rp, encoding="utf-8"))
added = 0
def add(set_, text, gold, pred, k):
    global added
    g, p = pad(gold), pad(pred); acc = {"G": [g], "P": [p], "B": [g, p], "Q": [p, g], "CT": [g, p]}[k]
    rev.append(dict(text=text, set=set_, gold=g, pred=p, ruling=k, accepted=acc, pmin=None, note="round 2 ruling (the second adapter's mistake)")); added += 1
for i, r in enumerate(new): add(r["set"], r["text"], r["gold"], r["pred"], rul[i])
eq = json.load(open(os.path.join(R, "results", "answer_equiv_r10.json"), encoding="utf-8"))
for r in eq:
    if r["verdict"] == "same": add(r["set"], r["text"], r["gold"], r["pred"], "B")
json.dump(rev, open(rp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
import collections
print("entries now", len(rev), "added", added, dict(collections.Counter(rul.values())))
