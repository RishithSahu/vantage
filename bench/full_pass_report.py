"""Summarise results/full_pass_*.json: limits (short <37 words 2 s, long >40 words 2.5 s, rest-after-model 0.2 s), worst questions, by set."""
import collections, json, statistics as st, sys
R = json.load(open(sys.argv[1])); n = len(R); lim = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
limit = lambda r: 2.0 if r["words"] < 37 else 2.5
print(n, "questions; median %.2f p95 %.2f p99 %.2f max %.1f" % (st.median(r["s"] for r in R), sorted(r["s"] for r in R)[int(.95 * n)], sorted(r["s"] for r in R)[int(.99 * n)], max(r["s"] for r in R)))
print("over their time limit:", sum(r["s"] > limit(r) for r in R), "| short over 2 s:", sum(r["s"] > 2 and r["words"] < 37 for r in R), "| long over 2.5 s:", sum(r["s"] > 2.5 and r["words"] >= 37 for r in R))
print("after-the-model time: median %.3f p95 %.3f; over %.2f s: %d; over 1 s: %d" % (st.median(r["rest"] for r in R), sorted(r["rest"] for r in R)[int(.95 * n)], lim, sum(r["rest"] > lim for r in R), sum(r["rest"] > 1 for r in R)))
by = collections.defaultdict(list)
for r in R: by[r["src"]].append(r)
print("sets with questions over the 0.2 s database limit:", sorted(((sum(x["rest"] > lim for x in v), k) for k, v in by.items() if any(x["rest"] > lim for x in v)), reverse=True)[:12])
for r in sorted(R, key=lambda r: -r["rest"])[:int(sys.argv[3]) if len(sys.argv) > 3 else 25]: print("rest %.2f model %.2f total %.2f w%d [%s] %s" % (r["rest"], r["model"], r["s"], r["words"], r["src"], r["q"][:85]))
