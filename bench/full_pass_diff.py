"""Compare two full-pass files: answers that changed (text) and how the time after the model moved.   python bench/full_pass_diff.py before.json after.json"""
import json, sys, statistics as st
A = {r["q"]: r for r in json.load(open(sys.argv[1]))}; B = {r["q"]: r for r in json.load(open(sys.argv[2]))}
ch = [(q, A[q], B[q]) for q in A if q in B and A[q]["a"] != B[q]["a"] and "Timed out" not in A[q]["a"] and "Timed out" not in B[q]["a"]]
print("questions", len(A), "| answers changed:", len(ch), "| timed out before:", sum("Timed out" in r["a"] for r in A.values()), "after:", sum("Timed out" in r["a"] for r in B.values()))
for q, a, b in ch[:int(sys.argv[3]) if len(sys.argv) > 3 else 12]: print("--", q[:90], "\n   before:", a["a"][:160].replace("\n", " | "), "\n   after: ", b["a"][:160].replace("\n", " | "))
for nm, D in (("before", A), ("after", B)):
    r = sorted(x["rest"] for x in D.values()); n = len(r); print("%s: after-model time median %.3f p95 %.3f max %.1f | over 0.03 s: %d, over 0.1 s: %d, over 1 s: %d | total p95 %.2f max %.1f" % (nm, st.median(r), r[int(.95 * n)], r[-1], sum(x > .03 for x in r), sum(x > .1 for x in r), sum(x > 1 for x in r), sorted(x["s"] for x in D.values())[int(.95 * n)], max(x["s"] for x in D.values())))
