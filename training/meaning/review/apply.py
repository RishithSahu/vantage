"""Apply the hand rulings (rulings_*.txt) to results/answer_equiv.json -> bench/meaning_eval/gold_review.json.

Each ruling is made by reading, for the gold frame and the predicted frame, what the resolver prints (bench/meaning_eval/answer_equiv.py) and what the question asks:
  G  the gold frame stands; the model is wrong
  P  the predicted frame is right; the gold label was wrong -> accepted = {pred}
  B  both frames are acceptable (the answers are equivalent or equally good) -> accepted = {gold, pred}
  Q  both acceptable and the predicted one is the better to teach
  CT which one is right depends on the car type (EV / petrol / diesel); both accepted here, the car-type hint decides
'same' (identical printed answers on all three vehicles) = B.  Follow-up rows are ruled by the ruling of the same question text in another set."""
import json, os, re, collections
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
rows = json.load(open(os.path.join(R, "results", "answer_equiv.json"), encoding="utf-8"))
diff = [r for r in rows if r["verdict"] == "differs"]
rul = {}
for f in ("rulings_a.txt", "rulings_b.txt", "rulings_c.txt"):
    for tok in open(os.path.join(H, f)).read().split():
        m = re.match(r"(\d+)([A-Z]+)$", tok); rul[int(m.group(1))] = m.group(2)
assert sorted(rul) == list(range(len(diff))), (len(rul), len(diff))
pad = lambda t: " ".join((t.split() + ["none"] * 8)[:8])
out = []; by_text = {}
for i, r in enumerate(diff):
    k = rul[i]; g, p = pad(r["gold"]), pad(r["pred"])
    acc = {"G": [g], "P": [p], "B": [g, p], "Q": [p, g], "CT": [g, p]}[k]
    e = dict(text=r["text"], set=r["set"], gold=g, pred=p, ruling=k, accepted=acc, pmin=r["pmin"]); out.append(e); by_text[r["text"].strip().lower()] = e
for r in rows:
    if r["verdict"] == "same": out.append(dict(text=r["text"], set=r["set"], gold=pad(r["gold"]), pred=pad(r["pred"]), ruling="B", accepted=[pad(r["gold"]), pad(r["pred"])], pmin=r["pmin"]))
    elif r["verdict"] == "skipped":
        e = by_text.get(r["text"].strip().lower()); k = e["ruling"] if e else "G"
        g, p = pad(r["gold"]), pad(r["pred"]); acc = {"G": [g], "P": [p], "B": [g, p], "Q": [p, g], "CT": [g, p]}[k]
        out.append(dict(text=r["text"], set=r["set"], gold=g, pred=p, ruling=k, accepted=acc, pmin=r["pmin"], note="follow-up row, ruled like the same question elsewhere" if e else "follow-up row, no twin: kept as gold"))
json.dump(out, open(os.path.join(R, "bench", "meaning_eval", "gold_review.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
c = collections.Counter(e["ruling"] for e in out); print(len(out), dict(c))
cg = [e for e in out if e["ruling"] == "G"]; print("true model errors (G):", len(cg), "of", len(out), "; confident:", sum((e["pmin"] or 1) >= 0.8 for e in cg))
