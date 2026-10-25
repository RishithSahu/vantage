"""The adoption gate for a new adapter (stated before the result is known):

    python bench/meaning_eval/gate.py NEW OLD     (names of results/meaning/<name>_r10/predictions.json; OLD = the shipped adapter, frame2w)

PASS needs all of
  1. no frozen set with at least 60 questions loses more than 1.5 points (reviewed score, the same rulings for both);
  2. the pooled frozen score (reviewed) does not fall;
  3. the pooled score of the fresh sets (sealed_v4, sealed_v5, sealed_v6, ice_v1, wording_v1, trend_v1/v2, anomaly_v1/v2; the held-out-family set is reported, not pooled) rises by at least 5 points (strict).
The tables are printed so a failure can be read, not only counted."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import eval_r10 as E                                          # noqa: E402

pad = lambda t: " ".join((str(t).split() + ["none"] * 8)[:8])
FRESH = ["sealed_v4", "sealed_v5", "sealed_v6", "ice_v1", "wording_v1", "trend_v1", "trend_v2", "anomaly_v1", "anomaly_v2"]


def load(name):
    review = {}
    for e in json.load(open(os.path.join(HERE, "gold_review.json"), encoding="utf-8")):
        for key in ((e["set"], e["text"].strip().lower()), ("*", e["text"].strip().lower())): review.setdefault(key, []).extend(x for x in (pad(y) for y in e["accepted"]) if x not in review.get(key, []))
    out = {}
    for r in json.load(open(os.path.join(ROOT, "results", "meaning", name + "_r10", "predictions.json"), encoding="utf-8"))["predictions"]:
        acc = review.get((r["set"], r["text"].strip().lower())) or review.get(("*", r["text"].strip().lower()))
        out[(r["set"], r["id"])] = (bool(r["strict"]), bool(r["strict"] or (acc is not None and pad(r["pred"]) in acc)))
    return out


def per_set(d):
    s = {}
    for (st, _), (strict, rev) in d.items():
        a = s.setdefault(st, [0, 0, 0]); a[0] += 1; a[1] += strict; a[2] += rev
    return s


def main():
    new, old = per_set(load(sys.argv[1])), per_set(load(sys.argv[2])); ok = True
    print(f"{'set':22s}{'n':>5s}   {'old rev':>8s}{'new rev':>8s}{'delta':>7s}   {'old str':>8s}{'new str':>8s}")
    for st in E.FROZEN:
        if st not in new or st not in old: continue
        n = new[st][0]; dr = 100 * (new[st][2] - old[st][2]) / n; flag = ""
        if n >= 60 and dr < -1.5: ok = False; flag = "  <-- loses more than 1.5"
        print(f"{st:22s}{n:5d}   {100 * old[st][2] / n:8.1f}{100 * new[st][2] / n:8.1f}{dr:+7.1f}   {100 * old[st][1] / n:8.1f}{100 * new[st][1] / n:8.1f}{flag}")
    fz = [s for s in E.FROZEN if s in new and s in old]; N = sum(new[s][0] for s in fz); po, pn = sum(old[s][2] for s in fz) / N * 100, sum(new[s][2] for s in fz) / N * 100
    print(f"\nFROZEN pooled ({N}): reviewed old {po:.2f}  new {pn:.2f}  delta {pn - po:+.2f}   strict old {sum(old[s][1] for s in fz) / N * 100:.2f}  new {sum(new[s][1] for s in fz) / N * 100:.2f}")
    if pn < po: ok = False
    print(f"\n{'fresh set':22s}{'n':>5s}   {'old str':>8s}{'new str':>8s}{'delta':>7s}")
    for st in FRESH + ["holdout_v10"]:
        if st not in new or st not in old: continue
        n = new[st][0]; print(f"{st:22s}{n:5d}   {100 * old[st][1] / n:8.1f}{100 * new[st][1] / n:8.1f}{100 * (new[st][1] - old[st][1]) / n:+7.1f}")
    fs = [s for s in FRESH if s in new and s in old]; M = sum(new[s][0] for s in fs); qo, qn = sum(old[s][1] for s in fs) / M * 100, sum(new[s][1] for s in fs) / M * 100
    print(f"\nFRESH pooled ({M}): strict old {qo:.2f}  new {qn:.2f}  delta {qn - qo:+.2f}")
    if qn - qo < 5.0: ok = False
    print("\nGATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
