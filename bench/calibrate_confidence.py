"""Calibrate the frame model's confidence on the frozen DEVELOPMENT sets (never the sealed, hold-out or acceptance sets).

    python bench/calibrate_confidence.py run  [--out results/calibration_rows.json]     run the product's model over every development question and keep (pmin, right / wrong)
    python bench/calibrate_confidence.py fit  [--rows results/calibration_rows.json]    fit the monotone map pmin -> P(frame right) on half of the questions, report it on the other half,
                                                                                         write vgpt/meaning/calibration.json

pmin is the lowest probability of any token the model wrote, read over the tokens the decoder may allow (the same measure in the llama.cpp and the ONNX Runtime backends).  It is a score, not a probability: the fit turns it into the
share of questions that were really read right at that score, which is what a driver (and the safe-refusal threshold) can use.  The split is by a hash of the question text, so the same question is always on the same side."""
import argparse
import glob
import json
import os
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
SKIP = ("sealed", "holdout", "acceptance", "expected")
OUT = os.path.join(ROOT, "vgpt", "meaning", "calibration.json")


def right(pred, gold): return pred.split()[:len(gold.split())] == gold.split()


def sets():
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        name = os.path.basename(f)[:-4]
        if any(k in name for k in SKIP): continue
        yield name, f


def questions():
    for name, f in sets():
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if name.startswith("followup") and len(p) >= 5: text, prev, gold = p[2], p[1], p[3]
            elif not name.startswith("followup") and len(p) >= 3: text, prev, gold = p[1], None, p[2]
            else: continue
            if not (6 <= len(gold.split()) <= 8): continue
            yield name, text, prev, gold


def run(a):
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import _frame_model
    ad = os.environ.get("VGPT_MEANING_ADAPTER"); m = _frame_model(ad, 4); rows = []; t0 = time.time(); seen = set()
    for name, text, prev, gold in questions():
        if (text, prev) in seen: continue
        seen.add((text, prev)); fr, info = m.parse(text, prev=prev) if prev is not None else m.parse(text)
        rows.append({"set": name, "text": text, "gold": gold, "pred": info["raw"], "pmin": info["pmin"], "right": bool(fr is not None and right(info["raw"], gold))})
        if len(rows) % 200 == 0: print(len(rows), f"{time.time() - t0:.0f}s", flush=True)
    from vgpt import bootstrap
    json.dump({"rows": rows, "model": bootstrap.describe_model(m), "adapter": os.path.basename(ad or ""), "adapter_sha256": bootstrap.file_sha256(ad) if ad else None}, open(os.path.join(ROOT, a.out), "w"), indent=0)
    print("rows", len(rows), "right", sum(r["right"] for r in rows) / len(rows))


def side(text): return zlib.crc32(text.encode("utf-8")) % 2                                 # 0: the fit, 1: the check


def pav(points):
    """Pool-adjacent-violators: -> [(x, y)] non-decreasing y for points sorted by x (y the share right, weights the counts)."""
    blocks = []
    for x, y, w in points:
        blocks.append([x, x, y * w, w])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            b = blocks.pop(); blocks[-1][1] = b[1]; blocks[-1][2] += b[2]; blocks[-1][3] += b[3]
    return [((b[0] + b[1]) / 2.0, b[2] / b[3]) for b in blocks]


def fit(a):
    d = json.load(open(os.path.join(ROOT, a.rows))); rows = [r for r in d["rows"] if r["pmin"] is not None]
    train = [r for r in rows if side(r["text"]) == 0]; test = [r for r in rows if side(r["text"]) == 1]
    # distinct scores with counts, rounded so that ties pool
    agg = {}
    for r in train: k = round(r["pmin"], 3); s = agg.setdefault(k, [0, 0]); s[0] += 1; s[1] += int(r["right"])
    pts = pav([(k, v[1] / v[0], v[0]) for k, v in sorted(agg.items())]); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]

    def cal(x):
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / ((xs[i] - xs[i - 1]) or 1.0)
    brier = lambda rs, f: sum((f(r["pmin"]) - int(r["right"])) ** 2 for r in rs) / len(rs)
    def ece(rs, f, bins=10):
        e = 0.0
        for b in range(bins):
            sel = [r for r in rs if min(int(f(r["pmin"]) * bins), bins - 1) == b]
            if sel: e += len(sel) / len(rs) * abs(sum(f(r["pmin"]) for r in sel) / len(sel) - sum(int(r["right"]) for r in sel) / len(sel))
        return e
    raw = lambda x: x
    rep = {"questions": len(rows), "fit_on": len(train), "checked_on": len(test), "accuracy_all": sum(int(r["right"]) for r in rows) / len(rows),
           "raw": {"brier": brier(test, raw), "ece": ece(test, raw)}, "calibrated": {"brier": brier(test, cal), "ece": ece(test, cal)}}
    # the refusal threshold: the calibrated confidence below which the answer is withheld; chosen on the fit half, reported on the check half
    table = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for nm, rs in (("fit", train), ("check", test)):
            held = [r for r in rs if cal(r["pmin"]) < thr]; wrong = [r for r in rs if not r["right"]]
            table.append({"threshold": thr, "half": nm, "withheld": len(held) / len(rs), "wrong_caught": (sum(1 for r in held if not r["right"]) / max(1, len(wrong))), "right_withheld": sum(1 for r in held if r["right"]) / max(1, sum(1 for r in rs if r["right"])),
                          "accuracy_of_answered": (sum(int(r["right"]) for r in rs if cal(r["pmin"]) >= thr) / max(1, sum(1 for r in rs if cal(r["pmin"]) >= thr)))})
    rep["thresholds"] = table
    out = {"what": "monotone map from the decoder's lowest token probability (pmin, over the tokens the decoder may allow) to the share of development questions read right at that score", "x": xs, "y": ys, "report": rep,
           "sets": sorted({r["set"] for r in rows}), "split": "crc32(question text) % 2: 0 fits, 1 checks", "excluded": list(SKIP), "model": d.get("model"), "adapter": d.get("adapter"), "adapter_sha256": d.get("adapter_sha256"), "made": time.strftime("%Y-%m-%d")}
    json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(rep, indent=1)); print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run"); r.add_argument("--out", default="results/calibration_rows.json")
    f = sub.add_parser("fit"); f.add_argument("--rows", default="results/calibration_rows.json")
    a = ap.parse_args()
    if a.cmd == "run": run(a)
    elif a.cmd == "fit": fit(a)
    else: ap.print_help()


if __name__ == "__main__": main()
