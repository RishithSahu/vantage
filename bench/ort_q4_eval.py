"""Does reading the decode passes with 4-bit weights change what the model writes?  The same question is run twice on the same ONNX Runtime model: decode passes with the int8 graph (as shipped) and with the 4-bit graph (VGPT_ORT_DECODE_DIR);
the prompt pass is int8 in both.  Reported on the development questions of bench/calibrate_confidence.py (never sealed sets): exact-frame accuracy of each, how many frames differ, the confidence shift, and the time of a decode pass on this machine.

    python bench/ort_q4_eval.py --q4 models/ort_q4_build [--limit 0] [--out results/ort_q4_eval.json]"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import calibrate_confidence as CC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q4", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="results/ort_q4_eval.json"); a = ap.parse_args()
    os.environ["VGPT_ORT_DECODE_DIR"] = os.path.abspath(a.q4)
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import _frame_model
    m = _frame_model(os.environ.get("VGPT_MEANING_ADAPTER"), 4); assert m.decode_bits == 4, "the 4-bit graph was not loaded"; q4 = m.sess_dec; i8 = m.sess
    qs = list(CC.questions()); seen = set(); qs = [x for x in qs if not ((x[1], x[2]) in seen or seen.add((x[1], x[2])))]
    if a.limit: qs = qs[::max(1, len(qs) // a.limit)]
    rows = []; t = {"int8": [0.0, 0], "q4": [0.0, 0]}
    for _, text, prev, gold in qs:
        r = {}
        for name, sess in (("int8", i8), ("q4", q4)):
            m.sess_dec = sess; fr, info = m.parse(text, prev=prev) if prev is not None else m.parse(text)
            r[name] = {"raw": info["raw"], "ok": bool(fr is not None and CC.right(info["raw"], gold)), "pmin": info["pmin"], "ms": info["ms"], "dec": m.stats.get("decode_ms", 0), "passes": m.stats.get("passes", 0)}
            t[name][0] += r[name]["dec"]; t[name][1] += r[name]["passes"]
        rows.append(dict(q=text, gold=gold, **{k + "_" + f: v for k, d in r.items() for f, v in d.items()}))
    n = len(rows); acc = lambda k: sum(x[k + "_ok"] for x in rows) / n
    diff = sum(1 for x in rows if x["int8_raw"] != x["q4_raw"]); fixed = sum(1 for x in rows if not x["int8_ok"] and x["q4_ok"]); broke = sum(1 for x in rows if x["int8_ok"] and not x["q4_ok"])
    out = {"n": n, "exact_int8": acc("int8"), "exact_q4": acc("q4"), "frames_differ": diff, "q4_fixed": fixed, "q4_broke": broke, "decode_ms_per_pass": {k: v[0] / max(1, v[1]) for k, v in t.items()}}
    print(json.dumps(out, indent=1)); json.dump({"summary": out, "rows": rows}, open(os.path.join(ROOT, a.out), "w"), indent=0)


if __name__ == "__main__": main()
