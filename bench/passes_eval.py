"""Passes over the weights per question (the cost of the decode on the Pi: ~140-250 ms each), on the acceptance questions and a sample of the frozen sets.  Frames must not change.

    python bench/passes_eval.py [--n 400] [--save results/passes_before.json] [--compare results/passes_before.json]"""
import argparse, glob, json, os, statistics as S, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.meaning.ort_backend import OrtFrameModel


def questions(n):
    out = []
    for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2: out.append(p[2])
    pool = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        if "followup" in f: continue
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) > 2: pool.append(p[1])
    out += pool[::max(1, len(pool) // n)]
    return out[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--save"); ap.add_argument("--compare"); a = ap.parse_args()
    m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=4); m.parse("how fast am I going"); rows = {}
    for q in questions(a.n):
        f, i = m.parse(q); rows[q] = {"raw": i["raw"], "pmin": i["pmin"], "passes": m.stats["passes"], "rows": m.stats.get("rows")}
    ps = [r["passes"] for r in rows.values()]
    print(f"{len(rows)} questions: passes mean {S.mean(ps):.3f}  1 pass {sum(p == 1 for p in ps) / len(ps):.0%}  2 passes {sum(p == 2 for p in ps) / len(ps):.0%}  3+ {sum(p >= 3 for p in ps) / len(ps):.0%}")
    if a.save: json.dump(rows, open(os.path.join(ROOT, a.save), "w"))
    if a.compare:
        old = json.load(open(os.path.join(ROOT, a.compare))); same = sum(old[q]["raw"] == rows[q]["raw"] for q in rows if q in old); n = sum(q in old for q in rows)
        print(f"frames identical to {a.compare}: {same}/{n}; pmin max difference {max(abs(old[q]['pmin'] - rows[q]['pmin']) for q in rows if q in old):.2e}; passes before mean {S.mean(old[q]['passes'] for q in rows if q in old):.3f}")


if __name__ == "__main__":
    main()
