"""How often does the frame change when the `Previous:` line is left out?  (the cost of that line is ~11 of the ~25 tokens read per question on the Pi)

    python bench/prev_cascade_eval.py [--per-set 25]

follow-up sets (id, previous frame, text, gold, tag): the previous frame is the real one.  Every other set: a previous frame drawn from ANOTHER question of the same set (an unrelated previous turn, which is what
the Pi benchmark and most real sessions look like).  For each item: frame with the line, frame without, and the gold."""
import argparse, glob, json, os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt import product


def load(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t"); rows.append(p)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-set", type=int, default=25); ap.add_argument("--out", default="results/prev_cascade.json"); a = ap.parse_args()
    m = product.frame_model() if hasattr(product, "frame_model") else None
    if m is None:
        from vgpt.meaning.ort_backend import OrtFrameModel
        m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=4)
    rnd = random.Random(5); out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        name = os.path.basename(f)[:-4]; rows = load(f)
        if not rows: continue
        follow = name.startswith("followup")
        items = []
        for r in rows:
            if follow and len(r) >= 5: items.append((r[2], r[1], r[3], r[4]))
            elif not follow and len(r) >= 3: items.append((r[1], None, r[2], ""))
        golds = [it[2] for it in items]
        rnd.shuffle(items); items = items[:a.per_set]
        for text, prev, gold, tag in items:
            pv = prev if prev else rnd.choice(golds)
            f1, i1 = m.parse(text, prev=pv); f0, i0 = m.parse(text)
            out.append({"set": name, "text": text, "real_prev": bool(prev), "gold": gold, "with": i1["raw"], "without": i0["raw"], "p_with": i1["pmin"], "p_without": i0["pmin"], "tail_with": m.stats.get("tail_tokens")})
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    def acc(rows, k): return sum(r[k] == r["gold"] for r in rows) / max(1, len(rows))
    for label, rows in (("follow-up (real previous frame)", [r for r in out if r["real_prev"]]), ("other sets (unrelated previous frame)", [r for r in out if not r["real_prev"]])):
        same = sum(r["with"] == r["without"] for r in rows)
        print(f"{label}: n={len(rows)} frame identical with/without {same/len(rows):.1%}; correct with {acc(rows,'with'):.1%}, without {acc(rows,'without'):.1%}")


if __name__ == "__main__":
    main()
