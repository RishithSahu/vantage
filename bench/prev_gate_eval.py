"""Accuracy and tokens of the previous-turn gate (vgpt/meaning/prevgate.py) against showing the previous frame every time, on the frozen evaluation sets.

    python bench/prev_gate_eval.py [--per-set 60]"""
import argparse, glob, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.meaning.ort_backend import OrtFrameModel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-set", type=int, default=60); ap.add_argument("--out", default="results/prev_gate_eval.json"); a = ap.parse_args()
    m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=4); rnd = random.Random(11); rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        name = os.path.basename(f)[:-4]; items = []; golds = []
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if name.startswith("followup") and len(p) >= 5: items.append((p[2], p[1], p[3], True))
            elif not name.startswith("followup") and len(p) >= 3: items.append((p[1], None, p[2], False)); golds.append(p[2])
        rnd.shuffle(items)
        for text, prev, gold, real in items[:a.per_set]:
            pv = prev or rnd.choice(golds).split()
            pv = prev or " ".join((pv + ["none"] * 8)[:8])
            row = {"set": name, "text": text, "gold": gold, "real_prev": real}
            for mode in ("always", "gate"):
                os.environ["VGPT_PREV"] = mode; fr, info = m.parse(text, prev=pv); row[mode] = info["raw"]; row[mode + "_tail"] = m.stats.get("tail_tokens"); row[mode + "_ms"] = info["ms"]; row[mode + "_used"] = info.get("prev_used", True)
            rows.append(row)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w"), indent=1)
    ok = lambda p, g: p.split()[:len(g.split())] == g.split()
    for label, sel in (("follow-up sets (real previous frame)", [r for r in rows if r["real_prev"]]), ("all other sets (unrelated previous frame)", [r for r in rows if not r["real_prev"]])):
        n = len(sel); ka = sum(ok(r["always"], r["gold"]) for r in sel); kg = sum(ok(r["gate"], r["gold"]) for r in sel); used = sum(r["gate_used"] for r in sel)
        print(f"{label}: n={n}  always-previous {ka/n:.1%}  gated {kg/n:.1%}  previous line shown for {used/n:.0%}  mean ms always {sum(r['always_ms'] for r in sel)/n:.0f} gated {sum(r['gate_ms'] for r in sel)/n:.0f}")


if __name__ == "__main__":
    main()
