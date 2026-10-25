"""What asking back ("did you mean 1 or 2") is worth, on the frozen evaluation sets.

    python bench/clarify_eval.py [--tau 0.8] [--limit 0]

For every question of every frozen set (the follow-up sets are given their real previous frame): the frame the model writes, its confidence, and the two readings it hesitated between.  Reported with the
threshold tau:  how many questions would be asked back, how many of the wrong readings that catches, how many correct readings are asked back needlessly, and - for the questions asked back - how often the
right frame is one of the two offered (so that "1" or "2" gets the person the right answer).  Gold frames of the older sets have six words; a prediction counts as right when it starts with the gold words."""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.meaning.ort_backend import OrtFrameModel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="results/clarify_eval.json"); a = ap.parse_args()
    m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=4); rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        name = os.path.basename(f)[:-4]; n = 0
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if name.startswith("followup") and len(p) >= 5: text, prev, gold = p[2], p[1], p[3]
            elif not name.startswith("followup") and len(p) >= 3: text, prev, gold = p[1], None, p[2]
            else: continue
            fr, info = m.parse(text, prev=prev); words = info["raw"].split(); alt = info.get("alt"); other = None
            if alt and alt["index"] < len(words) and words[alt["index"]] == alt["chosen"]: w2 = list(words); w2[alt["index"]] = alt["other"]; other = " ".join(w2)
            rows.append({"set": name, "text": text, "gold": gold, "pred": info["raw"], "pmin": info["pmin"], "other": other, "slot": (alt or {}).get("slot")})
            n += 1
            if a.limit and n >= a.limit: break
    json.dump(rows, open(os.path.join(ROOT, a.out), "w"), indent=1)
    ok = lambda p, g: p.split()[:len(g.split())] == g.split()
    N = len(rows); wrong = [r for r in rows if not ok(r["pred"], r["gold"])]
    print(f"{N} questions; {len(wrong)} read wrongly ({len(wrong) / N:.1%})")
    for tau in (0.6, 0.7, 0.8, 0.9):
        asked = [r for r in rows if (r["pmin"] or 1) < tau]; caught = [r for r in asked if not ok(r["pred"], r["gold"])]
        offered = [r for r in asked if r["other"]]; resolves = [r for r in offered if ok(r["pred"], r["gold"]) or ok(r["other"], r["gold"])]
        rest = [r for r in rows if (r["pmin"] or 1) >= tau]; rest_ok = sum(ok(r["pred"], r["gold"]) for r in rest)
        print(f"tau {tau}: asked back {len(asked) / N:.1%} (two readings offered for {len(offered) / max(1, len(asked)):.0%} of them); catches {len(caught) / len(wrong):.0%} of the wrong readings; "
              f"{len(asked) - len(caught)} correct readings asked needlessly; the right frame is one of the two offered in {len(resolves) / max(1, len(offered)):.0%}; accuracy of what is answered at once {rest_ok / len(rest):.1%}")


if __name__ == "__main__":
    main()
