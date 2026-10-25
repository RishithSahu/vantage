"""Does the model read a misspelled question the way it reads the clean one?  With the pre-parser off and on.

    python bench/typo_stress.py [--n 300] [--held]        --held uses the word list built without the held-out sets (results/_canon_vocab_holdout.json)

Questions: the clean ones of the held-out evaluation sets.  Each gets ONE random typo in one word of 5+ letters (swap, drop, double, substitute) and, in a second variant, a shorthand/filler change ('pls', 'avg').  A reading counts as kept when the frame
the model writes for the changed question equals the frame for the clean question."""
import argparse
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--held", action="store_true"); a = ap.parse_args()
    if a.held: os.environ["VGPT_CANON_VOCAB"] = os.path.join(ROOT, "results", "_canon_vocab_holdout.json")
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.meaning import canon
    from vgpt.product import _frame_model
    import json
    prod = json.load(open(os.path.join(ROOT, "models", "product.json"))); m = _frame_model(os.path.join(ROOT, "models", prod["adapter"]), 4)
    qs = []
    for b in ("sealed_v7", "sealed_v6", "user_v1", "hard_v5", "holdout_v10", "wording_v1", "novel_v1"):
        for line in open(os.path.join(ROOT, "bench", "meaning_eval", b + ".tsv"), encoding="utf-8"):
            if line.startswith("#") or "\t" not in line: continue
            c = line.rstrip("\n").split("\t")
            if len(c) > 3 and re.search(r"typo|noise|voice|abbrev|mixing|fragment", c[-1]): continue
            if re.search(r"[a-z]{5,}", c[1]) and len(c[1]) < 120: qs.append(c[1])
    rnd = random.Random(11); rnd.shuffle(qs); qs = qs[:a.n]

    def frame(t):
        r = m.parse(t); return str(getattr(r, "raw", None) or (r[0] if isinstance(r, tuple) else r))
    kept_off = kept_on = same_text = n = 0; broke = []
    for q in qs:
        ws = [(x.start(), x.group()) for x in re.finditer(r"[a-z]{5,}", q)]; pos, w = rnd.choice(ws); i = rnd.randrange(1, len(w) - 1); kind = rnd.choice(["swap", "drop", "dup", "sub"])
        t = {"swap": w[:i] + w[i + 1] + w[i] + w[i + 2:], "drop": w[:i] + w[i + 1:], "dup": w[:i] + w[i] + w[i:], "sub": w[:i] + "x" + w[i + 1:]}[kind]
        if t == w: continue
        q2 = q[:pos] + t + q[pos + len(w):]; fq = frame(q); n += 1; off = frame(q2) == fq; fixed = canon.normalise(q2)[0]; on = frame(fixed) == fq; kept_off += off; kept_on += on
        if off and not on: broke.append((q2, fixed))
    print(f"{n} questions with one typo each: reading kept without the pre-parser {kept_off} ({100 * kept_off / n:.1f} %), with it {kept_on} ({100 * kept_on / n:.1f} %); broken by the pre-parser: {len(broke)}")
    for b in broke[:8]: print("   broke:", b)


if __name__ == "__main__": main()
