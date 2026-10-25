"""Does a wrong frame change what the person is shown?

    python bench/meaning_eval/answer_equiv.py [--in results/clarify_eval.json] [--out results/answer_equiv.json]

For every row whose predicted frame differs from the gold frame, both frames are run through the real resolver on three generated vehicles (an EV, a petrol car, a diesel truck) and the printed answers are
compared (digits and words, case-folded).  `same` = identical on every vehicle: the frame error does not reach the person.  `differs` = at least one vehicle prints something else.  `error` = one of the two raised.
The follow-up sets need the previous turn and are skipped (marked `skipped`).  No model is loaded: the resolver is given the frames directly."""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
from vgpt.meaning import glossary as GL                      # noqa: E402
from vgpt.meaning.frame import Frame                         # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain               # noqa: E402

DBS = ["data/world_aster.db", "data/world_kestrel.db", "data/world_rhino.db"]
NOW = 1792364340


class Null:
    glossary = GL.load()
    def parse(self, text, hint=None, prev=None): raise RuntimeError("not used")


def pad(t):
    w = t.split()
    return " ".join(w + ["none"] * (8 - len(w)))


def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. %]+", " ", str(s).lower())).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="results/clarify_eval.json"); ap.add_argument("--out", default="results/answer_equiv.json"); a = ap.parse_args()
    rows = json.load(open(os.path.join(ROOT, a.inp), encoding="utf-8")); brains = []
    for d in DBS:
        b = MeaningBrain(os.path.join(ROOT, d), Null(), now=NOW, tau=0.0); brains.append(b)
    ok = lambda p, g: p.split()[:len(g.split())] == g.split()
    out = []; cache = {}
    def ans(b, i, text, frame):
        k = (i, text, frame)
        if k not in cache:
            try: cache[k] = norm(b.resolver.answer(text, Frame.from_text(frame))["answer"])
            except Exception as e: cache[k] = "ERR " + type(e).__name__
        return cache[k]
    for r in rows:
        if ok(r["pred"], r["gold"]): continue
        if r["set"].startswith("followup"): out.append(dict(r, verdict="skipped")); continue
        g, p = pad(r["gold"]), pad(r["pred"]); res = []
        for i, b in enumerate(brains): res.append((ans(b, i, r["text"], g), ans(b, i, r["text"], p)))
        if any(x.startswith("ERR") or y.startswith("ERR") for x, y in res): v = "error"
        elif all(x == y for x, y in res): v = "same"
        else: v = "differs"
        out.append(dict(r, verdict=v, gold_answer=res[0][0][:160], pred_answer=res[0][1][:160]))
    json.dump(out, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    import collections
    print(collections.Counter(x["verdict"] for x in out))


if __name__ == "__main__":
    main()
