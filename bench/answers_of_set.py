"""Answer every row of a tsv set with its GOLD frame through the real resolver (no language model), on the world of the row's car type, to read what the executors say.

    python bench/answers_of_set.py bench/meaning_eval/trend_v2.tsv [--out results/answers_trend_v2.txt] [--width 260]

Worlds: ev -> data/world_aster.db, petrol -> data/world_kestrel.db, diesel -> data/world_rhino.db (the same as bench/meaning_eval/answer_equiv.py)."""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
import ask_frame as AF                                         # noqa: E402
from vgpt.meaning.frame import Frame                           # noqa: E402
from vgpt.meaning.pipeline import MeaningBrain                 # noqa: E402
from vgpt.vehicle import lake as LK                            # noqa: E402

WORLD = {"ev": "data/world_aster.db", "petrol": "data/world_kestrel.db", "diesel": "data/world_rhino.db"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tsv"); ap.add_argument("--out"); ap.add_argument("--width", type=int, default=260); a = ap.parse_args()
    brains = {}; lines = []
    for ln in io.open(a.tsv, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t"); qid, q, gold = p[0], p[1], p[2]; veh = (p[3] if len(p) > 3 else "ev").split(":")[0]
        if veh not in brains:
            path = os.path.join(ROOT, WORLD.get(veh, WORLD["ev"])); LK.adopt_path(path); brains[veh] = MeaningBrain(path, AF.Null(), now=AF.last_ts(path), tau=0.0)
        b = brains[veh]; LK.adopt_path(os.path.join(ROOT, WORLD.get(veh, WORLD["ev"])))
        b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b._last_data = None
        t = gold.split(); f = Frame(*t[:8]) if len(t) == 8 else None
        try:
            r = b.resolver.answer(q, f); ans = str(r.get("answer", ""))
        except Exception as e: ans = f"EXCEPTION {type(e).__name__}: {e}"
        lines.append(f"[{veh:6s}] {qid} {q}\n     <{gold.replace(' none', '')}>\n     -> {ans[:a.width].replace(chr(10), ' / ')}")
    txt = "\n".join(lines)
    if a.out: io.open(a.out, "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
