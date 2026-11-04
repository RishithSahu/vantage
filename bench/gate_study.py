"""Why does the previous-turn gate read a question twice, and does the second reading ever change a correct first reading?  Over every audit question that has a previous frame:
the bare reading (no Previous line) and the reading with it are both scored against the gold; the gate reason is recorded (wording / incomplete frame / unsure).
    python bench/gate_study.py [--n 0]"""
import argparse, collections, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--only-pointed", action="store_true"); a = ap.parse_args()
from vgpt.meaning import ort_backend as OB, prevgate as PG
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=2, native=os.path.join(ROOT, "models/vg_frame2r12.vgw"))
rows = [(s, r) for s in E.FROZEN + E.NEW for r in E.load(s) if r["prev"]]
if a.n: rows = rows[::max(1, len(rows) // a.n)][:a.n]
C = collections.Counter(); ex = collections.defaultdict(list)
for s, r in rows:
    tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh)
    pointed = PG.wants_prev(r["text"]); tagp = "WORDING " if pointed else ""
    if a.only_pointed and not pointed: continue
    os.environ["VGPT_GATE_ABORT"] = "0"; f0, i0 = m._parse(r["text"], None, None); f1, i1 = m._parse(r["text"], None, r["prev"])
    if f0 is None: C["bare reading failed"] += 1; continue
    why = "incomplete frame" if (f0.act == "ambiguous" or (f0.concept in ("none", "unlisted") and f0.act not in PG.STANDS_ALONE)) else ("unsure (pmin %.2f)" % (i0["pmin"] or 0) if (i0["pmin"] or 0) < PG.MIN_P else "complete and sure: no second reading")
    ok0 = E.ok_prefix(i0["raw"], r["gold"]); ok1 = E.ok_prefix(i1["raw"], r["gold"]) if f1 is not None else False
    key = tagp + why.split(" (")[0]; C[key] += 1; C[key + ": bare right, with Previous right"] += ok0 and ok1; C[key + ": bare right, with Previous WRONG"] += ok0 and not ok1; C[key + ": bare wrong, with Previous right"] += (not ok0) and ok1; C[key + ": same frame"] += i0["raw"] == i1["raw"]
for k in sorted(C): print("%-70s %d" % (k, C[k]))
