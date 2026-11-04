"""Does dropping the trailing `none` words of the Previous line change a frame?  Every audit question with a previous frame is read with the full line and with the line cut after its last word that is not `none`; frames and strict scores are compared.
    python bench/prev_trim.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
from vgpt.meaning import ort_backend as OB
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=2, native=os.path.join(ROOT, "models/vg_frame2r12.vgw"))
def trim(p):
    w = p.split()
    while len(w) > 2 and w[-1] == "none": w.pop()
    return " ".join(w)
rows = [(s, r) for s in E.FROZEN + E.NEW for r in E.load(s) if r["prev"]]
same = ok_full = ok_trim = 0; lost = []; saved = 0
for s, r in rows:
    tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh)
    f0, i0 = m._parse(r["text"], None, r["prev"]); f1, i1 = m._parse(r["text"], None, trim(r["prev"]))
    same += i0["raw"] == i1["raw"]; a = E.ok_prefix(i0["raw"], r["gold"]); b = E.ok_prefix(i1["raw"], r["gold"]); ok_full += a; ok_trim += b
    if a and not b: lost.append((r["text"], r["prev"], i1["raw"], r["gold"]))
print(f"{len(rows)} questions: frames identical {same}; strict full {ok_full}, trimmed {ok_trim}; lost {len(lost)}")
for x in lost[:10]: print("  LOST", x)
