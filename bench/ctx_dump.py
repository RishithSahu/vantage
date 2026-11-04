"""Data for a context method that costs the model nothing: for every audit question with a previous frame, the BARE reading (no Previous line), the reading WITH it, the previous frame and the gold frame.
    python bench/ctx_dump.py > results/ctx_dump.json   (run on the Pi: models/vg_frame2r12.vgw + models/ort_frame2r12)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"; os.environ["VGPT_GATE_ABORT"] = "0"
from vgpt.meaning import ort_backend as OB, prevgate as PG
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=2, native=os.path.join(ROOT, "models/vg_frame2r12.vgw"))
out = []
for s in E.FROZEN + E.NEW:
    for r in E.load(s):
        if not r["prev"]: continue
        tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh)
        f0, i0 = m._parse(r["text"], None, None); f1, i1 = m._parse(r["text"], None, r["prev"])
        out.append({"set": s, "text": r["text"], "prev": r["prev"], "gold": r["gold"], "bare": i0["raw"], "bare_p": i0["pmin"], "withprev": i1["raw"], "wording": PG.wants_prev(r["text"])})
json.dump(out, open(os.path.join(ROOT, "results", "ctx_dump.json"), "w"), ensure_ascii=False); print("done", len(out))
