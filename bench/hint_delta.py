"""Does leaving more terms out of the hint line cost accuracy?  Only the audit questions whose hint line changes are read (with the native engine, as the product), once with the shipped skip list and once with VGPT_SKIP_EXTRA added; the strict scores are compared.
    python bench/hint_delta.py --extra "kwh,kw,wh/km,efficiency"      (run on the Pi: models/vg_frame2r12.vgw + models/ort_frame2r12)"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
ap = argparse.ArgumentParser(); ap.add_argument("--extra", required=True); ap.add_argument("--vgw", default="models/vg_frame2r12.vgw"); ap.add_argument("--ort", default="models/ort_frame2r12"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
from vgpt.meaning import ort_backend as OB, glossary as GL
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, a.ort), threads=a.threads, native=os.path.join(ROOT, a.vgw)); G = m.glossary
def hints(extra):
    GL._SKIP = None
    if extra: os.environ["VGPT_SKIP_EXTRA"] = extra
    else: os.environ.pop("VGPT_SKIP_EXTRA", None)
rows = []
for s in E.FROZEN + E.NEW:
    for r in E.load(s):
        tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev")
        hints(None); h0 = G.hint_line(r["text"], vehicle=veh if m.vehicle_hint else None); hints(a.extra); h1 = G.hint_line(r["text"], vehicle=veh if m.vehicle_hint else None)
        if h0 != h1: rows.append((s, r, veh))
print(len(rows), "audit questions have a different hint line", flush=True)
base = new = lost = won = 0; lost_l = []
for s, r, veh in rows:
    m.set_vehicle(veh)
    hints(None); f0, i0 = m.parse(r["text"], prev=r["prev"]); hints(a.extra); f1, i1 = m.parse(r["text"], prev=r["prev"]); hints(None)
    b = E.ok_prefix(i0["raw"], r["gold"]); n = E.ok_prefix(i1["raw"], r["gold"]); base += b; new += n; lost += b and not n; won += n and not b
    if b and not n: lost_l.append((r["text"], i1["raw"], r["gold"]))
print(f"shipped {base}, with {a.extra!r} left out {new}; lost {lost}, won {won}")
for x in lost_l: print("  LOST:", x)
