"""Does completing an ellipsis from the previous frame (vgpt/meaning/context.py) keep the frames right and save time?  Every audit question with a previous frame is read with VGPT_CONTEXT=0 (today) and =1; strict accuracy and time are compared.
    python bench/context_check.py"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
from vgpt.meaning import ort_backend as OB
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=2, native=os.path.join(ROOT, "models/vg_frame2r12.vgw"))
rows = [(s, r) for s in E.FROZEN + E.NEW for r in E.load(s) if r["prev"]]
ok0 = ok1 = done = 0; t0 = t1 = th0 = th1 = 0.0; fb = 0; diffs = []; nfb = 0; tfb0 = tfb1 = 0.0
for s, r in rows:
    tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh)
    os.environ["VGPT_CONTEXT"] = "0"; t = time.perf_counter(); f0, i0 = m.parse(r["text"], prev=r["prev"]); d0 = time.perf_counter() - t
    os.environ["VGPT_CONTEXT"] = "1"; t = time.perf_counter(); f1, i1 = m.parse(r["text"], prev=r["prev"]); d1 = time.perf_counter() - t
    a = E.ok_prefix(i0["raw"], r["gold"]); b = E.ok_prefix(i1["raw"], r["gold"]); ok0 += a; ok1 += b
    if i1.get("context_completed"): done += 1; th0 += d0; th1 += d1
    elif i1.get("prev_used") and r["text"]:
        from vgpt.meaning import context as CX, prevgate as PG
        if PG.wants_prev(r["text"]) and CX.eligible(r["text"]): nfb += 1; tfb0 += d0; tfb1 += d1
    t0 += d0; t1 += d1
    if i0["raw"] != i1["raw"]: diffs.append((r["text"], a, b, i0["raw"], i1["raw"]))
print(f"{len(rows)} follow-ups: strict today {ok0}, with context completion {ok1}; frames changed {len(diffs)}")
print(f"completed from the previous frame: {done}; time of those {th0:.1f}s -> {th1:.1f}s ({(th0 - th1) / max(done, 1) * 1000:.0f} ms saved each)")
print(f"eligible but not completed (bare reading then the normal one): {nfb}; time {tfb0:.1f}s -> {tfb1:.1f}s ({(tfb1 - tfb0) / max(nfb, 1) * 1000:+.0f} ms each)")
print(f"all {len(rows)}: {t0:.1f}s -> {t1:.1f}s")
for d in diffs[:12]: print("  CHANGED", d[0][:40], "today right" if d[1] else "today wrong", "| now right" if d[2] else "| now wrong", "|", d[3][:34], "->", d[4][:34])
