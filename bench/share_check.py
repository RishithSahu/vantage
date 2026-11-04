"""Does the second reading of a question (with the previous turn's frame) give the same frames when it starts from the first reading's work and the first reading stops as soon as the gate has decided?  Every audit question that has a previous frame is read with VGPT_SHARE=0 (the old way) and VGPT_SHARE=1; the frames must be identical, the time is compared.
    python bench/share_check.py [--vgw models/vg_frame2r12.vgw] [--ort models/ort_frame2r12] [--n 0] [--threads 2]"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
ap = argparse.ArgumentParser(); ap.add_argument("--vgw", default="models/vg_frame2r12.vgw"); ap.add_argument("--ort", default="models/ort_frame2r12"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
from vgpt.meaning import ort_backend as OB
import eval_r10 as E
m = OB.OrtFrameModel(os.path.join(ROOT, a.ort), threads=a.threads, native=os.path.join(ROOT, a.vgw))
rows = [(s, r) for s in E.FROZEN + E.NEW for r in E.load(s) if r["prev"]]
if a.n: rows = rows[::max(1, len(rows) // a.n)][:a.n]
print(len(rows), "questions with a previous frame", flush=True)
same = both = 0; t0 = t1 = 0.0; used = 0; saved = 0; worst = 0.0; diffs = []
for s, r in rows:
    tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh)
    os.environ["VGPT_SHARE"] = "0"; os.environ["VGPT_GATE_ABORT"] = "0"; t = time.perf_counter(); f0, i0 = m.parse(r["text"], prev=r["prev"]); d0 = time.perf_counter() - t
    os.environ["VGPT_SHARE"] = "1"; os.environ["VGPT_GATE_ABORT"] = "1"; t = time.perf_counter(); f1, i1 = m.parse(r["text"], prev=r["prev"]); d1 = time.perf_counter() - t
    same += i0["raw"] == i1["raw"]; worst = max(worst, abs((i0["pmin"] or 0) - (i1["pmin"] or 0)))
    if i0["raw"] != i1["raw"]: diffs.append((r["text"], i0["raw"], i1["raw"]))
    if i1.get("prev_used") and i1.get("ms", 0) > 0 and m.stats.get("reused_tokens"): used += 1; t0 += d0; t1 += d1
print(f"frames identical {same}/{len(rows)}; largest pmin difference {worst:.4f}")
print(f"questions read twice with work shared: {used}; their time {t0:.1f}s -> {t1:.1f}s ({(t0 - t1) / max(used, 1) * 1000:.0f} ms saved each)")
for d in diffs[:12]: print("  DIFF:", d)
