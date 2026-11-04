"""Does the native engine compute what the ONNX graph computes?  (1) logits of the same tokens through both, (2) the frames of real questions through both.
    python bench/native_parity.py [--n 200] [--vgw models/vg_frame2r12.vgw] [--ort models/ort_frame2r12]"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--vgw", default="models/vg_frame2r12.vgw"); ap.add_argument("--ort", default="models/ort_frame2r12"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
from vgpt.meaning import ort_backend as OB
ref = OB.OrtFrameModel(os.path.join(ROOT, a.ort), threads=a.threads); nat = OB.OrtFrameModel(os.path.join(ROOT, a.ort), threads=a.threads, native=os.path.join(ROOT, a.vgw))
print("engine kernel:", nat.eng.kernel)
rng = np.random.default_rng(3); worst = 0; top = 0; tot = 0
for T in (1, 3, 8, 40):
    ids = rng.integers(1000, 100000, T).tolist(); lr, kr = ref._fwd(ids, 0, ref._empty(), min(T, 4)); ln, kn = nat._fwd(ids, 0, nat._empty(), min(T, 4))
    d = np.abs(lr - ln).max(); worst = max(worst, d); top += int((lr.argmax(1) == ln.argmax(1)).sum()); tot += len(lr)
    kd = max(np.abs(x - y).max() for x, y in zip(kr, kn)); print(f"T={T:3d}: logits max |diff| {d:.4f} (logit range {np.abs(lr).max():.1f}), top-1 same {int((lr.argmax(1) == ln.argmax(1)).sum())}/{len(lr)}, kv max |diff| {kd:.5f}")
    ids2 = rng.integers(1000, 100000, 3).tolist(); lr2, _ = ref._fwd(ids2, T, kr, 3, True); ln2, _ = nat._fwd(ids2, T, kn, 3, True); print(f"      then 3-row decode pass: logits max |diff| {np.abs(lr2 - ln2).max():.4f}")
import eval_r10 as E
rows = []
for s in E.FROZEN + E.NEW:
    for r in E.load(s): rows.append((s, r))
step = max(1, len(rows) // a.n); rows = rows[::step][:a.n]; same = 0; pm = 0.0; tr = tn = 0.0
for s, r in rows:
    tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev")
    ref.set_vehicle(veh); nat.set_vehicle(veh)
    t = time.perf_counter(); f0, i0 = ref.parse(r["text"], prev=r["prev"]); tr += time.perf_counter() - t
    t = time.perf_counter(); f1, i1 = nat.parse(r["text"], prev=r["prev"]); tn += time.perf_counter() - t
    same += i0["raw"] == i1["raw"]; pm = max(pm, abs((i0["pmin"] or 0) - (i1["pmin"] or 0)))
    if i0["raw"] != i1["raw"]: print("  DIFF:", r["text"], "|", i0["raw"], "|", i1["raw"])
print(f"frames identical {same}/{len(rows)}; largest pmin difference {pm:.4f}; wall time ONNX {tr:.1f}s, native {tn:.1f}s")
