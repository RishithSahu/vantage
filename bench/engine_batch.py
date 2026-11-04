"""Many cars in one call.  (1) a batch of sequences of different lengths gives exactly the logits and caches each would give alone; (2) throughput against the number of cars.
    python bench/engine_batch.py [--vgw models/vg_frame2r12.vgw] [--lib engine/libvg_new.dll] [--threads 4]"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from vgpt.meaning import native as N
ap = argparse.ArgumentParser(); ap.add_argument("--vgw", default="models/vg_frame2r12.vgw"); ap.add_argument("--lib"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
eng = N.Engine(os.path.join(ROOT, a.vgw), threads=a.threads, lib=(os.path.join(ROOT, a.lib) if a.lib else None)); print("kernel", eng.kernel, "threads", a.threads)
rng = np.random.default_rng(5); cand = np.sort(rng.choice(150000, 3000, replace=False)).astype(np.int64); empty = [np.zeros((1, 2, 0, 64), np.float32) for _ in range(48)]
# (1) exactness of batching
lens = [5, 17, 1, 40, 9]; seqs = [(rng.integers(1000, 100000, n), 0, empty, 1) for n in lens]
alone = [eng.forward1(*s[:3], s[3], cand) for s in seqs]; batch = eng.forward(seqs, cand)
worst_l = max(np.abs(x[0] - y[0]).max() for x, y in zip(alone, batch)); worst_k = max(np.abs(p - q).max() for x, y in zip(alone, batch) for p, q in zip(x[1], y[1]))
print(f"prompt batch of {len(lens)} cars (lengths {lens}): largest logit difference against running each alone {worst_l:.2e}, cache {worst_k:.2e}")
seqs2 = [(rng.integers(1000, 100000, 3), n, alone[i][1], 3) for i, n in enumerate(lens)]
alone2 = [eng.forward1(*s[:3], s[3], cand) for s in seqs2]; batch2 = eng.forward(seqs2, cand)
print("decode batch (3 rows each, different memory lengths): largest logit difference %.2e" % max(np.abs(x[0] - y[0]).max() for x, y in zip(alone2, batch2)))
# (2) throughput
base = [eng.forward1(rng.integers(1000, 100000, 40), 0, empty, 1, cand)[1] for _ in range(32)]
print("\ncars | decode pass (1 row each, memory 40) ms | per car ms | passes/s total | speed-up over one car at a time")
t1 = None
for n in (1, 2, 4, 8, 16, 32):
    seqs = [(rng.integers(1000, 100000, 1), 40, base[i], 1) for i in range(n)]
    for _ in range(2): eng.forward(seqs, cand)
    t = time.perf_counter(); reps = 6
    for _ in range(reps): eng.forward(seqs, cand)
    dt = (time.perf_counter() - t) / reps
    if t1 is None: t1 = dt
    print(f"{n:4d} | {dt*1e3:8.1f} | {dt*1e3/n:8.2f} | {n/dt:8.1f} | {n*t1/dt:5.1f}x")
