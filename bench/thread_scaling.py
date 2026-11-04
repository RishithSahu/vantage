"""How does the native engine scale with cores?  Prompt tokens/s, one-row and 8-row decode passes, and batched throughput for 1..N threads.
    python bench/thread_scaling.py [--threads 1 2 4 8 12]"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.environ["VGPT_FRAME"] = "2"
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, nargs="*", default=[1, 2, 4]); ap.add_argument("--one", type=int, default=0); a = ap.parse_args()
if not a.one:                                                                                   # the engine's thread pool is process-wide: each thread count needs its own process
    import subprocess
    for th in a.threads: subprocess.run([sys.executable, __file__, "--one", str(th)])
    sys.exit(0)
a.threads = [a.one]
from vgpt.meaning import ort_backend as OB
def t(f, n=6):
    f(); s = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - s) / n * 1000
base = None
for th in a.threads:
    m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=th, native=os.path.join(ROOT, "models/vg_frame2r12.vgw")); m._prefix(); bids, bkv = m.base; P = len(bids)
    ids = [1000 + i for i in range(40)]; lg, kv = m._fwd(ids, P, bkv, 1); eng = m.eng
    pre = t(lambda: m._fwd(ids, P, bkv, 1)); d1 = t(lambda: m._fwd([5], P + 40, kv, 1, decode=True)); d8 = t(lambda: m._fwd([5] * 8, P + 40, kv, 8, decode=True))
    cand = m.cand
    for nseq in (4, 8):
        seqs = [(ids, P, bkv, 1) for _ in range(nseq)]
        tb = t(lambda: eng.forward(seqs, cand), 3)
        if nseq == 8: tput = nseq / (tb / 1000)
    print(f"threads {th:2d}: prompt 40 tok {pre:6.0f} ms ({40000/pre:5.0f} tok/s) | decode 1 row {d1:5.0f} ms  | decode 8 rows {d8:5.0f} ms | 8 prompts batched {tb:6.0f} ms = {tput:4.1f} prompts/s", flush=True)
