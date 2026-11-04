"""Throughput of K independent worker processes (one engine thread each, weights memory-mapped and shared): how many parser readings per second do K cores give?
    python bench/process_scaling.py --procs 1 2 4 8 [--seconds 20]"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, nargs="*", default=[1, 2, 4, 8]); ap.add_argument("--seconds", type=float, default=20); ap.add_argument("--worker", action="store_true"); ap.add_argument("--start", type=float, default=0); a = ap.parse_args()
if a.worker:
    sys.path.insert(0, ROOT); os.environ["VGPT_FRAME"] = "2"
    from vgpt.meaning import ort_backend as OB
    m = OB.OrtFrameModel(os.path.join(ROOT, "models/ort_frame2r12"), threads=1, native=os.path.join(ROOT, "models/vg_frame2r12.vgw")); m._prefix(); bids, bkv = m.base; P = len(bids)
    ids = [1000 + i for i in range(20)]; lg, kv = m._fwd(ids, P, bkv, 1)
    while time.time() < a.start: time.sleep(0.005)
    end = a.start + a.seconds; n = 0
    while time.time() < end:                                   # one reading = a 20-token prompt and three decode passes (the product's average)
        m._fwd(ids, P, bkv, 1); [m._fwd([5, 6, 7], P + 20, kv, 3, decode=True) for _ in range(3)]; n += 1
    print(n, flush=True); sys.exit(0)
for k in a.procs:
    start = time.time() + 25                                    # every worker loads first, then all start together
    ps = [subprocess.Popen([sys.executable, __file__, "--worker", "--start", str(start), "--seconds", str(a.seconds)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for _ in range(k)]
    tot = sum(int(p.communicate()[0].split()[-1]) for p in ps)
    print(f"{k:2d} processes: {tot / a.seconds:5.1f} readings/s ({tot / a.seconds / k:4.1f} per process)", flush=True)
