"""Time the native engine's forward passes on this machine (the Pi): prompt of 40 tokens, then decode passes of 1, 3 and 8 rows with a 60-token memory; and where the time goes.
    cd ~/vgengine && taskset -c 0,1 python3 engine_bench.py models/vg_frame2r12.vgw"""
import os, sys, time, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("native", os.path.join(os.path.dirname(os.path.abspath(__file__)), "native.py")); N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)
N.ENG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
eng = N.Engine(sys.argv[1], threads=int(os.environ.get("THREADS", "2"))); print("kernel", eng.kernel)
rng = np.random.default_rng(1); cand = np.sort(rng.choice(150000, 3000, replace=False)).astype(np.int64)
empty = [np.zeros((1, 2, 0, 64), np.float32) for _ in range(48)]
ids = rng.integers(1000, 100000, 40)
for _ in range(2): eng.forward1(ids, 0, empty, 1, cand)
eng.profile(True); t = time.perf_counter(); n = 6
for _ in range(n): lg, kv = eng.forward1(ids, 0, empty, 1, cand)
dt = (time.perf_counter() - t) / n; pr = eng.profile(True)
print(f"prompt pass, 40 tokens: {dt*1e3:.0f} ms ({dt*1e3/40:.1f} ms/token)")
print("   sections ms/call: embed+cache copy %.1f | norm+quantise %.1f | qkv %.1f | rope+attention %.1f | o-proj %.1f | gate+up %.1f | silu+quantise %.1f | down %.1f | head %.1f" % tuple(x / n for x in pr[:9]))
_, kv = eng.forward1(ids, 0, empty, 1, cand); _, kv = eng.forward1(ids[:20], 40, kv, 1, cand)
for rows in (1, 2, 3, 4, 6, 8):
    r = rng.integers(1000, 100000, rows)
    for _ in range(2): eng.forward1(r, 60, kv, rows, cand)
    eng.profile(True); t = time.perf_counter(); n = 12
    for _ in range(n): eng.forward1(r, 60, kv, rows, cand)
    dt = (time.perf_counter() - t) / n; pr = eng.profile(True)
    print(f"decode pass, {rows} rows, memory 60: {dt*1e3:6.1f} ms" + ("   sections: embed+copy %.1f | norm %.1f | qkv %.1f | attn %.1f | o %.1f | gu %.1f | silu %.1f | down %.1f | head %.1f" % tuple(x / n for x in pr[:9]) if rows in (1, 8) else ""))
