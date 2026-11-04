"""Where does the time go inside the product's ONNX graph?  Run on the Pi from the project folder:
    cd ~/vantage134 && VGPT_FRAME=2 taskset -c 0,1 ~/vehiclegpt2/.venv/bin/python ort_profile.py [model_dir]
Uses ONNX Runtime's own profiler on (a) prompt passes of 40 tokens and (b) one-row decode passes with a 60-token memory; prints time per operator type and the biggest single nodes, per call."""
import json, os, sys, time, collections
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.getcwd()); os.environ.setdefault("VGPT_FRAME", "2")
orig = ort.InferenceSession
def IS(path, so=None, providers=None, **kw):
    so = so or ort.SessionOptions(); so.enable_profiling = True; return orig(path, so, providers=providers, **kw)
ort.InferenceSession = IS
from vgpt.meaning import ort_backend as OB
md = sys.argv[1] if len(sys.argv) > 1 else "models/ort_frame2r12"
m = OB.OrtFrameModel(md, threads=2)

def report(title, path, calls):
    ev = json.load(open(path)); tot = collections.Counter(); node = collections.Counter(); n = collections.Counter(); run = 0
    for e in ev:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            op = e["args"].get("op_name", "?"); tot[op] += e["dur"]; node[e["name"].replace("_kernel_time", "")] += e["dur"]; n[op] += 1
        if e.get("cat") == "Session" and e["name"] == "model_run": run += e["dur"]
    s = sum(tot.values()); print(f"\n== {title}: {calls} calls, model_run {run/calls/1e3:.1f} ms/call, kernels {s/calls/1e3:.1f} ms/call ==")
    for op, d in tot.most_common(12): print(f"  {op:22s} {d/calls/1e3:7.2f} ms/call  {100*d/s:5.1f} %  ({n[op]//calls} nodes)")
    print("  biggest nodes:", ", ".join(f"{k[:40]}={v/calls/1e3:.1f}" for k, v in node.most_common(5)))

ids = np.arange(1000, 1040, dtype=np.int64)
for _ in range(2): m._fwd(list(ids), 0, m._empty())
m.sess.end_profiling()
m.sess = orig(os.path.join(md, "model.onnx"), m.sess.get_session_options() if False else None) if False else m.sess
ort.InferenceSession = IS
so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
m.sess = IS(os.path.join(md, "model.onnx"), so, providers=["CPUExecutionProvider"]); m.sess_dec = m.sess
N = 8
for _ in range(N): _, kv = m._fwd(list(ids), 0, m._empty())
p = m.sess.end_profiling(); report("prompt pass, 40 tokens", p, N)
m.sess = IS(os.path.join(md, "model.onnx"), so, providers=["CPUExecutionProvider"]); m.sess_dec = m.sess
_, kv = m._fwd(list(ids), 0, m._empty()); _, kv = m._fwd(list(ids[:20]), 40, kv)
t0 = time.perf_counter()
for i in range(N): m._fwd([1000 + i], 60, kv, decode=True)
print(f"\nwall per decode pass (one row, memory 60): {(time.perf_counter()-t0)/N*1e3:.1f} ms")
p = m.sess.end_profiling(); report("decode pass, 1 row", p, N + 0)
for r in (3, 8):
    m.sess = IS(os.path.join(md, "model.onnx"), so, providers=["CPUExecutionProvider"]); m.sess_dec = m.sess
    t0 = time.perf_counter()
    for i in range(N): m._fwd(list(range(1000, 1000 + r)), 60, kv, decode=True)
    print(f"wall per decode pass ({r} rows): {(time.perf_counter()-t0)/N*1e3:.1f} ms"); m.sess.end_profiling()
