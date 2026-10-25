"""One decode pass (a single token on top of a ~100-token cache) and one 24-token prefill, timed under different ONNX Runtime session settings.

    taskset -c 0,1 python bench/pi_session_knobs.py THREADS [key=value ...]     e.g.   2 allow_spinning=0     3     2 mem_pattern=0 mem_arena=0

Keys: allow_spinning (session config 0/1), mem_pattern, mem_arena (0/1), seq (execution_mode sequential/parallel), opt (graph optimisation level 0-3), denormal (1 = set denormals to zero).
Prints median ms over 25 decode passes and 8 prefills."""
import os, statistics as S, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import onnxruntime as ort

threads = int(sys.argv[1]); kv = dict(a.split("=") for a in sys.argv[2:])
_orig = ort.SessionOptions


class SO(_orig):
    pass


def make():
    so = _orig(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = {0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, 2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, 3: ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[int(kv.get("opt", 3))]
    so.add_session_config_entry("session.intra_op.allow_spinning", kv.get("allow_spinning", "1"))
    so.enable_mem_pattern = kv.get("mem_pattern", "1") == "1"; so.enable_cpu_mem_arena = kv.get("mem_arena", "1") == "1"
    if kv.get("seq", "1") == "0": so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    if kv.get("denormal") == "1": so.add_session_config_entry("session.set_denormal_as_zero", "1")
    return so


ort.SessionOptions = make
from vgpt.meaning.ort_backend import OrtFrameModel, L, NKV, HD
m = OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=threads)
m.parse("how fast am I going")                                                       # warm
ids = list(range(1000, 1024)); _, kvs = m._fwd(ids, 0, m._empty(), 1)
for _ in range(3): m._fwd([5], 24, kvs, 1)
d = []
for _ in range(25):
    t = time.perf_counter(); m._fwd([5], 24, kvs, 1); d.append((time.perf_counter() - t) * 1000)
p = []
for _ in range(8):
    t = time.perf_counter(); m._fwd(ids, 0, m._empty(), 1); p.append((time.perf_counter() - t) * 1000)
print(f"threads={threads} {kv}: decode pass median {S.median(d):.0f} ms (min {min(d):.0f}); 24-token prefill median {S.median(p):.0f} ms")
