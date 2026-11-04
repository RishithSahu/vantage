"""ONNX Runtime's own int8 matrix multiply on the model's shapes (the baseline for bench/arm/a72kernel.c).  Run on the Pi:
    taskset -c 0,1 ~/vehiclegpt2/.venv/bin/python ort_matmul_bench.py
Times three graphs per shape and token count M: MatMulInteger alone (uint8 activations x int8 weights -> int32); the product's per-token quantise + MatMulInteger + rescale chain; and com.microsoft DynamicQuantizeMatMul."""
import time, numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper

def sess(nodes, inputs, inits, outname, outtype):
    g = helper.make_graph(nodes, "g", inputs, [helper.make_tensor_value_info(outname, outtype, None)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)]); m.ir_version = 8
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

def timeit(f, reps):
    for _ in range(3): f()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best

rng = np.random.default_rng(1)
for K, N in ((896, 4864), (4864, 896), (896, 896)):
    W = rng.integers(-127, 128, (K, N), dtype=np.int8); ws = np.full(N, 0.01, np.float32)
    inits = [numpy_helper.from_array(W, "w"), numpy_helper.from_array(ws, "ws"), numpy_helper.from_array(np.array(128, np.uint8), "zp")]
    s1 = sess([helper.make_node("MatMulInteger", ["xq", "w", "zp"], ["y"])], [helper.make_tensor_value_info("xq", T.UINT8, ["M", K])], inits, "y", T.INT32)
    s2 = sess([helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "sx", "zx"]), helper.make_node("MatMulInteger", ["xq", "w", "zx"], ["yi"]), helper.make_node("Cast", ["yi"], ["yf"], to=T.FLOAT),
               helper.make_node("Mul", ["yf", "sx"], ["y1"]), helper.make_node("Mul", ["y1", "ws"], ["y"])], [helper.make_tensor_value_info("x", T.FLOAT, ["M", K])], inits, "y", T.FLOAT)
    s3 = sess([helper.make_node("DynamicQuantizeMatMul", ["x", "w", "ws"], ["y"], domain="com.microsoft")], [helper.make_tensor_value_info("x", T.FLOAT, ["M", K])], inits, "y", T.FLOAT)
    for M in (1, 2, 4, 8, 16, 32, 48):
        xq = rng.integers(0, 255, (M, K), dtype=np.uint8); x = rng.standard_normal((M, K)).astype(np.float32); ops = 2.0 * M * K * N; out = []
        for nm, s, feed in (("MatMulInteger", s1, {"xq": xq}), ("quantise+MMI+rescale", s2, {"x": x}), ("DynQuantMatMul", s3, {"x": x})):
            try: dt = timeit(lambda: s.run(None, feed), 15 if M < 16 else 8); out.append(f"{nm} {dt*1e3:7.3f} ms {ops/dt/1e9:5.1f} GOPS")
            except Exception as e: out.append(f"{nm} failed: {str(e)[:60]}")
        print(f"{K}x{N} M={M:<3}", " | ".join(out), flush=True)
