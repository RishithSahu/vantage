"""One layer's matrix multiply, the way the frame model's decode pass does it and in other shapes, on the target board: which formulation reads the weights at the memory's speed?

    python bench/ort_matmul_bench.py build DIR          (anywhere with the `onnx` package: writes the small graphs)
    taskset -c 0,1 python bench/ort_matmul_bench.py run DIR [--threads 2]      (on the board: needs only onnxruntime)

(a) as built: the activation [R,K] is A, the constant int8 weights [K,N] are B (B is packed once, A is padded to the kernel's row tile)
(b) transposed: the constant weights [N,K] (uint8) are A, the activation columns [K,R] (int8) are B   (the weights are streamed past the small operand)
(c) float32 weights, for reference."""
import glob, os, sys, time
import numpy as np

SHAPES = ((896, 4864, "gateup"), (4864, 896, "down"), (896, 1152, "qkv"))


def build(d):
    import onnx
    from onnx import helper, numpy_helper, TensorProto as T
    os.makedirs(d, exist_ok=True); rng = np.random.default_rng(0)
    def save(name, nodes, inits, inputs, outt):
        g = helper.make_graph(nodes, "t", inits and inputs or inputs, [helper.make_tensor_value_info("y", outt, None)], inits); m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9; onnx.save(m, os.path.join(d, name + ".onnx"))
    for K, N, nm in SHAPES:
        W = rng.integers(-127, 127, (K, N), dtype=np.int8)
        for R in (1, 3, 9):
            save(f"{nm}_a_{R}", [helper.make_node("MatMulInteger", ["x", "w", "zp"], ["y"])], [numpy_helper.from_array(W, "w"), numpy_helper.from_array(np.array(128, np.uint8), "zp")], [helper.make_tensor_value_info("x", T.UINT8, [1, R, K])], T.INT32)
            Wt = np.ascontiguousarray((W.T.astype(np.int16) + 128).astype(np.uint8))
            save(f"{nm}_b_{R}", [helper.make_node("MatMulInteger", ["wt", "xb"], ["y"])], [numpy_helper.from_array(Wt, "wt")], [helper.make_tensor_value_info("xb", T.INT8, [K, R])], T.INT32)
            save(f"{nm}_c_{R}", [helper.make_node("MatMul", ["x", "w"], ["y"])], [numpy_helper.from_array(W.astype(np.float32), "w")], [helper.make_tensor_value_info("x", T.FLOAT, [1, R, K])], T.FLOAT)
    # the output head over the candidate tokens only: the 151,936 x 896 matrix, 3,000 chosen rows
    V, K, C = 151936, 896, 3000; Wrow = rng.integers(-127, 127, (V, K), dtype=np.int8); rows = np.sort(rng.choice(V, C, replace=False)).astype(np.int64)
    def head(name, nodes, inits):
        g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", T.UINT8, [1, 1, K]), helper.make_tensor_value_info("rows", T.INT64, [C])], [helper.make_tensor_value_info("y", T.INT32, None)], inits); m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9; onnx.save(m, os.path.join(d, name + ".onnx"))
    head("head_old_1", [helper.make_node("Gather", ["w", "rows"], ["b"], axis=1), helper.make_node("MatMulInteger", ["x", "b", "zp"], ["y"])], [numpy_helper.from_array(np.ascontiguousarray(Wrow.T), "w"), numpy_helper.from_array(np.array(128, np.uint8), "zp")])
    head("head_new_1", [helper.make_node("Gather", ["w", "rows"], ["g"], axis=0), helper.make_node("Transpose", ["g"], ["b"], perm=[1, 0]), helper.make_node("MatMulInteger", ["x", "b", "zp"], ["y"])], [numpy_helper.from_array(Wrow, "w"), numpy_helper.from_array(np.array(128, np.uint8), "zp")])
    print("written", len(glob.glob(os.path.join(d, "*.onnx"))), "graphs to", d)


def run(d, threads):
    import onnxruntime as ort
    rng = np.random.default_rng(1); res = {}
    for f in sorted(glob.glob(os.path.join(d, "*.onnx"))):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"]); feed = {}
        for i in s.get_inputs():
            shp = [int(x) for x in i.shape]
            if "int64" in i.type: feed[i.name] = np.sort(rng.choice(151936, shp[0], replace=False)).astype(np.int64); continue
            feed[i.name] = rng.integers(0, 127, shp).astype(np.uint8 if "uint8" in i.type else np.int8) if "int" in i.type else rng.normal(size=shp).astype(np.float32)
        s.run(None, feed); ts = []
        for _ in range(25): t = time.perf_counter(); s.run(None, feed); ts.append(time.perf_counter() - t)
        parts = os.path.basename(f)[:-5].split("_")
        if parts[0] == "head": res[("head", 1, parts[1])] = 1000 * sorted(ts)[len(ts) // 2]; continue
        nm, v, R = parts; res[(nm, int(R), v)] = 1000 * sorted(ts)[len(ts) // 2]
    for K, N, nm in SHAPES:
        print(f"{nm}  {K}x{N} = {K * N / 1e6:.1f} MB int8, memory floor at 5 GB/s {K * N / 5e6:.0f} ms")
        for R in (1, 3, 9): print("   rows %d:  (a) as built %6.1f ms   (b) transposed %6.1f ms   (c) float32 %6.1f ms" % (R, res[(nm, R, "a")], res[(nm, R, "b")], res[(nm, R, "c")]))
    head_report(res)


def head_report(res):
    if ("head", 1, "old") in res: print("output head over 3,000 candidate tokens: as built (gather columns of [896,151936]) %.1f ms  |  rows of [151936,896] then transpose %.1f ms" % (res[("head", 1, "old")], res[("head", 1, "new")]))


if __name__ == "__main__":
    if sys.argv[1] == "build": build(sys.argv[2])
    else: run(sys.argv[2], int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2)
