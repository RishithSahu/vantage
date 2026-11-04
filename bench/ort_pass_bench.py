"""Time of one ONNX Runtime decode pass of the frame model, per graph and per number of rows (1 token, or 1 token + guessed ones), with a realistic key/value state.  Meant for the target board.

    python bench/ort_pass_bench.py DIR_WITH_GRAPHS [--graphs model.onnx model_q4.onnx ...] [--threads 2] [--pos 300] [--reps 12]"""
import argparse, os, time
import numpy as np
import onnxruntime as ort

L, NKV, HD = 24, 2, 64


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--graphs", nargs="+", default=["model.onnx"]); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--pos", type=int, default=300); ap.add_argument("--reps", type=int, default=12); ap.add_argument("--rows", nargs="+", type=int, default=[1, 3, 6, 9, 16]); ap.add_argument("--profile", action="store_true", help="per-operator time of the 1-row pass"); a = ap.parse_args()
    rng = np.random.default_rng(0); cand = np.sort(rng.choice(151936, 3000, replace=False)).astype(np.int64)
    for g in a.graphs:
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if a.profile: so.enable_profiling = True
        t = time.perf_counter(); s = ort.InferenceSession(os.path.join(a.dir, g), so, providers=["CPUExecutionProvider"]); load = time.perf_counter() - t; kv = {}
        for l in range(L): kv[f"pk{l}"] = rng.normal(size=(1, NKV, a.pos, HD)).astype(np.float32); kv[f"pv{l}"] = rng.normal(size=(1, NKV, a.pos, HD)).astype(np.float32)
        out = []
        for r in a.rows:
            feed = dict(kv, ids=rng.integers(0, 150000, (1, r)).astype(np.int64), pos=np.arange(a.pos, a.pos + r, dtype=np.int64)[None], nkeep=np.array([r], np.int64), cand=cand)
            s.run(None, feed); ts = []
            for _ in range(a.reps): t = time.perf_counter(); s.run(None, feed); ts.append(time.perf_counter() - t)
            out.append("%d rows %.0f ms" % (r, 1000 * sorted(ts)[len(ts) // 2]))
        if a.profile:
            import collections, json
            ev = json.load(open(s.end_profiling())); agg = collections.defaultdict(lambda: [0.0, 0]); runs = max(1, sum(1 for e in ev if e.get("name") == "model_run")); tot = 0.0
            for e in ev:
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"): x = agg[e["args"].get("op_name", "?")]; x[0] += e["dur"]; x[1] += 1; tot += e["dur"]
            wall = [e["dur"] for e in ev if e.get("name") == "model_run"]
            print("   per run: wall %.0f ms, sum of kernel times %.0f ms, %d nodes" % (sum(wall) / len(wall) / 1000, tot / runs / 1000, sum(v[1] for v in agg.values()) // runs))
            for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:10]: print("   %-26s %6.1f ms/run  %5d nodes  %5.1f us/node" % (k, v[0] / runs / 1000, v[1] // runs, v[0] / v[1]))
        print("%-18s load %.1fs | %s" % (g, load, " | ".join(out)), flush=True)


if __name__ == "__main__": main()
