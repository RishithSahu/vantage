"""Runs the understanding-versus-format diagnostic (bench/diag_format.py) for several models and output interfaces, one after the other on the same two pinned cores (never concurrently), writing results/dm_<model>_<config>.json.
    python bench/run_diag_matrix.py [--models lfm,granite,qwen] [--configs c1,c2,...] [--cores 8,9] [--stride 2]"""
import argparse, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"lfm": ("models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf", 12288), "granite": ("models/granite-4.0-1b-Q4_K_M.gguf", 12288), "qwen": ("models/qwen2.5-1.5b-instruct-q4_k_m.gguf", 12288)}
CONFIGS = {"intent_core": ["--format", "intent", "--decode", "grammar", "--shots", "core"],
           "intent_core_spans": ["--format", "intent", "--decode", "grammar", "--shots", "core", "--evidence", "spans"],
           "frame_core_spans_ops": ["--format", "frame", "--decode", "grammar", "--shots", "core", "--evidence", "spans_ops"],
           "tc_compact_core_spans": ["--format", "toolcall", "--decode", "grammar", "--schema", "compact", "--shots", "core", "--evidence", "spans"],
           "tc_compact_core_spans_ops": ["--format", "toolcall", "--decode", "grammar", "--schema", "compact", "--shots", "core", "--evidence", "spans_ops"],
           "frame_core_spans": ["--format", "frame", "--decode", "grammar", "--shots", "core", "--evidence", "spans"],
           "tc_compact_core": ["--format", "toolcall", "--decode", "grammar", "--schema", "compact", "--shots", "core"],
           "tc_rich_all": ["--format", "toolcall", "--decode", "grammar", "--schema", "rich", "--shots", "all"],
           "tc_rich_core": ["--format", "toolcall", "--decode", "grammar", "--schema", "rich", "--shots", "core"],
           "frame_core": ["--format", "frame", "--decode", "grammar", "--shots", "core"],
           "tc_free_compact_core": ["--format", "toolcall", "--decode", "free", "--schema", "compact", "--shots", "core"],
           "frame_free_core": ["--format", "frame", "--decode", "free", "--shots", "core"]}
ap = argparse.ArgumentParser(); ap.add_argument("--models", default="lfm,granite,qwen"); ap.add_argument("--configs", default=",".join(CONFIGS)); ap.add_argument("--cores", default="8,9"); ap.add_argument("--stride", default="2"); a = ap.parse_args()
for m in a.models.split(","):
    path, ctx = MODELS[m]
    for c in a.configs.split(","):
        out = os.path.join(HERE, "results", f"dm_{m}_{c}.json")
        if os.path.exists(out): print("skip (exists)", out, flush=True); continue
        cmd = [sys.executable, "tools_pin.py", a.cores, "bench/diag_format.py", "--model", path, "--n-ctx", "auto", "--stride", a.stride, "--out", out] + CONFIGS[c]
        print("RUN", m, c, flush=True); r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
        if r.returncode: print("FAILED", m, c, r.stderr[-600:], flush=True)
        else: print("done", m, c, flush=True)
