"""Run benchmark configurations as separate processes (each inside the simulated 2 GB / 2-core target). A crash = 'did not fit'.
  python bench/orchestrate.py <config-name> ...      (see CONFIGS)"""
import json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M05, M15 = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
CONFIGS = {   # name: (model, mmap, slowdown, os_mb)
    "rules_k1":  ("none", 0, 1, 700), "q05_k1": (M05, 0, 1, 700), "q15_k1": (M15, 0, 1, 700), "q15_mmap_k1": (M15, 1, 1, 700),
    "q05_k4": (M05, 0, 4, 700), "q05_k8": (M05, 0, 8, 700), "q15_mmap_k4": (M15, 1, 4, 700), "q15_lean_k1": (M15, 0, 1, 300), "q05_os1000_k4": (M05, 0, 4, 1000), "q05_mmap_k1": (M05, 1, 1, 700), "q05_mmap_k4": (M05, 1, 4, 700),
}
os.makedirs(f"{ROOT}/results", exist_ok=True)
for name in sys.argv[1:]:
    model, mmap, k, osmb = CONFIGS[name]; out = f"{ROOT}/results/{name}.json"
    if os.path.exists(out): os.remove(out)
    t = time.time()
    p = subprocess.run([sys.executable, f"{ROOT}/bench/run_bench.py", "--model", model, "--mmap", str(mmap), "--slowdown", str(k), "--os-mb", str(osmb), "--out", out, "--label", name],
                       capture_output=True, text=True, cwd=ROOT, timeout=900)
    txt = (p.stdout + p.stderr)
    print(f"\n##### {name} (exit {p.returncode}, {time.time() - t:.0f}s)")
    print("\n".join(l for l in txt.splitlines() if not any(x in l for x in ("mlflow", "llama_", "ggml_", "gguf"))) [-1800:])
    if p.returncode != 0 and not os.path.exists(out):
        json.dump({"config": {"name": name, "model": model, "mmap": mmap, "slowdown": k, "os_mb": osmb}, "did_not_fit": True, "exit": p.returncode, "tail": txt[-600:]}, open(out, "w"), indent=1)
