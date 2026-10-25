"""Runtime audit on a small FIXED batch (the first N stand-alone questions of the frozen slice), one configuration at a time on the designated cores: which checkpoint / quantization / runtime / CPU build, how many prompt tokens are cached (static prefix) versus evaluated per question (uncached), generated tokens, and the time of every stage (retrieval, state restore, prefill, decode, grammar) plus peak whole-process memory. Laptop measurements - not ARM.
    python tools_pin.py 10,11 bench/audit_runtime.py --model models/granite-4.0-1b-Q4_K_M.gguf --format intent --demos 8 --demo-cards no [--n 8]"""
import argparse, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import llama_cpp
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.modelplan import Planner, State


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--format", default="intent"); ap.add_argument("--demos", type=int, default=0); ap.add_argument("--demo-cards", default="yes"); ap.add_argument("--shots", default="core", choices=["all", "core", "none"]); ap.add_argument("--schema", default="rich"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out"); a = ap.parse_args()
    import psutil; proc = psutil.Process()
    ids = [l.strip() for l in open(os.path.join(HERE, "bench", "slice_v1.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; rows = [rows[i] for i in ids if not rows[i]["prev_text"]][: a.n]
    t = time.perf_counter(); P = Planner(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True), model_path=a.model, n_ctx="auto", dialect=(a.format if a.format != "toolcall" else None), schema_level=a.schema, core_shots={"core": True, "none": "none"}.get(a.shots, False), demo_k=a.demos, demo_cards=(a.demo_cards == "yes")); load_s = time.perf_counter() - t
    print("runtime: llama-cpp-python", llama_cpp.__version__, "|", llama_cpp.llama_print_system_info().decode()[:300].replace("\n", " "))
    print(f"checkpoint {os.path.basename(a.model)} ({os.path.getsize(a.model) / 1e6:.0f} MB file), arch {P.lm.arch}, dialect {P.dialect.name}, n_ctx {P.n_ctx}, threads 2; cold start {load_s:.1f}s (incl. static prefix {P.prefix_s:.1f}s); static prefix {P.lm.prefixes['plan']['tokens']} tokens (cached); rss after load {proc.memory_info().rss / 1e6:.0f} MB; affinity {proc.cpu_affinity()}")
    agg = {}; recs = []
    for r in rows:
        pr = P.plan(r["text"], State(), DEFAULT_NOW); tm = pr.timing; recs.append({"id": r["id"], **tm}); n_prompt = tm.get("prompt_tokens") or 0
        for k, v in tm.items():
            if isinstance(v, (int, float)): agg[k] = agg.get(k, 0) + v
    n = len(rows); mem = proc.memory_info(); peak = getattr(mem, "peak_wset", mem.rss)
    print(f"mean over {n} questions (ms): " + ", ".join(f"{k} {v / n:.0f}" for k, v in agg.items() if k.endswith("_ms")))
    print(f"prompt tokens evaluated per question (uncached, after the cached prefix): mean {agg.get('prompt_tokens', 0) / n:.0f} (incl. prefix: check prompt_tokens semantics in slm_backend); generated tokens mean {agg.get('out_tokens', 0) / n:.1f}")
    print(f"peak working set {peak / 1e6:.0f} MB; rss now {mem.rss / 1e6:.0f} MB; page faults {getattr(mem, 'num_page_faults', '?')}; cpu percent now {proc.cpu_percent(interval=0.5):.0f}; system memory used {psutil.virtual_memory().percent}%")
    if a.out: json.dump({"config": vars(a), "rows": recs}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
