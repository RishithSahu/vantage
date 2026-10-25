"""Everything that is measured after a frame-2 adapter has been trained, in one restartable command.

    python bench/meaning_eval/post_training.py --adapter models/meaning-lora-frame2.gguf --name frame2 [--skip answerability,variety]

Steps (each skipped when its result file exists, so an interrupted run continues):
  1. eval_v2        slots_v2, followup_v2, digest_v1, catalog_v1, followup_v3        -> results/meaning/v2_<name>_ext.json
  2. eval_frames    icevocab_diesel_v1, icevocab_petrol_v1, longwrap_v1               -> results/meaning/<name>_ice/predictions.json
  3. the sixteen older frozen sets (run_frame2.py does these; they are read here)
  4. rescore_v3     all older sets with the catalogue-equivalence rule                -> results/meaning/<name>_report.md
  5. answerability  every quantity x 3 forms x 7 vehicles, truth from the database    -> results/meaning/answerability_<name>.json
  6. answer_variety different prompts, different answers                              -> results/meaning/variety_<name>.json
Output of every step: results/meaning/post_training_<name>.log.  Nothing here edits a frozen file or trains anything."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); PY = sys.executable; RES = os.path.join(ROOT, "results", "meaning")


def run(cmd, log, env=None):
    t0 = time.time(); print("RUN", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, **(env or {}), PYTHONUTF8="1", HF_HUB_OFFLINE="1"))
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not l.startswith(("llama_", "load_tensors", "print_info", "ggml_", "llm_load", "sampler", "graph_", "init: ")))
    with open(log, "a", encoding="utf-8") as f: f.write(f"\n$ {' '.join(cmd)}  ({time.time() - t0:.0f} s, exit {p.returncode})\n{out}\n")
    print(out[-1800:], flush=True)
    if p.returncode: print("FAILED", p.returncode, flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--name", required=True); ap.add_argument("--skip", default=""); ap.add_argument("--old-sets", default="", help="comma separated subset of the older sets (to run several workers side by side)"); ap.add_argument("--threads", default="2"); a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x); env = {"VGPT_FRAME": "2"}; log = os.path.join(RES, f"post_training_{a.name}.log")
    if "eval_v2" not in skip and not os.path.exists(os.path.join(RES, f"v2_{a.name}_ext.json")):
        run([PY, "bench/meaning_eval/eval_v2.py", "--adapter", a.adapter, "--version", "2", "--name", a.name + "_ext", "--threads", a.threads, "--sets", "slots_v2,followup_v2,digest_v1,catalog_v1,followup_v3,followup_fresh_v4"], log, env)
    if "ice" not in skip and not os.path.exists(os.path.join(RES, f"{a.name}_ice", "predictions.json")):
        run([PY, "bench/meaning_eval/eval_frames.py", "--adapter", a.adapter, "--name", f"{a.name}_ice", "--threads", a.threads, "--sets", "icevocab_diesel_v1", "icevocab_petrol_v1", "longwrap_v1"], log, env)
    if "old" not in skip:                                                                                              # the sixteen older frozen sets (run_frame2.py already did them for the name `frame2`)
        for st in [x for x in ("novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5", "platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "cs5") if not a.old_sets or x in a.old_sets.split(",")]:
            if not os.path.exists(os.path.join(RES, f"{a.name}_{st}", "predictions.json")): run([PY, "bench/meaning_eval/eval_frames.py", "--adapter", a.adapter, "--name", f"{a.name}_{st}", "--threads", a.threads, "--sets", st], log, env)
    if "rescore" not in skip: run([PY, "bench/meaning_eval/rescore_v3.py", "--prefix", f"{a.name}_", "--out", os.path.join(RES, f"{a.name}_report.md")], log, env)
    if "answerability" not in skip and not os.path.exists(os.path.join(RES, f"answerability_{a.name}.json")):
        run([PY, "-u", "bench/meaning_eval/answerability.py", "--frame", "2", "--adapter", a.adapter, "--name", a.name, "--threads", a.threads], log, env)
    if "variety" not in skip and not os.path.exists(os.path.join(RES, f"variety_{a.name}.json")):
        run([PY, "bench/answer_variety.py", "--frame", "2", "--adapter", a.adapter, "--name", a.name, "--threads", a.threads], log, env)
    print("done; log:", log)


if __name__ == "__main__":
    main()
