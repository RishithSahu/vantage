"""Everything that follows the round-10 training, run unattended.

    python bench/after_train.py [--log logs_train_r10.txt] [--out training/out/meaning_frame2r10] [--name frame2r10]

1. waits for the trainer's "saved <out>" line;
2. converts the adapter to a llama.cpp LoRA (models/meaning-lora-<name>.gguf);
3. on the CPU: evaluates it on every frozen set and the new sets with the car type told to the model (eval_r10 --vehicle-hint), then the hint's value (the same weights without the car-type line, on the sets where it
   matters), writing results/meaning/<name>*_r10/ and logs_eval_<name>*.txt;
4. on the GPU, at the same time: the per-car-type experiment - the shared adapter fine-tuned for one epoch on (a) the combustion rows, (b) a random mix of the same size (the compute-matched control), each converted and
   evaluated on the combustion sets.
Each step writes its own log; a failing step is logged and the next one still runs."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(os.path.join(ROOT, "logs_after_train.txt"), "a", encoding="utf-8") as f: f.write(line + "\n")


def run(cmd, log=None, env=None, wait=True):
    say("RUN " + " ".join(cmd) + (f" > {log}" if log else ""))
    e = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", **(env or {}))
    fh = open(os.path.join(ROOT, log), "w", encoding="utf-8") if log else None
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh or subprocess.DEVNULL, stderr=subprocess.STDOUT, env=e)
    if not wait: return p
    rc = p.wait(); say(f"  exit {rc}")
    return rc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--log", default="logs_train_r10.txt"); ap.add_argument("--out", default="training/out/meaning_frame2r10"); ap.add_argument("--name", default="frame2r10"); a = ap.parse_args()
    say(f"waiting for the trainer ({a.log})")
    while True:
        try: txt = open(os.path.join(ROOT, a.log), encoding="utf-8", errors="replace").read()
        except OSError: txt = ""
        if "saved " + a.out.replace("/", os.sep) in txt or "saved " + a.out in txt: break
        time.sleep(60)
    say("training finished")
    gg = f"models/meaning-lora-{a.name}.gguf"
    run([PY, "-m", "training.convert_lora_gguf", a.out, "--out", gg], log=f"logs_convert_{a.name}.txt", env={"VGPT_FRAME": "2"})
    if not os.path.exists(os.path.join(ROOT, gg)): say("conversion failed; stopping"); return
    # CPU: the full evaluation with the car type told to the model, then the hint ablation
    cpu = run([PY, "bench/meaning_eval/eval_r10.py", "--adapter", gg, "--name", a.name + "_hint", "--vehicle-hint", "--default-vehicle", "ev"], log=f"logs_eval_{a.name}_hint.txt", wait=False, env={"VGPT_FRAME": "2"})
    # GPU: the per-car-type experiment, in parallel
    init = os.path.join(a.out, "adapter.safetensors")
    for tag in ("ice", "mix"):
        out = f"training/out/meaning_{a.name}_{tag}"; data = f"training/meaning/data/frame2_r10_{tag}"
        rc = run([PY, "training/meaning/train_meaning.py", "--data", data, "--out", out, "--epochs", "1", "--bs", "16", "--lr", "1e-4", "--rank", "32", "--alpha", "64", "--seed", "2", "--init-adapter", init], log=f"logs_train_{a.name}_{tag}.txt", env={"VGPT_FRAME": "2", "VGPT_VEHICLE_HINT": "1"})
        if rc == 0: run([PY, "-m", "training.convert_lora_gguf", out, "--out", f"models/meaning-lora-{a.name}_{tag}.gguf"], log=f"logs_convert_{a.name}_{tag}.txt", env={"VGPT_FRAME": "2"})
    cpu.wait(); say("hint evaluation finished")
    ice_sets = ["ice_v1", "icevocab_petrol_v1", "icevocab_diesel_v1", "sealed_v4", "sealed_v5", "wording_v1"]
    run([PY, "bench/meaning_eval/eval_r10.py", "--adapter", gg, "--name", a.name + "_nohint", "--default-vehicle", "none", "--sets", *ice_sets, "--no-vehicle-tag"], log=f"logs_eval_{a.name}_nohint.txt", env={"VGPT_FRAME": "2"})
    for tag in ("ice", "mix"):
        g = f"models/meaning-lora-{a.name}_{tag}.gguf"
        if os.path.exists(os.path.join(ROOT, g)): run([PY, "bench/meaning_eval/eval_r10.py", "--adapter", g, "--name", f"{a.name}_{tag}", "--vehicle-hint", "--default-vehicle", "ev", "--sets", *ice_sets], log=f"logs_eval_{a.name}_{tag}.txt", env={"VGPT_FRAME": "2"})
    say("all steps done")


if __name__ == "__main__":
    main()
