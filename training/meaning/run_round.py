"""One further round of training for the frame-2 adapter: continue from an adapter on a new build of the data, convert it, and measure everything in parallel workers.

    python training/meaning/run_round.py --name frame2b --data training/meaning/data/frame2_r2 --init training/out/meaning_frame2/adapter.safetensors --epochs 2 --lr 1.2e-4 --seed 2

Steps (each skipped when its output exists): train (train_meaning.py --init-adapter) -> training/out/meaning_<name>; convert -> models/meaning-lora-<name>.gguf; then bench/meaning_eval/post_training.py in four
workers side by side (extended sets + ICE sets / older sets / answerability / variety) and the rescoring.  Nothing frozen is edited; the previous adapter is never overwritten."""
import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); PY = sys.executable


def say(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, env=None):
    say("RUN " + " ".join(cmd)); p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, **(env or {}), PYTHONUTF8="1", HF_HUB_OFFLINE="1"))
    if p.returncode: raise SystemExit("FAILED\n" + p.stdout[-2500:] + p.stderr[-2500:])
    return p.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--data", required=True); ap.add_argument("--init", default=None); ap.add_argument("--epochs", default="2"); ap.add_argument("--lr", default="1.2e-4")
    ap.add_argument("--seed", default="2"); ap.add_argument("--warmup", default="40"); ap.add_argument("--skip-eval", action="store_true"); a = ap.parse_args()
    out = os.path.join("training", "out", "meaning_" + a.name); gguf = os.path.join("models", f"meaning-lora-{a.name}.gguf"); env = {"VGPT_FRAME": "2"}
    if not os.path.exists(os.path.join(ROOT, out, "adapter.safetensors")):
        cmd = [PY, "training/meaning/train_meaning.py", "--data", a.data, "--out", out, "--epochs", a.epochs, "--bs", "16", "--lr", a.lr, "--rank", "32", "--alpha", "64", "--seed", a.seed, "--warmup", a.warmup]
        if a.init: cmd += ["--init-adapter", a.init]
        run(cmd, env)
    if not os.path.exists(os.path.join(ROOT, gguf)): run([PY, "-m", "training.convert_lora_gguf", out, "--out", gguf], env)
    if a.skip_eval: say("trained and converted; evaluation skipped"); return
    pt = [PY, "-u", "bench/meaning_eval/post_training.py", "--adapter", gguf, "--name", a.name]; jobs = [
        ("extended + ICE", pt + ["--skip", "old,rescore,answerability,variety"]),
        ("older sets A", pt + ["--skip", "eval_v2,ice,rescore,answerability,variety", "--old-sets", "novel_v1,messy_v1,user_v1,sealed_v2,sealed_v3,hard_v1,hard_v2,hard_v3"]),
        ("older sets B", pt + ["--skip", "eval_v2,ice,rescore,answerability,variety", "--old-sets", "hard_v4,hard_v5,platform_v1,platform_v2,adv_v1,adv_v2,adv_v3,cs5"]),
        ("answerability", pt + ["--skip", "eval_v2,ice,old,rescore,variety"]),
        ("variety", pt + ["--skip", "eval_v2,ice,old,rescore,answerability"])]
    ts = []
    for label, cmd in jobs:
        t = threading.Thread(target=lambda c=cmd, l=label: (say("start " + l), run(c, env), say("done " + l))); t.start(); ts.append(t)
    for t in ts: t.join()
    say(run([PY, "bench/meaning_eval/rescore_v3.py", "--prefix", a.name + "_", "--out", os.path.join("results", "meaning", a.name + "_report.md")], env)[-3000:])
    say("round finished: " + a.name)


if __name__ == "__main__":
    main()
