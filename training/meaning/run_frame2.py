"""Build -> train -> convert -> evaluate the frame-version-2 adapter (docs/FRAME_V2.md).  NOT STARTED WITHOUT THE OWNER'S APPROVAL: training was stopped on 2026-10-20 after full_v11.

    python training/meaning/run_frame2.py                # everything, skipping finished steps (restartable)
    python training/meaning/run_frame2.py --dry-run      # print what each step would run and the size of the data, train nothing

Same recipe as full_v11 (LoRA rank 32 / alpha 64, 4 epochs, lr 2e-4, batch 16, seed 1, activation checkpointing; about 2 hours on the laptop GPU when it has the card to itself), on the data of build_v2.py.
The adapter is written to models/meaning-lora-frame2.gguf.  It is NOT made the default: `VGPT_FRAME=2 VGPT_MEANING_ADAPTER=models/meaning-lora-frame2.gguf` selects it (vgpt/product.py), so full_v11 stays the product
until the gate below has been read.

Gate (fixed before the run, in the spirit of docs/TRAINING_QUEUE.md): adopt only if
  (a) on the 16 older frozen sets, scored as eight words, no set loses more than 1.5 points against full_v11 (results/meaning/full_v11*) and the mean does not fall;
  (b) slots_v2: exact >= 70 % overall and >= 90 % on its controls;
  (c) followup_v2: exact >= 70 % overall and >= 85 % on `fresh` (a question that refers to nothing must inherit nothing);
  (d) answer latency p50 <= 1.0 s on two threads (the eight-word decode is two tokens longer);
  (e) the layer ablation (bench/meaning_eval/layer_ablation.py, VGPT_FRAME=2) shows `discourse` OFF is not worse than ON on followup_v2."""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); PY = sys.executable
DATA = os.path.join("training", "meaning", "data", "frame2"); OUT = os.path.join("training", "out", "meaning_frame2"); GGUF = os.path.join("models", "meaning-lora-frame2.gguf")
OLD_SETS = [("novel_v1", "novel_v1"), ("messy_v1", "messy_v1"), ("user_v1", "user_v1"), ("sealed_v2", "sealed_v2"), ("sealed_v3", "sealed_v3"), ("hard_v1", "hard_v1"), ("hard_v2", "hard_v2"), ("hard_v3", "hard_v3"), ("hard_v4", "hard_v4"),
            ("hard_v5", "hard_v5"), ("platform_v1", "platform_v1"), ("platform_v2", "platform_v2"), ("adv_v1", "adv_v1"), ("adv_v2", "adv_v2"), ("adv_v3", "adv_v3"), ("cs5", "cs5")]


def say(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, env=None, dry=False):
    say("RUN " + " ".join(cmd))
    if dry: return ""
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, **(env or {}), PYTHONUTF8="1"))
    if p.returncode: raise SystemExit("FAILED\n" + p.stdout[-2000:] + p.stderr[-2000:])
    return p.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args(); env = {"VGPT_FRAME": "2"}
    if not os.path.exists(os.path.join(ROOT, DATA, "train.jsonl")) or a.dry_run: run([PY, "training/meaning/build_v2.py", "--out", DATA], dry=a.dry_run and not os.path.exists(os.path.join(ROOT, DATA, "train.jsonl")))
    n = json.load(open(os.path.join(ROOT, DATA, "manifest.json")))["counts"]["train"] if os.path.exists(os.path.join(ROOT, DATA, "manifest.json")) else 15500
    say(f"data: {n} training rows"); epochs = 4
    if not os.path.exists(os.path.join(ROOT, GGUF)):
        run([PY, "training/meaning/train_meaning.py", "--data", DATA, "--out", OUT, "--epochs", str(epochs), "--bs", "16", "--lr", "2e-4", "--rank", "32", "--alpha", "64", "--seed", "1"], env=env, dry=a.dry_run)
        run([PY, "-m", "training.convert_lora_gguf", OUT, "--out", GGUF], env=env, dry=a.dry_run)
    if a.dry_run: say("dry run: nothing trained"); return
    if not os.path.exists(os.path.join(ROOT, "results", "meaning", "v2_frame2.json")): say(run([PY, "bench/meaning_eval/eval_v2.py", "--adapter", GGUF, "--version", "2", "--name", "frame2"], env=env))
    for name, tag in OLD_SETS:
        if os.path.exists(os.path.join(ROOT, "results", "meaning", "frame2_" + tag, "predictions.json")): continue
        say(run([PY, "bench/meaning_eval/eval_frames.py", "--adapter", GGUF, "--name", "frame2_" + tag, "--threads", "2", "--sets", name], env=env))
    say("done: read docs/FRAME_V2.md for the gate")


if __name__ == "__main__":
    main()
