"""Build -> train -> convert -> evaluate each ablation arm, sequentially (one GPU).

    python training/meaning/run_arms.py words_only natural_only template_only no_aux no_hints
    python training/meaning/run_arms.py full_v2

Arms (all share ontology, glossary, lexicon, seed, hyper-parameters; only the data sources differ):
  full           aux + template + natural + abstain, glossary hints on
  no_aux         template + natural + abstain                  (drops the word-level Term / Pair tasks)
  template_only  template + abstain-pool                       (ontology-generated sentences only: no hand-written natural sentences)
  words_only     aux                                           (words and their correlations only: no sentence is ever seen in training)
  natural_only   natural + natural_abstain                     (hand-written sentences only, incl. their own abstain rows; not the generated abstain pool)
  no_hints       aux + template + natural + abstain, hint probability 0 (the model never sees a glossary line; also evaluated without one)
  full_v2        `full` + natural_feedback.txt (rows written from failures) and sealed_v2 added to the leakage filter
  full_v3        `full_v2` + natural_honesty.txt (predictions, counterfactuals, blame, fleet, unrecorded content + near-boundary positives) and sealed_v3 added to the leakage filter
Every arm except full_v2 is built with `--eval-leak v1`, the filter `full` was built with (this reproduces training/meaning/data/full byte for byte), so the arms differ from `full` only in their sources.
Epochs are max(3, 600 steps' worth of examples) so a small arm is not under-trained by construction.
Evaluation: the four development sets (novel_v1, messy_v1, user_v1, cs5) under results/meaning/<arm>, and sealed_v2 once under results/meaning/<arm>_sealed.  Finished steps are skipped, so a
stopped run can simply be started again.  Evaluation uses 2 CPU threads (the product's setting)."""
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
THREADS = "2"
ALL = "aux,template,natural,abstain"
ARMS = {"full": dict(sources=ALL, leak="v1"), "no_aux": dict(sources="template,natural,abstain", leak="v1"), "template_only": dict(sources="template,abstain", leak="v1"),
        "words_only": dict(sources="aux", leak="v1"), "natural_only": dict(sources="natural,natural_abstain", leak="v1"), "no_hints": dict(sources=ALL, hint_p="0", leak="v1", eval_hints=False),
        "full_v2": dict(sources=ALL + ",feedback", leak="v2"), "full_v3": dict(sources=ALL + ",feedback,honesty", leak="v3"), "full_v4": dict(sources=ALL + ",feedback,honesty,hard1", leak="v4"), "full_v5": dict(sources=ALL + ",feedback,honesty,hard1,hard2", leak="v5"),
        "full_v6": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3", leak="v6", wrap_p="0.25"),
        "full_v7": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform", leak="v7", wrap_p="0.25"),
        "full_v8": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform,adv1", leak="v8", wrap_p="0.25"),
        "full_v9": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform,adv1,adv2", leak="v9", wrap_p="0.25"),
        "full_v10": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform,adv1,adv2,adv3", leak="v10", wrap_p="0.25", rank="32", alpha="64", epochs=4),
        "full_v11": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform,adv1,adv2,adv3,adv4", leak="v11", wrap_p="0.25", rank="32", alpha="64", epochs=4),
        "full_v12": dict(sources=ALL + ",feedback,honesty,hard1,hard2,hard3,platform,adv1,adv2,adv3,adv4,adv5", leak="v12", wrap_p="0.25", rank="32", alpha="64", epochs=4, rehearse=True)}
LOG = os.path.join(ROOT, "results", "meaning", "arms.log")


def say(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); open(LOG, "a", encoding="utf-8").write(line + "\n")


def run(cmd, env=None):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if p.returncode: say("FAILED: " + " ".join(cmd) + "\n" + p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit(1)
    return p.stdout


def evaluate(arm, gg, name, sets, hints=True):
    if os.path.exists(os.path.join(ROOT, "results", "meaning", name, "predictions.json")): say(f"{name}: already evaluated, skipped"); return
    say(f"{name}: evaluate {' '.join(sets)}{'' if hints else ' (no hints)'}")
    cmd = [PY, "bench/meaning_eval/eval_frames.py", "--adapter", gg, "--name", name, "--threads", THREADS, "--sets"] + sets + ([] if hints else ["--no-hints"])
    say(f"{name}: result\n" + run(cmd, env={"PYTHONUTF8": "1"}))


def main():
    for arm in sys.argv[1:]:
        cfg = ARMS[arm]; data = os.path.join("training", "meaning", "data", arm); out = os.path.join("training", "out", "meaning_" + arm); gg = os.path.join("models", f"meaning-lora-{arm}.gguf")
        if not os.path.exists(os.path.join(ROOT, gg)):
            say(f"{arm}: build data")
            cmd = [PY, "training/meaning/build_meaning.py", "--out", data, "--seed", "7", "--template-n", "7000", "--sources", cfg["sources"], "--eval-leak", cfg["leak"]]
            if "hint_p" in cfg: cmd += ["--hint-p", cfg["hint_p"]]
            if "wrap_p" in cfg: cmd += ["--wrap-p", cfg["wrap_p"]]
            if cfg.get("rehearse"): cmd += ["--rehearse"]
            run(cmd); n = json.load(open(os.path.join(ROOT, data, "manifest.json")))["counts"]["train"]
            epochs = cfg.get("epochs") or min(12, max(3, math.ceil(600 * 16 / n))); say(f"{arm}: train n={n} epochs={epochs} rank={cfg.get('rank', '16')}")
            run([PY, "training/meaning/train_meaning.py", "--data", data, "--out", out, "--epochs", str(epochs), "--bs", "16", "--lr", "2e-4", "--rank", cfg.get("rank", "16"), "--alpha", cfg.get("alpha", "32"), "--seed", "1"], env={"PYTHONUTF8": "1"})
            run([PY, "-m", "training.convert_lora_gguf", out, "--out", gg]); say(f"{arm}: trained + converted -> {gg}")
        hints = cfg.get("eval_hints", True)
        evaluate(arm, gg, arm, ["novel_v1", "messy_v1", "user_v1", "cs5"], hints)
        evaluate(arm, gg, arm + "_sealed", ["sealed_v2"], hints)
        if arm in ("full_v3", "full_v4", "full_v5", "full_v6", "full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_sealed3", ["sealed_v3"], hints)
        if arm in ("full_v4", "full_v5", "full_v6", "full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_hard1", ["hard_v1"], hints)
        if arm in ("full_v5", "full_v6", "full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_hard2", ["hard_v2"], hints); evaluate(arm, gg, arm + "_hard3", ["hard_v3"], hints)
        if arm in ("full_v6", "full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_hard4", ["hard_v4"], hints); evaluate(arm, gg, arm + "_hard5", ["hard_v5"], hints)
        if arm in ("full_v6", "full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_platform", ["platform_v1"], hints); evaluate(arm, gg, arm + "_platform2", ["platform_v2"], hints)
        if arm in ("full_v7", "full_v8", "full_v9", "full_v10", "full_v11", "full_v12"): evaluate(arm, gg, arm + "_adv1", ["adv_v1"], hints); evaluate(arm, gg, arm + "_adv2", ["adv_v2"], hints); evaluate(arm, gg, arm + "_adv3", ["adv_v3"], hints)
        if arm == "full": evaluate(arm, gg, "full_nohint", ["novel_v1", "messy_v1", "user_v1", "cs5"], False)


if __name__ == "__main__":
    main()
