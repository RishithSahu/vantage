"""Unattended training queue for the meaning-frame adapter: runs the arms one after the other (one GPU job at a time), evaluates each on every frozen set, applies the acceptance gate and writes a report.

    python tools_autotrain.py            start / resume (safe to run again after a crash or a reboot: finished steps are skipped)
    python tools_autotrain.py --report   only (re)write results/meaning/autopilot_report.md and autopilot_state.json

Chain (each arm = the previous sources + one more bank; a bank file that does not exist yet makes the queue stop and wait - run it again when the bank has been written):
    full_v7  platform bank                       (already trained when this script starts: it only adds the evaluations that are missing)
    full_v8  + natural_adv1.txt                  rows written from the failure FAMILIES of adv_v1
    full_v9  + natural_adv2.txt                  a second round of the same families / new ones found in v8's failures
    full_v12 + natural_adv5.txt, hard banks repeated (rehearsal) - added after v11 missed the older-set gate by 0.7 point on hard_v4
    full_v11 + natural_adv4.txt (families v10 still missed; same rank 32 / 4 epochs)
    full_v10 + natural_adv3.txt, LoRA rank 32 / alpha 64, 4 epochs   (families still failing after v7, more adapter capacity)

GATE (decided before any result was seen):
  target    every special set - platform_v1, platform_v2, adv_v1, adv_v2, adv_v3 - at or above 90 % exact frame;
  no regression   no older frozen set (novel, messy, user, sealed_v2, sealed_v3, cs5, hard_v1..v5) more than 1.5 points below the full_v6 baseline measured under the same ontology;
  the queue stops at the first arm that meets both, or when it runs out of arms.  The adapter that becomes the product default is chosen by `--adopt` in a separate step, never silently."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "meaning")
PY = sys.executable
CHAIN = [("full_v7", None), ("full_v8", "training/meaning/natural_adv1.txt"), ("full_v10", "training/meaning/natural_adv3.txt"), ("full_v11", "training/meaning/natural_adv4.txt"), ("full_v12", "training/meaning/natural_adv5.txt")]       # full_v9 (adv2 bank only, rank 16) is defined in run_arms.py but skipped: v10 contains adv1 + adv2 + adv3, so it is the one that is trained after v8
SPECIAL = {"platform_v1": 0.90, "platform_v2": 0.90, "adv_v1": 0.90, "adv_v2": 0.90, "adv_v3": 0.90}
OLD = ["novel_v1", "messy_v1", "user_v1", "cs5", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]
BASE_RUNS = (("v6_base_dev", ["novel_v1", "messy_v1", "user_v1", "cs5"]), ("v6_base_sealed", ["sealed_v2", "sealed_v3"]), ("v6_base_hard", ["hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]), ("v6_base_platform", ["platform_v1", "platform_v2"]), ("v6_base_adv", ["adv_v1", "adv_v2", "adv_v3"]))
LOG = os.path.join(RES, "autopilot.log")


def say(msg):
    os.makedirs(RES, exist_ok=True); line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"; print(line, flush=True); open(LOG, "a", encoding="utf-8").write(line + "\n")


def exact(run_dir):
    """{set: exact} of one results/meaning/<run> directory, or {}"""
    p = os.path.join(RES, run_dir, "predictions.json")
    if not os.path.exists(p): return {}
    try: d = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError): return {}
    return {k: float(v["exact"] if "exact" in v else v["act_and_concept"]) for k, v in d.get("summary", {}).items() if isinstance(v, dict) and ("exact" in v or "act_and_concept" in v)}       # cs5 is scored on act + concept (another author's gold)


def gather(prefix, runs):
    out = {}
    for r in runs: out.update(exact(r))
    return out


def arm_scores(arm):
    return gather(arm, [arm, arm + "_sealed", arm + "_sealed3", arm + "_hard1", arm + "_hard2", arm + "_hard3", arm + "_hard4", arm + "_hard5", arm + "_platform", arm + "_platform2", arm + "_adv1", arm + "_adv2", arm + "_adv3"])


def baseline():
    return gather("v6", [n for n, _ in BASE_RUNS])


def judge(arm):
    s, b = arm_scores(arm), baseline(); missing = [k for k in list(SPECIAL) + OLD if k not in s]; reasons = []
    for k, t in SPECIAL.items():
        if k in s and s[k] < t: reasons.append(f"{k} {100 * s[k]:.1f} % < {100 * t:.0f} %")
    for k in OLD:
        if k in s and k in b and s[k] < b[k] - 0.015: reasons.append(f"{k} regressed {100 * (b[k] - s[k]):.1f} points vs v6 baseline")
    return {"arm": arm, "scores": s, "missing": missing, "target_met": not reasons and not missing, "reasons": reasons}


def report():
    b = baseline(); arms = [a for a, _ in CHAIN if arm_scores(a)]; rows = []
    names = list(SPECIAL) + OLD
    head = "| set | v6 baseline | " + " | ".join(arms) + " |"; sep = "|---|---|" + "---|" * len(arms)
    for k in names: rows.append(f"| {k} | {'-' if k not in b else f'{100 * b[k]:.1f}'} | " + " | ".join(("-" if k not in arm_scores(a) else f"{100 * arm_scores(a)[k]:.1f}") for a in arms) + " |")
    verdicts = [judge(a) for a in arms]
    text = "# Autopilot report\n\nExact-frame accuracy (%), model alone (no arbitration, no dialogue rules).  Target: platform_v1, platform_v2, adv_v1, adv_v2 >= 90; no older set more than 1.5 points under the v6 baseline.\n\n" + "\n".join([head, sep] + rows) + "\n\n"
    for v in verdicts: text += f"* **{v['arm']}**: " + ("TARGET MET" if v["target_met"] else "not yet - " + "; ".join(v["reasons"] + ([f"missing: {', '.join(v['missing'])}"] if v["missing"] else []))) + "\n"
    open(os.path.join(RES, "autopilot_report.md"), "w", encoding="utf-8").write(text + f"\n_written {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    json.dump({"baseline": b, "arms": {v["arm"]: v for v in verdicts}, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(RES, "autopilot_state.json"), "w"), indent=1)
    return verdicts


def busy():
    """Another training / arm runner (not this script) is running."""
    try:
        import psutil
    except ImportError: return False
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try: c = " ".join(p.info["cmdline"] or [])
        except Exception: continue
        if p.info["pid"] != me and ("train_meaning.py" in c or "run_arms.py" in c): return True
    return False


def baseline_missing():
    """Run the v6 baseline evaluations that are not there yet (CPU only, alongside a GPU training)."""
    for name, sets in BASE_RUNS:
        if exact(name) and all(s in exact(name) for s in sets): continue
        say(f"baseline: evaluating v6 on {' '.join(sets)}")
        subprocess.run([PY, "bench/meaning_eval/eval_frames.py", "--adapter", "models/meaning-lora-full_v6.gguf", "--name", name, "--threads", "2", "--sets"] + sets, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))


def main():
    if "--report" in sys.argv: [print(v["arm"], "MET" if v["target_met"] else v["reasons"]) for v in report()]; return
    say("autopilot started")
    import threading
    t = threading.Thread(target=baseline_missing, daemon=True); t.start()
    for arm, bank in CHAIN:
        if bank and not os.path.exists(os.path.join(ROOT, bank)) or (bank and os.path.getsize(os.path.join(ROOT, bank)) < 200):
            say(f"{arm}: waiting for {bank} (not written yet) - queue stops here; run the script again when it exists"); break
        while busy(): time.sleep(30)
        say(f"{arm}: run_arms (finished steps are skipped)")
        p = subprocess.run([PY, "training/meaning/run_arms.py", arm], cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
        if p.returncode: say(f"{arm}: run_arms failed:\n{p.stdout[-1200:]}\n{p.stderr[-1200:]}"); break
        t.join(); verdict = [v for v in report() if v["arm"] == arm][0]
        say(f"{arm}: " + ("TARGET MET" if verdict["target_met"] else "not yet - " + "; ".join(verdict["reasons"])))
        if verdict["target_met"]: say("target met: the queue stops; the adapter is NOT adopted automatically"); break
    t.join(); report(); say("autopilot finished (report: results/meaning/autopilot_report.md)")


if __name__ == "__main__":
    main()
