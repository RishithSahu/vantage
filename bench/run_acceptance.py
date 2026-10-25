"""Runs the FROZEN configuration on the acceptance set exactly once.
    python tools_pin.py 8,9 bench/run_acceptance.py
Refuses to run unless (1) results/acceptance/freeze.json exists, (2) the question file, the scoring code, the evaluation databases and the system source still hash to the frozen values, and (3) no previous acceptance run exists. The result is written together with the freeze record.
If failures from this run later inform fixes, that must be disclosed and the set is regression data from then on. This script has NOT been run by the implementation (see VANTAGE_STATUS.md)."""
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.manifest import manifest, _file_hash
ACC = os.path.join(HERE, "bench", "acceptance_v1.tsv"); OUT = os.path.join(HERE, "results", "acceptance"); FR = os.path.join(OUT, "freeze.json"); RUN = os.path.join(OUT, "acceptance_run.json")


def main():
    if not os.path.exists(FR): sys.exit("not frozen: run bench/freeze_acceptance.py after the independent review")
    if os.path.exists(RUN): sys.exit("the acceptance set was already run once (results/acceptance/acceptance_run.json); it is regression data now")
    fz = json.load(open(FR)); cur = manifest(fz["config"]); bad = []
    if _file_hash(ACC) != fz["acceptance_questions"]: bad.append("acceptance questions changed since the freeze")
    if cur["source_hash"] != fz["source_hash"]: bad.append("system source changed since the freeze")
    if cur["databases"] != fz["databases"]: bad.append("evaluation databases changed since the freeze")
    if cur.get("models") != fz.get("models"): bad.append("model files changed since the freeze")
    for f, h in fz["scoring_code"].items():
        if hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16] != h: bad.append(f"{f} changed since the freeze")
    if bad: sys.exit("refusing to run: " + "; ".join(bad))
    out = os.path.join(OUT, "acceptance_eval.json"); t = time.time()
    cfg = fz["config"]; cmd = [sys.executable, os.path.join(HERE, "bench", "eval_v2.py"), "--system", cfg["system"], "--set", ACC, "--history", "own", "--out", out] + (["--model", os.path.join(HERE, cfg["selector_model"])] if cfg.get("selector_model") else [])
    r = subprocess.run(cmd, cwd=HERE); json.dump({"returncode": r.returncode, "seconds": round(time.time() - t), "freeze": fz, "result_file": out}, open(RUN, "w"), indent=1); print("acceptance run done; results in", out)


if __name__ == "__main__": main()
