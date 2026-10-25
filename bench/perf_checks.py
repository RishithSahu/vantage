"""Machine-dependent performance checks, kept out of the functional suite. Sets VGPT_PERF=1, pins to 2 cores, runs the timing assertions and records the conditions so the figures can be interpreted.
    python bench/perf_checks.py [--out results/perf_checks.json]
A pass here means 'within the thresholds on THIS machine under THESE conditions'; it is not evidence about the target hardware (see the guest benchmark and the open ARM requirement)."""
import json, os, platform, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = os.path.join(HERE, "results", "perf_checks.json")
if "--out" in sys.argv: out = sys.argv[sys.argv.index("--out") + 1]
env = dict(os.environ, VGPT_PERF="1"); t = time.time()
cond = {"machine": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "pinned_cores": 2, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
try:
    import psutil; cond["load_percent_before"] = psutil.cpu_percent(interval=1.0)
except Exception: pass
names = ["tests.test_camera", "tests.test_qc.QCTests", "tests.test_clf", "tests.test_robustness", "tests.test_vantage.PrefixBackend"]
r = subprocess.run([sys.executable, os.path.join(HERE, "tools_run_tests.py"), "tests.test_camera", "tests.test_clf", "tests.test_robustness", "tests.test_vantage.PrefixBackend", "tests.test_qc"], cwd=HERE, env=env, capture_output=True, text=True, timeout=1800)
cond["seconds"] = round(time.time() - t); cond["returncode"] = r.returncode; tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-12:]; cond["tail"] = tail
os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(cond, open(out, "w"), indent=1); print("\n".join(tail)); print("conditions:", {k: v for k, v in cond.items() if k != "tail"})
