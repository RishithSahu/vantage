"""DEV tool: every release gate in one command.   python tools_release_gate.py [--quick]

Gates (each prints PASS / FAIL with the number behind it; exit code 1 if any FAIL):
  1. the whole unit / integration test suite, with the model, the corpus, the driving fixture and the sample cars REQUIRED (a missing one fails, it does not skip); the legacy generated worlds are required only with --all-data
  1b. release provenance: a manifest of this tree (source, models, vocabulary, test data, runtime) verifies against the tree; the tree is a git commit with a tag (a warning, a failure with --require-tag)
  2. understanding of unseen everyday questions: tuning sets >= 95%, final held-out and messy sets >= 90%
  3. the older analysis routes untouched: torture sets >= 98%, extended eval 100%
  4. off-topic questions never answered as if about the car
  5. the package builds, extracts, passes doctor and its own smoke test in a clean environment
  6. a fresh virtual environment with ONLY requirements.txt installed can run the extracted package   (skipped with --quick; needs internet once)
Not automatable here, and listed in RELEASE_CHECKLIST.md: choose and add the LICENSE, resolve the VERIFY lines in THIRD_PARTY_NOTICES.md, try it on a second (not developer) computer."""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
results = []


def gate(name, ok, detail): results.append((name, ok, detail)); print(f"[{'WARN' if ok is None else 'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def run(args, timeout=1800, cwd=ROOT, env=None):
    return subprocess.run([PY, "-W", "ignore"] + args, cwd=cwd, env=env or ENV, capture_output=True, text=True, timeout=timeout)


def test_modules():
    return sorted("tests." + f[:-3] for f in os.listdir(os.path.join(ROOT, "tests")) if f.startswith("test_") and f.endswith(".py"))


def main(quick=False, all_data=False, require_tag=False):
    t0 = time.time()
    env = dict(ENV, VGPT_REQUIRE_TESTDATA="1" if all_data else "driving_fixture,corpus,sample_cars,models")                      # absent data FAILS here (tests/testdata.py)
    r = run(["-m", "unittest"] + test_modules(), env=env); tail = (r.stderr or "").strip().splitlines()[-3:]; m = re.search(r"Ran (\d+) tests", r.stderr or "")
    gate("tests", r.returncode == 0, f"{m[1] if m else '?'} tests; " + " | ".join(tail))
    import tools_release_manifest as RM
    from vgpt import releasecheck as RC
    man = RM.make(ROOT); bad = RC.verify(ROOT, man, models=False); gate("release manifest verifies against the tree", not bad, f"source {man['source']['digest'][:12]}, {len(man['source']['files'])} files" + ("" if not bad else "; " + bad[0]))
    v = man["vcs"]; gate("release is a tagged git commit", True if v["releasable_from_tag"] else (False if require_tag else None), f"{v['kind']}: " + (f"tag {v.get('tag')} at {str(v.get('commit'))[:10]}" if v["kind"] == "git" and v.get("tag") else v.get("note") or "no tag on this commit, or uncommitted changes"))
    import eval_everyday as E
    db = os.path.join(ROOT, "data", "gen_courier.db"); res = {}
    for f, floor in (("heldout_dev.tsv", .95), ("heldout_dev2.tsv", .95), ("heldout_final.tsv", .90), ("heldout_messy.tsv", .90)):
        ok, tot, _ = E.run(db, [os.path.join(ROOT, "bench", f)]); res[f] = ok / tot; gate(f"understanding {f}", ok / tot >= floor, f"{ok}/{tot} = {100 * ok / tot:.0f}% (floor {int(floor * 100)}%)")
    r = run([os.path.join("bench", "eval_torture.py")]); pct = [int(x) for x in re.findall(r"\((\d+)%\)", r.stdout)]; gate("older analysis routes (eval_torture)", bool(pct) and min(pct) >= 98, f"{pct}%")
    r = run([os.path.join("bench", "eval_ext.py"), "none"]); m = re.search(r"TOTAL (\d+)/(\d+)", r.stdout); gate("extended eval", bool(m) and m[1] == m[2], m[0] if m else r.stdout[-200:])
    from tests.test_routing import OOD
    from vgpt import brain
    b = brain.Brain(db); leaks = [q for q in OOD if b.ask(q, use_slm=False)["path"] in ("qa", "fast")]; gate("off-topic never answered as car", not leaks, f"{len(OOD) - len(leaks)}/{len(OOD)} declined" + (f"; leaked: {leaks}" if leaks else ""))
    import tools_release as R
    out = tempfile.mkdtemp(prefix="vgpt_gate_"); dst, zp = R.build(out, verbose=False); ok, tail = R.verify(dst); gate("package builds + self-test in clean environment", ok, f"{os.path.getsize(zp) / 1e6:.0f} MB zip" + ("" if ok else "; " + tail[-200:]))
    if not quick:
        venv = os.path.join(out, "venv"); subprocess.run([PY, "-m", "venv", venv], check=True); vpy = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
        p = subprocess.run([vpy, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", os.path.join(dst, "requirements.txt")], capture_output=True, text=True, timeout=1800)
        if p.returncode: gate("fresh environment with only requirements.txt", False, "pip failed: " + p.stderr[-200:])
        else:
            env = dict(R.clean_env(), VEHICLEGPT_HOME=tempfile.mkdtemp()); q = subprocess.run([vpy, os.path.join(dst, "vehiclegpt.py"), "smoke"], cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=900)
            gate("fresh environment with only requirements.txt", q.returncode == 0, (q.stdout.strip().splitlines() or [""])[-1] + ("" if q.returncode == 0 else q.stderr[-200:]))
    bad = [n for n, ok, _ in results if not ok]; print(f"\n{'ALL GATES PASSED' if not bad else 'FAILED: ' + ', '.join(bad)}   ({time.time() - t0:.0f} s)"); return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main("--quick" in sys.argv, "--all-data" in sys.argv, "--require-tag" in sys.argv))
