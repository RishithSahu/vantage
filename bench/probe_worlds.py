"""Does a 3-day world build for every car model?  (finds a bound quantity the world generator does not produce)   python bench/probe_worlds.py [id ...]"""
import glob, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ids = sys.argv[1:] or [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(ROOT, "vgpt", "vehicle", "packs", "*", "profile.json")))]
bad = 0
for pid in ids:
    r = subprocess.run([sys.executable, "-m", "vgpt.world.build", "--profile", pid, "--days", "3", "--seed", "3", "--out", os.path.join(ROOT, "data", "w4", "_probe.db")], capture_output=True, text=True, cwd=ROOT)
    if r.returncode: bad += 1; m = re.findall(r"KeyError: '([a-z_0-9]+)'", r.stderr); print(pid, "FAILED", ",".join(m) if m else r.stderr.strip().splitlines()[-1][:140])
print(f"{len(ids) - bad} of {len(ids)} build")
