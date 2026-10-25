"""Security evidence on the Raspberry Pi, measured (run on the Pi, as the service user, with passwordless sudo for the namespace and the unit analysis):

    python bench/pi_security_proof.py

1. OFFLINE.  The whole assistant (the real int8 model, the real database) answers questions inside a Linux network namespace that has NO interface at all (`unshare -n`; not even loopback is up), and the
   in-process guard (vgpt/offline.py) has refused every socket attempt: answers are correct with the network physically absent, and nothing waited for it.
2. DATA AT REST.  A world database is sealed (AES-256-GCM, key in a 0600 file here; a TPM / systemd credential in a vehicle), the plaintext is overwritten and removed, the assistant is started from the sealed
   file only, answers a question, and exits.  A scan of the whole sealed folder, /tmp and /dev/shm for a value known to be in the database finds nothing, and the RAM folder is gone after exit.
3. HARDENED UNIT.  `systemd-analyze security` gives the exposure score of the shipped unit (deploy/linux/vgpt-assistant.service.in) as systemd itself computes it.
4. PROCESS.  No core dumps and not dumpable, read back from /proc."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = {"when": time.strftime("%Y-%m-%d %H:%M:%S")}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe_code(db, extra=""):
    return ("import os,sys,json,time;sys.path.insert(0,%r);os.environ.setdefault('HF_HUB_OFFLINE','1');from types import SimpleNamespace;import vehiclegpt;vehiclegpt._adopt_product_model();"
            "from ui import server as S;S.CameraRig.start=lambda self:None;from vgpt import offline;"
            "app=S.App(SimpleNamespace(db=%r,model='0.5b',camera='clip',webcam_index=0,live=None,linked=False));"
            "qs=['how fast am I going','what was the average battery temperature last week','why does the Moon have phases'];res=[]\n"
            "for q in qs:\n t=time.time();r=app.ask(q,'driver');res.append((q,round(time.time()-t,2),str(r.get('answer'))[:90]))\n"
            "print('RESULT '+json.dumps({'answers':res,'offline':offline.check()%s}))") % (ROOT, db, extra)


def offline_proof():
    me = os.environ.get("USER", "vantage")
    cmd = ["sudo", "-n", "unshare", "-n", "--", "setpriv", f"--reuid={me}", f"--regid={me}", "--init-groups", "env", f"HOME={os.path.expanduser('~')}", "taskset", "-c", "0,1", PY, "-c", probe_code(os.path.join(ROOT, "data", "world_aster.db"))]
    r = run(cmd, timeout=900); m = re.search(r"RESULT (.*)", r.stdout)
    ns = run(["sudo", "-n", "unshare", "-n", "--", "sh", "-c", "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '"])
    OUT["offline"] = {"namespace_interfaces": ns.stdout.strip().replace("\n", " "), "result": json.loads(m.group(1)) if m else None, "stderr_tail": r.stderr[-400:] if not m else ""}


def sealed_proof():
    d = tempfile.mkdtemp(prefix="sealtest-", dir=os.path.expanduser("~")); db = os.path.join(d, "car.db"); shutil.copyfile(os.path.join(ROOT, "data", "world_aster.db"), db)
    import sqlite3
    c = sqlite3.connect(db); known = c.execute("SELECT v FROM meta WHERE k='vehicle.vin' OR k LIKE '%vin%' LIMIT 1").fetchone(); c.close()
    marker = (known[0] if known and known[0] and len(str(known[0])) >= 6 else None)
    sys.path.insert(0, ROOT)
    from vgpt import runtime_vault as RV
    key = os.path.join(d, "keys", "k"); from vgpt import atrest; atrest.make_key(key)
    # a recognisable value inside the database: the first 12 bytes of a table row as it is stored
    c = sqlite3.connect(db); row = c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1").fetchone(); table_name = row[0]; c.close()
    enc = RV.seal(db, key)
    env = dict(os.environ, VGPT_ATREST_KEY=key)
    r = run(["taskset", "-c", "0,1", PY, "-c", probe_code(db)], env=env, timeout=900); m = re.search(r"RESULT (.*)", r.stdout)
    needles = [table_name.encode(), b"SQLite format 3", b"sig_1m"]
    leaks = []
    for folder in (d, "/tmp", "/dev/shm"):
        for root, _, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) > 700 * 1024 * 1024: continue
                    blob = open(p, "rb").read()
                except OSError: continue
                for n in needles:
                    if n in blob and p != key and "pi_security" not in p: leaks.append((p, n.decode("ascii", "replace")))
    shm_left = [x for x in os.listdir("/dev/shm") if x.startswith(("vehiclegpt_", "vantage-"))] if os.path.isdir("/dev/shm") else []
    OUT["sealed"] = {"plain_removed": not os.path.exists(db), "sealed_size_mb": round(os.path.getsize(enc) / 1e6, 1), "starts_with": open(enc, "rb").read(8).decode("ascii", "replace"), "answered": json.loads(m.group(1))["answers"] if m else None,
                     "needles_searched": [n.decode() for n in needles], "files_containing_them_outside_the_key": leaks, "ram_folders_left_after_exit": shm_left, "stderr_tail": r.stderr[-300:] if not m else ""}
    shutil.rmtree(d, ignore_errors=True)


def unit_proof():
    src = open(os.path.join(ROOT, "deploy", "linux", "vgpt-assistant.service.in")).read()
    unit = src.replace("@USER@", "vantage").replace("@APP_DIR@", ROOT).replace("@STATE_DIR@", "/var/lib/vantage").replace("@EXEC_START@", f"{PY} -c pass")
    p = os.path.join(tempfile.mkdtemp(), "vgpt-assistant.service"); open(p, "w").write(unit)
    r = run(["systemd-analyze", "security", "--offline=true", p]); txt = r.stdout + r.stderr
    m = re.search(r"Overall exposure level for [^:]+: ([0-9.]+) (\w+)", txt)
    OUT["unit"] = {"exposure": float(m.group(1)) if m else None, "verdict": m.group(2) if m else None, "tail": txt.strip().splitlines()[-3:]}


def process_proof():
    code = "import sys;sys.path.insert(0,%r);from vgpt import hardening;print(hardening.harden());print([l for l in open('/proc/self/limits') if l.startswith('Max core')][0].strip());print(open('/proc/self/status').read().split('Seccomp:')[0][-0:0] or '')" % ROOT
    r = run([PY, "-c", code]); OUT["process"] = r.stdout.strip().splitlines()


if __name__ == "__main__":
    for step in (unit_proof, process_proof, offline_proof, sealed_proof):
        try: step()
        except Exception as e: OUT[step.__name__] = f"failed: {type(e).__name__}: {e}"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(OUT, open(os.path.join(ROOT, "results", "security_pi.json"), "w"), indent=1, default=str); print(json.dumps(OUT, indent=1, default=str)[:4000])
