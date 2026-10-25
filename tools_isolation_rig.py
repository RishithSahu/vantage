"""Freedom-from-interference rig, run ON the vehicle-class board (the Raspberry Pi), with the real kernel doing the isolating.

    python tools_isolation_rig.py [--seconds 45] [--only attack,ask,...] [--out results/isolation/pi.json]

What it does.  A stand-in for a safety function (tools/rt_task.c: a 100 Hz control loop with a 200 us computation, real-time priority, its own core) runs while the assistant, started as a hardened systemd unit
(deploy/linux/vgpt-assistant.service.in: no network, no CAN, no write access to the vehicle's files, no way to see or signal other processes, no real-time priority, only the cores and memory it was given), does
the worst it can.  The loop's wake-up lateness and deadline misses are compared with a run in which the assistant does nothing.

Scenarios: baseline | idle (model loaded, no questions) | ask (answering as fast as it can) | cpu (every core it may use at 100 %) | cpu-shared (allowed onto the SAME core as the safety loop) | mem (allocate until killed, restarted, again)
| fork (fork bomb) | io (write and fsync flood) | crash (SIGKILL every second or two, restarted) | attack (every escape attempt the sandbox is meant to close; must all be refused).
Needs passwordless sudo (systemd-run); changes nothing permanent (transient units, a scratch folder in /var/tmp)."""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__)); HOME = os.path.expanduser("~"); SCRATCH = "/var/lib/vgpt-scratch"
PY = os.path.join(HOME, "vehiclegpt2", ".venv", "bin", "python") if os.path.exists(os.path.join(HOME, "vehiclegpt2", ".venv", "bin", "python")) else sys.executable

HARDEN = ["User=" + os.environ.get("USER", "vantage"), "WorkingDirectory=" + ROOT, "NoNewPrivileges=yes", "CapabilityBoundingSet=", "AmbientCapabilities=", "PrivateNetwork=yes", "RestrictAddressFamilies=AF_UNIX", "IPAddressDeny=any",
          "ProtectSystem=strict", "ProtectHome=read-only", "PrivateTmp=yes", "PrivateDevices=yes", "ReadWritePaths=" + SCRATCH, "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectKernelLogs=yes", "ProtectControlGroups=yes",
          "ProtectClock=yes", "ProtectHostname=yes", "ProtectProc=invisible", "ProcSubset=pid", "PrivatePIDs=yes", "RestrictNamespaces=yes", "RestrictRealtime=yes", "RestrictSUIDSGID=yes", "LockPersonality=yes", "SystemCallArchitectures=native",
          "SystemCallFilter=@system-service", "SystemCallFilter=~@privileged", "SystemCallErrorNumber=EPERM", "UMask=0077", "LimitNOFILE=4096", "MemorySwapMax=0", "IOWeight=20"]
LIMITS = {"MemoryMax": "2G", "TasksMax": "256", "CPUQuota": "200%", "AllowedCPUs": "2-3"}
SAFETY_CPU = "0"


def sh(cmd, check=False, timeout=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode: raise RuntimeError(f"{' '.join(cmd)[:120]}: {r.stderr.strip()[:300]}")
    return r


def sd(unit, props, argv, wait=False, output=None, restart=None):
    cmd = ["sudo", "-n", "systemd-run", "--unit=" + unit, "--collect", "-p", "Type=exec"]
    for p in props: cmd += ["-p", p]
    if output: cmd += ["-p", "StandardOutput=file:" + output, "-p", "StandardError=file:" + output + ".err"]
    if restart: cmd += ["-p", "Restart=always", "-p", "RestartSec=1"]
    if wait: cmd += ["--wait"]
    return sh(cmd + argv, timeout=900 if wait else 60)


def rd(path):
    """Read a file the units wrote (they run as other users with a strict umask)."""
    r = sh(["sudo", "-n", "cat", path]); return r.stdout if r.returncode == 0 else ""


def stop(unit): sh(["sudo", "-n", "systemctl", "stop", unit + ".service"]); sh(["sudo", "-n", "systemctl", "reset-failed", unit + ".service"])


def prepare():
    sh(["sudo", "-n", "mkdir", "-p", SCRATCH], check=True); sh(["sudo", "-n", "chown", os.environ.get("USER", "vantage"), SCRATCH], check=True)
    rt = os.path.join(SCRATCH, "rt_task")
    if not os.path.exists(rt): sh(["gcc", "-O2", "-o", rt, os.path.join(ROOT, "tools", "rt_task.c")], check=True)
    for f in os.listdir(SCRATCH):
        if f.endswith((".json", ".err", ".out")): sh(["sudo", "-n", "rm", "-f", os.path.join(SCRATCH, f)])
    db = os.path.join(SCRATCH, "world.db")
    if not os.path.exists(db): sh(["cp", os.path.join(ROOT, "data", "world_aster.db"), db], check=True)         # the assistant's own working copy of a history it reads (the live one it cannot write)
    return rt, db


def run_scenario(name, rt, db, seconds):
    out = os.path.join(SCRATCH, f"safety_{name}.json"); res = {"scenario": name}
    stop("vgpt-safety"); stop("vgpt-assistant")
    sd("vgpt-safety", ["User=root", "CPUSchedulingPolicy=fifo", "CPUSchedulingPriority=50", "AllowedCPUs=" + SAFETY_CPU, "CPUAffinity=" + SAFETY_CPU], [rt, "10000", str(seconds), "200", "1000"], output=out)
    time.sleep(2.5)
    pid = sh(["systemctl", "show", "vgpt-safety.service", "-p", "MainPID", "--value"]).stdout.strip()
    props = HARDEN + [f"{k}={v}" for k, v in LIMITS.items()]; env = ["Environment=VGPT_SAFETY_PID=" + pid, "Environment=VGPT_DB=" + db, "Environment=VGPT_SCRATCH=" + SCRATCH, "Environment=HF_HUB_OFFLINE=1", "Environment=OMP_NUM_THREADS=2"]
    load = os.path.join(ROOT, "bench", "isolation_load.py"); aout = os.path.join(SCRATCH, f"assistant_{name}.out")
    if name == "baseline": pass
    elif name == "idle": sd("vgpt-assistant", props + env, [PY, load, "cpu", "0.1"], output=aout)                           # started, does nothing
    elif name == "ask": sd("vgpt-assistant", props + env, [PY, load, "ask", str(seconds - 3)], output=aout)
    elif name == "cpu": sd("vgpt-assistant", props + env, [PY, load, "cpu", str(seconds - 3)], output=aout)
    elif name == "cpu-shared": sd("vgpt-assistant", [p for p in props if not p.startswith("AllowedCPUs")] + ["AllowedCPUs=0-3", "CPUWeight=20"] + env, [PY, load, "cpu", str(seconds - 3)], output=aout)
    elif name == "mem": sd("vgpt-assistant", [p for p in props if not p.startswith("MemoryMax")] + ["MemoryMax=1G"] + env, [PY, load, "mem"], output=aout, restart=True)
    elif name == "fork": sd("vgpt-assistant", props + env, [PY, load, "fork", str(seconds - 3)], output=aout)
    elif name == "io": sd("vgpt-assistant", props + env, [PY, load, "io", str(seconds - 3)], output=aout)
    elif name == "crash": sd("vgpt-assistant", props + env, [PY, load, "crash"], output=aout, restart=True)
    elif name == "attack":
        r = sd("vgpt-assistant", props + env, [PY, load, "attack"], output=aout, wait=True); time.sleep(0.5)
        try: res["attack"] = json.loads(rd(aout).strip().splitlines()[-1])
        except Exception as e: res["attack_error"] = repr(e)[:200] + " " + rd(aout + ".err")[-400:]
    t0 = time.time()
    while time.time() - t0 < seconds + 12 and not rd(out).strip(): time.sleep(1)
    try: res["safety"] = json.loads(rd(out).strip().splitlines()[-1])
    except Exception as e: res["safety_error"] = repr(e)[:120]
    txt = rd(aout).strip().splitlines(); res["assistant_output"] = txt[-1][:200] if txt else ""
    st = sh(["systemctl", "show", "vgpt-assistant.service", "-p", "NRestarts", "-p", "Result", "-p", "ActiveState"]).stdout.split(); res["assistant_unit"] = " ".join(st); stop("vgpt-assistant"); stop("vgpt-safety")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=int, default=45); ap.add_argument("--only"); ap.add_argument("--out", default="results/isolation/pi.json"); a = ap.parse_args()
    rt, db = prepare(); names = ["baseline", "idle", "ask", "cpu", "cpu-shared", "mem", "fork", "io", "crash", "attack"]
    if a.only: names = [n for n in names if n in a.only.split(",")]
    results = []
    for n in names:
        r = run_scenario(n, rt, db, a.seconds); results.append(r); s = r.get("safety") or {}
        print("%-11s safety loop: cycles %s lateness p99 %.0f us, p99.99 %.0f us, max %.0f us, deadline misses %s | assistant: %s" % (n, s.get("cycles"), s.get("late_p99_us", -1), s.get("late_p9999_us", -1), s.get("late_max_us", -1), s.get("deadline_misses"), r.get("assistant_output", r.get("assistant_unit", ""))[:80]), flush=True)
        if "attack" in r: print("            escape attempts: %d, refused %d, ALLOWED %d %s" % (r["attack"]["attempts"], r["attack"]["refused"], len(r["attack"]["allowed"]), [x["attempt"] for x in r["attack"]["allowed"]]), flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True); json.dump({"host": os.uname().nodename, "time": time.strftime("%Y-%m-%d %H:%M"), "results": results}, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()
