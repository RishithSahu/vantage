"""What runs INSIDE the assistant's sandbox during the freedom-from-interference tests (tools_isolation_rig.py): ordinary work, hostile work, and attempts to get out.

    python bench/isolation_load.py attack                   try every way out of the sandbox the hardened unit is meant to close; print one JSON line (every attempt must have been refused)
    python bench/isolation_load.py ask   [seconds]          the real assistant answering questions as fast as it can (all the cores it is allowed)
    python bench/isolation_load.py cpu   [seconds]          every allowed core at 100 %
    python bench/isolation_load.py mem                      allocate memory until the kernel ends the process
    python bench/isolation_load.py fork                     fork as fast as possible until the task limit stops it
    python bench/isolation_load.py io    [seconds]          write and fsync as fast as possible in the scratch folder
    python bench/isolation_load.py crash [seconds]          end itself with SIGKILL after a short random time (the service manager restarts it)

Nothing here is part of the product."""
import ctypes
import errno
import json
import multiprocessing as mp
import os
import random
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("VGPT_SCRATCH", "/var/lib/vgpt-scratch")


def attempt(name, fn):
    """Run fn; the sandbox did its job when it raises (or returns a refusal).  -> (name, 'refused: <errno name>' | 'ALLOWED: <what happened>')."""
    try:
        r = fn()
    except OSError as e: return {"attempt": name, "result": "refused: " + errno.errorcode.get(e.errno, str(e.errno)), "blocked": True}
    except Exception as e: return {"attempt": name, "result": "refused: " + type(e).__name__ + " " + str(e)[:60], "blocked": True}
    return {"attempt": name, "result": "ALLOWED: " + str(r)[:80], "blocked": False}


def attack():
    libc = ctypes.CDLL(None, use_errno=True); out = []
    def sock(fam, typ=socket.SOCK_STREAM, proto=0):
        def f(): s = socket.socket(fam, typ, proto); s.close(); return "created"
        return f
    out.append(attempt("open an IPv4 socket", sock(socket.AF_INET)))
    out.append(attempt("open an IPv6 socket", sock(socket.AF_INET6)))
    out.append(attempt("open a raw packet socket", sock(getattr(socket, "AF_PACKET", 17), socket.SOCK_RAW, 0x0003)))
    out.append(attempt("open a CAN socket (a way onto a vehicle bus)", sock(getattr(socket, "AF_CAN", 29), socket.SOCK_RAW, getattr(socket, "CAN_RAW", 1))))
    out.append(attempt("open a netlink socket", sock(getattr(socket, "AF_NETLINK", 16), socket.SOCK_RAW, 0)))
    def unix_connect():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(1); s.connect("/run/vgpt-collector.sock"); return "connected"
    out.append(attempt("connect to the collector's local socket", unix_connect))
    def write_db():
        fd = os.open(os.path.join(ROOT, "data", "world_aster.db"), os.O_WRONLY); os.write(fd, b"x"); return "wrote"      # the vehicle's own history file, not the assistant's working copy
    out.append(attempt("write into the vehicle history database", write_db))
    def write_file(path):
        def f(): fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600); os.write(fd, b"x"); os.close(fd); return "wrote " + path
        return f
    out.append(attempt("create a file in /etc", write_file("/etc/vgpt-attack")))
    out.append(attempt("create a file in the home folder", write_file(os.path.expanduser("~/vgpt-attack"))))
    out.append(attempt("create a file next to the program", write_file(os.path.join(ROOT, "vgpt-attack"))))
    def other_process():
        pids = [p for p in os.listdir("/proc") if p.isdigit() and int(p) not in (os.getpid(), 1)]; return f"{len(pids)} other processes visible"
    r = attempt("see other processes (the safety function's, systemd)", other_process); r["blocked"] = r["result"].startswith("ALLOWED: 0 "); out.append(r)
    def kill_safety():
        pid = int(os.environ["VGPT_SAFETY_PID"]); os.kill(pid, 0); return f"process {pid} (the safety function) is reachable"
    if os.environ.get("VGPT_SAFETY_PID"): out.append(attempt("signal the safety function's process", kill_safety))
    def set_fifo():
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50)); return "real-time priority granted"
    out.append(attempt("take a real-time priority", set_fifo))
    def widen_affinity():
        os.sched_setaffinity(0, set(range(os.cpu_count()))); eff = os.sched_getaffinity(0); return f"cpus now {sorted(eff)}"
    allowed = sorted(os.sched_getaffinity(0)); r = attempt("move onto the cores reserved for the vehicle", widen_affinity)
    r["blocked"] = (set(os.sched_getaffinity(0)) <= set(allowed)); r["result"] = f"allowed cpus stay {sorted(os.sched_getaffinity(0))}"; out.append(r)
    def relax_cgroup():
        p = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split("::")[-1] + "/cpu.max"; open(p, "w").write("max 100000"); return "limit raised"
    out.append(attempt("raise its own CPU limit", relax_cgroup))
    def raw(call, *args):
        def f():
            r = libc.syscall(call, *args)
            if r == -1: e = ctypes.get_errno(); raise OSError(e, os.strerror(e))
            return r
        return f
    arch = os.uname().machine; SYS = {"aarch64": {"mount": 40, "reboot": 142, "init_module": 105, "settimeofday": 170, "ptrace": 117, "unshare": 97}, "x86_64": {"mount": 165, "reboot": 169, "init_module": 175, "settimeofday": 164, "ptrace": 101, "unshare": 272}}.get(arch, {})
    if SYS:
        out.append(attempt("mount a filesystem", raw(SYS["mount"], b"none", b"/mnt", b"tmpfs", 0, 0)))
        out.append(attempt("reboot the machine", raw(SYS["reboot"], 0xfee1dead, 672274793, 0x1234567, 0)))
        out.append(attempt("load a kernel module", raw(SYS["init_module"], 0, 0, 0)))
        out.append(attempt("set the system clock", raw(SYS["settimeofday"], 0, 0)))
        out.append(attempt("create a new namespace", raw(SYS["unshare"], 0x10000000)))
    def child_net():
        r = subprocess.run([sys.executable, "-c", "import socket; socket.socket(socket.AF_INET)"], capture_output=True, timeout=20)
        if r.returncode == 0: return "a child process opened a network socket"
        raise OSError(errno.EACCES, "child refused")
    out.append(attempt("start a child process and let IT open a network socket", child_net))
    def drop_in_path():
        os.makedirs(SCRATCH, exist_ok=True); fd = os.open(os.path.join(SCRATCH, "ok"), os.O_WRONLY | os.O_CREAT, 0o600); os.write(fd, b"x"); os.close(fd); return "wrote in its own scratch folder"
    ok = attempt("(control) write in its own scratch folder", drop_in_path); ok["blocked"] = not ok["blocked"]; ok["result"] = "control: " + ok["result"]; out.append(ok)
    refused = [o for o in out if o["blocked"]]; print(json.dumps({"attempts": len(out), "refused": len(refused), "allowed": [o for o in out if not o["blocked"]], "detail": out}))


def work_ask(seconds):
    sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import make_qwen05_brain
    sys.path.insert(0, os.path.join(ROOT, "bench")); from scale_curve import unique_question
    b = make_qwen05_brain(os.environ.get("VGPT_DB", os.path.join(ROOT, "data", "world_aster.db"))); b.now = 1792348500; n = 0; t0 = time.time()
    while time.time() - t0 < seconds: b.dialogue.reset(); b.ask(unique_question(n)); n += 1
    print(json.dumps({"answers": n, "seconds": round(time.time() - t0, 1)}))


def burn():
    x = 1.0
    while True: x = x * 1.0000001 + 1e-7


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "attack"; seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    if mode == "attack": return attack()
    if mode == "ask": return work_ask(seconds)
    if mode == "cpu":
        n = max(1, len(os.sched_getaffinity(0))) * 2; ps = [mp.Process(target=burn, daemon=True) for _ in range(n)]; [p.start() for p in ps]; time.sleep(seconds); print(json.dumps({"burners": n})); return
    if mode == "mem":
        held = []
        while True: held.append(bytearray(64 * 1024 * 1024)); held[-1][::4096] = b"x" * len(held[-1][::4096]); print(len(held) * 64, "MB", flush=True)
    if mode == "fork":
        n = 0
        try:
            while True:
                if os.fork() == 0: time.sleep(seconds); os._exit(0)
                n += 1
        except OSError as e: print(json.dumps({"forked": n, "stopped_by": errno.errorcode.get(e.errno)})); time.sleep(seconds); return
    if mode == "io":
        os.makedirs(SCRATCH, exist_ok=True); path = os.path.join(SCRATCH, "flood"); buf = os.urandom(8 * 1024 * 1024); t0 = time.time(); n = 0
        while time.time() - t0 < seconds:
            with open(path, "wb") as f: f.write(buf); f.flush(); os.fsync(f.fileno())
            n += 1
        os.remove(path); print(json.dumps({"writes_mb": n * 8})); return
    if mode == "crash":
        time.sleep(random.uniform(0.5, 2.0)); os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__": main()
