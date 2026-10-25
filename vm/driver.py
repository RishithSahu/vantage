"""Host-side harness (VANTAGE): runs the complete-answer benchmark inside an ARM64 QEMU guest exactly as the case study specifies
(~2 GB RAM, 2 ARM A5x cores, ARM virtualised environment = QEMU).  Usage:  python vm/driver.py <stage> [stage ...]
stages: boot | provision | bench | all
It boots Debian 12 arm64 (-M virt -cpu cortex-a55 -smp 2 -m 2048), talks to the serial console, installs Python + the ARM64 llama-cpp-python wheel, fetches the
project over a localhost-only HTTP server (guest -> 10.0.2.2), runs bench/run_guest.py inside the guest and stores the JSON in results/guest_*.json.
EMULATION NOTICE: QEMU TCG is software emulation on x86; guest timings are NOT real Cortex-A55 timings (typically far slower)."""
import json, os, re, socket, subprocess, sys, tarfile, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); IMG = os.environ.get("VM_IMG_DIR", os.path.join(os.path.dirname(ROOT), "vehiclegpt", "vm"))      # guest disk images are shared with the earlier project
QEMU = os.environ.get("QEMU_DIR", os.path.join(os.path.expanduser("~"), "qemu"))
PORT_SERIAL, PORT_HTTP = 4555, 8000
CPU = os.environ.get("VM_CPU", "cortex-a55"); MEM = int(os.environ.get("VM_MEM", "2048")); SMP = int(os.environ.get("VM_SMP", "2"))
LOG = open(os.path.join(HERE, "driver.log"), "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")


def stage_project():
    os.makedirs(os.path.join(HERE, "stage"), exist_ok=True); out = os.path.join(HERE, "stage", "proj2.tgz")
    with tarfile.open(out, "w:gz") as t:
        for rel in ["bench/run_guest_v2.py", "bench/indep_io.py", "bench/indep_dev.tsv", "data/gen_courier_ext.db"]: t.add(os.path.join(ROOT, rel), arcname="vantage/" + rel)
        for dirpath, dirs, files in os.walk(os.path.join(ROOT, "vgpt")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".py", ".json", ".dbc", ".txt")): t.add(os.path.join(dirpath, f), arcname="vantage/" + os.path.relpath(os.path.join(dirpath, f), ROOT).replace(os.sep, "/"))
    log("staged", out, os.path.getsize(out) // 1024, "KB")


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass


def start_http():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT_HTTP), partial(Quiet, directory=ROOT)); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


def start_qemu():
    for f, size in (("work.qcow2", None), ("data.img", None)):
        pass
    img = os.path.join(IMG, "work.qcow2")
    if not os.path.exists(img): subprocess.run([os.path.join(QEMU, "qemu-img.exe"), "create", "-f", "qcow2", "-b", os.path.join(IMG, "debian-12-nocloud-arm64.qcow2"), "-F", "qcow2", img], check=True, capture_output=True)
    data = os.path.join(IMG, "data.img")
    if not os.path.exists(data): subprocess.run([os.path.join(QEMU, "qemu-img.exe"), "create", "-f", "raw", data, "4G"], check=True, capture_output=True)
    cmd = [os.path.join(QEMU, "qemu-system-aarch64.exe"), "-M", "virt", "-cpu", CPU, "-smp", str(SMP), "-m", str(MEM), "-accel", "tcg,thread=multi",
           "-bios", os.path.join(QEMU, "share", "edk2-aarch64-code.fd"),
           "-drive", f"file={img},if=none,id=hd0,format=qcow2", "-device", "virtio-blk-pci,drive=hd0",
           "-drive", f"file={data},if=none,id=hd1,format=raw", "-device", "virtio-blk-pci,drive=hd1",
           "-netdev", "user,id=n0", "-device", "virtio-net-pci,netdev=n0", "-display", "none", "-monitor", "none",
           "-serial", f"tcp:127.0.0.1:{PORT_SERIAL},server=on,wait=off"]
    log("QEMU:", " ".join(cmd[1:12]), "...")
    return subprocess.Popen(cmd, stdout=open(os.path.join(HERE, "qemu.out"), "w"), stderr=subprocess.STDOUT)


class Console:
    def __init__(self):
        for _ in range(60):
            try: self.s = socket.create_connection(("127.0.0.1", PORT_SERIAL), timeout=2); break
            except OSError: time.sleep(1)
        else: raise RuntimeError("cannot reach serial console")
        self.s.settimeout(1.0); self.buf = ""

    def _read(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                d = self.s.recv(65536)
                if d: self.buf += d.decode("utf-8", "replace"); return True
            except socket.timeout: pass
        return False

    def wait_for(self, pattern, timeout, echo_every=30):
        t0 = time.time(); last = t0; rx = re.compile(pattern)
        while time.time() - t0 < timeout:
            m = rx.search(self.buf)
            if m: out = self.buf[:m.end()]; self.buf = self.buf[m.end():]; return out, m
            self._read(1.0)
            if time.time() - last > echo_every: log(f"  ... waiting ({int(time.time() - t0)} s) for {pattern!r}; tail: {self.buf[-120:]!r}"); last = time.time()
        raise TimeoutError(f"timeout waiting for {pattern!r}; tail: {self.buf[-300:]!r}")

    def send(self, line): self.s.sendall((line + "\r").encode())

    def run(self, cmd, timeout=600):
        t = time.time(); self.send(f'{cmd}; echo "__RC_$?__"')
        out, m = self.wait_for(r"__RC_(\d+)__", timeout); rc = int(m.group(1)); dt = time.time() - t
        out = re.sub(r"__RC_\d+__", "", out); out = out.split("\n", 1)[1] if "\n" in out else out
        log(f"$ {cmd[:110]}  -> rc={rc} in {dt:.0f}s"); return rc, out.strip(), dt


def boot():
    stage_project(); start_http(); q = start_qemu(); c = Console(); t0 = time.time()
    log("waiting for the guest to boot (software emulation: be patient) ...")
    c.wait_for(r"login: ", 900, echo_every=60); log(f"login prompt after {time.time() - t0:.0f} s"); c.send("root"); time.sleep(3)
    c.send("stty -echo; export PS1='# ' TERM=dumb LANG=C"); time.sleep(2); c.buf = ""
    return q, c


def refresh(c, model="LFM2.5-350M-Q4_K_M.gguf"):
    """Re-use the provisioned /data disk of the earlier project (venv with the ARM llama-cpp wheel stays): mount it, fetch this project and the model."""
    c.run("mkdir -p /data && (mountpoint -q /data || mount /dev/vdb /data) && ls /data | head", 120)
    c.run("mkdir -p /data/app && cd /data/app && rm -rf vantage && curl -s http://10.0.2.2:8000/vm/stage/proj2.tgz | tar xz && mkdir -p vantage/models vantage/cache && ls vantage", 900)
    c.run(f"cd /data/app/vantage && [ -f models/{model} ] || curl -s -o models/{model} http://10.0.2.2:8000/models/{model}; ls -la models", 1800)
    rc, out, _ = c.run("/data/venv/bin/python -c 'import llama_cpp, numpy; print(llama_cpp.__version__, numpy.__version__)'", 120); log("guest llama-cpp-python:", out[-80:])
    return out


def bench(c, label, args="", timeout=7200):
    c.run("sync; echo 3 > /proc/sys/vm/drop_caches; grep -E 'MemTotal|MemAvailable' /proc/meminfo", 60)
    rc, out, dt = c.run(f"cd /data/app/vantage && /data/venv/bin/python bench/run_guest_v2.py --label {label} {args}", timeout)
    m = re.search(r"@@RESULT_BEGIN@@\s*(\{.*\})\s*@@RESULT_END@@", out, re.S)
    if not m: log("NO RESULT; tail:", out[-2500:]); return None
    res = json.loads(m.group(1)); res["wall_s"] = round(dt); res["note"] = "QEMU TCG software emulation of cortex-a55 x2, 2048 MB; timings are emulation timings"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); json.dump(res, open(os.path.join(ROOT, "results", f"guest2_{label}.json"), "w"), indent=1); log("saved results/guest2_%s.json" % label); return res


def boot():
    stage_project(); start_http(); q = start_qemu(); c = Console(); t0 = time.time()
    log("waiting for the guest to boot (software emulation: be patient) ...")
    c.wait_for(r"login: ", 900, echo_every=60); log(f"login prompt after {time.time() - t0:.0f} s"); c.send("root"); time.sleep(3)
    c.send("stty -echo; export PS1='# ' TERM=dumb LANG=C"); time.sleep(2); c.buf = ""
    return q, c


if __name__ == "__main__":
    stages = sys.argv[1:] or ["smoke"]; q, c = boot()
    try:
        c.run("uname -m; nproc; grep -m1 'CPU part' /proc/cpuinfo; grep -E 'MemTotal|MemAvailable' /proc/meminfo; ip -br addr | head -3", 60)
        c.run("ping -c1 -W3 10.0.2.2 >/dev/null 2>&1 || (for i in $(ls /sys/class/net | grep -v lo); do ip link set $i up; dhclient -1 $i 2>/dev/null; done); ip -br addr | grep -v lo", 180)
        v = refresh(c)
        if "smoke" in stages:
            bench(c, "smoke_firstboot", "--questions 8")                    # builds and persists the prompt state on the target (one-time cost)
            bench(c, "smoke_secondboot", "--questions 8")                   # loads the persisted state
        if "full" in stages:
            bench(c, "full_static", "--questions 40"); bench(c, "full_bank_draft", "--questions 40 --bank 4 --draft")
    finally:
        try: c.run("poweroff", 30)
        except Exception: pass
        time.sleep(5); q.kill(); log("done")
