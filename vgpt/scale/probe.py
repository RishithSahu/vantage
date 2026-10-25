"""What hardware is this?  Read from the operating system, never assumed: the cores the process may really use (affinity, cgroup quota), the memory it may really use (cgroup limit), the instruction sets
that decide which kernels are fast, and any accelerator a runtime can use.  Every value can be overridden (VGPT_CORES, VGPT_MEM_MB, VGPT_ISA, VGPT_ACCEL) so that a laptop can be asked to behave as a 2-core
board and the planner can be tested against hardware that is not on the desk.

    p = probe()           # -> Hardware(cores=2, mem_mb=1950, isa={'neon', 'dotprod'}, accel=[], arch='aarch64', ...)

No third-party package is needed (psutil is used only if present)."""
import ctypes
import dataclasses
import os
import platform
import shutil
import subprocess
import sys


@dataclasses.dataclass
class Hardware:
    cores: int = 2                       # logical CPUs this process may run on (affinity and cgroup quota applied)
    physical: int = 0                    # physical cores when the OS says (0 = unknown)
    mem_mb: int = 2048                   # memory this process may use (cgroup limit applied)
    mem_total_mb: int = 0
    arch: str = ""
    isa: frozenset = frozenset()         # instruction sets that matter: neon, dotprod, i8mm, sve, avx2, avx512, avx_vnni, avx512_vnni, amx
    accel: tuple = ()                    # runtime providers: cuda, dml, tensorrt, coreml, qnn, openvino ...
    gpu_mb: int = 0
    free_disk_mb: int = 0
    os_name: str = ""
    source: dict = dataclasses.field(default_factory=dict)       # where each value came from (for the report)

    def describe(self):
        return f"{self.cores} cores, {self.mem_mb} MB, {self.arch}, isa={'+'.join(sorted(self.isa)) or '-'}, accel={'+'.join(self.accel) or '-'}"


def _cgroup_cpus():
    """Effective CPUs from a cgroup quota (v2 cpu.max, v1 cfs quota), or None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f: q, p = f.read().split()[:2]
        if q != "max": return max(1, int(round(int(q) / int(p))))
    except (OSError, ValueError): pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f: q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f: p = int(f.read())
        if q > 0: return max(1, int(round(q / p)))
    except (OSError, ValueError): pass
    return None


def _cgroup_mem_mb():
    if os.path.exists("/sys/fs/cgroup"):                                                       # cgroups are a Linux feature
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                v = open(path).read().strip()
                if v != "max" and int(v) < 1 << 50: return int(v) // (1 << 20)
            except (OSError, ValueError): pass
    return None


def _cpus():
    src = "os.cpu_count"; n = os.cpu_count() or 1
    try:
        if hasattr(os, "process_cpu_count") and os.process_cpu_count(): n = os.process_cpu_count(); src = "process affinity"
        elif hasattr(os, "sched_getaffinity"): n = len(os.sched_getaffinity(0)); src = "sched affinity"
    except OSError: pass
    q = _cgroup_cpus()
    if q is not None and q < n: n = q; src = "cgroup quota"
    return n, src


def _physical():
    try:
        if sys.platform.startswith("linux"):
            seen = set(); core = pkg = None
            for line in open("/proc/cpuinfo"):
                if line.startswith("physical id"): pkg = line.split(":")[1].strip()
                elif line.startswith("core id"): seen.add((pkg, line.split(":")[1].strip()))
            if seen: return len(seen)
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"], capture_output=True, text=True, timeout=15).stdout.strip()
            return int(out) if out.isdigit() else 0
    except (OSError, ValueError, subprocess.SubprocessError): pass
    return 0


def _memory():
    total = 0
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"): total = int(line.split()[1]) // 1024; break
        elif sys.platform == "win32":
            class MS(ctypes.Structure): _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); total = int(m.tp) // (1 << 20)
        elif sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) // (1 << 20)
    except (OSError, ValueError): pass
    lim = _cgroup_mem_mb()
    return (min(total, lim) if total and lim else (lim or total or 2048)), total, ("cgroup limit" if lim and (not total or lim < total) else "os")


def _isa():
    flags = set(); m = platform.machine().lower()
    if m in ("aarch64", "arm64"): flags.add("neon")
    try:
        if sys.platform.startswith("linux"):
            text = open("/proc/cpuinfo").read().lower()
            for key, name in (("asimddp", "dotprod"), ("asimdrdm", "rdm"), ("i8mm", "i8mm"), ("sve", "sve"), ("bf16", "bf16"), ("avx2", "avx2"), ("avx512f", "avx512"), ("avx512_vnni", "avx512_vnni"), ("avx_vnni", "avx_vnni"), ("amx_int8", "amx"), ("avx512_vnni", "vnni")):
                if key in text.replace("\n", " ").split() or (" " + key + " ") in text: flags.add(name)
        elif sys.platform == "win32":
            f = ctypes.windll.kernel32.IsProcessorFeaturePresent                                                         # numbers from winnt.h
            for pf, name in ((40, "avx2"), (41, "avx512"), (43, "dotprod")):
                try:
                    if f(pf): flags.add(name)
                except OSError: pass
            if m in ("amd64", "x86_64"): flags.update(_x86_flags_windows())
    except OSError: pass
    return frozenset(flags)


def _x86_flags_windows():
    """AVX-VNNI / AVX512-VNNI through CPUID is not reachable from pure Python; ask the CPU name instead: 12th generation Intel and later have AVX-VNNI, Zen4 and later Intel server parts have AVX512-VNNI."""
    out = set()
    try:
        name = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=15).stdout
        import re
        m = re.search(r"(\d+)(?:th|st|nd|rd) Gen", name)
        if m and int(m.group(1)) >= 12 and "Intel" in name: out.add("avx_vnni")
        if "Ryzen" in name and re.search(r"\b[789]\d{3}", name): out.add("avx512_vnni")
    except (OSError, subprocess.SubprocessError): pass
    return out


def _accelerators():
    acc = []; gpu_mb = 0
    try:
        import onnxruntime as ort
        names = {"CUDAExecutionProvider": "cuda", "DmlExecutionProvider": "dml", "TensorrtExecutionProvider": "tensorrt", "CoreMLExecutionProvider": "coreml", "QNNExecutionProvider": "qnn", "OpenVINOExecutionProvider": "openvino", "ROCMExecutionProvider": "rocm"}
        acc = [names[p] for p in ort.get_available_providers() if p in names]
    except Exception: pass
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            gpu_mb = max(int(x) for x in out if x.strip().isdigit()) if out else 0
        except (OSError, ValueError, subprocess.SubprocessError): pass
    return tuple(acc), gpu_mb


def probe():
    """Read the hardware this process runs on, with the override variables applied."""
    src = {}
    cores, src["cores"] = _cpus(); mem, total, src["mem"] = _memory(); acc, gpu = _accelerators()
    hw = Hardware(cores=cores, physical=_physical(), mem_mb=mem, mem_total_mb=total, arch=platform.machine().lower(), isa=_isa(), accel=acc, gpu_mb=gpu, os_name=sys.platform, source=src)
    try: hw.free_disk_mb = shutil.disk_usage(os.path.expanduser("~")).free // (1 << 20)
    except OSError: pass
    if os.environ.get("VGPT_CORES"): hw.cores = max(1, int(os.environ["VGPT_CORES"])); src["cores"] = "VGPT_CORES"
    if os.environ.get("VGPT_MEM_MB"): hw.mem_mb = max(256, int(os.environ["VGPT_MEM_MB"])); src["mem"] = "VGPT_MEM_MB"
    if os.environ.get("VGPT_ISA") is not None: hw.isa = frozenset(x for x in os.environ["VGPT_ISA"].split(",") if x); src["isa"] = "VGPT_ISA"
    if os.environ.get("VGPT_ACCEL") is not None: hw.accel = tuple(x for x in os.environ["VGPT_ACCEL"].split(",") if x); src["accel"] = "VGPT_ACCEL"
    return hw
