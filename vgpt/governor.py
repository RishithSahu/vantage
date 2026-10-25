"""Resource governor: simulate the case-study target (~2 GB RAM total, ~2 ARM A5x cores) on this x86 laptop with a Windows Job Object.

  * Job-wide COMMITTED-memory limit (default 2048 MB) shared by EVERYTHING in the "car": an OS-ballast process (stands for the vehicle OS,
    middleware and other ECUs' services), the ingest daemon, and the assistant. An allocation beyond the limit FAILS - this is a real
    enforcement, not a bookkeeping estimate.
  * CPU affinity pinned to 2 logical cores on different physical cores.
  * Optional CPU hard-cap (slowdown factor K): the job may use only 1/K of those 2 cores' time. This is a CRUDE stand-in for 'A55 cores are
    slower than a laptop core'; it is NOT an A55 measurement. Always report K together with the numbers.
Child processes inherit the job. KILL_ON_JOB_CLOSE guarantees nothing is left behind."""
import ctypes, os, subprocess, sys
from ctypes import wintypes as W

K32 = ctypes.WinDLL("kernel32", use_last_error=True)
LIM_AFFINITY, LIM_JOB_MEMORY, LIM_KILL_ON_CLOSE = 0x10, 0x200, 0x2000
CPU_ENABLE, CPU_HARD_CAP = 0x1, 0x4


class BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class IOC(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("ro", "wo", "oo", "rb", "wb", "ob")]


class EXT(ctypes.Structure):
    _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class CPURATE(ctypes.Structure):
    _fields_ = [("ControlFlags", W.DWORD), ("CpuRate", W.DWORD)]


K32.CreateJobObjectW.restype = ctypes.c_void_p
K32.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
K32.GetCurrentProcess.restype = ctypes.c_void_p
K32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, W.DWORD]
K32.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD)]
K32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_job = None


def apply(mem_mb=2048, cores=(0, 2), slowdown=1.0):
    """Put the CURRENT process (and everything it spawns later) under the simulated-target limits."""
    global _job
    _job = K32.CreateJobObjectW(None, None)
    e = EXT(); e.Basic.LimitFlags = LIM_JOB_MEMORY | LIM_AFFINITY | LIM_KILL_ON_CLOSE
    e.JobMemoryLimit = mem_mb * 1024 * 1024; e.Basic.Affinity = sum(1 << c for c in cores)
    if not K32.SetInformationJobObject(_job, 9, ctypes.byref(e), ctypes.sizeof(e)): raise OSError(ctypes.get_last_error(), "SetInformationJobObject(limits)")
    if slowdown > 1.0:
        total = os.cpu_count() or 12
        pct = len(cores) * 100.0 / total / slowdown                       # share of the WHOLE machine, in percent
        c = CPURATE(CPU_ENABLE | CPU_HARD_CAP, max(1, int(pct * 100)))
        if not K32.SetInformationJobObject(_job, 15, ctypes.byref(c), ctypes.sizeof(c)): raise OSError(ctypes.get_last_error(), "SetInformationJobObject(cpu)")
    if not K32.AssignProcessToJobObject(_job, K32.GetCurrentProcess()): raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
    return {"mem_mb": mem_mb, "cores": list(cores), "slowdown": slowdown}


def peak_job_mb():
    e = EXT(); n = W.DWORD()
    if not K32.QueryInformationJobObject(_job, 9, ctypes.byref(e), ctypes.sizeof(e), ctypes.byref(n)): return None
    return e.PeakJobMemoryUsed / 1048576.0


BALLAST = ("import sys,time\nn=int(sys.argv[1])*1048576\nb=bytearray(n)\nfor i in range(0,n,4096): b[i]=1\nprint('ballast',n//1048576,'MB committed',flush=True)\nsys.stdin.readline()\n")


def start_os_ballast(mb):
    """A process that commits `mb` MB and touches every page: stands for the vehicle OS + services sharing the same 2 GB."""
    p = subprocess.Popen([sys.executable, "-c", BALLAST, str(mb)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    if "committed" not in line: raise MemoryError("OS ballast could not be committed inside the job limit")
    return p


def working_sets_mb():
    """Resident working set (MB) of every process in this process tree, by name (psutil)."""
    import psutil
    me = psutil.Process(); out = {me.pid: ("assistant", me.memory_info().rss / 1048576)}
    for ch in me.children(recursive=True):
        try: out[ch.pid] = (" ".join(ch.cmdline()[-2:])[:30], ch.memory_info().rss / 1048576)
        except psutil.Error: pass
    return out
