"""Process hardening, done once at start-up before any vehicle data is read.

    harden() -> {"core_dumps": bool, "not_dumpable": bool|None, "umask": "0o77", "mlock": bool|None}

* no core dump (RLIMIT_CORE = 0) and not dumpable (prctl PR_SET_DUMPABLE = 0): a crash, or another process of the same user trying ptrace / reading /proc/<pid>/mem, gets nothing of the database that is in RAM;
* umask 077: every file this process creates is private to its user;
* optionally (VGPT_MLOCK=1) the memory is locked so that the clear database in RAM is never swapped out to the storage.

Linux only for the parts that exist there; elsewhere the call reports what it could not do.  The systemd unit adds the same from outside (LimitCORE=0, MemorySwapMax=0, ProtectProc=invisible)."""
import os
import sys


def harden():
    out = {"core_dumps": None, "not_dumpable": None, "umask": None, "mlock": None, "sigterm_cleans_up": None}
    try:
        os.umask(0o077); out["umask"] = "0o77"
    except OSError: pass
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); out["core_dumps"] = False
    except (ImportError, ValueError, OSError): pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            out["not_dumpable"] = libc.prctl(4, 0, 0, 0, 0) == 0 and libc.prctl(3, 0, 0, 0, 0) == 0        # PR_SET_DUMPABLE = 4, PR_GET_DUMPABLE = 3
            if os.environ.get("VGPT_MLOCK") == "1": out["mlock"] = libc.mlockall(3) == 0               # MCL_CURRENT | MCL_FUTURE
        except Exception: pass
    try:
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143)); out["sigterm_cleans_up"] = True                     # a stop request runs the exit handlers: the working copy is wiped
    except (ValueError, OSError, AttributeError): pass                                                         # not the main thread, or no SIGTERM here
    return out


def status():
    """What /proc says about this process right now (Linux): can it be dumped, is core size zero."""
    out = {}
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith(("VmLck", "TracerPid")): out[ln.split(":")[0]] = ln.split(":")[1].strip()
        out["dumpable"] = open("/proc/self/stat").read() and None
        for ln in open("/proc/self/limits"):
            if ln.startswith("Max core file size"): out["core_limit"] = ln.split()[4:6]
    except OSError: pass
    return out
