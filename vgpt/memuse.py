"""How much memory this process uses, on a system that has no /proc/self/status (QNX keeps its process data elsewhere) and no psutil.

The 2 GB budget of the case study is enforced by the operating system where it can (cgroups on Linux and Android, memory partitions on QNX) and by the process itself as a second line (`vehiclegpt.py _memory_guard`).  The
process needs a number to do that.  Where the live resident size cannot be read, the C library's accounting of the PEAK resident size is used instead: it never goes down, so it is the stricter reading (the guard acts when the
budget was crossed at any time, which is what a budget means), and where even that is unavailable the answer is 0.0, which the callers treat as "unknown", not as "small"."""
import sys


def peak_rss_mb():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError, ValueError, OSError): return 0.0
    return peak / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)                       # macOS reports bytes, Linux, Android and QNX kibibytes
