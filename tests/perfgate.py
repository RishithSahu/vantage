"""Machine-dependent timing assertions are NOT part of the functional suite. They run only when VGPT_PERF=1 (python bench/perf_checks.py sets it and records the machine, the load and the pinned cores).
A passing functional run says nothing about meeting a response-time objective; the measured latencies are reported by bench/run_guest_v2.py and the performance report."""
import os
PERF = os.environ.get("VGPT_PERF") == "1"
