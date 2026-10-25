"""Pin this process to explicit logical cores, then run a module or script:   python tools_pin.py 8,9 -m training.evaluate --backend rules   |   python tools_pin.py 8,9 script.py args
The target has two cores; a development machine has more, and several experiments may run side by side, so each is pinned to its own pair."""
import os, runpy, sys
import psutil
cores = [int(x) for x in sys.argv[1].split(",")]; psutil.Process().cpu_affinity(cores)
os.environ.setdefault("OMP_NUM_THREADS", str(len(cores))); os.environ.setdefault("OPENBLAS_NUM_THREADS", str(len(cores))); os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rest = sys.argv[2:]
if rest[0] == "-m": sys.argv = [rest[1]] + rest[2:]; runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
else: sys.argv = rest; runpy.run_path(rest[0], run_name="__main__")
