"""Run the test suite under the target's limits: pinned to 2 CPU cores, with the peak memory of the process reported against the 2 GB budget.

    python tools_run_tests.py                 whole suite
    python tools_run_tests.py tests.test_qc   selected modules

Only model TRAINING (training/train_lora.py) is allowed to use the whole development machine; tests, evaluation and benchmarks run like the vehicle: 2 cores, 2 inference threads."""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def pin(cores=2):
    try:
        import psutil
        psutil.Process().cpu_affinity(list(range(psutil.cpu_count()))[:cores]); return True
    except Exception as e:
        print("could not pin CPU cores:", e); return False


def main():
    pinned = pin(2)
    names = sys.argv[1:]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names) if names else unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if res.skipped:                                                                                 # every skip is listed with its reason: a skipped test is not a passed test
        print(f"\nSKIPPED ({len(res.skipped)}):")
        for t, why in res.skipped: print(f"  {t.id()}: {why}")
    try:
        import psutil
        mi = psutil.Process().memory_info(); peak = getattr(mi, "peak_wset", mi.rss) / 2 ** 20
        print(f"\npinned to 2 cores: {pinned}; peak memory of the test process: {peak:.0f} MB (budget 2048 MB for the whole system)")
    except Exception: pass
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__": main()
