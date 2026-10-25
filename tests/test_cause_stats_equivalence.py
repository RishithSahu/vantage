"""The speed-optimised vgpt/cause/stats.py must give the SAME numbers as the verbatim reference (tests/_cause_stats_reference.py): p-values come from shuffles, so the random stream is reset to the
same seed before each call and every comparison is exact.  This keeps the sealed causal test (results/causal/test.json) valid: the engine's decisions are unchanged, only faster."""
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.cause import stats as NEW

spec = importlib.util.spec_from_file_location("_cause_stats_reference", os.path.join(HERE, "tests", "_cause_stats_reference.py")); REF = importlib.util.module_from_spec(spec); spec.loader.exec_module(REF)


def series(seed):
    r = np.random.default_rng(seed); n = int(r.integers(10, 130)); y = 10 + r.normal(0, 1 + 2 * r.random(), n)
    kind = int(r.integers(0, 6))
    if kind == 1: y[n // 2:] += r.normal(4, 1)
    elif kind == 2: y[n - 12:] += np.linspace(0, 6, 12)
    elif kind == 3: y[n // 3: n // 3 + 3] += 12
    elif kind == 4: y[::7] += 25
    if r.random() < 0.5: y[r.integers(0, n, int(r.integers(1, max(2, n // 6))))] = np.nan
    return y


def same(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float): return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, np.ndarray): return np.array_equal(a, b, equal_nan=True)
    return a == b


def run(fn_name, y, *args, seed=7, **kw):
    out = []
    for mod in (REF, NEW):
        mod._RNG = np.random.default_rng(seed)
        out.append(getattr(mod, fn_name)(y, *args, **kw))
    return out


def test_best_step_identical():
    for s in range(120):
        a, b = run("best_step", series(s))
        assert same(a, b), (s, a, b)


def test_despike_identical():
    for s in range(150):
        a, b = run("despike", series(s)); assert same(a, b), s


def test_excursion_p_identical():
    n_checked = 0
    for s in range(160):
        y = series(s); ok = y[np.isfinite(y)]
        if len(ok) < 8: continue
        before = float(np.median(ok)); sd = float(max(np.std(ok), 1e-3)) * 0.6
        found = REF.excursion(y, before, sd, k=4.0, min_days=2)
        if found is None: continue
        a, b = run("excursion_p", y, before, sd, found); assert a == b, (s, a, b); n_checked += 1
    assert n_checked > 10


def test_mann_kendall_and_end_ramp_identical():
    for s in range(120):
        y = series(s)
        a, b = run("mann_kendall", y); assert same(a, b), s
        a, b = run("end_ramp", y, 0.8); assert same(a, b), s
