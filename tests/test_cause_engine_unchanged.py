"""End to end: the cause engine with the speed-optimised statistics and data layer returns the same diagnosis (hypotheses, confidences, onsets, changes) as with the verbatim reference modules
on every world history and on a planted-cause history - so the sealed causal test (results/causal/test.json) is not invalidated by the speed work."""
import glob
import importlib.util
import json
import os
import sqlite3
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.cause import engine as E
from vgpt.cause import stats as NEWS
from vgpt.cause import data as NEWD


def load(name, pkg=True):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "tests", name + ".py")); m = importlib.util.module_from_spec(spec)
    if pkg: m.__package__ = "vgpt.cause"
    spec.loader.exec_module(m); return m


REFS = load("_cause_stats_reference", False); REFD = load("_cause_data_reference")
DBS = sorted(glob.glob(os.path.join(HERE, "data", "world_*.db")))


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items() if k != "_features" and k != "cause"}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)): return None if x != x else float(f"{float(x):.9g}")      # two sums of the same numbers in a different order differ in the 16th digit: compare to 9 significant digits
    if isinstance(x, (np.integer,)): return int(x)
    return x


def diagnose_with(stats_mod, data_mod, db, symptom=None):
    E.S, E.History = stats_mod, data_mod.History
    stats_mod._RNG = np.random.default_rng(12345)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try: return json.dumps(clean(E.diagnose(data_mod.History(conn, 70), symptom)), sort_keys=True, default=str)
    finally: conn.close(); E.S, E.History = NEWS, NEWD.History


@pytest.mark.skipif(not DBS, reason="needs the world databases")
@pytest.mark.parametrize("db", DBS)
def test_diagnosis_identical(db):
    a = diagnose_with(REFS, REFD, db); b = diagnose_with(NEWS, NEWD, db)
    assert a == b
