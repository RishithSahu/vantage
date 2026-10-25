"""The one-scan History.daily must return exactly what the per-series queries returned (tests/_cause_data_reference.py), for every mapped quantity, every aggregate, masked and unmasked, on every world."""
import glob
import importlib.util
import os
import sqlite3
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import signal_registry as R
from vgpt.cause import data as NEW
spec = importlib.util.spec_from_file_location("_cause_data_reference", os.path.join(HERE, "tests", "_cause_data_reference.py")); REF = importlib.util.module_from_spec(spec)
sys.modules["_cause_data_reference"] = REF
import types
pkg = types.ModuleType("vgpt.cause"); 
REF.__package__ = "vgpt.cause"; spec.loader.exec_module(REF)
DBS = sorted(glob.glob(os.path.join(HERE, "data", "world_*.db")))[:3]


@pytest.mark.skipif(not DBS, reason="needs the world databases")
@pytest.mark.parametrize("db", DBS)
@pytest.mark.parametrize("horizon", [70, 30])
def test_daily_identical(db, horizon):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    a, b = REF.History(c, horizon_days=horizon), NEW.History(c, horizon_days=horizon); n = 0
    for concept, v in R.STORE.items():
        if not v.get("1m"): continue
        for agg in ("AVG", "MAX", "MIN"):
            for drive in (None, True, False):
                x, y = a.daily(concept, agg, drive), b.daily(concept, agg, drive)
                assert np.array_equal(x, y, equal_nan=True), (concept, agg, drive)
                n += 1
    assert n > 100
