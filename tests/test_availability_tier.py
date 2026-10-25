"""Profile.column_has_data answers from the daily tier but must give exactly the answer the scan gave, for every (table, column) the registry maps, on every world - including columns that are empty."""
import glob
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import signal_registry as NEW, store
spec = importlib.util.spec_from_file_location("_sr_ref", os.path.join(HERE, "tests", "_signal_registry_reference.py")); REF = importlib.util.module_from_spec(spec); REF.__package__ = "vgpt"; spec.loader.exec_module(REF)
DBS = sorted(glob.glob(os.path.join(HERE, "data", "world_*.db")))


@pytest.mark.skipif(not DBS, reason="needs the world databases")
@pytest.mark.parametrize("db", DBS)
def test_availability_identical(db):
    d = tempfile.mkdtemp(); p = os.path.join(d, "w.db"); shutil.copy(db, p); c = store.open_rw(p); store.ensure_daily_schema(c); store.roll_daily(c); c.close()
    ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True); a, b = REF.Profile(ro), NEW.Profile(ro); n = 0; empty = 0
    for concept, v in NEW.STORE.items():
        for tier, ent in v.items():
            ra, rb = a.column_has_data(ent[0], ent[1]), b.column_has_data(ent[0], ent[1]); assert ra == rb, (concept, tier, ent); n += 1; empty += (not ra)
    assert n > 40
