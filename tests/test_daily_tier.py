"""The daily tier (agg_1d): same numbers as scanning the minute rows, only the finished days are rolled, it extends incrementally, and a tier that does not belong to the data is ignored."""
import glob
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import signal_registry as R, store
from vgpt.cause import data as NEW
spec = importlib.util.spec_from_file_location("_cause_data_reference", os.path.join(HERE, "tests", "_cause_data_reference.py")); REF = importlib.util.module_from_spec(spec); REF.__package__ = "vgpt.cause"; spec.loader.exec_module(REF)
DBS = sorted(x for x in glob.glob(os.path.join(HERE, "data", "world_*.db")) if "3yr" not in x)       # the three-year world is for the history benchmark


def prepared(db):
    d = tempfile.mkdtemp(); p = os.path.join(d, "w.db"); shutil.copy(db, p)
    c = store.open_rw(p)
    store.ensure_daily_schema(c); c.execute("DELETE FROM agg_1d"); c.execute("DELETE FROM agg_1d_state"); c.commit()         # the shipped worlds are already rolled: start from none
    return p, c


@pytest.mark.skipif(not DBS, reason="needs the world databases")
def test_tier_gives_identical_series_and_skips_the_open_day():
    p, c = prepared(DBS[0]); n = store.roll_daily(c); assert n > 2000
    tz = store.tz_of(c); hi = c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]; last_day = (hi + tz) // 86400
    assert c.execute("SELECT MAX(d) FROM agg_1d").fetchone()[0] == last_day - 1                        # the day still filling is not rolled
    c.close(); ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    from vgpt.vehicle import lake
    lake.adopt_path(p)                                                                                    # the vehicle's own lake signals are mapped too
    for horizon in (70, 30, 200):
        a, b = REF.History(ro, horizon_days=horizon), NEW.History(ro, horizon_days=horizon)
        for concept, v in R.STORE.items():
            if not v.get("1m"): continue
            for agg in ("AVG", "MAX", "MIN"):
                for drive in (None, True, False): assert np.array_equal(a.daily(concept, agg, drive), b.daily(concept, agg, drive), equal_nan=True), (horizon, concept, agg, drive)


@pytest.mark.skipif(not DBS, reason="needs the world databases")
def test_rolling_is_incremental_and_idempotent():
    p, c = prepared(DBS[0]); first = store.roll_daily(c); again = store.roll_daily(c)
    assert first > 0 and again == 0


@pytest.mark.skipif(not DBS, reason="needs the world databases")
def test_a_tier_that_does_not_belong_to_the_data_is_ignored():
    p, c = prepared(DBS[0]); store.roll_daily(c)
    c.execute("UPDATE agg_1d_state SET sig_max_ts = sig_max_ts + 10 * 86400"); c.close()                  # claims data that this database does not have
    ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True); a, b = REF.History(ro, 70), NEW.History(ro, 70)
    assert np.array_equal(a.daily("speed", "AVG"), b.daily("speed", "AVG"), equal_nan=True)


@pytest.mark.skipif(not DBS, reason="needs the world databases")
def test_cutting_the_history_cuts_the_tier_too():
    from vgpt.live import replay as RP
    p, c = prepared(DBS[0]); store.roll_daily(c); tz = store.tz_of(c)
    hi = c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]; t = hi - 20 * 86400
    RP.cut_after(c, t)
    assert c.execute("SELECT MAX(d) FROM agg_1d").fetchone()[0] < (t + tz) // 86400                     # no day from the cut on survives in the tier
    c.close(); ro = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    a, b = REF.History(ro, 70), NEW.History(ro, 70)
    for concept in ("speed", "battery_temperature", "tyre_pressure_fl"): assert np.array_equal(a.daily(concept, "AVG"), b.daily(concept, "AVG"), equal_nan=True)
