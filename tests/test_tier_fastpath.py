"""Plain statistics over a LONG window come from the daily tier and must agree with scanning every minute: min / max / count / the minute of an extreme exactly, the mean to 1e-9 (a sum of products in a
different order); a window below the threshold, a condition, a trip selection or a derived quantity never takes the fast path."""
import os
import shutil
import sqlite3
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import store, signal_registry as R
from vgpt.query_ir import Query
from vgpt import qc as QC

DB = os.path.join(HERE, "data", "world_3yr.db")


@pytest.fixture(scope="module")
def paths():
    if not os.path.exists(DB): pytest.skip("needs data/world_3yr.db")
    d = tempfile.mkdtemp(); p = os.path.join(d, "w.db"); shutil.copy(DB, p); c = store.open_rw(p); store.ensure_daily_schema(c); store.roll_daily(c); c.close(); return p


def run(path, plan, fast):
    os.environ["VGPT_TIER_FAST"] = "1" if fast else "0"
    try:
        from vgpt.query_executor import Executor, context_for
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True); p = R.Profile(c); ex = Executor(c, p, context_for(c)); r = ex.run(Query.from_text(plan)); c.close(); return r
    finally: os.environ.pop("VGPT_TIER_FAST", None)


@pytest.mark.parametrize("sig", ["speed", "battery_temperature", "motor_temperature", "coolant_flow", "battery_soc", "ambient_temperature"])
@pytest.mark.parametrize("window", ["t=last:365:day", "t=last:900:day", "t=last:200:day"])
def test_aggregates_agree(paths, sig, window):
    for agg in ("mean", "min", "max", "count"):
        a = run(paths, f"op=aggregate tgt={sig} agg={agg} {window}", True); b = run(paths, f"op=aggregate tgt={sig} agg={agg} {window}", False)
        assert a.status == b.status
        if a.status != "ok": continue
        assert a.tier == "daily tier" and b.tier != "daily tier"
        assert a.samples == b.samples or agg != "count"
        assert abs(a.value - b.value) <= 1e-9 * max(1.0, abs(b.value)), (sig, window, agg, a.value, b.value)


@pytest.mark.parametrize("sig", ["speed", "battery_temperature", "ambient_temperature"])
@pytest.mark.parametrize("agg", ["max", "min"])
def test_extremes_agree_including_when(paths, sig, agg):
    a = run(paths, f"op=extreme tgt={sig} agg={agg} t=last:700:day", True); b = run(paths, f"op=extreme tgt={sig} agg={agg} t=last:700:day", False)
    assert a.status == b.status == "ok" and a.value == b.value and a.when == b.when and a.extra.get("state") == b.extra.get("state") and a.samples == b.samples


def test_short_windows_conditions_and_selection_never_take_the_fast_path(paths):
    for plan in ("op=aggregate tgt=speed agg=mean t=last:30:day", "op=aggregate tgt=speed agg=mean t=last:365:day c=highway", "op=aggregate tgt=speed agg=median t=last:365:day",
                 "op=aggregate tgt=tyre_pressure agg=mean t=last:365:day", "op=extreme tgt=speed agg=max t=last:30:day"):
        assert run(paths, plan, True).tier != "daily tier", plan


@pytest.mark.parametrize("sig", ["speed", "battery_temperature", "ambient_temperature", "coolant_flow"])
def test_trend_agrees(paths, sig):
    a = run(paths, f"op=trend tgt={sig} t=last:600:day", True); b = run(paths, f"op=trend tgt={sig} t=last:600:day", False)
    assert a.status == b.status
    if a.status != "ok": return
    assert a.tier == "daily tier" and abs(a.value - b.value) <= 1e-9 * max(1.0, abs(b.value)) and a.samples == b.samples and len(a.items) == len(b.items)
    assert abs(a.extra["p"] - b.extra["p"]) <= 1e-9 and abs(a.extra["change"] - b.extra["change"]) <= 1e-9
