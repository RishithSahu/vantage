import math
import os
import tempfile

import pytest

from vgpt import adapters, store
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate


class Echo:
    def decode(self, _bus, payload):
        return payload


def db_ingestor():
    path = os.path.join(tempfile.mkdtemp(), "audit.db")
    conn = store.open_rw(path)
    return conn, store.Ingestor(conn, Echo())


def test_compact_ir_rejects_duplicates_and_preserves_branch_b_conditions():
    with pytest.raises(IRError):
        Query.from_text("op=exists tgt=speed t=all p.thr=gt:10 p.thr=lt:20")
    q = Query.from_text("op=compare tgt=speed agg=mean t=last:1:day t2=prev:1:day c=city c2=highway")
    assert Query.from_text(q.to_text()).to_dict() == q.to_dict()


def test_typed_plan_contract_rejects_invalid_combinations_and_values():
    assert "BAD_AGG" in validate(Query.from_text("op=threshold_test tgt=speed agg=std t=all p.thr=gt:10")).codes()
    assert "BAD_PARAM" in validate(Query.from_text("op=percentile tgt=speed t=all p.pct=101")).codes()
    assert "BAD_COMPARE" in validate(Query.from_text("op=percentile_vs tgt=speed t=all grp=night,city p.pct=50")).codes()
    assert "BAD_PARAM" in validate(Query.from_text("op=session_attr t=all p.attr=distance")).codes()


def test_vhal_status_and_names_are_canonical():
    a = adapters.VhalAdapter()
    msg = a.v.encode("PERF_VEHICLE_SPEED", 10.0)
    assert a.decode(msg) == [("sig", "speed", 36.0)]
    msg["status"] = 1
    assert a.decode(msg) == []


def test_ingest_drops_late_and_nonfinite_but_keeps_other_signals_without_soc():
    c, ing = db_ingestor()
    ing.feed(100, "x", [("sig", "speed", 20.0), ("sig", "ign", 1.0), ("sig", "batt_temp", 25.0)])
    ing.feed(101, "x", [("sig", "soc", float("nan"))])
    ing.feed(100, "x", [("sig", "speed", 99.0)])
    ing.close()
    row = c.execute("SELECT speed,soc,batt FROM sig_1s WHERE ts=100").fetchone()
    assert row == (200, None, 250)
    assert ing.dropped_invalid == 1 and ing.dropped_late == 1


def test_stale_fast_signal_is_not_relabelled_as_current():
    c, ing = db_ingestor()
    ing.feed(100, "x", [("sig", "speed", 10.0), ("sig", "ign", 1.0)])
    ing.feed(190, "x", [("sig", "soc", 50.0)])
    ing.close()
    assert c.execute("SELECT speed FROM sig_1s WHERE ts=190").fetchone()[0] is None


def test_camera_missing_fields_are_null_and_finalized_minutes_ignore_late_frames():
    c, ing = db_ingestor()
    ing.feed(100, "x", [("sig", "cam_block", 0.3)])
    ing.feed(160, "x", [("sig", "cam_block", 0.1)])
    ing._cam(100, "cam_block", 0.9)
    ing.close()
    row = c.execute("SELECT frames,block_avg,sharp_avg,luma_avg,lane_avg FROM cam_1m WHERE ts=60").fetchone()
    assert row[0] == 1 and row[1] == pytest.approx(0.3)
    assert row[2:] == (None, None, None)


def test_invalid_gnss_fix_cannot_republish_old_coordinates():
    c, ing = db_ingestor()
    ing.feed(100, "x", [("sig", "lat", 12.0), ("sig", "lon", 77.0), ("sig", "gps_valid", 1.0)])
    ing.feed(101, "x", [("sig", "gps_valid", 0.0), ("sig", "hdop", 1.1)])
    ing.close()
    assert c.execute("SELECT ts FROM gps_1s ORDER BY ts").fetchall() == [(100,)]


def test_compaction_uses_represented_time_and_dominant_categorical_state():
    path = os.path.join(tempfile.mkdtemp(), "compact.db"); c = store.open_rw(path)
    now = 2_000_000_000; minute = ((now - store.RET_1S_DAYS * 86400 - 120) // 60) * 60
    # 10 km/h represents 50 seconds; 50 km/h represents 10 seconds.  Parked
    # represents 59 seconds and charging one second.
    c.execute("INSERT INTO sig_1s(ts,speed,mode) VALUES(?,?,?),(?,?,?),(?,?,?)", (minute, 100, 0, minute + 50, 500, 0, minute + 59, 500, 3))
    store.compact(c, now)
    speed, mode = c.execute("SELECT speed,mode FROM sig_1m WHERE ts=?", (minute // 60 * 60,)).fetchone()
    assert speed == pytest.approx((100 * 50 + 500 * 10) / 60, abs=1)
    assert mode == 0

