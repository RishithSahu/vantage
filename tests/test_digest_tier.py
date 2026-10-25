"""Digest answers (summary / anomaly / changes / usual) are identical whether the per-day aggregates come from the daily tier or from scanning the minute rows."""
import glob
import os
import shutil
import sqlite3
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import store

DBS = sorted(glob.glob(os.path.join(HERE, "data", "world_aster.db")))
ORT = os.path.join(HERE, "models", "ort_frame2w")
QS = ["is there anything unusual going on with the car", "driving style summary", "brief me on the month", "what changed since last week", "summarise the last 7 days", "cooling condition summary", "how is my usual commute"]


def variant(db, tier):
    d = tempfile.mkdtemp(); p = os.path.join(d, "w.db"); shutil.copy(db, p); c = sqlite3.connect(p)
    for t in ("agg_1d", "agg_1d_state"): c.execute(f"DROP TABLE IF EXISTS {t}")
    c.commit(); c.close()
    if tier:
        c = store.open_rw(p); store.ensure_daily_schema(c); store.roll_daily(c); c.close()
    return p


@pytest.mark.skipif(not DBS or not os.path.isdir(ORT), reason="needs the world database and the ONNX frame model")
def test_digest_answers_identical():
    os.environ["VGPT_ORT_DIR"] = ORT
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt import product
    answers = {}
    for tier in (False, True):
        p = variant(DBS[0], tier); b = product.make_meaning_brain(p, persona="driver"); out = []
        for q in QS: out.append(b.ask(q).get("answer"))
        answers[tier] = out
    assert answers[False] == answers[True]
