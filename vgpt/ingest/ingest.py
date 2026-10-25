"""Capture file -> the application's own database, with nothing configured:

    python vehiclegpt.py ingest capture.csv --db data/mycar.db [--dbc body.dbc] [--pack out_dir]

`ingest(path, db)` reads the capture, writes the profile pack it worked out (autoprofile), provisions the database from that pack and feeds every record through the same ingest runtime a vehicle's bus
traffic goes through (decoders -> plausibility -> events -> the one-second and one-minute tiers -> trips and charging sessions).  `read_back(db, concept)` returns what was stored, for tests and for a person
who wants to see that the speed in their CSV came out as the speed."""
import os
import shutil
import sqlite3
import tempfile

from . import autoprofile as AP
from . import readers as RD


def ingest(path, db, pack_dir=None, dbc=None, fmt=None, anchor=None, vehicle_id=None, model=None, rate_hz=1.0, limit=None):
    """-> {"analysis": Analysis, "profile": Profile, "records": n, "fed": n, "stats": {...}}.  Records without a timestamp (a CSV with no time column, an adapter trace) are spaced 1/rate_hz apart."""
    from ..live.runtime import Runtime
    pack = pack_dir or tempfile.mkdtemp(prefix="pack_"); profile, a = AP.autoprofile(path, pack, dbc=dbc, vehicle_id=vehicle_id, model=model, fmt=fmt, anchor=anchor)
    if os.path.exists(db):
        for ext in ("", "-wal", "-shm"):
            try: os.remove(db + ext)
            except OSError: pass
    rt = Runtime(db, profile=profile, monitor=False); fed = 0; stats = RD.Stats(); t = (a.t0 if a.t0 is not None else (anchor if anchor is not None else os.path.getmtime(path))) - 0.0; step = 1.0 / rate_hz
    try:
        for r in RD.read(path, fmt=fmt, anchor=anchor, stats=stats, limit=limit):
            ts = r.ts
            if ts is None: ts = t; t += step
            else: t = max(t, ts)
            if r.bus == "gnss" and r.ts is None: ts = t
            rt.feed(ts, r.bus, r.payload); fed += 1
        rt.publish()
    finally: rt.close()
    if pack_dir is None: shutil.rmtree(pack, ignore_errors=True)
    return {"analysis": a, "profile": profile, "records": stats.records, "fed": fed, "skipped": stats.skipped}


def read_back(db, concept, tier="1s"):
    """[(ts, value)] of a stored quantity in its canonical unit (the registry's), from the database's own tables."""
    from .. import signal_registry as R
    from ..vehicle import lake as LK
    from ..live.replay import CORE_MAP
    LK.adopt_path(db); ent = (R.STORE.get(concept) or {}).get(tier)
    if not ent and tier == "1s" and concept in {c for c, _, _ in CORE_MAP}:                      # a combustion vehicle's coolant temperature lives in the core column the electric vehicles call 'motor'
        from ..vehicle import caps
        col = caps.CORE.get(concept); ent = ("sig_1s", col, dict((c, k) for c, _, k in CORE_MAP)[concept]) if col else None
    if not ent: return []
    table, col, scale = ent
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try: return [(r[0], r[1] / scale) for r in c.execute(f"SELECT ts, {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY ts")]
    except sqlite3.Error: return []
    finally: c.close(); LK.release()
