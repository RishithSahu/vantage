"""A capture file as a live source: what the car side of vgpt/link.py plays to Vantage when the person picks a CAPTURE (a candump / ASC / BLF / CSV / JSON log) instead of a recorded vehicle database.

    prepare(path, dbc, workdir)   -> Prepared: the profile pack the capture implies (autoprofile), an empty provisioned history database, the records in time order, begin / end
    CaptureSource(sched, prepared)  the same face as replay.WireSource: `deliver(upto, deadline)` hands every message of the seconds up to `upto` to `sink`; the streamer releases each at its own moment

Nothing is configured: the pack is derived from the capture's own content (docs/DATA_SOURCES.md), sent to Vantage with the history, and Vantage builds its decoders from it."""
import os
import shutil
import tempfile

from . import autoprofile as AP
from . import readers as RD

STEP_NO_CLOCK = 0.02                                # a capture with no timestamps (an adapter trace) plays at 50 records a second


class Prepared:
    def __init__(self): self.profile = None; self.analysis = None; self.recs = []; self.begin = 0; self.end = 0; self.db = None; self.dir = None; self.format = None


def prepare(path, dbc=None, workdir=None, anchor=None, vehicle_id=None):
    """Read the capture once, derive the pack, make the empty history database (provisioned for that pack, so Vantage's runtime needs nothing else)."""
    from ..live.runtime import Runtime
    pr = Prepared(); pr.dir = workdir or tempfile.mkdtemp(prefix="capture_"); os.makedirs(pr.dir, exist_ok=True)
    pr.profile, pr.analysis = AP.autoprofile(path, os.path.join(pr.dir, "pack"), dbc=dbc, vehicle_id=vehicle_id, anchor=anchor); pr.format = pr.analysis.format
    anchor = anchor if anchor is not None else os.path.getmtime(path); recs = []; t = None
    for r in RD.read(path, anchor=anchor):
        ts = r.ts
        if ts is None: t = (t if t is not None else anchor) + STEP_NO_CLOCK; ts = t
        else: t = ts
        recs.append(RD.Rec(float(ts), r.bus, r.payload))
    recs.sort(key=lambda r: r.ts)
    if not recs: raise ValueError("the capture holds no records")
    pr.recs = recs; pr.begin = int(recs[0].ts) - 1; pr.end = int(recs[-1].ts) + 1; pr.db = os.path.join(pr.dir, "history.db")
    rt = Runtime(pr.db, profile=pr.profile, monitor=False)
    for k, v in (("t0", pr.begin), ("end", pr.begin)): rt.conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))
    rt.conn.commit(); rt.close()
    return pr


class CaptureSource:
    """Plays prepared records: `deliver(upto)` emits the messages of the seconds after `self.t` up to `upto`, in time order."""
    kind = "wire"

    def __init__(self, sched, prepared, **_):
        self.sched, self.p = sched, prepared; self.recs = prepared.recs; self.begin = prepared.begin; self.t = prepared.begin; self.i = 0; self.sink = None; self.msgs = 0

    def deliver(self, upto, deadline=None):
        import time
        n = 0; recs = self.recs
        while self.i < len(recs) and int(recs[self.i].ts) <= upto:
            r = recs[self.i]
            if int(r.ts) <= self.t: self.i += 1; continue                                         # already delivered (a resume)
            self.sink(r.ts, r.bus, r.payload); self.i += 1; self.msgs += 1; n += 1
            if deadline is not None and (n & 255) == 0 and time.perf_counter() > deadline:
                self.t = max(self.t, int(r.ts) - 1); return self.t
        self.t = upto; return upto

    def restart(self): self.i = 0; self.t = self.begin

    def seek(self, t):
        """Resume at second t (Vantage already holds everything up to there)."""
        self.t = int(t)
        while self.i < len(self.recs) and int(self.recs[self.i].ts) <= self.t: self.i += 1

    def close(self): pass


def drive_plan_wire(begin, end):
    """The replay plan of a capture, in the wire form Vantage's clock reads: real time from the first record to the last, then it stops."""
    return {"begin": begin, "end": end, "starts": [begin, end], "segs": [[1.0, end], [0.0, None]], "loop": False}
