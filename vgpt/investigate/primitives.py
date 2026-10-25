"""Building blocks of an investigation: events, series at the best available resolution, frozen event context, abnormal-signal detection, trip features.

Everything reads the assistant's read-only connection; nothing is written.  Every function says what resolution it used, because "the earliest abnormal signal" means something different at 1 Hz and
at one-minute averages, and an answer must not claim more precision than the data has."""
import json
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from .. import analytics as A, signal_registry as R, store

CTX_CORE = {"speed": ("speed", 10.0), "battery_soc": ("soc", 10.0), "battery_power": ("power", 1000.0), "battery_temperature": ("batt", 10.0), "coolant_flow": ("flow", 10.0), "ambient_temperature": ("amb", 10.0),
            "hvac_power": ("hvac", 1.0), "motor_temperature": ("motor", 10.0)}
CTX_COLS = ["ts", "speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"]
GENERIC_CONCEPT_RANGE = (0.0, 100.0)


@dataclass
class Event:
    ts: int
    type: str
    sev: str
    detail: str


@dataclass
class Series:
    concept: str
    t: np.ndarray
    v: np.ndarray
    res: float                      # seconds between samples (1 = 1 Hz, 60 = per-minute means)
    source: str
    want: tuple = (0, 0)

    def coverage(self):
        span = max(1.0, self.want[1] - self.want[0]); return min(1.0, len(self.t) * self.res / span)

    def label(self):
        return {1.0: "1 Hz", 60.0: "1-minute averages"}.get(self.res, f"{self.res:g} s") + f" ({self.source})"

    def span(self): return float(self.t[-1] - self.t[0]) if len(self.t) >= 2 else 0.0


@dataclass
class Departure:
    concept: str
    ts: int
    direction: str                  # up | down
    magnitude: float
    baseline: float
    value_at_anchor: float
    unit: str
    res: float


class Data:
    def __init__(self, brain):
        self.b = brain; self.c = brain.c; self.now = int(brain.now); self.tz = int(brain.tz); self.qc = brain.qc(); A.set_tz(self.tz)
        row = self.c.execute("SELECT v FROM meta WHERE k='t0'").fetchone(); self.t0 = int(float(row[0])) if row else 0
        row = self.c.execute("SELECT v FROM meta WHERE k='end'").fetchone(); self.end = int(float(row[0])) if row else self.now
        self.have = {r[0] for r in self.c.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        self._avail = {}

    # ------------------------------------------------------------------------------------------------ small helpers
    def stamp(self, ts): return A.stamp(int(ts))
    def day(self, ts): return A.day_of(int(ts))

    def expand(self, types):
        out = []
        for t in types:
            for x in (R.EVENT_GROUPS.get(t) or (t,)):
                if x not in out: out.append(x)
        return out

    def events(self, types, lo, hi, limit=None):
        types = self.expand(types)
        if not types: return []
        q = "SELECT ts,type,sev,detail FROM events WHERE type IN (%s) AND ts>=? AND ts<? ORDER BY ts" % ",".join("?" * len(types))
        rows = self.c.execute(q + (f" LIMIT {int(limit)}" if limit else ""), (*types, int(lo), int(hi))).fetchall()
        return [Event(*r) for r in rows]

    def all_events(self, lo, hi, exclude=("HARSH_BRAKE", "HARSH_ACCEL", "HARSH_CORNER", "CHARGE_START", "SW_UPDATE", "SERVICE_DUE", "CAMERA_RESTORED", "TYRE_INFLATED")):
        rows = self.c.execute("SELECT ts,type,sev,detail FROM events WHERE ts>=? AND ts<? ORDER BY ts", (int(lo), int(hi))).fetchall()
        return [Event(*r) for r in rows if r[1] not in exclude]

    def available(self, concept):
        if concept not in self._avail: self._avail[concept] = self.qc.profile.concept_available(concept) if concept in R.CONCEPTS else False
        return self._avail[concept]

    def dtc(self):
        """code -> {text, subsystem, severity} of the vehicle's DTC table (stored in meta at provisioning); {} for a database without one."""
        if not hasattr(self, "_dtc"):
            row = self.c.execute("SELECT v FROM meta WHERE k='vehicle.dtc'").fetchone(); self._dtc = json.loads(row[0]) if row else {}
        return self._dtc

    def unit(self, concept): return R.CONCEPTS[concept].unit if concept in R.CONCEPTS else ""
    def label(self, concept): return R.CONCEPTS[concept].label if concept in R.CONCEPTS else concept
    def rng(self, concept):
        c = R.CONCEPTS.get(concept)
        return (c.lo, c.hi) if c is not None and c.lo is not None and c.hi is not None else GENERIC_CONCEPT_RANGE

    # ------------------------------------------------------------------------------------------------ frozen context of an event
    def ctx_core(self, ev):
        row = self.c.execute("SELECT blob FROM event_ctx_v2 WHERE ts=? AND type=?", (ev.ts, ev.type)).fetchone() or self.c.execute("SELECT blob FROM event_ctx WHERE ts=? AND type=?", (ev.ts, ev.type)).fetchone()
        if row is None: return None
        rows = json.loads(zlib.decompress(row[0])); return np.array(rows, dtype=float) if rows else None

    def ctx_x(self, ev):
        if "event_ctx_x" not in self.have: return None
        row = self.c.execute("SELECT blob FROM event_ctx_x WHERE ts=? AND type=?", (ev.ts, ev.type)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    # ------------------------------------------------------------------------------------------------ series at the best resolution
    def series(self, concept, lo, hi, event=None):
        """-> Series or None.  Candidates: the frozen 1 Hz context of the event (core channels, lake channels), the 1 Hz tier, the 1-minute tier.  The finest one that covers at least 60 % of the window wins;
        if none does, the one that covers most."""
        lo, hi = int(lo), int(hi); want = (lo, hi); span = max(1.0, hi - lo); cands = []
        def cover(t): return float((t[-1] - t[0]) / span) if len(t) >= 2 else 0.0
        if event is not None:
            if concept in CTX_CORE:
                a = self.ctx_core(event)
                if a is not None and len(a):
                    col, k = CTX_CORE[concept]; j = CTX_COLS.index(col); t = a[:, 0]; v = a[:, j] / k; m = (t >= lo) & (t < hi) & ~np.isnan(v)
                    if m.sum() >= 60: cands.append(Series(concept, t[m], v[m], 1.0, "frozen event context", want))
            x = self.ctx_x(event)
            if x and concept in x["sig"]:
                t0 = x["t0"]; vals = x["sig"][concept]; t = np.arange(len(vals)) + t0; v = np.array([np.nan if y is None else y for y in vals], dtype=float); m = (t >= lo) & (t < hi) & ~np.isnan(v)
                if m.sum() >= 60: cands.append(Series(concept, t[m], v[m], 1.0, "frozen event context", want))
        store_ = R.STORE.get(concept) or {}
        for tier, res in (("1s", 1.0), ("1m", 60.0)):
            ent = store_.get(tier)
            if not ent or ent[0] not in self.have: continue
            table, col, scale = ent
            try: rows = self.c.execute(f"SELECT ts,{col} FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL ORDER BY ts", (lo, hi)).fetchall()
            except Exception: continue
            if len(rows) < (3 if tier == "1s" else 2): continue
            t = np.array([r[0] for r in rows], dtype=float); v = np.array([r[1] / scale for r in rows], dtype=float)
            res_eff = max(1.0, round(float(np.median(np.diff(t))))) if tier == "1s" and len(t) >= 2 else res
            cands.append(Series(concept, t, v, res_eff, "1 Hz tier" if tier == "1s" else "1-minute tier", want))
        if not cands: return None
        good = [c for c in cands if cover(c.t) >= 0.6]
        if good: return min(good, key=lambda c: c.res)
        fine = [c for c in cands if c.res <= 1.5]; coarse = [c for c in cands if c.res > 1.5 and cover(c.t) >= 0.6]
        if fine and coarse:                                                                          # 1 Hz around the event, one-minute averages before it
            f = max(fine, key=lambda c: len(c.t)); c0 = max(coarse, key=lambda c: cover(c.t)); keep = c0.t < f.t[0] - 30
            return Series(concept, np.r_[c0.t[keep], f.t], np.r_[c0.v[keep], f.v], 1.0, "1 Hz around the event, 1-minute before", want)
        return max(cands, key=lambda c: (cover(c.t), -c.res))

    # ------------------------------------------------------------------------------------------------ abnormal-signal detection
    def departure(self, s, anchor, direction=None, k=3.0):
        """First moment before `anchor` at which the series left its own baseline (the first 40 % of the window, robust median / MAD) and stayed away for 3 samples (2 at one-minute resolution)."""
        if s is None or len(s.t) < 6: return None
        pre = s.t < anchor
        if pre.sum() < 6: return None
        t, v = s.t[pre], s.v[pre]; nb = max(4, int(0.4 * len(t))); base = v[:nb]; med = float(np.median(base)); mad = float(np.median(np.abs(base - med)))
        lo, hi = self.rng(s.concept); floor = 0.02 * (hi - lo); thr = max(k * 1.4826 * mad, floor); dev = v - med
        for i in range(nb, len(t) - 1):
            step = float(np.median(np.diff(t[i:i + 4]))) if len(t) - i >= 3 else s.res; need = 2 if step >= 30 else 3
            if i + need > len(t): break
            win = dev[i:i + need]
            if np.all(np.abs(win) > thr) and (np.all(win > 0) or np.all(win < 0)):
                d = "up" if win[0] > 0 else "down"
                if direction and d != direction: continue
                end_v = float(v[-1]); return Departure(s.concept, int(t[i]), d, float(end_v - med), med, end_v, self.unit(s.concept), s.res)
        return None

    # ------------------------------------------------------------------------------------------------ trips
    def trips(self, lo, hi, min_km=0.5):
        rows = self.c.execute("SELECT t.start,t.end,t.km,t.kwh,t.wh_km,t.avg_speed,t.hwy_share,t.avg_amb,t.avg_hvac_w,t.soc_start,t.soc_end,x.harsh_brake,x.harsh_accel,x.harsh_corner,x.regen_kwh,x.rain_share,x.adas_share "
                              "FROM trips t LEFT JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<? AND t.km>=? ORDER BY t.start", (int(lo), int(hi), min_km)).fetchall()
        keys = ("start", "end", "km", "kwh", "wh_km", "avg_speed", "hwy_share", "avg_amb", "avg_hvac_w", "soc_start", "soc_end", "harsh_brake", "harsh_accel", "harsh_corner", "regen_kwh", "rain_share", "adas_share")
        return [dict(zip(keys, r)) for r in rows]

    def trip_features(self, tr):
        """Per-trip features from the one-minute tier (idle share, speed spread, peak temperatures, ...), added to the trip dict."""
        r = self.c.execute("SELECT COUNT(*), AVG(speed)/10.0, AVG(speed*speed)/100.0, SUM(speed<20), MAX(batt)/10.0, AVG(motor)/10.0, MAX(motor)/10.0, MIN(flow)/10.0, AVG(power)/1000.0, MIN(soc)/10.0 FROM sig_1m WHERE ts>=? AND ts<?", (tr["start"] - 59, tr["end"])).fetchone()
        n = r[0] or 0; f = dict(tr)
        if n:
            f["minutes"] = n; f["speed_std"] = math.sqrt(max(0.0, (r[2] or 0) - (r[1] or 0) ** 2)); f["idle_share"] = (r[3] or 0) / n; f["batt_max"] = r[4]; f["motor_mean"] = r[5]; f["motor_max"] = r[6]; f["flow_min"] = r[7]
            f["power_mean_kw"] = r[8]; f["duration_min"] = (tr["end"] - tr["start"]) / 60.0
        else: f["minutes"] = 0; f["duration_min"] = (tr["end"] - tr["start"]) / 60.0
        km = max(tr["km"], 0.1); f["harsh_per_100km"] = 100.0 * ((tr.get("harsh_brake") or 0) + (tr.get("harsh_accel") or 0)) / km; f["regen_share"] = (tr.get("regen_kwh") or 0) / max(abs(tr["kwh"]) + (tr.get("regen_kwh") or 0), 0.05)
        return f

    def log_lines(self, lo, hi, match=None, limit=8):
        q = "SELECT ts,src,msg FROM logs WHERE ts>=? AND ts<?"; args = [int(lo), int(hi)]
        if match: q += " AND msg LIKE ?"; args.append(f"%{match}%")
        return self.c.execute(q + " ORDER BY ts LIMIT ?", (*args, limit)).fetchall()


def robust_z(x, ref):
    ref = np.asarray(ref, dtype=float); med = float(np.median(ref)); mad = float(np.median(np.abs(ref - med))) * 1.4826 or float(np.std(ref)) or 1.0
    return (np.asarray(x, dtype=float) - med) / mad


def effect(a, b):
    """Difference of medians in pooled robust spread, and the share of `a` above the 75th percentile of `b`."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 3: return None
    sp = 0.5 * (float(np.median(np.abs(a - np.median(a)))) + float(np.median(np.abs(b - np.median(b))))) * 1.4826 or float(np.std(np.r_[a, b])) or 1.0
    return {"diff": float(np.median(a) - np.median(b)), "d": float((np.median(a) - np.median(b)) / sp), "a": float(np.median(a)), "b": float(np.median(b)), "above75": float(np.mean(a > np.percentile(b, 75))), "below25": float(np.mean(a < np.percentile(b, 25)))}
