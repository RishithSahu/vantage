"""Query executor: runs a validated Query against the READ-ONLY database and returns a structured Result - the number AND where it came from.

Responsibilities (each one is a deterministic function; nothing here calls a language model):
  * resolve the scope (time window x optional entity selection x conditions) into samples,
  * pick the data tier (1 Hz for short recent windows, per-minute for history; trip ledger for whole-trip accounting),
  * apply sample-level conditions (named predicates and signal thresholds),
  * compute measured / derived / accumulated values and the requested aggregation or analysis,
  * enforce limits (rows, intervals, windows), and
  * record provenance: tables, tier, window, sample count, data-quality flags, estimated / assumptions.
It returns data, never prose (answer_composer.py writes the words) and never raises for 'no data': it returns status "no_data" with the reason."""
import math
import os
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import analytics as A, derived as D, kinematics as K, live_snapshot as LIVE, qa_core as Q, sensor_quality as SQ, signal_registry as R
from .frame import Frame
from .query_ir import EVENT_PAGE
from .query_planner import Ctx, PlanError, assert_read_only, compile_entities, compile_events, compile_select, plan as make_plan, resolve_time, local_midnight, DAY

MAX_ROWS = 1_800_000              # per load: three years of minute rows are 1.6 M; a lower cap silently answered a long period from its first part only
MAX_INTERVALS = 200
TIER_FROM_DAYS = int(os.environ.get("VGPT_TIER_FROM_DAYS", "120"))        # windows at least this long are answered from the DAILY TIER when the question is a plain statistic (no conditions, no trip selection)
DEFAULT_CONTEXT = ("speed", "battery_power", "battery_temperature", "coolant_flow", "battery_soc", "ambient_temperature", "motor_temperature")


@dataclass
class Result:
    op: str
    status: str = "ok"                         # ok | no_data | unsupported
    value: object = None                       # float / str / dict, depends on op
    unit: str = ""
    target: str = ""
    agg: str = ""
    when: int = None                           # timestamp the value belongs to (current value, extreme, event)
    period: tuple = (0, 0)
    period_label: str = ""
    items: list = field(default_factory=list)  # rows: ranking, events, per-group values, per-day values ...
    extra: dict = field(default_factory=dict)  # op-specific details (stats, series, groups ...)
    tier: str = ""
    sources: list = field(default_factory=list)
    samples: int = 0
    quality: str = "high"                      # high | medium | low
    flags: list = field(default_factory=list)  # sensor-quality flags
    estimated: bool = False
    quality_kind: str = "measured"             # measured | derived | estimated
    assumptions: list = field(default_factory=list)
    via: list = field(default_factory=list)
    formula: str = ""
    reason: str = ""                           # why there is no value
    notes: list = field(default_factory=list)
    plan: list = field(default_factory=list)   # human-readable plan steps (provenance)
    query_text: str = ""


class Executor:
    def __init__(self, conn, profile, ctx, legacy=None, extra_rows=None):
        self.c = conn; self.p = profile; self.ctx = ctx; self.legacy = legacy; A.set_tz(ctx.tz)
        try: self.c.execute("PRAGMA query_only=ON")                          # belt and braces on top of mode=ro
        except sqlite3.Error: pass
        for pragma in ("PRAGMA cache_size=-32768", "PRAGMA temp_store=MEMORY", "PRAGMA mmap_size=268435456"):   # a question reads the same days several times: keep them in memory (32 MB cache, 256 MB mapped, both reclaimable)
            try: self.c.execute(pragma)
            except sqlite3.Error: pass

    # ------------------------------------------------------------------ SQL access (only compiled, asserted statements)
    def _rows(self, sql, args):
        assert_read_only(sql); return self.c.execute(sql, args).fetchall()

    def load_frame(self, concepts, lo, hi, tier):
        """Aligned Frame of the MEASURED concepts over [lo, hi) from the best table/tier for each; columns converted to engineering units.
        One question often asks for the same days several times (an explanation compares periods that share a window): the last few loads are kept, valid only while the database is unchanged (PRAGMA data_version), and handed out as copies."""
        try: ver = self.c.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error: return self._load_frame(concepts, lo, hi, tier)
        memo = self.__dict__.setdefault("_frame_memo", {}); key = (tuple(sorted(set(concepts))), int(lo), int(hi), tier, ver, getattr(self.ctx, "hz_lo", None))
        hit = memo.get(key)
        if hit is None:
            if len(memo) >= 6: memo.pop(next(iter(memo)))
            hit = memo[key] = self._load_frame(concepts, lo, hi, tier)
        return Frame(hit.ts.copy(), {k: v.copy() for k, v in hit.cols.items()}, hit.res, hit.tier, dict(hit.source), list(hit.intervals))

    def _load_frame(self, concepts, lo, hi, tier):
        by_table = {}
        if tier == "1s" and (self.ctx.hz_lo is None or lo < self.ctx.hz_lo):
            from . import archive
            self.archive_note = archive.hydrate(self.c, lo, hi)                                                          # the archived seconds of this window, for this question only
        for c in sorted(set(concepts)):
            tiers = self.p.tiers_of(c) or (self.p.resolve(c).tiers if self.p.resolve(c).status == "measured" else {})
            if not tiers: continue
            use = tier if tier in tiers else ("1m" if "1m" in tiers else next(iter(tiers)))
            tb, col, sc, *rest = tiers[use]; by_table.setdefault((tb, use), []).append((c, col, sc, rest[0] if rest else 0.0))       # a vehicle's storage may be offset as well as scaled (value = raw / scale + offset)
        if not by_table: return Frame(np.array([], dtype=np.int64), {}, 60 if tier == "1m" else 1, tier)
        # base = the sig_* table of the tier if it is needed, else the table with the finest resolution
        order = sorted(by_table, key=lambda k: (0 if k[0].startswith("sig_") else 1 if k[0].startswith("xsig") else 2, k[1] != tier))
        loaded = []
        for (tb, use) in order:
            items = by_table[(tb, use)]
            cols = []
            for _, col, _, _ in items:                                                          # the virtual "tyre pressure" concept needs all four wheel columns next to whatever else the table is asked for
                for x in (["tp_fl", "tp_fr", "tp_rl", "tp_rr"] if col == "tp_*" else [col]):
                    if x not in cols: cols.append(x)
            sql, args = compile_select(tb, cols, lo, hi)
            rows = self._rows(sql + f" LIMIT {MAX_ROWS}", args)
            loaded.append((tb, use, items, cols, rows))
        base_tb, base_use, _, _, base_rows = loaded[0]
        res = 1 if base_use == "1s" else 60 if base_use == "1m" else 86400
        ts = np.array([r[0] for r in base_rows], dtype=np.int64); frame = Frame(ts, {}, res, base_use if base_use in ("1s", "1m") else "day", {}, [(lo, hi)])
        for (tb, use, items, cols, rows) in loaded:
            if not rows:
                for c, col, sc, off in items: frame.cols[c] = np.full(len(ts), np.nan); frame.source[c] = f"{tb}.{col}"
                continue
            arr = np.array(rows, dtype=float); tts = arr[:, 0].astype(np.int64)
            idx = np.searchsorted(tts, ts, side="right") - 1; ok = (idx >= 0)
            tol = 90 if use == "1m" else 5 if use == "1s" else 86400
            ok &= (np.abs(ts - tts[np.clip(idx, 0, len(tts) - 1)]) <= tol) if tb != base_tb else True
            idx = np.clip(idx, 0, len(tts) - 1)
            for c, col, sc, off in items:
                if col == "tp_*":
                    v = arr[idx][:, [1 + cols.index(x) for x in ("tp_fl", "tp_fr", "tp_rl", "tp_rr")]] / sc + off; vals = np.where(ok, np.nanmin(v, axis=1), np.nan)
                else:
                    j = 1 + cols.index(col); vals = np.where(ok, arr[idx][:, j] / sc + off, np.nan)
                if c == "gps_speed": vals = np.where(vals < 0, np.nan, vals)
                frame.cols[c] = vals; frame.source[c] = f"{tb}.{col}"
        return frame

    # ------------------------------------------------------------------ long windows: the daily tier
    def _tier_days(self, q, scope, target, aggs):
        """For a plain statistic over a LONG window (no conditions, no trip selection, a measured column of sig_1m, an aggregation the daily tier holds exactly) -> (per-day {day: (n, avg, mx, mn)}, col, scale, offset,
        (lo, hi), label) read from the daily tier, else None (the minute rows are scanned as before).  mean = sum(n x daily mean) / sum(n); min / max / count are exact; the minute of an extreme is found by reading that one day."""
        if os.environ.get("VGPT_TIER_FAST", "1") != "1" or scope.select is not None or scope.conditions or target in D.ACCUMULATED or target not in R.CONCEPTS or R.CONCEPTS[target].dtype == "categorical": return None
        ent = (R.STORE.get(target) or {}).get("1m")
        if not ent or not (ent[0] == "sig_1m" or (str(ent[0]).startswith("lkv_") and ent[1] == "v")) or ent[1] == "tp_*" or (aggs is not None and (q.agg or "mean") not in aggs): return None       # a core column of the minute table, or a lake signal (its minute view)
        try: lo, hi, label = resolve_time(scope.time, self.ctx)
        except Exception: return None
        if hi - lo < TIER_FROM_DAYS * 86400: return None
        from . import dailytier as DT, store
        tz = store.tz_of(self.c)
        if DT._tier_state(self.c, ent[0], False, tz) is None: return None
        per = DT.per_day(self.c, tz, ent[0], False, [ent[1]], lo, hi, has_mode=ent[0] == "sig_1m").get(ent[1]) or {}
        if not per: return None
        return per, ent[1], float(ent[2]), float(ent[3]) if len(ent) > 3 else 0.0, (lo, hi), label, tz

    def _tier_value(self, q, scope, target, agg):
        got = self._tier_days(q, scope, target, ("mean", "min", "max", "count"))
        if got is None: return None
        per, col, sc, off, period, label, tz = got; c = R.CONCEPTS[target]; total = sum(v[0] for v in per.values())
        if total <= 0: return None
        agg = agg or "mean"
        if agg == "mean": val = sum(v[0] * v[1] for v in per.values()) / total / sc + off
        elif agg == "max": val = max(v[2] for v in per.values()) / sc + off
        elif agg == "min": val = min(v[3] for v in per.values()) / sc + off
        else: val = float(total)
        r = Result(q.op, target=target, agg=agg, period=period, period_label=label, extra={"entities": None}, samples=int(total), tier="daily tier", sources=[f"sig_1m.{col} (daily tier)"], quality="high")
        r.value = float(val); r.unit = c.unit if agg != "count" else "samples"
        r.notes.append(f"from the daily tier: the mean, maximum and minimum of every recorded minute of each of {len(per)} days (windows of {TIER_FROM_DAYS} days or more)")
        return r

    def _tier_extreme(self, q, agg):
        got = self._tier_days(q, q.scope, q.target, None)
        if got is None or (q.agg or "max") not in ("min", "max"): return None
        per, col, sc, off, period, label, tz = got; c = R.CONCEPTS[q.target]; total = sum(v[0] for v in per.values())
        days = sorted(per); key = (lambda d: per[d][2]) if agg == "max" else (lambda d: -per[d][3]); best = max(key(d) for d in days); day = next(d for d in days if key(d) == best)
        lo, hi = period; d0, d1 = max(lo, day * 86400 - tz), min(hi, (day + 1) * 86400 - tz)
        need = self._concepts_for(q) | ({"drive_mode"} if self.p.concept_available("drive_mode") else set()); need |= self._acc_needs(q.target)
        fr = self.load_frame(need, d0, d1, "1m")
        if len(fr) == 0: return None
        vals = self._series(fr, q.target)
        if vals is None or np.isnan(vals).all(): return None
        i = int(np.nanargmax(vals) if agg == "max" else np.nanargmin(vals))
        r = Result("extreme", value=float(vals[i]), unit=c.unit, target=q.target, agg=agg, when=int(fr.ts[i]), period=period, period_label=label, samples=int(total))
        r.extra["state"] = {k: float(fr.cols[k][i]) for k in ("speed", "battery_power", "ambient_temperature") if k in fr.cols and not np.isnan(fr.cols[k][i])}
        if q.unit and R.unit_family(q.unit) == R.unit_family(c.unit): r.value = R.convert(r.value, c.unit, q.unit); r.unit = q.unit
        r = self._finish(r, fr, [q.target]); r.samples = int(total); r.tier = "daily tier"; r.sources = [f"sig_1m.{col} (daily tier)"]
        r.notes.append(f"found in the daily tier ({len(per)} days), then read from the minute rows of the day it happened")
        return r

    # ------------------------------------------------------------------ scope
    def _intervals(self, scope, res, cap=MAX_INTERVALS):
        """-> (intervals, entity rows or None, label, (lo, hi))"""
        lo, hi, label = resolve_time(scope.time, self.ctx)
        if scope.select is None: return [(lo, hi)], None, label, (lo, hi)
        e = scope.select; sql, args = compile_entities(e.entity, e.klass, e.rank_by, e.order, e.k, lo, hi)
        rows = self._rows(sql, args)
        if cap and len(rows) > cap: rows = rows[:cap]
        ents = [{"start": int(r[0]), "end": int(r[1]), **({"km": r[2], "kwh": r[3], "wh_per_km": r[4], "avg_speed": r[5], "hwy_share": r[6], "avg_amb": r[7], "avg_hvac_w": r[8]} if e.entity == "trip" else {"kind": r[2], "charging_power": r[3], "energy_used": r[4]})} for r in rows]
        return [(x["start"], x["end"] + 1) for x in ents], ents, label, (lo, hi)

    def _last_known(self, t):
        """The newest recorded value of a live signal at or before 'now' (either tier), clearly labelled as not current (-> text or None)."""
        best = None
        for tier in ("1s", "1m"):
            st = R.STORE.get(t, {}).get(tier)
            if not st: continue
            tbl, col, scale = st[0], st[1], st[2]
            try: row = self.c.execute(f"SELECT ts, {col} FROM {tbl} WHERE ts <= ? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (int(self.ctx.now),)).fetchone()
            except Exception: continue
            if row and (best is None or row[0] > best[0]): best = (row[0], float(row[1]) / float(scale or 1.0))
        if best is None: return None
        c = R.CONCEPTS[t]
        if c.dtype == "categorical": return None
        age = int(self.ctx.now - best[0])
        ago = f"{age // 3600} h {age % 3600 // 60:02d} min ago" if age >= 3600 else f"{age // 60} min ago"
        return f"{best[1]:.1f} {c.unit} at {A.stamp(int(best[0]))} ({ago}), which is not a current reading"

    def _concepts_for(self, q, extra=()):
        from .capability import concepts_of
        from .query_planner import measured_needs
        names = [c for c in concepts_of(q) if c in R.CONCEPTS] + [c for c in extra if c]
        need = set()
        for n in names: need |= measured_needs(n, self.p)
        for n in list(need):
            for comp in R.QUALITY_COMPANIONS.get(n, ()):
                if self.p.concept_available(comp) and self.p.resolve(comp).status == "measured": need.add(comp)
        return need

    def _interval_mask(self, frame, intervals):
        m = np.zeros(len(frame.ts), bool)
        for lo, hi in intervals: m |= (frame.ts >= lo) & (frame.ts < hi)
        return m

    def _cond_mask(self, frame, conditions, tier):
        m = np.ones(len(frame.ts), bool); used = []
        for c in conditions:
            if c.predicate and c.op == "not":                                                   # the complement of a named state: the same rule, inverted
                inner, err = self._cond_mask(frame, [type(c)(predicate=c.predicate)], tier)
                if inner is None: return None, err
                m &= ~inner; used.append("not " + c.predicate); continue
            if c.predicate:
                rule = R.PREDICATES[c.predicate][1]; kind = rule[0]; used.append(c.predicate)
                if kind == "mode":
                    mode = frame.column("drive_mode")
                    if mode is None: return None, f"the driving state (city / highway / charging) is not recorded"
                    m &= (mode == rule[1])
                elif kind == "mode_in":
                    mode = frame.column("drive_mode")
                    if mode is None: return None, "the driving state is not recorded"
                    m &= np.isin(mode, rule[1])
                elif kind == "signal":
                    v = self._series(frame, rule[1])
                    if v is None: return None, f"{R.label_of(rule[1])} is not available"
                    op = rule[2]; thr = rule[3]; cond = (v < thr) if op == "<" else (v > thr)
                    if len(rule) > 4 and rule[4] == "driving":
                        mode = frame.column("drive_mode"); cond &= np.isin(mode, (1, 2)) if mode is not None else True
                    m &= cond & ~np.isnan(v)
                elif kind == "event":
                    em = self._event_mask(frame, rule[1], rule[2])
                    if not em.any() and not self.p.has_events(rule[1]): return None, f"no {R.EVENTS[rule[1]][0]} is recorded for this vehicle"
                    m &= em
                elif kind == "hour_not_in":
                    hrs = ((frame.ts + self.ctx.tz) % DAY) // 3600; m &= (hrs < rule[1]) | (hrs >= rule[2])
            else:
                v = self._series(frame, c.signal)
                if v is None: return None, f"{R.label_of(c.signal)} is not available"
                thr = float(c.value)
                if c.unit and R.unit_family(c.unit) == R.unit_family(R.CONCEPTS[c.signal].unit): thr = R.convert(thr, c.unit, R.CONCEPTS[c.signal].unit)
                m &= {"gt": v > thr, "lt": v < thr, "ge": v >= thr, "le": v <= thr, "eq": v == thr}[c.op] & ~np.isnan(v); used.append(c.signal)
        return m, ""

    def _event_mask(self, frame, etype, pad):
        if len(frame.ts) == 0: return np.zeros(0, bool)
        sql, args = compile_events([etype], int(frame.ts[0]) - pad, int(frame.ts[-1]) + frame.res + pad, "asc"); ev = [r[0] for r in self._rows(sql, args)]
        m = np.zeros(len(frame.ts), bool)
        for t in ev:
            if frame.res >= 60: m |= (frame.ts <= t) & (t < frame.ts + frame.res)
            else: m |= np.abs(frame.ts - t) <= pad
        return m

    def _series(self, frame, concept):
        """Values of a concept over the frame: measured column, or a pointwise derived series (None when not obtainable)."""
        v = frame.column(concept)
        if v is not None: return v
        if concept in D.POINTWISE:
            ds = D.compute(concept, frame, self.p)
            return None if ds is None else ds.values
        return None

    # ------------------------------------------------------------------ helpers
    def _finish(self, r, frame, concepts, now=None):
        if frame is None:                                                                        # daily values taken from the trip ledger (authoritative whole-trip accounting): there is no sample frame to assess
            r.tier = r.tier or "trip ledger"; r.sources = r.sources or ["trips"]; r.quality = "high"; r.flags = []; r.quality_kind = getattr(r, "quality_kind", None) or "derived"
            return r
        r.tier = {"1s": "1 Hz", "1m": "per-minute", "day": "daily"}.get(getattr(frame, "tier", ""), getattr(frame, "tier", ""))
        r.sources = sorted({v for k, v in frame.source.items() if k in concepts or not concepts}); r.samples = r.samples or int(len(frame))
        level, flags = SQ.assess_frame(frame, [c for c in concepts if c in frame.cols], now=now)
        for c in concepts:                                                                       # derived results are range-checked too (a stuck wheel sensor shows up as impossible slip)
            if c not in frame.cols and c in D.POINTWISE and c in R.CONCEPTS:
                try:
                    ds = self._series(frame, c)
                    if ds is not None: flags += SQ.range_violation(ds, c)
                except Exception: pass
        if any(f.severity == "major" for f in flags): level = "low"
        elif flags and level == "high": level = "medium"
        r.quality = level; r.flags = [f.to_dict() for f in flags]
        return r

    def _agg(self, v, agg):
        v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
        if len(v) == 0: return None
        return {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median, "std": lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "first": lambda x: x[0], "last": lambda x: x[-1], "delta": lambda x: x[-1] - x[0],
                "count": len, "p95": lambda x: np.percentile(x, 95), "sum": np.sum}[agg](v)

    def _no_data(self, op, reason, q, label="", period=(0, 0)):
        return Result(op, "no_data", reason=reason, target=q.target or "", period=period, period_label=label)

    # ------------------------------------------------------------------ entry point
    def run(self, q):
        pl = make_plan(q, self.p, self.ctx)
        fn = {"current_value": self._current, "aggregate": self._aggregate, "acceleration_band": self._acceleration_band, "extreme": self._extreme, "trend": self._trend, "compare": self._compare, "count": self._count, "list_events": self._list_events,
              "event_context": self._event_context, "rank": self._rank, "correlation": self._correlation, "percent_time": self._percent_time, "explain_event": self._explain, "capability": self._capability, "explain_change": self._explain_change, "detect_change": self._detect_change}[q.op]
        r = fn(q); r.plan = pl.explain(); r.query_text = q.to_text(); r.op = r.op or q.op
        res = self.p.resolve(q.target) if q.target else None
        if res is not None and res.status in ("derived", "estimated"):
            r.via = list(res.requires); r.formula = r.formula or res.formula; r.quality_kind = r.quality_kind if r.quality_kind != "measured" else res.status
            if res.status == "estimated": r.estimated = True; r.assumptions = list(dict.fromkeys(r.assumptions + res.assumptions))
        if res is not None and res.status == "estimated": r.estimated = True
        return r

    def _acceleration_band(self, q):
        lo, hi, label = resolve_time(q.scope.time, self.ctx)
        bounds = sorted(float(c.value) for c in q.scope.conditions if c.signal == "speed" and c.op in ("ge", "gt", "le", "lt"))
        if len(bounds) != 2: return self._no_data(q.op, "the speed band is incomplete", q, label, (lo, hi))
        fr = self.load_frame(["speed", "drive_mode"], lo, hi, "1s")
        if len(fr.ts) < 2 or "speed" not in fr.cols: return self._no_data(q.op, "there are not enough valid speed samples", q, label, (lo, hi))
        mode = fr.cols.get("drive_mode")
        valid = np.isfinite(fr.cols["speed"])
        if mode is not None: valid &= np.isin(mode, (1, 2))
        other = [c for c in q.scope.conditions if not (c.signal == "speed" and c.op in ("ge", "gt", "le", "lt"))]
        if other:
            cm, err = self._cond_mask(fr, other, "1s")
            if cm is None: return self._no_data(q.op, err, q, label, (lo, hi))
            valid &= cm
        runs = K.band_runs(fr.ts, fr.cols["speed"], bounds[0], bounds[1], valid=valid, max_gap_s=max(3, 3 * fr.res))
        if not runs: return self._no_data(q.op, f"no valid upward {bounds[0]:g} to {bounds[1]:g} km/h run was recorded", q, label, (lo, hi))
        value = float(np.mean([x.acceleration for x in runs]))
        r = Result(q.op, value=value, unit="m/s2", target="acceleration", agg="mean", period=(lo, hi), period_label=label,
                   samples=len(runs), tier="1s", sources=[fr.source.get("speed", "speed")], quality_kind="derived", via=["speed"],
                   formula="(upper speed - lower speed) / elapsed crossing time", extra={"low_kmh": bounds[0], "high_kmh": bounds[1], "runs": [x.__dict__ for x in runs]})
        return self._finish(r, fr, ["speed"])

    # ------------------------------------------------------------------ current value
    def _current(self, q):
        t = q.target; res = self.p.resolve(t); lo, hi, label = resolve_time(q.scope.time, self.ctx)
        if t == "range_estimate":
            er = A.est_range(self.c, self.ctx.now)
            if not er.get("ok"): return self._no_data("current_value", er.get("why", "not enough driving history"), q)
            r = Result("current_value", value=float(er["range_km"]), unit="km", target=t, quality_kind="estimated", estimated=True, samples=0, sources=["sig_1s/sig_1m (state of charge)", "trips (recent consumption)"], tier="trip ledger", period_label="right now",
                       assumptions=[f"based on your recent consumption of {er['wh_km']:.0f} Wh/km"], extra={"soc": er["soc"], "wh_per_km": er["wh_km"], "as_of": er["time"]}, via=["battery_soc", "wh_per_km"])
            return r
        # The dashboard and assistant share this exact snapshot contract for
        # core live signals.  It deliberately does not fall back to an old row:
        # no nearby sample means "no live value", not zero and not yesterday's
        # value presented as current.
        live = LIVE.reading(self.c, self.ctx.now, t)
        if t in LIVE.CONCEPT_TO_STATE:
            if live is None:
                last = self._last_known(t)
                return self._no_data("current_value", f"there is no live reading of {R.label_of(t)} at this moment" + (f"; the last recorded value was {last}" if last else ""), q, "right now")
            c = R.CONCEPTS[t]
            value = float(live["value"])
            if c.dtype == "categorical" and t == "drive_mode":
                value = ["parked", "city driving", "motorway driving", "charging"][int(value)] if 0 <= int(value) <= 3 else str(value)
            r = Result("current_value", value=value, unit="" if c.dtype == "categorical" else c.unit, target=t,
                       when=live["sample_ts"], period=(live["sample_ts"], live["sample_ts"] + (1 if live["tier"] in ("1 Hz", "live") else 60)),
                       period_label="right now", samples=1, tier=live["tier"],
                       sources=["live bus" if live.get("live") and t in live.get("live_concepts", ()) else "canonical live snapshot"], extra={"stale_s": live["age_s"], "snapshot_ts": live["sample_ts"], "live": bool(live.get("live") and t in live.get("live_concepts", ()))})
            if t == "speed" and live.get("gps") is not None:
                r.extra["gps"] = live["gps"].get("spd")
            return r
        bus = LIVE.LIVE_BUS                                                                                                      # a plain signal the bus carries (engine speed, a tyre pressure ...), asked about the present: straight from what is arriving
        lv = bus.value(t, self.ctx.now) if bus is not None and t not in LIVE.CONCEPT_TO_STATE else None
        if lv is not None:
            c = R.CONCEPTS[t]; ts = int(lv["ts"])
            return Result("current_value", value=float(lv["value"]), unit=c.unit, target=t, when=ts, period=(ts, ts + 1), period_label="right now", samples=1, tier="live", sources=["live bus"], extra={"stale_s": lv["age_s"], "live": True})
        need = self._concepts_for(q, extra=("speed", "gps_speed") if t == "speed" else ())
        need2 = need | {"drive_mode"} if self.p.concept_available("drive_mode") else need
        stale = None; fr = None; vals = None
        fine = self.ctx.hz_lo is not None and self.ctx.now - 6 * 3600 >= self.ctx.hz_lo - 1                                   # the 1 Hz tier covers the last 6 hours: take the newest second, not the newest minute mean
        for (wlo, whi, wtier) in ((lo, hi, "1s"), (self.ctx.now - 6 * 3600, self.ctx.now + 1, "1s" if fine else "1m"), (self.ctx.now - 6 * 3600, self.ctx.now + 1, "1m")) + (((self.ctx.hz_hi - 3600, self.ctx.hz_hi + 1, "1s"),) if self.ctx.hz_hi else ()) + ((max(self.ctx.lo, self.ctx.now - 30 * DAY), self.ctx.now + 1, "1m"),):   # now, then the last known reading
            f = self.load_frame(need2, wlo, whi, wtier)
            if len(f) == 0: continue
            v = self._series(f, t)
            if v is not None and (~np.isnan(v)).any(): fr, vals = f, v; break
        if fr is None: return self._no_data("current_value", f"there is no recent reading of {R.label_of(t)}", q, label)
        okidx = np.where(~np.isnan(vals))[0]; stale = int(self.ctx.now - fr.ts[okidx[-1]]) if len(okidx) else None
        ok = np.where(~np.isnan(vals))[0]
        if len(ok) == 0: return self._no_data("current_value", f"{R.label_of(t)} has no valid recent reading", q, label)
        i = int(ok[-1]); c = R.CONCEPTS[t]; val = float(vals[i]); r = Result("current_value", value=val, unit=c.unit, target=t, when=int(fr.ts[i]), period=(int(fr.ts[max(0, i - 5)]), int(fr.ts[i]) + fr.res), period_label=label, samples=int(len(ok)))
        if t == "tyre_pressure":
            r.extra["wheels"] = {w: (None if fr.column(f"tyre_pressure_{w}") is None else float(fr.column(f"tyre_pressure_{w}")[i])) for w in ("fl", "fr", "rl", "rr")}
        if c.dtype == "categorical" and t == "drive_mode": r.value = ["parked", "city driving", "motorway driving", "charging"][int(val)] if 0 <= int(val) <= 3 else str(val); r.unit = ""
        if t == "speed":
            gps = fr.column("gps_speed"); ref, flags = SQ.cross_check_speed(vals[ok][-30:], gps[ok][-30:] if gps is not None else None)
            r.value = ref; r.extra["gps"] = None if gps is None else float(gps[i]) if not np.isnan(gps[i]) else None; r.flags = [f.to_dict() for f in flags]
        r.extra["stale_s"] = stale
        fl = self._finish(r, fr, [t], now=self.ctx.now); r.flags = (r.flags or []) + [f for f in fl.flags if f not in (r.flags or [])]
        if stale is not None and stale > 300: r.flags.append({"code": "FRESHNESS", "severity": "major", "signal": t, "detail": f"the newest reading is {stale / 60:.0f} min old (the car is off)"}); r.quality = "low" if stale > 3600 else "medium"
        return r

    # ------------------------------------------------------------------ aggregate
    def _scope_values(self, q, scope, target, agg_samples=True):
        """Load frame + mask for a scope; -> (frame, mask, entities, label, period, error)."""
        intervals, ents, label, period = self._intervals(scope, None)
        if scope.select is not None and not intervals: return None, None, ents, label, period, "no trips matched the selection in that period"
        lo = min(a for a, _ in intervals); hi = max(b for _, b in intervals)
        need = self._concepts_for(q) | ({"drive_mode"} if self.p.concept_available("drive_mode") else set()); need |= self._acc_needs(target)
        tier = make_plan(q, self.p, self.ctx).tier
        if tier == "1s" and (hi - lo) > 6 * 3600: tier = "1m"
        fr = self.load_frame(need, lo, hi, tier)
        if len(fr) == 0: return None, None, ents, label, period, "there is no recorded data in that period"
        fr = fr.take(self._interval_mask(fr, intervals))                                         # only the samples inside the scope (several trips = several intervals)
        if len(fr) == 0: return None, None, ents, label, period, "there is no recorded data in that period"
        cm, err = self._cond_mask(fr, scope.conditions, tier)
        if cm is None: return None, None, ents, label, period, err
        return fr, cm, ents, label, period, ""

    def _acc_needs(self, target):
        if target in D.ACCUMULATED: return {"speed", "battery_power"} if target != "regen_energy" or not self.p.concept_available("regen_power") else {"regen_power", "speed"}
        return set()

    TRIP_WEATHER = {"cold": "avg_amb<5.0", "hot_weather": "avg_amb>28.0"}      # the same limits as signal_registry.PREDICATES, applied to a trip's average outside temperature

    def _trip_where(self, conditions):
        """SQL for conditions a whole trip can answer (weather: the trip's average outside temperature), or None when some condition needs the samples."""
        if "trips" not in self.p.tables or not conditions: return None
        parts = []
        for cd in conditions:
            w = self.TRIP_WEATHER.get(cd.predicate) if cd.signal is None else None
            if w is None: return None
            parts.append(w)
        return " AND ".join(parts)

    def _ledger(self, target, intervals, where=""):
        """Whole-trip accounting from the trip ledger (authoritative, matches the legacy answers) when no sample-level condition is involved; `where` = trip-level weather limits (_trip_where)."""
        if target not in ("energy_used", "distance_driven", "wh_per_km", "regen_energy"): return None        # regenerated energy comes from the trip ledger = the integral of the 1 Hz power (bench/regen_reconcile.py); one-minute averages cancel regeneration against traction and read ~15x low
        if "trips" not in self.p.tables: return None
        km = kwh = 0.0; regen = 0.0; n = 0
        for lo, hi in intervals:
            r = self.c.execute("SELECT COUNT(*), COALESCE(SUM(km),0), COALESCE(SUM(kwh),0) FROM trips WHERE start>=? AND start<? AND km>0.2" + (" AND " + where if where else ""), (int(lo), int(hi))).fetchone(); n += r[0]; km += r[1]; kwh += r[2]
            if target == "regen_energy" and "trip_ext" in self.p.tables:
                regen += self.c.execute("SELECT COALESCE(SUM(x.regen_kwh),0) FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<?" + (" AND " + where if where else ""), (int(lo), int(hi))).fetchone()[0]
        if n == 0: return None
        if target == "energy_used": return kwh, "kWh", n
        if target == "distance_driven": return km, "km", n
        if target == "wh_per_km": return (1000 * kwh / km, "Wh/km", n) if km > 0.5 else None
        if target == "regen_energy" and "trip_ext" in self.p.tables: return regen, "kWh", n
        return None

    def _value_over(self, q, scope, target, agg):
        """The single number for one scope. -> (Result-or-None, value, unit, frame, mask, extras) with a Result only when there is no data."""
        intervals, ents, label, period = self._intervals(scope, None)
        if scope.select is not None and not intervals: return self._no_data(q.op, "no trips matched the selection in that period", q, label, period), None
        tw = self._trip_where(scope.conditions) if scope.conditions else ""
        if target in D.ACCUMULATED and tw is not None:
            led = self._ledger(target, intervals, tw)
            if led is not None:
                r = Result(q.op, value=float(led[0]), unit=led[1], target=target, agg=agg or "", period=period, period_label=label, samples=led[2], tier="trip ledger", sources=["trips"], quality_kind="derived")
                r.extra["entities"] = ents; r.extra["trip_count"] = led[2]; return None, r
            if tw and "trips" in self.p.tables and target != "regen_energy":
                names = {"cold": "cold (an average outside temperature under 5 C)", "hot_weather": "hot (an average outside temperature over 28 C)"}
                return self._no_data(q.op, "no trip in that period was " + " and ".join(names[c.predicate] for c in scope.conditions), q, label, period), None
        fast = self._tier_value(q, scope, target, agg)
        if fast is not None: return None, fast
        fr, mask, ents, label, period, err = self._scope_values(q, scope, target)
        if fr is None: return self._no_data(q.op, err, q, label, period), None
        if int(mask.sum()) == 0: return self._no_data(q.op, "no samples satisfy those conditions in that period" if scope.conditions else "no samples in that period", q, label, period), None
        r = Result(q.op, target=target, agg=agg or "", period=period, period_label=label, extra={"entities": ents})
        if target in D.ACCUMULATED:
            acc = D.accumulate(target, fr, self.p, mask)
            if acc is None: return self._no_data(q.op, f"{R.label_of(target)} cannot be computed from the recorded signals in that period", q, label, period), None
            r.value, r.unit, r.quality_kind, r.via, r.assumptions, r.samples = acc[0], acc[1], acc[2], list(acc[3]), acc[4], acc[5]
            r.formula = "integrated from per-sample " + (" and ".join(R.label_of(v) for v in r.via) if r.via else "signals")                       # the signals this quantity really used (distance: speed only)
        else:
            vals = self._series(fr, target)
            if vals is None: return self._no_data(q.op, f"{R.label_of(target)} could not be computed from the recorded signals", q, label, period), None
            sel = vals[mask]; c = R.CONCEPTS[target]
            if agg in (None, "mean", "min", "max", "median", "std", "first", "last", "delta", "count", "p95", "sum"):
                v = self._agg(sel, agg or "mean")
                if v is None: return self._no_data(q.op, f"{R.label_of(target)} has no valid samples in that period", q, label, period), None
                r.value = float(v); r.unit = c.unit if agg != "count" else "samples"; r.samples = int((~np.isnan(sel)).sum())
            r.extra["_vals"] = sel; r.extra["_ts"] = fr.ts[mask]
            r = self._finish(r, fr, [target])
            if target in D.POINTWISE:
                ds = D.compute(target, fr, self.p);
                if ds is not None: r.quality_kind = ds.quality; r.formula = ds.formula; r.via = list(ds.via); r.assumptions = ds.assumptions
        if "tier" not in r.__dict__ or not r.tier: self._finish(r, fr, [target])
        return None, r

    def _aggregate(self, q):
        agg = q.agg or "mean"; bad, r = self._value_over(q, q.scope, q.target, agg)
        if bad is not None: return bad
        if q.unit and r.unit and R.unit_family(q.unit) == R.unit_family(r.unit): r.value = R.convert(r.value, r.unit, q.unit); r.unit = q.unit
        ents = r.extra.get("entities")                                                         # per-trip breakdown for selections
        if ents and len(ents) > 1: r.items = [{"start": e["start"], "end": e["end"], "wh_per_km": e.get("wh_per_km"), "km": e.get("km")} for e in ents[:20]]
        r.extra.pop("_vals", None); r.extra.pop("_ts", None)
        return r

    # ------------------------------------------------------------------ extreme
    def _extreme(self, q):
        agg = q.agg if q.agg in ("min", "max") else "max"
        fast = self._tier_extreme(q, agg)
        if fast is not None: return fast
        fr, mask, ents, label, period, err = self._scope_values(q, q.scope, q.target)
        if fr is None: return self._no_data("extreme", err, q, label, period)
        vals = self._series(fr, q.target)
        if vals is None: return self._no_data("extreme", f"{R.label_of(q.target)} could not be computed", q, label, period)
        v = np.where(mask & ~np.isnan(vals), vals, np.nan)
        if np.isnan(v).all(): return self._no_data("extreme", "no samples in that period" , q, label, period)
        i = int(np.nanargmax(v) if agg == "max" else np.nanargmin(v)); c = R.CONCEPTS[q.target]
        r = Result("extreme", value=float(v[i]), unit=c.unit, target=q.target, agg=agg, when=int(fr.ts[i]), period=period, period_label=label, samples=int(mask.sum()))
        r.extra["state"] = {k: float(fr.cols[k][i]) for k in ("speed", "battery_power", "ambient_temperature") if k in fr.cols and not np.isnan(fr.cols[k][i])}
        if q.unit and R.unit_family(q.unit) == R.unit_family(c.unit): r.value = R.convert(r.value, c.unit, q.unit); r.unit = q.unit
        return self._finish(r, fr, [q.target])

    # ------------------------------------------------------------------ day buckets (trend / compare significance)
    def _daily(self, q, scope, target, allow_tier=False):
        """[(day_start, value)] of the target per local day inside the scope (mean for signals, sum/ratio for accumulated metrics).  `allow_tier`: a long plain window reads the daily means from the daily tier (the trend)."""
        self._tier_hit = None; self._tier_kind = None
        if allow_tier:
            got = self._tier_days(q, scope, target, None)
            if got is not None:
                per, col, sc, off, period, label, tz = got
                out = [(int(d * DAY - tz), float(v[1]) / sc + off, int(v[0])) for d, v in sorted(per.items()) if v[0] >= 3]
                if out: self._tier_hit = col; return out, label, period, "", None
        if allow_tier and scope.conditions:
            got = self._daily_sql(q, scope, target)
            if got is not None: return got
        # For unrestricted complete local days, the trip ledger is the
        # authoritative accounting source for accumulated driving quantities.
        # It preserves trip-boundary energy and distance and therefore avoids
        # inventing/missing an interval when a downsampled telemetry tier has a
        # gap.  Partial days and conditioned/selected scopes still use samples.
        intervals, _, time_label, _ = self._intervals(scope, None)
        if (target in ("energy_used", "distance_driven", "wh_per_km") and not scope.conditions and scope.select is None
                and "trips" in self.p.tables and intervals
                and (scope.time.type == "all" or (max(b for _, b in intervals) - min(a for a, _ in intervals)) > 90 * DAY or          # a long window: partial first and last days matter less than minutes of loading per day
                     (min(a for a, _ in intervals) <= self.ctx.lo and max(b for _, b in intervals) >= self.ctx.hi) or
                     all((int(a) + self.ctx.tz) % DAY == 0 and (int(b) + self.ctx.tz) % DAY == 0 for a, b in intervals))):
            buckets = {}
            for a, b in intervals:
                rows = self.c.execute("SELECT start, km, kwh FROM trips WHERE start>=? AND start<? AND km>0.2 ORDER BY start", (int(a), int(b)))
                for start, km, kwh in rows:
                    try:
                        km = float(km)
                        if not math.isfinite(km) or km <= 0: continue
                    except (TypeError, ValueError):
                        continue
                    try:
                        kwh = float(kwh)
                        energy_ok = math.isfinite(kwh) and kwh >= 0
                    except (TypeError, ValueError):
                        energy_ok = False
                    # Do not turn an unavailable trip energy reading into
                    # zero.  Distance remains a valid independent ledger
                    # quantity; energy and Wh/km do not.
                    if target in ("energy_used", "wh_per_km") and not energy_ok:
                        continue
                    day = int(((int(start) + self.ctx.tz) // DAY) * DAY - self.ctx.tz)
                    x = buckets.setdefault(day, [0.0, 0.0, 0])
                    x[0] += km; x[1] += kwh if energy_ok else 0.0; x[2] += 1
            support_counts = {}
            if "sig_1m" in self.p.tables:
                for a, b in intervals:
                    for day_index, count in self.c.execute("SELECT CAST((ts + ?) / ? AS INTEGER), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? GROUP BY 1", (self.ctx.tz, DAY, int(a), int(b))):
                        day = int(day_index * DAY - self.ctx.tz)
                        support_counts[day] = support_counts.get(day, 0) + int(count)
            out = []
            for day in sorted(buckets):
                km, kwh, n = buckets[day]
                value = kwh if target == "energy_used" else km if target == "distance_driven" else (1000.0 * kwh / km if km > 0.5 else None)
                if value is not None: out.append((day, value, support_counts.get(day, n)))
            if out:
                lo = min(a for a, _ in intervals); hi = max(b for _, b in intervals)
                return out, time_label, (lo, hi), "", None
        fr, mask, ents, label, period, err = self._scope_values(q, scope, target)
        if fr is None: return None, label, period, err, None
        tz = self.ctx.tz; days = (fr.ts + tz) // DAY; out = []
        vals = None if target in D.ACCUMULATED else self._series(fr, target)
        if vals is None and target not in D.ACCUMULATED: return None, label, period, f"{R.label_of(target)} could not be computed", None
        for d in np.unique(days[mask]):
            m = mask & (days == d)
            if target in D.ACCUMULATED:
                acc = D.accumulate(target, fr, self.p, m);
                if acc is None: continue
                out.append((int(d * DAY - tz), float(acc[0]), int(m.sum())))
            else:
                v = vals[m]; v = v[~np.isnan(v)]
                if len(v) >= 3: out.append((int(d * DAY - tz), float(np.mean(v)), len(v)))
        return out, label, period, "", fr

    SQL_OPS = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "="}

    def _sql_condition(self, c):
        """One condition as SQL over the minute table `t` (sig_1m), or None when it needs the samples in Python.  Each atom is COALESCE(..., 0): a missing value is 'not true', as in the Python mask, so that `not` is the same complement."""
        if c.predicate and c.op == "not":
            inner = self._sql_condition(type(c)(predicate=c.predicate))
            return None if inner is None else f"NOT ({inner})"
        if c.predicate:
            rule = R.PREDICATES[c.predicate][1]; kind = rule[0]
            if kind == "mode": return f"COALESCE(t.mode = {int(rule[1])}, 0)"
            if kind == "mode_in": return "COALESCE(t.mode IN (" + ",".join(str(int(x)) for x in rule[1]) + "), 0)"
            if kind == "signal":
                e = self._sql_signal(rule[1])
                if e is None or rule[2] not in ("<", ">"): return None
                return f"COALESCE(({e} {rule[2]} {float(rule[3])!r}) AND t.{(R.STORE.get(rule[1]) or {})['1m'][1]} IS NOT NULL" + (" AND t.mode IN (1,2)" if len(rule) > 4 and rule[4] == "driving" else "") + ", 0)"
            return None
        e = self._sql_signal(c.signal)
        if e is None or c.op not in self.SQL_OPS: return None
        thr = float(c.value)
        if c.unit and R.unit_family(c.unit) == R.unit_family(R.CONCEPTS[c.signal].unit): thr = R.convert(thr, c.unit, R.CONCEPTS[c.signal].unit)
        return f"COALESCE(({e} {self.SQL_OPS[c.op]} {thr!r}) AND t.{R.STORE[c.signal]['1m'][1]} IS NOT NULL, 0)"

    def _sql_signal(self, concept):
        """The value of a measured column of the minute table in engineering units, as an SQL expression; None for anything else."""
        ent = (R.STORE.get(concept) or {}).get("1m")
        if not ent or ent[0] != "sig_1m" or ent[1] == "tp_*" or concept not in R.CONCEPTS: return None
        sc = float(ent[2]); off = float(ent[3]) if len(ent) > 3 else 0.0
        return f"(t.{ent[1]} * 1.0 / {sc!r} + {off!r})"

    def _daily_sql(self, q, scope, target):
        """The daily means of a measured quantity under conditions (hot weather, a driving mode, a threshold on another measured column), computed by SQLite over the minute table instead of loading millions of rows into Python: the same days, the same
        means (a day needs three values), a few hundred rows back.  None when the question needs anything the database cannot say (trip selections, events, derived or lake signals) - the Python path then runs as before."""
        if os.environ.get("VGPT_SQL_DAILY", "1") != "1" or scope.select is not None or target in D.ACCUMULATED or target not in R.CONCEPTS or R.CONCEPTS[target].dtype == "categorical": return None
        ent = (R.STORE.get(target) or {}).get("1m")
        if not ent or ent[0] != "sig_1m" or ent[1] == "tp_*" or "sig_1m" not in self.p.tables: return None
        try: intervals, _, label, period = self._intervals(scope, None)
        except Exception: return None
        if not intervals or max(b for _, b in intervals) - min(a for a, _ in intervals) < 6 * 3600: return None             # a short window is read from the 1 Hz tier as before
        parts = [self._sql_condition(c) for c in scope.conditions]
        if any(p is None for p in parts): return None
        sc = float(ent[2]); off = float(ent[3]) if len(ent) > 3 else 0.0; tz = int(self.ctx.tz); per = {}
        for a, b in intervals:
            sql = f"SELECT CAST((t.ts + {tz}) / {DAY} AS INTEGER) d, AVG(t.{ent[1]}), COUNT(t.{ent[1]}) FROM sig_1m t WHERE t.ts>=? AND t.ts<? AND " + " AND ".join(parts) + " GROUP BY d"
            assert_read_only(sql)
            for d, av, n in self.c.execute(sql, (int(a), int(b))):
                if av is not None and n:
                    k = int(d); x = per.setdefault(k, [0.0, 0]); x[0] += float(av) * n; x[1] += int(n)
        out = [(int(d * DAY - tz), x[0] / x[1] / sc + off, x[1]) for d, x in sorted(per.items()) if x[1] >= 3]
        if not out: return None
        self._tier_hit = ent[1]; self._tier_kind = "per-minute rows, aggregated in the database"
        return out, label, period, "", None

    def _trend(self, q):
        out, label, period, err, fr = self._daily(q, q.scope, q.target, allow_tier=True)
        if out is None: return self._no_data("trend", err, q, label, period)
        if len(out) < 4: return self._no_data("trend", f"only {len(out)} day(s) of data - too few to judge a trend", q, label, period)
        xs = [(d - out[0][0]) / DAY for d, _, _ in out]; ys = [v for _, v, _ in out]; st = Q.slope_test(xs, ys); k = max(2, len(ys) // 4)
        a, b = float(np.mean(ys[:k])), float(np.mean(ys[-k:])); c = R.CONCEPTS[q.target]
        r = Result("trend", value=float(st["slope"]) if st else 0.0, unit=f"{c.unit}/day", target=q.target, period=period, period_label=label, samples=sum(n for *_, n in out), items=[{"day": d, "value": v} for d, v, _ in out],
                   extra={"first": a, "last": b, "change": b - a, "pct": (100 * (b - a) / a) if abs(a) > 1e-9 and R.unit_family(c.unit) != "temperature" else None, "p": st["p"] if st else None, "r2": st["r2"] if st else None, "days": len(out), "significant": bool(st and st["p"] < 0.01 and abs(b - a) > 0.02 * max(abs(a), 1e-9))})
        r = self._finish(r, fr, [q.target])
        if getattr(self, "_tier_hit", None):
            r.tier = getattr(self, "_tier_kind", None) or "daily tier"; r.sources = [f"{R.STORE.get(q.target, {}).get('1m', ['sig_1m'])[0]}.{self._tier_hit} ({r.tier})"]; self._tier_kind = None; self._tier_hit = None
        return r

    # ------------------------------------------------------------------ compare
    def _compare(self, q):
        target = q.target; groups = []
        if q.scope_b is not None: scopes = [("A", q.scope), ("B", q.scope_b)]
        else:
            from .query_ir import EVENT_PAGE, Condition, Scope
            scopes = [(g, Scope(q.scope.time, q.scope.select, list(q.scope.conditions) + [Condition(predicate=g[4:], op="not") if g.startswith("not_") else Condition(predicate=g)])) for g in q.groups]
        vals = []; daily = []
        for name, sc in scopes:
            bad, r = self._value_over(q, sc, target, q.agg or "mean")
            if bad is not None: return self._no_data("compare", bad.reason if name in ("A", "B") else f"{name}: {bad.reason}", q, bad.period_label, bad.period)
            vals.append((name, r)); d, *_ = self._daily(q, sc, target); daily.append([v for _, v, _ in (d or [])])
        a, b = vals[0][1].value, vals[1][1].value; c = R.CONCEPTS[target]
        r0 = vals[0][1]; res = Result("compare", value=float(b - a), unit=c.unit, target=target, agg=q.agg or "mean", period=r0.period, period_label=r0.period_label, samples=sum(v[1].samples for v in vals), quality=min((v[1].quality for v in vals), key=["high", "medium", "low"].index) if False else r0.quality,
                                      items=[{"group": n, "value": r.value, "unit": r.unit, "samples": r.samples, "period": r.period_label, "quality_kind": r.quality_kind} for n, r in vals], tier=r0.tier, sources=r0.sources, quality_kind=r0.quality_kind, via=r0.via, formula=r0.formula, assumptions=r0.assumptions)
        res.unit = vals[0][1].unit or c.unit
        res.extra = {"a": a, "b": b, "ratio_pct": (100 * (b - a) / a) if abs(a) > 1e-9 and R.unit_family(c.unit) != "temperature" else None, "labels": [v[0] for v in vals], "periods": [v[1].period_label for v in vals]}
        w = Q.welch(daily[0], daily[1]) if len(daily[0]) >= 3 and len(daily[1]) >= 3 else None
        if w: res.extra["p"] = w[2]; res.extra["days"] = (len(daily[0]), len(daily[1]))
        res.flags = [f for v in vals for f in v[1].flags]; res.quality = "low" if any(v[1].quality == "low" for v in vals) else "medium" if any(v[1].quality == "medium" for v in vals) else "high"
        return res

    # ------------------------------------------------------------------ count / percent time
    def _events_under_conditions(self, rows, q, lo, hi, etype):
        """Events whose moment satisfies the question's sample-level conditions ("hard brakes in city driving", "faults excluding charging"). -> (rows, error)."""
        import copy
        conds = [c for c in q.scope.conditions if not (c.predicate and R.EVENT_AS_PREDICATE.get(etype) == c.predicate and c.op != "not")]
        if not conds or not rows: return rows, ""
        qq = copy.deepcopy(q); qq.op = "aggregate"; qq.target = "speed"; qq.event = None; qq.scope.conditions = conds
        fr = self.load_frame(self._concepts_for(qq) | ({"drive_mode"} if self.p.concept_available("drive_mode") else set()) | {"speed"}, lo, hi, "1m")
        # An empty telemetry frame means that the requested condition could not
        # be evaluated.  It is not evidence that no event matched it.
        if len(fr) == 0: return None, "there is no recorded data to evaluate the event condition"
        cm, err = self._cond_mask(fr, conds, "1m")
        if cm is None: return None, err
        keep = []
        for r in rows:
            ts = int(r["ts"]) if isinstance(r, dict) else int(r[0])
            i = int(np.searchsorted(fr.ts, ts, side="right")) - 1
            if 0 <= i < len(fr.ts) and ts < fr.ts[i] + max(1, int(fr.res)) and cm[i]: keep.append(r)
        return keep, ""

    def _count(self, q):
        lo, hi, label = resolve_time(q.scope.time, self.ctx)
        if q.event is not None:
            if q.event.type in R.DETECTED_EVENTS:
                rows = self._detect(q.event.type, lo, hi)
            else:
                sql, args = compile_events([q.event.type], lo, hi, "asc"); rows = self._rows(sql, args)
            rows, err = self._events_under_conditions(rows, q, lo, hi, q.event.type)
            if rows is None: return self._no_data("count", err, q, label, (lo, hi))
            res = Result("count", value=len(rows), unit="count", target=q.event.type, period=(lo, hi), period_label=label, samples=len(rows), tier="events", sources=["events"], items=[r if isinstance(r, dict) else {"ts": r[0], "type": r[1], "detail": r[3]} for r in rows[-10:]])
            if not rows and q.event.type not in R.DETECTED_EVENTS and lo > 0:                                                  # none in this period: the latest earlier one is worth a line (a lamp that looks at 24 h and a question about the week must not seem to disagree)
                try:
                    sql, args = compile_events([q.event.type], 0, lo, "desc", 1); prev = self._rows(sql, args)
                    if prev: res.extra["latest_before"] = {"ts": int(prev[0][0]), "detail": prev[0][3], "before_s": int(lo - prev[0][0])}
                except Exception: pass
            return res
        if q.scope.select is not None:
            ints, ents, label, period = self._intervals(q.scope, None, cap=None)                                  # counting needs no signal frame: every trip of a multi-year history is counted, not the first 200
            return Result("count", value=len(ents or []), unit="count", target=q.scope.select.entity, period=period, period_label=label, samples=len(ents or []), tier="trip ledger", sources=["trips"])
        fr, mask, ents, label, period, err = self._scope_values(q, q.scope, q.target)
        if fr is None: return self._no_data("count", err, q, label, period)
        edges = int(np.sum(mask[1:] & ~mask[:-1]) + (1 if mask[:1].any() else 0))
        return self._finish(Result("count", value=edges, unit="count", target=q.target or "", period=period, period_label=label, samples=int(mask.sum()), extra={"episodes": True}), fr, [])

    def _percent_time(self, q):
        scope_nocond = type(q.scope)(q.scope.time, q.scope.select, [])
        fr, mask_all, ents, label, period, err = self._scope_values(q, scope_nocond, q.target)
        if fr is None: return self._no_data("percent_time", err, q, label, period)
        cm, err = self._cond_mask(fr, q.scope.conditions, "1m")
        if cm is None: return self._no_data("percent_time", err, q, label, period)
        mode = fr.column("drive_mode"); mode_pred = any(c.predicate in ("charging", "parked", "driving", "highway", "city") for c in q.scope.conditions)
        if mode is None and not mode_pred:
            return self._no_data("percent_time", "driving-mode samples are required to calculate a share of driving time", q, label, period)
        denom = mask_all & (np.isin(mode, (1, 2)) if not mode_pred else True)
        n = int(denom.sum())
        if n == 0: return self._no_data("percent_time", "no driving samples in that period", q, label, period)
        pct = 100.0 * float((denom & cm).sum()) / n
        return self._finish(Result("percent_time", value=pct, unit="%", target=q.target or "", period=period, period_label=label, samples=n, extra={"denominator": "driving time" if not mode_pred else "all recorded time"}), fr, [])

    # ------------------------------------------------------------------ events
    def _events(self, q):
        e = q.event; lo, hi, label = resolve_time(q.scope.time, self.ctx)
        if e.type in R.DETECTED_EVENTS:
            rows = self._detect(e.type, lo, hi)
        else:
            sql, args = compile_events([e.type], lo, hi, "desc" if e.selector != "first" else "asc"); rows = self._rows(sql, args)
        rows, err = self._events_under_conditions(rows, q, lo, hi, e.type)
        if rows is None: return None, label, (lo, hi), err
        items = [r if isinstance(r, dict) else {"ts": r[0], "type": r[1], "sev": r[2], "detail": r[3]} for r in rows]
        if e.selector == "all" and e.n: items = items[: int(e.n)]                                  # the n newest (rows are newest first)
        if e.selector == "most_severe":
            severity = {"critical": 3, "warn": 2, "warning": 2, "info": 1}
            items.sort(key=lambda x: (severity.get(str(x.get("sev", "")).lower(), 0), x["ts"]), reverse=True)
        return items, label, (lo, hi), ""

    def _detect(self, etype, lo, hi):
        """Events found in the signals themselves. COOLANT_FLOW_DROP: first minute from which coolant flow stays below 75% of this vehicle's normal for 30+ minutes while the pump should be running."""
        b = Q.flow_baseline(self.c, self.ctx.now)
        if not b: return []
        rows = self._rows("SELECT ts, flow, mode FROM sig_1m WHERE ts>=? AND ts<? ORDER BY ts", (int(lo), int(hi)))
        thr = 0.75 * b["p50"] * 10; run = []; out = []; previous = None
        for ts, fl, mode in rows:
            if previous is not None and ts - previous > 60: run = []
            previous = ts
            if fl is not None and mode in (1, 2, 3) and 30 < fl < thr: run.append(ts)
            else: run = []
            if len(run) >= 30 and not out: out.append({"ts": run[0], "type": etype, "sev": "info", "detail": f"coolant flow fell below 75% of its normal {b['p50']:.1f} L/min"})
        return out

    def _list_events(self, q):
        evs, label, period, err = self._events(q)
        if err: return self._no_data("list_events", err, q, label, period)
        if not evs: return Result("list_events", status="ok", value=0, unit="count", target=q.event.type, period=period, period_label=label, tier="events", sources=["events"], items=[], reason="none recorded")
        sel = q.event.selector; PG = EVENT_PAGE; k = max(1, q.event.n or 1)
        items = evs[:1] if sel in ("last", "first", "most_severe") else evs[(q.event.n or 1) - 1:(q.event.n or 1)] if sel == "nth" else evs[(k - 1) * PG:k * PG] if sel == "page" else evs[:PG]
        if sel == "nth" and not items:
            return self._no_data("list_events", f"only {len(evs)} matching event(s) were recorded; event number {q.event.n} does not exist", q, label, period)
        r = Result("list_events", value=len(evs), unit="count", target=q.event.type, period=period, period_label=label, tier="events", sources=["events"], items=items, samples=len(evs))
        if sel in ("all", "page"): first = ((k - 1) * PG if sel == "page" else 0); r.extra["page"] = {"first": first + 1 if items else 0, "last": first + len(items), "size": PG, "total": len(evs)}
        return r

    def _event_context(self, q):
        evs, label, period, err = self._events(q)
        if err: return self._no_data("event_context", err, q, label, period)
        if not evs: return self._no_data("event_context", f"no {R.EVENTS[q.event.type][0]} was recorded", q, label, period)
        sel = q.event.selector
        if sel == "nth" and len(evs) < (q.event.n or 1): return self._no_data("event_context", f"only {len(evs)} matching event(s) were recorded; event number {q.event.n} does not exist", q, label, period)
        ev = evs[(q.event.n or 1) - 1] if sel == "nth" else evs[0]; ts = int(ev["ts"]); before = q.event.before or 60; after = q.event.after
        names = [q.target] if q.target else [n for n in DEFAULT_CONTEXT if self.p.concept_available(n)]
        need = set()
        from .query_planner import measured_needs
        for n in names: need |= measured_needs(n, self.p)
        fr = self.load_frame(need, ts - before, ts + after + 1, "1s")
        sourcename = None
        if len(fr) < 5 and before >= 120:                                                         # older than the 1 Hz tier: per-minute samples if the window is long enough
            fr = self.load_frame(need, ts - before, ts + after + 1, "1m")
        if len(fr) < 5:                                                                         # frozen 1 Hz context kept for every warning, even after the 7-day tier is gone
            sg = Q.event_sig(self.c, ts, before, after)
            if len(sg) >= 5:
                legacy = {"speed": "speed", "battery_soc": "soc", "battery_power": "power", "battery_temperature": "batt", "coolant_flow": "flow", "ambient_temperature": "amb", "hvac_power": "hvac", "motor_temperature": "motor", "drive_mode": "mode"}
                fr = Frame(np.array(sg.ts, dtype=np.int64), {k: np.array(sg.cols[v], dtype=float) for k, v in legacy.items() if v in sg.cols}, 1 if sg.res == 1 else 60, "1s" if sg.res == 1 else "1m", {k: f"{sg.tier}" for k in legacy}, [(ts - before, ts + after + 1)]); sourcename = sg.tier
        if not after and len(fr): fr = fr.take(fr.ts + fr.res <= ts)                                                 # "before the event": the event's own sample (or per-minute bucket) is the anchor and is not part of the figures
        if len(fr) < 3: return self._no_data("event_context", "there are no signal samples around that moment", q, label, (ts - before, ts + after))
        summ = []
        for n in names:
            v = self._series(fr, n)
            if v is None: continue
            ok = v[~np.isnan(v)]
            if len(ok) == 0: continue
            c = R.CONCEPTS[n]; summ.append({"signal": n, "label": c.label, "unit": c.unit, "first": float(ok[0]), "last": float(ok[-1]), "min": float(ok.min()), "max": float(ok.max()), "mean": float(ok.mean()), "change": float(ok[-1] - ok[0])})
        sqlE, argsE = compile_events(["ANY_WARNING", "HARSH_BRAKE", "HARSH_ACCEL", "ADAS_DISENGAGE", "CHARGE_START"], ts - before, ts + after + 1, "asc")
        near = [{"ts": r[0], "type": r[1], "detail": r[3]} for r in self._rows(sqlE, argsE) if r[0] != ts or r[1] != ev["type"]]
        r = Result("event_context", value=len(summ), unit="signals", target=q.target or "", when=ts, period=(ts - before, ts + after), period_label=f"{before} s before the event" + (f" and {after} s after" if after else ""), items=summ,
                   extra={"event": ev, "event_label": R.EVENTS[q.event.type][0], "window": (before, after), "near_events": near, "frozen_context": bool(sourcename), "series": {"t": (fr.ts - ts).tolist(), **{n: self._series(fr, n).tolist() for n in names if self._series(fr, n) is not None}}})
        return self._finish(r, fr, names)

    # ------------------------------------------------------------------ rank / correlation / explain / capability
    def _rank(self, q):
        intervals, ents, label, period = self._intervals(q.scope, None)
        if not ents: return self._no_data("rank", "nothing to rank in that period", q, label, period)
        e = q.scope.select; mcol, unit, _ = R.ENTITY_METRICS[e.entity][e.rank_by]
        key = {"wh_per_km": "wh_per_km", "distance_driven": "km", "energy_used": "energy_used" if e.entity == "charge_session" else "kwh", "avg_speed": "avg_speed", "ambient_temperature": "avg_amb", "hvac_power": "avg_hvac_w", "charging_power": "charging_power"}.get(e.rank_by)
        items = [{"start": x["start"], "end": x["end"], "value": (x.get(key) if key else x["end"] - x["start"]), "unit": unit, **{k: v for k, v in x.items() if k in ("km", "kwh", "avg_speed", "hwy_share")}} for x in ents]
        return Result("rank", value=len(items), unit="count", target=e.rank_by, period=period, period_label=label, items=items, samples=len(items), tier="trip ledger", sources=[R.ENTITY_TABLE[e.entity][0]], extra={"order": e.order, "entity": e.entity, "klass": e.klass})

    def _correlation(self, q):
        t1, t2 = q.target, q.target2; fr, mask, ents, label, period, err = self._scope_values(q, q.scope, t1)
        if fr is None: return self._no_data("correlation", err, q, label, period)
        a, b = self._series(fr, t1), self._series(fr, t2)
        if a is None or b is None: return self._no_data("correlation", "one of the signals could not be computed", q, label, period)
        ok = mask & ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 20: return self._no_data("correlation", f"only {int(ok.sum())} paired samples", q, label, period)
        x, y = a[ok].tolist(), b[ok].tolist()
        if len(x) > 4000: step = len(x) // 4000 + 1; x, y = x[::step], y[::step]
        pr, sp = Q.pearson(x, y), Q.spearman(x, y)
        if pr is None: return self._no_data("correlation", "one of the signals does not vary in that period", q, label, period)
        r = Result("correlation", value=float(pr), unit="r", target=t1, period=period, period_label=label, samples=int(ok.sum()), extra={"spearman": sp, "target2": t2, "samples_used": len(x)})
        return self._finish(r, fr, [t1, t2])

    def _explain(self, q):
        if self.legacy is None: return Result("explain_event", status="unsupported", reason="the causal-analysis engine is not attached")
        evs, label, period, err = self._events(q)
        if err: return self._no_data("explain_event", err, q, label, period)
        if not evs: return self._no_data("explain_event", f"no {R.EVENTS[q.event.type][0]} was recorded", q, label, period)
        if q.event.selector == "nth" and len(evs) < (q.event.n or 1): return self._no_data("explain_event", f"only {len(evs)} matching event(s) were recorded; event number {q.event.n} does not exist", q, label, period)
        e = evs[(q.event.n or 1) - 1] if q.event.selector == "nth" else evs[0]; row = (e["ts"], e["type"], e.get("sev", ""), e.get("detail", ""))
        d = self.legacy.diag_event(row)
        if d is None: return self._no_data("explain_event", "there is not enough signal data around that event to diagnose it", q, label, period)
        return Result("explain_event", value=d["alts"][0]["name"] if d.get("alts") else "", target=q.event.type, when=int(e["ts"]), period=period, period_label=label, tier="1 Hz event context", sources=["event_ctx", "sig_1s/sig_1m", "events"], extra={"diag": d, "event": e}, quality=("high" if d.get("confidence") == "high" else "medium"))

    def _explain_change(self, q):
        from . import change_ops
        return change_ops.explain_change(self, q)

    def _detect_change(self, q):
        from . import change_ops
        return change_ops.detect_change(self, q)

    def _capability(self, q):
        lab = R.label_of(q.target) if q.target else ""
        from .capability import describe_availability
        return Result("capability", value=describe_availability(self.p, q.target) if q.target else "", target=q.target or "", period_label="", samples=0, tier="registry", sources=["signal registry"])



_SPAN = {}


def _span(conn, now):
    """The history span (first/last second of each tier, the archive's range).  Reading it scans the ends of the big tables (70 ms on the Pi) and the assistant asks several times per question: the last answer is kept while the database is unchanged (PRAGMA data_version) and `now` is the same.  One slot, for the connection asked last."""
    from . import archive
    try: key = conn.execute("PRAGMA data_version").fetchone()[0]                                    # the span depends on the data, not on the clock: only its newest end is cut at `now`
    except sqlite3.Error: return Q.coverage(conn, now), archive.coverage(conn)
    hit = _SPAN.get("last")
    if not (hit and hit[0] is conn and hit[1] == key):
        hit = _SPAN["last"] = (conn, key, Q.coverage(conn, 2 ** 40), archive.coverage(conn))
    cv = dict(hit[2])
    if cv.get("hi"): cv["hi"] = min(cv["hi"], now)
    return cv, hit[3]


def context_for(conn, now=None):
    """Build the planner context (history span, tz, 1 Hz tier start) for a connection."""
    tz = 0
    try: tz = int(float(dict(conn.execute("SELECT k, v FROM meta").fetchall()).get("env.tz_h", 0)) * 3600)
    except sqlite3.Error: pass
    from . import archive
    cv, ar = _span(conn, now or 2 ** 40)
    return Ctx(now=int(now if now is not None else cv["hi"] or 0), tz=tz, lo=min(cv["lo"] or 0, ar[0]) if ar and cv["lo"] else (ar[0] if ar else cv["lo"] or 0), hi=cv["hi"] or 0, hz_lo=cv["hz_lo"], hz_hi=cv["hz_hi"], arch_lo=ar[0] if ar else None, arch_hi=ar[1] if ar else None)
