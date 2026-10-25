# VERBATIM COPY of vgpt/meaning/digest.py before the daily-tier read (reference for tests/test_digest_tier.py)
"""Digest executors: the kinds of answer a person asks a vehicle for when the question is not one number.

    summary  <none | topic | quantity | event>   [op change]  [ent trip | day]   what happened in the period, in a few lines: totals, how it compares with the period before, what stands out
    anomaly  <none | topic | quantity | event>                [ent trip | day]   readings and events that are unusual for THIS vehicle (its own history is the yardstick)
    stat     <quantity | event> usual                                          how the quantity usually behaves: the normal range, the pattern, where the latest value sits
    why      <quantity | event> ... ent trip | day                             why one trip / day stood out: how it differed from the usual ones (associations, never claimed causes)
    why      <energy level> [cond parked | night]                              why the battery / fuel level fell while the vehicle stood: the loss over the parked stretch against the usual loss, and what ran

Every number is recomputed from the vehicle's own tables (trips, events, the one-minute tier and the lake views) by plain SQL and robust statistics; nothing here reads the question's text.  The period and the trip
chosen ("last", "longest") arrive as typed values from `literals.py`.  What is said depends on what the vehicle records: an electric vehicle reports energy, a combustion vehicle fuel, and a quantity the vehicle
does not have is never mentioned.  Quantities are read while the vehicle was driving unless they do not depend on it (outside temperature, 12 V voltage).  Each answer ends with what it is based on, so "how did
you get that" has an answer."""
import statistics

import numpy as np

from .. import analytics as A
from .. import signal_registry as R
from ..investigate.primitives import Data, effect, robust_z
from ..plan_exec import resolve_time
from ..query_ir import TimeSpec
from . import fuelx as FX

DAY = 86400
NOISE = ("HARSH_BRAKE", "HARSH_ACCEL", "HARSH_CORNER", "CHARGE_START", "SW_UPDATE", "SERVICE_DUE", "CAMERA_RESTORED", "TYRE_INFLATED", "ADAS_STATE_CHANGE", "DTC_CLEAR")
STATES = {"drive_mode", "charging_state", "adas_state", "rain", "light", "heading", "location", "lane_confidence", "camera_blockage", "camera_sharpness", "camera_brightness", "gps_speed", "gps_speed_valid", "gps_hdop",
          "odometer", "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "battery_soc", "fuel_level", "range_estimate",
          "brake_pad_life", "dpf_regeneration", "reported_range"}
DRIVER_INPUT = {"speed", "brake_pedal", "brake_pressure", "accelerator_position", "steering_angle", "steering_rate", "motor_rpm", "motor_torque", "engine_rpm", "engine_torque", "battery_power", "motor_power", "hvac_power",
                "longitudinal_acceleration", "lateral_acceleration", "yaw_rate", "engine_load", "fuel_rate", "boost_pressure", "regen_power", "charging_power"}      # what the driver and the road ask for: a change in them is behaviour, not a fault
ANY_TIME = {"ambient_temperature", "aux_battery_voltage", "barometric_pressure"}              # not tied to driving: read all day
MIN_REL = 0.04                                      # a change smaller than this share of the usual value is not reported as a change, however steady
BASELINE_DAYS = 56


# ------------------------------------------------------------------------------------------------------------------ context and small helpers
class Ctx:
    def __init__(self, res, c, frame, lit, text):
        b = res.brain; self.res, self.b, self.c, self.frame, self.lit, self.text = res, b, c, frame, lit, text; self.D = Data(b); self.qc = b.qc(); A.set_tz(b.tz); self.cls = FX.caps_of(res).get("energy") or "unknown"
        self.steps = []; self.tz = int(b.tz); self.now = int(b.now); self._daily = {}; self._trips = {}
        default = TimeSpec("relative", 7, "day") if frame.act != "stat" else TimeSpec("relative", 28, "day")
        self.lo, self.hi, self.label = resolve_time(lit.times[0] if lit.times else default, self.qc.ctx); self.hi = min(int(self.hi), self.now + 1); self.lo = int(self.lo)
        span = max(DAY, self.hi - self.lo); self.plo, self.phi = self.lo - span, self.lo

    def step(self, s):
        if s not in self.steps: self.steps.append(s)

    def date(self, ts):
        d = A._lt(int(ts)); return f"{d.day} {d:%b}"

    def stamp(self, ts):
        d = A._lt(int(ts)); return f"{d.day} {d:%b}, {d:%H:%M}"

    def trips(self, lo, hi):
        k = (int(lo), int(hi))
        if k not in self._trips: self._trips[k] = self.D.trips(lo, hi, min_km=0.3)
        return self._trips[k]

    def label_of(self, concept): return self.D.label(concept)

    def daily(self, concept, agg="AVG", all_day=False):
        """{day: value} of the one-minute tier over the baseline and the period (read once): while driving unless the quantity does not depend on it (or `all_day`).  None when the vehicle does not store it."""
        key = (concept, agg, all_day)
        if key in self._daily: return self._daily[key]
        ent = (R.STORE.get(concept) or {}).get("1m"); out = None
        if ent and ent[0] in self.D.have:
            self._prefetch(ent[0], agg, all_day)                                                        # one scan of the table gives every quantity stored there (a digest asks for 30-40 of them)
            if key in self._daily and self._daily[key] is not None: return self._daily[key]
            table, col, scale = ent; col = ("vmax" if agg == "MAX" else "vmin" if agg == "MIN" else col) if table.startswith("lkv_") else col
            lo = min(self.plo, self.lo - BASELINE_DAYS * DAY); fn = {"AVG": "AVG", "MAX": "MAX", "MIN": "MIN"}[agg]
            for masked in (((concept not in ANY_TIME) and not all_day), False):
                join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""; where = " AND s.mode IN (1,2)" if masked and table != "sig_1m" else (" AND t.mode IN (1,2)" if masked else "")
                try:
                    rows = self.b.c.execute(f"SELECT CAST((t.ts + ?) / 86400 AS INTEGER) d, {fn}(t.{col}), COUNT(*) FROM {table} t{join} WHERE t.ts>=? AND t.ts<? AND t.{col} IS NOT NULL{where} GROUP BY d HAVING COUNT(*) >= ?",
                                            (self.tz, int(lo), int(self.hi), 5)).fetchall()
                except Exception: rows = []
                if rows: out = {int(d): float(v) / scale for d, v, n in rows}; break
        self._daily[key] = out; return out

    def _prefetch(self, table, agg, all_day):
        """The same per-day aggregate `daily` computes for one quantity, for all quantities of `table` in a single query (same filters, same thresholds).  A quantity this does not fill is left to `daily`, which then does exactly what it always did."""
        done = self.__dict__.setdefault("_pre", set())
        if (table, agg, all_day) in done: return
        done.add((table, agg, all_day)); fn = {"AVG": "AVG", "MAX": "MAX", "MIN": "MIN"}[agg]; items = []
        for c, tiers in R.STORE.items():
            ent = (tiers or {}).get("1m")
            if not ent or ent[0] != table: continue
            col = ("vmax" if agg == "MAX" else "vmin" if agg == "MIN" else ent[1]) if table.startswith("lkv_") else ent[1]; items.append((c, col, ent[2]))
        masked_of = lambda c: (c not in ANY_TIME) and not all_day
        for masked in (True, False):
            group = [(c, col, sc) for c, col, sc in items if masked_of(c) == masked and (c, agg, all_day) not in self._daily]
            if not group: continue
            join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""; where = (" AND s.mode IN (1,2)" if table != "sig_1m" else " AND t.mode IN (1,2)") if masked else ""
            cols = sorted({col for _, col, _ in group}); lo = min(self.plo, self.lo - BASELINE_DAYS * DAY)
            try:
                sel = ", ".join(f"{fn}(t.{c}), COUNT(t.{c})" for c in cols)
                rows = self.b.c.execute(f"SELECT CAST((t.ts + ?) / 86400 AS INTEGER) d, {sel} FROM {table} t{join} WHERE t.ts>=? AND t.ts<?{where} GROUP BY d", (self.tz, int(lo), int(self.hi))).fetchall()
            except Exception: continue
            for c, col, sc in group:
                j = cols.index(col); out = {int(r[0]): float(r[1 + 2 * j]) / sc for r in rows if r[1 + 2 * j] is not None and r[2 + 2 * j] >= 5}
                if out: self._daily[(c, agg, all_day)] = out

    def window(self, d, lo, hi):
        a, b_ = (lo + self.tz) // DAY, (hi - 1 + self.tz) // DAY; return {k: v for k, v in (d or {}).items() if a <= k <= b_}


def _num(v, nd=1):
    v = float(v)
    if round(v, nd) == 0: return "0"                                                                    # never "-0"
    if abs(v) >= 1000 or nd == 0: return f"{v:,.0f}"
    s = f"{v:,.{nd}f}"; return s.rstrip("0").rstrip(".") if "." in s else s


def _unit(u): return "" if u in ("none", "category", "", None) else (" " + ("°C" if u == "C" else u))


def _dur(sec):
    m = int(round(sec / 60.0)); return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _vs(new, old, more="higher", less="lower", noun=""):
    """'5% higher than', 'about the same as', '10% more distance than': how a value compares with the one before, for use as '<this> the period before'."""
    if not old: return None
    p = 100.0 * (new - old) / abs(old)
    return "about the same as" if abs(p) < 3 else f"{abs(p):.0f}% {more if p > 0 else less}{(' ' + noun) if noun else ''} than"


def _in(label):
    """'in the last 7 days', but 'this week' / 'yesterday' / 'last week' as they are."""
    return "in " + label if label.startswith("the ") else label


NEXT = {"summary": ["What changed since last week?", "Anything unusual this week?", "Tell me about my longest trip"], "changes": ["Summarise the last week", "Anything unusual this week?"],
        "anomaly": ["What changed since last week?", "Summarise the last week"], "trip": ["Summarise the last week", "Anything unusual this week?"], "usual": ["Summarise the last week", "Anything unusual this week?"],
        "parked": ["Summarise the last week", "How is the battery?"]}                                       # what a person usually asks next, shown as buttons under the answer


def _result(cx, text, route, **facts):
    from .resolve import _result as R_
    text = text.rstrip() + "\nBased on " + ", ".join(cx.steps[:3]) + ".\nEvidence: computed from the vehicle's recorded measurements and events."
    r = R_(text, route, "digest", cx.frame, **facts); r["facts"]["steps"] = list(cx.steps); r["facts"]["window"] = [cx.lo, cx.hi]
    kind = route.split(":")[1] if ":" in route else ""; r["facts"]["suggest"] = [x for x in NEXT.get(kind, []) if x.lower().rstrip("?") not in (getattr(cx, "text", "") or "").lower()][:3] or None; return r


def _gap(cx, text, route, **facts):
    from .resolve import _result as R_
    return R_(text, route, "digest", cx.frame, intent="CLARIFY", ok=False, **facts)


def _event_label(etype): return R.EVENTS[etype][0] if etype in R.EVENTS else etype.replace("_", " ").lower()


def _warning_events(cx, lo, hi, types=None):
    rows = [(ts, ty, det) for ts, ty, det in cx.b.c.execute("SELECT ts, type, detail FROM events WHERE ts>=? AND ts<? ORDER BY ts", (int(lo), int(hi))).fetchall() if ty not in NOISE]
    return [r for r in rows if not types or r[1] in types]


def _counts(rows):
    out = {}
    for ts, ty, det in rows: out[ty] = out.get(ty, 0) + 1
    return out


def _roster(cx, behaviour=True):
    """The measured quantities worth scanning on this vehicle: stored at one-minute resolution, numeric, not a state or a per-wheel duplicate; `behaviour=False` leaves out what the driver and the road ask for."""
    out = []
    for name, c in R.CONCEPTS.items():
        if name in STATES or c.unit in ("category", "none", "") or c.kind == "derived" or (not behaviour and name in DRIVER_INPUT): continue
        if "1m" not in (R.STORE.get(name) or {}) or not cx.D.available(name): continue
        out.append(name)
    return out[:30]


# ------------------------------------------------------------------------------------------------------------------ the period digest
def _energy_line(cx, trips, ptrips):
    km = sum(t["km"] for t in trips)
    if not km: return None
    if cx.cls in ("ice", "hybrid"):
        e = FX.economy(cx.b, cx.lo, cx.hi); pe = FX.economy(cx.b, cx.plo, cx.phi)
        if not e: return None
        cx.step("the fuel-rate signal against the distance driven"); c = _vs(e["l100"], pe["l100"]) if pe else None
        return f"Fuel: {_num(e['litres'])} L for {_num(e['km'], 0)} km, {_num(e['l100'])} L/100 km" + (f" ({c} the period before)" if c else "") + "."
    kwh = sum(t["kwh"] or 0 for t in trips)
    if kwh <= 0: return None
    pk = sum(t["km"] for t in ptrips); pkwh = sum(t["kwh"] or 0 for t in ptrips); cx.step("the energy and distance of each of those trips"); c = _vs(1000.0 * kwh / km, 1000.0 * pkwh / pk) if pk > 5 and pkwh > 0 else None
    return f"Energy: {_num(kwh)} kWh used, {_num(1000.0 * kwh / km, 0)} Wh/km" + (f" ({c} the period before)" if c else "") + "."


def _style_line(cx, trips):
    hb = sum((t.get("harsh_brake") or 0) for t in trips); ha = sum((t.get("harsh_accel") or 0) for t in trips); hc = sum((t.get("harsh_corner") or 0) for t in trips); km = sum(t["km"] for t in trips)
    if not km: return None
    tot = hb + ha + hc; cx.step("the hard braking, acceleration and cornering events of each trip")
    if tot == 0: return "Driving style: smooth, with no hard braking, acceleration or cornering."
    parts = [f"{n} hard {w}" for n, w in ((hb, "braking"), (ha, "acceleration"), (hc, "cornering")) if n]
    return f"Driving style: {', '.join(parts)} ({100.0 * tot / km:.1f} per 100 km)."


def _warning_line(cx, lo, hi):
    rows = _warning_events(cx, lo, hi); cx.step("the events recorded in the period")
    if not rows: return "Warnings: none recorded."
    cnt = _counts(rows); top = sorted(cnt.items(), key=lambda kv: -kv[1])[:3]; sev = {r[0]: r[1] for r in cx.b.c.execute("SELECT ts, sev FROM events WHERE ts>=? AND ts<? AND sev IN ('critical','error')", (int(lo), int(hi))).fetchall()}
    crit = sum(1 for r in rows if r[0] in sev)
    txt = ", ".join(f"{n} × {_event_label(t)}" for t, n in top) + (f" and {len(cnt) - 3} other kind{'s' if len(cnt) - 3 > 1 else ''}" if len(cnt) > 3 else "")
    return f"Warnings: {len(rows)} - {txt}; the latest on {cx.stamp(rows[-1][0])}." + (f" {crit} of them critical." if crit else "")


def _peak_lines(cx):
    out = []
    for concept in ("battery_temperature", "motor_temperature", "engine_oil_temperature", "transmission_temperature"):
        d = cx.daily(concept, "MAX")
        if not d: continue
        now, before = cx.window(d, cx.lo, cx.hi), cx.window(d, cx.plo, cx.phi)
        if not now: continue
        day, v = max(now.items(), key=lambda kv: kv[1]); u = _unit(cx.D.unit(concept)); lab = cx.label_of(concept)
        tail = f" (the period before peaked at {_num(max(before.values()))}{u})" if before else ""
        out.append(f"Highest {lab}: {_num(v)}{u} on {cx.date(day * DAY - cx.tz)}{tail}."); cx.step(f"the daily peaks of {lab}")
        if len(out) == 2: break
    return out


def _energy_level_line(cx):
    concept, word = ("fuel_level", "fuel level") if cx.cls == "ice" else ("battery_soc", "state of charge")
    ent = (R.STORE.get(concept) or {}).get("1m")
    if not ent or ent[0] not in cx.D.have: return None
    table, col, scale = ent; r = cx.b.c.execute(f"SELECT MIN({col}), COUNT(*) FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (cx.lo, cx.hi)).fetchone(); last = cx.b.c.execute(f"SELECT {col} FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (cx.lo, cx.hi)).fetchone()
    if not r or not r[1] or not last: return None
    cx.step(f"the {word} readings"); return f"Lowest {word}: {_num(r[0] / scale, 0)}%; at the end of the period {_num(last[0] / scale, 0)}%."


def _charge_line(cx):
    try: rows = cx.b.c.execute("SELECT start, kind, kwh FROM charges WHERE start>=? AND start<? AND kwh>0", (cx.lo, cx.hi)).fetchall()
    except Exception: return None
    if not rows or cx.cls == "ice": return None
    cx.step("the charging ledger"); dc = sum(1 for r in rows if r[1] == "DC")
    return f"Charging: {len(rows)} session{'s' if len(rows) > 1 else ''}, {_num(sum(r[2] for r in rows))} kWh added" + (f" ({dc} fast)" if dc else "") + "."


def period_summary(cx, concept=None):
    trips = cx.trips(cx.lo, cx.hi); ptrips = cx.trips(cx.plo, cx.phi); lines = []
    if not trips and not _warning_events(cx, cx.lo, cx.hi): return _gap(cx, f"No trip and no event is recorded {_in(cx.label)}, so there is nothing to summarise for that period.", "summary:empty")
    if concept is not None: return concept_summary(cx, concept, trips)
    if trips:
        km = sum(t["km"] for t in trips); sec = sum(t["end"] - t["start"] for t in trips); pk = sum(t["km"] for t in ptrips); cx.step(f"{len(trips)} trips from the trip ledger"); c = _vs(km, pk, "more", "less", "distance") if pk > 5 else None
        lines.append(f"{_in(cx.label).capitalize()}: {len(trips)} trip{'s' if len(trips) != 1 else ''}, {_num(km, 0)} km, {_dur(sec)} driving, average {_num(km / (sec / 3600.0), 0)} km/h" + (f" - {c} the period before" if c else "") + ".")
        long_ = max(trips, key=lambda t: t["km"]); lines.append(f"Longest trip: {_num(long_['km'], 0)} km on {cx.date(long_['start'])}.")
        for fn in (lambda: _energy_line(cx, trips, ptrips), lambda: _style_line(cx, trips)):
            x = fn()
            if x: lines.append(x)
    else: lines.append(f"{_in(cx.label).capitalize()}: the vehicle did not drive.")
    for fn in (lambda: _energy_level_line(cx), lambda: _charge_line(cx), lambda: _warning_line(cx, cx.lo, cx.hi)):
        x = fn()
        if x: lines.append(x)
    lines += _peak_lines(cx)
    ch = _changes(cx, limit=1)
    if ch: lines.append("Stands out: " + ch[0]["text"] + ".")
    return _result(cx, lines[0] + "\n" + "\n".join("- " + x for x in lines[1:]), "digest:summary", trips=len(trips))


def concept_summary(cx, c, trips):
    """A digest of one quantity or event over the period: its level, range, direction and what it did against the period before."""
    if c.kind == "event":
        from . import eventx as EX
        types = EX._types(c); rows = EX._events(cx.b, types, cx.lo, cx.hi); prev = EX._events(cx.b, types, cx.plo, cx.phi); cx.step(f"the {len(rows)} recorded occurrences")
        if not rows: return _gap(cx, f"{c.label.capitalize()} did not happen {_in(cx.label)}.", "digest:summary:event", count=0)
        days = {(int(r[0]) + cx.tz) // DAY for r in rows}; cmp_ = _vs(len(rows), len(prev), "more often", "less often")
        return _result(cx, f"{c.label.capitalize()} happened {len(rows)} time{'s' if len(rows) != 1 else ''} {_in(cx.label)}, on {len(days)} different day{'s' if len(days) != 1 else ''}; the latest on {cx.stamp(rows[-1][0])}.\n" + (f"- That is {cmp_} the period before ({len(prev)})." if cmp_ else "- Nothing of this kind was recorded in the period before."), "digest:summary:event", count=len(rows))
    now, before = cx.window(cx.daily(c.id, "AVG"), cx.lo, cx.hi), cx.window(cx.daily(c.id, "AVG"), cx.plo, cx.phi); mx, mn = cx.window(cx.daily(c.id, "MAX"), cx.lo, cx.hi), cx.window(cx.daily(c.id, "MIN"), cx.lo, cx.hi)
    if not now: return _gap(cx, f"There is no recorded {c.label} {_in(cx.label)}, so there is nothing to summarise.", "digest:summary:signal")
    u = _unit(c.unit); mean = float(np.mean(list(now.values()))); cx.step(f"the daily values of {c.label}"); lines = [f"{c.label.capitalize()} {_in(cx.label)}: average {_num(mean)}{u}, between {_num(min(mn.values()))} and {_num(max(mx.values()))}{u}."]
    if before and len(before) >= 2:
        pm = float(np.mean(list(before.values()))); lo_, hi_ = cx.D.rng(c.id)
        if abs(pm) < 0.05 * (hi_ - lo_): d_ = mean - pm; lines.append(f"- Compared with the period before ({_num(pm)}{u}): {'about the same' if abs(d_) < 0.02 * (hi_ - lo_) else _num(abs(d_)) + u + (' higher' if d_ > 0 else ' lower')}.")        # a ratio to a value near zero means nothing
        else: v_ = _vs(mean, pm, "above", "below"); lines.append(f"- Compared with the period before: {'about the same as it' if v_ == 'about the same as' else v_.replace(' than', ' it') if v_ else ''} ({_num(pm)}{u}).")
    if mx: day, v = max(mx.items(), key=lambda kv: kv[1]); lines.append(f"- Highest on {cx.date(day * DAY - cx.tz)}: {_num(v)}{u}.")
    an = _anomalies(cx, [c.id], events=False, limit=2)
    lines += [f"- Unusual: {a['text']}." for a in an]
    return _result(cx, lines[0] + "\n" + "\n".join(lines[1:]), "digest:summary:signal", mean=mean)


# ------------------------------------------------------------------------------------------------------------------ what changed
def _changes(cx, limit=4):
    """Quantities, trip figures and event rates that moved between the previous period and this one, largest first: [{score, text}]."""
    out = []
    for concept in _roster(cx):
        d = cx.daily(concept, "AVG")
        if not d: continue
        a, p = cx.window(d, cx.lo, cx.hi), cx.window(d, cx.plo, cx.phi)
        if len(a) < 3 or len(p) < 3: continue
        e = effect(list(a.values()), list(p.values()))
        if not e: continue
        lo_, hi_ = cx.D.rng(concept); rel = abs(e["diff"]) / max(abs(e["b"]), 0.02 * (hi_ - lo_), 1e-9)
        if abs(e["d"]) < 2.0 or rel < MIN_REL or abs(e["diff"]) < 0.04 * (hi_ - lo_): continue
        u = _unit(cx.D.unit(concept)); out.append({"score": abs(e["d"]), "concept": concept, "text": f"{cx.label_of(concept)} {'rose' if e['diff'] > 0 else 'fell'} from {_num(e['b'])} to {_num(e['a'])}{u} (daily average{' while driving' if concept not in ANY_TIME else ''})"})
    trips, ptrips = cx.trips(cx.lo, cx.hi), cx.trips(cx.plo, cx.phi)
    if len(trips) >= 3 and len(ptrips) >= 3:
        for key, label, unit, nd in (("avg_speed", "average trip speed", " km/h", 0), ("km", "trip length", " km", 0)):
            e = effect([t[key] for t in trips if t[key] is not None], [t[key] for t in ptrips if t[key] is not None])
            if e and abs(e["d"]) >= 1.2 and abs(e["diff"]) / max(abs(e["b"]), 1e-9) >= 0.1: out.append({"score": abs(e["d"]), "concept": key, "text": f"{label} {'rose' if e['diff'] > 0 else 'fell'} from {_num(e['b'], nd)} to {_num(e['a'], nd)}{unit}"})
    km, pk = sum(t["km"] for t in trips), sum(t["km"] for t in ptrips)
    if pk > 20 and km > 0 and abs(km - pk) / pk >= 0.25: out.append({"score": 1.0 + abs(km - pk) / pk, "concept": "distance", "text": f"distance driven {'rose' if km > pk else 'fell'} from {_num(pk, 0)} to {_num(km, 0)} km"})
    ev, pev = _counts(_warning_events(cx, cx.lo, cx.hi)), _counts(_warning_events(cx, cx.plo, cx.phi))
    for ty in set(ev) | set(pev):
        n, o = ev.get(ty, 0), pev.get(ty, 0)
        if abs(n - o) >= 3 and (o == 0 or n / max(o, 1) >= 2 or n / max(o, 1) <= 0.5): out.append({"score": 1.0 + min(abs(n - o), 20) / 10.0, "concept": ty, "text": f"{_event_label(ty)}: {n} now, {o} in the period before" + (" (new)" if o == 0 else "")})
    cx.step("the same measurements over the period before, compared by robust effect size")
    return sorted(out, key=lambda x: -x["score"])[:limit]


def what_changed(cx):
    ch = _changes(cx, limit=5)
    if not ch:
        trips = cx.trips(cx.lo, cx.hi); km = sum(t["km"] for t in trips)
        if not trips and not cx.trips(cx.plo, cx.phi) and not _warning_events(cx, cx.plo, cx.hi) and not any(cx.daily(c_, "AVG") and (cx.window(cx.daily(c_, "AVG"), cx.lo, cx.hi) or cx.window(cx.daily(c_, "AVG"), cx.plo, cx.phi)) for c_ in _roster(cx)):
            return _gap(cx, f"Nothing is recorded {_in(cx.label)} or in the period before it, so there is nothing to compare.", "digest:changes:empty")
        return _result(cx, f"Nothing has changed noticeably {_in(cx.label)}, compared with the period of the same length before it.\n- Every quantity I track stayed within its usual spread, and no kind of warning became more frequent.\n" + (f"- You drove {_num(km, 0)} km in {len(trips)} trips." if trips else "- The vehicle did not drive."), "digest:changes", changes=0)
    return _result(cx, f"{len(ch)} thing{'s' if len(ch) > 1 else ''} changed {_in(cx.label)}, compared with the period before:\n" + "\n".join(f"- {x['text']}." for x in ch), "digest:changes", changes=len(ch), items=[x["concept"] for x in ch])


# ------------------------------------------------------------------------------------------------------------------ unusual readings
def _anomalies(cx, concepts=None, events=True, limit=5):
    out = []; base_lo = cx.lo - BASELINE_DAYS * DAY
    for concept in (concepts or _roster(cx, behaviour=False)):
        best = None
        for agg, word in (("AVG", "average"), ("MAX", "peak"), ("MIN", "lowest value")):
            d = cx.daily(concept, agg)
            if not d: continue
            a, base = cx.window(d, cx.lo, cx.hi), cx.window(d, base_lo, cx.lo)
            if not a or len(base) < 6: continue
            ref = list(base.values()); z = robust_z(list(a.values()), ref); med = float(np.median(ref)); lo_, hi_ = cx.D.rng(concept); floor = 0.04 * (hi_ - lo_)
            for (day, v), zi in zip(a.items(), z):
                if abs(zi) >= 4.0 and abs(v - med) >= floor and (best is None or abs(zi) > best[0]): best = (abs(float(zi)), day, v, med, word, len(base))
        if best:
            zs, day, v, med, word, n = best; u = _unit(cx.D.unit(concept)); out.append({"score": zs / 4.0, "concept": concept, "ts": day * DAY - cx.tz, "text": f"{cx.label_of(concept)}: the daily {word} on {cx.date(day * DAY - cx.tz)} was {_num(v)}{u}, against {_num(med)}{u} on a usual day ({'above' if v > med else 'below'} anything in the previous {n} days with data)"})
    if events:
        ev = _warning_events(cx, cx.lo, cx.hi); prior = _counts(_warning_events(cx, base_lo, cx.lo)); weeks = max(1.0, (cx.lo - base_lo) / (7 * DAY)); span_w = max(1.0, (cx.hi - cx.lo) / (7 * DAY))
        for ty, n in _counts(ev).items():
            rate = prior.get(ty, 0) / weeks * span_w; last = [e[0] for e in ev if e[1] == ty][-1]
            if prior.get(ty, 0) == 0: out.append({"score": 1.5, "concept": ty, "ts": last, "text": f"{_event_label(ty)}: {n} time{'s' if n > 1 else ''}, the first in the previous {int((cx.lo - base_lo) / DAY)} days (latest {cx.stamp(last)})"})
            elif n >= 3 and n >= 3 * max(rate, 0.5): out.append({"score": 1.0 + n / max(rate, 1.0) / 5.0, "concept": ty, "ts": last, "text": f"{_event_label(ty)}: {n} times, against about {rate:.1f} in a usual period of this length"})
    cx.step(f"each quantity's daily values against its own previous {BASELINE_DAYS} days (robust z-score of at least 4)" + (" and the warning counts against their usual rate" if events else ""))
    return sorted(out, key=lambda x: -x["score"])[:limit]


def anomaly(cx, c=None):
    concept = c.id if c is not None and c.kind in ("signal", "derived", "declared", "event") else None
    if concept and c.kind == "event":
        from . import eventx as EX
        types = set(EX._types(c)); ev = _warning_events(cx, cx.lo, cx.hi, types); base = _warning_events(cx, cx.lo - BASELINE_DAYS * DAY, cx.lo, types); weeks = BASELINE_DAYS / 7.0; rate = len(base) / weeks * max(1.0, (cx.hi - cx.lo) / (7 * DAY))
        cx.step(f"the {len(ev)} occurrences against the {len(base)} of the previous {BASELINE_DAYS} days")
        if not ev: return _result(cx, f"{c.label.capitalize()} did not happen {_in(cx.label)}; that is not unusual" + (f" (about {rate:.1f} would be usual)." if rate >= 1 else "."), "digest:anomaly", anomalies=0)
        if len(ev) >= 3 and len(ev) >= 2 * max(rate, 1.0) or not base: return _result(cx, f"{c.label.capitalize()} is unusual {_in(cx.label)}: {len(ev)} time{'s' if len(ev) > 1 else ''}, against about {rate:.1f} in a usual period of this length.", "digest:anomaly", anomalies=1)
        return _result(cx, f"{c.label.capitalize()} is not unusual {_in(cx.label)}: {len(ev)} time{'s' if len(ev) > 1 else ''}, against about {rate:.1f} in a usual period of this length.", "digest:anomaly", anomalies=0)
    items = _anomalies(cx, [concept] if concept else None, events=concept is None)
    scope = cx.label_of(concept) if concept else "the vehicle"
    if not items:
        n = len(_roster(cx, behaviour=False)) if concept is None else 1
        if n == 0 or not any((cx.window(cx.daily(c_, "AVG"), cx.lo, cx.hi) for c_ in ([concept] if concept else _roster(cx, behaviour=False)))): return _gap(cx, f"I could not check anything for unusual behaviour {_in(cx.label)}: this vehicle records no quantity in that period that I can compare with its own history.", "digest:anomaly:empty")
        return _result(cx, f"Nothing looks unusual {_in(cx.label)} for {scope}.\n- I checked {n} measured quantit{'ies' if n > 1 else 'y'} against {'their' if n > 1 else 'its'} own recent history" + (" and the warnings against their usual rate" if concept is None else "") + "; every value stayed within the normal spread" + (" and no new kind of warning appeared." if concept is None else "."), "digest:anomaly", anomalies=0)
    return _result(cx, f"{len(items)} thing{'s' if len(items) > 1 else ''} look unusual {_in(cx.label)}:\n" + "\n".join(f"{i + 1}. {x['text']}." for i, x in enumerate(items)), "digest:anomaly", anomalies=len(items), items=[x["concept"] for x in items])


# ------------------------------------------------------------------------------------------------------------------ the usual behaviour of one quantity
def _band(h): return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"


def usual(cx, c):
    if c.kind == "event":
        from . import eventx as EX
        types = EX._types(c); rows = EX._events(cx.b, types, cx.lo, cx.hi); weeks = max(1, int((cx.hi - cx.lo) / (7 * DAY))); cx.step(f"the {len(rows)} recorded occurrences")
        if not rows: return _result(cx, f"{c.label.capitalize()} did not happen {_in(cx.label)}, so there is no usual rate.", "digest:usual", count=0)
        by_week = {}
        for r in rows: k = (int(r[0]) - cx.lo) // (7 * DAY); by_week[k] = by_week.get(k, 0) + 1
        vals = [by_week.get(k, 0) for k in range(weeks)]; cx.step(f"the {len(rows)} recorded occurrences, counted per week"); med = statistics.median(vals)
        return _result(cx, f"{EX._wording(c, 2).capitalize()} usually happen about {med:g} time{'s' if med != 1 else ''} a week (between {min(vals)} and {max(vals)}) {_in(cx.label)}; {len(rows)} in all.", "digest:usual", count=len(rows))
    ent = (R.STORE.get(c.id) or {}).get("1m")
    if not ent or ent[0] not in cx.D.have: return _gap(cx, f"{c.label.capitalize()} is not stored at a resolution that shows what is usual.", "digest:usual")
    table, col, scale = ent; masked = c.id not in ANY_TIME
    join = " JOIN sig_1m s ON s.ts = t.ts" if masked and table != "sig_1m" else ""; where = " AND s.mode IN (1,2)" if masked and table != "sig_1m" else (" AND t.mode IN (1,2)" if masked else "")
    rows = cx.b.c.execute(f"SELECT t.ts, t.{col} FROM {table} t{join} WHERE t.ts>=? AND t.ts<? AND t.{col} IS NOT NULL{where}", (cx.lo, cx.hi)).fetchall()
    if len(rows) < 30 and masked: rows = cx.b.c.execute(f"SELECT ts, {col} FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL", (cx.lo, cx.hi)).fetchall(); masked = False
    if len(rows) < 30: return _gap(cx, f"There is too little recorded {c.label} {_in(cx.label)} to say what is usual.", "digest:usual")
    t = np.array([r[0] for r in rows], dtype=float); v = np.array([r[1] for r in rows], dtype=float) / scale
    p10, p50, p90 = (float(x) for x in np.percentile(v, [10, 50, 90])); u = _unit(c.unit); cx.step(f"{len(v)} one-minute values of {c.label}" + (" while driving" if masked else ""))
    txt = f"{c.label.capitalize()} is usually between {_num(p10)} and {_num(p90)}{u}, median {_num(p50)}{u}" + (" while driving" if masked else "") + f"; it ranged from {_num(float(v.min()))} to {_num(float(v.max()))}{u} {_in(cx.label)}."
    hours = (((t + cx.tz) % DAY) // 3600).astype(int); bands = {}
    for h, x in zip(hours, v): bands.setdefault(_band(int(h)), []).append(x)
    meds = {k: float(np.median(x)) for k, x in bands.items() if len(x) >= 30}
    if len(meds) >= 2 and max(meds.values()) - min(meds.values()) >= 0.15 * max(p90 - p10, 1e-9):
        hi_b, lo_b = max(meds, key=meds.get), min(meds, key=meds.get); txt += f"\nPattern: highest in the {hi_b} ({_num(meds[hi_b])}{u}), lowest in the {lo_b} ({_num(meds[lo_b])}{u})."
    modesql = "t.mode" if table == "sig_1m" else "(SELECT mode FROM sig_1m s WHERE s.ts = t.ts)"
    last = cx.b.c.execute(f"SELECT t.{col}, {modesql} FROM {table} t WHERE t.ts<=? AND t.{col} IS NOT NULL ORDER BY t.ts DESC LIMIT 1", (cx.now,)).fetchone()
    if last:
        x = float(last[0]) / scale
        if masked and last[1] not in (1, 2): txt += f"\nThe vehicle is not driving now (the latest reading, {_num(x)}{u}, cannot be compared with the readings while driving)."
        else: where_ = "within the usual range" if p10 <= x <= p90 else ("above the usual range" if x > p90 else "below the usual range"); txt += f"\nThe latest value, {_num(x)}{u}, is {where_}."
    return _result(cx, txt, "digest:usual", median=p50, p10=p10, p90=p90)


# ------------------------------------------------------------------------------------------------------------------ one trip or one day
def _pick(cx):
    """The trip the question is about: named by a typed value (`literals.trip_pick`) or by a date; the latest by default."""
    pick = getattr(cx.lit, "pick", None); trips = cx.trips(cx.lo, cx.hi) if (cx.lit.times and not pick) else cx.trips(cx.now - 60 * DAY, cx.hi)
    if not trips: return None, "no trips"
    cx.step(f"{len(trips)} trips from the trip ledger")
    if pick == "longest": return max(trips, key=lambda t: t["km"]), "longest"
    if pick == "shortest": return min(trips, key=lambda t: t["km"]), "shortest"
    if pick == "first": return trips[0], "first"
    eff = [t for t in trips if t["wh_km"]]
    if pick == "most_efficient" and eff: return min(eff, key=lambda t: t["wh_km"]), "most efficient"
    if pick == "least_efficient" and eff: return max(eff, key=lambda t: t["wh_km"]), "least efficient"
    return trips[-1], "latest"


def _trip_peak(cx, concept, lo, hi):
    ent = (R.STORE.get(concept) or {}).get("1m")
    if not ent or ent[0] not in cx.D.have: return None
    table, col, scale = ent; col = "vmax" if table.startswith("lkv_") else col
    r = cx.b.c.execute(f"SELECT MAX({col}), AVG({col}), COUNT(*) FROM {table} WHERE ts>=? AND ts<=? AND {col} IS NOT NULL", (int(lo), int(hi))).fetchone()
    return None if not r or not r[2] else (float(r[0]) / scale, float(r[1]) / scale)


def _usual_trip_peaks(cx, concept, skip):
    out = [p for p in (_trip_peak(cx, concept, t["start"], t["end"]) for t in cx.trips(cx.now - 60 * DAY, cx.now + 1) if t["start"] != skip) if p is not None]
    return [p[0] for p in out]


def _trip_vs_usual(cx, tr):
    """How one trip differed from the trips around it: [(score, text)] over trip figures, the peak of the quantities it recorded, and the conditions."""
    allt = [t for t in cx.trips(cx.now - 60 * DAY, cx.now + 1) if t["start"] != tr["start"]]; out = []
    if len(allt) < 5: return out
    for key, label, unit, nd, k in (("km", "length", " km", 0, 1), ("avg_speed", "average speed", " km/h", 0, 1), ("avg_amb", "outside temperature", "°C", 0, 1), ("hwy_share", "share on the motorway", "%", 0, 100)):
        ref = [t[key] * k for t in allt if t.get(key) is not None]; x = tr.get(key)
        if x is None or len(ref) < 5: continue
        z = float(robust_z([x * k], ref)[0]); med = float(np.median(ref))
        if abs(z) >= 1.5: out.append((abs(z), f"{label} {_num(x * k, nd)}{unit} against a usual {_num(med, nd)}{unit}"))
    for concept in ("battery_temperature", "motor_temperature", "engine_oil_temperature", "coolant_flow", "hvac_power", "battery_power", "engine_load", "fuel_rate"):
        if not cx.D.available(concept): continue
        mine = _trip_peak(cx, concept, tr["start"], tr["end"]); peaks = _usual_trip_peaks(cx, concept, tr["start"])
        if mine is None or len(peaks) < 5: continue
        z = float(robust_z([mine[0]], peaks)[0]); u = _unit(cx.D.unit(concept))
        if abs(z) >= 1.5: out.append((abs(z) + 0.5, f"peak {cx.label_of(concept)} {_num(mine[0])}{u} against a usual trip peak of {_num(float(np.median(peaks)))}{u}"))
    return sorted(out, key=lambda x: -x[0])


def trip_focus(cx, c, kind):
    """kind: why | summary | anomaly about one trip."""
    tr, which = _pick(cx)
    if tr is None: return _gap(cx, "No trip is recorded in the last 60 days, so there is no trip to look at.", "digest:trip")
    head = f"The {which} trip, on {cx.stamp(tr['start'])}: {_num(tr['km'], 0)} km in {_dur(tr['end'] - tr['start'])}" + (f", {_num(tr['avg_speed'], 0)} km/h on average" if tr.get("avg_speed") else "")
    wh = tr.get("wh_km"); head += (f", {_num(wh, 0)} Wh/km" if wh and cx.cls != "ice" else "") + "."
    diff = _trip_vs_usual(cx, tr); ev = _warning_events(cx, tr["start"], tr["end"] + 60); lines = [head]; premise = None
    if c.id not in ("none", "unlisted") and c.kind in ("signal", "derived") and cx.D.available(c.id):
        mine = _trip_peak(cx, c.id, tr["start"], tr["end"]); peaks = _usual_trip_peaks(cx, c.id, tr["start"]); u = _unit(c.unit)
        if mine is not None:
            cx.step(f"the {cx.label_of(c.id)} readings between the start and the end of the trip"); lab = cx.label_of(c.id).capitalize(); line = f"{lab} on this trip: average {_num(mine[1])}{u}, peak {_num(mine[0])}{u}"
            if len(peaks) >= 5:
                z = float(robust_z([mine[0]], peaks)[0]); med = float(np.median(peaks)); ref = f"a usual trip peak of {_num(med)}{u}"
                premise = "high" if z >= 1.5 else "low" if z <= -1.5 else "normal"
                line += {"high": f" - clearly higher than {ref}.", "low": f" - clearly lower than {ref}.", "normal": f" - in line with {ref}, so it was not unusual."}[premise]
            lines.append(line)
    if kind == "why":
        if premise == "normal": lines.append("The data does not show anything unusual to explain here. For reference, what differed about the trip:" if diff else "Nothing about the trip stands out against your other recent trips either.")
        elif diff: lines.append("What was different about it, compared with your other trips of the last 60 days (associations, not proven causes):")
        else: lines.append("Nothing about the trip stands out against your other recent trips, so the data offers no condition that explains it.")
        lines += [f"- {t}" for _, t in diff[:4]]
    elif diff: lines.append("It stands out in: " + "; ".join(t for _, t in diff[:3]) + ".")
    elif kind == "anomaly": lines.append("Nothing about it is unusual for this vehicle.")
    if ev: cnt = _counts(ev); lines.append("During the trip: " + ", ".join(f"{n} × {_event_label(t)}" for t, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]) + ".")
    cx.step("the same figures for your other trips of the last 60 days, compared by robust z-score")
    return _result(cx, lines[0] + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else ""), f"digest:trip:{kind}", trip_start=int(tr["start"]))


# ------------------------------------------------------------------------------------------------------------------ energy lost while parked
def _parked_stretches(cx, table, col, scale, lo, hi, min_min=60):
    """[(start, end, level_at_start, level_at_end)] for every stretch of consecutive parked minutes of at least `min_min` minutes between lo and hi (a charging or driving minute ends a stretch)."""
    sql = f"SELECT s.ts, t.{col}, s.mode FROM sig_1m s LEFT JOIN {table} t ON t.ts = s.ts WHERE s.ts>=? AND s.ts<? ORDER BY s.ts" if table != "sig_1m" else f"SELECT ts, {col}, mode FROM sig_1m WHERE ts>=? AND ts<? ORDER BY ts"
    out, cur = [], None
    for ts, v, mode in cx.b.c.execute(sql, (int(lo), int(hi))):
        if mode == 0 and cur is not None and ts - cur[1] <= 180:
            cur[1] = ts
            if cur[2] is None: cur[2] = v
            if v is not None: cur[3] = v
        else:
            if cur is not None and cur[1] - cur[0] >= min_min * 60: out.append(tuple(cur))
            cur = [ts, ts, v, v] if mode == 0 else None
    if cur is not None and cur[1] - cur[0] >= min_min * 60: out.append(tuple(cur))
    return [(a, b, float(x) / scale, float(y) / scale) for a, b, x, y in out if x is not None and y is not None]


def _level_near(cx, table, col, scale, ts, within=900):
    r = cx.b.c.execute(f"SELECT {col} FROM {table} WHERE ts BETWEEN ? AND ? AND {col} IS NOT NULL ORDER BY ABS(ts - ?) LIMIT 1", (int(ts) - within, int(ts) + within, int(ts))).fetchone()
    return None if not r else float(r[0]) / scale


def _pts(n): return f"{_num(n, 1)} point{'' if abs(round(n, 1)) == 1 else 's'}"


def parked_loss(cx, c):
    """Why did the battery (or the fuel level) fall while the vehicle stood?  The parked stretch the question is about, the loss over it against the usual loss of this vehicle's other stretches, what was running."""
    ent = (R.STORE.get(c.id) or {}).get("1m")
    if not ent or ent[0] not in cx.D.have: return _gap(cx, f"{c.label.capitalize()} is not stored at a resolution that shows what happens while the vehicle stands.", "digest:parked")
    table, col, scale = ent; word = "state of charge" if c.id == "battery_soc" else "fuel level"
    if not cx.lit.times: cx.lo, cx.hi, cx.label = cx.now - 36 * 3600, cx.now + 1, "in the last day and a half"
    mine = _parked_stretches(cx, table, col, scale, cx.lo - 6 * 3600, min(cx.hi + 6 * 3600, cx.now + 1))
    if not mine: return _gap(cx, f"The vehicle was not parked for an hour or more {_in(cx.label)}, so there is no parked stretch to look at.", "digest:parked")
    lo_, hi_ = cx.lo, cx.hi; a, b, x, y = max(mine, key=lambda s_: max(0, min(s_[1], hi_) - max(s_[0], lo_)))
    if cx.lit.times and (a < lo_ or b > hi_):                                                       # the stretch is longer than the period asked about (a car parked all day, "last night"): measure the part inside it
        a2, b2 = max(a, lo_), min(b, hi_); xa, ya = _level_near(cx, table, col, scale, a2), _level_near(cx, table, col, scale, b2)
        if xa is not None and ya is not None and b2 - a2 >= 1800: a, b, x, y = a2, b2, xa, ya
    hours = (b - a) / 3600.0; drop = x - y; cx.step(f"the {word} at the start and the end of the parked stretch")
    lines = [f"The vehicle stood parked from {cx.stamp(a)} to {cx.stamp(b)} ({_dur(b - a)})."]
    caps = FX.caps_of(cx.res); cap = caps.get("cap_kwh") if c.id == "battery_soc" else caps.get("tank_l"); unit = "kWh" if c.id == "battery_soc" else "L"
    amount = f", about {_num(abs(drop) / 100.0 * cap, 1)} {unit}" if cap else ""
    if drop <= 0.05: lines.append(f"The {word} did not fall in that time: {_num(x, 0)}% at the start, {_num(y, 0)}% at the end" + (" (a rise of a point or two while standing is sensor drift)." if y > x else "."))
    else: nd = 1 if drop < 5 else 0; lines.append(f"The {word} fell from {_num(x, nd)}% to {_num(y, nd)}%: {_pts(drop)}{amount} in {_dur(b - a)}, {_num(drop / hours, 2)} points an hour.")
    rates = [(s_[2] - s_[3]) / ((s_[1] - s_[0]) / 3600.0) for s_ in _parked_stretches(cx, table, col, scale, cx.lo - BASELINE_DAYS * DAY, cx.lo - 6 * 3600, min_min=120) if s_[0] != a]
    if len(rates) >= 3 and drop > 0.05:
        cx.step(f"the same figure over {len(rates)} other parked stretches of two hours or more"); med = float(np.median(rates)); r = drop / hours
        same = abs(r - med) <= max(0.05, 0.25 * abs(med)); cmp_ = "about the same as" if same else (f"{r / med:.1f} times" if (r > med and med > 0.02) else "higher than") if r > med else "lower than"
        lines.append(f"That is {cmp_} your usual loss while parked ({_num(med, 2)} points an hour, median of {len(rates)} other stretches)" + (", so nothing abnormal is shown." if same else "."))
    ran = []
    if c.id == "battery_soc" and "sig_1m" in cx.D.have:
        hv = cx.b.c.execute("SELECT COUNT(*), AVG(hvac), MIN(ts) FROM sig_1m WHERE ts>=? AND ts<=? AND hvac>50", (int(a), int(b))).fetchone()
        if hv and hv[0]: ran.append(f"climate control was on for {_dur(hv[0] * 60)} (about {_num(hv[1], 0)} W on average, from {cx.stamp(hv[2])})"); cx.step("the climate-control power during the stretch")
    ev = _warning_events(cx, a, b)
    if ev: cnt = _counts(ev); ran.append("the vehicle recorded " + ", ".join(f"{n} × {_event_label(t)}" for t, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:2]))
    if ran: lines.append("While it stood, " + " and ".join(ran) + ". That is what the data shows running; it does not prove it caused the loss.")
    elif drop > 0.05: lines.append("Nothing else is recorded as running or happening in that time, so the loss looks like the vehicle's own standby use.")
    return _result(cx, lines[0] + "\n" + "\n".join("- " + x for x in lines[1:]), "digest:parked", drop=float(drop), hours=float(hours))


# ------------------------------------------------------------------------------------------------------------------ where a quantity stands, for a "why" about it
def status_line(res, c, frame, lit, text):
    """One factual line placed under the answer to a "why" about a measured quantity: where the quantity stood in the period asked about, against its own usual range.  "Why is it so high" about a value that
    is in its normal range is then answered with the fact, not with an explanation of a problem that is not in the data.  None when the vehicle has too little history to say."""
    if c.kind not in ("signal", "derived") or c.id in STATES or c.unit in ("category", "none", ""): return None
    cx = Ctx(res, c, frame, lit, text); d = cx.daily(c.id, "AVG", all_day=True)                                  # all minutes, the basis of the explanation printed above it
    if not d: return None
    now, base = cx.window(d, cx.lo, cx.hi), cx.window(d, cx.lo - BASELINE_DAYS * DAY, cx.lo)
    if not now or len(base) < 7: return None
    ref = np.array(list(base.values())); p10, p90 = (float(x) for x in np.percentile(ref, [10, 90])); x = float(np.mean(list(now.values()))); u = _unit(c.unit); lo_, hi_ = cx.D.rng(c.id)
    if p90 - p10 < 0.01 * (hi_ - lo_): return None
    where = "within" if p10 <= x <= p90 else ("above" if x > p90 else "below")
    return f"For reference: {cx.label_of(c.id)} averaged {_num(x)}{u} {_in(cx.label)}, {where} its usual range ({_num(p10)} to {_num(p90)}{u} over the {BASELINE_DAYS} days before)" + (", so there is no unusual level to explain." if where == "within" else ".")


# ------------------------------------------------------------------------------------------------------------------ the entry point used by the resolver
def maybe(res, c, frame, lit, text):
    """-> a result dict, or None when the frame is not one of the digest meanings."""
    kind = None
    if frame.act in ("summary", "anomaly"): kind = frame.act
    elif frame.act == "stat" and frame.op == "usual": kind = "usual"
    elif frame.act == "why" and frame.ent in ("trip", "day"): kind = "why"
    elif frame.act == "why" and c.id in ("battery_soc", "fuel_level") and (frame.cond in ("parked", "night") or any(getattr(t, "name", None) in ("last_night", "tonight") for t in lit.times)): kind = "parked"
    if kind is None: return None
    cx = Ctx(res, c, frame, lit, text)
    try:
        if kind == "usual": return usual(cx, c) if c.kind in ("signal", "derived", "declared", "event") else None
        if kind == "parked": return parked_loss(cx, c)
        if frame.ent == "trip": return trip_focus(cx, c, kind)
        if kind == "why":                                                                                           # one day: its summary against the days around it
            if not lit.times: cx.lo, cx.hi = cx.now - DAY, cx.now + 1; cx.label = "in the last day"
            return period_summary(cx, c if c.kind in ("signal", "derived", "declared", "event") else None)
        if kind == "summary":
            if frame.op == "change": return what_changed(cx)
            return period_summary(cx, c if c.kind in ("signal", "derived", "declared", "event") else None)
        return anomaly(cx, c)
    except Exception as e:                                                                                          # an executor that cannot run on this vehicle's data says so; it never takes the assistant down
        import logging; logging.getLogger("vehiclegpt").warning("digest executor failed: %s: %s", type(e).__name__, e)
        return _gap(cx, "I understood the question, but the routine that answers it could not run on this vehicle's data, so I will not guess.", "error:digest", error=type(e).__name__)
