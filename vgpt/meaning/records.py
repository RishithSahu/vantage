"""Executors that answer from the vehicle's own records: who reported an event, the state of the diagnostic codes, how long a source was silent, where a measurement comes from, charging sessions from the ledger.

    events <event> count | list   ent=source             the reporting unit of each event: the subsystem of a code, the unit that reset, the bus that went silent, the reason of a disengagement
    extreme <event> max | min     ent=source
    health diagnostic_status                              active / cleared / returned-after-clearing diagnostic codes
    events diagnostic_code ...    + a code typed in the question   the history of that code
    stat <event> count | duration                          how many; for a silent source: how long, in total
    spec <signal>                                          the sources, rate, unit and plausible range of a measurement, read from the vehicle profile stored in the database
    events charging_session ...   (cond fast_charging, ent=place)  the charging ledger: energy, duration, state of charge, place

Same rules as eventx.py: numbers come from SQL over events / ledger / profile tables, the question's own text is not inspected here (the frame names the concept, `lit` carries the parsed period and the typed code),
and tests/test_records.py recomputes every figure independently."""
import json
import math

from .. import analytics as A
from .. import signal_registry as R
from ..query_ir import TimeSpec
from .eventx import DAY, _cap, _events, _gap, _plural, _r, _stored, _types, _win, _window, _wording


# ------------------------------------------------------------------------------------------------------------------ who reported it
def _dtc_table(b):
    row = b.c.execute("SELECT v FROM meta WHERE k='vehicle.dtc'").fetchone()
    try: return json.loads(row[0]) if row else {}
    except (ValueError, TypeError): return {}


def _source_of(etype, detail, dtc):
    """The reporting unit of one event, read from the structured detail the decoder wrote (never from free text)."""
    d = str(detail or "")
    if etype in ("DTC_SET", "DTC_CLEAR"): code = d.split(" ")[0]; return (dtc.get(code) or {}).get("subsystem") or "unlisted subsystem"
    if etype == "ECU_RESET": return d.split(" reset (")[0] if " reset (" in d else (d.split(" (")[0] or "a control unit")
    if etype == "SOURCE_SILENT": return d.split(":")[0] if ":" in d else "a data source"
    if etype == "CAN_ERROR":
        i = d.find(" on the "); j = d.find(" bus", i + 1) if i >= 0 else -1
        return d[i + 8:j] + " bus" if i >= 0 and j > i else "the bus"
    return d[:60] or "unspecified"


def by_source(res, c, frame, lit):
    b = res.brain; qc = b.qc()
    if not _stored(b, c): return _gap(frame, c, "that event is detected from signals and has no stored occurrences to group")
    lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day")); rows = _events(b, _types(c), lo, hi); A.set_tz(b.tz)
    if not rows: return _r(res, f"No {c.label} was recorded {label}.", f"{frame.act}:{c.id}:source", frame, ok=False, count=0)
    types = _types(c); dtc = _dtc_table(b) if ("DTC_SET" in types or "DTC_CLEAR" in types) else {}; cnt = {}
    for ts, ty, det in rows: k = _source_of(ty, det, dtc); cnt[k] = cnt.get(k, 0) + 1
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])); n = len(rows)
    if frame.act == "extreme":
        k, v = order[0] if frame.op == "max" else order[-1]; ties = [x for x, y in order if y == v]
        return _r(res, f"The {'most' if frame.op == 'max' else 'fewest'} {_wording(c, 2)} {label} came from {', '.join(ties[:3])}: {_plural(v, 'event')} of {n} (by reporting unit: {'; '.join(f'{x} {y}' for x, y in order[:6])}).\n"
                       f"Based on {_plural(n, 'recorded event')}; the unit is read from the event's own record (a code's subsystem, the unit that reset, the bus that went silent).", f"extreme:{c.id}:source", frame, source=k, count=v, sources=dict(order))
    return _r(res, f"{_cap(c.label)} {label} by reporting unit: " + "; ".join(f"{x}: {y}" for x, y in order[:8]) + f" ({n} in all).\nBased on {_plural(n, 'recorded event')}; the unit is read from the event's own record.", f"events:{c.id}:source", frame, sources=dict(order), count=n)


# ------------------------------------------------------------------------------------------------------------------ diagnostic codes over time
def _code_events(b, lo, hi, codes=None):
    rows = b.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('DTC_SET','DTC_CLEAR') AND ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall(); out = {}
    for ts, ty, det in rows:
        code = str(det).split(" ")[0].upper()
        if codes and code not in codes: continue
        g = out.setdefault(code, {"set": [], "clear": [], "seq": [], "text": ""})
        g["set" if ty == "DTC_SET" else "clear"].append(ts); g["seq"].append((ts, ty))
        if ty == "DTC_SET" and not g["text"]: g["text"] = str(det)[len(code):].strip().split(" (status")[0]
    return out


def _state(g):
    """('active' | 'cleared', came back after a clear?)"""
    returned = False; cleared_before = False
    for ts, ty in g["seq"]:
        if ty == "DTC_CLEAR": cleared_before = True
        elif cleared_before: returned = True
    return ("active" if g["seq"][-1][1] == "DTC_SET" else "cleared"), returned


def _asof(b, qc, lit):
    """The window of a status question: "at the moment" / "right now" is the state reached by the whole history up to now, not an empty instant."""
    t = lit.times[0] if lit.times else None
    return _win(b, qc, TimeSpec("all") if t is None or t.type == "current" else t)


LAMP_TYPES = ("TPMS_LOW", "WARN_BATT_OVERTEMP", "WARN_OVERTEMP", "LOW_12V", "CAMERA_BLOCKED", "COOLANT_FLOW_DROP", "POWER_DERATE", "UNEXPECTED_SHUTDOWN")


def warning_note(b, hi):
    """A warning lamp is not always a stored code (a low tyre warns at a pressure where no code is set yet): the warnings of the last 24 hours that are not diagnostic codes, so that "0 codes" is never the whole story."""
    try:
        rows = b.c.execute("SELECT ts, type, detail FROM events WHERE ts>=? AND ts<=? AND type IN (%s) ORDER BY ts DESC" % ",".join("?" * len(LAMP_TYPES)), (int(hi) - 86400, int(hi), *LAMP_TYPES)).fetchall()
    except Exception: return ""
    if not rows: return ""
    seen, out = set(), []
    for ts, ty, det in rows:
        if ty in seen: continue
        seen.add(ty); out.append(f"{str(det)[:90]} ({A.stamp(ts)})")
    return "Warnings from the last 24 hours that are not stored as diagnostic codes: " + "; ".join(out[:4]) + "."


def dtc_status(res, c, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _asof(b, qc, lit); table = _dtc_table(b); codes = _code_events(b, lo, hi); note = warning_note(b, min(int(hi), int(b.now)))
    if not codes: return _r(res, f"No diagnostic trouble code was set or cleared {label}." + (f"\n{note}" if note else ""), "health:diagnostic_status", frame, ok=False, count=0)
    def name(code, g): return f"{code} {g['text'] or (table.get(code) or {}).get('text', '')}".strip()
    active = [(k, g) for k, g in codes.items() if _state(g)[0] == "active"]; cleared = [(k, g) for k, g in codes.items() if _state(g)[0] == "cleared"]; back = [(k, g) for k, g in codes.items() if _state(g)[1]]
    never = [(k, g) for k, g in codes.items() if not g["clear"]]
    lines = [f"Diagnostic codes {label}: {_plural(len(codes), 'different code')} - {len(active)} active now (latest event is 'set'), {len(cleared)} cleared and not seen since, {len(back)} came back after being cleared, {len(never)} never cleared."]
    if active: lines.append("Active: " + "; ".join(f"{name(k, g)} (set {_plural(len(g['set']), 'time')}, latest {A.stamp(g['set'][-1])})" for k, g in sorted(active, key=lambda x: -x[1]["set"][-1])[:6]))
    if back: lines.append("Returned after clearing: " + "; ".join(f"{name(k, g)} (cleared {_plural(len(g['clear']), 'time')}, set again {A.stamp(g['set'][-1])})" for k, g in back[:5]))
    if cleared: lines.append("Cleared, not seen since: " + "; ".join(f"{name(k, g)} (cleared {A.stamp(g['clear'][-1])})" for k, g in cleared[:5]))
    if note and not active: lines.append("No code is active now. " + note)
    elif note: lines.append(note)
    lines.append("Based on the recorded set / clear events; a cleared code is not proof that the fault is gone.")
    return _r(res, "\n".join(lines), "health:diagnostic_status", frame, codes=len(codes), active=[k for k, _ in active], recurring=[k for k, _ in back], never_cleared=[k for k, _ in never])


def code_history(res, c, frame, lit):
    """A diagnostic code typed in the question: its history (set, cleared, returned) over the period named - the whole record by default."""
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("all")); want = [x.upper() for x in lit.codes]; klass = list(lit.code_classes) if not want else []
    got = _code_events(b, lo, hi, want or None); table = _dtc_table(b)
    if klass: got = {k: g for k, g in got.items() if k[:1] in klass}; want = [f"{k}-class codes" for k in klass]
    if not got: return _r(res, f"{', '.join(want)} {'was' if len(want) == 1 and not klass else 'were'} not set or cleared {label}.", f"events:{c.id}:code", frame, ok=False, count=0, codes=want)
    parts = []; total = 0
    for code, g in got.items():
        st, back = _state(g); total += len(g["set"]); txt = g["text"] or (table.get(code) or {}).get("text", "")
        parts.append(f"{code} {txt}: set {_plural(len(g['set']), 'time')} ({A.stamp(g['set'][0])} to {A.stamp(g['set'][-1])})" if g["set"] else f"{code} {txt}: never set in this period")
        parts[-1] += (f", cleared {_plural(len(g['clear']), 'time')} (latest {A.stamp(g['clear'][-1])})" if g["clear"] else ", never cleared") + f"; now {st}" + ("; it came back after being cleared" if back else "")
    head = "\n".join(parts)
    if frame.op == "count": head = f"Set {_plural(total, 'time')} {label}.\n" + head
    return _r(res, head + "\nBased on the recorded set / clear events of that code.", f"events:{c.id}:code", frame, codes=list(got), count=total, state={k: _state(g)[0] for k, g in got.items()})


# ------------------------------------------------------------------------------------------------------------------ how long a source was silent
def event_stat(res, c, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz)
    if frame.op == "count":
        from .frame import Frame
        return res.answer("", Frame("events", frame.concept, "count", frame.cond, frame.vs, frame.ent), list(lit.times) if lit.times else None)
    if frame.op == "duration" and c.registry_key == "SOURCE_SILENT":
        lo, hi, label = _window(b, qc, lit, TimeSpec("relative", 30, "day"))
        have = b.c.execute("SELECT 1 FROM sqlite_master WHERE name='source_log'").fetchone()
        rows = b.c.execute("SELECT ts, source, status FROM source_log WHERE ts>=? AND ts<? ORDER BY source, ts", (lo, hi)).fetchall() if have else []
        total = {}; eps = {}; open_at = {}
        for ts, src, st in rows:
            if st == "silent": open_at[src] = ts
            elif st == "restored" and src in open_at: total[src] = total.get(src, 0) + (ts - open_at.pop(src)); eps[src] = eps.get(src, 0) + 1
        for src, t0 in open_at.items(): total[src] = total.get(src, 0) + max(0, int(b.now) - t0); eps[src] = eps.get(src, 0) + 1
        if not total: return _r(res, f"No data source was recorded as silent {label}.", f"stat:{c.id}:duration", frame, ok=False)
        def dur(s): return f"{int(s // 60)} min {int(s % 60):02d} s" if s >= 60 else f"{int(s)} s"
        names = "; ".join(f"{src}: {dur(t)} over {_plural(eps[src], 'episode')}" for src, t in sorted(total.items(), key=lambda kv: -kv[1]))
        return _r(res, f"Data sources were silent for {dur(sum(total.values()))} in total {label} ({names}).\nBased on the source-health log: a source counts as silent after it missed its usual period while the vehicle was awake; the cause is not recorded.",
                  f"stat:{c.id}:duration", frame, seconds=sum(total.values()), sources=total, episodes=eps)
    return _gap(frame, c, "how long that event lasted is not recorded (only the moment it happened)")


# ------------------------------------------------------------------------------------------------------------------ where a measurement comes from
def signal_spec(res, c, frame, lit):
    b = res.brain; row = b.c.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()
    try: prof = json.loads(row[0]) if row else None
    except (ValueError, TypeError): prof = None
    cd = R.CONCEPTS.get(c.id); unit = "" if cd is None or cd.unit in ("none", "category") else f" in {cd.unit}"; rng = f" (plausible range {cd.lo:g} to {cd.hi:g})" if cd is not None and cd.lo is not None and cd.hi is not None else ""
    if prof is None: return _r(res, f"{_cap(c.label)}{unit}{rng}. This database does not record which sources it was decoded from.", f"spec:{c.id}:source", frame, ok=False)
    bindings = [s for s in prof.get("signals", []) if s.get("concept") == c.id]
    if not bindings: return _r(res, f"This vehicle's data sources do not provide {c.label}.", f"spec:{c.id}:source", frame, ok=False)
    srcs = {s["id"]: s.get("name", s["id"]) for s in prof.get("sources", [])}; bindings.sort(key=lambda s: s.get("priority", 1)); lines = []
    for s in bindings:
        kind, _, rest = s["from"].partition(":"); sid = rest.split("/")[0] if "/" in rest else rest; key = rest.split("/", 1)[1] if "/" in rest else rest
        lines.append(f"{'primary' if s.get('priority', 1) == 1 else 'secondary'}: {srcs.get(sid, sid)} ({kind}), {key}" + (f", {s['rate_hz']:g} Hz" if s.get("rate_hz") else "") + (f", unit {s['unit']}" if s.get("unit") else ""))
    note = f" When the sources disagree by more than {bindings[1].get('conflict_abs', 'a tolerance')} for 5 s, a sensor conflict is recorded and both values are kept." if len(bindings) > 1 else ""
    return _r(res, f"{_cap(c.label)}{unit}{rng} comes from {_plural(len(bindings), 'source')}:\n" + "\n".join(" - " + x for x in lines) + note, f"spec:{c.id}:source", frame, sources=[s["from"] for s in bindings])


# ------------------------------------------------------------------------------------------------------------------ charging sessions from the ledger
def _place_name(row, places):
    pid = row["place"]
    if pid is not None and pid in places: return places[pid][0]
    if row["lat"] is None or row["lon"] is None: return "an unlisted place"
    best = None
    for nm, la, lo_ in places.values():
        d = ((la - row["lat"]) * 111.0) ** 2 + ((lo_ - row["lon"]) * 111.0 * math.cos(math.radians(row["lat"]))) ** 2
        if best is None or d < best[0]: best = (d, nm)
    return best[1] if best is not None and best[0] < 1.5 ** 2 else "an unlisted place"


def charge_events(res, c, frame, lit):
    b = res.brain; qc = b.qc(); A.set_tz(b.tz); lo, hi, label = _window(b, qc, lit, TimeSpec("all") if frame.op in ("last", "first", "exists") else TimeSpec("relative", 90, "day"))
    cols = ("start", "end", "kind", "kw", "kwh", "soc0", "soc1", "place", "lat", "lon")
    rows = [dict(zip(cols, r)) for r in b.c.execute("SELECT start, end, kind, kw, kwh, soc0, soc1, place, lat, lon FROM charges WHERE start>=? AND start<? ORDER BY start", (lo, hi))]
    word = "charging session"
    if frame.cond == "fast_charging": rows = [r for r in rows if r["kind"] == "DC"]; word = "fast (DC) charging session"
    elif frame.cond == "not_fast_charging": rows = [r for r in rows if r["kind"] != "DC"]; word = "slow (AC) charging session"
    if not rows: return _r(res, f"No {word} is recorded {label}.", f"events:{c.id}", frame, ok=False, count=0)
    have = b.c.execute("SELECT 1 FROM sqlite_master WHERE name='places'").fetchone()
    places = {r[0]: (r[1], r[2], r[3]) for r in b.c.execute("SELECT id, name, lat, lon FROM places")} if have else {}
    def one(r):
        mins = (r["end"] - r["start"]) / 60.0; soc = f", {r['soc0']:.0f} to {r['soc1']:.0f} %" if r["soc0"] is not None and r["soc1"] is not None else ""
        return f"{A.stamp(r['start'])}: {'DC' if r['kind'] == 'DC' else 'AC'} charge, {r['kwh']:.1f} kWh in {mins:.0f} min{soc}, peak {r['kw']:.0f} kW, at {_place_name(r, places)}"
    n = len(rows); wk = max((hi - lo) / (7 * DAY), 1e-9)
    if frame.ent == "place" or frame.act == "extreme":
        cnt = {}
        for r in rows: k = _place_name(r, places); cnt[k] = cnt.get(k, 0) + 1
        order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])); pick = order[0] if frame.op != "min" else order[-1]
        return _r(res, f"You charged {'most' if frame.op != 'min' else 'least'} often at {pick[0]} ({pick[1]} of {_plural(n, word)} {label}); by place: " + "; ".join(f"{k} {v}" for k, v in order[:6]) + ".\nBased on the charging ledger.", f"events:{c.id}:place", frame, place=pick[0], count=pick[1], places=dict(order))
    if frame.op in ("last", "first"):
        r = rows[-1] if frame.op == "last" else rows[0]
        return _r(res, f"The {'latest' if frame.op == 'last' else 'first'} {word} {label}: {one(r)}.\n{_plural(n, word)} {label}.\nBased on the charging ledger.", f"events:{c.id}", frame, start=r["start"], kwh=r["kwh"], count=n)
    if frame.op == "list": return _r(res, f"{_plural(n, word)} {label}" + (" (the latest 5):" if n > 5 else ":") + "\n" + "\n".join(" - " + one(r) for r in rows[-5:][::-1]) + "\nBased on the charging ledger.", f"events:{c.id}", frame, count=n)
    total = sum(r["kwh"] or 0 for r in rows)
    return _r(res, (f"Yes - {_plural(n, word)} {label}." if frame.op == "exists" else f"{_plural(n, word)} {label} ({n / wk:.1f} per week, {total:.0f} kWh in all).") + "\nBased on the charging ledger.", f"events:{c.id}", frame, count=n, per_week=n / wk, kwh=total)


def inventory(b):
    """What the vehicle's database holds: the span of time, the rows of every tier, the records and the signals.  Read from the stored tables themselves (counts), nothing estimated."""
    c = b.c; tz = int(getattr(b, "tz", 0) or 0)
    def n(sql, *a):
        try: return c.execute(sql, a).fetchone()[0] or 0
        except Exception: return None
    def span(table, col="ts"):
        try: lo, hi = c.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
        except Exception: return None
        return (lo, hi) if lo is not None else None
    tiers = [("one-second rows", n("SELECT COUNT(*) FROM sig_1s") + (n("SELECT COUNT(*) FROM lk_1s") or 0)), ("one-minute rows", n("SELECT COUNT(*) FROM sig_1m") + (n("SELECT COUNT(*) FROM lk_1m") or 0))]
    recs = [("trips", n("SELECT COUNT(*) FROM trips")), ("charging sessions", n("SELECT COUNT(*) FROM charges")), ("events and warnings", n("SELECT COUNT(*) FROM events")), ("log lines", n("SELECT COUNT(*) FROM logs")), ("GPS fixes", n("SELECT COUNT(*) FROM gps_1s"))]
    sig = n("SELECT COUNT(*) FROM sig_catalog"); s1 = span("sig_1m") or span("sig_1s"); lines = []
    if s1:
        d = (s1[1] - s1[0]) / 86400.0
        lines.append(f"The database covers {A.stamp(s1[0] - tz).split(' ')[0]} to {A.stamp(s1[1] - tz).split(' ')[0]} ({d:.0f} days).")
    lines.append("Measurements: " + "; ".join(f"{v:,} {k}" for k, v in tiers if v is not None) + (f", for {sig} signals." if sig else "."))
    lines.append("Records: " + "; ".join(f"{v:,} {k}" for k, v in recs if v) + ".")
    facts = {"ok": True, "route": "inventory", "executor": "inventory", "tiers": dict(tiers), "records": dict(recs), "signals": sig}
    return {"answer": " ".join(lines[:1]) + "\n" + "\n".join(lines[1:]), "brief": lines[0], "intent": "INVENTORY", "facts": facts}
