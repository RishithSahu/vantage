"""Deterministic analytics tools (stdlib + SQLite). ALL numbers in an answer come from here, never from the language model.
Every tool returns a `facts` dict (machine-checkable) and an `evidence` list (lines an engineer can audit)."""
import json, statistics, zlib, datetime as dt

from . import sqlfeatures

CAP_KWH = 60.0
METRICS = {   # name -> (sig_1m column, scale, unit, label)
    "batt_temp": ("batt", 10, "C", "battery temperature"), "coolant_flow": ("flow", 10, "L/min", "coolant flow"), "speed": ("speed", 10, "km/h", "speed"),
    "soc": ("soc", 10, "%", "state of charge"), "amb": ("amb", 10, "C", "outside temperature"), "hvac": ("hvac", 1, "W", "climate power"), "motor": ("motor", 10, "C", "motor temperature"),
}
EVENT_LABEL = {"ADAS_DISENGAGE": "ADAS disengagement", "WARN_BATT_OVERTEMP": "battery over-temperature warning", "DTC_SET": "diagnostic trouble code", "CHARGE_START": "charging session",
               "CAMERA_BLOCKED": "camera blockage", "CAMERA_DEGRADED": "camera image degradation", "CAMERA_RESTORED": "camera recovery", "CAN_ANOMALY": "CAN bus anomaly"}


TZ = 0                                   # vehicle local time offset from UTC (seconds); set from the database by the Brain


def set_tz(sec): 
    global TZ
    TZ = int(sec)


def _lt(ts): return dt.datetime.fromtimestamp(ts + TZ, dt.timezone.utc)
def day_of(ts): return _lt(ts).strftime("%Y-%m-%d")
def hhmm(ts): return _lt(ts).strftime("%H:%M")
def stamp(ts): return _lt(ts).strftime("%Y-%m-%d %H:%M")


def cap_kwh(c):
    """Usable battery capacity: nominal x state of health when the vehicle reports them, else the legacy 60 kWh."""
    try:
        r = dict(c.execute("SELECT k, v FROM meta WHERE k IN ('veh.cap_kwh','veh.soh')").fetchall())
        if "veh.cap_kwh" in r: return float(r["veh.cap_kwh"]) * float(r.get("veh.soh", 100)) / 100.0
    except Exception: pass
    return CAP_KWH


def window(now, when, c=None):
    """when: y yesterday | t today | we last weekend | sw since last weekend | tw this week | sc since last charge | 7 | 14 | 28 | a -> (lo, hi) epoch seconds (UTC)."""
    ln = now + TZ; today0 = ln - ln % 86400                          # local midnight (local-time epoch) ...
    wd = (today0 // 86400 + 3) % 7                                   # 0 = Monday
    if isinstance(when, str) and when.startswith("d:"):              # one named calendar day (local time)
        try:
            p = [int(x) for x in when[2:].split("-")]; y, mo, d = (p[0], p[1], p[2]) if len(p) == 3 else (dt.datetime.fromtimestamp(ln, dt.timezone.utc).year, p[0], p[1])
            day = dt.datetime(y, mo, d, tzinfo=dt.timezone.utc)
            if len(p) == 2 and day.timestamp() > ln: day = dt.datetime(y - 1, mo, d, tzinfo=dt.timezone.utc)
            lo = int(day.timestamp()) - TZ; return lo, lo + 86400
        except (ValueError, OverflowError): pass
    today0 -= TZ                                                     # ... back to a UTC epoch
    if when in ("we", "sw"):
        sat = today0 - ((wd - 5) % 7) * 86400                        # most recent Saturday
        if wd in (5, 6): sat -= 7 * 86400                            # weekend still running -> the previous one
        return (sat, sat + 2 * 86400) if when == "we" else (sat, now)
    if when == "tw": return today0 - wd * 86400, now
    if when == "sc" and c is not None:
        r = c.execute("SELECT MAX(ts) FROM events WHERE type='CHARGE_START' AND ts<=?", (now,)).fetchone()
        # Never substitute yesterday for the semantically different "since
        # last charge" interval.  An empty interval gives callers an honest
        # insufficient-data result when no charge start was recorded.
        return (r[0] if r and r[0] is not None else now), now
    return {"y": (today0 - 86400, today0), "t": (today0, now), "7": (today0 - 7 * 86400, now), "14": (today0 - 14 * 86400, now), "28": (today0 - 28 * 86400, now), "a": (0, now)}.get(when, (today0 - 14 * 86400, now))


class _WhenText(dict):
    def get(self, k, d=None):
        if isinstance(k, str) and k.startswith("d:"): return "on " + (k[2:] if len(k) > 7 else k[2:].replace("-", "/"))
        return dict.get(self, k, d)


WHEN_TEXT = _WhenText({"y": "yesterday", "t": "today", "we": "last weekend", "sw": "since last weekend", "tw": "this week", "sc": "since your last charge", "7": "in the last 7 days",
             "14": "in the last 14 days", "28": "in the last 28 days", "a": "overall"})


def usage(c, now, when):
    lo, hi = window(now, when, c)
    r = c.execute("SELECT COUNT(*), COALESCE(SUM(km),0), SUM(kwh), COALESCE(SUM(end-start),0), SUM(CASE WHEN kwh IS NULL OR kwh < 0 OR km IS NULL OR km <= 0 THEN 1 ELSE 0 END) FROM trips WHERE start>=? AND start<?", (lo, hi)).fetchone()
    n, km, kwh, secs, incomplete_energy = r
    energy_ok = bool(n) and incomplete_energy == 0 and kwh is not None
    return {"ok": True, "trips": n, "km": round(km, 1), "kwh": round(kwh, 1) if energy_ok else None, "minutes": round(secs / 60), "wh_km": round(1000 * kwh / km, 1) if energy_ok and km > 0.5 else None, "when": when, "lo": lo, "hi": hi,
            "energy_complete": energy_ok, "range_basis_km": round(cap_kwh(c) * 1000 / (1000 * kwh / km)) if energy_ok and km > 0.5 and kwh > 0 else None}


def summary(c, metric, now, when):
    lo, hi = window(now, when, c); col, sc, unit, label = METRICS[metric]
    modes = "" if metric == "amb" else (" AND mode IN (1,2)" if metric == "speed" else " AND mode IN (1,2,3)")
    hi_col = "batt_max" if metric == "batt_temp" else col; lo_col = "flow_min" if metric == "coolant_flow" else col
    if metric == "speed":
        # A maximum/minimum of minute averages is not a speed extremum.  Use
        # the raw tier only when it spans the requested interval; otherwise a
        # precise-looking value would overstate what the retained data proves.
        coverage = c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
        if (not coverage or coverage[0] is None or coverage[0] > lo) and hi - lo <= 45 * 86400:          # the seconds that expired from the database are in the archive: load them for this question
            from . import archive
            if archive.hydrate(c, lo, hi)["rows"]: coverage = c.execute("SELECT MIN(ts), MAX(ts) FROM sig_1s WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
        if not coverage or coverage[0] is None or coverage[0] > lo or coverage[1] < hi - 1:
            # The average of the minute tier is sound over any period (equal-weight minutes); only the extremes need the raw tier, so they are withheld, not the whole answer.
            r = c.execute(f"SELECT AVG({col}), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<?{modes}", (lo, hi)).fetchone()
            if not r or not r[1] or r[0] is None: return {"ok": False, "why": "no moving-speed data for that period"}
            return {"ok": True, "metric": metric, "avg": round(r[0] / sc, 1), "min": None, "max": None, "exact_extremes": False, "unit": unit, "minutes": r[1], "when": when}
        r = c.execute("SELECT AVG(speed), MIN(speed), MAX(speed), COUNT(*) FROM sig_1s WHERE ts>=? AND ts<? AND mode IN (1,2)", (lo, hi)).fetchone()
    else:
        r = c.execute(f"SELECT AVG({col}), MIN({lo_col}), MAX({hi_col}), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<?{modes}", (lo, hi)).fetchone()
    if not r or r[3] == 0 or any(v is None for v in r[:3]): return {"ok": False, "why": f"incomplete {label} data for that period"}
    return {"ok": True, "metric": metric, "avg": round(r[0] / sc, 1), "min": round(r[1] / sc, 1), "max": round(r[2] / sc, 1), "unit": unit, "minutes": r[3], "when": when}


def est_range(c, now):
    r = c.execute("SELECT ts, soc FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone() or c.execute("SELECT ts, soc FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()   # 1 Hz tier is only kept for 7 days
    if not r: return {"ok": False, "why": "no battery data"}
    lo = now - 7 * 86400
    k = c.execute("SELECT COALESCE(SUM(km),0), SUM(kwh), SUM(CASE WHEN kwh IS NULL OR kwh < 0 OR km IS NULL OR km <= 0 THEN 1 ELSE 0 END) FROM trips WHERE start>=? AND start<=?", (lo, now)).fetchone()
    basis = 7
    if k[0] < 20: k = c.execute("SELECT COALESCE(SUM(km),0), SUM(kwh), SUM(CASE WHEN kwh IS NULL OR kwh < 0 OR km IS NULL OR km <= 0 THEN 1 ELSE 0 END) FROM trips WHERE start<=?", (now,)).fetchone(); basis = 0
    if k[0] < 5 or k[1] is None or k[2]: return {"ok": False, "why": "not enough complete energy history to estimate range"}
    wh_km = 1000 * k[1] / k[0]; soc = r[1] / 10
    return {"ok": True, "range_km": round(soc / 100 * cap_kwh(c) * 1000 / wh_km), "soc": round(soc, 1), "wh_km": round(wh_km, 1), "basis_days": basis, "time": stamp(r[0])}


def daily_efficiency(c, lo, hi):
    rows = c.execute(f"SELECT ((start+({TZ}))/86400)*86400-({TZ}) d, SUM(kwh), SUM(km), AVG(avg_amb), AVG(avg_hvac_w), AVG(hwy_share) FROM trips WHERE start>=? AND start<? AND kwh IS NOT NULL AND kwh >= 0 AND km > 0 GROUP BY d ORDER BY d", (lo, hi)).fetchall()
    return [dict(day=day_of(d), ts=d, wh_km=1000 * k / max(m, 0.1), amb=a, hvac=h, hwy=hs) for d, k, m, a, h, hs in rows if m and m > 0.5]


def daily_signal(c, metric, lo, hi):
    col, sc, unit, _ = METRICS[metric]
    modes = "" if metric == "amb" else " AND mode IN (1,2,3)"
    rows = c.execute(f"SELECT ((ts+({TZ}))/86400)*86400-({TZ}) d, AVG({col}) FROM sig_1m WHERE ts>=? AND ts<?{modes} GROUP BY d ORDER BY d", (lo, hi)).fetchall()
    return [dict(day=day_of(d), ts=d, v=v / sc) for d, v in rows if v is not None]


def trend(c, metric, now, when="14"):
    lo, hi = window(now, when, c)
    if metric in ("energy_use", "range"):
        s = daily_efficiency(c, lo, hi)
        if len(s) < 2: return {"ok": False, "why": "not enough trips"}
        n = max(1, min(3, len(s) // 2)); a, b = statistics.fmean(x["wh_km"] for x in s[:n]), statistics.fmean(x["wh_km"] for x in s[-n:])
        if a <= 0 or b <= 0: return {"ok": False, "why": "consumption baseline is zero or missing"}
        f = {"ok": True, "metric": "energy_use", "first": round(a, 1), "last": round(b, 1), "change_pct": round(100 * (b - a) / a, 1), "unit": "Wh/km", "days": len(s)}
        f["range_first_km"], f["range_last_km"] = round(cap_kwh(c) * 1000 / a), round(cap_kwh(c) * 1000 / b)
        return f
    s = daily_signal(c, metric, lo, hi)
    if len(s) < 2: return {"ok": False, "why": "no data"}
    n = max(1, min(3, len(s) // 2)); a, b = statistics.fmean(x["v"] for x in s[:n]), statistics.fmean(x["v"] for x in s[-n:])
    return {"ok": True, "metric": metric, "first": round(a, 1), "last": round(b, 1), "change_pct": round(100 * (b - a) / a, 1) if a else None, "unit": METRICS[metric][2], "days": len(s)}


def largest_change(c, metric, now, when="a", worse=True):
    lo, hi = window(now, when, c)
    if metric in ("energy_use", "range"):
        s = daily_efficiency(c, lo, hi); key = "wh_km"
    else:
        s = daily_signal(c, metric, lo, hi); key = "v"
    if len(s) < 2: return {"ok": False, "why": "no data"}
    d = [(s[i][key] - s[i - 1][key], s[i]) for i in range(1, len(s))]
    delta, row = (max(d, key=lambda x: x[0]) if worse else min(d, key=lambda x: x[0]))
    return {"ok": True, "metric": "energy_use" if key == "wh_km" else metric, "day": row["day"], "delta": round(delta, 1), "value": round(row[key], 1), "unit": "Wh/km" if key == "wh_km" else METRICS[metric][2]}


def events(c, etype, now, when="y"):
    lo, hi = window(now, when, c)
    rows = c.execute("SELECT ts, sev, detail FROM events WHERE type=? AND ts>=? AND ts<? ORDER BY ts", (etype, lo, hi)).fetchall()
    out = {"ok": True, "type": etype, "count": len(rows), "when": when, "items": [{"ts": t, "time": stamp(t), "detail": d, "sev": s} for t, s, d in rows]}
    if etype == "CHARGE_START":
        try:
            r = c.execute("SELECT COUNT(*), COALESCE(SUM(kwh),0), COALESCE(SUM(CASE WHEN kind='DC' THEN kwh END),0) FROM charges WHERE start>=? AND start<?", (lo, hi)).fetchone()
            if r and r[0]: out["kwh"], out["dc_kwh"] = round(r[1], 1), round(r[2], 1)
        except Exception: pass
    return out


def _baseline(c, col, mode, ts, days=7):
    rows = [r[0] for r in c.execute(f"SELECT {col} FROM sig_1m WHERE ts<? AND ts>=? AND mode=?", (ts - 3600, ts - days * 86400, mode)).fetchall() if r[0] is not None]
    if len(rows) < 20: rows = [r[0] for r in c.execute(f"SELECT {col} FROM sig_1m WHERE ts<? AND ts>=? AND mode IN (1,2,3)", (ts - 3600, ts - days * 86400)).fetchall() if r[0] is not None]
    if len(rows) < 5: return None, None
    return statistics.fmean(rows), (statistics.pstdev(rows) or 1.0)


KB = [   # (predicate on dict of z-scores / context, cause text) - small expert knowledge base, not an LLM
    (lambda z, x: z.get("coolant_flow", 0) < -3 and z.get("batt_temp", 0) > 2, "Coolant flow collapsed while the pack heated - cooling-circuit or pump degradation (check pump / coolant level)."),
    (lambda z, x: x.get("dc_charge") and z.get("batt_temp", 0) > 2, "A high-power DC fast charge was running, raising the heat load."),
    (lambda z, x: z.get("speed", 0) > 3 and z.get("batt_temp", 0) > 2, "Sustained high-speed driving raised pack heat."),
    (lambda z, x: z.get("amb", 0) > 3, "Unusually high ambient temperature."),
]


def before_event(c, etype, now, minutes=60, which="last"):
    row = c.execute(f"SELECT ts, detail FROM events WHERE type=? AND ts<=? ORDER BY ts {'DESC' if which == 'last' else 'ASC'} LIMIT 1", (etype, now)).fetchone()
    if not row: return {"ok": False, "why": f"no {EVENT_LABEL.get(etype, etype)} found"}
    ts, detail = row
    ctx = c.execute("SELECT blob FROM event_ctx WHERE ts=?", (ts,)).fetchone()
    cols = ["ts", "speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"]
    if ctx: rows = json.loads(zlib.decompress(ctx[0])); src = "frozen event context (1 Hz)"
    else:
        rows = c.execute("SELECT ts,speed,soc,power,batt,flow,amb,hvac,motor,mode FROM sig_1s WHERE ts BETWEEN ? AND ?", (ts - minutes * 60, ts)).fetchall(); src = "1 Hz history"
    rows = [r for r in rows if ts - minutes * 60 <= r[0] <= ts]
    if len(rows) < 10: return {"ok": False, "why": "no signal history around that event"}
    mode = rows[-1][9]; z, dev = {}, []
    for name, idx, sc in (("batt_temp", 4, 10), ("coolant_flow", 5, 10), ("speed", 1, 10), ("soc", 2, 10), ("amb", 6, 10)):
        mu, sd = _baseline(c, {"batt_temp": "batt", "coolant_flow": "flow", "speed": "speed", "soc": "soc", "amb": "amb"}[name], mode, ts)
        if mu is None: continue
        values = [r[idx] for r in rows if r[idx] is not None]
        if len(values) < 2: continue
        first, last = values[0] / sc, values[-1] / sc; zz = (last - mu / sc) / (sd / sc)
        z[name] = round(zz, 1); dev.append(dict(signal=name, start=round(first, 1), end=round(last, 1), z=round(zz, 1), unit=METRICS[name][2]))
    dev.sort(key=lambda d: -abs(d["z"]))
    ev_around = c.execute("SELECT ts,type,detail FROM events WHERE ts BETWEEN ? AND ? AND ts<=? ORDER BY ts", (ts - 3 * 3600, ts, ts)).fetchall()
    dc = any(t == "CHARGE_START" and "DC" in d for _, t, d in ev_around)
    logs = c.execute("SELECT ts, src, msg FROM logs WHERE ts BETWEEN ? AND ? ORDER BY ts LIMIT 400", (ts - minutes * 60, ts)).fetchall()
    first_log = next(((t, m) for t, s, m in logs if "flow low" in m), None)
    causes = [txt for pred, txt in KB if pred(z, {"dc_charge": dc})]
    timeline = [dict(ts=t, time=hhmm(t), what=f"{EVENT_LABEL.get(ty, ty)}: {d}") for t, ty, d in ev_around if ty != "ADAS_DISENGAGE"]
    return {"ok": True, "type": etype, "ts": ts, "time": stamp(ts), "detail": detail, "source": src, "deviations": dev[:4], "timeline": timeline, "dc_fast_charge": dc,
            "first_log": (hhmm(first_log[0]), first_log[1]) if first_log else None, "causes": causes[:2], "window_min": minutes}


def compare_modes(c, now, when="a"):
    lo, hi = window(now, when, c)
    rows = c.execute("SELECT mode, SUM(power)/60.0, SUM(speed)/10.0/60.0, COUNT(*) FROM sig_1m WHERE mode IN (1,2) AND speed>30 AND ts>=? AND ts<? GROUP BY mode", (lo, hi)).fetchall()
    d = {m: (wh / km, n) for m, wh, km, n in rows if km is not None and km > 0 and wh is not None}
    if 1 not in d or 2 not in d: return {"ok": False, "why": "not enough city/highway data"}
    if d[1][0] == 0: return {"ok": False, "why": "city consumption baseline is zero"}
    return {"ok": True, "city": round(d[1][0], 1), "highway": round(d[2][0], 1), "diff_pct": round(100 * (d[2][0] - d[1][0]) / d[1][0], 1), "unit": "Wh/km", "minutes": d[1][1] + d[2][1]}


def why_range(c, now, when="14"):
    lo, hi = window(now, when, c)
    s = daily_efficiency(c, lo, hi)
    if len(s) < 4: return {"ok": False, "why": "not enough trips in that period"}
    n = max(1, min(3, len(s) // 2)); A, B = s[:n], s[-n:]; m = lambda L, k: statistics.fmean(x[k] for x in L)
    e0, e1 = m(A, "wh_km"), m(B, "wh_km")
    drag = c.execute("SELECT AVG(wh_km) FROM trips WHERE hwy_share>0.9").fetchone()[0]
    speed0 = statistics.fmean(r[0] for r in c.execute("SELECT avg_speed FROM trips WHERE start>=? AND start<?", (A[0]["ts"], A[-1]["ts"] + 86400)).fetchall()) or 40
    speed1 = statistics.fmean(r[0] for r in c.execute("SELECT avg_speed FROM trips WHERE start>=? AND start<?", (B[0]["ts"], B[-1]["ts"] + 86400)).fetchall()) or 40
    hvac_wh0, hvac_wh1 = m(A, "hvac") / speed0, m(B, "hvac") / speed1                                 # Wh per km spent on climate
    if e0 <= 0 or e1 <= 0: return {"ok": False, "why": "consumption baseline is zero or missing"}
    f = {"ok": True, "first": round(e0, 1), "last": round(e1, 1), "increase_pct": round(100 * (e1 - e0) / e0, 1), "range_first_km": round(cap_kwh(c) * 1000 / e0), "range_last_km": round(cap_kwh(c) * 1000 / e1),
         "amb_first": round(m(A, "amb"), 1), "amb_last": round(m(B, "amb"), 1), "hwy_first": round(100 * m(A, "hwy")), "hwy_last": round(100 * m(B, "hwy")),
         "hvac_wh_km_first": round(hvac_wh0, 1), "hvac_wh_km_last": round(hvac_wh1, 1)}
    f["factors"] = [("Climate/heating load", round(hvac_wh1 - hvac_wh0, 1)), ("Highway share", f["hwy_last"] - f["hwy_first"])]
    day, delta = None, 0
    step = largest_change(c, "energy_use", now, when)
    if step.get("ok"): f["step_day"], f["step_delta"] = step["day"], step["delta"]
    return f


def status(c, metric, now=None):
    col, sc, unit, label = METRICS[metric]
    bound = now if now is not None else 1 << 40
    r = c.execute(f"SELECT ts, {col} FROM sig_1s WHERE ts<=? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (bound,)).fetchone() or c.execute(f"SELECT ts, {col} FROM sig_1m WHERE ts<=? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (bound,)).fetchone()
    return {"ok": bool(r), "metric": metric, "value": round(r[1] / sc, 1), "unit": unit, "time": stamp(r[0])} if r else {"ok": False, "why": "no data"}


def search_logs(c, text, limit=5):
    toks = [t for t in "".join(ch if ch.isalnum() else " " for ch in text).split() if len(t) > 2][:6]
    if not toks: return []
    return sqlfeatures.search_logs(c, toks, limit)                                                          # the FTS5 index when this SQLite has one, a substring match when not (vgpt/sqlfeatures.py)


# ------------------------------------------------------------------------------------------------ camera feed (metadata only)
def camera_status(c, now):
    """Current camera state from the last camera event at or before `now` + the latest per-minute quality metrics."""
    try:
        ev = c.execute("SELECT ts, type, detail FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED','CAMERA_RESTORED') AND ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
        m = c.execute("SELECT ts, block_avg, sharp_avg, lane_avg, frames FROM cam_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    except Exception:
        return {"ok": False, "why": "no camera data has been recorded"}
    if ev is None and m is None: return {"ok": False, "why": "no camera data has been recorded"}
    newest = max(x[0] for x in (ev, m) if x is not None)
    if now - newest > 20 * 60:
        age = int((now - newest) // 60); ago = f"{age // 60} h {age % 60} min" if age >= 60 else f"{age} min"
        return {"ok": False, "why": f"camera evidence is stale (the newest camera record is {ago} old); its current state is unknown", "age_min": round((now - newest) / 60, 1)}
    state, since, detail = "clear", None, ""
    if ev and ev[1] in ("CAMERA_BLOCKED", "CAMERA_DEGRADED"):
        state, since, detail = ("blocked" if ev[1] == "CAMERA_BLOCKED" else "degraded"), ev[0], ev[2]
    out = {"ok": True, "state": state, "since": stamp(since) if since else None, "detail": detail}
    if m: out.update(lane_avg=round(m[3], 2), block_avg=round(m[1], 2), sharp_avg=round(m[2], 2), minute=stamp(m[0]), frames=m[4])
    if ev: out["last_event"] = {"time": stamp(ev[0]), "type": ev[1], "detail": ev[2]}
    return out


def camera_summary(c, now, when="7"):
    lo, hi = window(now, when, c)
    try:
        r = c.execute("SELECT SUM(frames), SUM(block_avg*frames)/NULLIF(SUM(frames),0), MAX(block_max), SUM(lane_avg*frames)/NULLIF(SUM(frames),0), SUM(sharp_avg*frames)/NULLIF(SUM(frames),0), COUNT(*) FROM cam_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
        n_ev = c.execute("SELECT type, COUNT(*) FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED') AND ts>=? AND ts<? GROUP BY type", (lo, hi)).fetchall()
    except Exception:
        return {"ok": False, "why": "no camera data has been recorded"}
    if not r or not r[0]: return {"ok": False, "why": "no camera data in that period"}
    return {"ok": True, "when": when, "frames": int(r[0]), "minutes": r[5], "block_avg": round(r[1] or 0, 3), "block_max": round(r[2] or 0, 2), "lane_avg": round(r[3] or 0, 2), "sharp_avg": round(r[4] or 0, 2),
            "blocked_events": dict(n_ev).get("CAMERA_BLOCKED", 0), "degraded_events": dict(n_ev).get("CAMERA_DEGRADED", 0)}
