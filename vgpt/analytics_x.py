"""Deterministic analytics for the extended vehicle data (tyres, rain / light, service and wear, driving style, ADAS use, places, health). stdlib + SQLite only.
Same contract as analytics.py: every number comes from here (never from the language model); every tool returns a facts dict; missing tables/columns degrade to {"ok": False, "why": ...}
so a database without the extension tables (e.g. the legacy simulator DB) still works."""
import math, re, sqlite3, statistics
from collections import Counter
from . import analytics as A

WHEELS = ["front left", "front right", "rear left", "rear right"]
TYPICAL_BRAKE_PER_100KM = (2.0, 6.0)             # hard-braking events per 100 km seen in real driving (VIF-AEGIS accelerometer, 0.28 g)


def _m(c, key, default=None):
    try: r = c.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    except sqlite3.Error: return default
    return r[0] if r else default


def _mf(c, key, default=None):
    v = _m(c, key)
    try: return float(v) if v is not None else default
    except (TypeError, ValueError): return default


def _safe(fn):
    def w(*a, **k):
        try: return fn(*a, **k)
        except sqlite3.OperationalError as e: return {"ok": False, "why": "this vehicle has no such data recorded"} if "no such" in str(e) else {"ok": False, "why": str(e)}
    w.__name__ = fn.__name__; w.__doc__ = fn.__doc__; return w


def _cold(kpa, tt10):
    """Gauge pressure reported at tyre temperature `tt10` (C x 10) -> equivalent at 20 C (ideal-gas law). Without a temperature the reading is returned as is."""
    if tt10 is None: return kpa
    return (kpa + 101.325) * 293.15 / (tt10 / 10.0 + 273.15) - 101.325


def haversine_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def sun_elevation(lat, lon, ts):
    """Solar elevation in degrees (NOAA approximation, stdlib)."""
    d = ts / 86400.0 + 2440587.5 - 2451545.0; g = math.radians((357.529 + 0.98560028 * d) % 360); q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360); e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L)); dec = math.asin(math.sin(e) * math.sin(L)); gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(((gmst + lon / 15.0) % 24) * 15.0) - ra; la = math.radians(lat)
    return math.degrees(math.asin(max(-1.0, min(1.0, math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))))


def clear_sky_lux(lat, lon, ts):
    """Ambient light under a clear sky (lux): Haurwitz global irradiance x ~105 lx per W/m2. 0 when the sun is below the horizon."""
    el = sun_elevation(lat, lon, ts)
    if el <= 0: return 0.0
    cz = math.sin(math.radians(el)); return 105.0 * 1098.0 * cz * math.exp(-0.057 / cz)


def light_class(lux, lat, lon, ts):
    el = sun_elevation(lat, lon, ts)
    if el < -6: return "dark", None
    if el < 2: return "dusk or dawn", None
    ref = clear_sky_lux(lat, lon, ts); ratio = lux / ref if ref > 0 else None
    if ratio is None: return "dim", None
    return ("sunny" if ratio >= 0.70 else "partly cloudy" if ratio >= 0.35 else "overcast" if ratio >= 0.12 else "heavily overcast"), round(ratio, 2)


# ------------------------------------------------------------------------------------------------ tyres
@_safe
def tyres(c, now, when="14"):
    sf, sr = _mf(c, "veh.tyre_spec_front"), _mf(c, "veh.tyre_spec_rear")
    if sf is None: return {"ok": False, "why": "this vehicle does not report tyre pressures"}
    r = c.execute("SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr, tt FROM ext_1m WHERE ts<=? AND (tp_fl IS NOT NULL OR tp_fr IS NOT NULL OR tp_rl IS NOT NULL OR tp_rr IS NOT NULL) ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    if not r: return {"ok": False, "why": "this vehicle does not record individual tyre pressures (or none has been recorded yet)"}
    r = list(r)
    for i, col in enumerate(("tp_fl", "tp_fr", "tp_rl", "tp_rr")):                                                  # one wheel missing from the newest row (a partial minute): its own latest reading from the last 15 minutes
        if r[1 + i] is None:
            o = c.execute(f"SELECT {col} FROM ext_1m WHERE ts<=? AND ts>=? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (r[0], r[0] - 900)).fetchone()
            if o: r[1 + i] = o[0]
    if r[5] is None:
        o = c.execute("SELECT tt FROM ext_1m WHERE ts<=? AND ts>=? AND tt IS NOT NULL ORDER BY ts DESC LIMIT 1", (r[0], r[0] - 900)).fetchone()
        if o: r[5] = o[0]
    implausible = [WHEELS[i] for i in range(4) if r[1 + i] is not None and not (0.0 < r[1 + i] / 10.0 <= 2000.0)]
    for i in range(4):
        if WHEELS[i] in implausible: r[1 + i] = None                                                     # a pressure of zero or less is a faulty sensor, not a flat tyre: it is left out and said so
    spec = [sf, sf, sr, sr]; wheels = []; missing = [WHEELS[i] for i in range(4) if r[1 + i] is None and WHEELS[i] not in implausible]; spec_missing = [WHEELS[i] for i in range(4) if spec[i] is None]
    for i in range(4):
        if r[1 + i] is None: continue                                                                # a wheel without a reading is reported as not recorded, never as a number
        raw = r[1 + i] / 10.0; p = _cold(raw, r[5]) if r[5] is not None else raw
        if spec[i] is None:
            wheels.append({"name": WHEELS[i], "kpa": round(p), "raw_kpa": round(raw), "spec": None, "pct": None, "status": "spec_unavailable"})
            continue
        pct = p / spec[i]                    # compare at the reference temperature (20 C), as the spec is a COLD pressure
        wheels.append({"name": WHEELS[i], "kpa": round(p), "raw_kpa": round(raw), "spec": round(spec[i]), "pct": round(100 * pct), "status": "ok" if pct >= 0.92 else "low" if pct >= 0.80 else "critical"})
    lo, hi = A.window(now, when, c)
    ev = c.execute("SELECT ts, type, detail FROM events WHERE type IN ('TPMS_LOW','TYRE_INFLATED') AND ts>=? AND ts<=? ORDER BY ts", (lo, min(hi, now))).fetchall()
    allev = c.execute("SELECT ts, type, detail FROM events WHERE type IN ('TPMS_LOW','TYRE_INFLATED') AND ts<=? ORDER BY ts", (now,)).fetchall()
    active = {}
    for ts, ty, de in allev:                                       # TPMS warning is active from TPMS_LOW until the next TYRE_INFLATED for that wheel
        w = next((n for n in WHEELS if de.startswith(n)), None)
        if w is None: continue
        if ty == "TPMS_LOW": active[w] = ts
        else: active.pop(w, None)
    # slow-leak trend: first reading of each local day (the cold-ish start-up reading) over the last 7 days
    rows = c.execute("SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr, tt FROM ext_1m WHERE ts>=? AND ts<=? ORDER BY ts", (now - 7 * 86400, now)).fetchall(); first = {}
    for row in rows:
        if any(row[1 + k] is None for k in range(4)) or row[5] is None: continue
        first.setdefault((row[0] + A.TZ) // 86400, (row[0],) + tuple(_cold(row[1 + k] / 10.0, row[5]) * 10 for k in range(4)))
    leak = None
    if len(first) >= 4:
        days = sorted(first); xs = [d - days[0] for d in days]
        for i in range(4):
            ys = [first[d][1 + i] / 10.0 for d in days]; mx, my = statistics.fmean(xs), statistics.fmean(ys); sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
            if spec[i] is not None and slope < -1.0 and ys[0] - ys[-1] > 4 and (leak is None or slope < leak["kpa_per_day"]):
                warn_at = 0.80 * spec[i] * 1.05; leak = {"wheel": WHEELS[i], "kpa_per_day": round(slope, 1), "days_to_warning": max(0, round((ys[-1] - warn_at) / -slope, 1)) if ys[-1] > warn_at else 0}
    return {"ok": True, "time": A.stamp(r[0]), "ts": r[0], "wheels": wheels, "not_recorded": missing, "implausible": implausible, "spec_unavailable": spec_missing, "active_warnings": sorted(active), "events": [{"time": A.stamp(t), "type": ty, "detail": d} for t, ty, d in ev], "leak": leak, "when": when}


# ------------------------------------------------------------------------------------------------ rain / light
@_safe
def weather(c, now, when="t", live=False):
    lat, lon = _mf(c, "env.lat"), _mf(c, "env.lon")
    if live:
        r = c.execute("SELECT ts, rain, lux FROM ext_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
        if not r or (r[1] is None and r[2] is None): return {"ok": False, "why": "no rain or light sensor reading has been recorded"}
        age_min = (now - r[0]) / 60.0; out = {"ok": True, "mode": "now", "time": A.stamp(r[0]), "age_min": round(age_min), "rain": r[1], "lux": r[2], "fresh": age_min <= 20}
        if lat is not None and lon is not None and r[2] is not None: out["light"], out["sky_ratio"] = light_class(r[2], lat, lon, r[0])
        amb = c.execute("SELECT amb FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
        if amb: out["amb_c"] = round(amb[0] / 10.0, 1)
        return out
    lo, hi = A.window(now, when, c)
    r = c.execute("SELECT COUNT(rain), SUM(rain>0), SUM(rain>=2), SUM(rain>=3), MAX(rain) FROM ext_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
    if not r or not r[0]: return {"ok": False, "why": "the car was not in use in that period, so its rain and light sensors recorded nothing"}
    days = c.execute("SELECT COUNT(DISTINCT (ts+?)/86400) FROM ext_1m WHERE rain>0 AND ts>=? AND ts<?", (A.TZ, lo, hi)).fetchone()[0]
    tot = c.execute("SELECT COUNT(DISTINCT (ts+?)/86400) FROM ext_1m WHERE rain IS NOT NULL AND ts>=? AND ts<?", (A.TZ, lo, hi)).fetchone()[0]
    ratios = []
    if lat is not None and lon is not None:
        for ts, lux in c.execute("SELECT ts, lux FROM ext_1m WHERE ts>=? AND ts<? AND (ts%300)=0", (lo, hi)).fetchall():
            if lux is not None and sun_elevation(lat, lon, ts) > 8: ratios.append(lux / max(1.0, clear_sky_lux(lat, lon, ts)))
    amb = c.execute("SELECT AVG(amb), MIN(amb), MAX(amb) FROM sig_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
    return {"ok": True, "mode": "period", "when": when, "active_min": r[0], "rain_min": r[1] or 0, "moderate_min": r[2] or 0, "heavy_min": r[3] or 0, "rain_days": days, "days_with_use": tot,
            "sky_ratio_avg": round(statistics.fmean(ratios), 2) if ratios else None, "amb_avg": round(amb[0] / 10.0, 1) if amb and amb[0] is not None else None}


@_safe
def vehicle_info(c, now):
    m = _m(c, "veh.model")
    if not m: return {"ok": True, "model": "60 kWh electric vehicle (simulated, scripted history)", "cap_kwh": 60.0, "soh": None, "legacy": True}
    soh = c.execute("SELECT soh FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    soh_value = soh[0] if soh else None
    try: soh_value = round(float(soh_value), 1) if soh_value is not None and math.isfinite(float(soh_value)) else _mf(c, "veh.soh")
    except (TypeError, ValueError): soh_value = _mf(c, "veh.soh")
    return {"ok": True, "model": m, "cap_kwh": _mf(c, "veh.cap_kwh"), "soh": soh_value, "home_kw": _mf(c, "veh.home_charger_kw"), "sw": _m(c, "veh.sw"), "city": _m(c, "env.city"),
            "tyre_front": _mf(c, "veh.tyre_spec_front"), "tyre_rear": _mf(c, "veh.tyre_spec_rear"), "legacy": False}


# ------------------------------------------------------------------------------------------------ service, wear, 12 V
@_safe
def service(c, now):
    odo_r = c.execute("SELECT ts, odo FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    if not odo_r: return {"ok": False, "why": "no service data has been recorded"}
    last = c.execute("SELECT ts, odo FROM ext_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    odo = (last[1] / 10.0 if last and last[1] is not None else odo_r[1]); ts_now = last[0] if last and last[1] is not None else odo_r[0]
    try:
        odo = float(odo)
        if not math.isfinite(odo): raise ValueError
    except (TypeError, ValueError): return {"ok": False, "why": "odometer data is incomplete"}
    interval, last_km, last_ts = _mf(c, "veh.service_km"), _mf(c, "veh.last_service_km"), _mf(c, "veh.last_service_ts")
    d = c.execute("SELECT ts, pad, soh, v12 FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    def finite_round(v, places):
        try:
            n = float(v)
            return round(n, places) if math.isfinite(n) else None
        except (TypeError, ValueError): return None
    out = {"ok": True, "odo_km": round(odo), "time": A.stamp(ts_now), "pad_pct": finite_round(d[1], 0) if d else None, "soh": finite_round(d[2], 1) if d else None, "v12": finite_round(d[3], 2) if d else None, "sw": _m(c, "veh.sw")}
    if interval is not None and last_km is not None and math.isfinite(interval) and math.isfinite(last_km):
        left = last_km + interval - odo; months = (now - last_ts) / (86400 * 30.4) if last_ts else None
        out.update(service_interval_km=int(interval), km_since_service=round(odo - last_km), km_to_service=round(left), months_since=round(months, 1) if months is not None else None,
                   months_to_service=round(12 - months, 1) if months is not None else None, due=left <= 0 or (months is not None and months >= 12))
    return out


@_safe
def health(c, now, when="14", focus=None):
    """Collect findings across systems. focus in {None, battery, tyres, brakes, camera, electrical, cooling}."""
    lo, hi = A.window(now, when, c); hi = min(hi, now); items = []; focus = focus or "all"

    def add(area, level, text, key=None): items.append({"area": area, "level": level, "text": text, "key": key or area})
    if focus in ("all", "battery", "cooling", "motor"):
        for ts, de in c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' AND ts>=? AND ts<=? ORDER BY ts", (lo, hi)).fetchall():
            if not isinstance(de, str): continue
            low = de.lower()
            if "coolant" in low or "pack" in low or "batt" in low: add("cooling", "problem", f"{de.split(' (')[0]} on {A.stamp(ts)}", "dtc-thermal")
        n_ot = c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts>=? AND ts<=?", (lo, hi)).fetchone()
        if n_ot[0]: add("battery", "problem", f"{n_ot[0]} battery over-temperature warning{'s' if n_ot[0] != 1 else ''} (latest {A.stamp(n_ot[1])})", "overtemp")
        a = c.execute("SELECT AVG(flow) FROM sig_1m WHERE ts>=? AND ts<=? AND mode IN (1,2,3) AND flow>5", (now - 3 * 86400, now)).fetchone()[0]
        b = c.execute("SELECT AVG(flow) FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2,3) AND flow>5", (now - 14 * 86400, now - 7 * 86400)).fetchone()[0]
        if a and b:
            ratio = a / b
            if ratio < 0.65: add("cooling", "problem", f"coolant flow is down {100 * (1 - ratio):.0f}% against two weeks ago", "flow")
            elif ratio < 0.85: add("cooling", "attention", f"coolant flow is down {100 * (1 - ratio):.0f}% against two weeks ago", "flow")
        try: d = c.execute("SELECT soh FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
        except sqlite3.OperationalError: d = None
        soh = None
        if d:
            try:
                soh = float(d[0])
                if not math.isfinite(soh): soh = None
            except (TypeError, ValueError): soh = None
        if soh is not None: add("battery", "ok" if soh >= 85 else "attention" if soh >= 80 else "problem", f"battery health {soh:.1f}% of its original capacity", "soh")
        if focus in ("all", "battery"):
            bt = c.execute("SELECT MAX(batt_max) FROM sig_1m WHERE ts>=? AND ts<=?", (lo, hi)).fetchone()[0]
            if bt is not None: bt /= 10.0; add("battery", "problem" if bt >= 52 else "attention" if bt >= 45 else "ok", f"pack temperature peaked at {bt:.0f} C {A.WHEN_TEXT.get(when, 'in that period')} (warning level 52 C)", "battpeak")
        if focus in ("all", "motor"):
            mt = c.execute("SELECT MAX(motor) FROM sig_1m WHERE ts>=? AND ts<=?", (lo, hi)).fetchone()[0]
            if mt is not None: mt /= 10.0; add("motor", "problem" if mt >= 130 else "attention" if mt >= 110 else "ok", f"drive motor peaked at {mt:.0f} C {A.WHEN_TEXT.get(when, 'in that period')} (normal operation stays below about 110 C)", "motorpeak")
    if focus in ("all", "tyres"):
        try: t = tyres(c, now, when)
        except sqlite3.OperationalError: t = {}
        if t.get("ok"):
            if t["active_warnings"]: add("tyres", "problem", "tyre-pressure warning: " + ", ".join(t["active_warnings"]), "tpms")
            elif t["leak"]: add("tyres", "attention", f"{t['leak']['wheel']} tyre is losing about {-t['leak']['kpa_per_day']:.1f} kPa a day", "leak")
            else:
                low = [w for w in t["wheels"] if w["status"] != "ok"]
                add("tyres", "attention" if low else "ok", ("check pressure: " + ", ".join(w["name"] for w in low)) if low else "tyre pressures are normal", "tyres")
    if focus in ("all", "brakes", "electrical"):
        try: s = service(c, now)
        except sqlite3.OperationalError: s = {}
        if s.get("ok"):
            if focus in ("all", "brakes") and s.get("pad_pct") is not None: add("brakes", "ok" if s["pad_pct"] >= 25 else "attention" if s["pad_pct"] >= 12 else "problem", f"brake pads about {s['pad_pct']}% left", "pads")
            if focus in ("all", "electrical") and s.get("v12") is not None: add("electrical", "ok" if s["v12"] >= 12.3 else "attention" if s["v12"] >= 12.0 else "problem", f"12 V battery {s['v12']} V at rest", "v12")
            if focus == "all" and s.get("due") is not None: add("service", "attention" if s["due"] else "ok", "service is due" if s["due"] else f"next service in about {max(0, s['km_to_service'])} km", "service")
        for ts, de in c.execute("SELECT ts, detail FROM events WHERE type='LOW_12V' AND ts>=? AND ts<=?", (lo, hi)).fetchall(): add("electrical", "attention", f"{de} on {A.stamp(ts)}", "low12v")
        if focus in ("all", "brakes"):
            for ts, de in c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' AND ts>=? AND ts<=? ORDER BY ts", (lo, hi)).fetchall():
                if not isinstance(de, str): continue
                low = de.lower()
                if any(word in low for word in ("brake", "abs", "esc", "stability")):
                    add("brakes", "problem", f"{de.split(' (')[0]} on {A.stamp(ts)}", "brake-dtc:" + de)
    if focus in ("all", "camera"):
        try: cs = A.camera_status(c, now)
        except sqlite3.OperationalError: cs = {}
        if cs.get("ok"): add("camera", "ok" if cs["state"] == "clear" else "problem" if cs["state"] == "blocked" else "attention", "front camera clear" if cs["state"] == "clear" else f"front camera {cs['state']}: {cs['detail']}", "camera")
    if focus == "all":
        known = {"cooling", "battery"}
        for ts, de in c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' AND ts>=? AND ts<=? ORDER BY ts", (lo, hi)).fetchall():
            if not isinstance(de, str): continue
            low = de.lower()
            if not any(k in low for k in ("coolant", "pack", "batt", "tyre", "voltage")): add("diagnostics", "attention", f"{de.split(' (')[0]} on {A.stamp(ts)}", "dtc")
    seen, out = set(), []
    for it in items:                                                                       # one line per key
        if it["key"] in seen: continue
        seen.add(it["key"]); out.append(it)
    order = {"problem": 0, "attention": 1, "ok": 2}; out.sort(key=lambda x: order[x["level"]])
    if not out: return {"ok": False, "why": "no health data is available"}
    return {"ok": True, "when": when, "focus": focus, "items": out, "problems": sum(1 for x in out if x["level"] == "problem"), "attention": sum(1 for x in out if x["level"] == "attention")}


# ------------------------------------------------------------------------------------------------ driving style, ADAS use, regeneration
@_safe
def driving(c, now, when="7"):
    lo, hi = A.window(now, when, c)
    r = c.execute("SELECT COUNT(*), COALESCE(SUM(t.km),0), COALESCE(SUM(t.end-t.start),0), COALESCE(SUM(t.kwh),0), COALESCE(SUM(t.km*t.hwy_share),0), COALESCE(SUM(x.harsh_brake),0), COALESCE(SUM(x.harsh_accel),0), "
                  "COALESCE(SUM(x.harsh_corner),0), COALESCE(SUM(x.regen_kwh),0), SUM(CASE WHEN x.start IS NOT NULL THEN x.adas_share/100.0*t.hwy_share*(t.end-t.start) END), SUM(CASE WHEN x.start IS NOT NULL THEN t.hwy_share*(t.end-t.start) END), COUNT(x.start) "
                  "FROM trips t LEFT JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<?", (lo, hi)).fetchone()
    n, km, secs, kwh, hwykm, hb, ha, hc, regen, adas_s, hwy_s, ext_n = r
    if not n: return {"ok": False, "why": "no trips in that period"}
    adas_dis = c.execute("SELECT COUNT(*) FROM events WHERE type='ADAS_DISENGAGE' AND ts>=? AND ts<?", (lo, hi)).fetchone()[0]
    per = lambda x: round(100 * x / km, 1) if km > 1 else None
    return {"ok": True, "when": when, "trips": n, "km": round(km, 1), "minutes": round(secs / 60), "hwy_km": round(hwykm, 1), "kwh": round(kwh, 1), "brake": int(hb), "accel": int(ha), "corner": int(hc), "brake_100": per(hb), "accel_100": per(ha),
            "regen_kwh": round(regen, 1) if ext_n else None, "regen_share": round(100 * regen / (kwh + regen), 1) if ext_n and kwh + regen > 0 else None, "adas_min": round(adas_s / 60) if adas_s is not None else None, "hwy_min": round(hwy_s / 60) if hwy_s is not None else None, "adas_share": round(100 * adas_s / hwy_s) if hwy_s is not None and hwy_s > 120 else None,
            "adas_disengage": adas_dis, "has_ext": bool(ext_n), "trips_with_ext": ext_n}


# ------------------------------------------------------------------------------------------------ places
def _places(c):
    """Only complete, finite saved places may participate in geometry."""
    out = {}
    for i, n, k, la, lo in c.execute("SELECT id, name, kind, lat, lon FROM places"):
        if not isinstance(n, str) or la is None or lo is None: continue
        try:
            if math.isfinite(la) and math.isfinite(lo) and -90 <= la <= 90 and -180 <= lo <= 180: out[i] = (n, k or "place", la, lo)
        except TypeError: continue
    return out


COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def _bearing(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2); x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _compass(deg): return COMPASS[int((deg + 22.5) // 45) % 8]


GENERIC = {"place", "town", "centre", "center"}


def _mentioned(pl, q):
    """Saved places named in the question, in order of appearance."""
    hits = {}
    for pid, (name, kind, la, lo) in pl.items():
        rxs = []
        if kind == "home": rxs.append(r"\b(home|house|flat|apartment)\b")
        elif kind == "work": rxs.append(r"\b(work|office|workplace)\b")
        elif kind == "far": rxs.append(r"out of town|outstation|\bairport\b")
        for w in re.findall(r"[a-z]{3,}", name.lower()):
            if w not in GENERIC: rxs.append(r"\b" + re.escape(w) + r"s?\b")
        for rx in rxs:
            m = re.search(rx, q)
            if m and (pid not in hits or m.start() < hits[pid]): hits[pid] = m.start()
    return [pid for pid, _ in sorted(hits.items(), key=lambda kv: kv[1])]


def _typical(c, a, b, now):
    rows = c.execute("SELECT t.km, (t.end-t.start)/60.0 FROM trips t JOIN trip_ext x ON x.start=t.start WHERE ((x.from_id=? AND x.to_id=?) OR (x.from_id=? AND x.to_id=?)) AND t.start<=?", (a, b, b, a, now)).fetchall()
    rows = [r for r in rows if r[0] is not None and r[1] is not None]
    if len(rows) < 2: return None
    return {"n": len(rows), "km": round(statistics.median(r[0] for r in rows), 1), "min": round(statistics.median(r[1] for r in rows))}


def _typical_speed(c, now, road_km=10.0):
    """Typical average speed (km/h) for a trip of about this length, from the car's own trips; sensible defaults when there is no comparable trip."""
    rows = [r[0] for r in c.execute("SELECT avg_speed FROM trips WHERE start>=? AND start<=? AND km BETWEEN ? AND ?", (now - 28 * 86400, now, road_km * 0.5, road_km * 1.6)).fetchall() if r[0]]
    if len(rows) >= 3: return max(15.0, statistics.median(rows))
    return 28.0 if road_km < 10 else 42.0 if road_km < 40 else 75.0


GNSS_CURRENT_MAX_AGE = 12 * 3600  # an overnight parked car can remain stationary; older fixes are historical, not current


def latest_fix(c, now):
    """Newest GNSS fix at or before `now`: the continuous 1 Hz stream first (position, course, speed, HDOP), else the per-minute position kept for the whole history."""
    def valid(lat, lon, scale):
        try:
            lat, lon = lat / scale, lon / scale
            return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
        except (TypeError, ValueError): return False
    try:
        rows = c.execute("SELECT ts, lat, lon, hdg, spd, hdop FROM gps_1s WHERE ts<=? ORDER BY ts DESC", (now,))
        for r in rows:
            if valid(r[1], r[2], 1e6) and now - r[0] <= GNSS_CURRENT_MAX_AGE:
                return {"ts": r[0], "lat": r[1] / 1e6, "lon": r[2] / 1e6, "hdg": r[3] / 10.0 if r[3] is not None else None, "spd": r[4] / 10.0 if r[4] is not None else None, "hdop": r[5] / 10.0 if r[5] is not None else None, "src": "gps"}
    except sqlite3.OperationalError: pass
    for r in c.execute("SELECT ts, lat, lon FROM ext_1m WHERE ts<=? ORDER BY ts DESC", (now,)).fetchall():
        if valid(r[1], r[2], 1e5) and now - r[0] <= GNSS_CURRENT_MAX_AGE: return {"ts": r[0], "lat": r[1] / 1e5, "lon": r[2] / 1e5, "spd": None, "hdg": None, "hdop": None, "src": "minute"}
    return None


@_safe
def track(c, now, seconds=1800, max_points=400):
    """Recent GNSS track up to `now` (for the map) + saved places + the newest fix."""
    fix = latest_fix(c, now)
    if not fix: return {"ok": False, "why": "no GPS position recorded"}
    rows = []
    try: rows = c.execute("SELECT ts, lat, lon FROM gps_1s WHERE ts>? AND ts<=? ORDER BY ts", (now - seconds, now)).fetchall(); rows = [(t, la / 1e6, lo / 1e6) for t, la, lo in rows if la is not None and lo is not None and -90e6 <= la <= 90e6 and -180e6 <= lo <= 180e6]
    except sqlite3.OperationalError: pass
    if len(rows) < 3: rows = [(t, la / 1e5, lo / 1e5) for t, la, lo in c.execute("SELECT ts, lat, lon FROM ext_1m WHERE ts>? AND ts<=? ORDER BY ts", (now - max(seconds, 3600), now)).fetchall() if la is not None and lo is not None and -90e5 <= la <= 90e5 and -180e5 <= lo <= 180e5]
    step = max(1, math.ceil(len(rows) / max_points)); pts = [[round(la, 6), round(lo, 6)] for t, la, lo in rows[::step]]
    return {"ok": True, "fix": fix, "points": pts, "places": [{"name": n, "kind": k, "lat": la, "lon": lo} for n, k, la, lo in _places(c).values()]}


@_safe
def places(c, now, when="y", focus="where", query=""):
    pl = _places(c)
    if not pl: return {"ok": False, "why": "no saved places or location history for this vehicle"}
    q = (query or "").lower(); r = latest_fix(c, now)
    out = {"ok": True, "focus": focus, "when": when, "saved": [n for n, k, la, lo in pl.values()]}; pos = None
    if r:
        pos = (r["lat"], r["lon"]); ds = sorted((haversine_km(pos, (la, lo)), pid) for pid, (n, k, la, lo) in pl.items()); d0, pid0 = ds[0]
        if r["spd"] is not None: moving = r["spd"] > 5 and now - r["ts"] < 30
        else:
            sp = c.execute("SELECT speed FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone(); moving = bool(sp and sp[0] > 50 and now - r["ts"] < 300)
        nm = pl[pid0]
        out["here"] = {"time": A.stamp(r["ts"]), "lat": round(pos[0], 4), "lon": round(pos[1], 4), "near": nm[0], "km": round(d0, 2), "at": d0 < 0.4, "moving": moving,
                       "compass": _compass(_bearing((nm[2], nm[3]), pos)) if d0 >= 0.4 else None, "near_id": pid0, "age_min": round((now - r["ts"]) / 60), "src": r["src"], "fix_age_s": int(now - r["ts"]),
                       "heading": round(r["hdg"]) if r["hdg"] is not None else None, "heading_dir": _compass(r["hdg"]) if r["hdg"] is not None else None, "speed": round(r["spd"], 1) if r["spd"] is not None else None, "hdop": r["hdop"]}
        if moving and r["hdg"] is not None:                                      # saved places roughly ahead of the car, nearest first
            ahead = []
            for pid, (n, k, la, lo) in pl.items():
                d = haversine_km(pos, (la, lo)); off = abs((_bearing(pos, (la, lo)) - r["hdg"] + 180) % 360 - 180)
                if d > 0.5 and off <= 40: ahead.append({"name": n, "km": round(d, 1), "off": round(off)})
            out["ahead"] = sorted(ahead, key=lambda x: x["km"])[:3]
        out["nearest"] = [{"name": pl[i][0], "km": round(d, 1), "compass": _compass(_bearing(pos, (pl[i][2], pl[i][3])))} for d, i in ds[:3]]
        city = _m(c, "env.city"); clat, clon = _mf(c, "env.lat"), _mf(c, "env.lon")
        if city and clat is not None and haversine_km(pos, (clat, clon)) < 60: out["here"]["area"] = city
    if focus == "here_only": return out
    if focus in ("distance", "eta", "coords"):
        if not r: return {"ok": False, "why": "no GPS position has been recorded yet"}
        ids = _mentioned(pl, q); here = out["here"]; tg = []
        out["car_from_me"] = bool(re.search(r"\b(my|the) car\b", q) and re.search(r"from (here|me)|to (me|here)|away from me", q))
        for pid in ids[:3]:
            n, k, la, lo = pl[pid]; km = haversine_km(pos, (la, lo)); road = km * 1.3; hist = _typical(c, here["near_id"], pid, now) if here["at"] and here["near_id"] != pid else None
            if hist: road = hist["km"]
            tg.append({"name": n, "kind": k, "lat": round(la, 4), "lon": round(lo, 4), "km": round(km, 1), "road_km": round(road, 1), "compass": _compass(_bearing(pos, (la, lo))), "typical": hist,
                       "eta_min": hist["min"] if hist else round(road / _typical_speed(c, now, road) * 60), "here_is_it": km < 0.4})
        out["targets"] = tg
        if len(ids) >= 2:
            a, b = ids[0], ids[1]; ka = haversine_km((pl[a][2], pl[a][3]), (pl[b][2], pl[b][3])); hist = _typical(c, a, b, now)
            out["pair"] = {"a": pl[a][0], "b": pl[b][0], "km": round(ka, 1), "road_km": hist["km"] if hist else round(ka * 1.3, 1), "typical": hist, "eta_min": hist["min"] if hist else round(ka * 1.3 / _typical_speed(c, now, ka * 1.3) * 60)}
            out["targets"] = []
        return out
    lo_, hi_ = A.window(now, when, c); hi_ = min(hi_, now)
    rows = c.execute("SELECT t.start, t.km, t.end, x.from_id, x.to_id FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<? ORDER BY t.start", (lo_, hi_)).fetchall()
    out["trips"] = [{"time": A.hhmm(s), "day": A.day_of(s), "km": round(km, 1), "min": round((e - s) / 60), "from": pl.get(f, ("somewhere", 0, 0, 0))[0] if f else "a charging stop", "to": pl.get(t, ("somewhere", 0, 0, 0))[0] if t else "a charging stop"} for s, km, e, f, t in rows]
    if focus == "going":                                                      # habit-based guess, clearly labelled as such
        last = c.execute("SELECT x.to_id, t.end FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start<=? AND x.to_id IS NOT NULL ORDER BY t.start DESC LIMIT 1", (now,)).fetchone()
        if last:
            origin = last[0]; wd = ((now + A.TZ) // 86400 + 3) % 7 < 5; hr = ((now + A.TZ) % 86400) / 3600.0; hist = []
            for s_, to in c.execute("SELECT x.start, x.to_id FROM trip_ext x WHERE x.from_id=? AND x.start<=? AND x.to_id IS NOT NULL", (origin, now)).fetchall():
                ls = s_ + A.TZ; wd2 = ((ls // 86400) + 3) % 7 < 5; h2 = (ls % 86400) / 3600.0
                if wd2 == wd and abs(h2 - hr) <= 2.0: hist.append(to)
            if hist:
                top, cnt = Counter(hist).most_common(1)[0]
                if top in pl and origin in pl:
                    out["guess"] = {"to": pl[top][0], "count": cnt, "of": len(hist), "from": pl[origin][0], "weekday": wd}
    return out


@_safe
def eco_speed(c, now, when="7"):
    """Energy per km at different STEADY speeds: 60-second blocks of the 1 Hz tier where the speed barely changed (so stop-and-go does not distort it)."""
    bins = [(20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 250)]; acc = {b: [0.0, 0.0, 0] for b in bins}
    rows = c.execute("SELECT ts, speed, power FROM sig_1s WHERE ts>=? AND ts<=? AND mode IN (1,2) ORDER BY ts", (now - 7 * 86400, now)).fetchall()
    blk = []
    for ts, sp, pw in rows:
        if sp is None or pw is None: continue
        try:
            if not math.isfinite(float(sp)) or not math.isfinite(float(pw)): continue
        except (TypeError, ValueError): continue
        if blk and ts - blk[0][0] >= 60:
            if len(blk) >= 50:
                v = [b[1] / 10.0 for b in blk]; m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
                if sd < 8.0 and m >= 20:
                    for b in bins:
                        if b[0] <= m < b[1]: acc[b][0] += sum(x[2] for x in blk); acc[b][1] += sum(v); acc[b][2] += 1; break
            blk = []
        blk.append((ts, sp, pw))
    out = [{"lo": b[0], "hi": b[1], "wh_km": round(a[0] / a[1], 1), "min": a[2]} for b, a in acc.items() if a[2] >= 3 and a[1] > 0]
    basis = "steady cruising, last 7 days"
    if len(out) < 2:                                                           # older moment: only per-minute averages exist
        lo, hi = A.window(now, "28", c); acc = {b: [0.0, 0.0, 0] for b in bins}
        for sp, pw in c.execute("SELECT speed, power FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) AND speed>=200 AND power IS NOT NULL", (lo, min(hi, now))).fetchall():
            v = sp / 10.0
            for b in bins:
                if b[0] <= v < b[1]: acc[b][0] += pw; acc[b][1] += v; acc[b][2] += 1; break
        out = [{"lo": b[0], "hi": b[1], "wh_km": round(a[0] / a[1], 1), "min": a[2]} for b, a in acc.items() if a[2] >= 20 and a[1] > 0]; basis = "per-minute averages, last 28 days"
    if len(out) < 2: return {"ok": False, "why": "not enough steady driving at different speeds yet"}
    best = min(out, key=lambda x: x["wh_km"]); worst = max(out, key=lambda x: x["wh_km"])
    return {"ok": True, "when": when, "bins": out, "best": best, "worst": worst, "pct": round(100 * (worst["wh_km"] / best["wh_km"] - 1)), "basis": basis}
