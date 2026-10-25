"""What the service engineer, the fleet operator and the validation team get that a driver does not.

    service_report(conn)      one vehicle: diagnostic codes with their history, warnings, component health with the evidence for each verdict, service due, and the checks to run
    fleet_overview(paths)     many vehicles: utilisation, efficiency, risk events and open alerts, and which vehicle needs attention - with no driver, no places, no pictures
    validation_report(conn)   one vehicle: coverage and quality of every signal, the protocol sources, events, and the injected faults scored against what the vehicle recorded
    evidence_pack(...)        a question, its answer and the data behind it, signed

Every number is read from the recorded history (read-only), at the vehicle's own moment (`now`), and every verdict says what it was based on.  Nothing here reads a question; nothing here writes."""
import hashlib
import json
import os
import sqlite3
import statistics
import time

from .vehicle import caps as CAPS

DAY = 86400
SEVERE = ("critical",)
WARNS = ("warn", "critical")


def _q(conn, sql, args=()):
    try: return conn.execute(sql, args).fetchall()
    except sqlite3.Error: return []


def _one(conn, sql, args=()):
    r = _q(conn, sql, args); return r[0] if r else None


def _meta(conn):
    return {k: v for k, v in _q(conn, "SELECT k, v FROM meta")}


def _num(x, default=None):
    try: return float(x)
    except (TypeError, ValueError): return default


def _now(conn):
    """The vehicle's own 'now': the end of its recorded history."""
    rs = [_one(conn, f"SELECT MAX(ts) FROM {t}") for t in ("sig_1s", "sig_1m", "events")]                                   # while a vehicle is delivering, the newest rows are in the 1 Hz table
    return int(max(x[0] for x in rs if x and x[0]) if any(x and x[0] for x in rs) else time.time())


def stamp(ts, tz_h=0.0):
    return time.strftime("%d %b %Y %H:%M", time.gmtime(int(ts) + int(tz_h * 3600)))


def active_codes(conn, lo=0, hi=None):
    """{code: {"text", "set", "cleared", "first", "last", "state"}} from the DTC_SET / DTC_CLEAR events; the code is the first word of the detail (how every other history here reads it)."""
    hi = hi or (_now(conn) + 1); meta = _meta(conn)
    try: table = json.loads(meta.get("vehicle.dtc", "{}"))
    except (TypeError, ValueError): table = {}
    out = {}
    for ts, ty, det in _q(conn, "SELECT ts, type, detail FROM events WHERE type IN ('DTC_SET','DTC_CLEAR') AND ts>=? AND ts<? ORDER BY ts", (lo, hi)):
        code = str(det).split(" ")[0].upper(); g = out.setdefault(code, {"code": code, "text": "", "set": 0, "cleared": 0, "first": ts, "last": ts, "state": "cleared"})
        g["last"] = ts
        if ty == "DTC_SET":
            g["set"] += 1; g["state"] = "active"
            if not g["text"]: g["text"] = str(det)[len(code):].strip().split(" (status")[0]
        else: g["cleared"] += 1; g["state"] = "cleared"
    for code, g in out.items():
        t = table.get(code) or {}; g["text"] = g["text"] or t.get("text", ""); g["subsystem"] = t.get("subsystem", "")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------- service
SERVICE_STEPS = {
    "codes": "Read the active trouble codes with a scan tool, note the freeze-frame data, then clear and re-test to see whether they return.",
    "warnings": "Open the event record around the most recent critical warning and check what the vehicle was doing at the time.",
    "tyres": "Check pressures cold against the placard, inspect the low tyre for a puncture or valve leak, and look at the trend for a slow leak.",
    "v12": "Load-test the 12 V battery and check the alternator / DC-DC converter output; look for brown-out resets in the record.",
    "brakes": "Measure pad thickness on every wheel and book the pad replacement before the wear limit.",
    "traction": "Run the high-voltage battery health report (cell balance, capacity) and check the cooling loop before the next hot spell.",
    "thermal": "Check coolant level and pump operation, and inspect the radiator and fans; compare the coolant temperature with load.",
    "dpf": "Check the particulate filter soot load and drive the regeneration cycle; inspect the differential-pressure sensor if it does not complete.",
    "adblue": "Refill the AdBlue / DEF tank and check the dosing system for faults.",
    "fuel_trim": "Check for intake leaks and a failing oxygen sensor; long-term trims beyond about +/-10 % indicate the mixture is being corrected hard.",
    "camera": "Clean and inspect the windscreen and camera housing, then recalibrate the forward camera if it stays degraded.",
    "service": "Book the scheduled service: oil / filters or brake fluid and coolant as the maintenance plan lists, and a general inspection.",
}


def _status(v, watch, attention, low_is_bad=False):
    if v is None: return "unknown"
    if low_is_bad: return "attention" if v <= attention else "watch" if v <= watch else "ok"
    return "attention" if v >= attention else "watch" if v >= watch else "ok"


def service_report(conn, tz_h=0.0):
    """The workshop's view of one vehicle.  Components with no data are reported as 'unknown', never as fine."""
    now = _now(conn); m = _meta(conn); k = CAPS.detect(conn); lo30, lo7 = now - 30 * DAY, now - 7 * DAY; checks = []
    def add(i, name, status, detail, evidence, step=None): checks.append({"id": i, "name": name, "status": status, "detail": detail, "evidence": evidence, "step": SERVICE_STEPS.get(step or i, "")})
    codes = active_codes(conn); act = [g for g in codes.values() if g["state"] == "active"]
    add("codes", "Diagnostic trouble codes", "attention" if act else "ok" if codes else "ok", f"{len(act)} active, {len(codes) - len(act)} cleared" if codes else "none recorded", f"{sum(g['set'] for g in codes.values())} set event(s) in the whole record")
    ev30 = _q(conn, "SELECT type, sev, COUNT(*), MAX(ts) FROM events WHERE ts>=? AND ts<=? AND sev IN ('warn','critical') GROUP BY type, sev ORDER BY 3 DESC", (lo30, now))
    crit = sum(n for _, s, n, _ in ev30 if s in SEVERE); warn = sum(n for _, s, n, _ in ev30)
    add("warnings", "Warnings in the last 30 days", "attention" if crit else "watch" if warn >= 5 else "ok", f"{warn} warning(s), {crit} critical" if warn else "none", "events table, severity warn / critical")
    spec = [_num(m.get("veh.tyre_spec_front")), _num(m.get("veh.tyre_spec_rear"))]; last = _one(conn, "SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr FROM ext_1m WHERE tp_fl IS NOT NULL ORDER BY ts DESC LIMIT 1")
    if last and spec[0]:
        names = ("front left", "front right", "rear left", "rear right"); kp = [x / 10.0 if x is not None else None for x in last[1:5]]; lowest = None
        for i, v in enumerate(kp):
            if v is None: continue
            dev = (v - spec[0 if i < 2 else 1]) / spec[0 if i < 2 else 1] * 100.0
            if lowest is None or dev < lowest[0]: lowest = (dev, names[i], v)
        st = "attention" if lowest[0] <= -15 else "watch" if lowest[0] <= -8 else "ok"
        add("tyres", "Tyre pressures", st, f"lowest: {lowest[1]} at {lowest[2]:.0f} kPa ({lowest[0]:+.0f} % against the {spec[0]:.0f} kPa placard)", f"latest reading {stamp(last[0], tz_h)}; {_one(conn, 'SELECT COUNT(*) FROM events WHERE type=? AND ts>=?', ('TPMS_LOW', lo30))[0]} low-pressure warning(s) in 30 days")
    else: add("tyres", "Tyre pressures", "unknown", "this vehicle does not report tyre pressures", "no tyre pressure data")
    v12 = _one(conn, "SELECT MIN(v12), COUNT(v12) FROM ext_1m WHERE ts>=? AND v12 IS NOT NULL", (lo7,)); thr = _num(m.get("thr.low_12v"), 11.8)
    if v12 and v12[0] is not None:
        mn = v12[0] / 100.0; dips = _one(conn, "SELECT COUNT(*) FROM events WHERE type IN ('VOLTAGE_DIP','LOW_12V') AND ts>=?", (lo30,))[0]
        add("v12", "12 V battery", "attention" if mn < thr else "watch" if (mn < thr + 0.5 or dips >= 3) else "ok", f"lowest {mn:.2f} V in the last 7 days (alert below {thr:.1f} V); {dips} dip / low-voltage event(s) in 30 days", "ext_1m.v12, events VOLTAGE_DIP / LOW_12V")
    else: add("v12", "12 V battery", "unknown", "no 12 V readings", "no data")
    pad = _one(conn, "SELECT ts, pad FROM vday WHERE pad IS NOT NULL ORDER BY ts DESC LIMIT 1")
    add("brakes", "Brake pads", _status(pad[1] if pad else None, 30, 15, low_is_bad=True), f"{pad[1]:.0f} % of the pad life left" if pad else "pad wear is not reported", f"daily wear estimate, {stamp(pad[0], tz_h)}" if pad else "no data")
    if k["battery"]:
        soh = _one(conn, "SELECT ts, soh FROM vday WHERE soh IS NOT NULL ORDER BY ts DESC LIMIT 1"); bmax = _one(conn, "SELECT MAX(batt_max) FROM sig_1m WHERE ts>=? AND batt_max IS NOT NULL", (lo30,)); thr_t = _num(m.get("thr.overtemp_c"), 45.0)
        bt = bmax[0] / 10.0 if bmax and bmax[0] is not None else None; ot = _one(conn, "SELECT COUNT(*) FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts>=?", (lo30,))[0]
        s1 = _status(soh[1] if soh else None, 85, 70, low_is_bad=True); s2 = "attention" if ot else _status(bt, thr_t - 5, thr_t)
        add("traction", "High-voltage battery", "attention" if "attention" in (s1, s2) else "watch" if "watch" in (s1, s2) else "ok" if "ok" in (s1, s2) else "unknown",
            (f"state of health {soh[1]:.0f} %" if soh else "state of health not reported") + (f"; hottest {bt:.0f} °C in 30 days" if bt is not None else "") + f"; {ot} over-temperature warning(s)", "vday.soh, sig_1m.batt_max, events WARN_BATT_OVERTEMP")
    if k["engine"] or k["engine_temp"]:
        mt = _one(conn, "SELECT MAX(motor) FROM sig_1m WHERE ts>=? AND motor IS NOT NULL", (lo30,)); t = mt[0] / 10.0 if mt and mt[0] is not None else None; ot = _one(conn, "SELECT COUNT(*) FROM events WHERE type='WARN_OVERTEMP' AND ts>=?", (lo30,))[0]
        add("thermal", "Coolant / motor temperature", "attention" if ot else _status(t, 100, 110), (f"hottest {t:.0f} °C in 30 days" if t is not None else "no temperature") + f"; {ot} over-temperature warning(s)", "sig_1m.motor, events WARN_OVERTEMP")
    rd = CAPS.Reader(conn) if k["engine"] else None
    if k["fuel_type"] == "diesel" and rd is not None:
        for cid, name, step, watch, att, low in (("dpf_soot_load", "Particulate filter soot load", "dpf", 80, 95, False), ("def_level", "AdBlue level", "adblue", 25, 12, True)):
            sid = rd.sid.get(cid)
            if sid is None: continue
            v = _one(conn, "SELECT v FROM lk_1m WHERE sid=? ORDER BY ts DESC LIMIT 1", (sid,)); mx = _one(conn, "SELECT MAX(v) FROM lk_1m WHERE sid=? AND ts>=?", (sid, lo30))
            val = v[0] if v else None; ref = (val if low else (mx[0] if mx and mx[0] is not None else val))
            add(cid, name, _status(ref, watch, att, low_is_bad=low), f"{val:.0f} % now" + ("" if low else f", up to {ref:.0f} % in 30 days") if val is not None else "no reading", f"lk_1m {cid}", step)
    if k["fuel_type"] == "petrol" and rd is not None and rd.sid.get("long_term_fuel_trim") is not None:
        v = _one(conn, "SELECT v FROM lk_1m WHERE sid=? ORDER BY ts DESC LIMIT 1", (rd.sid["long_term_fuel_trim"],)); val = v[0] if v else None
        add("fuel_trim", "Long-term fuel trim", _status(abs(val) if val is not None else None, 8, 12), f"{val:+.1f} %" if val is not None else "no reading", "lk_1m long_term_fuel_trim")
    cam = _one(conn, "SELECT COUNT(*) FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED') AND ts>=?", (lo30,))
    if k["camera"]: add("camera", "Forward camera", "watch" if cam and cam[0] >= 3 else "ok", f"{cam[0] if cam else 0} blocked / degraded event(s) in 30 days", "events CAMERA_BLOCKED / CAMERA_DEGRADED (the pictures themselves are not part of this report)")
    odo = _num(m.get("veh.odo_end")); last_km = _num(m.get("veh.last_service_km")); interval = _num(m.get("veh.service_km"))
    if odo and last_km is not None and interval:
        since = odo - last_km; left = interval - since
        add("service", "Scheduled service", "attention" if left < 0 else "watch" if left < 1000 else "ok", f"{since:,.0f} km since the last service of {interval:,.0f} km: " + (f"overdue by {-left:,.0f} km" if left < 0 else f"{left:,.0f} km left"), "veh.odo_end, veh.last_service_km, veh.service_km")
    else: add("service", "Scheduled service", "unknown", "no service interval or mileage is recorded", "no data")
    sw = _one(conn, "SELECT ts, detail FROM events WHERE type='SW_UPDATE' ORDER BY ts DESC LIMIT 1")
    steps = [{"for": c["name"], "status": c["status"], "step": c["step"]} for c in checks if c["status"] in ("attention", "watch") and c["step"]]
    order = {"attention": 0, "watch": 1}; steps.sort(key=lambda s: order[s["status"]])
    return {"vehicle": {"model": m.get("veh.model", "unknown vehicle"), "type": k["label"], "basis": k["summary"], "software": m.get("veh.sw"), "odometer_km": odo, "history_from": stamp(_num(m.get("t0"), now), tz_h), "history_to": stamp(now, tz_h)},
            "generated_at": stamp(now, tz_h), "codes": sorted(codes.values(), key=lambda g: (g["state"] != "active", -g["last"])), "warnings": [{"type": t, "severity": s, "count": n, "last": stamp(ts, tz_h)} for t, s, n, ts in ev30],
            "checks": checks, "steps": steps, "software_update": {"when": stamp(sw[0], tz_h), "detail": sw[1]} if sw else None,
            "summary": {"attention": sum(c["status"] == "attention" for c in checks), "watch": sum(c["status"] == "watch" for c in checks), "ok": sum(c["status"] == "ok" for c in checks), "unknown": sum(c["status"] == "unknown" for c in checks)}}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------- fleet
def vehicle_kpis(path):
    """The fleet-level numbers of one vehicle history (read-only, last 30 days of its own record).  No position, no place, no driver."""
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        m = _meta(c); k = CAPS.detect(c); end = _one(c, "SELECT MAX(end) FROM trips"); end = int(end[0]) if end and end[0] else _now(c); lo = end - 30 * DAY
        tr = _one(c, "SELECT COUNT(*), COALESCE(SUM(km),0), COALESCE(SUM(end-start),0), COALESCE(SUM(kwh),0), COALESCE(SUM(CASE WHEN kwh IS NOT NULL THEN km END),0) FROM trips WHERE start>=? AND end<=?", (lo, end + 1)) or (0, 0, 0, 0, 0)
        n, km, secs, kwh, km_kwh = tr; eff = unit = None
        mv = _one(c, "SELECT COUNT(*), COALESCE(SUM(speed), 0) / 600.0 FROM sig_1m WHERE ts>=? AND ts<=? AND speed>10", (lo, end))              # minutes the car moved, and the distance they add up to
        if mv and mv[0] and (km < 0.5 or (n and secs / n > 12 * 3600)):                                                                    # a recording whose 'trip' is the whole file (or has no distance): use the minute rows
            km = mv[1] if km < 0.5 else km; secs = mv[0] * 60.0
        if k["battery"] and kwh and km_kwh: eff, unit = kwh / km_kwh * 1000.0, "Wh/km"
        elif k["fuel"]:
            sid = dict(_q(c, "SELECT concept, sid FROM sig_catalog")).get("fuel_rate")
            if sid is not None:
                lit = _one(c, "SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=? AND ts>=? AND ts<?", (sid, lo, end + 1)); dist = _one(c, "SELECT SUM(speed) / 600.0 FROM sig_1m WHERE ts>=? AND ts<? AND speed IS NOT NULL", (lo, end + 1))
                if lit and lit[0] and dist and dist[0] and dist[0] > 5: eff, unit = lit[0] / dist[0] * 100.0, "L/100 km"
        harsh = _one(c, "SELECT COUNT(*) FROM events WHERE type IN ('HARSH_BRAKE','HARSH_ACCEL','HARSH_CORNER') AND ts>=? AND ts<=?", (lo, end))[0]
        has_harsh = bool(_one(c, "SELECT 1 FROM events WHERE type IN ('HARSH_BRAKE','HARSH_ACCEL','HARSH_CORNER') LIMIT 1"))
        alerts = _q(c, "SELECT sev, COUNT(*) FROM events WHERE sev IN ('warn','critical') AND ts>=? AND ts<=? GROUP BY sev", (lo, end)); a = dict(alerts)
        codes = [g for g in active_codes(c).values() if g["state"] == "active"]
        charges = _one(c, "SELECT COUNT(*) FROM charges WHERE start>=? AND start<=?", (lo, end))[0] + _one(c, "SELECT COUNT(*) FROM events WHERE type='REFUEL' AND ts>=? AND ts<=?", (lo, end))[0]
        odo = _num(m.get("veh.odo_end")); lastkm = _num(m.get("veh.last_service_km")); interval = _num(m.get("veh.service_km")); overdue = (odo - lastkm - interval) if (odo and lastkm is not None and interval and odo - lastkm > interval) else 0
        vid = "V-" + hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:4].upper()
        return {"id": vid, "model": m.get("veh.model", "vehicle"), "type": k["label"], "history_to": stamp(end), "trips": int(n), "km": round(km, 1), "driving_h": round(secs / 3600.0, 1), "utilisation_pct": round(secs / (30 * DAY) * 100.0, 1),
                "avg_trip_km": round(km / n, 1) if n else None, "efficiency": round(eff, 1) if eff else None, "efficiency_unit": unit, "harsh_per_100km": round(harsh / km * 100.0, 1) if has_harsh and km > 5 else None,
                "warnings": int(a.get("warn", 0) + a.get("critical", 0)), "critical": int(a.get("critical", 0)), "active_codes": len(codes), "energy_stops": int(charges), "service_overdue_km": round(overdue) if overdue else 0}
    finally: c.close()


def fleet_overview(paths):
    """`paths`: the history files that make up the fleet.  Returns the table, the fleet medians, and for every vehicle why it needs attention (if it does)."""
    rows = []
    for p in paths:
        try:
            r = vehicle_kpis(p)
            if r["trips"] >= 1 or r["km"]: rows.append(r)
        except (sqlite3.Error, OSError, ValueError, TypeError): continue
    med = {}
    for t in {r["type"] for r in rows}:
        e = [r["efficiency"] for r in rows if r["type"] == t and r["efficiency"]]
        if len(e) >= 2: med[t] = round(statistics.median(e), 1)
    hm = [r["harsh_per_100km"] for r in rows if r["harsh_per_100km"] is not None]; hmed = statistics.median(hm) if len(hm) >= 2 else None
    for r in rows:
        why = []
        if r["critical"]: why.append(f"{r['critical']} critical event(s) in 30 days")
        if r["active_codes"]: why.append(f"{r['active_codes']} active trouble code(s)")
        if r["service_overdue_km"]: why.append(f"service overdue by {r['service_overdue_km']:,} km")
        m = med.get(r["type"])
        if m and r["efficiency"] and r["efficiency"] > m * 1.15: why.append(f"uses {((r['efficiency'] / m) - 1) * 100:.0f} % more than the fleet median for its type")
        if hmed and r["harsh_per_100km"] and r["harsh_per_100km"] > max(hmed * 1.5, 1.0): why.append(f"{r['harsh_per_100km']} harsh events per 100 km (fleet median {hmed:.1f})")
        r["attention"] = why; r["status"] = "attention" if (r["critical"] or r["active_codes"] or r["service_overdue_km"]) else "watch" if why else "ok"
    rows.sort(key=lambda r: ({"attention": 0, "watch": 1, "ok": 2}[r["status"]], -r["km"]))
    return {"vehicles": rows, "medians": med, "totals": {"vehicles": len(rows), "km": round(sum(r["km"] for r in rows), 1), "trips": sum(r["trips"] for r in rows), "warnings": sum(r["warnings"] for r in rows), "needing_attention": sum(r["status"] != "ok" for r in rows)},
            "window": "the last 30 days of each vehicle's own record", "privacy": "No driver identity, places, routes or camera pictures are part of this view."}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------- validation
FAMILIES = {"ecu_resets_adas": "ECU_RESET", "voltage_dip": "VOLTAGE_DIP", "unexpected_shutdown": "UNEXPECTED_SHUTDOWN", "can_burst": "CAN_ERROR", "speed_conflict": "SENSOR_CONFLICT", "ambient_glitch": "SENSOR_CONFLICT", "aeb": "AEB_ACTIVATION", "dtc_cleared": "DTC_CLEAR"}
CORE = (("speed", "speed", "km/h", 10.0), ("soc", "battery state of charge", "%", 10.0), ("power", "battery power", "kW", 1.0), ("batt", "battery temperature", "°C", 10.0), ("flow", "coolant flow", "L/min", 10.0),
        ("amb", "outside temperature", "°C", 10.0), ("hvac", "climate power", "W", 1.0), ("motor", "motor / engine temperature", "°C", 10.0))


def scorecard(conn, truth, tol=30):
    """Injected faults (the world's ground truth) against the events the vehicle recorded: for each family, how many of the injected ones have an event of the matching type within `tol` seconds."""
    rows = []; faults = (truth or {}).get("faults", {}); by_type = {}; now = _now(conn)
    for fam, etype in FAMILIES.items():
        items = faults.get(fam) or []; ts = [int(x.get("ts") or x.get("cleared_ts") or 0) for x in items if isinstance(x, dict)]; ts = [t for t in ts if t and t <= now]                  # only what had happened by the vehicle's own moment
        if not ts: continue
        ev = by_type.get(etype)
        if ev is None: ev = by_type[etype] = [r[0] for r in _q(conn, "SELECT ts FROM events WHERE type=? ORDER BY ts", (etype,))]
        found = sum(1 for t in ts if any(abs(e - t) <= tol for e in ev)); rows.append({"family": fam.replace("_", " "), "event": etype, "injected": len(ts), "found": found, "recall": round(found / len(ts), 3), "recorded_events": len(ev)})
    return rows


def validation_report(conn, name=None, root=None, latency=None):
    now = _now(conn); m = _meta(conn); sig = []
    for col, label, unit, scale in CORE:
        r = _one(conn, f"SELECT COUNT({col}), MIN({col}), MAX({col}), MIN(CASE WHEN {col} IS NOT NULL THEN ts END), MAX(CASE WHEN {col} IS NOT NULL THEN ts END), COUNT(DISTINCT {col}) FROM sig_1m")
        if r and r[0]: sig.append({"signal": label, "unit": unit, "source": "core table", "rate": "1 / min", "samples": r[0], "min": round(r[1] / scale, 2), "max": round(r[2] / scale, 2), "from": stamp(r[3]), "to": stamp(r[4]), "flags": ["constant value"] if r[5] == 1 and r[0] > 100 else []})
    for sid, concept, label, unit, lo, hi, source, rate in _q(conn, "SELECT sid, concept, label, unit, lo, hi, source, rate_hz FROM sig_catalog ORDER BY concept"):
        r = _one(conn, "SELECT COUNT(*), MIN(v), MAX(v), MIN(ts), MAX(ts), COUNT(DISTINCT v) FROM lk_1m WHERE sid=?", (sid,))
        if not r or not r[0]: sig.append({"signal": label, "unit": unit, "source": source, "rate": f"{rate:g} Hz" if rate else "", "samples": 0, "min": None, "max": None, "from": "", "to": "", "flags": ["declared but no data"]}); continue
        flags = []
        if lo is not None and r[1] is not None and r[1] < lo - 1e-6: flags.append("below the physical range")
        if hi is not None and r[2] is not None and r[2] > hi + 1e-6: flags.append("above the physical range")
        if r[5] == 1 and r[0] > 100: flags.append("constant value")
        sig.append({"signal": label, "unit": unit, "source": source, "rate": f"{rate:g} Hz" if rate else "", "samples": r[0], "min": round(r[1], 2), "max": round(r[2], 2), "from": stamp(r[3]), "to": stamp(r[4]), "flags": flags})
    buses = {}
    for s in sig:
        b = (s["source"] or "core").split(":")[0]; e = buses.setdefault(b, {"bus": b, "signals": 0, "samples": 0}); e["signals"] += 1; e["samples"] += s["samples"]
    silent = {r[0]: r[1] for r in _q(conn, "SELECT source, COUNT(*) FROM source_log WHERE status='silent' GROUP BY source")}
    events = [{"type": t, "severity": s, "count": n} for t, s, n in _q(conn, "SELECT type, sev, COUNT(*) FROM events GROUP BY type, sev ORDER BY 3 DESC")]
    truth = None; card = []
    if name and root:
        p = os.path.join(root, "data", os.path.splitext(name)[0] + "_truth.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f: truth = json.load(f)
                card = scorecard(conn, truth)
            except (OSError, ValueError): card = []
    tot = sum(r["injected"] for r in card); fnd = sum(r["found"] for r in card)
    ass = {"version": None, "latency_ms": latency or None}
    if root:
        sp = os.path.join(root, "results", "meaning", "v11_summary.json")
        if os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8") as f: j = json.load(f)
                ass["whole_assistant_accuracy"] = j.get("e2e"); ass["note"] = "same answer as a perfect reader of the question, 2,176 frozen questions on the 105-day world (docs/V11_RESULTS.md)"
            except (OSError, ValueError): pass
        try:
            with open(os.path.join(root, "models", "product.json"), encoding="utf-8") as f: ass["product_adapter"] = json.load(f).get("adapter")
        except (OSError, ValueError): ass["product_adapter"] = None                                                                      # no product.json: the launcher's default adapter is in use
        if ass.get("whole_assistant_accuracy"):
            ass["note"] += ".  These figures belong to the earlier adapters they are named for (v6, v8, v11_final).  The adapter in use now is `product_adapter`; its own figures (exact executed frame on the same frozen questions, not whole-assistant answers) are in docs/SCORECARD.md, section 4"
    return {"vehicle": {"model": m.get("veh.model", ""), "history_to": stamp(now), "file": name, "synthetic": bool(truth)}, "signals": sig, "buses": sorted(buses.values(), key=lambda b: -b["samples"]), "silent_sources": silent, "events": events,
            "scorecard": {"rows": card, "injected": tot, "found": fnd, "recall": round(fnd / tot, 3) if tot else None, "tolerance_s": 30, "note": "Ground truth exists only for the generated worlds; a recorded file has none."} if card else None,
            "assistant": ass, "flags": sum(1 for s in sig if s["flags"])}


def evidence_pack(question, response, conn, role, meta_extra=None):
    """What a validation or service person can keep: the question, the answer, how it was read, and the data window - ready to be signed."""
    m = _meta(conn); r = response or {}
    f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
    return {"kind": "vehiclegpt.evidence/1", "created": int(time.time()), "role": role, "vehicle": {"model": m.get("veh.model"), "software": m.get("veh.sw"), "history_to": stamp(_now(conn))},
            "question": str(question), "question_sha256": hashlib.sha256(str(question).encode("utf-8")).hexdigest(), "answer": r.get("answer"), "path": r.get("path"), "route": r.get("route") or f.get("route"), "frame": f.get("frame") or ((r.get("slm") or {}).get("raw") if isinstance(r.get("slm"), dict) else None),
            "understood": r.get("understood"), "confidence": r.get("conf"), "latency_ms": (r.get("ms") or {}).get("total") if isinstance(r.get("ms"), dict) else None, "evidence": f.get("evidence") or r.get("evidence"), "meta": meta_extra or {}}
