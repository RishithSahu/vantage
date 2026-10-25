"""The ledger: trips and charging sessions derived from the signal tiers.

The stream layer (sig_1s, the lake, events) records WHAT the vehicle did second by second; the questions people ask are about journeys ("my last trip", "the three least efficient trips", "how many fast
charges this month").  `update()` turns 1 Hz rows into trip and charge records - the same tables (`trips`, `trip_ext`, `charges`) the older answer routes and the typed engine's entity operations read -
so a vehicle that only ever ran through live collectors has them too.  It is idempotent (a trip that is already recorded is not written again) and incremental (only rows after the last recorded trip are
read), so it runs every few minutes in the live runtime and once at the end of a bulk import.  A bulk-generated history gets its ledger from the generator; tests/test_world.py checks that this derivation
reproduces the generator's trips on the 1 Hz window within a small tolerance.

Definitions (shared by both paths): a TRIP is a run of rows in drive mode (city / highway) with no gap longer than 120 s; kilometres integrate speed, kWh integrate battery power (net of regeneration),
highway share is the share of trip seconds in highway mode.  A CHARGE is a run of charging-mode rows with no gap longer than 300 s; DC when the peak power exceeded 25 kW."""
import sqlite3

TRIP_GAP_S, CHARGE_GAP_S, MIN_TRIP_KM, MIN_TRIP_S = 120, 300, 0.2, 60


def derive(rows):
    """rows: iterable of (ts, speed_kmh, power_kw, amb_c, hvac_w, soc_pct, mode) ordered by ts -> (trips, charges)."""
    trips, charges = [], []; cur = chg = None; prev_ts = None
    for ts, spd, pw, amb, hvac, soc, mode in rows:
        dt = 1.0 if prev_ts is None else float(min(max(ts - prev_ts, 0), 2)); prev_ts_gap = None if prev_ts is None else ts - prev_ts; prev_ts = ts
        driving = mode in (1, 2); charging = mode == 3
        if cur is not None and (not driving or (prev_ts_gap is not None and prev_ts_gap > TRIP_GAP_S)):
            trips.append(_close_trip(cur)); cur = None
        if chg is not None and (not charging or (prev_ts_gap is not None and prev_ts_gap > CHARGE_GAP_S)):
            charges.append(_close_charge(chg)); chg = None
        if driving:
            if cur is None: cur = dict(start=ts, last=ts, km=0.0, kwh=0.0, secs=0.0, hwy=0.0, amb=0.0, hvac=0.0, speed_s=0.0, soc0=soc, soc1=soc)
            cur["last"] = ts; cur["km"] += (spd or 0.0) * dt / 3600.0; cur["kwh"] += (pw or 0.0) * dt / 3600.0; cur["secs"] += dt; cur["hwy"] += dt if mode == 2 else 0.0
            cur["amb"] += (amb or 0.0) * dt; cur["hvac"] += (hvac or 0.0) * dt; cur["speed_s"] += (spd or 0.0) * dt; cur["soc1"] = soc if soc is not None else cur["soc1"]
        elif charging:
            if chg is None: chg = dict(start=ts, last=ts, kwh=0.0, peak=0.0, soc0=soc, soc1=soc)
            chg["last"] = ts; kw = max(0.0, -(pw or 0.0)); chg["kwh"] += kw * dt / 3600.0; chg["peak"] = max(chg["peak"], kw); chg["soc1"] = soc if soc is not None else chg["soc1"]
    if cur is not None: trips.append(_close_trip(cur))
    if chg is not None: charges.append(_close_charge(chg))
    return [t for t in trips if t and t["km"] >= MIN_TRIP_KM and t["end"] - t["start"] >= MIN_TRIP_S], [c for c in charges if c["kwh"] > 0.05]


def _close_trip(t):
    secs = max(t["secs"], 1.0); km = t["km"]
    return dict(start=t["start"], end=t["last"] + 1, km=round(km, 2), kwh=round(t["kwh"], 3), wh_km=round(1000 * t["kwh"] / max(km, 0.1), 1), avg_speed=round(t["speed_s"] / secs, 1), hwy_share=round(t["hwy"] / secs, 3),
                avg_amb=round(t["amb"] / secs, 1), avg_hvac_w=round(t["hvac"] / secs, 0), soc_start=t["soc0"], soc_end=t["soc1"])


def _close_charge(c):
    return dict(start=c["start"], end=c["last"] + 1, kind="DC" if c["peak"] > 25 else "AC", kw=round(c["peak"], 1), kwh=round(c["kwh"], 2), soc0=c["soc0"], soc1=c["soc1"])


def rows_from_sig_1s(conn, since):
    q = "SELECT ts, speed, power, amb, hvac, soc, mode FROM sig_1s WHERE ts >= ? ORDER BY ts"
    for ts, sp, pw, am, hv, so, md in conn.execute(q, (int(since),)):
        yield (ts, None if sp is None else sp / 10.0, None if pw is None else pw / 1000.0, None if am is None else am / 10.0, hv, None if so is None else so / 10.0, md)


def update(conn, lookback_s=3600, now=None):
    """Write trips and charges derived from the 1 Hz rows after the last recorded one (re-reading `lookback_s` before it so an open trip is completed).  Returns (new_trips, new_charges)."""
    last_t = conn.execute("SELECT MAX(start) FROM trips").fetchone()[0]; last_c = conn.execute("SELECT MAX(start) FROM charges").fetchone()[0]
    since = max(0, min([x for x in (last_t, last_c) if x is not None] or [0]) - lookback_s) if (last_t or last_c) else 0
    trips, charges = derive(rows_from_sig_1s(conn, since)); nt = nc = 0
    if now is not None:                                                                                # a trip still in progress is not final
        trips = [t for t in trips if t["end"] < now - TRIP_GAP_S]; charges = [c for c in charges if c["end"] < now - CHARGE_GAP_S]
    conn.execute("BEGIN")
    for t in trips:
        if last_t is not None and t["start"] <= last_t: continue
        cur = conn.execute("INSERT INTO trips(start,end,km,kwh,wh_km,avg_speed,hwy_share,avg_amb,avg_hvac_w,soc_start,soc_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (t["start"], t["end"], t["km"], t["kwh"], t["wh_km"], t["avg_speed"], t["hwy_share"], t["avg_amb"], t["avg_hvac_w"], t["soc_start"], t["soc_end"])); nt += 1
        ev = dict(conn.execute("SELECT type, COUNT(*) FROM events WHERE ts>=? AND ts<? AND type IN ('HARSH_BRAKE','HARSH_ACCEL','HARSH_CORNER') GROUP BY type", (t["start"], t["end"])).fetchall())
        regen = conn.execute("SELECT COALESCE(SUM(CASE WHEN power<0 THEN -power ELSE 0 END),0)/3.6e6 FROM sig_1s WHERE ts>=? AND ts<? AND mode IN (1,2)", (t["start"], t["end"])).fetchone()[0]
        x = conn.execute("SELECT AVG(rain), AVG(adas) FROM ext_1m WHERE ts>=? AND ts<?", (t["start"] - 59, t["end"])).fetchone() if _has(conn, "ext_1m") else (None, None)
        g0 = conn.execute("SELECT lat, lon FROM gps_1s WHERE ts>=? ORDER BY ts LIMIT 1", (t["start"],)).fetchone() if _has(conn, "gps_1s") else None
        g1 = conn.execute("SELECT lat, lon FROM gps_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (t["end"],)).fetchone() if _has(conn, "gps_1s") else None
        conn.execute("INSERT OR REPLACE INTO trip_ext(start, from_id, to_id, harsh_brake, harsh_accel, harsh_corner, regen_kwh, rain_share, adas_share, lat0, lon0, lat1, lon1) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (t["start"], None, None, ev.get("HARSH_BRAKE", 0), ev.get("HARSH_ACCEL", 0), ev.get("HARSH_CORNER", 0), round(float(regen or 0), 3), None if x[0] is None else round(min(1.0, x[0] / 3.0), 3),
                      None if x[1] is None else round(x[1], 1), None if not g0 else g0[0] / 1e6, None if not g0 else g0[1] / 1e6, None if not g1 else g1[0] / 1e6, None if not g1 else g1[1] / 1e6))
    for c in charges:
        if last_c is not None and c["start"] <= last_c: continue
        g = conn.execute("SELECT lat, lon FROM gps_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (c["start"],)).fetchone() if _has(conn, "gps_1s") else None
        conn.execute("INSERT OR REPLACE INTO charges VALUES(?,?,?,?,?,?,?,?,?,?,?)", (c["start"], c["end"], c["kind"], c["kw"], c["kwh"], c["soc0"], c["soc1"], None, None if not g else g[0] / 1e6, None if not g else g[1] / 1e6, round(c["kwh"] / 0.93, 2))); nc += 1
    conn.execute("COMMIT"); return nt, nc


def _has(conn, table): return conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is not None
