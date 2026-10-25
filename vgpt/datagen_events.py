"""DEV tool - second half of the generator: derives events, ADAS use, camera health, tyre / 12 V / service data, places and the ground-truth file from the simulated arrays,
then seeds the on-device database (same schema the assistant reads) plus the extension tables."""
import datetime as dt, json, math, os
import numpy as np
from .datagen import THR_HARSH, THR_CORNER, bearing, haversine_km, offset, CITIES, movavg
from .protocols import nmea

WHEELS = ["front left", "front right", "rear left", "rear right"]
SECOND_ORDER = ("TPMS_LOW", "TYRE_INFLATED")


def _gnss_track(rng, t0, s, v, lat_t, lon_t, cut):
    """Per-second GNSS fixes for one trip: ~3 m correlated noise, course from the path, speed from the trip, occasional 5-30 s dropouts (urban canyon / tunnel) with a worse HDOP around them."""
    L = len(v); la = np.asarray(lat_t, np.float64).copy(); lo = np.asarray(lon_t, np.float64).copy(); coslat = max(0.2, float(np.cos(np.radians(la.mean()))))
    e = np.zeros((2, L)); w = rng.normal(0, 0.9, (2, L))
    for i in range(1, L): e[:, i] = 0.92 * e[:, i - 1] + w[:, i]
    la += e[0] / 111200.0; lo += e[1] / (111200.0 * coslat)
    dla = np.gradient(movavg(la, 5)); dlo = np.gradient(movavg(lo, 5)) * coslat; hd = (np.degrees(np.arctan2(dlo, dla)) + 360) % 360
    still = np.asarray(v) < 2.0; hdg = hd.copy(); last = hd[~still][0] if (~still).any() else 0.0
    for i in range(L):
        if still[i]: hdg[i] = last
        else: last = hdg[i]
    keep = np.ones(L, bool); hdop = 0.8 + np.abs(rng.normal(0, 0.25, L))
    for _ in range(int(rng.poisson(L / 1800.0))):
        a = int(rng.integers(0, max(1, L - 40))); d = int(rng.integers(5, 30)); keep[a:a + d] = False; hdop[max(0, a - 8):a + d + 8] += 2.5
    k0 = max(0, cut - s)                                                      # only the 1 Hz retention window (last 7 days)
    return [(t0 + s + i, int(round(la[i] * 1e6)), int(round(lo[i] * 1e6)), int(round(hdg[i] * 10)), int(round(float(v[i]) * 10)), int(round(hdop[i] * 10))) for i in range(k0, L) if keep[i]]


def gnss_stream(gv, t_from, seconds):
    """Yield (ts, "gnss", NMEA sentence) like a receiver would: RMC + GGA per fix (what the live ingest path consumes)."""
    for ts, la, lo, hd, sp, hp in gv.ex["gps_1s"]:
        if t_from <= ts < t_from + seconds:
            yield float(ts), "gnss", nmea.encode_rmc(ts, la / 1e6, lo / 1e6, sp / 10.0, hd / 10.0)
            yield float(ts) + 0.01, "gnss", nmea.encode_gga(ts, la / 1e6, lo / 1e6, 1, 9, hp / 10.0, 900.0)


def _edges(mask):
    d = np.diff(np.r_[False, mask].astype(np.int8)); return np.where(d == 1)[0]


def _runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(np.int8)); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _angdiff(a, b): return np.abs((a - b + 180) % 360 - 180)


def finish(gv, rng, verbose=True):
    life, world, V, C, DR, st, plan = gv.life, gv.world, gv.V, gv.C, gv.DR, gv.st, gv.plan
    t0, n = gv.t0, gv.n; m = n // 60; days = n // 86400; ev = []; ex = {}                       # ex: extension tables
    act = (gv.mode != 0)                                                                         # driving or charging
    first_active_after = lambda s: int(s + np.argmax(act[s:])) if act[s:].any() else None

    # ---------------------------------------------------------------- charging sessions
    charges = []
    for s in life.sessions:
        ev.append((t0 + s["start"], "CHARGE_START", "info", f"{s['kind']} {s['kw']:.0f} kW"))
        charges.append(dict(start=t0 + s["start"], end=t0 + s["end"], kind=s["kind"], kw=round(s["kw"], 1), kwh=round(s["kwh"], 2), soc0=round(s["soc0"], 1), soc1=round(s["soc1"], 1), place=s["place"]))
    ex["charges"] = charges

    # ---------------------------------------------------------------- per-minute context used by ADAS / camera synthesis
    mi = lambda s: min(s // 60, m - 1)
    rain_m = world.rain; elev, azim, cloud = world.elev, world.azim, world.cloud
    adas_on = np.zeros(n, bool); head_m = np.full(m, np.nan)
    road_days = set(st.get("roadworks", {}).get("days", []))
    # ---------------------------------------------------------------- per trip: harsh driving, ADAS, positions
    pos_lat = np.full(m, np.nan); pos_lon = np.full(m, np.nan); gps_rows = []; gps_cut = n - 7 * 86400; tracks = []      # GNSS: 1 Hz fixes while driving, 1/min parked, last 7 days
    for tr in gv.trips:
        s, v, lo, la = tr["s"], tr["v"], tr["lon"], tr["lat"]; L = len(v)
        hb = [i for i in _edges(lo < -THR_HARSH) if v[i] > 15]; ha = [i for i in _edges(lo > THR_HARSH) if v[i] > 15]; hc = [i for i in _edges(np.abs(la) > THR_CORNER) if v[i] > 20]
        for i in hb: ev.append((t0 + s + int(i), "HARSH_BRAKE", "info", f"{-lo[i]:.2f} g at {v[i]:.0f} km/h"))
        for i in ha: ev.append((t0 + s + int(i), "HARSH_ACCEL", "info", f"{lo[i]:.2f} g at {v[i]:.0f} km/h"))
        for i in hc: ev.append((t0 + s + int(i), "HARSH_CORNER", "info", f"{abs(la[i]):.2f} g at {v[i]:.0f} km/h"))
        tr["harsh_brake"], tr["harsh_accel"], tr["harsh_corner"] = len(hb), len(ha), len(hc)
        # ---- ADAS lane keeping on motorway stretches
        eng = 0; used = rng.random() < DR["adas"]; d = s // 86400
        if used:
            for (i0, i1) in _runs(v > 65):
                if i1 - i0 < 150: continue
                a0, a1 = i0 + int(rng.integers(20, 60)), i1 - int(rng.integers(10, 40)); adas_on[s + a0:s + a1] = True
                hrs = (a1 - a0) / 3600.0; tm = world.rain[np.minimum((np.arange(s + a0, s + a1)) // 60, m - 1)]
                rainy = float((tm >= 2).mean()) > 0.3; works = d in road_days and local_hour(gv, s) < 12.5
                rate = 5.0 if works else (3.0 if rainy else 0.35); pool = ([("lane marking lost", .9), ("driver override", .1)] if works else [("camera blocked", .5), ("lane marking lost", .4), ("driver override", .1)] if rainy
                                                                              else [("driver override", .65), ("lane marking lost", .3), ("camera blocked", .05)])
                for _ in range(int(rng.poisson(rate * hrs))):
                    k = int(a0 + rng.integers(0, max(1, a1 - a0))); r_ = pool[int(rng.choice(len(pool), p=np.array([p for _, p in pool]) / sum(p for _, p in pool)))][0]
                    ev.append((t0 + s + k, "ADAS_DISENGAGE", "info", r_)); adas_on[s + k:s + k + int(rng.integers(20, 120))] = False
        tr["adas_s"] = int(adas_on[s:s + L].sum())
        # ---- route position (one point per minute) and heading
        A_, B_ = plan.places[tr["oa"]], plan.places[tr["ob"]]; A = (A_["lat"], A_["lon"]); B = (B_["lat"], B_["lon"])
        cum = (tr["j_off"] + np.cumsum(v) / 3600.0) / max(tr["j_km"], 0.1); frac = np.clip(cum, 0, 1); km_route = max(0.5, haversine_km(A, B)); ph = float(rng.uniform(0, 6.28))
        wig = 0.07 * km_route / 111.2 * np.sin(2 * np.pi * frac * 1.6 + ph) * np.sin(np.pi * frac)
        brg = math.radians(bearing(A, B)); perp = brg + math.pi / 2
        lat_t = A[0] + (B[0] - A[0]) * frac + wig * math.cos(perp); lon_t = A[1] + (B[1] - A[1]) * frac + wig * math.sin(perp) / max(0.2, math.cos(math.radians(A[0])))
        minutes = np.arange(s // 60, min(m, (s + L) // 60 + 1)); sec_idx = np.clip(minutes * 60 - s, 0, L - 1)
        if s + L > gps_cut: gps_rows += _gnss_track(rng, t0, s, v, lat_t, lon_t, gps_cut)
        tracks.append((s, s + L, float(lat_t[-1]), float(lon_t[-1])))
        pos_lat[minutes] = lat_t[sec_idx]; pos_lon[minutes] = lon_t[sec_idx]; pos_lat[minutes[-1]] = lat_t[-1]; pos_lon[minutes[-1]] = lon_t[-1]; head_m[minutes] = bearing(A, B); tr["pos_end"] = (float(lat_t[-1]), float(lon_t[-1])); tr["pos_start"] = (float(lat_t[0]), float(lon_t[0]))
    # charging minutes: position = place or end of the previous trip leg
    ends = sorted(((tr["end"] - t0), tr) for tr in gv.trips)
    for c in charges:
        s = c["start"] - t0; e_ = c["end"] - t0
        if c["place"] >= 0: p = plan.places[c["place"]]; la_, lo_ = p["lat"], p["lon"]
        else:
            prev = [tr for e2, tr in ends if e2 <= s + 200]; la_, lo_ = prev[-1]["pos_end"] if prev else (C["lat"], C["lon"])
            la_, lo_ = la_ + rng.normal(0, 0.0004), lo_ + rng.normal(0, 0.0004)
        c["lat"], c["lon"] = la_, lo_; ms = np.arange(s // 60, min(m, e_ // 60 + 1)); pos_lat[ms] = la_; pos_lon[ms] = lo_
    # ---------------------------------------------------------------- thermal faults from the arrays
    healthy_low = (gv.flow > 0.3) & (gv.flow < 6.0) & (np.arange(n) >= life.pump_t)
    if healthy_low.any():
        i = int(np.argmax(healthy_low)); ev.append((t0 + i, "DTC_SET", "warn", "P1A00 Battery coolant flow low (illustrative code)"))
    over = gv.batt > 40.0; last = -10 ** 9
    for i in _edges(over):
        if i - last > 7200: ev.append((t0 + int(i), "WARN_BATT_OVERTEMP", "critical", "Battery temperature above 40 C")); last = i
    # ---------------------------------------------------------------- tyres (TPMS), 12 V, service, software
    for (sec, ty, sev, w) in life.tyre_events:
        j = first_active_after(min(int(sec), n - 2))
        if j is None: continue
        cold = life.cold_at(j)[w]
        if ty == "TPMS_LOW":
            ev.append((t0 + j, ty, sev, f"{WHEELS[w]} tyre pressure low ({cold:.0f} kPa cold, spec {life.spec[w]:.0f} kPa)")); ev.append((t0 + j, "DTC_SET", "warn", f"C0750 Tyre pressure low - {WHEELS[w]} (illustrative code)"))
        else: ev.append((t0 + j, ty, sev, f"{WHEELS[w]} tyre pressure restored to {life.cold_at(j)[w]:.0f} kPa"))
    v12_rest = np.full(days + 1, 12.65) - np.arange(days + 1) * 0.0008
    if "weak_12v" in st:
        d1 = st["weak_12v"]["start"]; v12_rest[d1:] = np.maximum(11.7, 12.62 - 0.07 * (np.arange(days + 1 - d1)))
        low = np.where(v12_rest < 12.1)[0]
        if len(low):
            d = int(low[0]); j = first_active_after(d * 86400 + 3600)
            if j is not None: ev.append((t0 + j, "LOW_12V", "warn", f"12 V battery low: {v12_rest[d]:.1f} V at rest")); ev.append((t0 + j, "DTC_SET", "warn", "P0562 System voltage low (illustrative code)"))
    gv.gps_tmp = (gps_rows, tracks, gps_cut)
    return _finish2(gv, rng, ev, ex, act, adas_on, head_m, pos_lat, pos_lon, v12_rest, verbose)


def local_hour(gv, s): return ((gv.t0 + s) % 86400 / 3600.0 + gv.tzh) % 24


def _finish2(gv, rng, ev, ex, act, adas_on, head_m, pos_lat, pos_lon, v12_rest, verbose):
    life, world, V, C, DR, st, plan = gv.life, gv.world, gv.V, gv.C, gv.DR, gv.st, gv.plan; t0, n = gv.t0, gv.n; m = n // 60; days = n // 86400
    # service interval & last service (kept in meta); SERVICE_DUE event when the interval is crossed inside the window
    svc_int = int(rng.choice([10000, 15000, 20000])); last_km = life.odo0 - float(rng.uniform(1500, svc_int * 0.92)); last_ts = t0 - int(86400 * rng.uniform(40, 330))
    due_km = last_km + svc_int
    if life.odo0 < due_km <= life.odo_km:
        j = int(np.searchsorted(gv.odo, due_km)); j = min(j, n - 1); ev.append((t0 + j, "SERVICE_DUE", "info", f"scheduled service due at {due_km:.0f} km"))
    if rng.random() < 0.4:
        j = int(rng.integers(86400, max(86401, n - 86400))); ev.append((t0 + j, "SW_UPDATE", "info", "infotainment software 24.3.1 -> 24.4.0 installed"))
    # ---------------------------------------------------------------- camera health (per driving minute) + camera events
    ign_m = gv.ign[: m * 60].reshape(m, 60); drv = ign_m.any(1); spd_m = gv.speed[: m * 60].reshape(m, 60).mean(1); hwy_m = spd_m > 72
    rain = world.rain.astype(np.float64); lux = world.lux; elev, azim = world.elev, world.azim; code = world.code[: m] if len(world.code) >= m else np.zeros(m, int)
    lane = np.where(hwy_m, 0.82, 0.55) + rng.normal(0, 0.07, m); block = np.abs(rng.normal(0.02, 0.015, m)); sharp = 1.0 + rng.normal(0, 0.08, m)
    lane *= np.where(rain == 1, 0.92, np.where(rain == 2, 0.78, np.where(rain >= 3, 0.58, 1.0))); block += np.where(rain == 2, 0.04, np.where(rain >= 3, 0.12, 0.0)); sharp *= np.where(rain > 0, 1 - 0.07 * rain, 1.0)
    fog = np.isin(code, (45, 48)); lane *= np.where(fog, 0.45, 1.0); sharp *= np.where(fog, 0.4, 1.0)
    night = lux < 60; lane *= np.where(night, 0.8, 1.0)
    glare = (elev > 0.5) & (elev < 12) & (_angdiff(azim, np.nan_to_num(head_m, nan=999)) < 35) & (world.cloud < 65) & (rain == 0) & drv
    lane *= np.where(glare, 0.45, 1.0)
    works_d = set(st.get("roadworks", {}).get("days", [])); dmin = (np.arange(m) * 60) // 86400
    lane *= np.where(np.isin(dmin, list(works_d)) & hwy_m, 0.5, 1.0)
    dirty = np.zeros(m, bool)
    if "dirty_windscreen" in st:
        d1, dd = st["dirty_windscreen"]["start"], st["dirty_windscreen"]["days"]; dirty = (dmin >= d1) & (dmin < d1 + dd); block += np.where(dirty, rng.uniform(0.30, 0.46, m), 0); lane *= np.where(dirty, 0.7, 1.0)
    heavy = (rain >= 3) & hwy_m & (rng.random(m) < 0.08); block += np.where(heavy, rng.uniform(0.2, 0.45, m), 0.0)
    lane = np.clip(lane, 0, 1); block = np.clip(block, 0, 1); sharp = np.clip(sharp, 0.05, 1.6); luma = np.clip(20 + 38 * np.log10(1 + lux / 50.0), 0, 255)
    block = np.where(drv, block, 0.0); lane = np.where(drv, lane, 0.0)
    state = np.zeros(m, np.int8); state[(block >= 0.30) & ~dirty] = 2; state[(state == 0) & (glare | fog)] = 1
    for (a, b) in _runs(state[:] == 2):                                                         # heavy-rain splash at speed: a few minutes
        if b - a < 2: continue
        ev.append((t0 + a * 60, "CAMERA_BLOCKED", "warn", f"front camera {int(round(block[a:b].max() * 100))}% of the view lost texture and structure")); ev.append((t0 + b * 60, "CAMERA_RESTORED", "info", f"front camera clear again after {(b - a) * 60:.0f} s"))
    if dirty.any():                                                                             # dirty windscreen: blocked from the first drive on the first dirty day until it is cleaned
        di = np.where(dirty & drv)[0]
        if len(di):
            a, b = int(di[0]), int(di[-1]); ev.append((t0 + a * 60, "CAMERA_BLOCKED", "warn", f"front camera {int(round(block[di].mean() * 100))}% of the view lost texture and structure (dirty windscreen)"))
            ev.append((t0 + b * 60 + 600, "CAMERA_RESTORED", "info", f"front camera clear again after {(b - a) / 1440:.1f} days (windscreen cleaned)"))
    for (a, b) in _runs(state[:] == 1):
        if b - a < 2: continue
        what = f"glare: {int(round(35 + 30 * float(rng.random())))}% of the view saturated" if glare[a:b].any() else "image blurred (sharpness 40% of normal)"
        ev.append((t0 + a * 60, "CAMERA_DEGRADED", "warn", f"front camera {what}")); ev.append((t0 + b * 60, "CAMERA_RESTORED", "info", f"front camera clear again after {(b - a) * 60:.0f} s"))
    idx = np.where(drv)[0]
    ex["cam_1m"] = [(t0 + int(i) * 60, 300, float(block[i]), float(min(1.0, block[i] * 1.6 + 0.03)), float(sharp[i]), float(luma[i]), float(lane[i])) for i in idx]
    # ---------------------------------------------------------------- extension signals per active minute (ext_1m)
    mode_m = gv.mode[: m * 60].reshape(m, 60); act_m = (mode_m != 0).any(1); tt_m = life.tyre_t[: m * 60].reshape(m, 60).mean(1); hours = np.minimum((np.arange(m) * 60) // 3600, life.cold.shape[1] - 1)
    cold_m = life.cold[:, hours]; pr = (cold_m + 101.325) * (tt_m[None, :] + 273.15) / (20 + 273.15) - 101.325 + rng.normal(0, 0.5, (4, m))
    odo_m = gv.odo[: m * 60][::60]; adas_m = adas_on[: m * 60].reshape(m, 60).mean(1) * 100; v12 = np.where(act_m, 13.95 + rng.normal(0, 0.07, m), 0.0); first_of_drive = drv & ~np.r_[False, drv[:-1]]
    rest_m = v12_rest[np.minimum(dmin, days)]; v12 = np.where(first_of_drive, rest_m - 0.35 + rng.normal(0, 0.04, m), v12)
    gv.adas_on = adas_on
    gv.minute = {"pr": pr, "v12": v12, "lane": lane, "block": block, "sharp": sharp, "luma": luma, "lux": lux, "rain": world.rain, "tt": tt_m, "adas_pct": adas_m, "act": act_m, "drv": drv, "v12_rest": v12_rest, "glare": glare}
    rows = []
    for i in np.where(act_m)[0]:
        if np.isnan(pos_lat[i]): continue
        rows.append((t0 + int(i) * 60, *(int(round(10 * pr[w, i])) for w in range(4)), int(world.rain[i]), int(lux[i]), int(round(100 * v12[i])), int(round(adas_m[i])), int(round(10 * odo_m[i])), int(round(pos_lat[i] * 1e5)), int(round(pos_lon[i] * 1e5)), int(round(10 * tt_m[i]))))
    ex["ext_1m"] = rows
    # ---------------------------------------------------------------- daily vehicle state, places, trips
    fric_cum = np.zeros(days + 1)
    for tr in gv.trips: fric_cum[min(days, tr["day"] + 1):] += tr["fric_kwh"]
    pad0 = float(rng.uniform(35, 92)); odo_d = [float(gv.odo[min(n - 1, d * 86400)]) for d in range(days + 1)]
    ex["vday"] = [(t0 + d * 86400, odo_d[d], max(2.0, pad0 - 0.032 * fric_cum[d]), 100 * life.soh0 - 0.0021 * d + float(rng.normal(0, 0.08)), float(v12_rest[d])) for d in range(days + 1)]
    gps_rows, tracks, gps_cut = gv.gps_tmp
    tracks.sort(); prev_end, plat, plon = 0, None, None
    for (a_, b_, la_, lo_) in tracks + [(n, n, None, None)]:
        if plat is None and la_ is not None: plat, plon = la_, lo_
        if plat is not None:
            for k in range(max(prev_end + 30, gps_cut), a_, 60): gps_rows.append((t0 + k, int(round(plat * 1e6)), int(round(plon * 1e6)), 0, 0, 10))     # parked: one fix a minute
        prev_end, plat, plon = b_, la_, lo_
    gps_rows.sort(); ex["gps_1s"] = list({r[0]: r for r in gps_rows}.values())
    ex["places"] = [dict(id=i + 1, name=p["name"], kind=p["kind"], lat=round(p["lat"], 5), lon=round(p["lon"], 5)) for i, p in enumerate(plan.places)]
    texts = []
    for tr in gv.trips:
        a, b = tr["a"], tr["b"]; texts.append((tr["start"], (a + 1) if a >= 0 else None, (b + 1) if b >= 0 else None, tr["harsh_brake"], tr["harsh_accel"], tr["harsh_corner"], tr["regen_kwh"], tr["rain_share"],
                                               round(100 * tr["adas_s"] / max(1, len(tr["v"])), 1), round(tr["pos_start"][0], 5), round(tr["pos_start"][1], 5), round(tr["pos_end"][0], 5), round(tr["pos_end"][1], 5)))
    ex["trip_ext"] = texts
    ev.sort()
    gv.events = ev; gv.ex = ex
    gv.meta = {"veh.model": V["label"], "veh.cap_kwh": f"{V['cap']:.1f}", "veh.soh": f"{100 * life.soh0:.1f}", "veh.odo_start": f"{life.odo0:.1f}", "veh.odo_end": f"{life.odo_km:.1f}", "veh.service_km": str(svc_int),
               "veh.last_service_km": f"{last_km:.0f}", "veh.last_service_ts": str(last_ts), "veh.tyre_spec_front": f"{V['tyre'][0]}", "veh.tyre_spec_rear": f"{V['tyre'][1]}", "veh.pad_start": f"{pad0:.1f}",
               "veh.home_charger_kw": f"{life.home_kw}", "veh.sw": "24.4.0" if any(e[1] == "SW_UPDATE" for e in ev) else "24.3.1", "env.city": gv.C["label"], "env.lat": f"{C['lat']}", "env.lon": f"{C['lon']}",
               "env.tz_h": f"{gv.tzh}", "gen.seed": str(gv.seed), "gen.driver": gv.driver, "gen.vehicle": gv.vehicle, "gen.stories": ",".join(gv.st), "gen.weather": world.source, "gen.driving_source": getattr(getattr(life, "seg", None), "source", "real")}
    gv.truth = make_truth(gv, svc_int, last_km, pad0)
    if verbose:
        from collections import Counter
        print("  events:", dict(Counter(e[1] for e in ev)))
    return gv


def make_truth(gv, svc_int, last_km, pad0):
    """Ground truth from the generator's arrays (independent of the in-car analytics)."""
    t0, n = gv.t0, gv.n; days = n // 86400; life = gv.life; d = lambda k: [0] * days; daily = []
    by = {}
    for tr in gv.trips: by.setdefault(tr["day"], []).append(tr)
    rain_active_min = []
    ign_m = (gv.mode[: (n // 60) * 60].reshape(-1, 60) != 0).any(1); rain = gv.world.rain          # minutes the car was awake (driving or charging): what its sensors can see
    for k in range(days):
        T = by.get(k, []); mm = slice(k * 1440, (k + 1) * 1440); ch = [c for c in gv.ex["charges"] if (c["start"] - t0) // 86400 == k]
        daily.append(dict(day=k, date=(gv.d0 + dt.timedelta(days=k)).isoformat(), trips=len(T), km=round(sum(t["km"] for t in T), 2), kwh=round(sum(t["kwh"] for t in T), 3), duration_min=round(sum((t["end"] - t["start"]) for t in T) / 60, 1),
                          harsh_brake=sum(t["harsh_brake"] for t in T), harsh_accel=sum(t["harsh_accel"] for t in T), harsh_corner=sum(t["harsh_corner"] for t in T), adas_s=sum(t["adas_s"] for t in T),
                          hwy_km=round(sum(t["km"] * t["hwy_share"] for t in T), 2), rain_active_min=int(((rain[mm] > 0) & ign_m[mm]).sum()), rain_heavy_active_min=int(((rain[mm] >= 3) & ign_m[mm]).sum()),
                          charge_sessions=len(ch), charge_kwh=round(sum(c["kwh"] for c in ch), 2), places=[(gv.plan.places[t["a"]]["name"] if t["a"] >= 0 else "stop", gv.plan.places[t["b"]]["name"] if t["b"] >= 0 else "stop") for t in T]))
    et = {}
    for ts, ty, sv, de in gv.events: et.setdefault(ty, []).append(ts)
    cold_end = [float(x) for x in life.cold[:, -2]]
    return {"meta": {"seed": gv.seed, "city": gv.city, "driver": gv.driver, "vehicle": gv.vehicle, "stories": gv.st, "t0": t0, "days": days, "tz_h": gv.tzh, "end": gv.end_d.isoformat(), "weather": gv.world.source},
            "daily": daily, "event_ts": et, "tyre_cold_end_kpa": cold_end, "tyre_spec_kpa": [float(x) for x in life.spec], "odo_start": life.odo0, "odo_end": life.odo_km, "service_km": svc_int, "last_service_km": last_km,
            "pad_start": pad0, "fric_kwh": life.fric_kwh, "regen_kwh": life.regen_kwh, "soh0": life.soh0, "trips": len(gv.trips), "total_km": round(sum(t["km"] for t in gv.trips), 1),
            "total_kwh": round(sum(t["kwh"] for t in gv.trips), 2), "wh_km": round(1000 * sum(t["kwh"] for t in gv.trips) / max(1, sum(t["km"] for t in gv.trips)), 1)}


# ------------------------------------------------------------------------------------------------ seeding
def seed(path, gv):
    from . import store
    store.seed_from_sim(path, gv)
    c = store.open_rw(path); c.execute("BEGIN")
    c.executemany("INSERT OR REPLACE INTO ext_1m VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", gv.ex["ext_1m"])
    c.executemany("INSERT OR REPLACE INTO cam_1m VALUES(?,?,?,?,?,?,?)", gv.ex["cam_1m"])
    c.executemany("INSERT OR REPLACE INTO vday VALUES(?,?,?,?,?)", gv.ex["vday"])
    c.executemany("INSERT OR REPLACE INTO gps_1s VALUES(?,?,?,?,?,?)", gv.ex["gps_1s"])
    c.executemany("INSERT OR REPLACE INTO places VALUES(?,?,?,?,?)", [(p["id"], p["name"], p["kind"], p["lat"], p["lon"]) for p in gv.ex["places"]])
    c.executemany("INSERT OR REPLACE INTO charges VALUES(?,?,?,?,?,?,?,?,?,?,?)", [(x["start"], x["end"], x["kind"], x["kw"], x["kwh"], x["soc0"], x["soc1"], (x["place"] + 1) if x["place"] >= 0 else None, round(x["lat"], 5), round(x["lon"], 5), round(x["kwh"] / (0.95 if x["kind"] == "DC" else 0.92), 2)) for x in gv.ex["charges"]])
    c.executemany("INSERT OR REPLACE INTO trip_ext VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", gv.ex["trip_ext"])
    c.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", list(gv.meta.items()))
    c.execute("COMMIT"); c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.execute("VACUUM"); c.close()
