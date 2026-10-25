"""Vehicle simulator (DEV TOOL - uses numpy; not part of the in-car footprint).

Creates 28 days of a coherent EV life and exposes it two ways:
  * Vehicle arrays + events + trips  -> bulk-loaded into the store (fast history pre-load)
  * live_messages(t0, seconds)       -> protocol-NATIVE messages (raw CAN frames, SOME/IP-style events, VHAL props, QNX/Linux log lines,
                                        UDS DTC reports, camera-pipeline metadata) that the in-car ingestion adapters must decode.
Scenarios are scripted and seeded (the brief allows CARLA OR simulated/recorded data; this is the simulated-data route and the same
adapter interface would accept CARLA-bridge or recorded-CAN replay):
  A  cold weather + more highway driving      -> energy per km rises, range falls
  B  cooling-pump degradation + DC fast charge -> coolant flow collapse, DTC, battery over-temperature warning (day 26)
  C  ADAS: lane-marking loss on roadworks days, camera blocked on rainy days
Ground truth for every benchmark answer is computed here from the same arrays (truth.json), independently of the in-car analytics.
"""
import json, math
from dataclasses import dataclass, field
import numpy as np

START = "2026-09-04"           # day 0; "now" in the demo is 2026-10-02 08:00 (so 'yesterday' = 2026-10-01)
DAYS = 28
CAP_KWH = 60.0


@dataclass
class Vehicle:
    t0: int
    n: int
    ign: np.ndarray; speed: np.ndarray; mode: np.ndarray; soc: np.ndarray; power: np.ndarray; batt: np.ndarray; flow: np.ndarray
    amb: np.ndarray; hvac: np.ndarray; motor: np.ndarray; odo: np.ndarray; chg: np.ndarray
    events: list = field(default_factory=list)      # (ts, type, severity, detail)
    trips: list = field(default_factory=list)       # dicts
    truth: dict = field(default_factory=dict)


try:
    from . import realcycles as _realcycles
    REAL = _realcycles.load()                       # None if the dataset is not available -> synthetic fallback
except Exception:
    REAL = None


def _smooth(x, k):
    p = np.pad(x, (k, k), 'edge'); return np.convolve(p, np.ones(k) / k, 'same')[k:-k]


MODE = {"parked": 0, "city": 1, "highway": 2, "charging": 3}


def _ts(day, hh, mm=0):
    return day * 86400 + hh * 3600 + mm * 60


def build(seed=11, days=DAYS):
    r = np.random.default_rng(seed)
    t0 = int(np.datetime64(START + "T00:00:00", "s").astype("int64"))
    n = days * 86400
    ign = np.zeros(n, bool); speed = np.zeros(n, np.float32); mode = np.zeros(n, np.int8); power = np.zeros(n, np.float32)
    chg = np.zeros(n, np.float32); hvac = np.zeros(n, np.float32)
    sec = np.arange(n); day = sec // 86400; hod = (sec % 86400) / 3600.0
    # ambient: slow cooling trend + daily cycle (story A)
    amb = (20 - 0.46 * (sec / 86400) + 5 * np.sin((hod - 9) / 24 * 2 * np.pi) + r.normal(0, .3, n).cumsum() * 0 + r.normal(0, .4, n)).astype(np.float32)
    trips, events = [], []
    rain_days, works_days = {15, 16, 22, 27}, {9, 10, 11, 12, 13}
    for d in range(days):
        wk = (t0 // 86400 + d + 3) % 7 < 5                     # 1970-01-01 was a Thursday -> (days+3)%7: 0=Mon
        plan = []
        hw = min(0.75, 0.30 + 0.014 * d)                       # story A: highway share grows
        if wk: plan += [(7.67 + r.uniform(-.1, .1), 55, hw), (17.1 + r.uniform(-.15, .15), 52, hw * 0.9)]
        else: plan += [(r.uniform(10, 16), r.uniform(25, 70), hw)]
        if d == 26: plan = [(10.5, 120, 1.0)]                 # story B: long highway trip, arrives at the fast charger with a low battery
        for (h0, mins, hshare) in plan:
            s0 = int(d * 86400 + h0 * 3600); L = int(mins * 60)
            # alternate city/highway blocks
            sp = np.zeros(L); md = np.zeros(L, np.int8); i = 0
            while i < L:
                hwy = r.random() < hshare; blk = int(r.uniform(300, 900)); blk = min(blk, L - i)
                if REAL is not None: v = _realcycles.block(r, REAL["hwy"] if hwy else REAL["city"], blk)       # REAL recorded OBD speed traces
                elif hwy: v = np.clip(100 + _smooth(r.normal(0, 7, blk), 30) * 3, 80, 125)
                else:
                    v = np.clip(35 + _smooth(r.normal(0, 25, blk), 8) * 2.2, 0, 65)
                    stops = r.random(blk // 60 + 1)
                    for j, sj in enumerate(stops):
                        if sj < 0.8: v[min(blk - 1, j * 60 + 20):min(blk, j * 60 + 32)] *= 0.05
                sp[i:i + blk] = v; md[i:i + blk] = 2 if hwy else 1; i += blk
            seg = slice(s0, s0 + L); ign[seg] = True; speed[seg] = sp; mode[seg] = md
            trips.append(dict(start=t0 + s0, end=t0 + s0 + L, day=d))
    # physics: power draw, hvac, soc, temperatures
    v = speed.astype(np.float64)
    hv = np.where(ign, np.clip((21 - amb) * 150 + 150, 0, 2800), 0) + np.where(ign, 300, 0)    # W: heating + auxiliaries
    hvac[:] = hv
    roof = np.where(sec >= 17 * 86400 + 12 * 3600, 1.30, 1.0)      # story D: roof box fitted on day 17 -> aerodynamic penalty
    wh_km = np.where(ign & (v > 3), (88 + 0.0085 * v ** 2 + np.where(mode == 1, 20, 0)) * roof, 0)
    power[:] = np.where(ign, wh_km * v / 1000 + hv / 1000, 0.0)
    flow = np.full(n, 12.0) + r.normal(0, .15, n)
    batt = np.empty(n); soc = np.empty(n); b = amb[0] + 3.0; s = 82.0
    chg_sessions = []
    for d in range(days):                                       # nightly AC charge + two DC fast charges
        s0 = d * 86400 + 22 * 3600; chg_sessions.append((s0, 300 * 60, 11.0, "AC"))
    for d_dc, h_dc in ((8, 19.5), (26, 12.95)):
        chg_sessions.append((int(d_dc * 86400 + h_dc * 3600), 55 * 60, 118.0, "DC"))
    for (s0, L, kw, kind) in chg_sessions:
        if s0 + L < n: chg[s0:s0 + L] = kw; mode[s0:s0 + L] = MODE["charging"]
    # story B: cooling pump degrades from day 20 (slow drift), collapses during the day-26 fast charge
    drift = np.clip((sec - 20 * 86400) / (8 * 86400), 0, 1); flow -= 2.2 * drift
    k0 = 26 * 86400 + 13 * 3600 + 2 * 60; collapse = np.clip((sec - k0) / (15 * 60), 0, 1); flow = np.where((sec >= k0) & (sec < k0 + 3600), flow - 7.0 * collapse, flow)
    batt = np.empty(n); soc = np.empty(n); b = float(amb[0]) + 6.0; s = 82.0
    C_TH, UA0 = 150e3, 250.0                                     # pack thermal mass J/K, liquid-cooling conductance W/K at nominal flow
    for i in range(n):                                           # 1 s step: SOC + lumped battery thermal model
        kw = chg[i]
        if kw > 0:
            kw = kw if (s < 80 or kw > 50) else kw * max(0.05, (100 - s) / 20.0)   # AC tapers above 80 %; DC held (story needs sustained heat)
            if s >= 100: kw = 0.0
            power[i] = -kw
        p = power[i]
        heat_w = (0.075 if abs(p) > 50 else 0.04) * abs(p) * 1000.0   # conversion loss -> heat (aged pack: higher resistive loss at DC power)
        ua = UA0 * (max(flow[i], 0.0) / 12.0)
        tcool = amb[i] if (ign[i] or kw > 0) else amb[i]
        b += (heat_w - ua * (b - tcool)) / C_TH
        batt[i] = b
        s = min(100.0, s - p / 3600.0 / CAP_KWH * 100.0); soc[i] = s
    motor = (amb + 18 + np.clip(power, 0, None) * 0.6 + r.normal(0, .5, n)).astype(np.float32)
    odo = 12000 + np.cumsum(np.where(ign, v / 3600.0, 0))
    pack_v = None
    veh = Vehicle(t0, n, ign, speed, mode, soc.astype(np.float32), power, batt.astype(np.float32), flow.astype(np.float32), amb, hvac, motor, odo.astype(np.float32), chg)
    veh.trips = trips
    # events -------------------------------------------------------------------
    for tr in trips:
        a, e = tr["start"] - t0, tr["end"] - t0
        sl = slice(a, e); dist = float(v[sl].sum() / 3600.0); kwh = float(power[sl].sum() / 3600.0)
        tr.update(km=round(dist, 2), kwh=round(kwh, 3), wh_km=round(1000 * kwh / max(dist, 0.1), 1), avg_speed=round(float(v[sl].mean()), 1),
                  hwy_share=round(float((mode[sl] == 2).mean()), 3), avg_amb=round(float(amb[sl].mean()), 1), avg_hvac_w=round(float(hvac[sl].mean()), 0),
                  soc_start=float(soc[a]), soc_end=float(soc[e - 1]))
    for tr in trips:                                            # story C: ADAS disengagements while on highway segments
        a, e = tr["start"] - t0, tr["end"] - t0; d = tr["day"]; hrs = (e - a) / 3600.0
        hwy = np.where(mode[a:e] == 2)[0]
        if len(hwy) < 120: continue
        rate = 0.25
        reason_pool = [("driver override", 0.6), ("lane marking lost", 0.3), ("camera blocked", 0.1)]
        morning = (a % 86400) < 12 * 3600
        if d in works_days and morning: rate, reason_pool = 5.0, [("lane marking lost", 0.9), ("driver override", 0.1)]
        if d in rain_days: rate, reason_pool = 3.0, [("camera blocked", 0.8), ("driver override", 0.2)]
        for _ in range(r.poisson(rate * len(hwy) / 3600.0)):
            ts = t0 + a + int(r.choice(hwy)); reasons, w = zip(*reason_pool)
            events.append((ts, "ADAS_DISENGAGE", "info", str(r.choice(reasons, p=np.array(w) / sum(w)))))
    for (s0, L, kw, kind) in chg_sessions:
        if s0 + L < n: events.append((t0 + s0, "CHARGE_START", "info", f"{kind} {kw:.0f} kW"))
    # story B events
    dtc_idx = int(np.argmax((flow < 6.0) & (sec > k0))); ev_t = t0 + dtc_idx; events.append((ev_t, "DTC_SET", "warn", "P1A00 Battery coolant flow low (illustrative code)"))
    warn_idx = int(np.argmax((batt > 52.0) & (sec > k0)))
    events.append((t0 + warn_idx, "WARN_BATT_OVERTEMP", "critical", "Battery temperature above 52 C"))
    events.sort()
    veh.events = events
    # ground truth ---------------------------------------------------------------
    dd = {}
    for tr in trips: dd.setdefault(tr["day"], []).append(tr)
    daily = {d: (sum(t["kwh"] for t in L), sum(t["km"] for t in L)) for d, L in dd.items()}
    daily_whkm = {d: 1000 * k / max(m, 0.1) for d, (k, m) in daily.items()}
    days_sorted = sorted(daily_whkm); deltas = {d: daily_whkm[d] - daily_whkm[p] for p, d in zip(days_sorted, days_sorted[1:])}
    wd = max(deltas, key=deltas.get)
    c_mask = (mode == 1) & ign & (v > 3); h_mask = (mode == 2) & ign & (v > 3)
    def seg_whkm(mask): return round(float(1000 * (power[mask].sum() / 3600.0) / (v[mask].sum() / 3600.0)), 1)
    first, last = days_sorted[:3], days_sorted[-3:]
    ev_by = {}
    for ts, ty, sv, de in events: ev_by.setdefault(ty, []).append((ts, de))
    yday = 27; ys = t0 + yday * 86400; ye = ys + 86400
    veh.truth = {
        "now": "2026-10-02T08:00:00", "yesterday": "2026-10-01",
        "whkm_first3": round(float(np.mean([daily_whkm[d] for d in first])), 1), "whkm_last3": round(float(np.mean([daily_whkm[d] for d in last])), 1),
        "amb_first3": round(float(np.mean([t["avg_amb"] for d in first for t in dd[d]])), 1), "amb_last3": round(float(np.mean([t["avg_amb"] for d in last for t in dd[d]])), 1),
        "hwy_share_first3": round(float(np.mean([t["hwy_share"] for d in first for t in dd[d]])), 3), "hwy_share_last3": round(float(np.mean([t["hwy_share"] for d in last for t in dd[d]])), 3),
        "largest_eff_drop_day": (np.datetime64(START) + int(wd)).astype(str), "largest_eff_drop_delta_whkm": round(float(deltas[wd]), 1),
        "city_whkm": seg_whkm(c_mask), "highway_whkm": seg_whkm(h_mask),
        "adas_total": len(ev_by.get("ADAS_DISENGAGE", [])), "adas_yesterday": sum(1 for ts, _ in ev_by.get("ADAS_DISENGAGE", []) if ys <= ts < ye),
        "adas_works_reason_lane": sum(1 for ts, de in ev_by.get("ADAS_DISENGAGE", []) if de == "lane marking lost"),
        "overtemp_ts": int(ev_by["WARN_BATT_OVERTEMP"][0][0]), "dtc_ts": int(ev_by["DTC_SET"][0][0]),
        "flow_before_overtemp": round(float(flow[warn_idx - 3000]), 1), "flow_at_overtemp": round(float(flow[warn_idx]), 1), "batt_at_overtemp": round(float(batt[warn_idx]), 1),
        "fastcharge_before_overtemp": True, "trips": len(trips), "total_km": round(float(v[ign].sum() / 3600.0), 1),
    }
    # extended reference facts used to score EVERY answer family (computed from the arrays, independent of the in-car SQL/analytics)
    now_ts = t0 + n + 8 * 3600; today0 = now_ts - now_ts % 86400
    wins = {"y": (today0 - 86400, today0), "t": (today0, now_ts), "7": (today0 - 7 * 86400, now_ts), "14": (today0 - 14 * 86400, now_ts), "28": (today0 - 28 * 86400, now_ts), "a": (0, now_ts)}
    veh.truth["events_counts"] = {ty: {w: sum(1 for ts, t2, sv, de in events if t2 == ty and lo <= ts < hi) for w, (lo, hi) in wins.items()}
                                  for ty in ("ADAS_DISENGAGE", "WARN_BATT_OVERTEMP", "DTC_SET", "CHARGE_START", "CAMERA_BLOCKED")}
    lo14, hi14 = wins["14"]; dd14 = {}
    for tr in trips:
        if lo14 <= tr["start"] < hi14: dd14.setdefault(tr["day"], []).append(tr)
    dk = sorted(dd14); d14 = {d: 1000 * sum(t["kwh"] for t in dd14[d]) / max(sum(t["km"] for t in dd14[d]), 0.1) for d in dk}
    f14, l14 = float(np.mean([d14[d] for d in dk[:3]])), float(np.mean([d14[d] for d in dk[-3:]]))
    veh.truth["why14"] = {"first": round(f14, 1), "last": round(l14, 1), "increase_pct": round(100 * (l14 - f14) / f14, 1)}
    return veh


# ------------------------------------------------------------------- real wire-format live messages (see livegen.py)
def live_messages(veh, start_s, seconds, can_hz=10):
    """Protocol-native traffic for the simulated vehicle: CAN/CAN-FD (DBC), ISO-TP/UDS, OBD-II, SOME/IP(+SD), Android VHAL, QNX/Android logs."""
    from .livegen import live_messages as _lm
    return _lm(veh, start_s, seconds, can_hz)


def save_truth(veh, path):
    json.dump(veh.truth, open(path, "w"), indent=1)


if __name__ == "__main__":
    import time
    t = time.time(); v = build(); print("built in %.1fs" % (time.time() - t)); print(json.dumps(v.truth, indent=1))
    print("events:", len(v.events), "trips:", len(v.trips))
