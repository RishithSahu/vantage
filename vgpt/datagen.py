"""DEV tool - random but realistic EV life generator (numpy). Replaces hand-scripted stories with a seeded, profile-driven simulation built on REAL data:

  real weather (Open-Meteo hourly: temperature, rain, cloud, sunlight)  -> ambient temperature, rain-sensor/wiper state, ambient light, HVAC load, glare, rainy-day behaviour
  real driving (VIF-AEGIS 35 trips with accelerometer + EPA UDDS/HWFET/US06/SC03/FTP) -> speed traces, hard braking / acceleration / cornering from the REAL accelerometer
  vehicle physics (mass, CdA, rolling resistance vs tyre pressure and temperature, regen, heat-pump HVAC, battery thermal model, charging curves vs temperature)
  driver profiles (commuter, delivery, long-distance, retiree, rideshare), vehicle profiles (28-78 kWh), cities (7), places (home, work, shops ...) and trips between them
  random faults: cooling-pump degradation, slow tyre leak, weak 12 V battery, dirty windscreen, roadworks lane loss, roof box ... (chosen by the seed, recorded in the truth file)
Everything the assistant can be asked about (tyres, rain, light, service, brakes, driving style, ADAS use, places, charging, camera health) therefore has data behind it, with ground truth
computed here from the arrays - independently of the in-car analytics."""
import datetime as dt, json, math, os, pickle, time
import numpy as np
from . import datasets as D
from .sim import Vehicle

G = 9.80665
MODE = {"parked": 0, "city": 1, "highway": 2, "charging": 3}

CITIES = {   # lat, lon, IANA zone, motorway speed cap (km/h), congestion (1 = free flowing), place names
    "bengaluru": dict(label="Bengaluru", lat=12.9716, lon=77.5946, tz="Asia/Kolkata", cap=100, cong=1.35),
    "graz": dict(label="Graz", lat=47.0707, lon=15.4395, tz="Europe/Vienna", cap=130, cong=0.9),
    "london": dict(label="London", lat=51.5074, lon=-0.1278, tz="Europe/London", cap=112, cong=1.25),
    "oslo": dict(label="Oslo", lat=59.9139, lon=10.7522, tz="Europe/Oslo", cap=110, cong=0.9),
    "chicago": dict(label="Chicago", lat=41.8781, lon=-87.6298, tz="America/Chicago", cap=120, cong=1.1),
    "phoenix": dict(label="Phoenix", lat=33.4484, lon=-112.0740, tz="America/Phoenix", cap=120, cong=1.0),
    "singapore": dict(label="Singapore", lat=1.3521, lon=103.8198, tz="Asia/Singapore", cap=90, cong=1.15),
    "sydney": dict(label="Sydney", lat=-33.8688, lon=151.2093, tz="Australia/Sydney", cap=110, cong=1.1),
}
VEHICLES = {   # nominal usable kWh, mass kg, CdA m2, rolling coeff, drivetrain eff, regen kW, aux W, heat-pump?, AC kW, DC kW, tyre cold pressure kPa (front, rear), top km/h
    "city": dict(label="City EV 28 kWh", cap=28, mass=1250, cda=0.60, crr=0.0100, eta=0.89, regen=40, aux=300, hp=False, ac=7.4, dc=50, tyre=(230, 230), top=140),
    "compact": dict(label="Compact EV 42 kWh", cap=42, mass=1550, cda=0.66, crr=0.0095, eta=0.90, regen=55, aux=330, hp=True, ac=11.0, dc=100, tyre=(250, 250), top=150),
    "family": dict(label="Family EV 60 kWh", cap=60, mass=1850, cda=0.72, crr=0.0095, eta=0.91, regen=65, aux=350, hp=True, ac=11.0, dc=150, tyre=(240, 260), top=160),
    "suv": dict(label="SUV EV 78 kWh", cap=78, mass=2150, cda=0.85, crr=0.0100, eta=0.90, regen=80, aux=380, hp=True, ac=11.0, dc=175, tyre=(250, 270), top=170),
}
DRIVERS = {   # style: aggressiveness weight for picking real driving segments; adas: probability the driver uses lane keeping on motorways
    "commuter": dict(label="Weekday commuter", aggr=0.0, adas=0.6, plug=0.7, target=0.9, work_charge=0.25, desc="home-work trips on weekdays, errands and the odd long weekend trip"),
    "delivery": dict(label="Delivery driver", aggr=0.4, adas=0.0, plug=1.0, target=1.0, work_charge=0.0, desc="12-30 short city trips a day, six days a week"),
    "long_distance": dict(label="Long-distance sales rep", aggr=0.1, adas=0.85, plug=0.8, target=0.9, work_charge=0.0, desc="long motorway days (120-320 km) plus short trips"),
    "retiree": dict(label="Relaxed local driver", aggr=-0.6, adas=0.3, plug=0.5, target=0.85, work_charge=0.0, desc="a few short midday trips, rarely on motorways"),
    "rideshare": dict(label="Rideshare driver", aggr=0.2, adas=0.0, plug=1.0, target=1.0, work_charge=0.0, desc="10-hour shifts of continuous city trips"),
}
PLACE_NAMES = ["Supermarket", "Gym", "School", "Shopping mall", "Restaurant", "Clinic", "Friends' place", "Park", "Cinema", "Garden centre"]
STORIES = ("cooling_pump", "tyre_leak", "weak_12v", "dirty_windscreen", "roadworks", "roof_box", "harsh_week", "alignment_drift")
THR_HARSH, THR_CORNER = 0.28, 0.35            # g, 1 s means. Real VIF-AEGIS accelerometer rates per 100 km: 0.28 g -> ~3-6 hard brakes, 0.35 g -> ~3 hard corners


# ------------------------------------------------------------------------------------------------ helpers
def tz_offset_h(zone, date):
    try:
        import zoneinfo
        d = dt.datetime(date.year, date.month, date.day, 12, tzinfo=zoneinfo.ZoneInfo(zone)); return d.utcoffset().total_seconds() / 3600.0
    except Exception:
        return {"Asia/Kolkata": 5.5, "Europe/Vienna": 1, "Europe/London": 0, "Europe/Oslo": 1, "America/Chicago": -6, "America/Phoenix": -7, "Asia/Singapore": 8, "Australia/Sydney": 10}.get(zone, 0)


def solar(lat, lon, t):
    """Sun elevation / azimuth (deg, azimuth clockwise from north) for epoch seconds `t` (NOAA approximation)."""
    t = np.asarray(t, np.float64); d = t / 86400.0 + 2440587.5 - 2451545.0
    g = np.radians((357.529 + 0.98560028 * d) % 360); q = (280.459 + 0.98564736 * d) % 360
    L = np.radians((q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360); e = np.radians(23.439 - 0.00000036 * d)
    ra = np.arctan2(np.cos(e) * np.sin(L), np.cos(L)); dec = np.arcsin(np.sin(e) * np.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24; lst = np.radians(((gmst + lon / 15.0) % 24) * 15.0); ha = lst - ra; la = np.radians(lat)
    sin_el = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha); el = np.arcsin(np.clip(sin_el, -1, 1))
    az = np.arctan2(-np.sin(ha) * np.cos(dec), np.sin(dec) * np.cos(la) - np.cos(dec) * np.sin(la) * np.cos(ha))
    return np.degrees(el), (np.degrees(az) + 360) % 360


def movavg(x, k):
    """Same-length moving average (edge padded), safe for short arrays."""
    x = np.asarray(x, np.float64); k = max(1, min(k, len(x)))
    c = np.cumsum(np.r_[0.0, x]); out = (c[k:] - c[:-k]) / k; lo = k // 2
    return np.pad(out, (lo, len(x) - len(out) - lo), "edge")


def haversine_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); dphi = p2 - p1; dl = math.radians(b[1] - a[1])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2; return 12742 * math.asin(math.sqrt(h))


def bearing(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0]); dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2); x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl); return (math.degrees(math.atan2(y, x)) + 360) % 360


def offset(p, km, brg):
    la = math.radians(p[0]); b = math.radians(brg)
    return (p[0] + km * math.cos(b) / 111.2, p[1] + km * math.sin(b) / (111.2 * max(0.2, math.cos(la))))


# ------------------------------------------------------------------------------------------------ weather world
class World:
    """Per-second / per-minute environment from REAL hourly weather (or a synthetic fallback when offline)."""
    def __init__(self, rng, city, t0, n, offline=False):
        c = CITIES[city]; self.city, self.t0, self.n = c, t0, n; self.source = "Open-Meteo archive (real hourly weather)"
        d0 = (dt.datetime.fromtimestamp(t0, dt.timezone.utc) - dt.timedelta(days=1)).date(); d1 = (dt.datetime.fromtimestamp(t0 + n, dt.timezone.utc) + dt.timedelta(days=1)).date()
        w = None
        if not offline:
            try: w = D.weather(c["lat"], c["lon"], d0.isoformat(), d1.isoformat())
            except Exception as e: print("  (real weather is not in the local cache: %s - using synthetic weather)" % type(e).__name__)
        if w is None: w = self._synthetic(rng, c, d0, d1); self.source = "synthetic weather (offline fallback)"
        self.w = w; m = n // 60; tmin = t0 + np.arange(m) * 60.0; tsec = t0 + np.arange(n, dtype=np.float64)
        self.amb = (np.interp(tsec, w["t"], w["temp"]) + 0.6 + rng.normal(0, 0.15, n).cumsum() * 0.0).astype(np.float32)       # urban heat island +0.6 C
        wob = np.convolve(rng.normal(0, 0.35, n // 60 + 40), np.ones(20) / 20, "same")[: n // 60 + 1]; self.amb += np.repeat(wob, 60)[:n].astype(np.float32)
        hidx = np.clip(((tmin - w["t"][0]) // 3600).astype(int), 0, len(w["t"]) - 1)
        self.precip_h = w["precip"][hidx]; self.cloud = np.interp(tmin, w["t"], w["cloud"]).astype(np.float32); self.wind = np.interp(tmin, w["t"], w["wind"]).astype(np.float32)
        self.sun = np.interp(tmin, w["t"], w["sun"]).astype(np.float32); self.code = w["code"][hidx]
        self.lux = (self.sun * 105.0 * rng.uniform(0.9, 1.1, m) + np.where(self.sun < 1, rng.uniform(1, 8, m), 0)).astype(np.float32)
        lvl = np.zeros(m, np.int8)
        for h in np.unique(hidx):                                                # rain arrives in bursts inside the hour, not for the whole hour
            p = float(w["precip"][h])
            if p < 0.05: continue
            idx = np.where(hidx == h)[0]; L = int(round(len(idx) * min(1.0, 0.25 + p / 2.5))); s = int(rng.integers(0, max(1, len(idx) - L + 1)))
            lvl[idx[s:s + L]] = 1 if p < 0.5 else (2 if p < 2.5 else 3)
        self.rain = lvl; self.elev, self.azim = solar(c["lat"], c["lon"], tmin)

    @staticmethod
    def _synthetic(rng, c, d0, d1):
        hrs = int((d1 - d0).days * 24 + 24); t = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.timezone.utc).timestamp()) + np.arange(hrs) * 3600
        hod = ((t // 3600) % 24); doy = ((t // 86400) % 365)
        temp = 22 - 14 * np.cos((doy - 15) / 365 * 2 * np.pi) * (1 if c["lat"] > 0 else -1) * (abs(c["lat"]) / 60) + 5 * np.sin((hod - 9) / 24 * 2 * np.pi) + np.convolve(rng.normal(0, 2, hrs), np.ones(24) / 12, "same")
        wet = np.convolve(rng.random(hrs) < 0.04, np.ones(5), "same") > 0; precip = np.where(wet, rng.gamma(1.2, 1.2, hrs), 0)
        el, _ = solar(c["lat"], c["lon"], t + 1800); sun = np.clip(900 * np.sin(np.radians(np.clip(el, 0, 90))), 0, None) * np.where(wet, 0.25, rng.uniform(0.7, 1.0, hrs))
        return {"t": t.astype(np.int64), "temp": temp.astype(np.float32), "precip": precip.astype(np.float32), "cloud": np.where(wet, 90, rng.uniform(5, 60, hrs)).astype(np.float32), "rh": np.full(hrs, 60, np.float32),
                "wind": rng.uniform(2, 20, hrs).astype(np.float32), "code": np.where(wet, 61, 1).astype(np.int16), "sun": sun.astype(np.float32), "is_day": (el > 0).astype(np.float32), "pressure": np.full(hrs, 1013, np.float32)}


# ------------------------------------------------------------------------------------------------ real-data speed traces
class Segments:
    def __init__(self, cache=os.path.join(D.DATA, "segments.pkl")):
        self.pools = None; self.source = "real"
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(D.VIF_CACHE if os.path.exists(D.VIF_CACHE) else cache):
            with open(cache, "rb") as f: self.pools = pickle.load(f)
            if not (self.pools.get("city") and self.pools.get("hwy")): self.pools = None                  # an empty cache (made when no real trace was there) is not a cache
        if self.pools is None:
            self.pools = D.segments(); self.source = D.LAST_SOURCE["source"]
            if self.source == "real" and self.pools["city"] and self.pools["hwy"]:
                with open(cache, "wb") as f: pickle.dump(self.pools, f)                                      # only real traces are cached; a fixture is never written over the real cache
        if not self.pools["city"] or not self.pools["hwy"]: raise RuntimeError("no real driving data found (VIF-AEGIS zip in Downloads / EPA files in data/ref/epa) - see vgpt/datasets.py")
        self.info = {}
        for k, pool in self.pools.items():
            if not pool: continue
            a = np.array([float(np.percentile(np.abs(s["lon_g"]), 95)) for s in pool]); self.info[k] = ((a - a.mean()) / (a.std() + 1e-6), np.array([s["v"].sum() / 3600.0 for s in pool]))

    def take(self, rng, kind, km, aggr):
        pool = self.pools[kind] if self.pools.get(kind) else self.pools["hwy" if kind == "fast" else "city"]; z, kms = self.info[kind if self.pools.get(kind) else ("hwy" if kind == "fast" else "city")]
        w = np.exp(np.clip(aggr, -1.5, 1.5) * z); w /= w.sum(); out, tot = [], 0.0
        while tot < km:
            i = int(rng.choice(len(pool), p=w)); out.append(pool[i]); tot += float(kms[i])
        return out


def smooth_join(v, j, half=5):
    a, b = max(0, j - half), min(len(v) - 1, j + half)
    if b > a: v[a:b + 1] = np.linspace(v[a], v[b], b - a + 1)


def limit_accel(v, up=3.4, down=6.0):
    """Rate-limit a 1 Hz speed trace (km/h) to what a road car can do: acceleration fades with speed (power limited), braking up to ~0.6 g. Removes the jumps where real traces were stitched together."""
    v = np.asarray(v, dtype=np.float64).copy()
    for _ in range(2):
        for i in range(1, len(v)):
            lim = up * (1.0 - 0.65 * min(1.0, v[i - 1] / 130.0)) * 3.6
            if v[i] - v[i - 1] > lim: v[i] = v[i - 1] + lim
        for i in range(len(v) - 2, -1, -1):
            if v[i] - v[i + 1] > down * 3.6: v[i] = v[i + 1] + down * 3.6
    return v


def build_speed(rng, seg, km, hwy, aggr, traffic, rain, cap):
    """-> v km/h (1 Hz), lon_g, lat_g for a trip of ~km kilometres, `hwy` share on motorway, from real segments."""
    city_km, hwy_km = km * (1 - hwy), km * hwy
    if hwy_km < 0.8: city_km, hwy_km = km, 0.0
    head = city_km if hwy_km == 0 else min(city_km, max(1.2, 0.3 * city_km)); tail = city_km - head
    plan = [("city", head)] + ([("fast" if cap >= 118 else "hwy", hwy_km)] if hwy_km > 0 else []) + ([("city", tail)] if tail > 0.3 else [])
    sc = min(1.05, traffic * (0.93 if rain else 1.0))

    def proc(items):
        V, LO, LA, junctions, n0 = [], [], [], [], 0
        for kind, k in items:
            for s_ in seg.take(rng, kind, k, aggr):
                V.append(s_["v"]); LO.append(s_["lon_g"]); LA.append(s_["lat_g"]); n0 += len(s_["v"]); junctions.append(n0)
        v = np.concatenate(V).astype(np.float64); lo = np.concatenate(LO).astype(np.float64); la = np.concatenate(LA).astype(np.float64)
        for j in junctions[:-1]: smooth_join(v, j)
        scale = sc + (1.0 - min(sc, 1.0)) * 0.8 * np.clip((v - 50) / 40.0, 0, 1)                # traffic slows urban speeds, far less so on the motorway; continuous in v, so no artificial jumps
        v = v * scale; over = v > cap; v[over] = cap - np.abs(rng.normal(2, 1.5, int(over.sum())))   # national motorway limit
        lo = lo * scale; la = la * scale ** 2
        for _ in range(int(len(v) / 60 * (0.20 if traffic > 0.95 else 0.45) * (1.4 if rain else 1.0))):     # extra stop-and-go: smooth 30-80 s dips, only in slow traffic
            i = int(rng.integers(30, max(31, len(v) - 100))); L = int(rng.integers(30, 80)); w = 1 - np.sin(np.linspace(0, np.pi, L)) ** 2 * 0.97
            if v[i:i + L].max() < 62: v[i:i + L] *= w
        return limit_accel(v), lo, la
    v, lo, la = proc(plan); tries = 0
    while v.sum() / 3600.0 < km and tries < 6:                                                  # traffic slows the real traces, so draw more until the distance is reached
        kind = plan[-1][0]; v2, lo2, la2 = proc([(kind, max(1.0, (km - v.sum() / 3600.0) * 1.25))]); smooth_join(np.r_[v, v2], len(v)); v, lo, la = np.r_[v, v2], np.r_[lo, lo2], np.r_[la, la2]; tries += 1
    cum = np.cumsum(v) / 3600.0; k = int(np.searchsorted(cum, km)) + 1; k = max(60, min(k, len(v)))
    v, lo, la = v[:k].copy(), lo[:k].copy(), la[:k].copy(); r = min(14, k // 4); v[-r:] *= np.linspace(1, 0, r); v[:10] *= np.linspace(0, 1, 10) ** 1.2        # park / pull away
    v[-1] = 0.0; v = limit_accel(v); v[-1] = 0.0                                              # stopping ramps obey the same braking limit as everything else
    lon_d = np.gradient(v / 3.6) / G
    lon = np.where(np.abs(lo) > np.abs(lon_d), lo, lon_d)
    return v, lon, la


# ------------------------------------------------------------------------------------------------ planning
class Plan:
    def __init__(self, rng, city, driver, days, d0, tzh, work_km=None):
        self.rng, self.city, self.driver, self.days, self.d0, self.tzh = rng, CITIES[city], DRIVERS[driver], days, d0, tzh
        c = self.city; ctr = (c["lat"], c["lon"]); home = offset(ctr, rng.uniform(2, 11), rng.uniform(0, 360))
        self.places = [{"name": "Home", "kind": "home", "lat": home[0], "lon": home[1]}]
        dw = {"commuter": rng.uniform(5, 28), "delivery": 0, "long_distance": rng.uniform(4, 15), "retiree": 0, "rideshare": 0}[driver]
        if work_km and driver in ("commuter", "long_distance"): dw = float(work_km)
        if dw:
            w = offset(home, dw, rng.uniform(0, 360)); self.places.append({"name": "Work", "kind": "work", "lat": w[0], "lon": w[1]})
        for nm in rng.choice(PLACE_NAMES, size=5, replace=False): p = offset(home, rng.uniform(0.8, 9), rng.uniform(0, 360)); self.places.append({"name": str(nm), "kind": "other", "lat": p[0], "lon": p[1]})
        far = offset(home, rng.uniform(60, 170), rng.uniform(0, 360)); self.places.append({"name": "Out of town", "kind": "far", "lat": far[0], "lon": far[1]})
        self.by = {p["name"]: i for i, p in enumerate(self.places)}; self.route_km = {}
        self.wfh = {d for d in range(days) if rng.random() < 0.2}
        self.has_work = "Work" in self.by

    def pid(self, name): return self.by[name]

    def km_between(self, a, b):
        key = (min(a, b), max(a, b))
        if key not in self.route_km: self.route_km[key] = haversine_km((self.places[a]["lat"], self.places[a]["lon"]), (self.places[b]["lat"], self.places[b]["lon"])) * self.rng.uniform(1.22, 1.42)
        return self.route_km[key] * float(self.rng.uniform(0.96, 1.06))

    def hwy_share(self, a, b, km):
        key = ("h", min(a, b), max(a, b))
        if key not in self.route_km:
            self.route_km[key] = 0.85 if "far" in (self.places[a]["kind"], self.places[b]["kind"]) else (float(self.rng.uniform(0.35, 0.8)) if km > 14 else (float(self.rng.uniform(0.0, 0.3)) if km > 7 else 0.0))
        return self.route_km[key]

    def trip(self, t, a, b, **k): return dict(t=t, a=a, b=b, **k)

    def schedule(self):
        r = self.rng; out = []; H = self.pid("Home"); others = [i for i, p in enumerate(self.places) if p["kind"] == "other"]; far = self.pid("Out of town"); drv = None
        for d in range(self.days):
            date = self.d0 + dt.timedelta(days=d); wd = date.weekday(); base = d * 86400; dayT = []
            def at(h): return base + int(h * 3600)
            kind = self.driver_kind = next(k for k, v in DRIVERS.items() if v is self.driver)
            if kind == "commuter":
                if wd < 5 and d not in self.wfh and self.has_work:
                    W = self.pid("Work"); dayT.append(self.trip(at(r.normal(8.0, 0.45)), H, W, rush=True))
                    ret = at(r.normal(17.8, 0.8))
                    if r.random() < 0.25: s = int(r.choice(others)); dayT += [self.trip(ret, W, s), self.trip(ret + int(r.uniform(1500, 3500)), s, H)]
                    else: dayT.append(self.trip(ret, W, H, rush=True))
                    if r.random() < 0.10: s = int(r.choice(others)); dayT += [self.trip(at(12.5), W, s), self.trip(at(13.3), s, W)]
                else:
                    if r.random() < (0.5 if wd < 5 else 0.75):
                        t = at(r.uniform(10, 17))
                        for _ in range(int(r.integers(1, 3))): s = int(r.choice(others)); dayT += [self.trip(t, H, s), self.trip(t + int(r.uniform(2400, 6000)), s, H)]; t += int(r.uniform(9000, 14000))
                    if wd == 5 and r.random() < 0.25: t = at(r.uniform(8, 11)); dayT += [self.trip(t, H, far), self.trip(t + int(r.uniform(18000, 28000)), far, H)]
            elif kind == "delivery":
                if wd != 6:
                    t = at(r.normal(8.0, 0.3)); cur = H
                    for _ in range(int(r.integers(12, 28))):
                        s = int(r.choice(others)); dayT.append(self.trip(t, cur, s)); t += int(r.uniform(900, 2400)); cur = s
                        if t > at(19): break
                    dayT.append(self.trip(t, cur, H))
            elif kind == "long_distance":
                if wd < 4 and r.random() < 0.5:
                    t = at(r.uniform(5.5, 8.5)); dayT += [self.trip(t, H, far, rush=False), self.trip(t + int(r.uniform(30000, 40000)), far, H)]
                elif r.random() < 0.55:
                    t = at(r.uniform(9, 17)); s = int(r.choice(others)); dayT += [self.trip(t, H, s), self.trip(t + 4200, s, H)]
                if self.has_work and wd < 5 and r.random() < 0.3: dayT += [self.trip(at(8.2), H, self.pid("Work"), rush=True), self.trip(at(r.normal(17.5, 0.6)), self.pid("Work"), H, rush=True)]
            elif kind == "retiree":
                if r.random() < 0.65:
                    t = at(r.uniform(9.5, 15))
                    for _ in range(int(r.integers(1, 3))): s = int(r.choice(others)); dayT += [self.trip(t, H, s), self.trip(t + int(r.uniform(2400, 5400)), s, H)]; t += int(r.uniform(9000, 16000))
                if wd == 6 and r.random() < 0.15: t = at(10); dayT += [self.trip(t, H, far), self.trip(t + 24000, far, H)]
            elif kind == "rideshare":
                if wd != 2:
                    t = at(r.normal(9.0, 0.4)); cur = H
                    for _ in range(int(r.integers(14, 24))):
                        s = int(r.choice(others)); dayT.append(self.trip(t, cur, s, rush=True)); t += int(r.uniform(700, 2000)); cur = s
                        if t > at(21): break
                    dayT.append(self.trip(t, cur, H))
            out += sorted(dayT, key=lambda x: x["t"])
        return out


# ------------------------------------------------------------------------------------------------ physics helpers
def hvac_w(veh, T, sun_wm2, t_since_s):
    """Electrical climate-control power (W): heat-pump / PTC heating below ~17 C, A/C above ~24 C, with a cabin warm-up / cool-down boost after start."""
    T = np.asarray(T, np.float64); t = np.asarray(t_since_s, np.float64); UA = 105.0
    cop = np.clip(0.075 * (T + 15) + 1.25, 1.2, 3.5) if veh["hp"] else np.full_like(T, 1.0)
    heat = np.clip(21 - T, 0, None) * UA + np.where(T < 14, 2200 * np.exp(-t / 650), 0)
    cool = np.clip(T - 23, 0, None) * UA * 0.9 + 0.30 * sun_wm2 + np.where(T > 27, 1500 * np.exp(-t / 500), 0)
    return np.clip(heat / cop + cool / 2.8, 0, 4200)


def dc_curve(soc):    # fraction of peak DC power vs state of charge
    return np.interp(soc, [0, 0.5, 0.8, 0.9, 1.0], [1.0, 1.0, 0.55, 0.25, 0.08])


def temp_derate(tb):  # cold or hot packs accept less power
    return np.interp(tb, [-10, 0, 10, 20, 45, 55], [0.35, 0.55, 0.8, 1.0, 1.0, 0.5])


def batt_heat_kw(p_kw, tb, dc=False):
    a = abs(p_kw); return (0.0004 * a * a + 0.010 * a + (0.0001 * a * a if dc else 0)) * (1.0 + 0.03 * max(0.0, 20.0 - tb))


