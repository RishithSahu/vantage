"""DEV tool - loaders for the REAL datasets behind the data generator (vgpt/datagen.py). Nothing is redistributed: archives are read in place, derived caches go to data/.

  VIF-AEGIS   35 real trips of one car (OBD speed + 24 Hz accelerometer).  -> 1 Hz speed, longitudinal and lateral acceleration (g)  [your Downloads zip]
  EPA         dynamometer schedules UDDS / HWFET / US06 / SC03 / FTP (public domain)  -> city, highway and aggressive speed traces        [data/ref/epa/*.txt]
  Open-Meteo  real hourly weather (temperature, precipitation, cloud, humidity, wind, sunlight, weather code) for any place/date  -> cached JSON  [data/ref/weather/]
Accelerometer axes of the VIF logger are identified per trip (longitudinal = the axis that correlates with d(speed)/dt, sign and bias removed)."""
import csv, collections, io, json, os, pickle, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); REF = os.path.join(DATA, "ref")
VIF_ZIP = os.path.join(os.path.expanduser("~"), "Downloads", "Automotive-ResearchDataSet-VIF-AEGIS.zip")
VIF_CACHE = os.path.join(DATA, "vif_trips.pkl")
EPA_FILES = {"udds": "uddscol", "hwfet": "hwycol", "us06": "us06col", "sc03": "sc03col", "ftp": "ftpcol"}
G = 9.80665


def _t(s): return np.datetime64(s.replace(" ", "T")).astype("datetime64[us]").astype("int64") / 1e6


# ------------------------------------------------------------------------------------------------ VIF-AEGIS (real car, real accelerometer)
def load_vif(zip_path=VIF_ZIP, cache=VIF_CACHE):
    """-> list of {id, v (km/h, 1 Hz), lon_g (+ = accelerating), lat_g (signed), acc_ok}. Cached."""
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    if not os.path.exists(zip_path): return []
    z = zipfile.ZipFile(zip_path)

    def rows(name, trip_col=1):
        out = collections.defaultdict(list)
        with z.open(name) as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="utf-8")); next(r)
            for row in r: out[row[trip_col]].append(row)
        return out
    obd, acc = rows("obdData.csv"), rows("accelerations.csv"); trips = []
    for tid, rs in obd.items():
        s = np.array([(_t(r[4]), float(r[3])) for r in rs if r[2] == "0D"])
        if len(s) < 120: continue
        s = s[np.argsort(s[:, 0])]; t = np.arange(s[0, 0], s[-1, 0], 1.0); v = np.interp(t, s[:, 0], s[:, 1]).astype(np.float32)
        d = {"id": int(tid), "v": v, "lon_g": np.zeros(len(v), np.float32), "lat_g": np.zeros(len(v), np.float32), "acc_ok": False}
        a = acc.get(tid)
        if a and len(a) > 200:
            a = np.array([(_t(r[5]), float(r[2]), float(r[3])) for r in a]); a = a[np.argsort(a[:, 0])]
            ax = np.interp(t, a[:, 0], a[:, 1]); ay = np.interp(t, a[:, 0], a[:, 2])
            # 1 s means of the raw 24 Hz signal (what a telematics box reports), then axis / sign / bias identification against d(speed)/dt
            def one_s(col):
                idx = np.clip(((a[:, 0] - t[0]) // 1).astype(int), 0, len(t) - 1); sm = np.bincount(idx, a[:, col], len(t)); n = np.maximum(np.bincount(idx, None, len(t)), 1); return (sm / n)
            ax1, ay1 = one_s(1), one_s(2); dv = np.gradient(v / 3.6) / G
            c = [np.corrcoef(dv, x)[0, 1] if np.std(x) > 1e-6 and np.std(dv) > 1e-6 else 0.0 for x in (ax1, ay1)]
            k = int(np.argmax(np.abs(c)))
            if abs(c[k]) > 0.45:
                lon = (ax1, ay1)[k]; lat = (ax1, ay1)[1 - k]
                d["lon_g"] = (np.sign(c[k]) * (lon - np.median(lon))).astype(np.float32); d["lat_g"] = (lat - np.median(lat)).astype(np.float32); d["acc_ok"] = True
        trips.append(d)
    os.makedirs(DATA, exist_ok=True); pickle.dump(trips, open(cache, "wb"))
    return trips


# ------------------------------------------------------------------------------------------------ EPA dynamometer schedules
def epa_folder():
    """The folder the EPA schedules must be in (data/ref/epa).  This product is fully offline: it does not fetch them; they were put there before it was installed (the retired downloader is in
    vehiclegpt2_retired_networking_2026-10-21/)."""
    d = os.path.join(REF, "epa")
    missing = [f + ".txt" for f in EPA_FILES.values() if not os.path.exists(os.path.join(d, f + ".txt"))]
    if missing: raise FileNotFoundError(f"EPA schedule files missing in {d}: {', '.join(missing)} (this program does not download anything)")
    return d


def load_epa():
    """-> {name: {v (km/h 1 Hz), lon_g, lat_g}}  (speed from the published mph target trace; acceleration = d(v)/dt)."""
    d = os.path.join(REF, "epa"); out = {}
    for key, f in EPA_FILES.items():
        p = os.path.join(d, f + ".txt")
        if not os.path.exists(p): continue
        pts = []
        for line in open(p, encoding="utf-8", errors="ignore"):
            parts = line.replace('"', "").split()
            if len(parts) >= 2:
                try: pts.append((float(parts[0]), float(parts[1])))
                except ValueError: pass
        a = np.array(pts)
        if len(a) < 100: continue
        v = (a[:, 1] * 1.609344).astype(np.float32)
        out[key] = {"id": key, "v": v, "lon_g": (np.gradient(v / 3.6) / G).astype(np.float32), "lat_g": np.zeros(len(v), np.float32), "acc_ok": True}
    return out


# ------------------------------------------------------------------------------------------------ a small derived fixture (tests only)
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "driving_segments.npz")
LAST_SOURCE = {"source": "real"}


def fixture_segments(path=None):
    """The driving fixture of the tests: 3-minute windows of speed and acceleration taken from the 1 Hz tail of the generated sample cars (tools_testdata.py make-driving), small enough to ship.  -> pools like `segments()`, or None.
    It stands in for the real traces (VIF-AEGIS, EPA schedules) only where the person asks for it (VGPT_DRIVING_FIXTURE=1), and a world made from it says so (`gen.driving_source = fixture`)."""
    path = path or FIXTURE
    if not os.path.exists(path): return None
    z = np.load(path); out = {"city": [], "hwy": [], "fast": [], "sport": []}
    for k in out:
        if k + "_v" not in z: continue
        for v, lo, la in zip(z[k + "_v"], z[k + "_lon"], z[k + "_lat"]): out[k].append({"v": v.copy(), "lon_g": lo.copy(), "lat_g": la.copy(), "src": "fixture"})
    return out


# ------------------------------------------------------------------------------------------------ segments for the generator
def segments(window=180, stride=120):
    """Cut every real trace into 3-minute windows. -> {"city": [...], "hwy": [...], "fast": [...], "sport": [...]} of dicts {v, lon_g, lat_g, src}.
    city: 10-60 km/h mean, hwy: >= 85, fast: >= 108 (motorway at speed), sport: US06-like (hard acceleration)."""
    out = {"city": [], "hwy": [], "fast": [], "sport": []}
    srcs = [("vif", t) for t in load_vif()] + [(k, e) for k, e in load_epa().items()]
    for name, tr in srcs:
        v, lo, la = tr["v"], tr["lon_g"], tr["lat_g"]
        for i in range(0, len(v) - window, stride):
            w = v[i:i + window]; m = float(w.mean()); seg = {"v": w.copy(), "lon_g": lo[i:i + window].copy(), "lat_g": la[i:i + window].copy(), "src": str(name)}
            if name == "us06": out["sport"].append(seg)
            if m >= 108: out["fast"].append(seg)
            if m >= 85: out["hwy"].append(seg)
            elif 10 <= m <= 62: out["city"].append(seg)
    LAST_SOURCE["source"] = "real"
    if not (out["city"] and out["hwy"]) and os.environ.get("VGPT_DRIVING_FIXTURE") == "1":
        fx = fixture_segments()
        if fx and fx["city"] and fx["hwy"]: LAST_SOURCE["source"] = "fixture"; return fx
    return out


# ------------------------------------------------------------------------------------------------ Open-Meteo (real weather)
HOURLY = "temperature_2m,precipitation,cloud_cover,relative_humidity_2m,wind_speed_10m,weather_code,shortwave_radiation,is_day,surface_pressure"


def weather(lat, lon, start, end, cache_dir=os.path.join(REF, "weather")):
    """Hourly real weather (UTC) for start..end (YYYY-MM-DD, inclusive), from the local cache only. -> dict of numpy arrays: t (epoch s), temp, precip, cloud, rh, wind, code, sun (W/m2), is_day, pressure."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{lat:.2f}_{lon:.2f}_{start}_{end}.json")
    if not os.path.exists(p): raise FileNotFoundError(f"weather for {lat:.2f},{lon:.2f} {start}..{end} is not in the local cache {p} (this program does not download anything)")
    h = json.load(open(p))["hourly"]
    t = np.array([int(np.datetime64(s).astype("datetime64[s]").astype("int64")) for s in h["time"]], np.int64)
    f = lambda k, d=0.0: np.array([d if x is None else x for x in h[k]], np.float32)
    return {"t": t, "temp": f("temperature_2m", 15), "precip": f("precipitation"), "cloud": f("cloud_cover"), "rh": f("relative_humidity_2m", 60), "wind": f("wind_speed_10m"),
            "code": f("weather_code").astype(np.int16), "sun": f("shortwave_radiation"), "is_day": f("is_day"), "pressure": f("surface_pressure", 1013)}


if __name__ == "__main__":
    print("EPA:", list(epa_folder() and load_epa()))
    vif = load_vif(); print("VIF trips:", len(vif), "| with accelerometer:", sum(t["acc_ok"] for t in vif), "| hours:", round(sum(len(t["v"]) for t in vif) / 3600, 1))
    seg = segments(); print({k: len(v) for k, v in seg.items()})
    # harsh-event rates on the REAL VIF accelerometer (events/100 km) at candidate thresholds
    km = sum(float(t["v"].sum()) / 3600 for t in vif if t["acc_ok"])
    for thr in (0.20, 0.25, 0.30):
        nb = na = nc = 0
        for t in vif:
            if not t["acc_ok"]: continue
            b = (t["lon_g"] < -thr).astype(int); a = (t["lon_g"] > thr).astype(int); c = (np.abs(t["lat_g"]) > thr).astype(int)
            nb += int((np.diff(np.r_[0, b]) == 1).sum()); na += int((np.diff(np.r_[0, a]) == 1).sum()); nc += int((np.diff(np.r_[0, c]) == 1).sum())
        print(f"  threshold {thr:.2f} g: brake {100 * nb / km:.1f}  accel {100 * na / km:.1f}  corner {100 * nc / km:.1f}  per 100 km (over {km:.0f} km real driving)")
