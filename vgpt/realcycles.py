"""DEV tool: real recorded driving cycles from the VIF-AEGIS automotive research dataset (35 real OBD-II trips, one car, Jan 2017).
Reads the zip IN PLACE (never copied/redistributed), takes the OBD PID 0x0D (vehicle speed, km/h) per trip, resamples to 1 Hz and cuts it into
city-like and highway-like segments. The EV simulation then uses these REAL speed traces instead of synthetic ones -> the driving scenarios are
recreated from recorded data (the case study allows 'simulated/recorded data'). Derived segments are cached to data/real_cycles.pkl (git-ignored)."""
import csv, io, os, pickle, zipfile
import numpy as np

ZIP = os.path.join(os.path.expanduser("~"), "Downloads", "Automotive-ResearchDataSet-VIF-AEGIS.zip")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_cycles.pkl")


def _t(s):  # "2017-01-19 16:19:03.045593" -> epoch seconds (naive)
    return np.datetime64(s.replace(" ", "T")).astype("datetime64[us]").astype("int64") / 1e6


def load(zip_path=ZIP, cache=CACHE):
    if os.path.exists(cache):
        with open(cache, "rb") as f: return pickle.load(f)
    if not os.path.exists(zip_path): return None
    z = zipfile.ZipFile(zip_path); per = {}
    for r in csv.DictReader(io.TextIOWrapper(z.open("obdData.csv"), encoding="utf-8")):
        if r["obdPid"] == "0D":
            try: per.setdefault(r["trip_id"], []).append((_t(r["timestamp"]), float(r["data"])))
            except ValueError: pass
    city, hwy, n_trips = [], [], 0
    for tid, rows in per.items():
        a = np.array(rows); a = a[np.argsort(a[:, 0])]
        if len(a) < 300: continue
        n_trips += 1; t = a[:, 0] - a[0, 0]; v = np.interp(np.arange(0, t[-1], 1.0), t, a[:, 1])
        for i in range(0, len(v) - 180, 180):                                    # 3-minute windows classified by their own mean speed
            w = v[i:i + 180]
            if w.mean() >= 85: hwy.append(w.astype(np.float32))
            elif 12 <= w.mean() <= 62: city.append(w.astype(np.float32))
    out = {"city": city, "hwy": hwy, "trips": n_trips}
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(out, open(cache, "wb"))
    return out


def block(rng, segs, n):
    """A speed trace of n seconds assembled from random real 3-minute segments."""
    out = []
    while sum(len(x) for x in out) < n: out.append(segs[int(rng.integers(len(segs)))])
    return np.concatenate(out)[:n].astype(np.float64)


if __name__ == "__main__":
    d = load(); print("real trips:", d["trips"], "| city segments:", len(d["city"]), "| highway segments:", len(d["hwy"]))
    print("city mean speed %.1f, highway mean speed %.1f km/h" % (np.mean([s.mean() for s in d["city"]]), np.mean([s.mean() for s in d["hwy"]])))
