"""Evidence engine for the analytical questions (stdlib only): time parsing, access to every tier of the store, derived signals, statistics, event helpers,
anomaly definition and the efficiency / overheating hypothesis models. Everything returns plain numbers + the timestamps they came from, so answers can cite and
recompute them. Nothing here talks to the language model."""
import math, operator, re, statistics, datetime as dt
import numpy as np
from . import analytics as A, analytics_x as AX, fastcols as FC

# ------------------------------------------------------------------------------------------------ metrics
SIGS = {"speed": ("speed", 10.0, "km/h", "vehicle speed"), "soc": ("soc", 10.0, "%", "state of charge"), "power": ("power", 1000.0, "kW", "battery power"), "batt": ("batt", 10.0, "C", "battery temperature"),
        "flow": ("flow", 10.0, "L/min", "coolant flow"), "amb": ("amb", 10.0, "C", "outside temperature"), "hvac": ("hvac", 1.0, "W", "climate power"), "motor": ("motor", 10.0, "C", "motor temperature")}
DERIVED = {"rpm": ("rpm", "estimated motor speed"), "accel": ("m/s2", "acceleration")}
MISSING = {"cabin": "cabin temperature", "throttle": "accelerator-pedal position (throttle)", "steering": "steering angle", "radar": "radar objects", "engine_load": "engine load", "gear": "gear position",
           "rear_camera": "rear or side camera images", "fuel": "fuel level"}
METRIC_RX = [("rpm", r"\brpm\b|revs\b|revolutions|motor speed|engine speed"), ("batt", r"battery temp\w*|pack temp\w*|battery heat|\bpack\b.{0,30}\b(hot\w*|warm\w*|temp\w*|heat\w*)"), ("motor", r"motor temp\w*|engine temp\w*|engine heat|(motor|engine)\b.{0,30}\b(hot\w*|warm\w*|temp\w*|heat\w*|cool\w*)"),
             ("soc", r"state of charge|\bsoc\b|battery (level|percentage|charge)|charge level"), ("flow", r"coolant flow|cooling flow"), ("amb", r"outside temp\w*|ambient|external temp\w*"),
             ("tyre", r"tire pressure|tyre pressure|\btires?\b pressure"), ("cabin", r"cabin temp\w*|interior temp\w*|inside temp\w*"), ("throttle", r"throttle|accelerator|pedal"),
             ("steering", r"steering"), ("radar", r"radar"), ("engine_load", r"engine load"), ("hvac", r"climate power|hvac|heating power"), ("power", r"\bpower\b|motor load|\bload\b"), ("speed", r"\bspeed\b|\bvelocity\b")]
GEAR_RATIO, WHEEL_R = 9.0, 0.33


def metric_of(t):
    for k, rx in METRIC_RX:
        if re.search(rx, t): return k
    return None


def label(m): return SIGS.get(m, DERIVED.get(m, ("", "", "", m)))[-1] if m in SIGS else {"rpm": "estimated motor speed", "tyre": "tyre pressure", "accel": "acceleration"}.get(m, MISSING.get(m, m))


def unit(m): return {"rpm": "rpm", "tyre": "kPa", "accel": "m/s2"}.get(m) or SIGS.get(m, ("", 1, ""))[2]


def rpm_of(kmh, ratio=GEAR_RATIO, r=WHEEL_R): return kmh / 3.6 / (2 * math.pi * r) * 60.0 * ratio


def kmh_of_rpm(rpm, ratio=GEAR_RATIO, r=WHEEL_R): return rpm * (2 * math.pi * r) / 60.0 / ratio * 3.6


# ------------------------------------------------------------------------------------------------ time parsing
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60}
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})


def number(s):
    s = s.lower().strip()
    return float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else NUMW.get(s)


def find_clock(t):
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b", t)
    if m:
        h, mi, se = int(m[1]), int(m[2]), int(m[3] or 0)
        if m[4] == "pm" and h < 12: h += 12
        if m[4] == "am" and h == 12: h = 0
        return h * 3600 + mi * 60 + se if h < 24 and mi < 60 else None
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
    if m: h = int(m[1]) % 12 + (12 if m[2] == "pm" else 0); return h * 3600
    return None


def find_date(t, now, tz):
    """-> (date, explicit, text) ; relative words resolve against the vehicle's local 'today'."""
    today = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc).date()
    if re.search(r"day before yesterday", t): return today - dt.timedelta(days=2), False, "the day before yesterday"
    if re.search(r"\byesterday\b|last night", t): return today - dt.timedelta(days=1), False, "yesterday"
    if re.search(r"\btoday\b|this morning|this afternoon|tonight", t): return today, False, "today"
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try: return dt.date(int(m[1]), int(m[2]), int(m[3])), True, m[0]
        except ValueError: return None, True, "INVALID " + m[0]
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?(?: of)? (" + mon + r")\b(?:,? (\d{4}))?", t) or None
    if m: d, mo, y = int(m[1]), MONTHS[m[2]], m[3]
    else:
        m = re.search(r"\b(" + mon + r") (\d{1,2})(?:st|nd|rd|th)?\b(?:,? (\d{4}))?", t)
        if not m: return None, False, ""
        mo, d, y = MONTHS[m[1]], int(m[2]), m[3]
    yr = int(y) if y else today.year
    try:
        date = dt.date(yr, mo, d)
        if not y and date > today: date = dt.date(yr - 1, mo, d)
        return date, True, m[0]
    except ValueError: return None, True, "INVALID " + m[0]


def local_midnight(date, tz): return int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp()) - tz


def day_bounds(date, tz): lo = local_midnight(date, tz); return lo, lo + 86400


def ts_s(ts): return dt.datetime.fromtimestamp(ts + A.TZ, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ts_m(ts): return A.stamp(ts)


def dur(sec):
    sec = int(round(sec))
    if sec < 90: return f"{sec} s"
    if sec < 5400: return f"{sec / 60:.0f} min" if sec % 60 == 0 or sec > 600 else f"{sec // 60} min {sec % 60} s"
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min"


def find_duration(t, default=None):
    m = re.search(r"\b(\d+|one|two|three|four|five|ten|fifteen|twenty|thirty|sixty)\s*(second|sec|s|minute|min|hour|hr)s?\b", t)
    if not m: return default
    n = number(m[1]); u = m[2]; return n * (1 if u.startswith("s") else 60 if u.startswith("m") else 3600)


# ------------------------------------------------------------------------------------------------ data access
def meta(c, k, d=None):
    try: r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    except Exception: return d
    return r[0] if r else d


def minmax_ts(c, table):
    """(first, last) timestamp of a table keyed by ts.  Two statements, never `SELECT MIN(ts), MAX(ts)`: SQLite answers a single MIN or MAX from the end of the index, but both together scan the whole table (0.6 s for the two million minute rows of a four-year history on the Pi, at every question)."""
    lo = c.execute(f"SELECT MIN(ts) FROM {table}").fetchone()[0]
    return (lo, c.execute(f"SELECT MAX(ts) FROM {table}").fetchone()[0]) if lo is not None else (None, None)


def coverage(c, now):
    """Span of the history that exists (first/last second, and the 1 Hz tier's start)."""
    a = minmax_ts(c, "sig_1m"); b = minmax_ts(c, "sig_1s")
    lo = min(x for x in (a[0], b[0]) if x is not None) if (a[0] is not None or b[0] is not None) else None
    hi = max(x for x in (a[1], b[1]) if x is not None) if lo is not None else None
    return {"lo": lo, "hi": min(hi, now) if hi else None, "hz_lo": b[0], "hz_hi": b[1]}


class Sig:
    """Signals over a window as parallel lists in engineering units; res = seconds per sample (1 = 1 Hz tier, 60 = per-minute tier)."""
    def __init__(self, ts, cols, res, tier): self.ts, self.cols, self.res, self.tier = ts, cols, res, tier
    def __len__(self): return len(self.ts)
    def col(self, m):
        if m == "rpm": return [rpm_of(v) for v in self.cols["speed"]]
        if m == "accel": return self.accel()
        return self.cols[m]
    def accel(self):
        if getattr(self, "_acc", None) is not None: return self._acc
        s = self.cols["speed"]; out = [float("nan")] * len(s)
        for i in range(1, len(s)):
            dt_ = self.ts[i] - self.ts[i - 1]
            if 0 < dt_ <= 3 * self.res and math.isfinite(s[i]) and math.isfinite(s[i - 1]): out[i] = (s[i] - s[i - 1]) / 3.6 / dt_
        self._acc = out
        return out


def load(c, lo, hi, prefer="auto", cols=("speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"), edge=False):
    """1 Hz tier when the window lies inside it (last 7 days), else the per-minute tier.  `edge`: the window may run past the newest row (a question about right now on a live vehicle): the 1 Hz tier still serves it."""
    hz_lo, hz_hi = minmax_ts(c, "sig_1s")
    if prefer == "1s" and (hz_lo is None or hz_lo > lo) and hi - lo <= 45 * 86400:
        from . import archive
        if archive.hydrate(c, lo, hi)["rows"]: hz_lo, hz_hi = minmax_ts(c, "sig_1s")
    hz_covers = hz_lo is not None and hz_lo <= lo and hz_hi is not None and (hz_hi >= hi - 1 or (edge and hz_hi >= lo + 25))
    use_hz = hz_covers and prefer in ("1s", "auto")
    if use_hz:
        rows = c.execute("SELECT ts, speed, soc, power, batt, flow, amb, hvac, motor, mode FROM sig_1s WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall(); res, tier = 1, "1 Hz"
        sc = (10.0, 10.0, 1000.0, 10.0, 10.0, 10.0, 1.0, 10.0, 1.0)
    else:
        res, tier = 60, "per-minute"; sc = (10.0, 10.0, 1000.0, 10.0, 10.0, 10.0, 1.0, 10.0, 1.0)
        names = ("speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"); fast = FC.minute_columns(c, names, lo, hi)
        if fast is not None: return Sig(fast[0].astype(np.int64).tolist(), {n: (fast[i + 1] / sc[i]).tolist() for i, n in enumerate(names)}, res, tier)        # NULL is NaN in the copy, and NaN / scale is NaN
        rows = c.execute("SELECT ts, speed, soc, power, batt, flow, amb, hvac, motor, mode FROM sig_1m WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall()
    names = ("speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode")
    # SQL NULL is a missing observation, not zero.  Keep the parallel numeric
    # arrays usable by analytics while allowing the snapshot boundary to turn
    # the non-finite marker back into an explicit unavailable value.
    ts = [r[0] for r in rows]
    data = {n: [(float("nan") if r[i + 1] is None else r[i + 1] / sc[i]) for r in rows] for i, n in enumerate(names)}
    return Sig(ts, data, res, tier)


def nearest(sig, ts, tol):
    """Index of the sample closest to ts within tol seconds, else None."""
    import bisect
    i = bisect.bisect_left(sig.ts, ts); best = None
    for j in (i - 1, i):
        if 0 <= j < len(sig.ts) and abs(sig.ts[j] - ts) <= tol and (best is None or abs(sig.ts[j] - ts) < abs(sig.ts[best] - ts)): best = j
    return best


def state_at(c, ts, now=None):
    """Everything the car recorded at (about) `ts`: signals, derived values, position, camera, charging, tyres. `gap` = no sample (car switched off)."""
    out = {"ts": ts}
    sg = load(c, ts - 30, ts + 31, edge=True); i = nearest(sg, ts, 5) if sg.res == 1 else None
    if i is None:
        sg = load(c, ts - 120, ts + 121, prefer="1m"); i = nearest(sg, ts, 60)
    if i is None:
        out["gap"] = True
        a = c.execute("SELECT MAX(ts) FROM sig_1m WHERE ts<?", (ts,)).fetchone()[0]; b = c.execute("SELECT MIN(ts) FROM sig_1m WHERE ts>?", (ts,)).fetchone()[0]; out["prev"], out["next"] = a, b
    else:
        for m in SIGS: out[m] = sg.cols[m][i]
        raw_mode = sg.cols["mode"][i]
        out["mode"] = int(raw_mode) if raw_mode is not None and math.isfinite(raw_mode) else None
        out["rpm"] = rpm_of(out["speed"]) if out.get("speed") is not None and math.isfinite(out["speed"]) else None
        out["sample_ts"] = sg.ts[i]; out["tier"] = sg.tier
        if sg.res == 1 and i > 0 and sg.ts[i] - sg.ts[i - 1] <= 3 and all(x is not None and math.isfinite(x) for x in (sg.cols["speed"][i], sg.cols["speed"][i - 1])): out["accel"] = (sg.cols["speed"][i] - sg.cols["speed"][i - 1]) / 3.6 / (sg.ts[i] - sg.ts[i - 1])
    try:
        g = c.execute("SELECT ts, lat, lon, hdg, spd FROM gps_1s WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (ts, ts - 180)).fetchone()
        if g: out["gps"] = {"lat": g[1] / 1e6, "lon": g[2] / 1e6, "hdg": g[3] / 10.0 if g[3] is not None else None, "spd": g[4] / 10.0 if g[4] is not None else None, "ts": g[0]}          # a recording may carry no heading or speed with its fixes
        else:
            g = c.execute("SELECT ts, lat, lon FROM ext_1m WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (ts, ts - 300)).fetchone()
            if g and g[1] is not None and g[2] is not None:
                lat, lon = g[1] / 1e5, g[2] / 1e5
                if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                    out["gps"] = {"lat": lat, "lon": lon, "hdg": None, "spd": None, "ts": g[0]}
        t = c.execute("SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr, tt, v12, rain, lux FROM ext_1m WHERE ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (ts, ts - 300)).fetchone()
        if t:
            raw_tyres = [x / 10.0 if x is not None else None for x in t[1:5]]
            out["ext"] = {"ts": t[0], "tyres": [AX._cold(x, t[5]) if x is not None and t[5] is not None else None for x in raw_tyres], "raw_tyres": raw_tyres, "v12": t[6] / 100.0 if t[6] is not None else None, "rain": t[7], "lux": t[8]}
    except Exception: pass
    try:
        cam = c.execute("SELECT ts, type FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED','CAMERA_RESTORED') AND ts<=? ORDER BY ts DESC LIMIT 1", (ts,)).fetchone()
        out["camera"] = ("blocked" if cam[1] == "CAMERA_BLOCKED" else "degraded" if cam[1] == "CAMERA_DEGRADED" else "clear") if cam else "clear"
        ch = c.execute("SELECT start, end, kind, kw FROM charges WHERE start<=? AND end>=? LIMIT 1", (ts, ts)).fetchone()
        if ch: out["charging"] = {"kind": ch[2], "kw": ch[3], "start": ch[0]}
    except Exception: pass
    return out


def fmt_state(s, brief=False):
    if "speed" not in s: return "no signal sample was recorded at that moment (the car was switched off)"
    mode = {0: "parked", 1: "driving (city)", 2: "driving (motorway)", 3: "charging"}.get(s.get("mode"), "drive mode unavailable")
    def value(name, text, digits=1):
        v = s.get(name)
        return text.format(v) if v is not None and math.isfinite(v) else None
    bits = [mode, value("speed", "speed {:.0f} km/h"), value("rpm", "motor speed about {:.0f} rpm (estimated)"), value("batt", "battery {:.1f} C"), value("soc", "state of charge {:.0f}%"), value("power", "power {:.1f} kW"), value("flow", "coolant flow {:.1f} L/min"), value("motor", "motor {:.0f} C"), value("amb", "outside {:.1f} C")]
    if "accel" in s: bits.insert(2, f"acceleration {s['accel']:+.1f} m/s2")
    if not brief:
        if s.get("gps"): bits.append(f"GPS {s['gps']['lat']:.4f}, {s['gps']['lon']:.4f}")
        bits.append(f"camera {s['camera']}")
        if s.get("charging"): bits.append(f"{s['charging']['kind']} charging at {s['charging']['kw']:.0f} kW")
        if s.get("ext"):
            tyres = [x for x in s["ext"]["tyres"] if x is not None and math.isfinite(x)]
            if tyres: bits.append("tyres " + "/".join(f"{x:.0f}" for x in tyres) + " kPa (20 C equivalent)")
    return ", ".join(b for b in bits if b)


# ------------------------------------------------------------------------------------------------ statistics
def mean(x): return sum(x) / len(x) if x else float("nan")


def sd(x):
    if len(x) < 2: return 0.0
    m = mean(x); d = [v - m for v in x]; return (sum(map(operator.mul, d, d)) / (len(x) - 1)) ** 0.5            # the same sum of the same squares in the same order, in C (23 calls over 20 000 values took 0.34 s on the Pi)


def pearson(x, y):
    n = len(x)
    if n < 3: return None
    mx, my = mean(x), mean(y); sx = sum((a - mx) ** 2 for a in x); sy = sum((b - my) ** 2 for b in y)
    if sx <= 0 or sy <= 0: return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def ols_line(x, y):
    n = len(x)
    if n < 3: return None
    mx, my = mean(x), mean(y); sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0: return None
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx; a0 = my - b * mx; ss_tot = sum((c - my) ** 2 for c in y)
    ss_res = sum((c - (a0 + b * a)) ** 2 for a, c in zip(x, y)); return {"slope": b, "intercept": a0, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0, "n": n}


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
    for m in range(1, 200):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30; de = d * c; h *= de
        if abs(de - 1) < 3e-7: break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def welch(a, b):
    """Welch's t-test -> (t, df, two-sided p) or None."""
    if len(a) < 2 or len(b) < 2: return None
    va, vb = sd(a) ** 2 / len(a), sd(b) ** 2 / len(b)
    if va + vb <= 0: return None
    t = (mean(a) - mean(b)) / math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); p = betainc(df / 2, 0.5, df / (df + t * t))
    return t, df, p


def solve(Amat, bvec):
    n = len(bvec); M = [row[:] + [bvec[i]] for i, row in enumerate(Amat)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12: return None
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for k in range(i, n + 1): M[r][k] -= f * M[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1): x[i] = (M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return x


def ols_multi(X, y, ridge=1e-6):
    """Multiple linear regression with intercept (normal equations) -> (coefs [b0, b1..], r2) or None."""
    n, p = len(X), len(X[0]) if X else 0
    if n < p + 3: return None
    k = p + 1; Z = np.ones((n, k)); Z[:, 1:] = np.asarray(X, dtype=float).reshape(n, p); yy = np.asarray(y, dtype=float)       # the normal equations as array products (the same sums, 2900 trips x 7 terms took 0.4 s as Python loops on the Pi)
    A_ = Z.T @ Z
    for a in range(1, k): A_[a, a] += ridge
    beta = solve(A_.tolist(), (Z.T @ yy).tolist())
    if beta is None: return None
    my = float(yy.mean()); ss_tot = float(((yy - my) ** 2).sum()); ss_res = float(((yy - Z @ np.asarray(beta)) ** 2).sum())
    return beta, (1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


# ------------------------------------------------------------------------------------------------ events
KINDS = {"overheat": ("WARN_BATT_OVERTEMP",), "diagnostic": ("DTC_SET",), "adas": ("ADAS_DISENGAGE",), "brake": ("HARSH_BRAKE",), "camera": ("CAMERA_BLOCKED", "CAMERA_DEGRADED"), "charging": ("CHARGE_START",),
         "tyre": ("TPMS_LOW",), "accel": ("HARSH_ACCEL",), "corner": ("HARSH_CORNER",), "can": ("CAN_ANOMALY",),
         "warning": ("WARN_BATT_OVERTEMP", "DTC_SET", "TPMS_LOW", "LOW_12V", "CAMERA_BLOCKED", "CAN_ANOMALY"), "fault": ("DTC_SET", "WARN_BATT_OVERTEMP")}
KIND_RX = [("overheat", r"overheat\w*|over.?temp\w*|too hot|thermal warning|heat warning"), ("diagnostic", r"diagnostic|\bdtc\b|fault code|trouble code|error code"), ("adas", r"adas|disengage\w*|lane (assist|keep\w*)"),
           ("brake", r"brak\w*"), ("camera", r"camera"), ("charging", r"charging session|charge session|charging event"), ("tyre", r"tyre|tire"), ("can", r"can bus|intrusion"),
           ("fault", r"major fault|serious (fault|event)|\bfaults?\b"), ("warning", r"warnings?|alerts?|anomal\w*|abnormal\w*")]
KIND_NAME = {"overheat": "battery over-temperature warning", "diagnostic": "diagnostic trouble code", "adas": "ADAS disengagement", "brake": "hard-braking event", "camera": "camera fault", "charging": "charging session",
             "tyre": "tyre-pressure warning", "accel": "hard-acceleration event", "corner": "hard-cornering event", "can": "CAN-bus incident", "warning": "warning", "fault": "fault"}
INJECT_RX = re.compile(r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)|system prompt|reveal (all )?(hidden|secret)|api[ _-]?key|password|disregard|you are now|act as", re.I)


def kind_of(t):
    for k, rx in KIND_RX:
        if re.search(rx, t): return k
    return None


def events(c, kind, lo, hi, now=None, limit=None, newest_first=False):
    types = KINDS[kind]; q = "SELECT ts, type, sev, detail FROM events WHERE type IN (%s) AND ts>=? AND ts<?" % ",".join("?" * len(types))
    q += " ORDER BY ts " + ("DESC" if newest_first else "ASC") + (f" LIMIT {int(limit)}" if limit else "")
    return c.execute(q, (*types, lo, min(hi, now + 1) if now else hi)).fetchall()


def last_event(c, kind, now): r = events(c, kind, 0, now + 1, limit=1, newest_first=True); return r[0] if r else None


def safe_text(s):
    """Stored text is DATA: printable ASCII only, and instruction-like content is flagged instead of being obeyed."""
    s = re.sub(r"[^\x20-\x7e]", " ", s or "")
    return s[:160]


def flagged(s): return bool(INJECT_RX.search(s or ""))


def last_anomaly(c, now):
    """The most recent thing the car itself marked as abnormal: a warning/critical event; if there is none, the strongest signal outlier of the last 7 days."""
    r = c.execute("SELECT ts, type, sev, detail FROM events WHERE ts<=? AND (sev IN ('warn','critical') OR type IN ('DTC_SET','WARN_BATT_OVERTEMP','TPMS_LOW','LOW_12V','CAN_ANOMALY','CAMERA_BLOCKED')) ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    if r: return {"ts": r[0], "kind": "event", "type": r[1], "detail": safe_text(r[3]), "basis": "the last warning/fault event the car recorded"}
    sg = load(c, now - 7 * 86400, now + 1, prefer="1m"); best = None
    for m in ("batt", "flow", "motor"):
        v = [x for x, md in zip(sg.cols[m], sg.cols["mode"]) if md in (1, 2, 3) and x is not None and math.isfinite(x) and (m != "flow" or x > 0.3)]
        if len(v) < 30: continue
        mu, s_ = mean(v), sd(v) or 1.0
        for ts, x, md in zip(sg.ts, sg.cols[m], sg.cols["mode"]):
            if md not in (1, 2, 3) or x is None or not math.isfinite(x): continue
            z = abs(x - mu) / s_
            if best is None or z > best["z"]: best = {"ts": ts, "kind": "signal", "type": m, "z": z, "detail": f"{label(m)} {x:.1f} {unit(m)} (mean {mu:.1f}, sd {s_:.1f})", "basis": "the strongest statistical outlier in the last 7 days (no warning events exist)"}
    return best


# ------------------------------------------------------------------------------------------------ trips + efficiency attribution
def trips_between(c, lo, hi):
    try:
        rows = c.execute("SELECT t.start, t.end, t.km, t.kwh, t.wh_km, t.avg_speed, t.hwy_share, t.avg_amb, t.avg_hvac_w, COALESCE(x.harsh_brake,0)+COALESCE(x.harsh_accel,0), COALESCE(x.rain_share,0), COALESCE(x.regen_kwh,0), x.from_id, x.to_id "
                         "FROM trips t LEFT JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<? AND t.km>1.5 ORDER BY t.start", (lo, hi)).fetchall()
    except Exception:                                                                 # legacy database without the extension tables
        rows = c.execute("SELECT t.start, t.end, t.km, t.kwh, t.wh_km, t.avg_speed, t.hwy_share, t.avg_amb, t.avg_hvac_w, 0, 0, 0, NULL, NULL FROM trips t WHERE t.start>=? AND t.start<? AND t.km>1.5 ORDER BY t.start", (lo, hi)).fetchall()
    out = []
    for r in rows:
        # Efficiency attribution is an energy-per-distance analysis.  Keeping
        # a trip with missing energy, distance, or its requested numeric
        # covariates would turn NULL into zero or make a ranking crash.  It is
        # safer to omit that incomplete trip from this *analysis* than to
        # silently score it as the most efficient one.
        required = r[:9]
        try:
            if any(v is None or not math.isfinite(float(v)) for v in required):
                continue
            if float(r[2]) <= 0 or float(r[3]) < 0 or float(r[4]) < 0:
                continue
        except (TypeError, ValueError):
            continue
        out.append(dict(start=r[0], end=r[1], km=r[2], kwh=r[3], wh_km=r[4], speed=r[5], hwy=r[6], amb=r[7], hvac=r[8], harsh=r[9] * 100.0 / max(r[2], 1.0), rain=r[10], regen=r[11], a=r[12], b=r[13]))
    return out


def tyre_deficit(c, ts):
    """Mean cold-equivalent shortfall (% of spec) across the four tyres at the last reading before ts; 0 when the vehicle has no TPMS data."""
    try:
        sf, sr = AX._mf(c, "veh.tyre_spec_front"), AX._mf(c, "veh.tyre_spec_rear")
        if sf is None: return 0.0
        r = c.execute("SELECT tp_fl, tp_fr, tp_rl, tp_rr, tt FROM ext_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (ts,)).fetchone()
        if not r: return 0.0
        spec = [sf, sf, sr, sr]; return mean([max(0.0, 100 * (1 - AX._cold(r[i] / 10.0, r[4]) / spec[i])) for i in range(4)])
    except Exception: return 0.0


def attach_tyre(c, trips):
    """trip["tyre"] = tyre_deficit(c, trip["start"]) for every trip.  Many trips are answered from one read of the tyre-pressure minutes (each trip takes the last reading at or before its start, as tyre_deficit does), not one query per trip (a health summary has thousands)."""
    if len(trips) < 4:
        for t in trips: t["tyre"] = tyre_deficit(c, t["start"])
        return
    try:
        sf, sr = AX._mf(c, "veh.tyre_spec_front"), AX._mf(c, "veh.tyre_spec_rear")
        if sf is None:
            for t in trips: t["tyre"] = 0.0
            return
        last = max(int(t["start"]) for t in trips) + 1
        from . import colstore
        cs = colstore.of(c); arr = cs.read(c, "ext_1m", ["tp_fl", "tp_fr", "tp_rl", "tp_rr", "tt"], 0, last) if cs is not None else None
        if arr is None: arr = np.array(c.execute("SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr, tt FROM ext_1m WHERE ts<? ORDER BY ts", (last,)).fetchall(), dtype=float).reshape(-1, 6)
    except Exception:
        for t in trips: t["tyre"] = tyre_deficit(c, t["start"])
        return
    ts = arr[:, 0]; spec = [sf, sf, sr, sr]
    for t in trips:
        i = int(np.searchsorted(ts, t["start"], side="right")) - 1
        if i < 0 or np.isnan(arr[i, 1:5]).any(): t["tyre"] = 0.0; continue
        tt = None if np.isnan(arr[i, 5]) else float(arr[i, 5])
        try: t["tyre"] = mean([max(0.0, 100 * (1 - AX._cold(float(arr[i, 1 + k]) / 10.0, tt) / spec[k])) for k in range(4)])
        except Exception: t["tyre"] = 0.0


FEATURES = [("heating", "cold weather (heating load)", lambda t: max(0.0, 18 - t["amb"]), "degrees below 18 C"), ("cooling", "hot weather (air conditioning)", lambda t: max(0.0, t["amb"] - 24), "degrees above 24 C"),
            ("speed", "higher average speed", lambda t: t["speed"], "km/h"), ("hwy", "more motorway driving", lambda t: 100 * t["hwy"], "% of distance"), ("rain", "rain on the road", lambda t: 100 * t["rain"], "% of the trip in rain"),
            ("harsh", "harder driving", lambda t: t["harsh"], "hard events per 100 km"), ("tyre", "low tyre pressure", lambda t: t["tyre"], "% below spec")]


PHYS = ("heating", "cooling", "rain", "harsh", "tyre", "hwy")


def eff_fit(c, lo, hi):
    """Trip-level multiple regression of Wh/km on the conditions in FEATURES (standardised, lightly ridge-regularised). -> dict(T, feats, beta, mu, sg, r2) or None."""
    T = trips_between(c, lo, hi)
    attach_tyre(c, T)
    if len(T) < 12: return None
    feats = [f for f in FEATURES if sd([f[2](t) for t in T]) > 1e-6]
    if not feats: return None
    X = [[f[2](t) for f in feats] for t in T]; y = [t["wh_km"] for t in T]
    mu = [mean([r[i] for r in X]) for i in range(len(feats))]; sg_ = [sd([r[i] for r in X]) or 1.0 for i in range(len(feats))]
    fit = ols_multi([[(r[i] - mu[i]) / sg_[i] for i in range(len(feats))] for r in X], y, ridge=0.5)
    if fit is None: return None
    beta, r2 = fit
    bb = [beta[i + 1] / sg_[i] for i in range(len(feats))]
    bb = [0.0 if (f[0] in PHYS and b < 0) else b for f, b in zip(feats, bb)]                      # a physically impossible sign (e.g. low tyre pressure saving energy) is statistical noise -> no effect
    return {"T": T, "feats": feats, "beta": bb, "b0": beta[0], "mu": mu, "sg": sg_, "r2": r2}


def eff_period(fit, bad, good):
    """Contribution of each condition to the Wh/km gap between two sets of trips (coefficient x difference in the condition)."""
    if not fit or len(bad) < 2 or len(good) < 2: return None
    out = []
    for i, f in enumerate(fit["feats"]):
        xb, xg = mean([f[2](t) for t in bad]), mean([f[2](t) for t in good]); out.append({"key": f[0], "name": f[1], "unit": f[3], "bad": xb, "good": xg, "dx": xb - xg, "per_unit": fit["beta"][i], "contrib": fit["beta"][i] * (xb - xg)})
    out.sort(key=lambda r: -abs(r["contrib"]))
    gap = mean([t["wh_km"] for t in bad]) - mean([t["wh_km"] for t in good])
    return {"factors": out, "gap": gap, "explained": sum(r["contrib"] for r in out), "bad_mean": mean([t["wh_km"] for t in bad]), "good_mean": mean([t["wh_km"] for t in good]), "n_bad": len(bad), "n_good": len(good), "r2": fit["r2"], "n": len(fit["T"])}


def eff_trip(fit, t):
    """Why one trip used more/less than the typical trip: contribution of each condition relative to the mean trip."""
    typical = mean([x["wh_km"] for x in fit["T"]]); out = []
    for i, f in enumerate(fit["feats"]):
        x = f[2](t); out.append({"key": f[0], "name": f[1], "unit": f[3], "value": x, "mean": fit["mu"][i], "contrib": fit["beta"][i] * (x - fit["mu"][i])})
    out.sort(key=lambda r: -abs(r["contrib"])); return {"factors": out, "typical": typical, "actual": t["wh_km"], "r2": fit["r2"], "n": len(fit["T"])}


def daily_series(c, lo, hi):
    """Per local day: km, kWh, Wh/km, mean ambient, motorway share, trips."""
    T = trips_between(c, lo, hi); days = {}
    for t in T: days.setdefault((t["start"] + A.TZ) // 86400, []).append(t)
    out = []
    for d in sorted(days):
        L = days[d]; km = sum(x["km"] for x in L); kwh = sum(x["kwh"] for x in L)
        if km < 3: continue
        out.append({"day": d, "ts": d * 86400 - A.TZ, "date": dt.date.fromordinal(719163 + d).isoformat(), "km": km, "wh_km": 1000 * kwh / km, "amb": mean([x["amb"] for x in L]), "hwy": sum(x["hwy"] * x["km"] for x in L) / km,
                    "speed": mean([x["speed"] for x in L]), "trips": len(L), "harsh": sum(x["harsh"] * x["km"] / 100 for x in L) * 100 / km, "rain": sum(x["rain"] * x["km"] for x in L) / km, "hvac": mean([x["hvac"] for x in L])})
    return out


# ------------------------------------------------------------------------------------------------ formatting + windows around events
F1 = lambda x: f"{x:.1f}"
F0 = lambda x: f"{x:.0f}"
SIGNED = lambda x, d=1: f"{x:+.{d}f}"


def plural(n, w, pl=None): return f"{n} {w if n == 1 else (pl or w + 's')}"


def event_sig(c, ts, before, after):
    """Signals around ts. Uses the frozen 1 Hz event context (kept for every warning/fault, even after the 7-day 1 Hz tier is gone) when it covers the window; else the live tiers."""
    import json, zlib
    try:
        # v2 can retain more than one event context at a shared timestamp.
        # Prefer the context that actually covers the requested investigation
        # window; old databases retain the timestamp-only compatibility table.
        rs = c.execute("SELECT blob FROM event_ctx_v2 WHERE ts=?", (ts,)).fetchall()
        if not rs: rs = c.execute("SELECT blob FROM event_ctx WHERE ts=?", (ts,)).fetchall()
    except Exception: rs = []
    best = None
    for r in rs:
        try: rows = [x for x in json.loads(zlib.decompress(r[0])) if ts - before <= x[0] <= ts + after]
        except Exception: continue
        if best is None or len(rows) > len(best): best = rows
    if best and len(best) >= 10:
        rows = best
        live = load(c, ts - before, ts + after + 1)
        if live.res == 1 and len(live) >= len(rows): return live
        names = ("speed", "soc", "power", "batt", "flow", "amb", "hvac", "motor", "mode"); sc = (10.0, 10.0, 1000.0, 10.0, 10.0, 10.0, 1.0, 10.0, 1.0)
        return Sig([x[0] for x in rows], {n: [x[i + 1] / sc[i] for x in rows] for i, n in enumerate(names)}, 1, "frozen 1 Hz event context")
    return load(c, ts - before, ts + after + 1)


def summarise(sg, lo=None, hi=None):
    """Per signal: first, last, min (+time), max (+time), mean over the Sig (optionally restricted to lo<=ts<hi)."""
    idx = [i for i, t in enumerate(sg.ts) if (lo is None or t >= lo) and (hi is None or t < hi)]
    if not idx: return {}
    out = {}
    for m in list(SIGS) + ["rpm", "accel"]:
        col = sg.col(m); v = [col[i] for i in idx]; a = min(range(len(v)), key=v.__getitem__); b = max(range(len(v)), key=v.__getitem__)
        out[m] = {"first": v[0], "last": v[-1], "min": v[a], "tmin": sg.ts[idx[a]], "max": v[b], "tmax": sg.ts[idx[b]], "mean": mean(v), "n": len(v)}
    return out


def slope(ts, v):
    """Least-squares slope per second."""
    n = len(ts)
    if n < 3: return 0.0
    mt, mv = mean(ts), mean(v); den = sum((a - mt) ** 2 for a in ts)
    return sum((a - mt) * (b - mv) for a, b in zip(ts, v)) / den if den > 0 else 0.0


def baseline_stats(c, col, lo, hi, modes=(1, 2), scale=10.0, minv=None):
    """Mean, sd, 1st and 99th percentile of a per-minute signal while driving (or the given modes) over [lo, hi)."""
    v = FC.baseline_values(c, col, lo, hi, modes, scale, minv)
    if v is None:
        q = "SELECT %s FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (%s) AND %s IS NOT NULL" % (col, ",".join(str(m) for m in modes), col) + (f" AND {col}>{int(minv * scale)}" if minv is not None else "")
        v = sorted(r[0] / scale for r in c.execute(q, (lo, hi)).fetchall())
    if len(v) < 30: return None
    return {"p50": v[len(v) // 2], "mean": mean(v), "sd": sd(v) or 0.1, "p1": v[int(0.01 * len(v))], "p99": v[min(len(v) - 1, int(0.99 * len(v)))], "n": len(v)}


def last_abnormal(c, now):
    """Latest per-minute sample outside the car's own normal band (learned from the first half of the history): pack temperature, motor temperature, coolant flow while the pump runs."""
    cv = coverage(c, now)
    if cv["lo"] is None: return None
    mid = cv["lo"] + (cv["hi"] - cv["lo"]) // 2; lo = cv["lo"]; hi = min(mid, lo + 14 * 86400); best = None
    for col, key, kind in (("(batt-amb)", "batt", "high"), ("(motor-amb)", "motor", "high"), ("flow", "flow", "low")):         # temperatures are judged as the excess over the outside temperature, so a warm spell is not 'abnormal'
        b = baseline_stats(c, col, lo, hi, modes=(1, 2, 3), minv=3.0 if key == "flow" else None)
        if not b: continue
        if kind == "high": thr = b["p99"] + max(2.0, 2.0 * b["sd"]); cond = f"{col} > ?"; lim = thr * 10
        else: thr = 0.6 * b["p50"]; cond = f"{col} < ? AND {col} > 30"; lim = thr * 10
        if kind == "low":
            rows = FC.recent_below(c, col, hi, cv["hi"], lim, 30, 600)
            if rows is None: rows = c.execute(f"SELECT ts, {col} FROM sig_1m WHERE ts>=? AND ts<=? AND {cond} ORDER BY ts DESC LIMIT 600", (hi, cv["hi"], lim)).fetchall()
            have = {x[0] for x in rows}
            r = next((x for x in rows if (x[0] - 60 in have and x[0] + 60 in have) or (x[0] - 60 in have and x[0] - 120 in have)), None)           # three low minutes in a row = a real condition, not a spin-up / spin-down edge
        else:
            r = FC.latest_above(c, col, hi, cv["hi"], lim)
            if r is None: r = c.execute(f"SELECT ts, {col} FROM sig_1m WHERE ts>=? AND ts<=? AND {cond} ORDER BY ts DESC LIMIT 1", (hi, cv["hi"], lim)).fetchone()
        if r and (best is None or r[0] > best["ts"]):
            best = {"ts": r[0], "signal": key, "value": r[1] / 10.0, "normal": (f"{b['p50']:.1f} typical" if kind == "low" else f"up to {b['p99']:.1f}"), "threshold": thr, "kind": kind, "baseline": b, "label": SIGS[key][3] + (" above the outside temperature" if col.startswith("(") else ""), "unit": SIGS[key][2]}
    return best


def slope_test(x, y):
    """OLS slope of y on x with its two-sided p-value: {slope, p, r2, n} or None."""
    n = len(x)
    if n < 4: return None
    f = ols_line(x, y)
    if not f: return None
    mx = mean(x); sxx = sum((a - mx) ** 2 for a in x); ss_res = sum((c - (f["intercept"] + f["slope"] * a)) ** 2 for a, c in zip(x, y))
    if ss_res <= 0:
        # A perfectly flat series is not evidence of a trend.  A genuinely
        # non-zero perfect line remains maximally significant.
        return {"slope": f["slope"], "p": 1.0 if abs(f["slope"]) < 1e-12 else 0.0, "r2": f["r2"], "n": n}
    se = math.sqrt(ss_res / (n - 2) / sxx); t = f["slope"] / se if se > 0 else 0.0; df = n - 2
    return {"slope": f["slope"], "p": betainc(df / 2, 0.5, df / (df + t * t)), "r2": f["r2"], "n": n}


def pfmt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def flow_days(c, lo, hi, minn=15):
    """Median coolant flow (L/min, pump running) per local day: [(day_start_ts, median, n)]. Daily medians ignore the partial minutes when the pump spins up or down."""
    fast = FC.flow_days(c, lo, hi, minn)
    if fast is not None: return fast
    days = {}
    for ts, f in c.execute("SELECT ts, flow FROM sig_1m WHERE ts>=? AND ts<? AND flow>30 AND mode IN (1,2,3)", (lo, hi)):
        days.setdefault((ts + A.TZ) // 86400 * 86400 - A.TZ, []).append(f / 10.0)
    out = []
    for d in sorted(days):
        v = sorted(days[d])
        if len(v) >= minn: out.append((d, v[len(v) // 2], len(v)))
    return out


def flow_baseline(c, now):
    cv = coverage(c, now)
    if cv["lo"] is None: return None
    hi = min(cv["lo"] + (cv["hi"] - cv["lo"]) // 2, cv["lo"] + 14 * 86400)
    return baseline_stats(c, "flow", cv["lo"], hi, modes=(1, 2, 3), minv=3.0)


def flow_onset(c, now, upto=None):
    """First local day after which the daily median coolant flow stayed below ~90% of normal: (day_ts, median that day, normal) or None."""
    b = flow_baseline(c, now); cv = coverage(c, now)
    if not b: return None
    days = flow_days(c, cv["lo"], (upto or cv["hi"]) + 1)
    for i, (d, m, n) in enumerate(days):
        if m < 0.85 * b["p50"] and all(x[1] < 0.93 * b["p50"] for x in days[i:]): return d, m, b["p50"]
    return None


# ------------------------------------------------------------------------------------------------ more statistics (stdlib only)
def tcrit(df, conf=0.95):
    """Two-sided critical value of Student's t (bisection on the regularised incomplete beta)."""
    df = max(1.0, float(df)); target = 1 - conf; lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if betainc(df / 2, 0.5, df / (df + mid * mid)) > target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def ranks(v):
    order = sorted(range(len(v)), key=v.__getitem__); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y): return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def corr_p(r, n):
    if r is None or n < 4 or abs(r) >= 1: return 0.0 if r is not None and abs(r) >= 1 else 1.0
    t = r * math.sqrt((n - 2) / (1 - r * r)); return betainc((n - 2) / 2, 0.5, (n - 2) / ((n - 2) + t * t))


def partial_corr(x, y, z_list):
    """Correlation of x and y after regressing both on the variables in z_list (list of lists)."""
    if not z_list: return pearson(x, y)
    Z = [list(r) for r in zip(*z_list)]
    fx = ols_multi(Z, x, ridge=1e-9); fy = ols_multi(Z, y, ridge=1e-9)
    if fx is None or fy is None: return None
    bx, by = fx[0], fy[0]
    rx = [x[i] - bx[0] - sum(bx[k + 1] * Z[i][k] for k in range(len(Z[0]))) for i in range(len(x))]
    ry = [y[i] - by[0] - sum(by[k + 1] * Z[i][k] for k in range(len(Z[0]))) for i in range(len(y))]
    return pearson(rx, ry)


def inv(M):
    n = len(M); A = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(M)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12: return None
        A[i], A[p] = A[p], A[i]; d = A[i][i]; A[i] = [x / d for x in A[i]]
        for r in range(n):
            if r != i:
                f = A[r][i]; A[r] = [a - f * b for a, b in zip(A[r], A[i])]
    return [r[n:] for r in A]


def ols_se(X, y):
    """OLS with intercept -> {beta, se, t, p, r2, n} (lists incl. intercept first) or None."""
    n = len(X); k = len(X[0]) + 1
    if n < k + 3: return None
    Z = [[1.0] + list(r) for r in X]; XtX = [[sum(Z[i][a] * Z[i][b] for i in range(n)) for b in range(k)] for a in range(k)]; Xty = [sum(Z[i][a] * y[i] for i in range(n)) for a in range(k)]
    I = inv(XtX)
    if I is None: return None
    beta = [sum(I[a][b] * Xty[b] for b in range(k)) for a in range(k)]; res = [y[i] - sum(beta[a] * Z[i][a] for a in range(k)) for i in range(n)]
    s2 = sum(r * r for r in res) / (n - k); se = [math.sqrt(max(s2 * I[a][a], 0.0)) for a in range(k)]; t = [b / s if s > 0 else 0.0 for b, s in zip(beta, se)]
    p = [betainc((n - k) / 2, 0.5, (n - k) / ((n - k) + tt * tt)) for tt in t]; my = mean(y); sst = sum((v - my) ** 2 for v in y)
    return {"beta": beta, "se": se, "t": t, "p": p, "r2": 1 - sum(r * r for r in res) / sst if sst > 0 else 0.0, "n": n, "df": n - k}


def welch_ci(a, b, conf=0.95):
    w = welch(a, b)
    if not w: return None
    t, df, p = w; se = math.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b)); d = mean(b) - mean(a); h = tcrit(df, conf) * se
    sp = math.sqrt(((len(a) - 1) * sd(a) ** 2 + (len(b) - 1) * sd(b) ** 2) / max(len(a) + len(b) - 2, 1)); g = d / sp if sp > 0 else 0.0
    return {"diff": d, "lo": d - h, "hi": d + h, "t": t, "df": df, "p": p, "d": g, "n": (len(a), len(b))}


def mann_whitney_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation with tie correction)."""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3: return None
    r = ranks(list(a) + list(b)); u = sum(r[:n1]) - n1 * (n1 + 1) / 2; mu = n1 * n2 / 2; N = n1 + n2
    from collections import Counter
    ties = sum(c ** 3 - c for c in Counter(list(a) + list(b)).values()); sig = math.sqrt(n1 * n2 / 12 * ((N + 1) - ties / (N * (N - 1))))
    if sig == 0: return 1.0
    z = (u - mu) / sig; return math.erfc(abs(z) / math.sqrt(2))


def holm(ps, alpha=0.05):
    """-> list of booleans: rejected after Holm's step-down correction."""
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [False] * len(ps)
    for k, i in enumerate(idx):
        if ps[i] <= alpha / (len(ps) - k): out[i] = True
        else: break
    return out


def bh(ps, q=0.05):
    """Benjamini-Hochberg false-discovery control -> list of booleans."""
    n = len(ps); idx = sorted(range(n), key=lambda i: ps[i]); kmax = -1
    for k, i in enumerate(idx):
        if ps[i] <= q * (k + 1) / n: kmax = k
    out = [False] * n
    for k, i in enumerate(idx):
        if k <= kmax: out[i] = True
    return out


def bootstrap_slope(x, y, n=2000, seed=7, conf=0.95):
    import random
    rnd = random.Random(seed); N = len(x); out = []
    for _ in range(n):
        ii = [rnd.randrange(N) for _ in range(N)]; xs = [x[i] for i in ii]; ys = [y[i] for i in ii]
        if sd(xs) == 0: continue
        f = ols_line(xs, ys)
        if f: out.append(f["slope"])
    out.sort()
    if len(out) < 100: return None
    return out[int((1 - conf) / 2 * len(out))], out[int((1 + conf) / 2 * len(out)) - 1], len(out)


def autocorr(y, lag):
    n = len(y)
    if n - lag < 4: return None
    m = mean(y); den = sum((v - m) ** 2 for v in y)
    return sum((y[i] - m) * (y[i + lag] - m) for i in range(n - lag)) / den if den > 0 else None


def aic(rss, n, k): return n * math.log(max(rss, 1e-12) / n) + 2 * k
