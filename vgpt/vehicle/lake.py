"""The signal lake: where every signal that is not one of the nine core columns lives, for any vehicle, without a schema change.

    sig_catalog(sid, concept, label, unit, ...)   one row per mapped signal: the DATABASE describes its own vehicle (concepts, units, plausible ranges, source, rate)
    lk_1s(sid, ts, v)                             1 Hz while the car is awake, 1 per minute while parked; kept `fine_days` (default 7)
    lk_1m(sid, ts, v, vmin, vmax, n)              one row per active minute (parked: every 10th minute); written LIVE, kept `coarse_days` (default 120 = 17 weeks)
    lkv_<sid>_1s / lkv_<sid>_1m                   one VIEW per signal exposing (ts, v[, vmin, vmax]): to the typed query engine a lake signal looks exactly like a column of a wide table,
                                                  so every operation (statistics, extremes, thresholds, trends, change detection, "why") works on it unchanged

The nine core signals (speed, soc, power, battery/motor temperature, coolant flow, ambient, hvac, drive mode) stay in the compact wide tables sig_1s / sig_1m (fastest path, backward
compatible).  `provision()` writes the profile and the catalogue; `adopt()` reads the catalogue back and registers the vehicle's concepts and storage mapping with the registries, so opening ANY
vehicle database makes the assistant speak that vehicle's vocabulary.  Everything here is stdlib sqlite3; the assistant itself only reads (the views are read-only)."""
import json
import math
import sqlite3

from .. import signal_registry as R
from .profile import Profile
from .router import ALIAS

LAKE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sig_catalog(sid INTEGER PRIMARY KEY, concept TEXT UNIQUE, label TEXT, unit TEXT, dtype TEXT, category TEXT, synonyms TEXT, lo REAL, hi REAL, source TEXT, key TEXT, rate_hz REAL, ttl REAL, core INTEGER, description TEXT);
CREATE TABLE IF NOT EXISTS lk_1s(sid INTEGER, ts INTEGER, v REAL, PRIMARY KEY(sid, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lk_1m(sid INTEGER, ts INTEGER, v REAL, vmin REAL, vmax REAL, n INTEGER, PRIMARY KEY(sid, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS event_ctx_x(ts INTEGER, type TEXT, blob BLOB, PRIMARY KEY(ts, type)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS source_log(ts INTEGER, source TEXT, status TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS i_source_log ON source_log(source, ts);
"""
CORE_STORED = {"speed", "battery_soc", "battery_power", "battery_temperature", "coolant_flow", "ambient_temperature", "hvac_power", "motor_temperature", "drive_mode", "camera_blockage", "camera_sharpness", "camera_brightness", "lane_confidence"}
SPEC_META = {"veh.model", "veh.cap_kwh", "veh.soh"}
EXT_COLS = {"tyre_pressure_fl": ("tp_fl", 10), "tyre_pressure_fr": ("tp_fr", 10), "tyre_pressure_rl": ("tp_rl", 10), "tyre_pressure_rr": ("tp_rr", 10), "rain": ("rain", 1), "light": ("lux", 1),
            "aux_battery_voltage": ("v12", 100), "adas_state": ("adas", 100), "odometer": ("odo", 10), "tyre_temperature": ("tt", 10)}          # the per-minute table the question handlers of the older routes read
TYRES = ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr")
CTX_TYPES_X = ("ADAS_DISENGAGE", "ECU_RESET", "UNEXPECTED_SHUTDOWN", "VOLTAGE_DIP", "SENSOR_CONFLICT", "AEB_ACTIVATION", "CAN_ERROR", "WARN_BATT_OVERTEMP", "WARN_OVERTEMP", "POWER_DERATE", "DTC_SET", "CAMERA_BLOCKED",
               "ADAS_STATE_CHANGE", "LOW_12V", "TPMS_LOW", "SOURCE_SILENT")                                                       # events whose surrounding channels are frozen into event_ctx_x
CTX_X_CONCEPTS = ("speed", "battery_voltage", "battery_current", "battery_temperature", "coolant_flow", "motor_temperature", "motor_torque", "aux_battery_voltage", "adas_state", "lane_confidence", "lead_vehicle_distance",
                  "brake_pressure", "steering_angle", "longitudinal_acceleration", "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "camera_blockage", "brake_temperature", "reported_range")



def lake_concepts(profile):
    """Concepts of the profile that live in the lake: everything bound that the wide tables do not already hold."""
    return [c for c in profile.concept_names() if not c.startswith("_") and c not in CORE_STORED]


def provision(conn, profile):
    """Create the schema, write the profile and the signal catalogue into `conn` (idempotent).  Returns {concept: sid}."""
    release()                                                                                  # validate and provision against the full canonical registry, never against another vehicle's overlay
    profile.check()
    conn.executescript(LAKE_SCHEMA)
    meta = {"vehicle.id": profile.id, "vehicle.profile": json.dumps(profile.spec, separators=(",", ":")), "vehicle.profile_schema": profile.spec.get("schema", ""),
            "ret.fine_days": str(profile.retention["fine_days"]), "ret.coarse_days": str(profile.retention["coarse_days"]), "ret.log_days": str(profile.retention["log_days"]), "ret.context_days": str(profile.retention["context_days"]),
            "ret.archive_days": str(profile.retention.get("archive_days", 365)), "ret.records_days": str(profile.retention.get("records_days", 1825))}
    for k, v in profile.specs.items(): meta[k] = str(v)
    for k, v in profile.thresholds.items(): meta["thr." + k] = str(v)
    try: meta["vehicle.dtc"] = json.dumps(profile.dtc_table(), separators=(",", ":"))                 # code -> text / subsystem / severity: investigations group faults by subsystem
    except (OSError, ValueError): pass
    conn.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", list(meta.items()))
    sids = {}
    declared = {c["name"]: c for c in profile.concepts}
    rows = []
    for concept in profile.concept_names():
        if concept.startswith("_"): continue
        b = next(x for x in sorted(profile.bindings, key=lambda x: x.priority) if x.concept == concept)
        cd = R.CONCEPTS.get(concept); d = declared.get(concept, {})
        label = cd.label if cd else d["label"]; unit = cd.unit if cd else profile.canonical_unit(concept); dtype = cd.dtype if cd else d.get("dtype", "numeric")
        lo, hi = (cd.lo, cd.hi) if cd else (d.get("lo"), d.get("hi")); cat = cd.category if cd else d.get("category", "vehicle")
        syn = list(cd.synonyms) if cd else list(d.get("synonyms", ()))
        ttl = float(b.ttl) if b.ttl else max(2.0, 3.0 / max(b.rate_hz, 0.01))
        rows.append((concept, label, unit, dtype, cat, json.dumps(syn), lo, hi, f"{b.bus}:{b.source}", b.key, b.rate_hz, ttl, 1 if concept in CORE_STORED else 0, (cd.description if cd else d.get("description", ""))))
    have = dict(conn.execute("SELECT concept, sid FROM sig_catalog").fetchall())
    nxt = max(have.values(), default=0) + 1
    for r in rows:
        if r[0] in have: sid = have[r[0]]
        else: sid = nxt; nxt += 1
        sids[r[0]] = sid
        conn.execute("INSERT OR REPLACE INTO sig_catalog(sid, concept, label, unit, dtype, category, synonyms, lo, hi, source, key, rate_hz, ttl, core, description) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (sid,) + r)
        if not r[12]: conn.execute(f"CREATE VIEW IF NOT EXISTS lkv_{sid}_1s AS SELECT ts, v FROM lk_1s WHERE sid={sid}"); conn.execute(f"CREATE VIEW IF NOT EXISTS lkv_{sid}_1m AS SELECT ts, v, vmin, vmax FROM lk_1m WHERE sid={sid}")
    ids = [sids.get(t) for t in TYRES]
    if all(i is not None for i in ids):                                                       # four wheels -> the aggregate concept "tyre pressure" (the lowest of the four), as a view
        lst = ",".join(str(i) for i in ids)
        conn.execute(f"CREATE VIEW IF NOT EXISTS lkv_tyre_1s AS SELECT ts, MIN(v) AS v FROM lk_1s WHERE sid IN ({lst}) GROUP BY ts HAVING COUNT(*)=4")
        conn.execute(f"CREATE VIEW IF NOT EXISTS lkv_tyre_1m AS SELECT ts, MIN(v) AS v FROM lk_1m WHERE sid IN ({lst}) GROUP BY ts HAVING COUNT(*)=4")
    return sids


# ------------------------------------------------------------------------------------------------ making a database speak its vehicle's vocabulary
_ADOPTED = {"id": None, "token": None}


def overlay_of(conn):
    """The schema overlay (schema_overlay.py) described by a database's own catalogue, or None for a database without one (the legacy generated vehicles)."""
    try: rows = conn.execute("SELECT sid, concept, label, unit, dtype, category, synonyms, lo, hi, core, description FROM sig_catalog ORDER BY sid").fetchall()
    except sqlite3.Error: return None
    if not rows: return None
    concepts, store = [], {}
    for sid, concept, label, unit, dtype, cat, syn, lo, hi, core, desc in rows:
        if concept not in R.CONCEPTS:
            concepts.append({"name": concept, "label": label, "unit": unit, "category": cat, "synonyms": json.loads(syn or "[]") + [label], "dtype": dtype, "lo": lo, "hi": hi, "description": desc or ""})
        if not core: store[concept] = {"1s": [f"lkv_{sid}_1s", "v", 1.0], "1m": [f"lkv_{sid}_1m", "v", 1.0]}
    names = {c for _, c, *_ in rows}
    if all(t in names for t in TYRES): store["tyre_pressure"] = {"1s": ["lkv_tyre_1s", "v", 1.0], "1m": ["lkv_tyre_1m", "v", 1.0]}
    drop = []
    try:
        spec = json.loads(conn.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()[0]); drop = list(spec.get("drop", []))
        if spec.get("vehicle", {}).get("powertrain") not in ("bev", "phev", "hev", None): drop += [d for d in EV_ONLY if d != "range_estimate"]        # a combustion vehicle has no battery hardware, but "how far can I go" is still a question about it (answered in fuel terms)
    except (sqlite3.Error, TypeError, ValueError): pass
    return {"concepts": concepts, "store": store, "drop": [d for d in dict.fromkeys(drop) if d in R.CONCEPTS]}


def declared_names(conn):
    """The concepts this vehicle's own profile declares (signals the shared registry did not have): the vocabulary only this vehicle brings."""
    try: return [c["name"] for c in json.loads(conn.execute("SELECT v FROM meta WHERE k='vehicle.profile'").fetchone()[0]).get("concepts", [])]
    except (sqlite3.Error, TypeError, ValueError): return []


EV_ONLY = ["battery_soc", "battery_voltage", "battery_current", "battery_temperature", "battery_power", "battery_soh", "cell_temperature_max", "cell_temperature_min", "cell_voltage_min", "charging_power", "charging_state",
           "regen_power", "regen_energy", "coolant_flow", "range_estimate", "thermal_rise_rate", "battery_efficiency", "motor_torque", "motor_power", "motor_rpm", "motor_temperature"]


def release():
    """Undo the adopted vehicle overlay (back to the full canonical registry) - used before provisioning another vehicle in the same process and by tests."""
    from .. import schema_overlay
    from ..meaning import ontology
    if _ADOPTED["token"] is not None:
        schema_overlay.restore(_ADOPTED["token"]); _ADOPTED.update(id=None, token=None); ontology.set_absent(()); ontology.reload()


def adopt(conn, key=None):
    """Register the database's vehicle with the process-wide registries (idempotent; opening another vehicle's database replaces the previous overlay).  Returns the overlay applied, or None."""
    from .. import schema_overlay
    from ..meaning import ontology, glossary
    ident = key or id(conn)
    if _ADOPTED["token"] is not None:
        schema_overlay.restore(_ADOPTED["token"]); _ADOPTED.update(id=None, token=None); ontology.set_absent(()); ontology.reload()
    ov = overlay_of(conn)
    if ov is None: ontology.set_absent(()); ontology.reload(); return None
    _ADOPTED["token"] = schema_overlay.apply(ov); _ADOPTED["id"] = ident; ontology.set_absent(ov["drop"])
    for tiers in ov["store"].values():
        for table, col, _ in tiers.values(): R.TABLE_COLUMNS.setdefault(table, set()).update({"ts", "v", "vmin", "vmax"})
    ontology.reload()
    glossary.load().extend([{"term": c["label"], "forms": [s for s in c["synonyms"] if s != c["label"]], "kind": "term", "meaning": c.get("description") or c["label"], "concept": c["name"], "hint": True} for c in ov["concepts"]])
    return ov


def adopt_path(path):
    """Open `path` read-only just long enough to adopt its vehicle (call BEFORE building the frame model, whose decoding grammar is generated from the registered concepts)."""
    import os
    if not os.path.exists(path): return None
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    try: return adopt(c, key=os.path.abspath(path))
    finally: c.close()


# ------------------------------------------------------------------------------------------------ live writer
class Lake:
    """Per-second writer, driven by the ingestor: current values of every lake concept -> lk_1s (active: every second, parked: every minute) and live 1-minute aggregates -> lk_1m."""
    def __init__(self, conn, profile=None):
        self.c = conn; self.sig = {}; self.bucket = {}; self.minute = None; self.last_s = {}; self.rows = 0; self.last_min = {}
        self.name_of = {}
        for sid, concept, ttl, core in conn.execute("SELECT sid, concept, ttl, core FROM sig_catalog").fetchall():
            if not core: self.sig[concept] = (sid, float(ttl or 3.0)); self.name_of[sid] = concept
        try: self.tz = int(float(conn.execute("SELECT v FROM meta WHERE k='env.tz_h'").fetchone()[0]) * 3600)
        except (TypeError, sqlite3.Error): self.tz = 0
        self.day = None; self.dayagg = {}

    def tick(self, sec, latest, active):
        m = sec // 60 * 60
        if self.minute is None: self.minute = m
        if m != self.minute: self._flush_minute(); self.minute = m
        batch = []
        for concept, (sid, ttl) in self.sig.items():
            item = latest.get(concept)
            if item is None: continue
            v, src_ts = item
            if not (0 <= sec - src_ts <= ttl) or not math.isfinite(v): continue
            b = self.bucket.get(sid)
            if b is None: self.bucket[sid] = [v, v, v, 1]
            else: b[0] += v; b[1] = min(b[1], v); b[2] = max(b[2], v); b[3] += 1
            if active or sec - self.last_s.get(sid, -1e9) >= 60: batch.append((sid, sec, v)); self.last_s[sid] = sec
        if batch: self.c.executemany("INSERT OR REPLACE INTO lk_1s VALUES(?,?,?)", batch); self.rows += len(batch)
        self._active_min = getattr(self, "_active_min", False) or active

    def _flush_minute(self):
        if self.minute is None or not self.bucket: self.bucket = {}; self._active_min = False; return
        every = getattr(self, "_active_min", False) or (self.minute // 60) % 10 == 0                  # parked minutes: one in ten is enough for a coarse history ...
        keep = {sid: b for sid, b in self.bucket.items() if self._changed(sid, b[0] / b[3]) or every}   # ... plus a reading the first time it is seen and whenever it changes (a tyre losing pressure in a parked car)
        if keep:
            self.c.executemany("INSERT OR REPLACE INTO lk_1m VALUES(?,?,?,?,?,?)", [(sid, self.minute, b[0] / b[3], b[1], b[2], b[3]) for sid, b in keep.items()])
            self._compat(self.minute, {self.name_of[sid]: b for sid, b in keep.items() if sid in self.name_of})
        self.bucket = {}; self._active_min = False

    def _changed(self, sid, avg):
        prev = self.last_min.get(sid); self.last_min[sid] = avg
        return prev is None or abs(avg - prev) > 0.002 * max(1.0, abs(prev))

    def _compat(self, m, by):
        """The minute extras (tyres, rain, light, 12 V, ADAS share, odometer) in ext_1m and the daily row in vday: what the older answer routes read, kept live from the lake."""
        row = {}
        for concept, (col, k) in EXT_COLS.items():
            b = by.get(concept)
            if b is not None: row[col] = int(round(b[0] / b[3] * k))
        if row:
            cols = ["ts"] + list(row); self.c.execute(f"INSERT OR REPLACE INTO ext_1m({','.join(cols)}) VALUES({','.join('?' * len(cols))})", [m] + list(row.values()))
        d = ((m + self.tz) // 86400) * 86400 - self.tz
        if self.day != d: self._flush_day(); self.day = d; self.dayagg = {}
        a = self.dayagg
        for concept, key in (("odometer", "odo"), ("brake_pad_life", "pad"), ("battery_soh", "soh"), ("aux_battery_voltage", "v12")):
            b = by.get(concept)
            if b is not None: a[key] = b[2] if key == "odo" else b[0] / b[3]
        if (m // 60) % 10 == 0: self._flush_day()

    def _flush_day(self):
        if self.day is not None and self.dayagg:
            a = self.dayagg; self.c.execute("INSERT OR REPLACE INTO vday(ts, odo, pad, soh, v12) VALUES(?,?,?,?,?)", (self.day, a.get("odo"), a.get("pad"), a.get("soh"), a.get("v12")))

    def close(self): self._flush_minute(); self._flush_day()


def compact_lake(conn, now_ts):
    """Expire the lake tiers by the retention the database was provisioned with (meta ret.*)."""
    meta = dict(conn.execute("SELECT k, v FROM meta WHERE k LIKE 'ret.%'").fetchall())
    fine, coarse = int(float(meta.get("ret.fine_days", 7))), int(float(meta.get("ret.coarse_days", 120)))
    conn.execute("DELETE FROM lk_1s WHERE ts < ?", (now_ts // 60 * 60 - fine * 86400,)); conn.execute("DELETE FROM lk_1m WHERE ts < ?", (now_ts - coarse * 86400,))
