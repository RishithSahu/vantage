"""What a vehicle reports, read from the data it has produced - never from a setting.

A real vehicle does not announce "I am an EV" or "I am a diesel": it sends signals.  A battery state of charge means a traction battery; a fuel level or an engine speed means a combustion engine; both
mean a hybrid; neither means the head unit has nothing to say about energy and must not pretend.  `detect(conn)` looks at the database (the signal catalogue and the rows actually stored) and returns the
capabilities the page and the assistant adapt to; `Reader` fetches the latest values of the signals that live in the lake (fuel level, engine speed, oil temperature ...) at a given moment;
`fuel_economy(conn, now)` derives litres per 100 km and the range of a combustion vehicle from its own fuel-rate and speed history.

    caps = detect(conn)          {"energy": "ev" | "ice" | "hybrid" | "unknown", "battery": bool, "fuel": bool, "engine": bool, "charging": bool, "refuel": bool, "tyres": ..., "concepts": {...}}
"""
import sqlite3

CORE = {"speed": "speed", "battery_soc": "soc", "battery_power": "power", "battery_temperature": "batt", "coolant_flow": "flow", "ambient_temperature": "amb", "hvac_power": "hvac", "motor_temperature": "motor", "drive_mode": "mode"}
LAKE_EXTRAS = ("fuel_level", "fuel_rate", "engine_rpm", "engine_load", "engine_torque", "engine_oil_temperature", "fuel_temperature", "barometric_pressure", "reported_range", "battery_voltage", "charging_power", "regen_power")
ENGINE_SIGNALS = ("engine_rpm", "engine_load", "engine_torque", "fuel_rate", "engine_oil_temperature")
DIESEL_ONLY = ("dpf_soot_load", "dpf_regeneration", "def_level", "boost_pressure", "exhaust_gas_temperature", "fuel_rail_pressure")          # diesel-only hardware: particulate filter, AdBlue / DEF tank, turbo, exhaust sensor, common rail
PETROL_ONLY = ("lambda", "short_term_fuel_trim", "long_term_fuel_trim", "intake_manifold_pressure", "catalyst_temperature", "ignition_timing", "mass_air_flow")   # spark-ignition: closed-loop mixture, three-way catalyst, spark
READ = ("fuel_level", "fuel_rate", "engine_rpm", "engine_load", "engine_torque", "engine_oil_temperature", "fuel_temperature", "barometric_pressure", "accelerator_position") + DIESEL_ONLY + PETROL_ONLY + ("intake_air_temperature",)
NAMES = {"engine_rpm": "engine speed", "engine_load": "engine load", "engine_torque": "engine torque", "fuel_rate": "fuel rate", "fuel_level": "fuel level", "engine_oil_temperature": "oil temperature",
         "dpf_soot_load": "particulate filter (DPF) soot load", "dpf_regeneration": "DPF regeneration state", "def_level": "AdBlue / DEF level", "boost_pressure": "boost pressure", "exhaust_gas_temperature": "exhaust gas temperature",
         "fuel_rail_pressure": "rail pressure", "lambda": "lambda (air-fuel ratio)", "short_term_fuel_trim": "short-term fuel trim", "long_term_fuel_trim": "long-term fuel trim", "intake_manifold_pressure": "manifold pressure",
         "catalyst_temperature": "catalyst temperature", "ignition_timing": "ignition timing", "mass_air_flow": "mass air flow", "battery_soc": "state of charge", "battery_voltage": "pack voltage", "battery_current": "pack current",
         "battery_temperature": "pack temperature", "charging_power": "charging power", "regen_power": "regenerative braking power"}
HOLD = {"fuel_level": 3 * 86400, "def_level": 3 * 86400}                    # slow-changing levels: seconds a last reading still describes a parked car
BUS = {"can": "CAN", "uds": "UDS / OBD-II diagnostics", "someip": "SOME/IP Ethernet", "vhal": "Android VHAL", "log": "system logs", "camera": "camera", "gnss": "GNSS"}


def _any(conn, sql, args=()):
    try: return conn.execute(sql, args).fetchone() is not None
    except sqlite3.Error: return False


def declared_powertrain(raw):
    """What a recorded file says about its own vehicle (meta `veh.powertrain`, e.g. 'diesel_ice', 'petrol_ice', 'bev') -> {"energy", "fuel_type", "raw"} or None.  A label is not a measurement: it is used only when the signals are silent."""
    if not raw: return None
    words = {w for w in "".join(ch if ch.isalpha() else " " for ch in str(raw).lower()).split()}
    if words & {"hybrid", "phev", "hev"}: return {"energy": "hybrid", "fuel_type": None, "raw": str(raw)}                  # before petrol / diesel: 'petrol_hybrid' is a hybrid
    if words & {"bev", "ev", "electric"}: return {"energy": "ev", "fuel_type": None, "raw": str(raw)}
    if words & {"diesel"}: return {"energy": "ice", "fuel_type": "diesel", "raw": str(raw)}
    if words & {"petrol", "gasoline", "gas"}: return {"energy": "ice", "fuel_type": "petrol", "raw": str(raw)}
    if words & {"ice", "combustion"}: return {"energy": "ice", "fuel_type": None, "raw": str(raw)}
    return None


def vehicle_kind(conn):
    """The powertrain class a model prompt is told about: 'ev', 'petrol', 'diesel', 'hybrid', 'ice' (combustion, fuel unknown) or None when the file says nothing."""
    k = detect(conn); e = k["energy"]
    if e == "unknown": e = (k.get("declared") or {}).get("energy", "unknown"); ft = (k.get("declared") or {}).get("fuel_type")
    else: ft = k.get("fuel_type")
    if e == "ev": return "ev"
    if e == "hybrid": return "hybrid"
    if e == "ice": return ft or "ice"
    return None


def is_combustion(k):
    """A combustion vehicle by its signals, or - when they are silent - by what its own file declares."""
    return k["energy"] == "ice" or (k["energy"] == "unknown" and (k.get("declared") or {}).get("energy") == "ice")


def detect(conn):
    """The capabilities of the vehicle that produced this database."""
    has, info = {}, {}
    for concept, col in CORE.items():
        has[concept] = _any(conn, f"SELECT 1 FROM sig_1m WHERE {col} IS NOT NULL LIMIT 1") or _any(conn, f"SELECT 1 FROM sig_1s WHERE {col} IS NOT NULL LIMIT 1")
    try: cat = conn.execute("SELECT sid, concept, label, unit FROM sig_catalog").fetchall()
    except sqlite3.Error: cat = []
    sids = {}
    for sid, concept, label, unit in cat:
        sids[concept] = sid; info[concept] = {"label": label, "unit": unit}
        if concept not in CORE and (_any(conn, "SELECT 1 FROM lk_1m WHERE sid=? LIMIT 1", (sid,)) or _any(conn, "SELECT 1 FROM lk_1s WHERE sid=? LIMIT 1", (sid,))): has[concept] = True
    battery = bool(has.get("battery_soc")); fuel = bool(has.get("fuel_level")) or bool(has.get("fuel_rate")); engine = any(has.get(c) for c in ENGINE_SIGNALS)
    energy = "hybrid" if battery and (fuel or engine) else "ev" if battery else "ice" if (fuel or engine) else "unknown"
    d_sig = [c for c in DIESEL_ONLY if has.get(c)]; p_sig = [c for c in PETROL_ONLY if has.get(c)]; fuel_type, why_fuel = None, ""
    if energy in ("ice", "hybrid"):
        if len(d_sig) > len(p_sig): fuel_type, why_fuel = "diesel", "diesel-only signals: " + ", ".join(NAMES.get(c, c) for c in d_sig)
        elif len(p_sig) > len(d_sig): fuel_type, why_fuel = "petrol", "spark-ignition signals: " + ", ".join(NAMES.get(c, c) for c in p_sig)
        elif sids.get("engine_rpm") is not None:                                                                  # nothing type-specific is reported: the rev range tells (a diesel is governed below ~4500 rpm, a petrol engine revs past it)
            try: mx = conn.execute("SELECT MAX(vmax) FROM lk_1m WHERE sid=?", (sids["engine_rpm"],)).fetchone()[0]
            except sqlite3.Error: mx = None
            if mx: fuel_type, why_fuel = ("petrol" if mx > 4600 else "diesel"), f"inferred from the engine speed range (peak {mx:.0f} rpm)"
    meta = {}
    try: meta = dict(conn.execute("SELECT k, v FROM meta WHERE k IN ('veh.tank_l','veh.cap_kwh','veh.powertrain')").fetchall())
    except sqlite3.Error: pass
    declared = declared_powertrain(meta.get("veh.powertrain")) if energy == "unknown" else None                     # the signals decide whenever they exist; a recorded file's own label only when they say nothing
    f = lambda k: float(meta[k]) if k in meta and meta[k] not in (None, "") else None
    sources = sorted({str(r[0]) for r in conn.execute("SELECT DISTINCT source FROM sig_catalog").fetchall() if r[0]}) if cat else []
    buses = [BUS.get(x.split(":")[0], x.split(":")[0]) + (" (" + x.split(":", 1)[1] + ")" if ":" in x and x.split(":")[0] == "can" else "") for x in sources]
    have = lambda *ks: [NAMES[k] for k in ks if has.get(k)]
    if energy == "ev": ident = "Battery-electric vehicle", ["state of charge", *have("battery_voltage", "battery_current", "battery_temperature", "charging_power", "regen_power")], "no fuel or engine signals"
    elif energy == "hybrid": ident = "Hybrid vehicle", [*have("battery_soc"), *have("fuel_level", "engine_rpm")], "both a traction battery and a combustion engine report"
    elif energy == "ice": ident = (fuel_type.capitalize() if fuel_type else "Combustion-engine") + " vehicle", [*have("engine_rpm", "engine_load", "fuel_rate", "fuel_level", "engine_oil_temperature")], (why_fuel or "no signal identifies the fuel type") + "; no traction-battery signals"
    elif declared: ident = ("Declared " + (declared["fuel_type"] + " " if declared["fuel_type"] else "") + {"ice": "combustion", "ev": "electric", "hybrid": "hybrid"}[declared["energy"]] + " vehicle"), [], f"the file labels the vehicle '{declared['raw']}' but holds no battery, fuel or engine signal (speed and position only), so nothing in the data confirms it"
    else: ident = "Energy source not reported", [], "this data holds no battery, fuel or engine signal (speed and position only)"
    summary = f"{ident[0]}." + (" Reports " + ", ".join(ident[1]) + "." if ident[1] else "") + f" Basis: {ident[2]}." + (" Data arrives over " + ", ".join(buses) + "." if buses else "")
    label = {"ev": "Electric", "hybrid": "Hybrid", "ice": (fuel_type or "combustion").capitalize()}.get(energy, "Energy source unknown")
    if declared: label = (declared["fuel_type"] or {"ice": "combustion", "ev": "electric", "hybrid": "hybrid"}[declared["energy"]]).capitalize() + " (declared)"
    return {"energy": energy, "declared": declared, "fuel_type": fuel_type, "label": label, "aftertreatment": (["particulate filter"] if has.get("dpf_soot_load") else []) + (["AdBlue / SCR"] if has.get("def_level") else []) + (["catalytic converter"] if has.get("catalyst_temperature") else []),
            "summary": summary, "buses": buses, "read": [c for c in READ if has.get(c)], "battery": battery, "fuel": fuel, "engine": engine, "tank_l": f("veh.tank_l"), "cap_kwh": f("veh.cap_kwh"),
            "charging": bool(has.get("charging_power")) or _any(conn, "SELECT 1 FROM charges LIMIT 1") or _any(conn, "SELECT 1 FROM events WHERE type='CHARGE_START' LIMIT 1"),
            "refuel": _any(conn, "SELECT 1 FROM events WHERE type='REFUEL' LIMIT 1"),
            "tyres": _any(conn, "SELECT 1 FROM ext_1m WHERE tp_fl IS NOT NULL LIMIT 1") or bool(has.get("tyre_pressure_fl")),
            "gps": _any(conn, "SELECT 1 FROM gps_1s LIMIT 1") or _any(conn, "SELECT 1 FROM ext_1m WHERE lat IS NOT NULL LIMIT 1"),
            "camera": _any(conn, "SELECT 1 FROM cam_1m LIMIT 1") or _any(conn, "SELECT 1 FROM events WHERE type LIKE 'CAMERA%' LIMIT 1"),
            "adas": _any(conn, "SELECT 1 FROM ext_1m WHERE adas IS NOT NULL LIMIT 1") or bool(has.get("adas_state")),
            "ambient": bool(has.get("ambient_temperature")), "coolant_flow": bool(has.get("coolant_flow")), "engine_temp": bool(has.get("motor_temperature")), "pack_temp": bool(has.get("battery_temperature")),
            "oil": bool(has.get("engine_oil_temperature")), "range_reported": bool(has.get("reported_range")), "speed": bool(has.get("speed")), "hvac": bool(has.get("hvac_power")),
            "signals": sorted(c for c, v in has.items() if v), "concepts": {c: info[c] for c in info if has.get(c)}}


class Reader:
    """Latest value of a lake signal at a moment (1 Hz row within `fresh` seconds, else the minute row within two minutes)."""
    def __init__(self, conn):
        self.conn = conn
        try: self.sid = dict(conn.execute("SELECT concept, sid FROM sig_catalog").fetchall())
        except sqlite3.Error: self.sid = {}

    def at(self, concepts, t, fresh=5):
        out, t = {}, int(t)
        for c in concepts:
            sid = self.sid.get(c)
            if sid is None: continue
            try:
                r = self.conn.execute("SELECT v FROM lk_1s WHERE sid=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (sid, t, t - fresh)).fetchone() or self.conn.execute("SELECT v FROM lk_1m WHERE sid=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (sid, t, t - 120)).fetchone()
                if r is None and c in HOLD:                                                          # a parked car sends little: the tank level is still what it was when the car last reported it
                    r = self.conn.execute("SELECT v FROM lk_1m WHERE sid=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (sid, t, t - HOLD[c])).fetchone()
            except sqlite3.Error: r = None
            if r is not None and r[0] is not None: out[c] = float(r[0])
        return out

    def fuel_economy(self, now, tank_l):
        """{"l100": litres per 100 km, "range_km": ...} from the last drives (two hours, then a day, then the whole record); None when there is no fuel-rate history."""
        sid = self.sid.get("fuel_rate")
        if sid is None: return None
        now = int(now)
        for span in (7200, 86400, 86400 * 30):
            try:
                lit = self.conn.execute("SELECT SUM(v) / 60.0 FROM lk_1m WHERE sid=? AND ts>? AND ts<=?", (sid, now - span, now)).fetchone()[0]
                km = self.conn.execute("SELECT SUM(speed) / 600.0 FROM sig_1m WHERE ts>? AND ts<=? AND speed IS NOT NULL", (now - span, now)).fetchone()[0]
            except sqlite3.Error: return None
            if lit and km and km >= 5.0 and lit > 0.2: return {"l100": lit / km * 100.0, "litres": lit, "km": km, "window_s": span}
        return None


def range_km(level_pct, tank_l, l100):
    if level_pct is None or not tank_l or not l100: return None
    return level_pct / 100.0 * tank_l / l100 * 100.0
