# VERBATIM COPY of vgpt/signal_registry.py before the 2026-10-23 availability speed-up (reference for tests/test_availability_tier.py)
"""Signal registry: the single formal description of what a vehicle COULD know (the ontology) and, separately, what THIS vehicle history actually contains (the data profile).

    ONTOLOGY != DATA.  A concept is "known" when it is in CONCEPTS. It is "available" only when a storage mapping exists AND the database in front of us really has rows for it
    (or, for a derived metric, when a dependency set that is available can produce it). The three answers the rest of the system needs are therefore:
        known + available      -> can be answered from data
        known + unavailable    -> understood, but the data is missing (say exactly which signals would be needed)
        unknown                -> not a vehicle concept we model

Nothing in here talks to a language model; the model only ever picks names from this registry (its output grammar is generated from it, see query_grammar.py).
Storage mappings were written from the real schema (sig_1s / sig_1m / ext_1m / gps_1s / cam_1m / vday / trips / trip_ext / events / charges) - not from what a vehicle "ought" to have."""
import sqlite3
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------------------------ units
UNIT_FAMILIES = {"speed": ("km/h", "mph", "m/s"), "temperature": ("C", "F", "K"), "pressure": ("kPa", "bar", "psi"), "distance": ("km", "mi", "m"), "energy": ("kWh", "Wh", "J", "MJ"), "power": ("kW", "W", "hp"),
                 "consumption": ("Wh/km", "Wh/mi", "kWh/100km"), "torque": ("Nm",), "angle": ("deg", "rad"), "angular_rate": ("deg/s", "rad/s"), "acceleration": ("m/s2", "g"), "flow": ("L/min",), "voltage": ("V",), "current": ("A",),
                 "percent": ("%",), "rpm": ("rpm",), "time": ("s", "min", "h"), "illuminance": ("lux",), "ratio": ("ratio",), "count": ("count",), "rate": ("C/min", "kW/min"), "category": ("category",), "none": ("none",), "fuel_flow": ("L/h", "gal/h"), "length": ("mm", "in")}
_UNIT_TO_FAMILY = {u: f for f, us in UNIT_FAMILIES.items() for u in us}


def unit_family(u): return _UNIT_TO_FAMILY.get(u)


def convert(value, frm, to):
    """Deterministic unit conversion inside one family; raises ValueError across families."""
    if frm == to: return value
    if unit_family(frm) != unit_family(to) or unit_family(frm) is None: raise ValueError(f"cannot convert {frm} to {to}")
    f = unit_family(frm)
    if f == "temperature":
        c = value if frm == "C" else (value - 32) * 5 / 9 if frm == "F" else value - 273.15
        return c if to == "C" else c * 9 / 5 + 32 if to == "F" else c + 273.15
    table = {"speed": {"km/h": 1.0, "mph": 1.609344, "m/s": 3.6}, "pressure": {"kPa": 1.0, "bar": 100.0, "psi": 6.894757}, "distance": {"km": 1.0, "mi": 1.609344, "m": 0.001}, "energy": {"kWh": 1.0, "Wh": 0.001, "J": 1 / 3.6e6, "MJ": 1 / 3.6},
             "power": {"kW": 1.0, "W": 0.001, "hp": 0.7456999}, "fuel_flow": {"L/h": 1.0, "gal/h": 3.785411784}, "length": {"mm": 1.0, "in": 25.4}, "angle": {"deg": 1.0, "rad": 57.29577951}, "angular_rate": {"deg/s": 1.0, "rad/s": 57.29577951}, "acceleration": {"m/s2": 1.0, "g": 9.80665}, "time": {"s": 1.0, "min": 60.0, "h": 3600.0}}
    if f == "consumption":
        wh_km = value if frm == "Wh/km" else value * 1.609344 if frm == "Wh/mi" else value * 10.0
        return wh_km if to == "Wh/km" else wh_km / 1.609344 if to == "Wh/mi" else wh_km / 10.0
    return value * table[f][frm] / table[f][to]


# ------------------------------------------------------------------------------------------------ concepts
@dataclass(frozen=True)
class Concept:
    name: str
    label: str
    unit: str
    category: str                                   # powertrain | battery | chassis | adas | environment | vehicle | camera | trip
    dtype: str = "numeric"                          # numeric | categorical | binary
    kind: str = "measured"                          # measured | derived
    synonyms: tuple = ()
    lo: float = None                                # plausible physical range (sensor-quality engine)
    hi: float = None
    description: str = ""
    levels: tuple = ()                              # categorical levels


def C(name, label, unit, category, synonyms=(), lo=None, hi=None, dtype="numeric", kind="measured", description="", levels=()):
    return Concept(name, label, unit, category, dtype, kind, tuple(synonyms), lo, hi, description, tuple(levels))


CONCEPTS = {c.name: c for c in [
    # ---- powertrain
    C("motor_torque", "motor torque", "Nm", "powertrain", ("torque", "motor torque", "engine torque", "wheel torque"), -500, 500),
    C("motor_rpm", "motor RPM", "rpm", "powertrain", ("rpm", "revs", "revolutions", "motor speed", "motor rpm", "engine speed", "engine rpm"), -2000, 20000),
    C("motor_power", "motor (mechanical) power", "kW", "powertrain", ("motor power", "mechanical power", "shaft power", "drive power"), -250, 400),
    C("motor_temperature", "motor temperature", "C", "powertrain", ("motor temperature", "motor temp", "motor heat", "drive motor temperature", "engine temperature", "engine temp"), -40, 200),
    C("engine_rpm", "engine rpm", "rpm", "powertrain", ("engine rpm",), 0, 8000), C("engine_torque", "engine torque", "Nm", "powertrain", ("engine torque",), 0, 800),
    C("engine_load", "engine load", "%", "powertrain", ("engine load",), 0, 100), C("fuel_rate", "fuel rate", "L/h", "powertrain", ("fuel rate", "fuel consumption rate"), 0, 100),
    C("transmission_temperature", "transmission temperature", "C", "powertrain", ("transmission temperature", "gearbox temperature", "gearbox temp"), -40, 200),
    C("drive_mode", "driving state", "category", "powertrain", ("driving state", "drive mode", "mode"), dtype="categorical", levels=("parked", "city", "highway", "charging")),
    # ---- battery
    C("battery_power", "battery power", "kW", "battery", ("battery power", "electrical power", "pack power", "power draw", "power"), -300, 400, description="electrical; positive = discharging, negative = charging or regeneration"),
    C("battery_voltage", "battery voltage", "V", "battery", ("battery voltage", "pack voltage", "traction voltage", "hv voltage"), 0, 1000), C("battery_current", "battery current", "A", "battery", ("battery current", "pack current", "traction current"), -900, 900),
    C("battery_soc", "state of charge", "%", "battery", ("state of charge", "soc", "charge level", "battery level", "battery percentage", "battery charge level", "how much charge", "full battery", "empty battery", "fully charged", "battery is full"), 0, 100),
    C("battery_temperature", "battery temperature", "C", "battery", ("battery temperature", "battery temp", "pack temperature", "pack temp", "battery heat", "battery pack temperature", "how hot the battery", "how hot is the battery", "how hot was the battery"), -40, 90),
    C("cell_temperature_max", "hottest cell temperature", "C", "battery", ("cell temperature", "hottest cell", "max cell temperature"), -40, 90), C("cell_temperature_min", "coolest cell temperature", "C", "battery", ("coolest cell", "min cell temperature"), -40, 90),
    C("cell_voltage_min", "lowest cell voltage", "V", "battery", ("cell voltage", "lowest cell voltage", "weakest cell"), 0, 5), C("battery_soh", "battery health", "%", "battery", ("battery health", "state of health", "soh", "battery degradation", "battery capacity", "healthy battery", "is it healthy", "is healthy", "battery condition"), 0, 100),
    C("charging_power", "charging power", "kW", "battery", ("charging power", "charge rate", "charging rate", "charging speed", "charger power"), 0, 400), C("charging_state", "charging state", "category", "battery", ("charging state",), dtype="categorical", levels=("not_charging", "ac", "dc")),
    C("regen_power", "regenerative braking power", "kW", "battery", ("regen power", "regeneration power", "regenerative power"), 0, 250),
    C("coolant_flow", "coolant flow", "L/min", "battery", ("coolant flow", "cooling flow", "pump flow", "coolant", "flow rate", "coolant pump flow"), 0, 60),
    C("hvac_power", "climate-control power", "W", "vehicle", ("climate power", "hvac power", "heater power", "ac power", "climate control", "hvac", "heating power"), 0, 8000),
    C("aux_battery_voltage", "12 V battery voltage", "V", "vehicle", ("12v battery", "12 v battery", "auxiliary battery", "12 volt battery", "aux battery"), 6, 16),
    # ---- chassis
    C("speed", "vehicle speed", "km/h", "chassis", ("speed", "velocity", "how fast", "vehicle speed"), -10, 300),
    C("steering_angle", "steering angle", "deg", "chassis", ("steering angle", "steering wheel angle", "steering"), -900, 900), C("steering_rate", "steering rate", "deg/s", "chassis", ("steering rate",), -1500, 1500), C("steering_torque", "steering torque", "Nm", "chassis", ("steering torque",), -20, 20),
    C("yaw_rate", "yaw rate", "deg/s", "chassis", ("yaw rate", "yaw"), -120, 120), C("longitudinal_acceleration", "longitudinal acceleration", "m/s2", "chassis", ("longitudinal acceleration", "forward acceleration", "g force"), -15, 15),
    C("lateral_acceleration", "lateral acceleration", "m/s2", "chassis", ("lateral acceleration", "sideways acceleration", "cornering acceleration"), -15, 15),
    C("wheel_speed_fl", "front-left wheel speed", "km/h", "chassis", ("front left wheel speed",), -10, 300), C("wheel_speed_fr", "front-right wheel speed", "km/h", "chassis", ("front right wheel speed",), -10, 300),
    C("wheel_speed_rl", "rear-left wheel speed", "km/h", "chassis", ("rear left wheel speed",), -10, 300), C("wheel_speed_rr", "rear-right wheel speed", "km/h", "chassis", ("rear right wheel speed",), -10, 300),
    C("wheel_speed", "wheel speed", "km/h", "chassis", ("wheel speed", "wheel speeds", "speed of a wheel", "wheel rotation speed"), -10, 300, description="one of the four wheels (the wheel named in the question selects the signal)"),
    C("brake_pressure", "brake pressure", "bar", "chassis", ("brake pressure", "braking pressure"), 0, 200), C("brake_pedal", "brake pedal position", "%", "chassis", ("brake pedal",), 0, 100), C("accelerator_position", "accelerator position", "%", "chassis", ("accelerator position", "throttle", "throttle position", "accelerator pedal", "pedal position"), 0, 100),
    C("tyre_pressure_fl", "front-left tyre pressure", "kPa", "chassis", ("front left tyre pressure",), 0, 600), C("tyre_pressure_fr", "front-right tyre pressure", "kPa", "chassis", ("front right tyre pressure",), 0, 600),
    C("tyre_pressure_rl", "rear-left tyre pressure", "kPa", "chassis", ("rear left tyre pressure",), 0, 600), C("tyre_pressure_rr", "rear-right tyre pressure", "kPa", "chassis", ("rear right tyre pressure",), 0, 600),
    C("tyre_pressure", "tyre pressure", "kPa", "chassis", ("tyre pressure", "tire pressure", "tyre pressures", "tire pressures", "tyres", "tires"), 0, 600, description="all four tyres (the lowest one is reported)"),
    C("tyre_temperature", "tyre temperature", "C", "chassis", ("tyre temperature", "tire temperature"), -40, 120),
    # ---- ADAS
    C("adas_state", "driver-assistance state", "category", "adas", ("adas", "driver assistance", "lane assist state", "lane keeping"), dtype="categorical", levels=("off", "active")), C("adas_disengagement", "ADAS disengagement", "count", "adas", ("adas disengagement", "lane assist disengagement"), dtype="binary"),
    C("lane_confidence", "lane-marking confidence", "ratio", "adas", ("lane confidence", "lane marking confidence", "lane detection"), 0, 1), C("lane_departure", "lane departure", "count", "adas", ("lane departure",), dtype="binary"),
    C("lead_vehicle_distance", "distance to the vehicle ahead", "m", "adas", ("lead vehicle distance", "following distance", "distance to the car ahead"), 0, 300), C("aeb_state", "emergency-braking state", "category", "adas", ("aeb", "emergency braking state"), dtype="categorical"),
    C("acc_state", "adaptive cruise state", "category", "adas", ("acc", "adaptive cruise", "cruise control"), dtype="categorical"), C("blindspot_state", "blind-spot warning state", "category", "adas", ("blind spot", "blindspot"), dtype="categorical"),
    # ---- camera metadata
    C("camera_blockage", "camera blockage", "ratio", "camera", ("camera blockage", "lens blockage", "camera blocked"), 0, 1), C("camera_sharpness", "camera sharpness", "ratio", "camera", ("camera sharpness", "image sharpness", "camera focus"), 0, 5), C("camera_brightness", "camera brightness", "none", "camera", ("camera brightness", "image brightness"), 0, 255),
    # ---- environment
    C("ambient_temperature", "outside temperature", "C", "environment", ("outside temperature", "ambient temperature", "external temperature", "air temperature", "weather temperature", "outside temp"), -50, 60),
    C("rain", "rain", "none", "environment", ("rain", "raining", "wet weather"), dtype="binary"), C("light", "ambient light", "lux", "environment", ("ambient light", "light level", "brightness outside", "daylight"), 0, 130000),
    C("humidity", "humidity", "%", "environment", ("humidity",), 0, 100), C("wind_speed", "wind speed", "km/h", "environment", ("wind speed", "wind"), 0, 200), C("wind_direction", "wind direction", "deg", "environment", ("wind direction",), 0, 360),
    # ---- position
    C("gps_speed", "GPS speed", "km/h", "vehicle", ("gps speed",), 0, 300), C("heading", "heading", "deg", "vehicle", ("heading", "compass"), 0, 360), C("odometer", "odometer", "km", "vehicle", ("odometer", "mileage on the clock"), 0, 2e6),
    C("brake_pad_life", "brake pad life", "%", "vehicle", ("brake pad", "brake pads", "pad wear", "brake wear"), 0, 100),
    # Known-but-unavailable component observations remain separate ontology
    # entries. This lets capability analysis refuse honestly instead of
    # substituting a nearby measurement such as brake-pad wear.
    C("brake_temperature", "brake temperature", "C", "chassis", ("brake temperature", "brake temp", "brake temps", "brake pad temperature", "brake pad temp", "brake disc temperature", "brake disc temp", "brake rotor temperature"), -40, 1000, description="temperature of the brake discs and pads"),
    C("brake_fluid_condition", "brake fluid condition", "category", "vehicle", ("brake fluid", "brake fluid condition", "brake fluid health", "brake fluid status"), dtype="categorical", levels=("ok", "attention", "service")),
    C("tyre_tread_depth", "tyre tread depth", "mm", "vehicle", ("tyre tread depth", "tire tread depth", "tread depth", "tyre tread", "tire tread", "tread wear"), 0, 15),
    C("rated_max_speed", "rated maximum speed", "km/h", "vehicle", ("rated maximum speed", "rated top speed", "top speed specification", "maximum speed specification"), 0, 500),
    C("cabin_temperature", "cabin temperature", "C", "vehicle", ("cabin temperature", "interior temperature", "inside temperature", "cabin temp"), -20, 80), C("fuel_level", "fuel level", "%", "vehicle", ("fuel level", "fuel"), 0, 100),
    # ---- derived (declared here, implemented in derived.py)
    C("acceleration", "signed acceleration", "m/s2", "chassis", ("acceleration", "signed acceleration", "mean signed acceleration"), -15, 15, kind="derived"),
    C("acceleration_magnitude", "acceleration magnitude", "m/s2", "chassis", ("acceleration magnitude", "absolute acceleration", "mean absolute acceleration", "average absolute acceleration"), 0, 15, kind="derived"),
    C("braking_intensity", "braking intensity", "g", "chassis", ("braking intensity", "braking force", "deceleration"), 0, 2, kind="derived"),
    C("horsepower", "motor horsepower", "hp", "powertrain", ("horsepower", "hp", "bhp", "how many horses"), -350, 600, kind="derived"), C("turn_radius", "turning radius", "m", "chassis", ("turn radius", "turning radius", "turning circle"), 1, 5000, kind="derived"),
    C("energy_used", "energy used", "kWh", "trip", ("energy", "energy used", "electricity used", "kwh used", "how much energy"), kind="derived"), C("distance_driven", "distance driven", "km", "trip", ("distance", "distance driven", "how far", "km driven"), kind="derived"),
    C("wh_per_km", "energy consumption", "Wh/km", "trip", ("consumption", "efficiency", "wh per km", "energy per km", "energy consumption", "wh/km", "economy", "mileage"), 0, 2000, kind="derived"),
    C("regen_energy", "energy recovered by regenerative braking", "kWh", "battery", ("regen energy", "regenerated energy", "energy recovered", "regen", "regenerative braking", "recuperation"), kind="derived"),
    C("wheel_slip", "wheel slip", "ratio", "chassis", ("wheel slip", "slip"), -1, 1, kind="derived"), C("thermal_rise_rate", "battery temperature rise rate", "C/min", "battery", ("thermal rise rate", "heating rate", "temperature rise rate"), -5, 5, kind="derived"),
    C("range_estimate", "estimated range", "km", "battery", ("range", "remaining range", "how far can i go"), 0, 2000, kind="derived"), C("battery_efficiency", "battery-to-wheel efficiency", "ratio", "battery", ("drivetrain efficiency", "battery efficiency"), 0, 1.2, kind="derived"),
    C("motor_power_estimate", "motor power (estimated)", "kW", "powertrain", (), -250, 400, kind="derived"),
]}
from .ingest import catalog as _catalog                                        # the universal signal catalogue: the quantities of any vehicle (engine oil temperature, DPF soot load, throttle ...) are known from the start
_catalog.install(CONCEPTS, C)

# ------------------------------------------------------------------------------------------------ storage: where each MEASURED concept lives in the real schema
# (table, column, scale: stored/scale = engineering unit, sign applied after scaling). Tiers: "1s" = sig_1s (7 days), "1m" = per-minute tables (90 days).
STORE = {
    "speed": {"1s": ("sig_1s", "speed", 10.0), "1m": ("sig_1m", "speed", 10.0)}, "battery_soc": {"1s": ("sig_1s", "soc", 10.0), "1m": ("sig_1m", "soc", 10.0)}, "battery_power": {"1s": ("sig_1s", "power", 1000.0), "1m": ("sig_1m", "power", 1000.0)},
    "battery_temperature": {"1s": ("sig_1s", "batt", 10.0), "1m": ("sig_1m", "batt", 10.0)}, "coolant_flow": {"1s": ("sig_1s", "flow", 10.0), "1m": ("sig_1m", "flow", 10.0)}, "ambient_temperature": {"1s": ("sig_1s", "amb", 10.0), "1m": ("sig_1m", "amb", 10.0)},
    "hvac_power": {"1s": ("sig_1s", "hvac", 1.0), "1m": ("sig_1m", "hvac", 1.0)}, "motor_temperature": {"1s": ("sig_1s", "motor", 10.0), "1m": ("sig_1m", "motor", 10.0)}, "drive_mode": {"1s": ("sig_1s", "mode", 1.0), "1m": ("sig_1m", "mode", 1.0)},
    "tyre_pressure_fl": {"1m": ("ext_1m", "tp_fl", 10.0)}, "tyre_pressure_fr": {"1m": ("ext_1m", "tp_fr", 10.0)}, "tyre_pressure_rl": {"1m": ("ext_1m", "tp_rl", 10.0)}, "tyre_pressure_rr": {"1m": ("ext_1m", "tp_rr", 10.0)}, "tyre_temperature": {"1m": ("ext_1m", "tt", 10.0)},
    "rain": {"1m": ("ext_1m", "rain", 1.0)}, "light": {"1m": ("ext_1m", "lux", 1.0)}, "aux_battery_voltage": {"1m": ("ext_1m", "v12", 100.0)}, "adas_state": {"1m": ("ext_1m", "adas", 1.0)}, "odometer": {"1m": ("ext_1m", "odo", 10.0)},
    "gps_speed": {"1s": ("gps_1s", "spd", 10.0)}, "heading": {"1s": ("gps_1s", "hdg", 10.0)},
    "lane_confidence": {"1m": ("cam_1m", "lane_avg", 1.0)}, "camera_blockage": {"1m": ("cam_1m", "block_avg", 1.0)}, "camera_sharpness": {"1m": ("cam_1m", "sharp_avg", 1.0)}, "camera_brightness": {"1m": ("cam_1m", "luma_avg", 1.0)},
    "battery_soh": {"day": ("vday", "soh", 1.0)}, "brake_pad_life": {"day": ("vday", "pad", 1.0)},
}
# extended signals (additive tables created by the simulator / a future ingest: xsig_1s, xsig_1m - one wide row per timestamp, same scaling convention: plain engineering units as REAL)
XSIGNALS = ("steering_angle", "steering_rate", "yaw_rate", "longitudinal_acceleration", "lateral_acceleration", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "motor_rpm", "motor_torque", "motor_power",
            "battery_voltage", "battery_current", "brake_pressure", "accelerator_position", "charging_power", "regen_power", "brake_pedal", "cell_temperature_max", "cell_temperature_min", "cell_voltage_min", "gps_speed_valid")
for _n in XSIGNALS: STORE.setdefault(_n, {"1s": ("xsig_1s", _n, 1.0), "1m": ("xsig_1m", _n, 1.0)})

# column whitelist per table: the SQL compiler may only name these (nothing a user or a model typed ever reaches an SQL string)
TABLE_COLUMNS = {}
for _c, _tiers in STORE.items():
    for _t, (_tb, _col, _sc) in _tiers.items(): TABLE_COLUMNS.setdefault(_tb, set()).add(_col)
TABLE_COLUMNS["sig_1m"] |= {"batt_max", "flow_min"}; [TABLE_COLUMNS[t].add("ts") for t in list(TABLE_COLUMNS) if t != "vday"]; TABLE_COLUMNS["vday"].add("ts")
TIER_TABLE_PREF = {"1s": ("sig_1s", "gps_1s", "xsig_1s"), "1m": ("sig_1m", "ext_1m", "cam_1m", "xsig_1m")}

# ------------------------------------------------------------------------------------------------ events
EVENTS = {
    "ADAS_DISENGAGE": ("ADAS disengagement", ("adas disengagement", "adas disengagements", "lane assist disengagement", "lane keeping dropout", "driver assistance disengaged", "lane assist switching off", "lane assist switch off")),
    "WARN_BATT_OVERTEMP": ("battery over-temperature warning", ("overheating", "overheating warning", "overtemperature", "over temperature warning", "battery overheating", "battery got too hot", "thermal warning", "overheat")),
    "DTC_SET": ("diagnostic trouble code", ("diagnostic", "diagnostics", "diagnostic code", "diagnostic trouble code", "trouble code", "fault code", "dtc", "diagnostic event", "diagnostic events", "error code")),
    "HARSH_BRAKE": ("hard-braking event", ("hard braking", "harsh braking", "emergency stop", "hard brake", "braking hard", "heavy braking")), "HARSH_ACCEL": ("hard-acceleration event", ("hard acceleration", "harsh acceleration", "aggressive acceleration", "accelerating hard")),
    "HARSH_CORNER": ("hard-cornering event", ("hard cornering", "harsh cornering")), "TPMS_LOW": ("tyre-pressure warning", ("tyre warning", "tire warning", "low tyre pressure warning", "tpms")), "TYRE_INFLATED": ("tyre inflation", ("tyre inflated",)),
    "LOW_12V": ("low 12 V battery warning", ("low 12v", "low 12 v battery")), "SERVICE_DUE": ("service reminder", ("service due", "service reminder")), "CAMERA_BLOCKED": ("camera blockage", ("camera blocked", "camera blockage", "blocked camera")),
    "CAMERA_DEGRADED": ("camera degradation", ("camera degraded",)), "CHARGE_START": ("charging session", ("charging session", "charge session", "plugged in", "started charging")), "CAN_ANOMALY": ("CAN-bus anomaly", ("can anomaly", "can bus attack", "bus intrusion")),
    "ANY_WARNING": ("warning", ("warning", "warnings", "alert", "alerts", "fault", "faults")),
}
EVENTS.update({
    "ECU_RESET": ("ECU reset", ("ecu reset", "ecu resets", "ecu reboot", "controller reset", "module reset", "watchdog reset", "ecu restarted", "ecu rebooted")),
    "UNEXPECTED_SHUTDOWN": ("unexpected shutdown", ("unexpected shutdown", "unexpected shutdowns", "unexpected power loss", "sudden shutdown", "system shut down unexpectedly", "unexpected reboot", "vehicle shut down")),
    "VOLTAGE_DIP": ("supply voltage dip", ("voltage dip", "voltage dips", "voltage drop", "voltage fluctuation", "voltage fluctuations", "brown-out", "brownout", "supply dip", "power dip")),
    "SENSOR_CONFLICT": ("sensor conflict", ("sensor conflict", "sensor conflicts", "conflicting sensor", "conflicting values", "sources disagree", "sensor disagreement", "sensor discrepancy", "source discrepancy", "conflicting readings")),
    "AEB_ACTIVATION": ("emergency-braking activation", ("emergency braking activation", "aeb activation", "aeb triggered", "automatic emergency braking", "collision braking")),
    "CAN_ERROR": ("CAN communication error", ("can error", "can errors", "communication error", "communication errors", "bus error", "bus errors", "can communication fault", "network error")),
    "SENSOR_INVALID": ("invalid sensor reading", ("invalid sensor reading", "invalid reading", "implausible reading", "sensor out of range", "bad sensor value")),
    "ADAS_STATE_CHANGE": ("ADAS state change", ("adas state change", "adas status change", "adas switching", "driver assistance state change", "adas toggled")),
    "POWER_DERATE": ("power derate", ("power derate", "power derating", "power limitation", "reduced power", "reduced acceleration", "limp mode", "torque limit", "power limited", "derated")),
    "WARN_OVERTEMP": ("overtemperature warning", ("coolant overheating", "engine overheating", "overtemperature warning", "engine overtemperature", "coolant overtemperature")),
    "DTC_CLEAR": ("cleared diagnostic code", ("dtc cleared", "fault code cleared", "diagnostic code cleared", "diagnostic memory cleared", "fault memory cleared", "code was cleared", "codes were cleared", "code erased", "cleared the fault code", "cleared the codes")),
    "SW_UPDATE": ("software update", ("software update", "software updates", "firmware update", "ota update", "over the air update", "software was updated", "updated the software", "new software version", "software installed")),
    "SOURCE_SILENT": ("data source dropout", ("source dropout", "bus dropout", "bus went silent", "source went silent", "went silent", "lost communication with the bus", "ecu stopped transmitting", "signal source dropped out", "data source stopped", "communication dropout", "bus silent", "stopped sending")),
    "LOG_MESSAGE": ("service log message", ("log message", "log messages", "log line", "log lines", "log entry", "log entries", "service log", "service logs", "system log", "system logs", "qnx log", "qnx logs", "syslog", "error log", "error logs", "logs")),
})
EVENT_GROUPS = {"ANY_WARNING": ("WARN_BATT_OVERTEMP", "DTC_SET", "TPMS_LOW", "LOW_12V", "CAMERA_BLOCKED", "CAMERA_DEGRADED", "CAN_ANOMALY", "UNEXPECTED_SHUTDOWN", "VOLTAGE_DIP", "WARN_OVERTEMP", "ECU_RESET", "SENSOR_CONFLICT", "POWER_DERATE", "SOURCE_SILENT")}
# detector events: found in the signals themselves, not in the events table (implemented in query_executor)
DETECTED_EVENTS = {"COOLANT_FLOW_DROP": ("start of the coolant-flow drop", ("coolant flow started dropping", "coolant flow dropped", "coolant flow drop", "coolant flow fell", "coolant flow started falling", "coolant flow started to drop", "flow started dropping", "coolant flow fault", "coolant flow warning", "coolant flow problem", "coolant flow failure", "coolant flow alert", "low coolant flow event", "coolant fault", "cooling fault"))}
for _k, (_l, _s) in DETECTED_EVENTS.items(): EVENTS[_k] = (_l, _s)
EVENT_EXTRA = {"WARN_BATT_OVERTEMP": ["battery overheating warnings", "over-temperature warnings", "times the battery got too hot"], "DTC_SET": ["fault codes", "diagnostic trouble codes", "dtcs"], "HARSH_BRAKE": ["hard braking events", "harsh brakes", "emergency stops"],
                "HARSH_ACCEL": ["hard acceleration events", "harsh accelerations"], "ADAS_DISENGAGE": ["lane assist disengagements", "adas disengagements", "driver assistance dropouts"], "TPMS_LOW": ["tyre pressure warnings", "low tyre warnings"],
                "ANY_WARNING": ["warnings", "alerts", "faults"], "CAMERA_BLOCKED": ["camera blockages", "blocked camera events"], "CHARGE_START": ["charging sessions", "charges"], "LOW_12V": ["low 12 v battery warnings"],
                "COOLANT_FLOW_DROP": ["coolant flow drops", "drops in coolant flow"], "HARSH_CORNER": ["hard cornering events"], "CAN_ANOMALY": ["can bus anomalies"], "SERVICE_DUE": ["service reminders"]}
for _k, _extra in EVENT_EXTRA.items():
    if _k in EVENTS: EVENTS[_k] = (EVENTS[_k][0], tuple(EVENTS[_k][1]) + tuple(_extra))


# ------------------------------------------------------------------------------------------------ named conditions (sample level) and trip classes (entity level)
PREDICATES = {   # name -> (description, how it is evaluated: ("mode", level) | ("signal", name, op, value) | ("derived", name, op, value) | ("trip", column, op, value))
    "highway": ("on the motorway / highway", ("mode", 2)), "city": ("in city driving", ("mode", 1)), "charging": ("while charging", ("mode", 3)), "parked": ("while parked", ("mode", 0)), "driving": ("while driving", ("mode_in", (1, 2))),
    "fast_charging": ("while fast (DC) charging", ("signal", "battery_power", "<", -20.0)), "raining": ("while it was raining", ("signal", "rain", ">", 0)), "hard_braking": ("during hard braking", ("event", "HARSH_BRAKE", 10)),
    "hard_acceleration": ("during hard acceleration", ("event", "HARSH_ACCEL", 10)), "regenerating": ("while regenerating", ("signal", "battery_power", "<", -1.0, "driving")), "night": ("at night", ("hour_not_in", 6, 20)), "cold": ("in cold weather", ("signal", "ambient_temperature", "<", 5.0)),
    "hot_weather": ("in hot weather", ("signal", "ambient_temperature", ">", 28.0)), "high_speed": ("at high speed", ("signal", "speed", ">", 110.0)), "idling": ("while standing still", ("signal", "speed", "<", 1.0)),
}
PRED_SYN = {   # predicate -> the ways people say it (the parser and the training-data generator both read this; one vocabulary)
    "highway": ("highway driving", "motorway driving", "freeway driving", "highway", "motorway", "freeway", "expressway", "highways"), "city": ("city driving", "urban driving", "town driving", "city", "urban", "in town"),
    "charging": ("charging", "plugged in", "on charge", "charge session", "the charger", "on charger"), "parked": ("parked", "parking"), "driving": ("driving", "moving", "in motion", "on the move"),
    "fast_charging": ("fast charging", "dc charging", "dc fast charging", "rapid charging", "fast charge"), "raining": ("in the rain", "raining", "it rained", "it was raining", "rainy", "in rain", "wet weather", "it was wet", "while it rained"),
    "hard_braking": ("hard braking", "harsh braking", "hard brake", "braking hard", "heavy braking", "hard brakes"), "hard_acceleration": ("hard acceleration", "harsh acceleration", "aggressive acceleration", "accelerating hard"),
    "regenerating": ("regenerating", "regen active", "during regen", "when regen"), "night": ("at night", "night time", "nighttime", "after dark", "in the dark", "during the night", "the dark", "the night"),
    "cold": ("cold weather", "in the cold", "it was cold", "cold conditions", "when cold", "the cold"), "hot_weather": ("hot weather", "in the heat", "it was hot", "hot conditions", "on hot days", "the heat"),
    "high_speed": ("high speed", "high speeds", "at speed"), "idling": ("idling", "stationary", "standing still", "at a standstill", "stopped"),
}
QUALITY_COMPANIONS = {"steering_angle": ("yaw_rate", "speed"), "wheel_speed_fl": ("speed",), "wheel_speed_fr": ("speed",), "wheel_speed_rl": ("speed",), "wheel_speed_rr": ("speed",),
                      "motor_rpm": ("speed",), "motor_torque": ("speed",), "motor_power": ("speed",), "yaw_rate": ("speed",), "brake_pressure": ("speed",)}   # signals loaded next to a target so its quality can be judged (stuck while moving, bias while straight)
MODE_PREDICATES = ("highway", "city", "charging", "parked", "driving", "fast_charging", "idling")        # mutually exclusive: a newer one replaces an older one in a follow-up
EVENT_AS_PREDICATE = {"HARSH_BRAKE": "hard_braking", "HARSH_ACCEL": "hard_acceleration", "CHARGE_START": "charging"}
TRIP_CLASSES = {"highway": ("trips mostly on the motorway", "hwy_share", ">=", 0.5), "city": ("mostly city trips", "hwy_share", "<", 0.2), "rainy": ("trips in the rain", "rain_share", ">=", 0.3), "long": ("long trips", "km", ">=", 30.0), "short": ("short trips", "km", "<", 5.0)}
ENTITY_METRICS = {   # entity -> metric name -> (column in the entity table, unit, higher_is_worse). Whitelist for ranking / selection SQL.
    "trip": {"wh_per_km": ("wh_km", "Wh/km", True), "distance_driven": ("km", "km", None), "energy_used": ("kwh", "kWh", None), "duration": ("(end-start)", "s", None), "avg_speed": ("avg_speed", "km/h", None), "ambient_temperature": ("avg_amb", "C", None), "hvac_power": ("avg_hvac_w", "W", None), "recency": ("start", "s", None)},
    "charge_session": {"energy_used": ("kwh", "kWh", None), "charging_power": ("kw", "kW", None), "duration": ("(end-start)", "s", None), "recency": ("start", "s", None)}}
ENTITY_TABLE = {"trip": ("trips", "start", "end"), "charge_session": ("charges", "start", "end")}

# ------------------------------------------------------------------------------------------------ derived-metric dependency declarations (implementation: derived.py)
DERIVED_REQUIRES = {   # name -> ordered alternatives: (requires_concepts, requires_params, quality: "derived" exact physics | "estimated" uses a stated assumption, formula text)
    "motor_power_estimate": [], "acceleration": [(("speed",), (), "derived", "d(speed)/dt")], "acceleration_magnitude": [(("longitudinal_acceleration",), (), "derived", "abs(a_long)"), (("speed",), (), "derived", "abs(d(speed)/dt)")], "braking_intensity": [(("longitudinal_acceleration",), (), "derived", "max(0, -a_long) / g"), (("speed",), (), "derived", "max(0, -d(speed)/dt) / g")],
    "horsepower": [(("motor_power",), (), "derived", "kW x 1.341"), (("motor_torque", "motor_rpm"), (), "derived", "torque x rpm / 7120")],
    "turn_radius": [(("steering_angle",), ("wheelbase_m",), "derived", "R = L / tan(delta)  (delta = road-wheel angle = steering angle / steering ratio)"), (("speed", "yaw_rate"), (), "derived", "R = v / yaw rate"), (("speed", "lateral_acceleration"), (), "derived", "R = v^2 / a_lat")],
    "energy_used": [(("battery_power",), (), "derived", "integral of battery power dt (discharge only)")], "distance_driven": [(("speed",), (), "derived", "integral of speed dt")], "wh_per_km": [(("battery_power", "speed"), (), "derived", "energy / distance")],
    "regen_energy": [(("battery_power", "speed"), (), "derived", "integral of -battery power dt while moving and power < 0"), (("regen_power",), (), "derived", "integral of regen power dt")], "wheel_slip": [(("wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "speed"), (), "derived", "(wheel - vehicle) / vehicle")],
    "thermal_rise_rate": [(("battery_temperature",), (), "derived", "d(battery temperature)/dt")], "range_estimate": [(("battery_soc", "battery_power", "speed"), ("battery_capacity_kwh",), "estimated", "SoC x capacity / recent Wh/km (recent trips)")],
    "battery_efficiency": [(("motor_power", "battery_power"), (), "derived", "motor power / battery power")],
}
# motor_* concepts that can also be DERIVED when the vehicle does not record them directly
DERIVED_ALTERNATIVES = {
    "motor_rpm": [(("speed",), ("gear_ratio", "wheel_radius_m"), "estimated", "speed / (2 pi r) x gear ratio - nominal constants, not read from the vehicle")],
    "motor_power": [(("motor_torque", "motor_rpm"), (), "derived", "torque x rpm / 9549")],
    "motor_torque": [(("motor_power", "motor_rpm"), (), "derived", "power x 9549 / rpm")],
    "longitudinal_acceleration": [(("speed",), (), "derived", "d(speed)/dt")],
    "battery_power": [(("battery_voltage", "battery_current"), (), "derived", "V x I / 1000")],
}
# vehicle parameters: read from the meta table when present; the project's nominal constants are used only where a signal is explicitly marked "estimated"
PARAM_META = {"battery_capacity_kwh": "veh.cap_kwh", "wheelbase_m": "veh.wheelbase_m", "wheel_radius_m": "veh.wheel_radius_m", "gear_ratio": "veh.gear_ratio", "steering_ratio": "veh.steering_ratio", "drivetrain_efficiency": "veh.drivetrain_eff"}
NOMINAL_PARAMS = {"gear_ratio": 9.0, "wheel_radius_m": 0.33}          # the constants the existing project already uses for its (labelled) rpm estimate
AGG_NUMERIC = ("mean", "min", "max", "median", "std", "first", "last", "delta", "count", "p95", "sum")
AGG_BINARY = ("count", "percent_time")
AGG_CATEGORICAL = ("percent_time", "mode", "count")
AGG_EXTENSIVE = ("sum",)                                                   # sum is only meaningful for quantities that accumulate (energy, distance, regen energy, counts)
EXTENSIVE = {"energy_used", "distance_driven", "regen_energy"}


# ------------------------------------------------------------------------------------------------ the data profile of ONE database
class Profile:
    """What this database really contains. Built once per connection (a handful of cheap queries); everything 'is it available?' goes through here."""
    def __init__(self, conn):
        self.conn = conn; self.tables = {}; self.params = {}
        for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')"):
            try: cols = [r[1] for r in conn.execute(f"PRAGMA table_info({n})")]; self.tables[n] = {"cols": set(cols), "rows": None}
            except sqlite3.Error: pass
        try: self.meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        except sqlite3.Error: self.meta = {}
        for p, k in PARAM_META.items():
            try: self.params[p] = float(self.meta[k])
            except (KeyError, TypeError, ValueError): pass
        self._avail = {}

    def has_rows(self, table):
        t = self.tables.get(table)
        if t is None: return False
        if t["rows"] is not True:                                             # a live table that was empty may receive data later
            try: t["rows"] = bool(self.conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone())
            except sqlite3.Error: t["rows"] = False
        return t["rows"]

    def column_has_data(self, table, col):
        key = (table, col)
        if self._avail.get(key) is not True:                                  # cache presence, never permanent absence
            ok = False
            if self.has_rows(table) and col in self.tables[table]["cols"]:
                try: ok = self.conn.execute(f"SELECT 1 FROM {table} WHERE {col} IS NOT NULL LIMIT 1").fetchone() is not None
                except sqlite3.Error: ok = False
            self._avail[key] = ok
        return self._avail[key]

    def tiers_of(self, concept):
        """Storage tiers that really hold data for a measured concept: {"1s": (table, col, scale), ...}"""
        return {t: m for t, m in STORE.get(concept, {}).items() if self.column_has_data(m[0], m[1])}

    def param(self, name):
        """-> (value, provenance) where provenance is 'vehicle' (meta table) or 'nominal' (project constant) or None."""
        if name in self.params: return self.params[name], "vehicle"
        if name in NOMINAL_PARAMS: return NOMINAL_PARAMS[name], "nominal"
        return None, None

    def has_events(self, etype):
        if etype in DETECTED_EVENTS: return self.concept_available("coolant_flow")
        if etype == "ANY_WARNING": return self.has_events_group("ANY_WARNING")
        try: return self.conn.execute("SELECT 1 FROM events WHERE type=? LIMIT 1", (etype,)).fetchone() is not None
        except sqlite3.Error: return False

    def has_events_group(self, g):
        try:
            return self.conn.execute("SELECT 1 FROM events WHERE type IN (%s) LIMIT 1" % ",".join("?" * len(EVENT_GROUPS[g])), EVENT_GROUPS[g]).fetchone() is not None
        except sqlite3.Error: return False

    def event_table_exists(self): return "events" in self.tables

    # ---- availability of a concept: measured / derived / unavailable, with the way to get it
    def resolve(self, concept, _depth=0):
        """-> Resolution(status, how, tiers, requires, quality, assumptions, formula, missing). status in measured | derived | estimated | unavailable | unknown."""
        c = CONCEPTS.get(concept)
        if c is None: return Resolution("unknown", concept=concept)
        if _depth > 4: return Resolution("unavailable", concept=concept, missing=[concept])
        tiers = self.tiers_of(concept) if c.kind == "measured" or concept in DERIVED_ALTERNATIVES else {}
        if tiers: return Resolution("measured", concept=concept, tiers=tiers, quality="measured")
        if concept == "tyre_pressure":
            parts = [self.tiers_of(x) for x in ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr")]
            if all(parts): return Resolution("measured", concept=concept, tiers={"1m": ("ext_1m", "tp_*", 10.0)}, quality="measured")
        alts = DERIVED_REQUIRES.get(concept, []) + DERIVED_ALTERNATIVES.get(concept, []); best_missing = None
        for req, params, quality, formula in alts:
            missing = []; sub = {}
            for r in req:
                rr = self.resolve(r, _depth + 1)
                if rr.status in ("unavailable", "unknown"): missing.append(r)
                else: sub[r] = rr
            assumptions = []
            for p in params:
                v, prov = self.param(p)
                if v is None: missing.append(p)
                elif prov == "nominal": assumptions.append(f"{p} = {v:g} (project nominal value, not read from the vehicle)")
            if not missing:
                q = quality
                if any(s.status == "estimated" for s in sub.values()) or assumptions: q = "estimated"
                assumptions += [a for s in sub.values() for a in s.assumptions]
                if q == "estimated" and not assumptions: assumptions = [f"estimated by {formula}; it assumes recent driving conditions continue"]
                return Resolution("estimated" if q == "estimated" else "derived", concept=concept, requires=req, params=params, quality=q, assumptions=assumptions, formula=formula, sub=sub)
            if best_missing is None or len(missing) < len(best_missing): best_missing = missing
        return Resolution("unavailable", concept=concept, missing=best_missing or [concept], alternatives=[a for a in alts])

    def concept_available(self, concept): return self.resolve(concept).status in ("measured", "derived", "estimated")

    def available_concepts(self): return sorted(n for n in CONCEPTS if self.concept_available(n))
    def known_but_missing(self): return sorted(n for n in CONCEPTS if not self.concept_available(n))


@dataclass
class Resolution:
    status: str
    concept: str = ""
    tiers: dict = field(default_factory=dict)
    requires: tuple = ()
    params: tuple = ()
    quality: str = ""
    assumptions: list = field(default_factory=list)
    formula: str = ""
    sub: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)

    @property
    def available(self): return self.status in ("measured", "derived", "estimated")


# ------------------------------------------------------------------------------------------------ vocabulary (for the parsers and for the model's output grammar)
# known paraphrases beyond the base synonyms (shared with the training-data generator, training/phrases.py)
EXTRA_SYNONYMS = {
    "battery_temperature": ["pack temperature", "how hot the battery got", "battery heat", "temperature of the battery pack", "battery temp", "traction battery temperature", "pack heat level"],
    "battery_soc": ["state of charge", "charge level", "soc", "how much charge was left", "battery percentage", "battery level", "remaining charge"],
    "speed": ["vehicle speed", "speed", "how fast the car went", "driving speed", "velocity", "road speed"],
    "battery_power": ["battery power", "power draw", "power from the battery", "electrical power", "pack power"],
    "coolant_flow": ["coolant flow", "coolant flow rate", "cooling flow", "flow of the battery coolant", "pump flow", "cooling circuit flow"],
    "ambient_temperature": ["outside temperature", "ambient temperature", "air temperature outside", "external temperature", "temperature outside"],
    "hvac_power": ["climate control power", "hvac power", "heater power", "power used by the climate control", "cabin heating power", "ac power"],
    "motor_temperature": ["motor temperature", "drive motor temperature", "motor temp", "how hot the motor got", "electric motor temperature"],
    "motor_rpm": ["motor rpm", "motor speed", "rpm", "revs", "motor revolutions"],
    "motor_torque": ["motor torque", "torque", "drive torque", "torque at the motor", "wheel torque"],
    "motor_power": ["motor power", "mechanical power", "shaft power", "power at the motor"],
    "horsepower": ["horsepower", "hp", "how many horsepower", "motor horsepower"],
    "battery_voltage": ["battery voltage", "pack voltage", "traction voltage", "hv voltage", "high voltage battery voltage"],
    "battery_current": ["battery current", "pack current", "traction current", "current drawn from the battery"],
    "steering_angle": ["steering angle", "steering wheel angle", "how far the wheel was turned"], "yaw_rate": ["yaw rate", "rotation rate", "how fast the car rotated"],
    "lateral_acceleration": ["lateral acceleration", "sideways acceleration", "cornering acceleration", "side g"], "longitudinal_acceleration": ["longitudinal acceleration", "forward acceleration", "front-to-back acceleration"],
    "brake_pressure": ["brake pressure", "braking pressure", "pressure in the brakes"], "accelerator_position": ["accelerator position", "throttle position", "pedal position", "how far the accelerator was pressed"],
    "tyre_pressure": ["tyre pressure", "tire pressure", "pressure in the tyres", "tyre pressures"], "tyre_temperature": ["tyre temperature", "tire temperature", "how warm the tyres got"],
    "wheel_speed_fl": ["front left wheel speed", "front-left wheel speed"], "wheel_speed_rr": ["rear right wheel speed", "rear-right wheel speed"],
    "regen_power": ["regen power", "regenerative braking power", "regeneration power"], "charging_power": ["charging power", "charge rate", "charging rate", "charging speed", "power while charging"],
    "cell_temperature_max": ["hottest cell temperature", "maximum cell temperature", "temperature of the hottest cell"], "cell_voltage_min": ["lowest cell voltage", "weakest cell voltage", "minimum cell voltage"],
    "aux_battery_voltage": ["12 v battery voltage", "12v battery voltage", "auxiliary battery voltage"], "battery_soh": ["battery health", "state of health", "battery degradation", "soh"],
    "wheel_slip": ["wheel slip", "slip", "tyre slip"], "turn_radius": ["turn radius", "turning radius", "turning circle"], "braking_intensity": ["braking intensity", "braking force", "deceleration"],
    "range_estimate": ["remaining range", "estimated range", "driving range", "range left"], "wh_per_km": ["consumption", "energy consumption", "efficiency", "wh per km", "energy use per kilometre", "economy"],
    "energy_used": ["energy used", "electricity used", "kwh used", "energy consumed", "how much electricity", "electricity"], "distance_driven": ["distance driven", "distance covered", "kilometres driven", "kilometers driven", "km driven", "how many kilometres", "how many kilometers", "how many km"],
    "regen_energy": ["regen energy", "energy recovered by regeneration", "energy regenerated", "recuperated energy", "energy i got back from regenerative braking", "energy did regen recover", "energy back from regen", "energy regen recovered", "energy did i get back from regenerative braking"], "cabin_temperature": ["cabin temperature", "interior temperature", "temperature inside the car"],
    "thermal_rise_rate": ["temperature rise rate", "heating rate of the battery", "how fast the battery heated"], "lane_confidence": ["lane confidence", "lane marking confidence", "lane detection confidence"],
    "camera_blockage": ["camera blockage", "camera blockage level", "lens blockage"], "light": ["ambient light", "light level", "brightness outside"], "rain": ["rain", "rainfall"],
}

def _build_lexicon():
    lex = {}
    for c in CONCEPTS.values():
        for s in c.synonyms + (c.label,) + tuple(EXTRA_SYNONYMS.get(c.name, ())): lex.setdefault(s.lower(), set()).add(c.name)
    return lex


LEXICON = _build_lexicon()                                    # phrase -> {concept names}; ambiguity (more than one) is resolved by the parser with context, never silently


def concept_names(): return sorted(CONCEPTS)


def label_of(name):
    c = CONCEPTS.get(name)
    return c.label if c else EVENTS.get(name, (name,))[0] if name in EVENTS else name
