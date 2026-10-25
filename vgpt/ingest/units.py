"""Units as they are written in the wild -> a physical class and a conversion to the unit the registry uses.

A CSV header says "Speed (OBD)(km/h)", a JSON key says `speed_mph`, a DBC says `unit "m/s"`, a VHAL property is metres per second, a GNSS sentence is in knots, a J1939 SPN is 1/256 km/h per bit.  This module
turns any of those spellings into (class, factor, offset) so that value_in_registry_unit = value * factor + offset, and says which class a value belongs to, so that a name that matches two quantities can be
decided by its unit (a "temperature" in kPa is not a temperature).  Pure data and arithmetic."""
import re

# class -> {spelling (normalised) -> (factor to the class's registry unit, offset)}
TABLE = {
    "speed": {"km/h": (1, 0), "kph": (1, 0), "kmh": (1, 0), "kmph": (1, 0), "km/hr": (1, 0), "kilometersperhour": (1, 0), "m/s": (3.6, 0), "mps": (3.6, 0), "ms-1": (3.6, 0), "mph": (1.609344, 0), "mi/h": (1.609344, 0), "milesperhour": (1.609344, 0),
              "kn": (1.852, 0), "kt": (1.852, 0), "kts": (1.852, 0), "knot": (1.852, 0), "knots": (1.852, 0), "cm/s": (0.036, 0), "mm/s": (0.0036, 0)},
    "temperature": {"c": (1, 0), "degc": (1, 0), "celsius": (1, 0), "degreec": (1, 0), "f": (5 / 9, -160 / 9), "degf": (5 / 9, -160 / 9), "fahrenheit": (5 / 9, -160 / 9), "k": (1, -273.15), "kelvin": (1, -273.15), "mk": (0.001, -273.15)},
    "pressure": {"kpa": (1, 0), "pa": (0.001, 0), "hpa": (0.1, 0), "mbar": (0.1, 0), "millibar": (0.1, 0), "bar": (100, 0), "mpa": (1000, 0), "psi": (6.894757, 0), "inhg": (3.386389, 0), "mmhg": (0.133322, 0), "atm": (101.325, 0), "kgf/cm2": (98.0665, 0)},
    "voltage": {"v": (1, 0), "volt": (1, 0), "volts": (1, 0), "mv": (0.001, 0), "kv": (1000, 0)},
    "current": {"a": (1, 0), "amp": (1, 0), "amps": (1, 0), "ma": (0.001, 0), "ka": (1000, 0)},
    "power": {"kw": (1, 0), "w": (0.001, 0), "watt": (0.001, 0), "mw": (1000, 0), "megawatt": (1000, 0), "hp": (0.7456999, 0), "ps": (0.7354988, 0), "bhp": (0.7456999, 0), "cv": (0.7354988, 0)},
    "energy": {"kwh": (1, 0), "wh": (0.001, 0), "mwh": (1000, 0), "j": (1 / 3.6e6, 0), "kj": (1 / 3600.0, 0), "mj": (1 / 3.6, 0)},
    "distance": {"km": (1, 0), "m": (0.001, 0), "mi": (1.609344, 0), "miles": (1.609344, 0), "ft": (0.0003048, 0), "cm": (0.00001, 0), "mm": (0.000001, 0), "dm": (0.0001, 0)},
    "rpm": {"rpm": (1, 0), "r/min": (1, 0), "1/min": (1, 0), "min-1": (1, 0), "rps": (60, 0), "hz": (60, 0)},
    "torque": {"nm": (1, 0), "n.m": (1, 0), "knm": (1000, 0), "lbft": (1.3558179, 0), "lb-ft": (1.3558179, 0)},
    "angle": {"deg": (1, 0), "degree": (1, 0), "degrees": (1, 0), "rad": (57.29577951, 0), "mrad": (0.05729578, 0), "mdeg": (0.001, 0)},
    "angular_rate": {"deg/s": (1, 0), "dps": (1, 0), "rad/s": (57.29577951, 0), "deg/ms": (1000, 0)},
    "acceleration": {"m/s2": (1, 0), "m/s^2": (1, 0), "mps2": (1, 0), "g": (9.80665, 0), "mg": (0.00980665, 0), "km/h/s": (0.2777778, 0), "cm/s2": (0.01, 0)},
    "flow": {"l/min": (1, 0), "lpm": (1, 0), "l/s": (60, 0), "m3/h": (16.666667, 0), "gpm": (3.785411784, 0), "ml/min": (0.001, 0), "l/h": (1 / 60.0, 0)},
    "fuel_flow": {"l/h": (1, 0), "l/hr": (1, 0), "lph": (1, 0), "gal/hr": (3.785411784, 0), "us gal/h": (3.785411784, 0), "ml/s": (3.6, 0), "ml/min": (0.06, 0), "gal/h": (3.785411784, 0), "gph": (3.785411784, 0), "l/min": (60, 0), "cc/min": (0.06, 0), "ml/h": (0.001, 0), "kg/h": (1.19, 0)},
    "mass_flow": {"g/s": (1, 0), "kg/h": (0.2777778, 0), "lb/min": (7.559873, 0), "mg/s": (0.001, 0), "kg/s": (1000, 0)},
    "percent": {"%": (1, 0), "pct": (1, 0), "percent": (1, 0), "perc": (1, 0), "permille": (0.1, 0), "‰": (0.1, 0)},
    "time": {"s": (1, 0), "sec": (1, 0), "ms": (0.001, 0), "min": (60, 0), "h": (3600, 0), "hr": (3600, 0), "hours": (3600, 0), "us": (0.000001, 0)},
    "illuminance": {"lux": (1, 0), "lx": (1, 0), "klx": (1000, 0)},
    "length": {"mm": (1, 0), "in": (25.4, 0), "cm": (10, 0)},
}
REGISTRY_UNIT = {"speed": "km/h", "temperature": "C", "pressure": "kPa", "voltage": "V", "current": "A", "power": "kW", "energy": "kWh", "distance": "km", "rpm": "rpm", "torque": "Nm", "angle": "deg", "angular_rate": "deg/s",
                 "acceleration": "m/s2", "flow": "L/min", "fuel_flow": "L/h", "mass_flow": "g/s", "percent": "%", "time": "s", "illuminance": "lux", "length": "mm"}
# the registry's own unit strings -> class (for concepts)
OF_REGISTRY = {"km/h": "speed", "C": "temperature", "kPa": "pressure", "bar": "pressure", "V": "voltage", "A": "current", "kW": "power", "W": "power", "hp": "power", "kWh": "energy", "km": "distance", "m": "distance", "rpm": "rpm",
               "Nm": "torque", "deg": "angle", "deg/s": "angular_rate", "m/s2": "acceleration", "g": "acceleration", "L/min": "flow", "L/h": "fuel_flow", "g/s": "mass_flow", "%": "percent", "s": "time", "min": "time", "h": "time",
               "lux": "illuminance", "mm": "length"}
_AMBIGUOUS = {"min"}                                           # 'min' as a unit is minutes only when nothing says otherwise


def normalise(text):
    """'Speed (OBD)(km/h)' keeps its unit part; '°C' -> 'degc'; ' m / s^2 ' -> 'm/s^2'."""
    s = str(text).strip().lower().replace("°", "deg").replace("µ", "u").replace("²", "2").replace("·", ".").replace("Ω", "ohm")
    s = re.sub(r"\s+", "", s); s = s.replace("per", "/") if "per" in s and "/" not in s and s not in ("percent", "perc") else s
    return s.strip("[]()")


def find(unit):
    """-> (class, factor, offset) for a unit spelling, or None."""
    n = normalise(unit)
    if not n: return None
    for cls, table in TABLE.items():
        if n in table: return (cls,) + table[n]
    return None


def split_header(name):
    """('Engine RPM(rpm)') -> ('Engine RPM', 'rpm'); ('speed_kph') -> ('speed', 'kph'); ('Coolant [degC]') -> ('Coolant', 'degC'); no unit -> (name, None)."""
    s = str(name).strip()
    m = re.match(r"^(.*?)[\s_]*[\[(]\s*([^\])]{1,12})\s*[\])]\s*$", s)
    if m and find(m.group(2)): return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^(.*?)[\s_.\-]+([A-Za-z/%°]{1,6}\d?)$", s)
    if m and find(m.group(2)) and not (m.group(2).lower() in _AMBIGUOUS): return m.group(1).strip(), m.group(2)
    return s, None


def convert_to_registry(value, unit):
    """value in `unit` -> (value in the registry unit of its class, class) or None."""
    f = find(unit)
    if f is None: return None
    cls, k, o = f; return value * k + o, cls


def factor_between(unit_from, registry_unit):
    """(factor, offset) taking `unit_from` to the registry unit `registry_unit`, or None when they are different classes."""
    f = find(unit_from); cls = OF_REGISTRY.get(registry_unit)
    if f is None or cls is None or f[0] != cls: return None
    k, o = f[1], f[2]; r = TABLE[cls].get(normalise(registry_unit)) if normalise(registry_unit) in TABLE[cls] else None
    if r is None: return k, o
    rk, ro = r
    return k / rk, (o - ro) / rk
