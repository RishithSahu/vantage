"""SAE J1979 / ISO 15031-5 OBD-II service 01 (current data) response decoding, listen-only (stdlib only).
A response arrives as an ISO-TP payload from 0x7E8..0x7EF:  0x41 <PID> <data bytes A B C D>.  Formulas are the standard J1979 ones."""

PIDS = {  # pid: (name, unit, nbytes, formula(A,B,C,D))
    0x04: ("engine_load", "%", 1, lambda a, b, c, d: a * 100 / 255), 0x05: ("coolant_temp", "C", 1, lambda a, b, c, d: a - 40),
    0x06: ("fuel_trim_st1", "%", 1, lambda a, b, c, d: (a - 128) * 100 / 128), 0x07: ("fuel_trim_lt1", "%", 1, lambda a, b, c, d: (a - 128) * 100 / 128),
    0x0A: ("fuel_pressure", "kPa", 1, lambda a, b, c, d: a * 3), 0x0B: ("intake_map", "kPa", 1, lambda a, b, c, d: a),
    0x0C: ("rpm", "rpm", 2, lambda a, b, c, d: (256 * a + b) / 4), 0x0D: ("speed", "km/h", 1, lambda a, b, c, d: a),
    0x0E: ("timing_advance", "deg", 1, lambda a, b, c, d: a / 2 - 64), 0x0F: ("intake_air_temp", "C", 1, lambda a, b, c, d: a - 40),
    0x10: ("maf", "g/s", 2, lambda a, b, c, d: (256 * a + b) / 100), 0x11: ("throttle", "%", 1, lambda a, b, c, d: a * 100 / 255),
    0x1F: ("run_time", "s", 2, lambda a, b, c, d: 256 * a + b), 0x21: ("distance_with_mil", "km", 2, lambda a, b, c, d: 256 * a + b),
    0x2F: ("fuel_level", "%", 1, lambda a, b, c, d: a * 100 / 255), 0x31: ("distance_since_clear", "km", 2, lambda a, b, c, d: 256 * a + b),
    0x33: ("barometric", "kPa", 1, lambda a, b, c, d: a), 0x42: ("module_voltage", "V", 2, lambda a, b, c, d: (256 * a + b) / 1000),
    0x46: ("ambient_temp", "C", 1, lambda a, b, c, d: a - 40), 0x5C: ("oil_temp", "C", 1, lambda a, b, c, d: a - 40), 0x5E: ("fuel_rate", "L/h", 2, lambda a, b, c, d: (256 * a + b) / 20),
}


def decode(payload):
    """ISO-TP payload of a mode-01 response -> {"pid","name","unit","value"} or None. A negative response (7F 01 NRC) returns {"error": nrc}."""
    if len(payload) >= 3 and payload[0] == 0x7F and payload[1] == 0x01: return {"error": payload[2]}
    if len(payload) < 3 or payload[0] != 0x41: return None
    pid = payload[1]; spec = PIDS.get(pid)
    if spec is None: return {"pid": pid, "name": f"pid_{pid:02X}", "unit": "", "value": None, "raw": bytes(payload[2:])}
    name, unit, n, f = spec
    if len(payload) != 2 + n: return None
    data = list(payload[2:2 + n]) + [0] * 4
    return {"pid": pid, "name": name, "unit": unit, "value": f(*data[:4])}


def encode(pid, value):
    """Inverse of the single-/two-byte formulas (tests / simulator): returns the 0x41 response payload."""
    name, unit, n, f = PIDS[pid]
    if n == 1: a = max(0, min(255, round(_inv1(pid, value)))); return bytes([0x41, pid, a])
    raw = max(0, min(65535, round(_inv2(pid, value)))); return bytes([0x41, pid, raw >> 8, raw & 0xFF])


def _inv1(pid, v):
    return {0x04: v * 255 / 100, 0x05: v + 40, 0x06: v * 128 / 100 + 128, 0x07: v * 128 / 100 + 128, 0x0A: v / 3, 0x0B: v, 0x0D: v, 0x0E: (v + 64) * 2, 0x0F: v + 40, 0x11: v * 255 / 100,
            0x2F: v * 255 / 100, 0x33: v, 0x46: v + 40, 0x5C: v + 40}[pid]


def _inv2(pid, v):
    return {0x0C: v * 4, 0x10: v * 100, 0x1F: v, 0x21: v, 0x31: v, 0x42: v * 1000, 0x5E: v * 20}[pid]
