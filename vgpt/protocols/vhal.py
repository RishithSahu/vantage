"""Android Vehicle HAL (VHAL) property decoding (stdlib only). Property ids, value types and units come from vhal_props.json, which is GENERATED from the
real AOSP VehicleProperty.aidl (tools_gen_vhal.py) - nothing is typed from memory.

Input is the VehiclePropValue parcelable as a client sees it from the Binder interface (IVehicle / VehicleHal callbacks):
    {"prop": 0x11600207, "area": 0, "timestamp": ns_since_boot, "status": 0, "int32": [..], "float": [..], "int64": [..], "bytes": b"", "string": ""}
(The Binder transport itself is out of scope; on a real Android Automotive head unit a small client service subscribes with SubscribeOptions and forwards these.)"""
import json, math, os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vhal_props.json")
UNIT_TO = {"METER_PER_SEC": ("km/h", 3.6), "MILLIWATTS": ("kW", 1e-6), "WATT_HOUR": ("kWh", 1e-3), "METER": ("km", 1e-3), "KILOMETER": ("km", 1.0), "CELSIUS": ("C", 1.0),
           "PERCENTAGE": ("%", 1.0), "VOLT": ("V", 1.0), "MILLIAMPERE": ("A", 1e-3)}
# the subset the assistant stores, mapped to its signal names (everything else is parsed, typed, and counted but not stored)
SIGNALS = {"PERF_VEHICLE_SPEED": "speed", "PERF_ODOMETER": "odo_km", "ENV_OUTSIDE_TEMPERATURE": "amb_temp", "EV_BATTERY_LEVEL": "batt_kwh", "RANGE_REMAINING": "range_km",
           "EV_BATTERY_INSTANTANEOUS_CHARGE_RATE": "chg_kw", "HVAC_TEMPERATURE_CURRENT": "cabin_temp", "ENGINE_COOLANT_TEMP": "engine_coolant_c"}


class Vhal:
    def __init__(self, path=_PATH):
        d = json.load(open(path))
        self.props = d["properties"]; self.enums = d["enums"]
        self.by_id = {p["id"]: n for n, p in self.props.items()}
        self.unknown = 0

    def decode(self, v):
        """VehiclePropValue dict -> list of ("sig", name, value) / ("state", name, text); unknown ids are counted, never fatal."""
        if not isinstance(v, dict): return []
        # VehiclePropertyStatus.AVAILABLE is zero.  A value with any other
        # status is not evidence and must never update derived vehicle state.
        if v.get("status") != 0: return []
        pid = v.get("prop"); name = self.by_id.get(pid)
        if name is None: self.unknown += 1; return []
        p = self.props[name]; t = p["type"]
        try:
            raw = (v.get("float") or [None])[0] if t == "FLOAT" else (v.get("int32") or [None])[0] if t in ("INT32", "BOOLEAN") else (v.get("int64") or [None])[0] if t == "INT64" else None
        except (TypeError, IndexError): raw = None
        if raw is None: return []
        if name == "IGNITION_STATE":
            inv = {n: x for x, n in ((x, n) for n, x in self.enums["VehicleIgnitionState"].items())}
            state = next((n for n, x in self.enums["VehicleIgnitionState"].items() if x == raw), None)
            if state is None: return []
            return [("sig", "ign", 1.0 if state in ("ON", "START") else 0.0), ("state", "ignition", state)]
        if name == "EV_CHARGE_STATE":
            state = next((n for n, x in self.enums["EvChargeState"].items() if x == raw), None)
            if state is None: return []
            return [("sig", "chg_state", 1.0 if state == "CHARGING" else 0.0), ("state", "charge_state", state)]
        sig = SIGNALS.get(name)
        if sig is None: return []
        unit, k = UNIT_TO.get(p["unit"], (p["unit"], 1.0))
        value = float(raw) * k
        if not math.isfinite(value): return []
        return [("sig", sig, value)]

    def encode(self, name, value, area=0, ts=0):
        """Build the parcelable for a property by name (tests / simulator). `value` is in the property's NATIVE unit."""
        p = self.props[name]; v = {"prop": p["id"], "area": area, "timestamp": ts, "status": 0, "int32": [], "float": [], "int64": [], "bytes": b"", "string": ""}
        if p["type"] == "FLOAT": v["float"] = [float(value)]
        elif p["type"] in ("INT32", "BOOLEAN"): v["int32"] = [int(value)]
        elif p["type"] == "INT64": v["int64"] = [int(value)]
        return v
