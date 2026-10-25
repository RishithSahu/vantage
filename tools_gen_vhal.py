"""DEV tool: build vgpt/protocols/vhal_props.json from the REAL AOSP files (VehicleProperty.aidl, VehicleIgnitionState.aidl, EvChargeState.aidl, VehicleUnit.aidl).
Property ids are COMPUTED from the AIDL expressions (nothing typed from memory); units/access/change mode come from each property's doc comment.
Usage: python tools_gen_vhal.py   (reads data/ref/*.aidl, downloaded from android.googlesource.com / platform/hardware/interfaces)"""
import json, os, re

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ref")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgpt", "protocols", "vhal_props.json")
TYPES = {0x00100000: "STRING", 0x00200000: "BOOLEAN", 0x00300000: "INT32", 0x00400000: "INT32", 0x00410000: "INT32_VEC", 0x00500000: "INT64", 0x00510000: "INT64_VEC",
         0x00600000: "FLOAT", 0x00610000: "FLOAT_VEC", 0x00700000: "BYTES", 0x00E00000: "MIXED"}
AREAS = {0x01000000: "GLOBAL", 0x03000000: "WINDOW", 0x04000000: "MIRROR", 0x05000000: "SEAT", 0x06000000: "DOOR", 0x07000000: "WHEEL"}


def enum_values(name):
    """AIDL enum body -> {NAME: value}; entries without '= n' continue from the previous value (AIDL auto-numbering)."""
    t = open(os.path.join(REF, name + ".aidl"), encoding="utf-8").read()
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S); t = re.sub(r"//[^\n]*", "", t)
    body = t[t.index("{") + 1:t.rindex("}")]; out, nxt = {}, 0
    for part in body.split(","):
        m = re.match(r"\s*([A-Z0-9_]+)\s*(?:=\s*(0x[0-9A-Fa-f]+|-?\d+))?\s*$", part.strip())
        if not m: continue
        v = int(m[2], 0) if m[2] else nxt; out[m[1]] = v; nxt = v + 1
    return out


text = open(os.path.join(REF, "VehicleProperty.aidl"), encoding="utf-8").read()
props = {}
for m in re.finditer(r"/\*\*(.*?)\*/\s*([A-Z0-9_]+)\s*=\s*((?:[^,/]|//[^\n]*\n)*?),", text, re.S):
    doc, name, expr = m[1], m[2], re.sub(r"//[^\n]*", "", m[3])
    ids = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", expr)]
    if not ids: continue
    pid = sum(ids)
    unit = re.search(r"@unit\s+VehicleUnit\.(\w+)", doc); access = re.search(r"@access\s+VehiclePropertyAccess\.(\w+)", doc); chg = re.search(r"@change_mode\s+VehiclePropertyChangeMode\.(\w+)", doc)
    props[name] = {"id": pid, "type": TYPES.get(pid & 0x00FF0000, "OTHER"), "area": AREAS.get(pid & 0x0F000000, "OTHER"), "unit": unit[1] if unit else None,
                   "access": access[1] if access else None, "change": chg[1] if chg else None}
out = {"source": "AOSP platform/hardware/interfaces automotive/vehicle/aidl_property (main)", "properties": props,
       "enums": {"VehicleIgnitionState": enum_values("VehicleIgnitionState"), "EvChargeState": enum_values("EvChargeState"), "VehicleUnit": enum_values("VehicleUnit")}}
json.dump(out, open(OUT, "w"), indent=0, separators=(",", ":"))
print(f"{len(props)} properties -> {OUT} ({os.path.getsize(OUT) // 1024} KB)")
for k in ("PERF_VEHICLE_SPEED", "PERF_ODOMETER", "IGNITION_STATE", "ENV_OUTSIDE_TEMPERATURE", "EV_BATTERY_LEVEL", "EV_CHARGE_STATE", "EV_BATTERY_INSTANTANEOUS_CHARGE_RATE", "RANGE_REMAINING"):
    p = props.get(k); print(f"  {k:40s} 0x{p['id']:08X} {p['type']:9s} {p['area']:7s} unit={p['unit']}" if p else f"  {k}: MISSING")
print("enums:", {k: v for k, v in out["enums"].items() if k != "VehicleUnit"})
