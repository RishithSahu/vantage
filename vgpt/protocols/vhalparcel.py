"""A VehiclePropValue as bytes: the layout an Android Vehicle HAL client reads from the Binder interface (IVehicle / IVehicleCallback), written the way an AIDL parcelable is written.

    int32 size of the whole parcelable | int64 timestamp (ns since boot) | int32 areaId | int32 prop | int32 status | RawPropValues: int32 size, int32[] int32Values, float[] floatValues, int64[] int64Values,
    byte[] byteValues (padded to 4), String stringValue (UTF-16 with its length in characters, NUL-terminated, padded to 4)       - arrays are an int32 count followed by the elements, little endian

This is the structure of the AOSP definition (VehiclePropValue.aidl, RawPropValues.aidl) and it round-trips here; it has NOT been compared byte for byte with a parcel written by an Android device, so a real
head unit's service would hand Vantage the same fields through CarPropertyManager instead of through this encoding (docs/REAL_DATA_PATHS.md).  Stdlib only."""
import struct

TYPE_MASK, GROUP_MASK, AREA_MASK = 0x00FF0000, 0xF0000000, 0x0F000000
TYPES = {0x00100000, 0x00200000, 0x00400000, 0x00410000, 0x00500000, 0x00510000, 0x00600000, 0x00610000, 0x00700000, 0x00E00000}
TYPE_BITS = {t & TYPE_MASK for t in TYPES}
GROUPS = {0x10000000, 0x20000000, 0x30000000}
AREAS = {0x01000000, 0x02000000, 0x03000000, 0x04000000, 0x05000000, 0x06000000, 0x07000000, 0x08000000}
MAX_ARRAY = 4096


def _pad4(n): return (4 - n % 4) % 4


def pack(v):
    """{"prop", "area", "timestamp", "status", "int32": [], "float": [], "int64": [], "bytes": b"", "string": ""} -> bytes."""
    i32, fl, i64 = list(v.get("int32") or []), list(v.get("float") or []), list(v.get("int64") or []); by = bytes(v.get("bytes") or b""); s = str(v.get("string") or "")
    raw = struct.pack("<i", len(i32)) + struct.pack(f"<{len(i32)}i", *i32) + struct.pack("<i", len(fl)) + struct.pack(f"<{len(fl)}f", *fl) + struct.pack("<i", len(i64)) + struct.pack(f"<{len(i64)}q", *i64)
    raw += struct.pack("<i", len(by)) + by + b"\0" * _pad4(len(by))
    u = s.encode("utf-16-le"); raw += struct.pack("<i", len(s)) + u + b"\0\0" + b"\0" * _pad4(len(u) + 2)
    inner = struct.pack("<i", 4 + len(raw)) + raw
    prop = int(v["prop"]) & 0xFFFFFFFF
    head = struct.pack("<qiii", int(v.get("timestamp", 0)), int(v.get("area", 0)), prop - 2 ** 32 if prop >= 2 ** 31 else prop, int(v.get("status", 0)))
    body = head + inner
    return struct.pack("<i", 4 + len(body)) + body


def _take(b, o, n):
    if o + n > len(b): raise ValueError("a VehiclePropValue parcel ends early")
    return b[o:o + n], o + n


def unpack(b):
    """bytes -> the dict form.  Raises ValueError for anything that is not a well-formed parcel (the receiver then treats the payload as something else)."""
    b = bytes(b)
    if len(b) < 4 + 8 + 12 + 8: raise ValueError("too short for a VehiclePropValue")
    size = struct.unpack_from("<i", b, 0)[0]
    if size != len(b): raise ValueError("the parcel size does not match")
    ts, area, prop, status = struct.unpack_from("<qiii", b, 4); o = 4 + 20
    isize = struct.unpack_from("<i", b, o)[0]
    if isize != len(b) - o: raise ValueError("the RawPropValues size does not match")
    o += 4; out = {"prop": prop & 0xFFFFFFFF, "area": area, "timestamp": ts, "status": status}
    for key, fmt, w in (("int32", "i", 4), ("float", "f", 4), ("int64", "q", 8)):
        n = struct.unpack_from("<i", b, o)[0]; o += 4
        if not 0 <= n <= MAX_ARRAY: raise ValueError("array length out of range")
        raw, o = _take(b, o, n * w); out[key] = list(struct.unpack(f"<{n}{fmt}", raw))
    n = struct.unpack_from("<i", b, o)[0]; o += 4
    if not 0 <= n <= MAX_ARRAY * 8: raise ValueError("byte array length out of range")
    raw, o = _take(b, o, n); out["bytes"] = raw; o += _pad4(n)
    n = struct.unpack_from("<i", b, o)[0]; o += 4
    if not 0 <= n <= MAX_ARRAY: raise ValueError("string length out of range")
    raw, o = _take(b, o, 2 * n); out["string"] = raw.decode("utf-16-le", errors="replace")
    return out


def looks_like(b):
    """A cheap, strict test used by the sniffer: the sizes agree and the property id has the structure of an Android one (group, area and type bits)."""
    b = bytes(b)
    if len(b) < 40 or len(b) > 1 << 16: return False
    try:
        if struct.unpack_from("<i", b, 0)[0] != len(b): return False
        ts, area, prop, status = struct.unpack_from("<qiii", b, 4)
        p = prop & 0xFFFFFFFF
        if (p & GROUP_MASK) not in GROUPS or (p & AREA_MASK) not in AREAS or (p & TYPE_MASK) not in TYPE_BITS: return False
        if not 0 <= status <= 2 or ts < 0: return False
        return struct.unpack_from("<i", b, 24)[0] == len(b) - 24
    except struct.error: return False
