"""Diagnostics over IP, ISO 13400-2 (DoIP), listen-only.

Automotive Ethernet carries the same UDS requests as CAN does, wrapped in a DoIP frame: a generic header (protocol version, its bitwise inverse, a 16-bit payload type, a 32-bit payload length) and a payload.
This module only READS what a tester and an entity say to each other (the assistant never transmits); `encode` exists for the simulator and the tests.

    decode(frame)        one complete frame (a UDP datagram, or one frame cut from a TCP stream) -> {"type", "name", "version", ...fields...}  or raises DoipError
    Stream()             .feed(bytes) -> [frames' dicts]   a TCP byte stream cut at arbitrary places; a bad header is counted, skipped one byte at a time, and never fatal;  .errors
    diagnostic(frame)    -> (source address, target address, UDS bytes) when the frame carries a diagnostic message, else None - the UDS bytes then go through protocols/uds.py like those of an ISO-TP frame
    encode(name, **f)    -> frame bytes

Payload types: 0x0000 generic negative acknowledgement, 0x0001-0x0004 vehicle identification request / with EID / with VIN / announcement-response, 0x0005 / 0x0006 routing activation request / response,
0x0007 / 0x0008 alive check, 0x4001-0x4004 entity status and diagnostic power mode, 0x8001 diagnostic message, 0x8002 / 0x8003 its positive / negative acknowledgement.  Protocol versions 0x01 (2010),
0x02 (2012), 0x03 (2019) and 0xFF (default, used in a vehicle identification request) are accepted."""
import struct

HEADER = 8
MAX_PAYLOAD = 1 << 20                      # 1 MB: far above any diagnostic message; bounds the memory a hostile length can claim
VERSIONS = (0x01, 0x02, 0x03, 0xFF)
NAMES = {0x0000: "generic_nack", 0x0001: "vehicle_identification_request", 0x0002: "vehicle_identification_request_eid", 0x0003: "vehicle_identification_request_vin", 0x0004: "vehicle_announcement",
         0x0005: "routing_activation_request", 0x0006: "routing_activation_response", 0x0007: "alive_check_request", 0x0008: "alive_check_response", 0x4001: "entity_status_request",
         0x4002: "entity_status_response", 0x4003: "power_mode_request", 0x4004: "power_mode_response", 0x8001: "diagnostic_message", 0x8002: "diagnostic_ack", 0x8003: "diagnostic_nack"}
TYPES = {v: k for k, v in NAMES.items()}


class DoipError(ValueError): pass


def _u16(b, o): return (b[o] << 8) | b[o + 1]


def decode(frame):
    f = bytes(frame)
    if len(f) < HEADER: raise DoipError("short header")
    ver, inv, ptype, n = struct.unpack(">BBHI", f[:HEADER])
    if inv != (ver ^ 0xFF) or ver not in VERSIONS: raise DoipError("bad version / inverse")
    if n > MAX_PAYLOAD: raise DoipError("payload too long")
    if len(f) != HEADER + n: raise DoipError("length does not match the frame")
    p = f[HEADER:]; out = {"version": ver, "type": ptype, "name": NAMES.get(ptype, "reserved")}
    try:
        if ptype == 0x0000: out["code"] = p[0]
        elif ptype == 0x0003: out["vin"] = p[:17].decode("ascii", "replace")
        elif ptype == 0x0002: out["eid"] = p[:6].hex()
        elif ptype == 0x0004:
            if len(p) not in (32, 33): raise DoipError("announcement length")
            out.update(vin=p[:17].decode("ascii", "replace"), logical_address=_u16(p, 17), eid=p[19:25].hex(), gid=p[25:31].hex(), further_action=p[31])
            if len(p) == 33: out["sync_status"] = p[32]
        elif ptype == 0x0005:
            if len(p) not in (7, 11): raise DoipError("routing activation request length")
            out.update(source_address=_u16(p, 0), activation_type=p[2], reserved=int.from_bytes(p[3:7], "big"))
            if len(p) == 11: out["oem"] = p[7:11].hex()
        elif ptype == 0x0006:
            if len(p) not in (9, 13): raise DoipError("routing activation response length")
            out.update(tester_address=_u16(p, 0), entity_address=_u16(p, 2), response_code=p[4], reserved=int.from_bytes(p[5:9], "big"))
            if len(p) == 13: out["oem"] = p[9:13].hex()
        elif ptype == 0x4002:
            if len(p) not in (3, 7): raise DoipError("entity status length")
            out.update(node_type=p[0], max_sockets=p[1], open_sockets=p[2])
            if len(p) == 7: out["max_data_size"] = int.from_bytes(p[3:7], "big")
        elif ptype == 0x4004: out["power_mode"] = p[0]
        elif ptype in (0x8001, 0x8002, 0x8003):
            if len(p) < 4 or (ptype == 0x8001 and len(p) < 5): raise DoipError("diagnostic message length")
            out.update(source_address=_u16(p, 0), target_address=_u16(p, 2))
            if ptype == 0x8001: out["data"] = bytes(p[4:])
            else: out["code"] = p[4] if len(p) > 4 else None; out["previous"] = bytes(p[5:])
        elif ptype == 0x0007: pass
        elif ptype == 0x0008:
            if len(p) != 2: raise DoipError("alive check response length")
            out["source_address"] = _u16(p, 0)
    except IndexError: raise DoipError("payload shorter than its type needs")
    return out


def looks_like_frame(block):
    try: decode(block); return True
    except DoipError: return False


def diagnostic(frame):
    d = frame if isinstance(frame, dict) else decode(frame)
    return (d["source_address"], d["target_address"], d["data"]) if d.get("type") == 0x8001 else None


class Stream:
    """DoIP over TCP: frames arrive cut and glued at any place."""

    def __init__(self, max_buffer=2 * MAX_PAYLOAD):
        self.buf = bytearray(); self.errors = 0; self.max_buffer = max_buffer

    def feed(self, data):
        self.buf += data; out = []
        while len(self.buf) >= HEADER:
            ver, inv, ptype, n = struct.unpack(">BBHI", bytes(self.buf[:HEADER]))
            if inv != (ver ^ 0xFF) or ver not in VERSIONS or n > MAX_PAYLOAD: self.errors += 1; del self.buf[0]; continue        # not a header: look for one a byte further
            if len(self.buf) < HEADER + n: break
            frame = bytes(self.buf[:HEADER + n]); del self.buf[:HEADER + n]
            try: out.append(decode(frame))
            except DoipError: self.errors += 1
        if len(self.buf) > self.max_buffer: self.errors += 1; self.buf.clear()
        return out


def encode(name, version=0x02, **f):
    t = TYPES[name]; g = f.get
    if t == 0x0000: p = bytes([g("code", 0)])
    elif t == 0x0001: p = b""
    elif t == 0x0002: p = bytes.fromhex(g("eid"))
    elif t == 0x0003: p = g("vin").encode("ascii").ljust(17, b"\0")[:17]
    elif t == 0x0004:
        p = g("vin").encode("ascii").ljust(17, b"\0")[:17] + struct.pack(">H", g("logical_address")) + bytes.fromhex(g("eid")) + bytes.fromhex(g("gid")) + bytes([g("further_action", 0)])
        if g("sync_status") is not None: p += bytes([g("sync_status")])
    elif t == 0x0005:
        p = struct.pack(">HB", g("source_address"), g("activation_type", 0)) + int(g("reserved", 0)).to_bytes(4, "big") + (bytes.fromhex(g("oem")) if g("oem") else b"")
    elif t == 0x0006:
        p = struct.pack(">HHB", g("tester_address"), g("entity_address"), g("response_code")) + int(g("reserved", 0)).to_bytes(4, "big") + (bytes.fromhex(g("oem")) if g("oem") else b"")
    elif t == 0x0007: p = b""
    elif t == 0x0008: p = struct.pack(">H", g("source_address"))
    elif t == 0x8001: p = struct.pack(">HH", g("source_address"), g("target_address")) + bytes(g("data"))
    elif t in (0x8002, 0x8003): p = struct.pack(">HHB", g("source_address"), g("target_address"), g("code", 0)) + bytes(g("previous", b""))
    else: raise DoipError("cannot encode " + name)
    return struct.pack(">BBHI", version, version ^ 0xFF, t, len(p)) + p
