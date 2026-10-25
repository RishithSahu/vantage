"""AUTOSAR SOME/IP and SOME/IP-SD decoding (stdlib only; listen-only - nothing here can transmit).

Header (16 bytes, big endian): message id (service id 16 | method/event id 16; event ids have bit 15 set), length (32, counts from request id to the end),
request id (client id 16 | session id 16), protocol version (1), interface version (8), message type (8), return code (8).
SOME/IP-SD is service 0xFFFF / method 0x8100: flags(8) reserved(24) entries-length(32) entries[16 B each] options-length(32) options[].
Payload layouts of application services come from a small service catalog (Franca-IDL-like JSON) so new services need data, not code."""
import json, struct

MSG_TYPES = {0x00: "REQUEST", 0x01: "REQUEST_NO_RETURN", 0x02: "NOTIFICATION", 0x80: "RESPONSE", 0x81: "ERROR",
             0x20: "TP_REQUEST", 0x21: "TP_REQUEST_NO_RETURN", 0x22: "TP_NOTIFICATION", 0xA0: "TP_RESPONSE", 0xA1: "TP_ERROR"}
RETURN_CODES = {0: "E_OK", 1: "E_NOT_OK", 2: "E_UNKNOWN_SERVICE", 3: "E_UNKNOWN_METHOD", 4: "E_NOT_READY", 5: "E_NOT_REACHABLE", 6: "E_TIMEOUT",
                7: "E_WRONG_PROTOCOL_VERSION", 8: "E_WRONG_INTERFACE_VERSION", 9: "E_MALFORMED_MESSAGE", 10: "E_WRONG_MESSAGE_TYPE"}
SD_SERVICE, SD_METHOD = 0xFFFF, 0x8100
SD_ENTRY = {0x00: "FindService", 0x01: "OfferService", 0x06: "SubscribeEventgroup", 0x07: "SubscribeEventgroupAck"}
HEADER = struct.Struct(">HHIHHBBBB")


class SomeIpError(ValueError): pass


def parse(buf):
    """-> dict(service, method, is_event, length, client, session, proto, iface, type, type_name, rc, rc_name, payload) for ONE message at buf[0:]; also returns total size used."""
    if len(buf) < 16: raise SomeIpError("short SOME/IP header")
    svc, meth, length, client, session, proto, iface, mtype, rc = HEADER.unpack_from(buf, 0)
    if length < 8: raise SomeIpError("length field below the minimum of 8")
    total = 8 + length
    if len(buf) < total: raise SomeIpError("truncated SOME/IP message")
    if proto != 1: raise SomeIpError(f"unsupported protocol version {proto}")
    return {"service": svc, "method": meth, "is_event": bool(meth & 0x8000), "length": length, "client": client, "session": session, "proto": proto, "iface": iface,
            "type": mtype, "type_name": MSG_TYPES.get(mtype, f"0x{mtype:02X}"), "rc": rc, "rc_name": RETURN_CODES.get(rc, f"0x{rc:02X}"), "payload": bytes(buf[16:total]), "size": total}


def split(stream):
    """Split a byte stream (UDP datagram can carry several messages; TCP is a stream) into messages. Returns (messages, remainder)."""
    out, pos = [], 0
    while len(stream) - pos >= 16:
        length = struct.unpack_from(">I", stream, pos + 4)[0]
        if length < 8 or length > (1 << 24): raise SomeIpError("corrupt length - cannot resynchronise")
        if len(stream) - pos < 8 + length: break
        out.append(parse(stream[pos:pos + 8 + length])); pos += 8 + length
    return out, stream[pos:]


def build(service, method, payload=b"", client=0, session=1, iface=1, mtype=0x02, rc=0):
    return HEADER.pack(service, method, 8 + len(payload), client, session, 1, iface, mtype, rc) + payload


# ------------------------------------------------------------------------------------------------ SOME/IP-SD
def parse_sd(payload):
    if len(payload) < 12: raise SomeIpError("short SD payload")
    flags = payload[0]; n_ent = struct.unpack_from(">I", payload, 4)[0]
    if 8 + n_ent + 4 > len(payload) or n_ent % 16: raise SomeIpError("bad SD entries length")
    n_opt = struct.unpack_from(">I", payload, 8 + n_ent)[0]
    if 12 + n_ent + n_opt != len(payload): raise SomeIpError("SD payload length does not match declared entries and options")
    opts, pos, end = [], 12 + n_ent, 12 + n_ent + n_opt
    while pos + 3 <= end:
        ln, typ = struct.unpack_from(">HB", payload, pos); body = payload[pos + 4: pos + 3 + ln]
        if ln < 1 or pos + 3 + ln > end: raise SomeIpError("truncated SD option")
        o = {"type": typ}
        if typ in (0x04, 0x06, 0x14, 0x16) and len(body) >= 8:               # IPv4/IPv6 endpoint, multicast
            a = 4 if typ in (0x04, 0x14) else 16
            o.update(addr=".".join(str(b) for b in body[:4]) if a == 4 else body[:16].hex(), proto={0x06: "TCP", 0x11: "UDP"}.get(body[a + 1], body[a + 1]), port=struct.unpack_from(">H", body, a + 2)[0])
        opts.append(o); pos += 3 + ln
    if pos != end: raise SomeIpError("truncated SD option header")
    entries = []
    for i in range(8, 8 + n_ent, 16):
        e = payload[i:i + 16]; typ = e[0]; idx1, idx2, nn = e[1], e[2], e[3]; svc, inst = struct.unpack_from(">HH", e, 4); major = e[8]; ttl = int.from_bytes(e[9:12], "big")
        d = {"type": typ, "type_name": SD_ENTRY.get(typ, f"0x{typ:02X}"), "service": svc, "instance": inst, "major": major, "ttl": ttl, "opt1": (idx1, nn >> 4), "opt2": (idx2, nn & 0xF)}
        if typ in (0x00, 0x01): d["minor"] = struct.unpack_from(">I", e, 12)[0]
        else: d["counter"] = e[13] & 0xF; d["eventgroup"] = struct.unpack_from(">H", e, 14)[0]
        if typ == 0x01 and ttl == 0: d["type_name"] = "StopOfferService"
        entries.append(d)
    return {"reboot": bool(flags & 0x80), "unicast": bool(flags & 0x40), "entries": entries, "options": opts}


class ServiceCatalog:
    """Payload layouts: {"0x4A11": {"name": "AdasState", "events": {"0x8002": {"name": "Disengaged", "fmt": ">BH", "fields": ["reason", "speed_kmh"], "enums": {"reason": {"0": "driver override"}}}}}}"""
    def __init__(self, spec):
        if not isinstance(spec, dict): raise SomeIpError("service catalog must be an object")
        self.spec = {}
        for key, service in spec.items():
            try: sid = int(key, 16) if isinstance(key, str) else int(key)
            except (TypeError, ValueError) as e: raise SomeIpError("invalid service id in catalog") from e
            if not isinstance(service, dict) or not isinstance(service.get("events"), dict): raise SomeIpError("service catalog requires an events object")
            for event in service["events"].values():
                if not isinstance(event, dict) or not isinstance(event.get("fmt"), str) or not isinstance(event.get("fields"), list): raise SomeIpError("catalog event requires fmt and fields")
                try: layout = struct.Struct(event["fmt"])
                except struct.error as e: raise SomeIpError("invalid catalog struct format") from e
                try: arity = len(layout.unpack(bytes(layout.size)))
                except struct.error as e: raise SomeIpError("catalog struct format cannot be unpacked") from e
                if len(event["fields"]) != arity or not all(isinstance(f, str) and f for f in event["fields"]): raise SomeIpError("catalog field count does not match payload format")
                if not isinstance(event.get("allow_trailing", False), bool): raise SomeIpError("catalog allow_trailing must be boolean")
                unknown = set((event.get("enums") or {})) - set(event["fields"])
                if unknown: raise SomeIpError("catalog enum names unknown field(s): " + ", ".join(sorted(unknown)))
                event["_layout"] = layout
            self.spec[sid] = service

    @classmethod
    def load(cls, path): return cls(json.load(open(path, encoding="utf-8")))

    def decode(self, svc, method, payload):
        s = self.spec.get(svc)
        if not s: return None
        ev = s["events"].get(f"0x{method:04X}") or s["events"].get(method)
        if not ev: return None
        st = ev["_layout"]
        if len(payload) < st.size: raise SomeIpError("payload shorter than the catalogued layout")
        if len(payload) != st.size and not ev.get("allow_trailing", False): raise SomeIpError("payload length differs from the catalogued layout")
        vals = dict(zip(ev["fields"], st.unpack_from(payload, 0)))
        for f, m in (ev.get("enums") or {}).items():
            if f in vals: vals[f + "_text"] = m.get(str(vals[f]), str(vals[f]))
        return {"service": s["name"], "event": ev["name"], "values": vals}


def build_sd_offer(services, endpoint=("192.168.0.10", 30490), session=1, reboot=True, ttl=3):
    """SOME/IP-SD message offering `services` [(service_id, instance, major, minor)] with one IPv4 UDP endpoint option (tests / simulator)."""
    entries = b"".join(bytes([0x01, 0, 0, 0x10]) + struct.pack(">HH", sid, inst) + bytes([major]) + ttl.to_bytes(3, "big") + struct.pack(">I", minor) for sid, inst, major, minor in services)
    opt = struct.pack(">HBB", 9, 0x04, 0) + bytes(int(x) for x in endpoint[0].split(".")) + bytes([0, 0x11]) + struct.pack(">H", endpoint[1])
    payload = bytes([0x80 if reboot else 0, 0, 0, 0]) + struct.pack(">I", len(entries)) + entries + struct.pack(">I", len(opt)) + opt
    return build(SD_SERVICE, SD_METHOD, payload, client=0, session=session, iface=1, mtype=0x02)
