"""CANopen (CiA 301) over CAN, listen-only: city buses, delivery vans, forklifts, agricultural and construction machines, and the battery / charger nodes of many electric vehicles.

CANopen has no message identifiers of its own: the 11-bit identifier is a function code (the top 4 bits) and a node number (the low 7 bits), the "COB-ID".  What this module reads off the wire:

    0x000               NMT command (start, stop, pre-operational, reset) to a node, or to all nodes (0)
    0x080               SYNC
    0x081..0x0FF        EMCY: emergency of a node (error code, error register, 5 manufacturer bytes)
    0x100               TIME
    0x180+n .. 0x4FF    process data objects: TPDO1..4 (0x180, 0x280, 0x380, 0x480) and RPDO1..4 (0x200, 0x300, 0x400, 0x500) - their layout is the PDO mapping, written as a DBC message of that COB-ID
    0x580+n / 0x600+n   SDO response of node n / request to node n: expedited reads and writes of one object (index, sub-index) are decoded; segmented and block transfers are recognised and not unpacked
    0x700+n             heartbeat / boot-up of node n (the NMT state)

    classify(can_id) -> (kind, node)            decode(can_id, data) -> a dict describing the message
    heartbeat(node, state) / emcy(node, code, register, manufacturer) / sdo_read_request(node, index, sub) / sdo_read_response(node, index, sub, data) / sdo_abort(node, index, sub, code) / nmt(command, node) / sync()
        the frames a node would send (for the simulator and the tests): (can_id, data)

An object read through an SDO is bound like a UDS data identifier: the source lists `od`: {"0x2000:1": {"node": "0x23", "concept": ..., "fmt": "<H", "scale": 0.1, "unit": ...}}."""
import struct

NMT_COMMANDS = {0x01: "start", 0x02: "stop", 0x80: "pre-operational", 0x81: "reset node", 0x82: "reset communication"}
NMT_STATES = {0x00: "boot-up", 0x04: "stopped", 0x05: "operational", 0x7F: "pre-operational"}
EMCY_CLASSES = {0x00: "no error / reset", 0x10: "generic error", 0x20: "current", 0x21: "current, device input side", 0x22: "current inside the device", 0x23: "current, device output side", 0x30: "voltage", 0x31: "mains voltage",
                0x32: "voltage inside the device", 0x33: "output voltage", 0x40: "temperature", 0x41: "ambient temperature", 0x42: "device temperature", 0x50: "device hardware", 0x60: "device software", 0x61: "internal software",
                0x62: "user software", 0x63: "data set", 0x70: "additional modules", 0x80: "monitoring", 0x81: "communication", 0x82: "protocol error", 0x90: "external error", 0xF0: "additional functions", 0xFF: "device specific"}
EMCY_EXACT = {0x8110: "CAN overrun (objects lost)", 0x8120: "CAN in error passive mode", 0x8130: "life guard or heartbeat error", 0x8140: "recovered from bus off", 0x8150: "COB-ID collision", 0x8210: "PDO not processed (length)", 0x8220: "PDO length exceeded"}
ERROR_REGISTER_BITS = ("generic", "current", "voltage", "temperature", "communication", "device profile", "reserved", "manufacturer")
SDO_ABORTS = {0x05030000: "toggle bit not alternated", 0x05040000: "SDO protocol timed out", 0x05040001: "command specifier invalid", 0x06010000: "unsupported access", 0x06010001: "attempt to read a write-only object",
              0x06010002: "attempt to write a read-only object", 0x06020000: "object does not exist", 0x06040041: "object cannot be mapped to a PDO", 0x06070010: "data type or length does not match", 0x06090011: "sub-index does not exist",
              0x06090030: "value range exceeded", 0x08000000: "general error", 0x08000020: "data cannot be transferred or stored", 0x08000022: "data cannot be transferred, device state"}


class CanopenError(ValueError): pass


def classify(can_id):
    """-> (kind, node) for an 11-bit COB-ID."""
    if not 0 <= can_id <= 0x7FF: raise CanopenError("a CANopen COB-ID is 11 bits")
    if can_id == 0x000: return "nmt", 0
    if can_id == 0x080: return "sync", 0
    if can_id == 0x100: return "time", 0
    fn, node = can_id & 0x780, can_id & 0x7F
    if fn == 0x080 and node: return "emcy", node
    pdo = {0x180: "tpdo1", 0x200: "rpdo1", 0x280: "tpdo2", 0x300: "rpdo2", 0x380: "tpdo3", 0x400: "rpdo3", 0x480: "tpdo4", 0x500: "rpdo4"}
    if fn in pdo and node: return pdo[fn], node
    if fn == 0x580 and node: return "sdo_tx", node
    if fn == 0x600 and node: return "sdo_rx", node
    if fn == 0x700 and node: return "heartbeat", node
    return "other", node


def emcy_text(code):
    return EMCY_EXACT.get(code) or EMCY_CLASSES.get(code >> 8) or f"error 0x{code:04X}"


def decode(can_id, data):
    kind, node = classify(can_id); d = bytes(data); out = {"kind": kind, "node": node}
    if kind == "nmt":
        if len(d) != 2: raise CanopenError("NMT command is 2 bytes")
        out.update(command=NMT_COMMANDS.get(d[0], f"0x{d[0]:02X}"), target=d[1])
    elif kind == "heartbeat":
        if len(d) != 1: raise CanopenError("heartbeat is 1 byte")
        out.update(state=NMT_STATES.get(d[0] & 0x7F, f"0x{d[0]:02X}"), code=d[0] & 0x7F, toggle=bool(d[0] & 0x80))
    elif kind == "emcy":
        if len(d) != 8: raise CanopenError("EMCY is 8 bytes")
        code, reg = struct.unpack_from("<HB", d)
        out.update(code=code, text=emcy_text(code), register=reg, register_bits=[n for i, n in enumerate(ERROR_REGISTER_BITS) if reg >> i & 1], manufacturer=bytes(d[3:]), reset=code == 0)
    elif kind == "sdo_rx" or kind == "sdo_tx":
        if len(d) != 8: raise CanopenError("an SDO frame is 8 bytes")
        cs = d[0] >> 5; out["command"] = cs
        if kind == "sdo_tx" and cs == 4: out.update(type="abort", index=struct.unpack_from("<H", d, 1)[0], sub=d[3], abort=struct.unpack_from("<I", d, 4)[0], text=SDO_ABORTS.get(struct.unpack_from("<I", d, 4)[0], "abort"))
        elif kind == "sdo_tx" and cs == 2:
            e, s, n = d[0] >> 1 & 1, d[0] & 1, d[0] >> 2 & 3; out.update(type="upload", index=struct.unpack_from("<H", d, 1)[0], sub=d[3])
            if e: out["data"] = bytes(d[4:8 - n] if s else d[4:8])
            else: out["size"] = struct.unpack_from("<I", d, 4)[0] if s else None; out["segmented"] = True
        elif kind == "sdo_rx" and cs == 2: out.update(type="read_request", index=struct.unpack_from("<H", d, 1)[0], sub=d[3])
        elif kind == "sdo_rx" and cs == 1: out.update(type="write_request", index=struct.unpack_from("<H", d, 1)[0], sub=d[3], data=bytes(d[4:8 - (d[0] >> 2 & 3)] if d[0] & 1 else d[4:8]), expedited=bool(d[0] >> 1 & 1))
        elif kind == "sdo_tx" and cs == 3: out.update(type="write_response", index=struct.unpack_from("<H", d, 1)[0], sub=d[3])
        else: out["type"] = "segment"
    elif kind == "time":
        if len(d) != 6: raise CanopenError("TIME is 6 bytes")
        ms, days = struct.unpack_from("<IH", d); out.update(ms_of_day=ms & 0x0FFFFFFF, days_since_1984=days)
    elif kind in ("sync",): pass
    else: out["data"] = d
    return out


# ------------------------------------------------------------------------------------------------ the frames a node sends
def nmt(command, node=0): return 0x000, bytes([{v: k for k, v in NMT_COMMANDS.items()}[command], node])
def sync(): return 0x080, b""
def heartbeat(node, state=0x05): return 0x700 + node, bytes([state])
def emcy(node, code, register=0x01, manufacturer=b""): return 0x080 + node, struct.pack("<HB", code, register) + bytes(manufacturer).ljust(5, b"\0")[:5]


def sdo_read_request(node, index, sub): return 0x600 + node, bytes([0x40]) + struct.pack("<HB", index, sub) + b"\0\0\0\0"


def sdo_read_response(node, index, sub, data):
    data = bytes(data)
    if not 1 <= len(data) <= 4: raise CanopenError("an expedited read carries 1 to 4 bytes")
    return 0x580 + node, bytes([0x40 | ((4 - len(data)) << 2) | 0x03]) + struct.pack("<HB", index, sub) + data.ljust(4, b"\0")


def sdo_abort(node, index, sub, code): return 0x580 + node, bytes([0x80]) + struct.pack("<HBI", index, sub, code)
