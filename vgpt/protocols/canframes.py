"""CAN / CAN-FD frame formats (stdlib only).

  Frame                      ts, can_id, extended, fd, brs, esi, data (bytes)
  DLC_TO_LEN / len_to_dlc    CAN-FD data length codes (9..15 -> 12,16,20,24,32,48,64)
  parse_asc(line)            Vector ASC trace lines (classic 'd' lines and 'CANFD' lines)
  parse_candump(line)        SocketCAN candump -L / log text: (ts) can0 123#DEAD  |  1F334455#...  |  123##1AABB (CAN-FD, flags nibble)  |  123#R
  parse_socketcan_struct(b)  raw struct can_frame (16 bytes) or canfd_frame (72 bytes)
  pack_socketcan(frame)      the inverse, for tests and replay
Cross-checked against python-can on real CAN-FD ASC files."""
import re, struct
from dataclasses import dataclass

DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
CAN_EFF, CAN_RTR, CAN_ERR = 0x80000000, 0x40000000, 0x20000000
CANFD_BRS, CANFD_ESI, CANFD_FDF = 0x01, 0x02, 0x04                  # FDF: the kernel marks a frame read from a CAN-FD socket as an FD frame (Linux 5.x and later)


def len_to_dlc(n):
    for dlc, ln in enumerate(DLC_TO_LEN):
        if ln >= n: return dlc
    raise ValueError("CAN-FD payload longer than 64 bytes")


@dataclass
class Frame:
    ts: float
    can_id: int
    data: bytes
    extended: bool = False
    fd: bool = False
    brs: bool = False
    esi: bool = False
    rtr: bool = False
    error_frame: bool = False
    error_class: int = 0
    channel: str = ""


_ASC_CLASSIC = re.compile(r"^\s*([0-9.]+)\s+(\d+)\s+([0-9a-fA-F]+)(x?)\s+(Rx|Tx)\s+d\s+(\d+)\s*((?:[0-9a-fA-F]{2}\s*)*)")
_ASC_FD = re.compile(r"^\s*([0-9.]+)\s+CANFD\s+(\d+)\s+(Rx|Tx)\s+([0-9a-fA-F]+)(x?)\s+(.*)$")


def parse_asc(line):
    m = _ASC_FD.match(line)
    if m:
        rest = m[6].split()
        if rest and rest[0] not in ("0", "1"): rest = rest[1:]            # optional symbolic message name
        if len(rest) < 4: return None
        brs, esi, dlc, ln = rest[0] == "1", rest[1] == "1", int(rest[2], 16 if rest[2].isalpha() else 10), int(rest[3])
        if ln not in DLC_TO_LEN or dlc > 15 or DLC_TO_LEN[dlc] != ln: return None
        toks = rest[4:4 + ln]                                                                          # exactly the announced number of data bytes: Vector writes a CRC, bit timing and more columns after them
        if len(toks) != ln or any(len(t) != 2 for t in toks): return None
        try: data = bytes(int(b, 16) for b in toks)
        except ValueError: return None
        cid = int(m[4], 16); ext = bool(m[5])
        if cid > (0x1FFFFFFF if ext else 0x7FF): return None
        return Frame(float(m[1]), cid, data, extended=ext, fd=True, brs=brs, esi=esi, channel=m[2])
    m = _ASC_CLASSIC.match(line)
    if m:
        dlc = int(m[6]); tokens = m[7].split()[:dlc]; cid = int(m[3], 16); ext = bool(m[4])                  # trailing "Length = ... BitCount = ..." columns of newer Vector files are not data
        if dlc > 8 or len(tokens) != dlc or any(len(t) != 2 for t in tokens) or cid > (0x1FFFFFFF if ext else 0x7FF): return None
        try: data = bytes(int(x, 16) for x in tokens)
        except ValueError: return None
        return Frame(float(m[1]), cid, data, extended=ext, channel=m[2])
    return None


_CANDUMP = re.compile(r"^\s*\(\s*([0-9.]+)\s*\)\s+(\S+)\s+([0-9A-Fa-f]+)(#{1,2})(R?)([0-9A-Fa-f]*)\s*$")


def parse_candump(line):
    m = _CANDUMP.match(line)
    if not m: return None
    cid = int(m[3], 16); ext = len(m[3]) > 3; fd = m[4] == "##"; body = m[6]
    if cid > (0x1FFFFFFF if ext else 0x7FF): return None
    flags = 0
    if fd:
        if not body: return None
        flags = int(body[0], 16); body = body[1:]
        if flags & ~(CANFD_BRS | CANFD_ESI): return None
    if len(body) % 2: return None
    data = bytes.fromhex(body)
    if (not fd and len(data) > 8) or (fd and len(data) not in DLC_TO_LEN): return None
    if m[5] and data: return None
    return Frame(float(m[1]), cid, data, extended=ext, fd=fd, brs=bool(flags & CANFD_BRS), esi=bool(flags & CANFD_ESI), rtr=bool(m[5]), channel=m[2])


def parse_socketcan_struct(b):
    if len(b) == 16:                                                      # struct can_frame
        cid, dlc = struct.unpack_from("<IB", b, 0)
        if dlc > 8: raise ValueError("classic CAN payload length exceeds 8 bytes")
        err = bool(cid & CAN_ERR); ident = cid & 0x1FFFFFFF; data = bytes(b[8:8 + dlc])
        return Frame(0.0, ident, data, extended=bool(cid & CAN_EFF), rtr=bool(cid & CAN_RTR), error_frame=err, error_class=ident if err else 0)
    if len(b) == 72:                                                      # struct canfd_frame
        cid, ln, fl = struct.unpack_from("<IBB", b, 0)
        if ln > 64 or ln not in DLC_TO_LEN: raise ValueError("invalid CAN-FD payload length")
        if fl & ~(CANFD_BRS | CANFD_ESI | CANFD_FDF): raise ValueError("unsupported CAN-FD flags")
        err = bool(cid & CAN_ERR); ident = cid & 0x1FFFFFFF
        return Frame(0.0, ident, bytes(b[8:8 + ln]), extended=bool(cid & CAN_EFF), fd=True, brs=bool(fl & CANFD_BRS), esi=bool(fl & CANFD_ESI), error_frame=err, error_class=ident if err else 0)
    raise ValueError("a SocketCAN frame is 16 bytes (classic) or 72 bytes (FD)")


def pack_socketcan(f):
    if f.can_id < 0 or f.can_id > (0x1FFFFFFF if (f.extended or f.can_id > 0x7FF) else 0x7FF): raise ValueError("CAN identifier out of range")
    cid = f.can_id | (CAN_EFF if f.extended or f.can_id > 0x7FF else 0) | (CAN_RTR if f.rtr else 0) | (CAN_ERR if f.error_frame else 0)
    if f.fd:
        return struct.pack("<IBBBB", cid, len(f.data), (CANFD_BRS if f.brs else 0) | (CANFD_ESI if f.esi else 0) | CANFD_FDF, 0, 0) + f.data.ljust(64, b"\0")
    return struct.pack("<IBBBB", cid, len(f.data), 0, 0, 0) + f.data.ljust(8, b"\0")
