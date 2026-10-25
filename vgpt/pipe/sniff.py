"""What is in this block of bytes?  Vantage is never told how a car sends its data; it looks.

`classify(payload)` names the form of ONE block from its structure alone - the Linux CAN frame structure (16 or 72 bytes), a SOME/IP message, an Android VehiclePropValue parcel, a camera picture (JPEG/PNG),
an NMEA sentence, a JSON object, a system-log line, other text - and says "unknown" for anything else.  `ChannelSniffer` keeps the votes of one channel so that a block that is ambiguous on its own (a 16-byte
SOME/IP header is also the size of a CAN frame) is read the way the rest of its channel reads.  `to_bus` turns a block of a known form into what the decoder layer takes: (bus, payload) pairs.

Every test is structural and strict (sizes must agree, checksums must match, reserved bytes must be what they are), so a car that sends something new is called "unknown" rather than mis-read."""
import collections
import json
import struct

from ..protocols import canframes as cf, doip as _doip, flexray as _fx, lin as _lin, logs as _logs, nmea as _nmea, someip as _someip, vhalparcel

KINDS = ("can", "someip", "vhal", "lin", "doip", "flexray", "image", "nmea", "json", "log", "text", "unknown")
SOMEIP_TYPES = {0x00, 0x01, 0x02, 0x40, 0x41, 0x42, 0x80, 0x81, 0xC0, 0xC1, 0x20, 0x21, 0x22, 0xA0, 0xA1, 0xA2}
JPEG, PNG = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"


def _is_can(b):
    n = len(b)
    if n == 16:
        cid, dlc = struct.unpack_from("<IB", b, 0)
        if dlc > 8 or b[5] != 0 or b[6] != 0: return False
        if b[7] not in (0, dlc): return False
    elif n == 72:
        cid, ln, fl = struct.unpack_from("<IBB", b, 0)
        if ln not in cf.DLC_TO_LEN or fl & ~(cf.CANFD_BRS | cf.CANFD_ESI | cf.CANFD_FDF) or b[6] != 0 or b[7] != 0: return False
        if any(b[8 + ln:]): return False                                              # bytes beyond the length are zero in a frame
    else: return False
    if cid & cf.CAN_ERR: return True
    ident = cid & 0x1FFFFFFF
    return ident <= 0x1FFFFFFF if cid & cf.CAN_EFF else ident <= 0x7FF


def _is_someip(b):
    n = len(b); pos = 0; count = 0
    if n < 16: return False
    while pos < n:
        if n - pos < 16: return False
        length = struct.unpack_from(">I", b, pos + 4)[0]
        if length < 8 or pos + 8 + length > n: return False
        if b[pos + 12] != 1 or b[pos + 14] not in SOMEIP_TYPES or b[pos + 15] > 0x5E: return False
        pos += 8 + length; count += 1
    return count >= 1


def _text(b):
    try: s = b.decode("utf-8")
    except UnicodeDecodeError: return None
    if not s or not all(ch.isprintable() or ch in "\r\n\t" for ch in s): return None
    return s


def classify(b):
    b = bytes(b)
    if not b: return "unknown"
    if b[:3] == JPEG or b[:8] == PNG: return "image"
    if len(b) >= 8 and b[1] == (b[0] ^ 0xFF) and _doip.looks_like_frame(b): return "doip"                   # ISO 13400 header: version, its inverse, a known payload type, a length that agrees with the block
    if len(b) in (16, 72) and _is_can(b):
        if len(b) == 16 and _is_someip(b): return "someip"                           # a bare SOME/IP header is the size of a CAN frame: a header whose fields are consistent is read as SOME/IP
        return "can"
    if _is_someip(b): return "someip"
    if vhalparcel.looks_like(b): return "vhal"
    if len(b) >= 9 and b[0] in (0, 1) and (len(b) - 9) % 2 == 0 and _fx.looks_like_frame(b): return "flexray"          # channel byte, header whose CRC agrees, payload, frame CRC that agrees
    if 4 <= len(b) <= 11 and b[0] == _lin.SYNC and _lin.looks_like_frame(b): return "lin"                    # sync byte, protected identifier with valid parity, checksum that agrees
    s = _text(b)
    if s is None: return "unknown"
    t = s.strip()
    if t.startswith("$"):
        lines = [x for x in t.splitlines() if x.strip()]
        if lines and all(_nmea_ok(x) for x in lines): return "nmea"
    if t.startswith("{") and t.endswith("}"):
        try:
            if isinstance(json.loads(t), dict): return "json"
        except ValueError: pass
    first = next((x for x in t.splitlines() if x.strip()), "")
    if _logs.parse(first) is not None: return "log"
    return "text" if len(t) >= 3 else "unknown"


def _nmea_ok(line):
    """A sentence whose checksum agrees (a sentence type Vantage does not use is still an NMEA sentence)."""
    try: _nmea.decode(line.strip()); return True
    except _nmea.NmeaError: return False


class ChannelSniffer:
    """The votes of one channel.  `verdict()` is (kind, share, n): the form most of its blocks have, how large their share is, how many blocks were looked at."""
    def __init__(self): self.votes = collections.Counter(); self.n = 0; self.size = collections.Counter()

    def observe(self, payload):
        k = classify(payload); self.votes[k] += 1; self.n += 1
        if self.n <= 64: self.size[len(payload)] += 1
        return k

    def verdict(self):
        if not self.n: return "unknown", 0.0, 0
        known = [(v, k) for k, v in self.votes.items() if k != "unknown"]
        if not known: return "unknown", 1.0, self.n
        v, k = max(known); return k, v / self.n, self.n

    def settled(self, min_blocks=4, share=0.8):
        k, s, n = self.verdict(); return n >= min_blocks and k != "unknown" and s >= share


# ------------------------------------------------------------------------------------------------ from bytes to what the decoders take
def to_bus(kind, payload, ts, decode_image=None):
    """[(bus, payload)] for one block of a known form (an empty list for a block that cannot be read)."""
    b = bytes(payload)
    try:
        if kind == "can":
            f = cf.parse_socketcan_struct(b); f.ts = ts or 0.0; return [("can", f)]
        if kind == "someip": return [("someip", b)]
        if kind == "lin": return [("lin", b)]
        if kind == "doip": return [("doip", b)]
        if kind == "flexray": return [("flexray", b)]
        if kind == "vhal":
            v = vhalparcel.unpack(b); return [("vhal", v)]
        if kind == "image":
            if decode_image is None: return []
            y = decode_image(b)
            return [("cam", {"cam": "front", "t": ts or 0.0, "w": y[1], "h": y[2], "y": y[0]})] if y else []
        if kind in ("nmea", "log", "text"):
            s = b.decode("utf-8", errors="replace"); bus = "gnss" if kind == "nmea" else "log"
            return [(bus, line.strip()) for line in s.splitlines() if line.strip()]
        if kind == "json":
            o = json.loads(b.decode("utf-8"))
            if not isinstance(o, dict): return []
            if "prop" in o and any(k in o for k in ("float", "int32", "int64")): o = dict(o, bytes=bytes.fromhex(o["bytes"]) if isinstance(o.get("bytes"), str) else o.get("bytes", b"")); return [("vhal", o)]
            if "cam" in o or "blockage" in o or "lane_conf" in o: return [("cam", o)]
            return [("table", {k: v for k, v in o.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})]
    except (ValueError, struct.error, KeyError, TypeError, _someip.SomeIpError, _nmea.NmeaError): return []
    return []


def luma_decoder():
    """A function bytes -> (luma plane bytes, width, height) for a JPEG/PNG block, or None when OpenCV is not present (the camera is then reported, not analysed)."""
    try:
        import cv2
        import numpy as np
    except ImportError: return None

    def decode(b):
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        return img.tobytes(), int(img.shape[1]), int(img.shape[0])
    return decode
