"""LIN (ISO 17987 / LIN 2.x), listen-only: the body network of a car (mirrors, seats, climate flaps, rain / light sensor, steering-wheel switches, window lifts).

A LIN frame on the wire is a break, a sync byte (0x55), a protected identifier (6 identifier bits and 2 parity bits), 1 to 8 data bytes and a checksum.  The break is a line condition and is not in a capture; a
captured frame is therefore the bytes `55 PID d0 .. dn CHK`.  The checksum of frames 0..59 is the ENHANCED one (it also sums the PID) from LIN 2.0 on and the CLASSIC one (data only) in LIN 1.3; the
diagnostic frames 0x3C / 0x3D always use the classic one.

    protected_id(i) / identifier(pid)         the parity of the identifier (P0 = ID0^ID1^ID2^ID4, P1 = ~(ID1^ID3^ID4^ID5))
    checksum(data, pid=None)                  classic (pid None) or enhanced
    encode_frame(i, data, enhanced=True)      -> bytes
    decode_frame(block)                       -> {"id", "pid", "data", "enhanced"} or raises LinError (bad sync, parity or checksum, wrong length)
    Ldf.load(path) / Ldf.parse(text)          the LIN Description File: frames, signals (bit position, size), signal encodings (physical value = raw * scale + offset, logical values)
    ldf.decode(i, data) -> {signal: physical value or text}        ldf.encode(frame name, {signal: physical value}) -> data bytes

Only what a listening assistant needs of the LDF is read: Signals, Frames, Signal_encoding_types and Signal_representation.  Event-triggered / sporadic frames and the schedule tables are not needed to read a
frame that is on the wire."""
import re

SYNC = 0x55
DIAG_REQUEST, DIAG_RESPONSE = 0x3C, 0x3D


class LinError(ValueError): pass


def protected_id(i):
    if not 0 <= i <= 0x3F: raise LinError("identifier out of range")
    b = [(i >> k) & 1 for k in range(6)]
    p0 = b[0] ^ b[1] ^ b[2] ^ b[4]; p1 = 1 ^ (b[1] ^ b[3] ^ b[4] ^ b[5])
    return i | (p0 << 6) | (p1 << 7)


def identifier(pid):
    i = pid & 0x3F
    if protected_id(i) != pid: raise LinError("identifier parity error")
    return i


def checksum(data, pid=None):
    s = (pid or 0) + sum(data)
    while s > 0xFF: s = (s & 0xFF) + (s >> 8)
    return (~s) & 0xFF


def encode_frame(i, data, enhanced=True):
    data = bytes(data)
    if not 1 <= len(data) <= 8: raise LinError("data length")
    pid = protected_id(i); enh = enhanced and i not in (DIAG_REQUEST, DIAG_RESPONSE)
    return bytes([SYNC, pid]) + data + bytes([checksum(data, pid if enh else None)])


def decode_frame(block):
    b = bytes(block)
    if not 4 <= len(b) <= 11: raise LinError("a LIN frame is 4 to 11 bytes")
    if b[0] != SYNC: raise LinError("no sync byte")
    i = identifier(b[1]); data = b[2:-1]; got = b[-1]
    if got == checksum(data, b[1]) and i not in (DIAG_REQUEST, DIAG_RESPONSE): enh = True
    elif got == checksum(data): enh = False
    else: raise LinError("checksum")
    return {"id": i, "pid": b[1], "data": data, "enhanced": enh}


def looks_like_frame(block):
    try: decode_frame(block); return True
    except LinError: return False


# ------------------------------------------------------------------------------------------------ LIN Description File
_TOKEN = re.compile(r'\s*(?:(?P<str>"[^"]*")|(?P<num>[-+]?(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][-+]?\d+)?))|(?P<id>[A-Za-z_][A-Za-z_0-9.]*)|(?P<p>[{};:,=]))')


def _strip(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _tokens(text):
    pos = 0; out = []
    while True:
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip(): raise LinError("cannot read the LDF near: " + text[pos:pos + 30].strip())
            return out
        pos = m.end()
        if m.group("str") is not None: out.append(("s", m.group("str")[1:-1]))
        elif m.group("num") is not None:
            t = m.group("num"); out.append(("n", int(t, 16) if t.lower().lstrip("+-").startswith("0x") else (float(t) if any(c in t for c in ".eE") else int(t))))
        elif m.group("id") is not None: out.append(("i", m.group("id")))
        else: out.append(("p", m.group("p")))


def _blocks(toks):
    """{section name: list of statements}; a statement is a list of tokens up to ';' (a nested { } stays inside it as ('block', [...]))."""
    pos = 0; sections = {}
    while pos < len(toks):
        kind, val = toks[pos]
        if kind == "i" and pos + 1 < len(toks) and toks[pos + 1] == ("p", "{"):
            body, pos = _block(toks, pos + 2); sections.setdefault(val, []).extend(body)
        else:
            while pos < len(toks) and toks[pos] != ("p", ";"): pos += 1
            pos += 1
    return sections


def _block(toks, pos):
    stmts = []; cur = []
    while pos < len(toks):
        t = toks[pos]
        if t == ("p", "}"): return stmts, pos + 1
        if t == ("p", "{"):
            body, pos = _block(toks, pos + 1); cur.append(("block", body)); stmts.append(cur); cur = []; continue                  # a frame or an encoding ends with its closing brace
        if t == ("p", ";"): stmts.append(cur); cur = []
        else: cur.append(t)
        pos += 1
    raise LinError("unbalanced braces in the LDF")


class Signal:
    def __init__(self, name, size, init, publisher=""): self.name, self.size, self.init, self.publisher = name, size, init, publisher


class Frame:
    def __init__(self, name, ident, publisher, length): self.name, self.id, self.publisher, self.length, self.signals = name, ident, publisher, length, []    # signals: [(Signal, bit offset)]


class Encoding:
    def __init__(self, name): self.name, self.physical, self.logical, self.bcd, self.ascii = name, [], {}, False, False     # physical: [(lo, hi, scale, offset, unit)]; logical: {value: text}

    def to_physical(self, raw):
        if raw in self.logical: return self.logical[raw]
        for lo, hi, scale, off, unit in self.physical:
            if lo <= raw <= hi: return raw * scale + off
        return float(raw)

    def to_raw(self, value):
        if isinstance(value, str):
            for k, v in self.logical.items():
                if v == value: return k
            raise LinError("no such logical value: " + value)
        for lo, hi, scale, off, unit in self.physical:
            r = int(round((float(value) - off) / (scale or 1.0)))
            if lo <= r <= hi: return r
        return int(round(float(value)))

    @property
    def unit(self): return next((u for *_, u in self.physical if u), "")


class Ldf:
    def __init__(self): self.signals = {}; self.frames = {}; self.by_id = {}; self.encodings = {}; self.repr = {}; self.speed = None; self.version = ""

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: return cls.parse(f.read())

    @classmethod
    def parse(cls, text):
        """A file from outside can be cut or damaged anywhere: a statement that does not have the expected shape raises LinError (a ValueError), never a TypeError / IndexError."""
        try: return cls._parse(text)
        except LinError: raise
        except (TypeError, IndexError, ValueError, KeyError, AttributeError) as e: raise LinError(f"malformed LIN description file ({type(e).__name__}: {e})") from e

    @classmethod
    def _parse(cls, text):
        toks = _tokens(_strip(text)); sec = _blocks(toks); L = cls()
        for st in sec.get("Signals", ()):
            if len(st) >= 7 and st[1] == ("p", ":"): L.signals[st[0][1]] = Signal(st[0][1], int(st[2][1]), st[4][1] if st[4][0] == "n" else 0, st[-3][1] if len(st) > 6 and st[-3][0] == "i" else "")
        for st in sec.get("Frames", ()):
            if len(st) < 7 or st[1] != ("p", ":"): continue
            name = st[0][1]; ident = int(st[2][1]); pub = st[4][1]; length = int(st[6][1]); fr = Frame(name, ident, pub, length)
            for blk in [t for t in st if t[0] == "block"]:
                for s2 in blk[1]:
                    if len(s2) >= 3 and s2[0][0] == "i" and s2[1] == ("p", ","): sig = L.signals.get(s2[0][1]); fr.signals.append((sig, int(s2[2][1]))) if sig else None
            L.frames[name] = fr; L.by_id[ident] = fr
        for st in sec.get("Signal_encoding_types", ()):
            if len(st) == 2 and st[0][0] == "i" and st[1][0] == "block":
                e = Encoding(st[0][1])
                for s2 in st[1][1]:
                    key = s2[0][1] if s2 and s2[0][0] == "i" else ""; vals = [t[1] for t in s2[2:] if t[0] in ("n", "s")]
                    if key == "physical_value" and len(vals) >= 4: e.physical.append((int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3]), str(vals[4]) if len(vals) > 4 else ""))
                    elif key == "logical_value" and vals: e.logical[int(vals[0])] = str(vals[1]) if len(vals) > 1 else str(vals[0])
                    elif key == "bcd_value": e.bcd = True
                    elif key == "ascii_value": e.ascii = True
                L.encodings[e.name] = e
        for st in sec.get("Signal_representation", ()):
            if len(st) >= 3 and st[1] == ("p", ":"):
                for t in st[2:]:
                    if t[0] == "i": L.repr[t[1]] = st[0][1]
        return L

    def decode(self, ident, data):
        fr = self.by_id.get(ident)
        if fr is None: raise LinError("no such frame in the LDF")
        if len(data) != fr.length: raise LinError("frame length")
        n = int.from_bytes(bytes(data), "little"); out = {}
        for sig, off in fr.signals:
            raw = (n >> off) & ((1 << sig.size) - 1) if sig.size <= 16 or sig.size % 8 else None
            if raw is None: raw = int.from_bytes(bytes(data)[off // 8:off // 8 + sig.size // 8], "little")
            enc = self.encodings.get(self.repr.get(sig.name, ""))
            out[sig.name] = enc.to_physical(raw) if enc else raw
        return out

    def encode(self, frame_name, values):
        fr = self.frames[frame_name]; n = 0
        for sig, off in fr.signals:
            enc = self.encodings.get(self.repr.get(sig.name, ""))
            if sig.name in values: v = values[sig.name]; raw = enc.to_raw(v) if enc else int(round(float(v)))
            else: raw = sig.init if isinstance(sig.init, int) else 0
            raw = max(0, min((1 << sig.size) - 1, int(raw))); n |= raw << off
        return n.to_bytes(fr.length, "little")
