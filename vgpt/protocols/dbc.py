"""DBC (CAN database) parser and signal decoder - stdlib only.

Supports: BO_ messages (standard and extended ids), SG_ signals with Intel (@1) / Motorola (@0) byte order, signed/unsigned, factor/offset,
IEEE float/double (SIG_VALTYPE_), multiplexing (M / mN), value tables (VAL_), CAN-FD payloads (> 8 bytes). Cross-checked against `cantools`."""
import math, os, re, struct
from dataclasses import dataclass

EXT_FLAG = 0x80000000
_BO = re.compile(r"^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)")
_SG = re.compile(r"^\s*SG_\s+(\w+)\s*(M|m\d+M?)?\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*\[\s*([-+0-9.eE]+)\s*\|\s*([-+0-9.eE]+)\s*\]\s*\"([^\"]*)\"")
_VT = re.compile(r"^SIG_VALTYPE_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s*;")
_VAL = re.compile(r"^VAL_\s+(\d+)\s+(\w+)\s+(.*);\s*$")


@dataclass(frozen=True)
class DecodedSignal:
    raw: int
    physical: float
    enum: str = None
    unit: str = ""


class Signal:
    __slots__ = ("name", "start", "length", "little", "signed", "factor", "offset", "minimum", "maximum", "unit", "mux", "float_kind", "choices", "_msb")

    def __init__(self, name, start, length, little, signed, factor, offset, minimum, maximum, unit, mux):
        self.name, self.start, self.length, self.little, self.signed = name, start, length, little, signed
        self.factor, self.offset, self.minimum, self.maximum, self.unit, self.mux = factor, offset, minimum, maximum, unit, mux
        self.float_kind, self.choices = None, {}

    def raw(self, data):
        n = len(data)
        if self.little:
            v = (int.from_bytes(data, "little") >> self.start) & ((1 << self.length) - 1)
        else:                                                           # Motorola: start = MSB in sawtooth numbering
            msb = (n - 1 - self.start // 8) * 8 + self.start % 8
            v = (int.from_bytes(data, "big") >> (msb - self.length + 1)) & ((1 << self.length) - 1)
        return v

    def decode(self, data):
        v = self.raw(data)
        if self.float_kind == 32: x = struct.unpack("<f" if self.little else ">f", v.to_bytes(4, "little" if self.little else "big"))[0]
        elif self.float_kind == 64: x = struct.unpack("<d" if self.little else ">d", v.to_bytes(8, "little" if self.little else "big"))[0]
        else:
            if self.signed and v >> (self.length - 1): v -= 1 << self.length
            x = v
        return x * self.factor + self.offset

    def put(self, data, raw):
        """Write the raw integer into a bytearray at this signal's bit position (the inverse of raw())."""
        mask = (1 << self.length) - 1; raw &= mask; n = len(data)
        if self.little:
            v = int.from_bytes(data, "little"); v = (v & ~(mask << self.start)) | (raw << self.start); data[:] = v.to_bytes(n, "little")
        else:
            shift = (n - 1 - self.start // 8) * 8 + self.start % 8 - self.length + 1
            v = int.from_bytes(data, "big"); v = (v & ~(mask << shift)) | (raw << shift); data[:] = v.to_bytes(n, "big")

    def encode(self, data, physical):
        """Physical value (or an enum label) -> raw -> into `data`.  Out-of-range values are clamped to what the bit field can hold."""
        if isinstance(physical, str):
            inv = {label: code for code, label in self.choices.items()}
            if physical not in inv: raise ValueError(f"{self.name}: unknown enum label {physical!r}")
            physical = inv[physical] * self.factor + self.offset
        if self.float_kind:
            kind = "<" if self.little else ">"
            raw = int.from_bytes(struct.pack(kind + ("f" if self.float_kind == 32 else "d"), float((physical - self.offset) / self.factor)), "little" if self.little else "big")
        else:
            raw = int(round((float(physical) - self.offset) / self.factor))
            lo, hi = (-(1 << (self.length - 1)), (1 << (self.length - 1)) - 1) if self.signed else (0, (1 << self.length) - 1)
            raw = max(lo, min(hi, raw))
        self.put(data, raw)


class Message:
    def __init__(self, frame_id, name, dlc, sender):
        self.frame_id, self.name, self.dlc, self.sender = frame_id & 0x1FFFFFFF, name, dlc, sender
        self.extended = bool(frame_id & EXT_FLAG) or frame_id > 0x7FF
        self.signals, self.mux_signal = [], None

    def decode(self, data):
        """-> {signal: physical value}; multiplexed signals only when their mux value matches."""
        return {name: value.physical for name, value in self.decode_with_metadata(data).items()}

    def encode(self, values):
        """{signal name: physical value or enum label} -> payload bytes of this message's DLC (signals not given are zero; a multiplexed signal needs its mux value in `values`)."""
        data = bytearray(self.dlc); mux = values.get(self.mux_signal.name) if self.mux_signal is not None else None
        for s in self.signals:
            if s.name not in values: continue
            if s.mux is not None and s.mux != mux: continue
            s.encode(data, values[s.name])
        return bytes(data)

    def decode_with_metadata(self, data):
        """Decode a complete DBC message while preserving enum semantics."""
        if len(data) != self.dlc: raise ValueError(f"invalid payload length for {self.name}: expected {self.dlc}, got {len(data)}")
        out, mux_val = {}, None
        if self.mux_signal is not None and len(data) * 8 >= self.mux_signal.start + 1: mux_val = self.mux_signal.raw(data)
        for s in self.signals:
            if s.mux is not None and s.mux != mux_val: continue
            if (s.start + s.length > len(data) * 8 and s.little) or (not s.little and ((len(data) - 1 - s.start // 8) * 8 + s.start % 8) < s.length - 1): continue   # does not fit this payload
            raw = s.raw(data); choice_raw = raw
            if s.signed and raw >> (s.length - 1): choice_raw = raw - (1 << s.length)
            value = s.decode(data)
            if not math.isfinite(value): continue
            if s.minimum < s.maximum and not (s.minimum <= value <= s.maximum): continue
            label = s.choices.get(choice_raw)
            if s.choices and label is None: label = f"UNKNOWN({choice_raw})"
            out[s.name] = DecodedSignal(choice_raw, value, label, s.unit)
        return out


class Database:
    def __init__(self): self.messages, self.extended_messages = {}, {}; self.skipped = []  # standard and extended identifiers are distinct namespaces; `skipped`: what a lenient load left out, and why

    @classmethod
    def load(cls, path_or_text, lenient=False):
        """`lenient=True` (used when a decoder file from outside is installed): a signal that does not fit its message, or a message defined twice, is left out and listed in `skipped` instead of refusing the file, and declared value ranges are not enforced."""
        text = open(path_or_text, encoding="utf-8", errors="replace").read() if isinstance(path_or_text, str) and os.path.isfile(path_or_text) else str(path_or_text)
        db, cur, sigs = cls(), None, {}
        for line in text.splitlines():
            m = _BO.match(line)
            if m:
                cur = Message(int(m[1]), m[2], int(m[3]), m[4]); namespace = db.extended_messages if cur.extended else db.messages
                if cur.frame_id in namespace:
                    if not lenient: raise ValueError(f"duplicate DBC message identity {cur.frame_id:#x} extended={cur.extended}")
                    db.skipped.append(f"message {m[2]}: identity {cur.frame_id:#x} is already defined"); cur = None; continue
                namespace[cur.frame_id] = cur; continue
            m = _SG.match(line)
            if m and cur is not None:
                mux = None; kind = m[2]
                start, width = int(m[3]), int(m[4])
                factor, offset, minimum, maximum = float(m[7]), float(m[8]), float(m[9]), float(m[10])
                if width <= 0: raise ValueError(f"DBC signal {m[1]} has non-positive width")
                if start < 0 or not all(math.isfinite(x) for x in (factor, offset, minimum, maximum)):
                    raise ValueError(f"DBC signal {m[1]} has invalid numeric metadata")
                fits = start + width <= cur.dlc * 8 if m[5] == "1" else ((cur.dlc - 1 - start // 8) * 8 + start % 8) - width + 1 >= 0
                if not fits:
                    if not lenient: raise ValueError(f"DBC signal {m[1]} does not fit message DLC {cur.dlc}")
                    db.skipped.append(f"{cur.name}.{m[1]}: does not fit a {cur.dlc}-byte message"); continue
                if lenient: minimum = maximum = 0.0                                                                   # a declared [min|max] in a database from outside is advisory (many contradict their own offset); plausibility is checked against the registry
                sig = Signal(m[1], start, width, m[5] == "1", m[6] == "-", factor, offset, minimum, maximum, m[11], None)
                if kind == "M": cur.mux_signal = sig
                elif kind and kind[0] == "m": sig.mux = int(re.match(r"m(\d+)", kind)[1])
                cur.signals.append(sig); sigs[(cur.frame_id, cur.extended, sig.name)] = sig; continue
            if line.lstrip().startswith("SG_MUL_VAL_"):
                if not lenient: raise ValueError("extended multiplexing (SG_MUL_VAL_) is not supported")
                continue
            if line.lstrip().startswith("SG_") and cur is not None:
                # Do not silently discard a syntactically signal-like line:
                # it can contain nan/inf coefficients or a corrupt bit layout.
                if lenient: db.skipped.append(f"{cur.name}: malformed signal definition: {line.strip()[:80]}"); continue
                raise ValueError(f"malformed DBC signal definition: {line.strip()}")
            m = _VT.match(line)
            if m:
                fid = int(m[1]); key = (fid & 0x1FFFFFFF, bool(fid & EXT_FLAG) or fid > 0x7FF, m[2])
                if key in sigs:
                    code = m[3]
                    if code not in ("1", "2"): raise ValueError(f"unsupported SIG_VALTYPE_ code {code} for {m[2]}")
                    expected = 32 if code == "1" else 64
                    if sigs[key].length != expected: raise ValueError(f"SIG_VALTYPE_ width mismatch for {m[2]}")
                    sigs[key].float_kind = expected
                continue
            m = _VAL.match(line)
            if m:
                fid = int(m[1]); key = (fid & 0x1FFFFFFF, bool(fid & EXT_FLAG) or fid > 0x7FF, m[2])
                if key in sigs:
                    pairs = re.findall(r'(-?\d+)\s+"([^"]*)"', m[3]); sigs[key].choices = {int(a): b for a, b in pairs}
        for msg in list(db.messages.values()) + list(db.extended_messages.values()): # a mux signal must also be decodable as a plain signal
            if msg.mux_signal is not None and msg.mux_signal not in msg.signals: msg.signals.insert(0, msg.mux_signal)
        return db

    def _message(self, frame_id, extended=None):
        embedded = bool(frame_id & EXT_FLAG) or frame_id > 0x7FF
        ident = frame_id & 0x1FFFFFFF
        if extended is None:
            if embedded: extended = True
            elif ident in self.messages and ident in self.extended_messages: raise ValueError("ambiguous CAN identity: extended flag required")
            else: extended = ident in self.extended_messages and ident not in self.messages
        return (self.extended_messages if extended else self.messages).get(ident)

    def by_name(self, name):
        """The message called `name` (standard or extended namespace), or None."""
        return next((m for d in (self.messages, self.extended_messages) for m in d.values() if m.name == name), None)

    def decode(self, frame_id, data, extended=None):
        msg = self._message(frame_id, extended)
        return msg.decode(data) if msg else None

    def decode_with_metadata(self, frame_id, data, extended=None):
        msg = self._message(frame_id, extended)
        return msg.decode_with_metadata(data) if msg else None
