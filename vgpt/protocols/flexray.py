"""FlexRay (ISO 17458 / FlexRay protocol specification 2.1), listen-only: the chassis and powertrain network of premium cars (stability control, active steering, adaptive dampers) at up to 10 Mbit/s.

A communication cycle (here 5 ms) is divided into slots; a frame sits in the slot whose number is its frame ID (1..2047), on channel A, B or both.  A frame is a 40-bit header, the payload (2 x length bytes) and a
24-bit CRC:

    bit 39 reserved   38 payload preamble indicator   37 null frame indicator (0 = the payload is not valid)   36 sync frame   35 startup frame
    bits 34..24 frame ID   23..17 payload length in 16-bit words   16..6 header CRC (11 bits)   5..0 cycle count

    header CRC   polynomial 0x385, initial value 0x1A, over the 20 bits sync indicator, startup indicator, frame ID, payload length
    frame CRC    polynomial 0x5D6DCB, initial value 0xFEDCBA on channel A and 0xABCDEF on channel B, over the header (40 bits, CRC included) and the payload

A capture block of this module is the channel byte (0 = A, 1 = B) followed by the frame as above.

    encode_frame(slot, payload, cycle, channel=0, sync=False, startup=False, null=False) -> bytes        decode_frame(block) -> {"slot", "cycle", "channel", "payload", "null", "sync", "startup"} or FlexrayError
    Description.load(path)                          a FIBEX-like JSON: frames (slot, repetition / base cycle, channel, length in words, signals with start bit, length, factor, offset, sign, unit)
    description.decode(slot, cycle, payload) -> {frame name: {signal: physical value}}      description.encode(frame name, {signal: value}) -> payload bytes

Signal bits are counted from the most significant bit of the first payload byte (big-endian, as FlexRay tools do).  Only the static segment is carried; dynamic-segment frames are read by the same decoder when
they are present in a capture."""
import json

HEADER_POLY, HEADER_INIT = 0x385, 0x1A
FRAME_POLY, FRAME_INIT_A, FRAME_INIT_B = 0x5D6DCB, 0xFEDCBA, 0xABCDEF


class FlexrayError(ValueError): pass


def _bits(value, n): return [(value >> i) & 1 for i in range(n - 1, -1, -1)]


def crc_bits(bits, poly, width, init):
    """The CRC register after clocking `bits` (most significant first) through the polynomial - one bit at a time, as the specification describes it."""
    reg = init; top = 1 << (width - 1); mask = (1 << width) - 1
    for b in bits:
        fb = ((reg >> (width - 1)) & 1) ^ b
        reg = (reg << 1) & mask
        if fb: reg ^= poly
    return reg


def header_crc(sync, startup, slot, length_words):
    return crc_bits(_bits(sync, 1) + _bits(startup, 1) + _bits(slot, 11) + _bits(length_words, 7), HEADER_POLY, 11, HEADER_INIT)


def frame_crc(header5, payload, channel=0):
    bits = []
    for byte in bytes(header5) + bytes(payload): bits += _bits(byte, 8)
    return crc_bits(bits, FRAME_POLY, 24, FRAME_INIT_B if channel else FRAME_INIT_A)


def encode_frame(slot, payload, cycle, channel=0, sync=False, startup=False, null=False):
    payload = bytes(payload)
    if not 1 <= slot <= 2047: raise FlexrayError("frame id out of range")
    if len(payload) % 2 or len(payload) > 254: raise FlexrayError("payload is 0..127 words")
    if not 0 <= cycle <= 63: raise FlexrayError("cycle count 0..63")
    n = len(payload) // 2; hcrc = header_crc(int(sync), int(startup), slot, n)
    word = (0 << 39) | (0 << 38) | ((0 if null else 1) << 37) | (int(sync) << 36) | (int(startup) << 35) | (slot << 24) | (n << 17) | (hcrc << 6) | cycle
    header = word.to_bytes(5, "big"); return bytes([channel & 1]) + header + payload + frame_crc(header, payload, channel).to_bytes(3, "big")


def decode_frame(block):
    b = bytes(block)
    if len(b) < 9: raise FlexrayError("a FlexRay capture block is at least 9 bytes")
    channel = b[0]
    if channel not in (0, 1): raise FlexrayError("channel byte")
    header = b[1:6]; word = int.from_bytes(header, "big"); n = word >> 17 & 0x7F
    if len(b) != 1 + 5 + 2 * n + 3: raise FlexrayError("length does not match the header")
    if word >> 39 & 1: raise FlexrayError("reserved bit set")
    slot = word >> 24 & 0x7FF
    if slot == 0: raise FlexrayError("frame id 0")
    if (word >> 6 & 0x7FF) != header_crc(word >> 36 & 1, word >> 35 & 1, slot, n): raise FlexrayError("header CRC")
    payload = b[6:6 + 2 * n]
    if int.from_bytes(b[-3:], "big") != frame_crc(header, payload, channel): raise FlexrayError("frame CRC")
    return {"slot": slot, "cycle": word & 0x3F, "channel": channel, "payload": payload, "null": not (word >> 37 & 1), "sync": bool(word >> 36 & 1), "startup": bool(word >> 35 & 1), "words": n}


def looks_like_frame(block):
    try: decode_frame(block); return True
    except FlexrayError: return False


class Signal:
    def __init__(self, d):
        self.name = d["name"]; self.start = int(d["start"]); self.length = int(d["len"]); self.factor = float(d.get("factor", 1.0)); self.offset = float(d.get("offset", 0.0)); self.signed = bool(d.get("signed", False)); self.unit = d.get("unit", "")
        self.choices = {int(k): v for k, v in (d.get("choices") or {}).items()}

    def decode(self, payload):
        total = len(payload) * 8
        if self.start + self.length > total: raise FlexrayError("signal outside the payload")
        raw = (int.from_bytes(payload, "big") >> (total - self.start - self.length)) & ((1 << self.length) - 1)
        if self.signed and raw >> (self.length - 1): raw -= 1 << self.length
        return self.choices.get(raw) if self.choices else raw * self.factor + self.offset

    def raw(self, value):
        if self.choices:
            inv = {v: k for k, v in self.choices.items()}
            return inv[value] if value in inv else int(value)
        r = int(round((float(value) - self.offset) / self.factor))
        lo, hi = (-(1 << self.length - 1), (1 << self.length - 1) - 1) if self.signed else (0, (1 << self.length) - 1)
        return max(lo, min(hi, r))


class Frame:
    def __init__(self, name, d):
        self.name = name; self.slot = int(d["slot"]); self.base = int(d.get("base", 0)); self.repetition = int(d.get("repetition", 1)); self.words = int(d["words"]); self.channel = d.get("channel", "A")
        self.signals = {s["name"]: Signal(s) for s in d["signals"]}

    def sent_in(self, cycle): return cycle % self.repetition == self.base % self.repetition


class Description:
    def __init__(self, spec):
        self.cycle_ms = float(spec.get("cycle_ms", 5.0)); self.frames = {n: Frame(n, d) for n, d in spec["frames"].items()}; self.by_slot = {f.slot: f for f in self.frames.values()}

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: return cls(json.load(f))

    def decode(self, slot, cycle, payload):
        fr = self.by_slot.get(slot)
        if fr is None or not fr.sent_in(cycle): return None
        if len(payload) != 2 * fr.words: raise FlexrayError("payload length differs from the description")
        return fr, {name: s.decode(bytes(payload)) for name, s in fr.signals.items()}

    def encode(self, frame_name, values):
        fr = self.frames[frame_name]; n = 0; total = fr.words * 16
        for name, s in fr.signals.items():
            if name in values: n |= (s.raw(values[name]) & ((1 << s.length) - 1)) << (total - s.start - s.length)
        return n.to_bytes(fr.words * 2, "big")
