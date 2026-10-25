"""ISO 15765-2 (ISO-TP) reassembly over CAN / CAN-FD, listen-only (the assistant never transmits flow control).

PCI types: 0 single frame (classic: len in low nibble; CAN-FD >8 bytes: byte0 = 0x00, byte1 = len), 1 first frame (12-bit length, or 0 + 32-bit
escape), 2 consecutive frame (sequence number 0..15 wrapping), 3 flow control (observed, ignored). Malformed sequences are reported, never fatal."""

import time

SF, FF, CF, FC = 0, 1, 2, 3
MAX_MESSAGE = 65535                 # supported read-only UDS/OBD responses; bound RAM on the target
ASSEMBLY_TIMEOUT = 1.0


def encode(payload, can_id_dlc=8):
    """Segment a payload into frames (used by tests/simulator; classic 8-byte CAN unless can_id_dlc > 8)."""
    n = len(payload); w = can_id_dlc
    if n <= (7 if w == 8 else w - 2):
        return [bytes([n]) + payload] if w == 8 else [bytes([0, n]) + payload]
    frames = []
    if n <= 4095: head = bytes([0x10 | (n >> 8), n & 0xFF]); take = w - 2
    else: head = bytes([0x10, 0]) + n.to_bytes(4, "big"); take = w - 6
    frames.append(head + payload[:take]); pos, sn = take, 1
    while pos < n:
        frames.append(bytes([0x20 | (sn & 0xF)]) + payload[pos:pos + w - 1]); pos += w - 1; sn += 1
    return frames


class Reassembler:
    """One instance per (arbitration id) stream. feed(bytes) -> payload bytes when a message completes, else None. .errors counts anomalies."""

    def __init__(self):
        self.buf, self.expect, self.sn, self.errors, self.last = bytearray(), 0, 0, 0, None
        self.session = None; self.error_counts = {"timeout": 0, "sequence": 0, "oversize": 0, "replaced": 0, "reserved_pci": 0, "malformed": 0, "session": 0}

    def _reset(self):
        self.buf = bytearray(); self.expect = 0; self.sn = 0; self.session = None

    def _error(self, reason):
        self.errors += 1; self.error_counts[reason] = self.error_counts.get(reason, 0) + 1

    def feed(self, data, now=None, session=None):
        if not data: return None
        now = time.monotonic() if now is None else float(now)
        if self.expect and self.last is not None and now - self.last > ASSEMBLY_TIMEOUT:
            self._error("timeout"); self._reset()
        if self.expect and self.session is not None and session is not None and session != self.session:
            self._error("session"); self._reset()
        self.last = now
        t = data[0] >> 4
        if t == SF:
            if data[0] & 0x0F:                                           # classic single frame
                n = data[0] & 0x0F; msg = bytes(data[1:1 + n]); ok = len(msg) == n
            else:                                                        # CAN-FD single frame with escape length
                if len(data) < 2: self._error("malformed"); return None
                n = data[1]; msg = bytes(data[2:2 + n]); ok = n > 0 and len(msg) == n
            self._reset()
            if not ok: self._error("malformed"); return None
            return msg
        if t == FF:
            if len(data) < 2: self._error("malformed"); return None
            n = ((data[0] & 0x0F) << 8) | data[1]; off = 2
            if n == 0:
                if len(data) < 6: self._error("malformed"); return None
                n = int.from_bytes(data[2:6], "big"); off = 6
            if n <= 0 or n > MAX_MESSAGE:
                self._error("oversize" if n > MAX_MESSAGE else "malformed"); self._reset(); return None
            # A First Frame is only legal for a payload that cannot fit in the
            # corresponding Single Frame, and it cannot carry more bytes than
            # it declares.  Never truncate malformed trailing bytes into a
            # seemingly valid UDS/OBD response.
            initial = len(data) - off
            single_limit = 7 if len(data) <= 8 else len(data) - 2
            if n <= single_limit or initial > n:
                self._error("malformed"); self._reset(); return None
            if self.expect: self._error("replaced")                         # a previous message was never finished
            self.buf = bytearray(data[off:min(len(data), off + n)]); self.expect = n; self.sn = 1; self.session = session
            if len(self.buf) >= n: m = bytes(self.buf[:n]); self._reset(); return m
            return None
        if t == CF:
            if not self.expect: self._error("sequence"); return None
            if (data[0] & 0x0F) != (self.sn & 0x0F): self._error("sequence"); self._reset(); return None
            self.sn += 1; remaining = self.expect - len(self.buf); self.buf += data[1:1 + remaining]
            if len(self.buf) >= self.expect:
                m = bytes(self.buf[:self.expect]); self._reset(); return m
            return None
        if t == FC: return None                                             # observed by a passive listener; no payload
        self._error("reserved_pci"); self._reset(); return None
