"""SAE J1939 on CAN, listen-only: the 29-bit identifier, the transport protocol that carries messages longer than 8 bytes, and the diagnostic messages that ride on it.

    fields(can_id)         priority, PGN, destination or group extension, source address
    encode_id(...)         the inverse (used by tests and the simulator)
    Reassembler            J1939-21 TP.CM / TP.DT (broadcast announce BAM, connection-mode RTS/CTS observed from the wire) and ETP.CM / ETP.DT (up to MAX_MESSAGE bytes), per (source, destination) pair
    decode_dm1(payload)    J1939-73 DM1 / DM2 active and previously-active fault codes: lamp status, SPN, FMI, occurrence count, conversion method
    decode_vin(payload)    PGN 65260 vehicle identification;  decode_component_id(payload)  PGN 65259 make*model*serial*unit
    FMI_TEXT               the 32 failure-mode identifiers

The assistant never transmits (no CTS, no acknowledgements): connection-mode transfers are rebuilt from the data packets the two ECUs already exchange.  Malformed sequences are counted and dropped, never fatal."""

TP_CM, TP_DT, ETP_CM, ETP_DT = 0xEC00, 0xEB00, 0xC800, 0xC700
PGN_DM1, PGN_DM2, PGN_VIN, PGN_COMPONENT_ID = 0xFECA, 0xFECB, 0xFEEC, 0xFEEB
GLOBAL = 0xFF
MAX_MESSAGE = 1 << 20                  # TP carries at most 1785 bytes; ETP 117 440 505 - the assistant keeps the first megabyte at most (RAM bound on the 2 GB target)
TIMEOUT_S = 1.25                       # J1939-21: T1 0.75 s between data packets, T2/T3 1.25 s; the longest gap allowed is used so that a slow but legal sender is never cut
BAM, RTS, CTS, EOMA, ABORT = 32, 16, 17, 19, 255
E_RTS, E_CTS, E_DPO, E_EOMA = 20, 21, 22, 23

FMI_TEXT = {0: "data valid but above normal operational range - most severe", 1: "data valid but below normal operational range - most severe", 2: "data erratic, intermittent or incorrect", 3: "voltage above normal or shorted to high source",
            4: "voltage below normal or shorted to low source", 5: "current below normal or open circuit", 6: "current above normal or grounded circuit", 7: "mechanical system not responding or out of adjustment",
            8: "abnormal frequency, pulse width or period", 9: "abnormal update rate", 10: "abnormal rate of change", 11: "root cause not known", 12: "bad intelligent device or component", 13: "out of calibration",
            14: "special instructions", 15: "data valid but above normal operating range - least severe", 16: "data valid but above normal operating range - moderately severe", 17: "data valid but below normal operating range - least severe",
            18: "data valid but below normal operating range - moderately severe", 19: "received network data in error", 20: "data drifted high", 21: "data drifted low", 31: "condition exists"}
for _k in range(22, 31): FMI_TEXT.setdefault(_k, "reserved for future assignment")
LAMP = {0: "off", 1: "on", 2: "error", 3: "not available"}
FLASH = {0: "slow flash", 1: "fast flash", 2: "reserved", 3: "steady / not flashing"}


def fields(can_id):
    """-> (priority, pgn, ps, sa).  PDU1 (PF < 240): the PS byte is a destination address and is not part of the PGN; PDU2: it is the group extension and is."""
    can_id &= 0x1FFFFFFF; pf = (can_id >> 16) & 0xFF; ps = (can_id >> 8) & 0xFF; dp = (can_id >> 24) & 1; edp = (can_id >> 25) & 1
    return (can_id >> 26) & 7, (edp << 17) | (dp << 16) | (pf << 8) | (ps if pf >= 240 else 0), ps, can_id & 0xFF


def encode_id(priority, pgn, sa, da=GLOBAL):
    """The identifier of a message of this PGN from this source; `da` is the destination for a PDU1 PGN and is ignored for a PDU2 one."""
    pf = (pgn >> 8) & 0xFF; base = (priority & 7) << 26 | ((pgn >> 16) & 3) << 24 | pf << 16 | (pgn & 0xFF if pf >= 240 else da & 0xFF) << 8
    return base | (sa & 0xFF)


def is_transport(can_id):
    """True for the four transport PGNs: the frame is a piece of a longer message, not a message."""
    _, pgn, _, _ = fields(can_id); return pgn in (TP_CM, TP_DT, ETP_CM, ETP_DT)


class Reassembler:
    """Feed every 29-bit frame of a J1939 bus: feed(can_id, data, now) -> a list of (pgn, source, destination, payload) for every message the frame completes (a single-frame message is returned as it is).  .errors counts anomalies by kind."""

    def __init__(self, timeout=TIMEOUT_S, max_open=64):
        self.timeout = timeout; self.max_open = max_open; self.open = {}; self.errors = {}; self.completed = 0

    def _err(self, kind): self.errors[kind] = self.errors.get(kind, 0) + 1

    def _expire(self, now):
        if now is None: return
        for k in [k for k, s in self.open.items() if now - s["t"] > self.timeout]: self._err("timeout"); del self.open[k]

    def feed(self, can_id, data, now=None):
        _, pgn, ps, sa = fields(can_id); data = bytes(data); self._expire(now)
        if pgn == TP_CM or pgn == ETP_CM: return self._cm(pgn, sa, ps, data, now)
        if pgn == TP_DT or pgn == ETP_DT: return self._dt(pgn, sa, ps, data, now)
        return [(pgn, sa, ps if (pgn >> 8) & 0xFF < 240 else GLOBAL, data)]

    def _cm(self, pgn, sa, da, d, now):
        if len(d) < 8: self._err("short_cm"); return []
        ext = pgn == ETP_CM; ctl = d[0]; key = (ext, sa, da)
        if ctl == ABORT:
            if self.open.pop(key, None) is not None: self._err("aborted")
            return []
        if (not ext and ctl in (BAM, RTS)) or (ext and ctl == E_RTS):
            size = (d[1] | d[2] << 8) if not ext else (d[1] | d[2] << 8 | d[3] << 16 | d[4] << 24); packets = (size + 6) // 7; target = d[5] | d[6] << 8 | d[7] << 16
            if not ext and (not 9 <= size <= 1785 or d[3] != packets): self._err("bad_size"); self.open.pop(key, None); return []
            if ext and size < 1786: self._err("bad_size"); self.open.pop(key, None); return []
            if size > MAX_MESSAGE: self._err("too_large"); self.open.pop(key, None); return []
            if ctl == BAM and da != GLOBAL: self._err("bam_not_global")
            if key in self.open: self._err("restarted")
            if len(self.open) >= self.max_open: self.open.pop(min(self.open, key=lambda k: self.open[k]["t"])); self._err("evicted")
            self.open[key] = {"size": size, "pgn": target, "buf": bytearray(), "next": 1, "t": now if now is not None else 0.0, "ext": ext}
        elif ext and ctl == E_DPO:                                                                                         # sent by the transmitter, like the data it announces: the packets that follow continue at this offset
            s = self.open.get(key)
            if s is None: self._err("orphan_dpo"); return []
            if (d[2] | d[3] << 8 | d[4] << 16) * 7 != len(s["buf"]): self._err("sequence"); del self.open[key]; return []
            s["next"] = 1
            if now is not None: s["t"] = now
        return []

    def _dt(self, pgn, sa, da, d, now):
        ext = pgn == ETP_DT; key = (ext, sa, da)
        if len(d) < 2: self._err("short_dt"); return []
        s = self.open.get(key)
        if s is None: self._err("orphan_dt"); return []
        seq = d[0]
        if seq != s["next"]:
            if seq == s["next"] - 1 and not ext: self._err("duplicate_dt"); return []
            self._err("sequence"); del self.open[key]; return []
        s["buf"] += d[1:8]; s["next"] = seq + 1 if seq < 255 else 1
        if now is not None: s["t"] = now
        if len(s["buf"]) >= s["size"]:
            del self.open[key]; self.completed += 1; return [(s["pgn"], sa, da if (s["pgn"] >> 8) & 0xFF < 240 else GLOBAL, bytes(s["buf"][:s["size"]]))]
        return []


def encode_tp(pgn, payload, sa, da=GLOBAL, priority=6):
    """-> [(can_id, 8 data bytes)] of a TP.CM announce followed by its data packets: a BAM to the global address, an RTS to anyone else (the clear-to-send of the receiver is not part of this list).  For tests and the simulator."""
    n = len(payload); packets = (n + 6) // 7; out = []
    ctl = BAM if da == GLOBAL else RTS
    out.append((encode_id(priority, TP_CM, sa, da), bytes([ctl, n & 0xFF, n >> 8, packets, 0xFF if ctl == BAM else packets, pgn & 0xFF, (pgn >> 8) & 0xFF, (pgn >> 16) & 0xFF])))
    for i in range(packets): out.append((encode_id(7, TP_DT, sa, da), bytes([i + 1]) + payload[i * 7:i * 7 + 7].ljust(7, b"\xFF")))
    return out


def decode_dm1(payload):
    """J1939-73 DM1 (active faults) / DM2 (previously active): lamp status in bytes 0-1, then 4 bytes per fault.
    -> {"lamps": {...}, "faults": [{"spn", "fmi", "fmi_text", "count", "conversion_method"}]}; one all-zero fault is 'no fault'.  None when too short."""
    p = bytes(payload)
    if len(p) < 2: return None
    b0, b1 = p[0], p[1]
    lamps = {"malfunction": LAMP[(b0 >> 6) & 3], "red_stop": LAMP[(b0 >> 4) & 3], "amber_warning": LAMP[(b0 >> 2) & 3], "protect": LAMP[b0 & 3],
             "malfunction_flash": FLASH[(b1 >> 6) & 3], "red_stop_flash": FLASH[(b1 >> 4) & 3], "amber_warning_flash": FLASH[(b1 >> 2) & 3], "protect_flash": FLASH[b1 & 3]}
    faults = []; body = p[2:]
    for i in range(0, len(body) - 3, 4):
        a, b, c, d = body[i:i + 4]
        if a == b == c == d == 0xFF: continue                                                                               # padding to a multiple of 7 or an 8-byte frame
        spn = a | b << 8 | (c >> 5) << 16; fmi = c & 0x1F
        if spn == 0 and fmi == 0 and d == 0 and len(body) <= 6: continue                                                    # "no active faults"
        faults.append({"spn": spn, "fmi": fmi, "fmi_text": FMI_TEXT.get(fmi, "reserved"), "count": d & 0x7F if (d & 0x7F) != 0x7F else None, "conversion_method": d >> 7})
    return {"lamps": lamps, "faults": faults}


def encode_dm1(faults, malfunction=0, red_stop=0, amber=0, protect=0):
    """The DM1 payload for a list of (spn, fmi, count): lamp bytes then 4 bytes per fault; an empty list is the 'no active faults' record."""
    out = bytearray([(malfunction & 3) << 6 | (red_stop & 3) << 4 | (amber & 3) << 2 | (protect & 3), 0xFF])
    for spn, fmi, cnt in faults or [(0, 0, 0)]: out += bytes([spn & 0xFF, (spn >> 8) & 0xFF, (spn >> 16) << 5 | (fmi & 0x1F), cnt & 0x7F])
    return bytes(out)


def _star_fields(payload):
    text = bytes(payload).rstrip(b"\xFF\x00").decode("ascii", "replace"); return text.split("*")


def decode_vin(payload):
    """PGN 65260: the VIN as ASCII terminated by '*'.  -> 17-character VIN or None when it does not look like one."""
    s = bytes(payload).split(b"*")[0].decode("ascii", "replace").strip()
    return s if len(s) == 17 and s.isalnum() and not any(c in "IOQ" for c in s.upper()) else None


def decode_component_id(payload):
    """PGN 65259: make * model * serial number * unit number (the trailing fields are optional)."""
    f = [x for x in _star_fields(payload)]; keys = ("make", "model", "serial", "unit")
    return {k: v for k, v in zip(keys, f) if v}
