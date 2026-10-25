"""The standards a vehicle may speak, as code: SAE J1939 identifiers and parameter groups, OBD-II service 01 over ISO-TP, and what to do with their "not available" values.

    j1939_fields(can_id)        priority, PGN, destination / group extension, source address of a 29-bit identifier
    looks_like_j1939(frames)    is this traffic J1939 (heavy vehicles, buses, agricultural, marine, FMS telematics)?
    j1939_dbc(pgns)             a DBC text for the parameter groups the catalogue knows (one message per PGN, source address ignored)
    install_obd()               teach protocols/obd.py every service-01 parameter of the catalogue

J1939 sends "not available" as an all-ones field (0xFF, 0xFFFF, 0xFFFFFFFF) and "error" as 0xFE.. ; those are not readings and must never be stored (`is_na`).  Scales and offsets are the SAE J1939-71 ones recorded
in catalog.json (for example SPN 84 wheel-based speed = 1/256 km/h per bit, SPN 190 engine speed = 0.125 rpm per bit, SPN 110 coolant temperature = 1 degC per bit with a -40 offset)."""
from . import catalog as K

CANON_PRIORITY = 6                                                  # the identifier a catalogue message is stored under: priority 6, source address 0


def j1939_fields(can_id):
    """-> (priority, pgn, ps, sa).  PDU1 (PF < 240): the PS byte is a destination address and is not part of the PGN; PDU2: it is the group extension and is."""
    can_id &= 0x1FFFFFFF; pf = (can_id >> 16) & 0xFF; ps = (can_id >> 8) & 0xFF; dp = (can_id >> 24) & 1; edp = (can_id >> 25) & 1
    pgn = (edp << 17) | (dp << 16) | (pf << 8) | (ps if pf >= 240 else 0)
    return (can_id >> 26) & 7, pgn, ps, can_id & 0xFF


def canonical_id(can_id):
    """The identifier under which a catalogue message for this PGN is stored: priority 6, any source address -> 0."""
    _, pgn, _, _ = j1939_fields(can_id); return (CANON_PRIORITY << 26) | (pgn << 8)


def looks_like_j1939(frames, sample=4000):
    """frames: iterable of canframes.Frame.  True when most are 29-bit and a fair share of them are parameter groups the catalogue knows, or nearly all are PDU2 (broadcast) groups."""
    known = K.j1939_table(); n = ext = hit = pdu2 = 0
    for f in frames:
        n += 1
        if f.extended: ext += 1; _, pgn, _, _ = j1939_fields(f.can_id); hit += pgn in known; pdu2 += ((f.can_id >> 16) & 0xFF) >= 240
        if n >= sample: break
    return n >= 20 and ext >= 0.6 * n and (hit >= 0.10 * ext or pdu2 >= 0.5 * ext)


def is_na(raw, bits):
    """A J1939 field that does not carry a reading: valid values stop at 0xFA (0xFAFF, 0xFAFFFFFF ...); the top of the range is reserved (0xFB..0xFD), error (0xFE..) or not available (0xFF..).  Flag fields under a byte are not checked."""
    return bits >= 8 and raw >= (0xFB << (bits - 8))


def j1939_dbc(pgns=None):
    """A DBC text with one 8-byte extended message per PGN of the catalogue (or only `pgns`), signals little-endian unsigned with the catalogue's scale and offset.  Returns (text, {signal name: concept})."""
    table = K.j1939_table(); lines = ['VERSION ""', "", "NS_ :", "", "BS_:", "", "BU_: ECU", ""]; names = {}
    for pgn in sorted(table):
        if pgns is not None and pgn not in pgns: continue
        ident = (CANON_PRIORITY << 26) | (pgn << 8) | 0x80000000
        lines.append(f"BO_ {ident} J1939_{pgn}: 8 ECU")
        for concept, spn, start, ln, scale, offset in table[pgn]:
            name = f"SPN{spn}_{concept}"; names[name] = concept
            lines.append(f' SG_ {name} : {(start - 1) * 8}|{ln * 8}@1+ ({scale:.10g},{offset:.10g}) [0|0] "" Vector__XXX')
        lines.append("")
    return "\n".join(lines) + "\n", names


def install_obd():
    """Extend protocols/obd.py with every service-01 parameter of the catalogue that it does not have (idempotent); -> {pid: (name, concept)}."""
    from ..protocols import obd
    out = {}
    for pid, (concept, nbytes, scale, offset) in K.obd_table().items():
        if pid not in obd.PIDS:
            if nbytes == 1: f = (lambda s, o: lambda a, b, c, d: a * s + o)(scale, offset)
            elif nbytes == 2: f = (lambda s, o: lambda a, b, c, d: (256 * a + b) * s + o)(scale, offset)
            else: f = (lambda s, o: lambda a, b, c, d: (((a * 256 + b) * 256 + c) * 256 + d) * s + o)(scale, offset)
            obd.PIDS[pid] = (concept, "", nbytes, f)
        out[pid] = (obd.PIDS[pid][0], concept)
    return out
