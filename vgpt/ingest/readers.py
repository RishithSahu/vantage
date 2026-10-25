"""Readers: a capture in ANY of the usual formats -> one stream of records `Rec(ts, bus, payload)` that the profile router understands.

    sniff(path)                    what format is this file (by its first bytes and lines, never by its extension alone)
    read(path, fmt=None, anchor=None, limit=None)    -> iterator of Rec

    bus "can"     payload canframes.Frame        candump (-L and the default view), Vector ASC, PEAK TRC (1.x / 2.x), Vector BLF (CAN, CAN FD), ELM327 / OBD adapter traces
    bus "table"   payload {name: value}          CSV / TSV (OBD apps, loggers, spreadsheets), JSON lines and JSON documents (flat, nested, COVESA VSS paths, OpenXC, {"name","value"} lists)
    bus "vhal"    payload dict                   Android VehicleHAL property values as JSON lines ({"prop": ..., "area": ..., "float": [...]})
    bus "gnss"    payload str                    NMEA 0183 sentences (also with a timestamp in front of each line)
    bus "log"     payload str                    syslog / logcat / QNX slog2info text, AUTOSAR DLT binary logs

Timestamps are normalised to seconds since the epoch: nanoseconds, microseconds, milliseconds and seconds are told apart by magnitude, ISO-8601 and the common log formats are parsed, and a capture whose clock starts at 0
is placed after `anchor` (default: the file's modification time).  A line or object that cannot be understood is skipped and counted (`Stats.skipped`), never guessed.  Pure standard library (a Vector BLF is
read with zlib; ASAM MDF4 needs the optional `asammdf` package and says so when it is missing)."""
import csv
import datetime as dt
import io
import json
import os
import re
import struct
import zlib
from collections import namedtuple

from ..protocols import canframes

Rec = namedtuple("Rec", "ts bus payload")
TS_KEYS = ("timestamp", "ts", "time", "t", "epoch", "datetime", "date_time", "time_stamp", "utc", "unix", "unixtime", "gpstime", "timestamp_ms", "time_ms", "elapsed", "seconds", "sec", "@timestamp", "systime", "ts_ms", "ts_ns")
NAME_KEYS = ("name", "signal", "sig", "path", "key", "id", "pid", "channel", "parameter", "sensor")
VALUE_KEYS = ("value", "val", "v", "reading", "data", "measurement")
_NMEA = re.compile(r"^(?:[\d.:T\-Z+ ]*?)(\$[A-Z]{2}[A-Z]{3},[^\r\n]*)$")


class Stats:
    def __init__(self): self.records = 0; self.skipped = 0; self.format = None; self.notes = []


# ------------------------------------------------------------------------------------------------------------------ timestamps and numbers
def to_epoch(x, anchor):
    """A number or string in any of the common clocks -> epoch seconds (float), or None."""
    if x is None: return None
    if isinstance(x, str):
        s = x.strip()
        if not s: return None
        try: x = float(s.replace(",", ".") if s.count(",") == 1 and "." not in s else s)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%m-%d %H:%M:%S.%f", "%H:%M:%S.%f", "%H:%M:%S"):
                try:
                    d = dt.datetime.strptime(s, fmt)
                    if d.year == 1900: d = d.replace(year=dt.datetime.fromtimestamp(anchor, dt.timezone.utc).year) if "%m-%d" in fmt or "%H" in fmt[:2] else d
                    return d.replace(tzinfo=dt.timezone.utc).timestamp()
                except ValueError: continue
            try: return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if "+" in s or s.endswith("Z") else dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()
            except ValueError: return None
    try: x = float(x)
    except (TypeError, ValueError): return None
    if x != x: return None
    a = abs(x)
    if a >= 1e17: x /= 1e9
    elif a >= 1e14: x /= 1e6
    elif a >= 1e11: x /= 1e3
    return x if abs(x) >= 1e9 else anchor + x


def num(v):
    """A cell or JSON value -> float, or None (text, empty, NaN)."""
    if isinstance(v, bool): return 1.0 if v else 0.0
    if isinstance(v, (int, float)): return float(v) if v == v and abs(v) != float("inf") else None
    if not isinstance(v, str): return None
    s = v.strip().lower()
    if s in ("", "-", "--", "nan", "null", "none", "n/a", "na", "undefined"): return None
    if s in ("true", "on", "yes", "open", "engaged", "active"): return 1.0
    if s in ("false", "off", "no", "closed", "inactive"): return 0.0
    try: return float(s.replace(",", ".") if s.count(",") == 1 and "." not in s else s)
    except ValueError: return None


# ------------------------------------------------------------------------------------------------------------------ sniffing
def _head(path, n=65536):
    with open(path, "rb") as f: return f.read(n)


def sniff(path):
    """-> 'blf' | 'dlt' | 'mf4' | 'pcap' | 'candump' | 'asc' | 'trc' | 'elm' | 'nmea' | 'jsonl' | 'json' | 'csv' | 'log' | 'unknown'."""
    b = _head(path)
    if b[:4] == b"LOGG": return "blf"
    if b[:4] == b"DLT\x01": return "dlt"
    if b[:4] in (b"MDF ", b"MDF3") or b[8:12] == b"MDF ": return "mf4"
    if b[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a"): return "pcap"
    text = b.decode("utf-8", errors="replace"); lines = [l for l in text.splitlines() if l.strip()][:60]
    if not lines: return "unknown"
    def frac(rx): return sum(1 for l in lines if re.search(rx, l)) / len(lines)
    if frac(r"^\s*\(\d+\.\d+\)\s+\S+\s+[0-9A-Fa-f]+#") > 0.6 or frac(r"^\s*\S+\s+[0-9A-Fa-f]{3,8}\s+\[\d+\]\s*(?:[0-9A-Fa-f]{2}\s*)*$") > 0.6: return "candump"
    if any(l.lstrip().lower().startswith(("base hex", "date ")) for l in lines[:6]) or frac(r"^\s*[\d.]+\s+(?:CANFD\s+)?\d+\s+(?:[0-9A-Fa-f]+x?\s+(?:Rx|Tx)|(?:Rx|Tx)\s+[0-9A-Fa-f]+x?)\s+") > 0.5: return "asc"
    if any(l.startswith(";$FILEVERSION") or l.startswith(";   Start time") for l in lines[:8]) or frac(r"^\s*\d+\)?\s+[\d.]+\s+(?:DT|FD)\s+\d+\s+[0-9A-Fa-f]+\s+(?:Rx|Tx)") > 0.6 or frac(r"^\s*\d+\)\s+[\d.]+\s+(?:DT\s+)?\d+\s+[0-9A-Fa-f]+\s+(?:Rx\s+|Tx\s+)?\d+\s") > 0.6: return "trc"
    if frac(_NMEA.pattern) > 0.6: return "nmea"
    first = lines[0].lstrip()
    if first[:1] in "{[":
        return "jsonl" if frac(r"^\s*\{.*\}\s*,?\s*$") > 0.6 else "json"
    if frac(r"^(?:>?\s*)(?:[0-9A-Fa-f]{3}\s+)?(?:[0-9A-Fa-f]{2}\s+){2,}[0-9A-Fa-f]{2}\s*$") > 0.6 and frac(r"41\s+[0-9A-Fa-f]{2}") > 0.3: return "elm"
    try:
        d = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|")
        if len(lines[0].split(d.delimiter)) >= 2 and sum(len(l.split(d.delimiter)) >= 2 for l in lines) >= 0.8 * len(lines): return "csv"
    except csv.Error: pass
    if frac(r"\d\d:\d\d:\d\d") > 0.4: return "log"
    return "unknown"


# ------------------------------------------------------------------------------------------------------------------ CAN text formats
_CANDUMP_DEF = re.compile(r"^\s*(\S+)\s+([0-9A-Fa-f]{3,8})\s+\[(\d+)\]\s*((?:[0-9A-Fa-f]{2}\s*)*)$")
_TRC1 = re.compile(r"^\s*\d+\)?\s+([\d.]+)\s+(?:\d+\s+)?([0-9A-Fa-f]+)\s+(\d+)\s+((?:[0-9A-Fa-f]{2}\s*)*)$")
_TRC2 = re.compile(r"^\s*\d+\)?\s+([\d.]+)\s+(?:DT|FD|BI|RR)\s+(?:\d+\s+)?([0-9A-Fa-f]+)\s+(?:Rx|Tx)\s+(?:-\s+)?(\d+)\s+((?:[0-9A-Fa-f]{2}\s*)*)$")
_ELM_CAN = re.compile(r"^\s*>?\s*([0-9A-Fa-f]{3}|[0-9A-Fa-f]{8})\s+((?:[0-9A-Fa-f]{2}\s*)+)$")
_ELM_RAW = re.compile(r"^\s*>?\s*((?:4[1-9A-Fa-f]|7[Ff])\s+(?:[0-9A-Fa-f]{2}\s*)+)$")


def _hex_frame(ident, hexbytes, ts, ext_hint=None, expect=None):
    try:
        cid = int(ident, 16); data = bytes.fromhex("".join(hexbytes.split()))
    except ValueError: return None
    if expect is not None and len(data) != expect: return None
    ext = len(ident) > 3 if ext_hint is None else ext_hint
    if cid > (0x1FFFFFFF if ext else 0x7FF) or len(data) > 64: return None
    return canframes.Frame(ts, cid, data, extended=ext, fd=len(data) > 8)


def _can_lines(f, anchor, stats, fmt):
    for line in f:
        fr = None; ts = None
        if fmt == "candump":
            line = re.sub(r"\s+[RT]\s*$", "", line)                                                        # python-can's candump writer appends the direction
            fr = canframes.parse_candump(line)
            if fr is None:
                m = _CANDUMP_DEF.match(line)
                if m:
                    fr = _hex_frame(m[2], m[4], 0.0, expect=int(m[3]))
                    if fr is not None: fr = canframes.Frame(0.0, fr.can_id, fr.data, fr.extended, False, channel=m[1])
        elif fmt == "asc": fr = canframes.parse_asc(line)
        elif fmt == "trc":
            m = _TRC2.match(line) or _TRC1.match(line)
            if m:
                fr = _hex_frame(m[2], m[4], float(m[1]) / 1000.0, expect=int(m[3]))                           # PEAK trace files stamp in milliseconds
        elif fmt == "elm":
            m = _ELM_CAN.match(line)
            if m:
                fr = _hex_frame(m[1], m[2], 0.0)
                if fr is not None and len(fr.data) < 8 and not fr.fd: fr = canframes.Frame(0.0, fr.can_id, fr.data.ljust(8, b"\x00"), fr.extended)
            else:
                m = _ELM_RAW.match(line)
                if m:
                    data = bytes.fromhex("".join(m[1].split()))
                    if 2 <= len(data) <= 7: fr = canframes.Frame(0.0, 0x7E8, bytes([len(data)]) + data + b"\x00" * (7 - len(data)))        # an adapter that hides the headers: a single-frame ISO-TP response
        if fr is None:
            if line.strip() and not line.lstrip().startswith((";", "//", "date", "base", "Begin", "End", "no ", "1)")): stats.skipped += 1
            continue
        if fmt == "elm" and fr.ts == 0.0: stats.records += 1; yield Rec(None, "can", fr); continue
        ts = to_epoch(fr.ts, anchor); stats.records += 1
        yield Rec(ts, "can", canframes.Frame(ts, fr.can_id, fr.data, fr.extended, fr.fd, fr.brs, fr.esi, fr.rtr, fr.error_frame, fr.error_class, fr.channel))


# ------------------------------------------------------------------------------------------------------------------ Vector BLF
_BLF_BASE = struct.Struct("<4sHHLL"); _BLF_V1 = struct.Struct("<LHHQ"); _BLF_V2 = struct.Struct("<LBBHQQ")
_BLF_CAN = struct.Struct("<HBBL8s"); _BLF_CAN2 = struct.Struct("<HBBL"); _BLF_FD = struct.Struct("<HBBLLBBB5x64s")
_BLF_CONTAINER = struct.Struct("<H6xL4x")
_BLF_EXT = 0x80000000


def _blf_objects(buf, base_time, out):
    """Parse the objects at the start of `buf`; -> the offset of the first byte not consumed (an object may continue in the next container)."""
    pos = 0; n = len(buf)
    while pos + _BLF_BASE.size <= n:
        sig, hsize, hver, osize, otype = _BLF_BASE.unpack_from(buf, pos)
        if sig != b"LOBJ":
            pos += 1; continue
        if osize < _BLF_BASE.size: pos += 1; continue
        if pos + osize > n: break
        if otype in (1, 86, 100):
            meta = pos + _BLF_BASE.size
            try:
                if hver == 1: flags, _, _, stamp = _BLF_V1.unpack_from(buf, meta)
                else: flags, _, _, _, stamp, _ = _BLF_V2.unpack_from(buf, meta)
                body = pos + hsize; ts = base_time + stamp * (1e-5 if flags & 1 else 1e-9)
                if otype == 1: ch, fl, dlc, aid, data = _BLF_CAN.unpack_from(buf, body); data = data[:min(dlc, 8)]; fd = False; cfl = 0
                elif otype == 86: ch, fl, dlc, aid = _BLF_CAN2.unpack_from(buf, body); data = bytes(buf[body + _BLF_CAN2.size: body + _BLF_CAN2.size + min(dlc, 8)]); fd = False; cfl = 0
                else: ch, fl, dlc, aid, flen, abc, cfl, valid, data = _BLF_FD.unpack_from(buf, body); data = data[:valid]; fd = True
                ext = bool(aid & _BLF_EXT); cid = aid & 0x1FFFFFFF
                if cid <= (0x1FFFFFFF if ext else 0x7FF): out.append(Rec(ts, "can", canframes.Frame(ts, cid, bytes(data), extended=ext, fd=fd, brs=bool(fd and cfl & 2), esi=bool(fd and cfl & 4), channel=str(ch))))
            except struct.error: pass
        pos += osize + (osize & 3)
    return pos


def _read_blf(path, anchor, stats):
    data = open(path, "rb").read()
    if data[:4] != b"LOGG": raise ValueError("not a Vector BLF file")
    hsize = struct.unpack_from("<L", data, 4)[0]; st = struct.unpack_from("<8H", data, 40) if len(data) >= 56 else (1970, 1, 3, 1, 0, 0, 0, 0)
    try: base = dt.datetime(st[0], st[1], st[3], st[4], st[5], st[6], st[7] * 1000, tzinfo=dt.timezone.utc).timestamp()
    except ValueError: base = anchor
    base = base if base > 1e9 else anchor; pos = hsize; tail = b""
    while pos + _BLF_BASE.size <= len(data):
        sig, h, hv, osize, otype = _BLF_BASE.unpack_from(data, pos)
        if sig != b"LOBJ" or osize < _BLF_BASE.size or pos + osize > len(data): break
        batch = []
        if otype == 10:
            method, usize = _BLF_CONTAINER.unpack_from(data, pos + _BLF_BASE.size); payload = data[pos + _BLF_BASE.size + _BLF_CONTAINER.size: pos + osize]
            if method == 2: payload = zlib.decompress(payload)
            buf = tail + payload; used = _blf_objects(buf, base, batch); tail = buf[used:]
        else: _blf_objects(data[pos: pos + osize + 4], base, batch)
        for r in batch: stats.records += 1; yield r
        pos += osize + (osize & 3)


# ------------------------------------------------------------------------------------------------------------------ tables: CSV / JSON
def _flatten(o, prefix=""):
    """{"a": {"b": 1}, "list": [{"name": "x", "value": 2}]} -> {"a.b": 1, "x": 2}: scalars only, nested names joined with dots, a list of {name, value} objects becomes name -> value."""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)): out.update(_flatten(v, key))
            else: out[key] = v
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, dict):
                nk = next((k for k in NAME_KEYS if k in v and isinstance(v[k], str)), None); vk = next((k for k in VALUE_KEYS if k in v), None)
                if nk and vk is not None and not isinstance(v[vk], (dict, list)): out[str(v[nk])] = v[vk]; continue
                out.update(_flatten(v, f"{prefix}[{i}]"))
            elif isinstance(v, list): out.update(_flatten(v, f"{prefix}[{i}]"))
            else: out[f"{prefix}[{i}]"] = v
    return out


def _split_ts(flat, anchor):
    """-> (ts or None, the rest): the timestamp is the first key of TS_KEYS (case-insensitive, also as the last path segment)."""
    lower = {k.lower(): k for k in flat}
    for k in TS_KEYS:
        real = lower.get(k) or next((lower[l] for l in lower if l.endswith("." + k)), None)
        if real is not None:
            ts = to_epoch(flat[real], anchor)
            if ts is not None: return ts, {a: b for a, b in flat.items() if a != real}
    return None, flat


def _json_records(objs, anchor, stats):
    for o in objs:
        if not isinstance(o, dict): stats.skipped += 1; continue
        if "prop" in o and "area" in o and any(k in o for k in ("float", "int32", "int64", "string", "bytes")):
            ts = to_epoch(o.get("timestamp"), anchor); stats.records += 1; yield Rec(ts, "vhal", o); continue
        flat = _flatten(o); ts, rest = _split_ts(flat, anchor)
        if len(rest) == 2 and set(rest) <= set(NAME_KEYS + VALUE_KEYS) and any(k in rest for k in VALUE_KEYS): rest = {str(next(rest[k] for k in NAME_KEYS if k in rest)): next(rest[k] for k in VALUE_KEYS if k in rest)}
        vals = {k: v for k, v in rest.items() if num(v) is not None}
        if not vals: stats.skipped += 1; continue
        stats.records += 1; yield Rec(ts, "table", vals)


def _read_json(path, anchor, stats, lines):
    text = open(path, encoding="utf-8", errors="replace").read()
    if lines:
        def objs():
            for l in text.splitlines():
                l = l.strip().rstrip(",")
                if not l or l in ("[", "]"): continue
                try: yield json.loads(l)
                except ValueError: stats.skipped += 1
        yield from _json_records(objs(), anchor, stats); return
    doc = json.loads(text)
    if isinstance(doc, dict):
        for k in ("data", "records", "rows", "samples", "signals", "measurements", "items", "frames"):
            if isinstance(doc.get(k), list): doc = doc[k]; break
        else: doc = [doc]
    yield from _json_records(doc, anchor, stats)


def _read_csv(path, anchor, stats):
    raw = open(path, encoding="utf-8-sig", errors="replace", newline="").read(); sample = "\n".join(raw.splitlines()[:30])
    try: delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error: delim = ","
    rows = csv.reader(io.StringIO(raw), delimiter=delim); header = None; ts_col = None; n = 0; first_ts = None
    for row in rows:
        if not row or all(not c.strip() for c in row): continue
        if header is None:
            if sum(1 for c in row if num(c) is None and c.strip()) >= max(1, len(row) // 2): header = [c.strip() for c in row]; ts_col = next((i for i, h in enumerate(header) if h.lower().split("(")[0].split("[")[0].strip().replace(" ", "_") in TS_KEYS or h.lower() in TS_KEYS), None); continue
            header = [f"col{i}" for i in range(len(row))]
        if len(row) < len(header) - 1: stats.skipped += 1; continue
        ts = to_epoch(row[ts_col], anchor) if ts_col is not None else None
        if ts_col is None and first_ts is None:
            for i, h in enumerate(header):                                                                  # a time column under an odd name: the first column whose cells parse as times
                if to_epoch(row[i], anchor) is not None and num(row[i]) is None: ts_col = i; ts = to_epoch(row[i], anchor); break
        vals = {}
        for i, (h, c) in enumerate(zip(header, row)):
            if i == ts_col or not h: continue
            v = num(c)
            if v is not None: vals[h] = v
        if not vals: stats.skipped += 1; continue
        n += 1; stats.records += 1; yield Rec(ts if ts is not None else None, "table", vals)


# ------------------------------------------------------------------------------------------------------------------ NMEA, text logs, DLT
def _read_nmea(path, anchor, stats):
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _NMEA.match(line.strip())
        if not m: stats.skipped += 1; continue
        stamp = line.strip()[:line.strip().index("$")].strip(); ts = to_epoch(stamp, anchor) if stamp else None
        stats.records += 1; yield Rec(ts, "gnss", m[1])


def _read_log(path, anchor, stats):
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\r\n")
        if not line.strip(): continue
        m = re.match(r"^(\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?Z?)\s+(.*)$", line) or re.match(r"^(\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+(.*)$", line)
        ts = to_epoch(m[1], anchor) if m else None; stats.records += 1; yield Rec(ts, "log", line)


def _read_dlt(path, anchor, stats):
    """AUTOSAR DLT storage files: storage header 'DLT\\x01', standard header, optional extended header, verbose string / numeric arguments -> one text line per message."""
    data = open(path, "rb").read(); pos = 0
    while pos + 20 <= len(data):
        if data[pos:pos + 4] != b"DLT\x01": pos += 1; continue
        sec, usec = struct.unpack_from("<LL", data, pos + 4); ecu = data[pos + 12:pos + 16].rstrip(b"\x00").decode("ascii", "replace"); p = pos + 16
        htyp = data[p]; length = struct.unpack_from(">H", data, p + 2)[0]
        if length < 4 or p + length > len(data): pos += 4; continue
        q = p + 4
        if htyp & 0x04: q += 4
        if htyp & 0x08: q += 4
        if htyp & 0x10: q += 4
        apid = ctid = ""; verbose = False
        if htyp & 0x01:
            msin, noar = data[q], data[q + 1]; apid = data[q + 2:q + 6].rstrip(b"\x00").decode("ascii", "replace"); ctid = data[q + 6:q + 10].rstrip(b"\x00").decode("ascii", "replace"); q += 10; verbose = bool(msin & 1)
        body = data[q:p + length]; parts = []
        if verbose:
            i = 0
            while i + 4 <= len(body):
                info = struct.unpack_from("<L", body, i)[0]; i += 4
                if info & 0x200:                                                                   # string
                    ln = struct.unpack_from("<H", body, i)[0]; i += 2; parts.append(body[i:i + ln].split(b"\x00")[0].decode("utf-8", "replace")); i += ln
                elif info & 0x20 or info & 0x40 or info & 0x10 or info & 0x08:                    # ints
                    ln = {1: 1, 2: 2, 3: 4, 4: 8}.get(info & 0xF, 4); parts.append(str(int.from_bytes(body[i:i + ln], "little", signed=bool(info & 0x20)))); i += ln
                elif info & 0x80:
                    ln = {2: 2, 3: 4, 4: 8}.get(info & 0xF, 4); parts.append(str(struct.unpack("<f" if ln == 4 else "<d", body[i:i + ln])[0]) if ln in (4, 8) else ""); i += ln
                else: break
        else: parts.append(body.hex())
        stats.records += 1; yield Rec(float(sec) + usec / 1e6, "log", f"{ecu} {apid}.{ctid} " + " ".join(parts))
        pos = p + length


# ------------------------------------------------------------------------------------------------------------------ the entry points
def read(path, fmt=None, anchor=None, limit=None, stats=None):
    """Iterate the records of a capture.  `stats` (a Stats) is filled as it goes: format, records, skipped, notes."""
    stats = stats if stats is not None else Stats(); fmt = fmt or sniff(path); stats.format = fmt
    anchor = anchor if anchor is not None else os.path.getmtime(path)
    if fmt in ("candump", "asc", "trc", "elm"): it = _can_lines(open(path, encoding="utf-8", errors="replace"), anchor, stats, fmt)
    elif fmt == "blf": it = _read_blf(path, anchor, stats)
    elif fmt == "csv": it = _read_csv(path, anchor, stats)
    elif fmt in ("json", "jsonl"): it = _read_json(path, anchor, stats, lines=fmt == "jsonl")
    elif fmt == "nmea": it = _read_nmea(path, anchor, stats)
    elif fmt == "log": it = _read_log(path, anchor, stats)
    elif fmt == "dlt": it = _read_dlt(path, anchor, stats)
    elif fmt == "mf4": it = _read_mf4(path, anchor, stats)
    elif fmt == "pcap": raise ValueError("a packet capture (pcap) needs the vehicle's SOME/IP service description (FIBEX / ARXML); export the traffic as candump / ASC / BLF or give the service catalogue in a profile pack")
    else: raise ValueError(f"cannot recognise the format of {path}")
    n = 0
    for r in it:
        yield r; n += 1
        if limit and n >= limit: break


def _read_mf4(path, anchor, stats):
    try: from asammdf import MDF
    except ImportError as e: raise ValueError("an ASAM MDF4 (.mf4) log needs the optional package asammdf:  pip install asammdf") from e
    mdf = MDF(path); times = {}
    for ch in mdf.iter_channels(raster=None):
        name = ch.name; sig = ch
        for t, v in zip(sig.timestamps, sig.samples):
            try: val = float(v)
            except (TypeError, ValueError): continue
            times.setdefault(round(float(t), 3), {})[name] = val
    for t in sorted(times): stats.records += 1; yield Rec(anchor + t, "table", times[t])
