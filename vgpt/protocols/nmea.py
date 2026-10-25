"""NMEA 0183 GNSS sentences (stdlib only): the continuous position stream of a car's GNSS receiver (gpsd on QNX/Linux, the Android location service, a telematics unit).
Decoded: RMC (time, validity, position, speed, course, date) and GGA (fix quality, satellites, HDOP, altitude). Checksums are verified; malformed lines raise NmeaError and are counted by the adapter.
Encoders exist for the simulator and tests. Cross-checked against the independent `pynmea2` library (tests/test_gnss.py) and against REAL GPS fixes from the VIF-AEGIS drives."""
import datetime as dt
import math
import re

KNOT_KMH = 1.852


class NmeaError(ValueError): pass


def checksum(body):
    c = 0
    try: raw = body.encode("ascii")
    except UnicodeEncodeError as e: raise NmeaError("non-ASCII sentence") from e
    for ch in raw: c ^= ch
    return f"{c:02X}"


def _sentence(body): return f"${body}*{checksum(body)}"


def _dm(deg, lat):
    """decimal degrees -> NMEA ddmm.mmmm / dddmm.mmmm + hemisphere"""
    hemi = ("N" if deg >= 0 else "S") if lat else ("E" if deg >= 0 else "W"); a = abs(deg); d = int(a); m = (a - d) * 60.0
    if round(m, 4) >= 60.0: d += 1; m = 0.0
    return (f"{d:02d}{m:07.4f}" if lat else f"{d:03d}{m:07.4f}"), hemi


def encode_rmc(ts, lat, lon, speed_kmh, course, valid=True, talker="GP"):
    t = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    if not valid: return _sentence(f"{talker}RMC,{t:%H%M%S}.{int(t.microsecond / 10000):02d},V,,,,,,,{t:%d%m%y},,,N")          # no fix: the receiver still talks, with empty fields
    la, ns = _dm(lat, True); lo, ew = _dm(lon, False)
    body = f"{talker}RMC,{t:%H%M%S}.{int(t.microsecond / 10000):02d},{'A' if valid else 'V'},{la},{ns},{lo},{ew},{speed_kmh / KNOT_KMH:.2f},{course:.1f},{t:%d%m%y},,,{'A' if valid else 'N'}"
    return _sentence(body)


def encode_gga(ts, lat, lon, quality=1, sats=9, hdop=1.0, alt=0.0, talker="GP"):
    t = dt.datetime.fromtimestamp(ts, dt.timezone.utc); la, ns = _dm(lat, True); lo, ew = _dm(lon, False)
    return _sentence(f"{talker}GGA,{t:%H%M%S}.{int(t.microsecond / 10000):02d},{la},{ns},{lo},{ew},{quality},{sats:02d},{hdop:.1f},{alt:.1f},M,0.0,M,,")


def _deg(v, hemi, lat=None):
    if not v: raise NmeaError("empty coordinate")
    if lat is True and hemi not in ("N", "S"): raise NmeaError("invalid latitude hemisphere")
    if lat is False and hemi not in ("E", "W"): raise NmeaError("invalid longitude hemisphere")
    if lat is None and hemi not in ("N", "S", "E", "W"): raise NmeaError("invalid hemisphere")
    dot = v.index("."); d = float(v[: dot - 2]); m = float(v[dot - 2:])
    limit = 90 if (lat is True or hemi in ("N", "S")) else 180
    if not math.isfinite(d) or not math.isfinite(m) or not 0 <= m < 60 or not 0 <= d <= limit or (d == limit and m != 0): raise NmeaError("coordinate out of range")
    x = d + m / 60.0
    return -x if hemi in ("S", "W") else x


def decode(line):
    """One sentence -> dict {"type": "RMC"|"GGA", ...} or None for sentence types we do not use. Raises NmeaError on a bad checksum / malformed fields."""
    if not isinstance(line, str): raise NmeaError("sentence must be text")
    line = line.rstrip("\r\n")
    if not line.startswith("$") or "*" not in line: raise NmeaError("not a sentence")
    body, cs = line[1:].rsplit("*", 1)
    if not re.fullmatch(r"[0-9A-Fa-f]{2}", cs): raise NmeaError("malformed checksum field")
    if checksum(body) != cs.upper(): raise NmeaError("bad checksum")
    f = body.split(","); kind = f[0][2:]
    try:
        if kind == "RMC":
            hh = f[1]; ddmmyy = f[9]
            if f[2] not in ("A", "V"): raise NmeaError("invalid RMC status")
            valid = f[2] == "A"
            yy = int(ddmmyy[4:6]) if ddmmyy else 0; year = (1900 + yy) if yy >= 80 else (2000 + yy)
            ts = dt.datetime(year, int(ddmmyy[2:4]), int(ddmmyy[0:2]), int(hh[0:2]), int(hh[2:4]), int(hh[4:6]), int(round(float(hh[6:] or 0) * 1e6)), tzinfo=dt.timezone.utc).timestamp() if hh and ddmmyy else None
            out = {"type": "RMC", "valid": valid, "ts": ts}
            if valid:
                speed = float(f[7] or 0) * KNOT_KMH; course = float(f[8]) if f[8] else None
                if not math.isfinite(speed) or speed < 0: raise NmeaError("invalid speed")
                if course is not None and (not math.isfinite(course) or not 0 <= course <= 360): raise NmeaError("invalid course")
                out.update(lat=_deg(f[3], f[4], True), lon=_deg(f[5], f[6], False), speed_kmh=speed, course=course)
            return out
        if kind == "GGA":
            q = int(f[6] or 0); hdop = float(f[8]) if f[8] else None; alt = float(f[9]) if f[9] else None
            if hdop is not None and (not math.isfinite(hdop) or hdop < 0): raise NmeaError("invalid HDOP")
            if alt is not None and not math.isfinite(alt): raise NmeaError("invalid altitude")
            sats = int(f[7] or 0)
            if not 0 <= q <= 8 or not 0 <= sats <= 99 or (q > 0 and sats == 0): raise NmeaError("impossible GGA fix quality or satellite count")
            out = {"type": "GGA", "quality": q, "sats": sats, "hdop": hdop, "alt": alt}
            if q > 0: out.update(lat=_deg(f[2], f[3], True), lon=_deg(f[4], f[5], False))
            return out
    except (ValueError, IndexError) as e: raise NmeaError(str(e))
    return None
