"""System-log line parsers (stdlib only): Android logcat, Linux syslog (RFC 3164 / 5424), QNX slog2info.
Each returns a dict {ts_text, host, source, pid, severity (0 emerg .. 7 debug, syslog scale), level (E/W/I..), message, fmt} or None.
Validated on real Android and Linux logs (Loghub). Severity for formats without one is inferred from the message with a small keyword list (documented, conservative)."""
import re

SEV_NAMES = ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
LOGCAT_LEVEL = {"F": 2, "E": 3, "W": 4, "I": 6, "D": 7, "V": 7, "S": 7}
SLOG2_SEV = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 7}                      # QNX slog2: 0 shutdown 1 critical 2 error 3 warning 4 notice 5 info 6 debug1 7 debug2

_LOGCAT = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\s+(\d+)\s+(\d+)\s+([VDIWEFS])\s+([^:]+?)\s*:\s?(.*)$")
_BSD = re.compile(r"^(?:<(\d{1,3})>)?([A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d)\s+(\S+)\s+([^\s:\[]+?(?: v?\d[\d.]*)?)(?:\[(\d+)\])?:\s?(.*)$")
_BSD_PAREN = re.compile(r"^(?:<(\d{1,3})>)?([A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d)\s+(\S+)\s+(\S+?)\[(\d+)\]:\s?(.*)$")
_RFC5424 = re.compile(r"^<(\d{1,3})>(\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(?:-|\[.*?\])\s*(.*)$")
_SLOG2 = re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d\.\d{3})\s+(\S+)\s+(\d+)\s+(\d+)\s+(.*)$")
_BAD = re.compile(r"fail|error|denied|refused|invalid|unable|cannot|can't|timeout|timed out|critical|fatal|panic|crash|oom|killed|abort|exception|unreachable|corrupt", re.I)
_NOTE = re.compile(r"warn|retry|deprecated|slow|low|degrad|drop|missing|unknown|not found", re.I)


def infer_severity(msg):
    if _BAD.search(msg): return 3
    if _NOTE.search(msg): return 4
    return 6


def parse(line):
    line = line.rstrip("\r\n")
    m = _LOGCAT.match(line)
    if m: return {"fmt": "logcat", "ts_text": m[1], "host": "", "source": m[5].strip(), "pid": int(m[2]), "tid": int(m[3]), "level": m[4], "severity": LOGCAT_LEVEL[m[4]], "message": m[6]}
    m = _RFC5424.match(line)
    if m:
        pri = int(m[1])
        if not 0 <= pri <= 191: return None
        return {"fmt": "rfc5424", "ts_text": m[3], "host": m[4], "source": m[5], "pid": int(m[6]) if m[6].isdigit() else None, "severity": pri & 7, "facility": pri >> 3, "level": SEV_NAMES[pri & 7], "message": m[8]}
    m = _SLOG2.match(line)
    if m and 0 <= int(m[3]) <= 7 and 0 <= int(m[4]) < 8 and "." in m[1]:
        sev = SLOG2_SEV[int(m[3])]
        return {"fmt": "slog2info", "ts_text": m[1], "host": "", "source": m[2], "pid": None, "level": SEV_NAMES[sev], "severity": sev, "message": m[5]}
    m = _BSD.match(line) or _BSD_PAREN.match(line)
    if m:
        pri = int(m[1]) if m[1] else None
        if pri is not None and not 0 <= pri <= 191: return None
        sev = (pri & 7) if pri is not None else infer_severity(m[6])
        return {"fmt": "rfc3164", "ts_text": m[2], "host": m[3], "source": m[4], "pid": int(m[5]) if m[5] else None, "severity": sev, "level": SEV_NAMES[sev], "message": m[6], "inferred": pri is None}
    return None


def format_slog2(ts, name, sev_slog2, msg):
    """slog2info text line: 'Mon DD HH:MM:SS.mmm name <severity> <verbosity> message' (tests / simulator)."""
    return f"{ts} {name} {sev_slog2} 0 {msg}"
