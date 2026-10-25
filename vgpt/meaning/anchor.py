"""Periods named by an event of this vehicle's own life, not by the calendar: "before and after my last service", "since the last service".

The date is read from the vehicle's record (the service meta of its profile), never guessed.  "After" is the period from that moment to the end of the record; "before" is the period of the same length that ends at it (or as much as the record holds), so that
the two can be compared fairly.  The result is typed periods (the same TimeSpec every other period is), so every executor that takes a period takes these."""
import re

from ..query_ir import TimeSpec

SERVICE = re.compile(r"\b(?P<kind>before and after|after and before|before or after|before|after|since|following|prior to)\s+(?:(?:my|the|our|this)\s+)?(?:last|latest|previous|most recent|recent)\s+(?:service|servicing|maintenance|garage visit|workshop visit)\b")
DAY = 86400


def service_periods(brain, text):
    """-> [TimeSpec, ...] in the order the words give them, or None when the text names no such anchor or the vehicle has no service on record."""
    m_ = SERVICE.search(" " + str(text).lower() + " ")
    if not m_: return None
    try:
        r = dict(brain.c.execute("SELECT k, v FROM meta WHERE k IN ('veh.last_service_ts')").fetchall()); ts = int(float(r["veh.last_service_ts"]))
    except Exception: return None
    now = int(brain.now); tz = int(brain.tz or 0); first = brain.c.execute("SELECT MIN(ts) FROM sig_1m").fetchone()[0] or ts - 365 * DAY
    if not (first < ts < now): return None
    mid = lambda x: ((int(x) + tz) // DAY) * DAY - tz
    today = mid(now - 1 if (now + tz) % DAY == 0 else now); n = max(1, (today - mid(ts)) // DAY); room = max(1, (mid(ts) - mid(first)) // DAY); m = min(n, room)          # whole local days, the form every executor takes
    after = TimeSpec("between", n, "day", value2=0); before = TimeSpec("between", n + m, "day", value2=n + 1)
    k = m_.group("kind")
    if k in ("before and after", "before or after"): return [before, after]
    if k == "after and before": return [after, before]
    if k in ("before", "prior to"): return [before]
    return [after]
