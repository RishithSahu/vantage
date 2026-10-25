"""Typed temporal operations: threshold crossing, duration, episodes, state timelines, percentiles and event joins.

Each operation is a small, explicit computation over the vehicle's recorded one-minute samples (the 1 Hz tier is never mixed in, so every number is on one declared basis):

  * a *sample* is one recorded one-minute row; its duration is 60 s.  A gap longer than 120 s between rows breaks continuity: nothing is assumed about the unrecorded time.
  * the *state* of a sample is the vehicle's recorded drive mode (parked / city / highway / charging); a state that is not recorded is unknown, never assumed.
  * a *condition* is a conjunction of named states and their complements ("not charging and not parked"); negation binds to the state it precedes.
  * a *percentile* is the linear-interpolation (Hyndman-Fan type 7) percentile of the valid samples of the stated population, and is only reported when the sample is large
    enough to estimate that tail (at least 10 / min(p, 1 - p) samples).

Everything is read from the recorded data through the executor's compiled queries; no operation reads SQL text from the question.  Anything that cannot be computed on that
basis is declined with the reason, never replaced by a nearby quantity."""
import datetime as dt
import math
import functools
import re

import numpy as np

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition
from .query_planner import resolve_time, local_midnight, DAY
from . import composed as C

STEP = 60              # seconds represented by one recorded sample
BREAK = 120            # a gap larger than this between consecutive samples ends a continuous stretch
FULL_DAY = 0.90        # a calendar day is "fully covered" when at least this share of its 1440 minutes has recorded samples
MODE = {"parked": 0, "city": 1, "highway": 2, "charging": 3}
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
CMP_WORDS = [(r"(?:rise|rises|rose|risen|go|goes|went|gone|get|gets|got|climb\w*|cross\w*|exceed\w*|surpass\w*|top\w*) (?:above|over|beyond|past)|above|over|exceed\w*|more than|greater than|higher than|faster than|hotter than|warmer than|at more than|beyond|cross\w*", "gt"),
             (r"(?:fall|falls|fell|fallen|drop|drops|dropped|go|goes|went|gone|get|gets|got|sink\w*) (?:below|under)|below|under|less than|lower than|slower than|colder than|cooler than|beneath", "lt"),
             (r"at least|no less than|minimum of|not below", "ge"), (r"at most|no more than|maximum of|not above|up to", "le")]
UNIT_RX = r"(km/h|kph|mph|°c|degrees? c(?:elsius)?|degrees?|celsius|c\b|°f|fahrenheit|l/min|litres? per minute|kpa|psi|bar|%|percent|kw|wh/km|rpm|v\b)?"
NUMRX = r"(-?\d+(?:\.\d+)?|" + "|".join(WORDNUM) + r")"


def _num(s):
    s = str(s).strip().lower()
    return float(WORDNUM[s]) if s in WORDNUM else float(s)


_rx = functools.lru_cache(maxsize=None)(re.compile)                    # see semantic_ops._rx


def _numeric_signals(t):
    """Registry concepts named in the text (longest wording first), numeric only, in order of appearance."""
    t_ = " " + re.sub(r"[^a-z0-9%/.-]+", " ", re.sub(r"interquartile range|inter quartile range|\biqr\b|range of values|value range|\brange (?:between|of|from|in)\b|\bthe range\b", " ", t.lower())) + " "
    hits = []
    for phrase, names in R.LEXICON.items():
        p = phrase.lower().replace("-", " ")
        m = _rx(r"(?<![a-z0-9])" + re.escape(p).replace(r"\ ", r"[ -]") + r"s?(?![a-z0-9])").search(t_)
        if m:
            for n in sorted(names):
                c = R.CONCEPTS.get(n)
                if c is not None and c.dtype == "numeric": hits.append((m.start(), -len(p), n))
    out = []
    for _, _, n in sorted(hits):
        if n not in out: out.append(n)
    return out


def threshold(t, signal=None):
    """First stated threshold -> (op, value, unit-or-None, span) or None.  'more than 60' with no unit takes the unit of the signal."""
    for rx, op in CMP_WORDS:
        m = re.search(r"\b(?:" + rx + r")\s+(?:than\s+)?" + NUMRX + r"\s*" + UNIT_RX, t)
        if m:
            v = _num(m.group(1)); u = (m.group(2) or "").strip() or None
            if u in ("degrees", "degree", "degrees c", "degree c", "degrees celsius", "celsius", "c", "°c"): u = "°C"
            elif u in ("kph",): u = "km/h"
            elif u in ("percent",): u = "%"
            elif u and u.startswith("litre"): u = "L/min"
            return op, v, u, (m.start(), m.end())
    return None


def to_signal_unit(value, unit, signal):
    if unit is None or signal not in R.CONCEPTS: return value
    su = R.CONCEPTS[signal].unit
    try:
        if R.unit_family(unit) == R.unit_family(su): return R.convert(value, unit, su)
    except Exception: pass
    return value


def duration_requirement(t):
    """'for at least five minutes', 'for more than 300 seconds', 'for ten minutes' -> seconds, or None."""
    m = re.search(r"\b(?:for|lasting|of) (?:at least |more than |over |longer than |a minimum of |no less than |a full )?" + NUMRX + r"[- ](?:continuous |consecutive |straight |uninterrupted |unbroken )?(second|sec|minute|min|hour|hr)s?\b", t)
    if not m: return None
    n = _num(m.group(1)); u = m.group(2)
    if not math.isfinite(n) or n <= 0: return None
    return n * (1 if u.startswith("sec") else 60 if u.startswith("min") else 3600)


def _samples(sig, mask=None):
    return sig


# ------------------------------------------------------------------ data access
class Data:
    """One loaded window: aligned arrays of the requested concepts plus the drive mode."""

    def __init__(self, qc, concepts, lo, hi):
        self.qc = qc; self.lo = int(lo); self.hi = int(hi)
        need = set(c for c in concepts if c) | {"drive_mode"}
        fr = qc.executor.load_frame(need, self.lo, self.hi, "1m")
        self.frame = fr
        self.ts = fr.ts.astype(np.int64) if len(fr) else np.array([], dtype=np.int64)
        self.vals = {}
        for c in concepts:
            if not c: continue
            v = qc.executor._series(fr, c) if len(fr) else None
            self.vals[c] = None if v is None else np.asarray(v, dtype=float)
        mode = fr.column("drive_mode") if len(fr) else None
        self.mode = None if mode is None else np.asarray(mode, dtype=float)
        self.n = len(self.ts)

    def has(self, c): return self.vals.get(c) is not None

    def state_mask(self, conds):
        """Conjunction of named states / complements over the samples; unknown state never satisfies a state or its complement."""
        m = np.ones(self.n, dtype=bool)
        for c in conds or []:
            name = c.predicate
            if name in MODE or name in ("driving", "moving"):
                if self.mode is None: return None
                known = ~np.isnan(self.mode)
                if name in MODE: inside = self.mode == MODE[name]
                else: inside = np.isin(self.mode, (1, 2))
                m &= known & (~inside if c.op == "not" else inside)
            else:
                cm, err = self.qc.executor._cond_mask(self.frame, [c], "1m")
                if cm is None: return None
                m &= cm
        return m


def runs(ts, ok):
    """Maximal stretches of consecutive qualifying samples that are no more than BREAK seconds apart -> [(start_index, end_index)] (inclusive)."""
    out = []; start = None; prev = None
    for i in range(len(ts)):
        if ok[i]:
            if start is not None and ts[i] - ts[prev] > BREAK: out.append((start, prev)); start = i
            elif start is None: start = i
            prev = i
        else:
            if start is not None: out.append((start, prev)); start = None
    if start is not None: out.append((start, prev))
    return out


def run_seconds(ts, a, b): return int(ts[b] - ts[a]) + STEP


def fmt_dur(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    if h: return f"{h} h {m:02d} min"
    if m or not s: return f"{m} min" + (f" {s} s" if s else "")
    return f"{s} s"


def stamp(ts, tz):
    d = dt.datetime.fromtimestamp(int(ts) + tz, dt.timezone.utc); return f"{d.day} {d.strftime('%b')}, {d:%H:%M}"


def day_label(ts, tz): return C._day_label(ts, tz)


# ------------------------------------------------------------------ parsing helpers shared by the operations
def states_of(t):
    """Named states in the sentence with negation bound to the state it precedes -> list of Condition (predicate / op='not')."""
    from . import nl2ir
    S = nl2ir._S(nl2ir.normalise(t)); hits = nl2ir._predicates(S, [])
    neg = getattr(S, "negated", set()); out = []
    for _, name in hits:
        if name in ("hard_braking", "hard_acceleration", "fast_charging"): continue                      # the event itself, not a state the car was in
        nm = {"city": "city", "highway": "highway", "charging": "charging", "parked": "parked", "driving": "driving"}.get(name, name)
        c = Condition(predicate=nm, op="not" if name in neg else None)
        if all((x.predicate, x.op) != (c.predicate, c.op) for x in out): out.append(c)
    if re.search(r"\bmoving\b|\bin motion\b", t) and not any(x.predicate == "driving" for x in out):
        if re.search(r"\b(?:not|isn't|wasn't|never) (?:\w+ )?moving\b", t): out.append(Condition(predicate="driving", op="not"))
        else: out.append(Condition(predicate="driving"))
    return out


def describe_states(conds):
    names = {"parked": "parked", "charging": "charging", "city": "in city driving", "highway": "on the motorway / highway", "driving": "driving"}
    bits = [(("not " if c.op == "not" else "") + names.get(c.predicate, (R.PREDICATES.get(c.predicate, (c.predicate,))[0]).replace("while ", ""))) for c in conds]
    return " and ".join(bits)


def window_of(brain, qc, t, default_days=7):
    from . import nl2ir
    try:
        ts = [x for x in nl2ir._times(nl2ir._S(nl2ir.normalise(t)), brain.now, brain.tz) if x.type != "current"]
    except Exception: ts = []
    ws = C._windows(t, qc)
    for x in ts:                                                                                     # an explicit span ("over the same 14 days", "the last two weeks") outranks a bare "last week"
        if x.type == "relative" and x.unit in ("day", "week", "month") and (x.value or 0) not in (7,) : return x
    if ws: return ws[0]
    if ts: return ts[0]
    return TimeSpec("relative", default_days, "day")


def coverage_text(d, lo, hi):
    span_h = max(0.0, (hi - lo) / 3600.0)
    return f"{d.n:,} recorded one-minute samples ({d.n * STEP / 3600.0:.1f} h of {span_h:.0f} h in the period)"


def _ans(text, route, **facts): return C._result(text, route, **facts)


def _label(qc, spec):
    return C._label(spec, qc) if spec.type == "range" else resolve_time(spec, qc.ctx)[2]
