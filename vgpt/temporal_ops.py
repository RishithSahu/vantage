"""Operations of the temporal layer (see temporal.py for the shared definitions: sample, gap, state, percentile).  Entry point: handle(brain, raw, t)."""
import datetime as dt
import math
import re

import numpy as np

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition
from .query_planner import resolve_time, local_midnight, DAY
from . import composed as C
from .temporal import (STEP, BREAK, FULL_DAY, MODE, NUMRX, _num, _numeric_signals, threshold, to_signal_unit, duration_requirement, Data, runs, run_seconds, fmt_dur, stamp,
                       day_label, states_of, describe_states, window_of, coverage_text, _ans, _label)

POLAR = r"^(?:did|do|does|has|have|was|were|is|are|can|could|will)\b"


class Ctx:
    """The parsed pieces every operation needs."""

    def __init__(self, brain, qc, t):
        self.brain, self.qc, self.t = brain, qc, t; self.tz = qc.ctx.tz
        self.signals = _numeric_signals(t)
        hint, _ = C._implicit_measure(t)
        if not self.signals and hint: self.signals = [hint]
        if re.search(r"\b(?:faster|slower|speeding|km/h|kph|mph)\b", t) and "speed" not in self.signals: self.signals.insert(0, "speed")
        self.signal = self.signals[0] if self.signals else None
        th = threshold(t, self.signal)
        self.th = None
        if th:
            op, v, u, _span = th
            sig = self.signal
            if sig is None and u:
                for cand, un in (("speed", "km/h"), ("battery_temperature", "°C"), ("coolant_flow", "L/min"), ("tyre_pressure", "kPa"), ("battery_soc", "%")):
                    if u.lower() == un.lower(): sig = cand; break
                if sig is None: sig = self.signal
            self.signal = sig or self.signal
            self.th = (op, to_signal_unit(v, u, self.signal), v, u)
        self.states = states_of(t)
        self.min_dur = duration_requirement(t)
        self.spec = window_of(brain, qc, t)

    def window(self):
        lo, hi, label = resolve_time(self.spec, self.qc.ctx)
        if self.spec.type == "range": label = C._label(self.spec, self.qc)
        return lo, min(hi, self.qc.ctx.now + 1), label

    def load(self, concepts=None, lo=None, hi=None):
        l, h, label = self.window()
        d = Data(self.qc, concepts if concepts is not None else ([self.signal] if self.signal else []), lo if lo is not None else l, hi if hi is not None else h)
        return d, label

    def compare(self, d, arr=None):
        arr = d.vals.get(self.signal) if arr is None else arr
        if arr is None or self.th is None: return None
        op, v = self.th[0], self.th[1]
        return {"gt": arr > v, "ge": arr >= v, "lt": arr < v, "le": arr <= v}[op] & ~np.isnan(arr)

    def th_text(self):
        op, v, raw, u = self.th; unit = u or (R.CONCEPTS[self.signal].unit if self.signal in R.CONCEPTS else "")
        return f"{ {'gt': 'above', 'ge': 'at least', 'lt': 'below', 'le': 'at most'}[op] } {raw:g} {unit}".strip()

    def sig_label(self): return R.label_of(self.signal) if self.signal else "the signal"


def _need(c, d, concepts=()):
    if d.n == 0: return False
    if any(not d.has(x) for x in concepts): return False
    return True


def _nodata(c, label, what="the requested quantity"):
    return _ans(f"There is no recorded {what} {label}, so I cannot answer that.", "temporal_nodata", ok=False)


def _scope_text(c):
    return (" " + describe_states(c.states)) if c.states else ""


def _vals_in_state(c, d, concept=None):
    arr = d.vals.get(concept or c.signal)
    if arr is None: return None, None
    m = d.state_mask(c.states)
    if m is None: return None, None
    return arr, m & ~np.isnan(arr)


# ------------------------------------------------------------------ A: threshold semantics
def op_average_vs_threshold(c):
    t = c.t
    if not (c.th and c.signal and re.search(r"\b(?:average|mean|typical|on average)\b", t) and re.match(POLAR, t.strip())): return None
    d, label = c.load()
    arr, m = _vals_in_state(c, d)
    if arr is None or not m.any(): return _nodata(c, label, f"{c.sig_label()}{_scope_text(c)}")
    mean = float(np.mean(arr[m])); op, v, raw, u = c.th
    ok = {"gt": mean > v, "ge": mean >= v, "lt": mean < v, "le": mean <= v}[op]
    unit = R.CONCEPTS[c.signal].unit
    return _ans(f"{'Yes' if ok else 'No'}. The average {c.sig_label()}{_scope_text(c)} {label} was {mean:.1f} {unit}, which is {'' if ok else 'not '}{c.th_text()} (from {int(m.sum()):,} recorded one-minute samples).", "temporal_avg_vs_threshold", value=mean, verdict=bool(ok))


def op_duration(c):
    t = c.t
    if not re.search(r"\bhow (?:many (?:minutes|hours|seconds|mins)|long|much time)\b|\bfor how (?:long|many)\b", t): return None
    if re.search(r"\bhow long (?:ago|since)\b|\bhow (?:many|much) (?:minutes|hours|days|time) (?:ago|since)\b", t): return None      # elapsed time SINCE an event is a recency question, not the duration of a state
    if re.search(r"\bpercent|%|\bshare\b|\bfraction\b|\bproportion\b", t): return None
    if re.search(r"\bmoving\b.*\bnot charging\b|\bboth\b", t) is None and not (c.th or c.states): return None
    d, label = c.load(([c.signal] if c.th and c.signal else []))
    if d.n == 0: return _nodata(c, label, "samples")
    sm = d.state_mask(c.states)
    if sm is None: return _ans(f"The drive state is not recorded for {label}, so I cannot apply the condition ({describe_states(c.states)}).", "temporal_duration", ok=False)
    m = sm.copy()
    if c.th:
        cm = c.compare(d)
        if cm is None: return _nodata(c, label, c.sig_label())
        m &= cm
    n = int(m.sum()); sec = n * STEP
    if c.th: subj = f"{c.sig_label().capitalize()} was {c.th_text()}" + (f" while {describe_states(c.states)}" if c.states else "")
    else: subj = f"The car was {describe_states(c.states)}"
    unknown = int(np.isnan(d.mode).sum()) if d.mode is not None else d.n
    extra = f" Samples with no recorded drive state: {unknown}." if c.states and unknown else ""
    both = "" if not re.search(r"\bboth\b", t) else ""
    overlap = ""
    if len(c.states) >= 2 and all(x.op != "not" for x in c.states) and len({x.predicate for x in c.states if x.predicate in MODE}) >= 2 and n == 0:
        overlap = " This vehicle records one drive state at a time (parked, city, highway or charging), so two of those states cannot overlap in its data."
    hrs = sec / 3600.0
    return _ans(f"{subj} for {fmt_dur(sec)} ({hrs:.1f} hours) {label}. Each recorded one-minute sample counts as 60 s; {n:,} of {d.n:,} recorded samples qualified.{extra}{overlap}", "temporal_duration", seconds=sec, samples=n)


def op_share(c):
    t = c.t
    if not re.search(r"\bwhat (?:percent|percentage|%|share|fraction|proportion)|\bpercent(?:age)? of\b|\bshare of\b|\bfraction of\b|\bhow much of (?:the|my|this|that) (?:time|week|day|period|drive|driving)\b|\bhow much of (?:yesterday|today|last week)\b", t): return None
    if not c.th: return None
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, "samples")
    den = d.state_mask(c.states)
    if den is None: return _ans(f"The drive state is not recorded for {label}.", "temporal_share", ok=False)
    mm = re.search(r"\bof (?:my |the )?(moving|driving|parked|charging|city|highway)(?: driving)? time\b", t)
    positive_mode = any(x.predicate in MODE or x.predicate in ("driving", "moving") for x in c.states if x.op != "not")
    default_driving = (mm is None) and not positive_mode and not re.search(r"\ball (?:the )?(?:recorded )?(?:time|samples)\b|\bwhole (?:period|time)\b|\bof (?:the )?recorded time\b|\bof the (?:week|day|period)\b", t)
    if default_driving:
        inner0 = d.state_mask([Condition(predicate="driving")])
        if inner0 is not None: den = den & inner0
    den_note = ("driving time" if default_driving else "all recorded samples") if not mm else f"{mm.group(1)} time"
    if mm:
        key = {"moving": "driving", "driving": "driving"}.get(mm.group(1), mm.group(1))
        inner = d.state_mask([Condition(predicate=key)])
        if inner is None: return _ans("The drive state is not recorded, so that denominator cannot be built.", "temporal_share", ok=False)
        den = den & inner
    cmp_ = c.compare(d)
    if cmp_ is None: return None                          # no comparable level/signal in the request: this is not a time-share question (trip-share and similar go elsewhere)
    num = den & cmp_
    nd = int(den.sum())
    if nd == 0: return _ans(f"There are no recorded samples of {den_note}{_scope_text(c)} {label}.", "temporal_share", ok=False)
    pct = 100.0 * int(num.sum()) / nd
    shown = [x for x in c.states if not (mm and x.predicate == key and x.op != "not")] if mm else c.states
    scope = ""
    if shown:
        scope = ", " + " and ".join((("excluding the time " + describe_states([Condition(predicate=x.predicate)])) if x.op == "not" else ("only while " + describe_states([x]))) for x in shown)
    return _ans(f"{pct:.1f}% of {'the ' + den_note if mm else den_note}{scope} {label} was spent with {c.sig_label()} {c.th_text()} ({int(num.sum()):,} of {nd:,} recorded one-minute samples, about {int(num.sum())} minutes)." + (" The share is of driving time by default; ask for \"of all recorded time\" to use every recorded minute." if default_driving else ""), "temporal_share", pct=pct, num=int(num.sum()), den=nd)


def op_episodes(c):
    t = c.t
    if not (c.th and re.search(r"\b(?:how many|count|number of|how often|tally)\b.{0,30}\b(?:times|episodes|occasions|instances|occurrences|stretches|spells|periods)\b|\b(?:separate|distinct|individual|different) (?:times|episodes|occasions|instances|stretches)\b", t)): return None
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = d.state_mask(c.states)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    ok = cm & sm
    rs = runs(d.ts, ok); n = len(rs)
    longest = max((run_seconds(d.ts, a, b) for a, b in rs), default=0)
    first = f" The first began at {stamp(d.ts[rs[0][0]], c.tz)}." if rs else ""
    return _ans(f"{c.sig_label().capitalize()} went {c.th_text()} on {n} separate occasion{'' if n == 1 else 's'} {label}{_scope_text(c)}; the longest lasted {fmt_dur(longest)}.{first} An occasion is a stretch of consecutive recorded samples (no gap over {BREAK // 60} min), counted once however long it lasts.",
                "temporal_episodes", n=n, longest=longest)


def op_continuous(c):
    t = c.t
    if not (c.th and (c.min_dur or re.search(r"\bcontinuous\w*|\bconsecutive\b|\bwithout a break\b|\bremain\w*|\bstay\w*|\bkept\b", t))): return None
    if re.search(r"\bon which days\b|\bwhich days\b|\bhow many days\b", t): return None
    if re.search(r"\bhow (?:many (?:minutes|hours|seconds|mins)|long)\b|\bfor how (?:long|many)\b", t) and not c.min_dur and not re.search(r"\bcontinuous\w*|\bconsecutive\b|\bwithout a break\b|\bunbroken\b", t): return None      # a duration question, not a continuity one
    need = c.min_dur or 0
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = d.state_mask(c.states)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    rs = runs(d.ts, cm & sm)
    if not rs: return _ans(f"No. {c.sig_label().capitalize()} was never {c.th_text()}{_scope_text(c)} {label} in the recorded samples.", "temporal_continuous", verdict=False)
    best = max(rs, key=lambda r: run_seconds(d.ts, r[0], r[1])); bs = run_seconds(d.ts, *best)
    good = [r for r in rs if run_seconds(d.ts, r[0], r[1]) >= need]
    if need and not good:
        return _ans(f"No. The longest stretch with {c.sig_label()} {c.th_text()}{_scope_text(c)} {label} lasted {fmt_dur(bs)} ({stamp(d.ts[best[0]], c.tz)}), shorter than the {fmt_dur(need)} asked about. A single qualifying sample is one minute and does not count as continuous.", "temporal_continuous", verdict=False, longest=bs)
    if need:
        return _ans(f"Yes. {c.sig_label().capitalize()} stayed {c.th_text()}{_scope_text(c)} for at least {fmt_dur(need)} on {len(good)} occasion{'' if len(good) == 1 else 's'} {label}; the longest was {fmt_dur(bs)} starting {stamp(d.ts[best[0]], c.tz)}. Continuity requires recorded samples no more than {BREAK // 60} min apart.", "temporal_continuous", verdict=True, longest=bs, n=len(good))
    return _ans(f"The longest unbroken stretch with {c.sig_label()} {c.th_text()}{_scope_text(c)} {label} lasted {fmt_dur(bs)}, starting {stamp(d.ts[best[0]], c.tz)}.", "temporal_continuous", longest=bs)


def op_days_continuous(c):
    t = c.t
    if not (c.th and c.min_dur and re.search(r"\bon which days\b|\bwhich days\b|\bwhat days\b", t)): return None
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = d.state_mask(c.states)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    rs = [r for r in runs(d.ts, cm & sm) if run_seconds(d.ts, r[0], r[1]) >= c.min_dur]
    days = {}
    for a, b in rs:
        day = local_midnight(int(d.ts[a]), c.tz); days[day] = max(days.get(day, 0), run_seconds(d.ts, a, b))
    if not days: return _ans(f"On none of the days {label} was {c.sig_label()} {c.th_text()} for {fmt_dur(c.min_dur)} or more without a break.", "temporal_days", n=0)
    txt = "; ".join(f"{day_label(k, c.tz)} (longest {fmt_dur(v)})" for k, v in sorted(days.items()))
    return _ans(f"{c.sig_label().capitalize()} was {c.th_text()} for at least {fmt_dur(c.min_dur)} without a break on {len(days)} day{'' if len(days) == 1 else 's'} {label}: {txt}.", "temporal_days", n=len(days))


def op_halves(c):
    t = c.t
    m = re.search(r"\b(first|second) half\b", t)
    if not m or not re.search(r"\b(?:higher|lower|more|less|greater|faster|slower|hotter|colder|compared|than|versus|vs)\b", t): return None
    if not c.signal: return None
    lo, hi, label = c.window(); mid = (lo + hi) // 2
    A, B = (lo, mid), (mid, hi)
    da, _ = c.load([c.signal], A[0], A[1]); db, _ = c.load([c.signal], B[0], B[1])
    def mean(d):
        arr, mm = _vals_in_state(c, d)
        return (float(np.mean(arr[mm])), int(mm.sum())) if arr is not None and mm.any() else (None, 0)
    ma, na = mean(da); mb, nb = mean(db)
    if ma is None or mb is None: return _ans(f"One of the two halves of {label} has no recorded {c.sig_label()}, so they cannot be compared.", "temporal_halves", ok=False)
    first_half = m.group(1) == "first"
    subj, other = (ma, mb) if first_half else (mb, ma)
    up = re.search(r"\b(?:higher|more|greater|faster|hotter|bigger|larger)\b", t); down = re.search(r"\b(?:lower|less|slower|colder|smaller)\b", t)
    d_ = subj - other; yes = (d_ > 0) if (up or not down) else (d_ < 0)
    unit = R.CONCEPTS[c.signal].unit
    return _ans(f"{'Yes' if yes else 'No'}. The average {c.sig_label()}{_scope_text(c)} was {mb:.1f} {unit} in the second half of {label} (from {stamp(mid, c.tz)}) and {ma:.1f} {unit} in the first half (until then): a difference of {abs(mb - ma):.1f} {unit}. The period was split at its midpoint.", "temporal_halves", first=ma, second=mb)


def op_drop_vs_baseline(c):
    t = c.t
    m = re.search(r"\b(?:fall|fell|drop|dropped|decrease\w*|go)\b.{0,12}\bby (?:more than |over |at least )?" + NUMRX + r" ?(%|percent).{0,40}\b(?:relative to|compared (?:to|with)|against|from)\b.{0,25}\b(?:preceding|previous|prior|last)\b.{0,20}\b" + NUMRX + r"[- ]?(minute|min|second|hour)", t)
    if not m or not c.signal: return None
    frac = _num(m.group(1)) / 100.0; win = int(_num(m.group(3))) * (60 if m.group(4).startswith("min") else 1 if m.group(4).startswith("sec") else 3600)
    k = max(1, win // STEP)
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    arr = d.vals.get(c.signal); sm = d.state_mask(c.states)
    if arr is None or sm is None: return _nodata(c, label, c.sig_label())
    hit = np.zeros(d.n, dtype=bool); usable = 0
    for i in range(k, d.n):
        if d.ts[i] - d.ts[i - k] > win + BREAK or not sm[i] or np.isnan(arr[i]): continue
        base = arr[i - k:i]
        if np.isnan(base).any(): continue
        mu = float(base.mean()); usable += 1
        if mu > 1e-9 and arr[i] < (1.0 - frac) * mu: hit[i] = True
    rs = runs(d.ts, hit); n = len(rs)
    return _ans(f"{c.sig_label().capitalize()} fell by more than {frac * 100:g}% below its preceding {fmt_dur(win)} average on {n} separate occasion{'' if n == 1 else 's'} {label}{_scope_text(c)}. Only samples with a complete preceding {fmt_dur(win)} of recorded data and a positive baseline were tested ({usable:,} of {d.n:,}); consecutive qualifying samples count as one occasion.", "temporal_drop", n=n, tested=usable)


def op_class_duration_compare(c):
    t = c.t
    if not (c.th and re.search(r"\bmore time\b|\blonger\b|\bfor more\b|\bgreater duration\b", t) and re.search(r"\bhighway|motorway\b", t) and re.search(r"\bcity\b", t)): return None
    d, label = c.load()
    if d.n == 0 or d.mode is None: return _nodata(c, label, c.sig_label())
    cm = c.compare(d)
    hw = int((cm & (d.mode == 2)).sum()); ct = int((cm & (d.mode == 1)).sum())
    tot_h = int((d.mode == 2).sum()); tot_c = int((d.mode == 1).sum())
    first_is_hw = t.find("highway") < t.find("city") if t.find("highway") >= 0 and t.find("city") >= 0 else True
    a, b = (hw, ct) if first_is_hw else (ct, hw)
    yes = a > b
    return _ans(f"{'Yes' if yes else 'No'}. Over {label} the vehicle spent {hw} minutes {c.th_text()} on the motorway / highway (of {tot_h} recorded highway minutes) and {ct} minutes in the city (of {tot_c} recorded city minutes).", "temporal_class_compare", highway=hw, city=ct)


# ------------------------------------------------------------------ B: parking and occupancy
def _day_table(c, d):
    """Per local day: recorded minutes, parked minutes, longest parked stretch (s)."""
    days = {}
    if d.mode is None: return days
    local = ((d.ts + c.tz) // DAY) * DAY - c.tz
    for day in np.unique(local):
        idx = np.where(local == day)[0]
        md = d.mode[idx]; rec = int((~np.isnan(md)).sum()); parked = int((md == 0).sum())
        rs = runs(d.ts[idx], md == 0)
        days[int(day)] = {"recorded": rec, "parked": parked, "longest": max((run_seconds(d.ts[idx], a, b) for a, b in rs), default=0), "n": len(idx)}
    return days


def op_parked_days(c):
    t = c.t
    if not re.search(r"\bparked\b", t): return None
    if not re.search(r"\ball day\b|\bwhole day\b|\bentire\b.{0,15}\bday\b|\bfull day\b|\bcomplete days?\b|\bat least \d+ hours\b|\bat least " + NUMRX + r" hours\b|\bhow many days\b|\bwhich days\b", t): return None
    d, label = c.load([])
    if d.n == 0 or d.mode is None: return _nodata(c, label, "drive-state samples")
    tab = _day_table(c, d)
    need_h = None
    mh = re.search(r"\bat least " + NUMRX + r" hours?\b", t)
    if mh: need_h = _num(mh.group(1))
    lo, hi, _ = c.window()
    rows = []
    for day, v in sorted(tab.items()):
        span = min(DAY, hi - max(day, lo)) if day < lo else min(DAY, hi - day)
        full = v["recorded"] >= FULL_DAY * 1440
        rows.append((day, v, full))
    if need_h is not None:
        okd = [(day, v) for day, v, full in rows if v["parked"] * STEP >= need_h * 3600]
        unclass = [day for day, v, full in rows if not full and v["parked"] * STEP < need_h * 3600]
        txt = f"{len(okd)} of {len(rows)} calendar days {label} had at least {need_h:g} hours of recorded parked time"
        if okd: txt += ": " + "; ".join(f"{day_label(day, c.tz)} ({v['parked'] * STEP / 3600:.1f} h)" for day, v in okd)
        if unclass: txt += f". {len(unclass)} day{'s' if len(unclass) != 1 else ''} had too little recorded data to reach that, so the true count could be higher"
        return _ans(txt + ".", "temporal_parked_hours", n=len(okd), of=len(rows))
    allday = [(day, v) for day, v, full in rows if full and v["parked"] == v["recorded"]]
    anypark = [day for day, v, full in rows if v["parked"] > 0]
    unclass = [day for day, v, full in rows if not full]
    covmax = max((v["recorded"] for _, v, _ in rows), default=0)
    parts = [f"{len(allday)} of {len(rows)} calendar days {label} were parked for the whole day"]
    if allday: parts[0] += ": " + ", ".join(day_label(day, c.tz) for day, _ in allday)
    parts.append(f"{len(anypark)} had some parked time")
    if unclass: parts.append(f"{len(unclass)} could not be classified because fewer than {int(FULL_DAY * 100)}% of their 1,440 minutes are recorded (the best-covered day has {covmax} recorded minutes)")
    return _ans("; ".join(parts) + f". A day counts as parked all day only when at least {int(FULL_DAY * 100)}% of it is recorded and every recorded sample is parked.", "temporal_parked_days", n=len(allday), of=len(rows), unclassified=len(unclass))


def op_longest_parked(c):
    t = c.t
    if not (re.search(r"\blongest\b", t) and re.search(r"\bparked|stationary|still\b", t)): return None
    d, label = c.load([])
    if d.n == 0 or d.mode is None: return _nodata(c, label, "drive-state samples")
    rs = runs(d.ts, d.mode == 0)
    if not rs: return _ans(f"The car was not recorded as parked at any time {label}.", "temporal_longest_parked", ok=False)
    best = max(rs, key=lambda r: run_seconds(d.ts, r[0], r[1])); bs = run_seconds(d.ts, *best)
    return _ans(f"The longest continuous parked stretch {label} lasted {fmt_dur(bs)}, from {stamp(d.ts[best[0]], c.tz)} to {stamp(d.ts[best[1]] + STEP, c.tz)}. Stretches are broken wherever recorded samples are more than {BREAK // 60} min apart, so unrecorded time is never counted as parked.", "temporal_longest_parked", seconds=bs)


# ------------------------------------------------------------------ C: percentiles and distributions
_PCT_RX = re.compile(r"\bp(\d{1,2}(?:\.\d+)?)\b|\b(\d{1,2}(?:\.\d+)?) ?(?:(?:st|nd|rd|th)[- ]?(?:percentile|percentage)|[- ]?percentile)\b|\bpercentile (\d{1,2}(?:\.\d+)?)\b|\b(median)\b|\b(interquartile range|iqr)\b|\b(quartile)\b")


def percentiles_of(t):
    out = []
    for m in _PCT_RX.finditer(t):
        if m.group(4): out.append(50.0)
        elif m.group(5): out.extend([25.0, 75.0])
        else:
            v = m.group(1) or m.group(2) or m.group(3)
            if v: out.append(float(v))
    seen = []
    for v in out:
        if v not in seen: seen.append(v)
    return seen


def pct_ok(n, p):
    tail = min(p, 100 - p) / 100.0
    need = 20 if tail >= 0.5 else math.ceil(10.0 / max(tail, 1e-6))
    return n >= max(need, 20), need


def op_percentiles(c):
    t = c.t
    ps = percentiles_of(t)
    if not ps or not c.signal: return None
    if re.search(r"\b(?:compare|than|versus|vs)\b", t) and len(re.findall(r"\b(?:city|highway|motorway|charging|driving|parked)\b", t)) >= 2: return None
    if re.search(r"\bmaximum|max\b|\bhighest\b", t) and re.search(r"\bthan\b|\bgap\b|\bdifference\b", t): return None
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    arr, m = _vals_in_state(c, d)
    if arr is None or not m.any(): return _nodata(c, label, f"{c.sig_label()}{_scope_text(c)}")
    if re.search(r"\bexcluding stationary\b|\bnot stationary\b|\bwhile moving\b|\bmoving\b", t) and "speed" in d.vals:
        sp = d.vals.get("speed")
        if sp is None:
            d2 = Data(c.qc, [c.signal, "speed"], d.lo, d.hi); d = d2; arr = d.vals[c.signal]; m = d.state_mask(c.states) & ~np.isnan(arr)
            sp = d.vals.get("speed")
        if sp is not None: m = m & (sp >= 1.0)
    vals = arr[m]; n = len(vals); unit = R.CONCEPTS[c.signal].unit
    lines = []; reported = {}
    for p in ps:
        ok, need = pct_ok(n, p)
        name = "median" if p == 50 else f"{p:g}th percentile"
        if not ok:
            lines.append(f"the {name} cannot be estimated reliably from {n:,} samples (a {p:g}th percentile needs at least {need:,})")
        else:
            val = float(np.percentile(vals, p, method="linear")); reported[p] = val
            lines.append(f"{name} {val:.1f} {unit}")
    extra = ""
    if re.search(r"\binterquartile|\biqr\b", t) and 25.0 in reported and 75.0 in reported:
        extra = f" The interquartile range is {reported[75.0] - reported[25.0]:.1f} {unit} (p75 {reported[75.0]:.1f} minus p25 {reported[25.0]:.1f})."
    st = _scope_text(c)
    if re.search(r"\bexcluding stationary\b|\bnot stationary\b", t): st += ", excluding stationary samples (speed below 1 km/h)"
    return _ans(f"{c.sig_label().capitalize()}{st} {label}: " + "; ".join(lines) + f".{extra} Basis: {n:,} valid recorded one-minute samples, linear-interpolation percentiles.", "temporal_percentiles", values=reported, n=n)


def op_percentile_compare(c):
    t = c.t
    ps = percentiles_of(t)
    if not ps or not c.signal: return None
    if not re.search(r"\b(?:than|compare|versus|vs|higher|lower|greater|compared)\b", t): return None
    names = [m.group(0) for m in re.finditer(r"\b(city|highway|motorway|charging|driving|parked)\b", t)]
    uniq = []
    for n_ in names:
        k = "highway" if n_ == "motorway" else n_
        if k not in uniq: uniq.append(k)
    if len(uniq) < 2: return None
    a, b = uniq[0], uniq[1]
    d, label = c.load()
    if d.n == 0 or d.mode is None: return _nodata(c, label, c.sig_label())
    arr = d.vals.get(c.signal)
    if arr is None: return _nodata(c, label, c.sig_label())
    res = {}
    for k in (a, b):
        mk = {"city": d.mode == 1, "highway": d.mode == 2, "charging": d.mode == 3, "parked": d.mode == 0, "driving": np.isin(d.mode, (1, 2))}[k] & ~np.isnan(arr)
        res[k] = arr[mk]
    p = ps[0]; unit = R.CONCEPTS[c.signal].unit; name = "median" if p == 50 else f"{p:g}th-percentile"
    for k in (a, b):
        ok, need = pct_ok(len(res[k]), p)
        if not ok: return _ans(f"There are only {len(res[k]):,} recorded {k} samples {label}, too few for a {p:g}th percentile (needs {need:,}).", "temporal_pct_compare", ok=False)
    va, vb = float(np.percentile(res[a], p)), float(np.percentile(res[b], p))
    up = re.search(r"\b(?:higher|greater|more|faster|hotter)\b", t); down = re.search(r"\b(?:lower|less|slower|colder)\b", t)
    polar = re.match(POLAR, t.strip()) and (up or down)
    lead = ""
    if polar:
        yes = (va > vb) if up or not down else (va < vb)
        lead = "Yes. " if yes else "No. "
    return _ans(lead + f"The {name} {c.sig_label()} {label} was {va:.1f} {unit} {'while ' + a if a in ('charging', 'driving', 'parked') else 'in ' + a + ' driving'} against {vb:.1f} {unit} {'while ' + b if b in ('charging', 'driving', 'parked') else 'in ' + b + ' driving'} (n = {len(res[a]):,} and {len(res[b]):,} samples; the two populations do not overlap).", "temporal_pct_compare", a=va, b=vb)


def op_max_vs_percentile(c):
    t = c.t
    ps = percentiles_of(t)
    if not ps or not c.signal or not re.search(r"\b(?:maximum|max|highest|peak)\b", t): return None
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    arr, m = _vals_in_state(c, d)
    if arr is None or not m.any(): return _nodata(c, label, c.sig_label())
    vals = arr[m]; mx = float(vals.max()); p = ps[0]; ok, need = pct_ok(len(vals), p); unit = R.CONCEPTS[c.signal].unit
    if not ok: return _ans(f"Too few samples ({len(vals):,}) for a {p:g}th percentile.", "temporal_max_vs_pct", ok=False)
    pv = float(np.percentile(vals, p)); gap = mx - pv
    if re.search(r"\bthan\b|\bhigher\b|\bgreater\b", t) and re.match(POLAR, t.strip()):
        yes = mx > pv
        return _ans(f"{'Yes' if yes else 'No'}. The maximum {c.sig_label()}{_scope_text(c)} {label} was {mx:.1f} {unit} and the {p:g}th percentile {pv:.1f} {unit} (gap {gap:.1f} {unit}; {len(vals):,} samples). The maximum is by definition at least as large as any percentile of the same samples.", "temporal_max_vs_pct", max=mx, p=pv, verdict=yes)
    return _ans(f"The gap between the maximum {c.sig_label()}{_scope_text(c)} and its {p:g}th percentile {label} was {gap:.1f} {unit}: maximum {mx:.1f} {unit}, {p:g}th percentile {pv:.1f} {unit} ({len(vals):,} samples).", "temporal_max_vs_pct", gap=gap)


OPS = [op_parked_days, op_longest_parked, op_percentile_compare, op_max_vs_percentile, op_percentiles, op_halves, op_drop_vs_baseline, op_days_continuous, op_continuous,
       op_class_duration_compare, op_episodes, op_average_vs_threshold, op_share, op_duration]


_HYPOTHETICAL = re.compile(r"\bwould (?:not |never |still )?have\b|\bwould (?:the|it|that|this) (?:fault|warning|event|alarm|dtc)\b|\bhad (?:i|we|the driver|it) (?:driven|braked|charged|accelerated|parked)\b", re.I)


def handle(brain, qc, t):
    t = re.sub(r"(\d) ?: ?(\d)", lambda m: m.group(1) + ":" + m.group(2), t); t = re.sub(r" ([,;.?!])", lambda m: m.group(1), t); t = re.sub(r" ?/ ?", "/", t); t = re.sub(r"(\d) %", lambda m: m.group(1) + "%", t); t = re.sub(r"° ?c\b", "°c", t)       # the spell-repaired form spaces these out
    if _HYPOTHETICAL.search(t): return None            # a statement about a world that did not happen is not a measurement query: the counterfactual / epistemic routes own it
    c = Ctx(brain, qc, t)
    try:
        from . import temporal_events, temporal_misc
        mods = (temporal_misc, temporal_events)
    except Exception as e:
        import logging; logging.getLogger("vehiclegpt2").warning("temporal extension unavailable: %s", e); mods = ()
    for mod in mods:                                   # the specific joins and cross-source operations first; the generic threshold operations below
        try: r = mod.handle(brain, qc, t, c)
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("temporal %s failed: %s: %s", mod.__name__, type(e).__name__, e); r = None
        if r: return r
    for fn in OPS:
        try: r = fn(c)
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("temporal op %s failed: %s: %s", fn.__name__, type(e).__name__, e); r = None
        if r: return r
    return None
