"""Deterministic executor of the plan operations (query_ir.PLAN_OPS).

Input is a validated Query - never text.  Every decision that used to be taken by matching words in the question (which operation, which threshold, which states, which window, which
percentiles, ...) has already been made by the interpreter and is present as typed fields:

    target / target2, agg, scope.time (+ scope_b.time), scope.conditions (named states with `not`, signal thresholds), groups, event, scope.select, and Query.params (PLAN_PARAMS).

The arithmetic here is the domain policy of vgpt/temporal.py (one-minute samples, 120 s continuity gap, 90 % day coverage, type-7 percentiles with a minimum sample, one drive state at a
time with unknown staying unknown, 21:00-06:00 night).  A plan that cannot be executed on the recorded data returns an honest explanation; nothing is guessed or substituted."""
import datetime as dt
import math
import re

import numpy as np

from . import signal_registry as R
from .query_ir import Query, TimeSpec, Condition, EventRef
from .query_planner import resolve_time, local_midnight, DAY
from .temporal import STEP, BREAK, FULL_DAY, MODE, Data, runs, run_seconds, fmt_dur, stamp, day_label, describe_states, to_signal_unit
from . import composed as C

OPS_WITH_EVENT = ("count", "list_events")


def _ans(text, route, **facts): return C._result(text, route, **facts)


class PlanCtx:
    """The typed arguments of one plan, shaped like the legacy parsed context so that the shared arithmetic is the same."""

    def __init__(self, brain, qc, q):
        self.brain, self.qc, self.q = brain, qc, q; self.tz = qc.ctx.tz; self.t = ""
        self.signal = q.target; self.signal2 = q.target2; self.agg = q.agg; self.params = dict(q.params or {})
        self.states = [c for c in q.scope.conditions]
        self.spec = q.scope.time; self.spec2 = q.scope_b.time if q.scope_b is not None else None
        self.th = None
        thr = self.params.get("thr")
        if thr and self.signal:
            op, _, v = str(thr).partition(":")
            try:
                raw = float(v); unit = (q.unit or "").replace("_per_", "/") or None
                self.th = (op, to_signal_unit(raw, unit, self.signal), raw, unit)
            except ValueError: self.th = None
        try: self.min_dur = int(float(self.params["mind"])) if self.params.get("mind") not in (None, "", "0") else None
        except ValueError: self.min_dur = None

    # ---- window and data
    def window(self, spec=None):
        spec = spec or self.spec
        lo, hi, label = resolve_time(spec, self.qc.ctx)
        return lo, min(hi, self.qc.ctx.now + 1), label

    def load(self, concepts=None, spec=None, lo=None, hi=None):
        l, h, label = self.window(spec)
        # A plan is allowed to use this compatibility executor, but it must
        # retain the complete typed scope.  In particular condition signals
        # must be loaded and a trip/session selection must constrain the
        # samples rather than merely change a label.
        scope = self.q.scope_b if spec is self.spec2 and self.q.scope_b is not None else self.q.scope
        base = set(concepts if concepts is not None else ([self.signal] if self.signal else []))
        base |= self.qc.executor._concepts_for(self.q)
        d = Data(self.qc, base, l if lo is None else lo, h if hi is None else hi)
        d.plan_conditions = list(scope.conditions)
        d.scope_mask = np.ones(d.n, dtype=bool)
        if scope.select is not None and d.n:
            intervals, _ents, _label, _period = self.qc.executor._intervals(scope, None)
            d.scope_mask = self.qc.executor._interval_mask(d.frame, intervals)
        return d, label

    def compare(self, d, arr=None):
        arr = d.vals.get(self.signal) if arr is None else arr
        if arr is None or self.th is None: return None
        op, v = self.th[0], self.th[1]
        return {"gt": arr > v, "ge": arr >= v, "lt": arr < v, "le": arr <= v, "eq": np.isclose(arr, v, rtol=0.0, atol=1e-9)}[op] & ~np.isnan(arr)

    def th_text(self):
        op, v, raw, u = self.th; unit = u or (R.CONCEPTS[self.signal].unit if self.signal in R.CONCEPTS else "")
        return f"{ {'gt': 'above', 'ge': 'at least', 'lt': 'below', 'le': 'at most', 'eq': 'equal to'}[op] } {raw:g} {unit}".strip()

    def sig_label(self): return R.label_of(self.signal) if self.signal else "the signal"
    def state_text(self): return (" " + describe_states([x for x in self.states if x.predicate])) if any(x.predicate for x in self.states) else ""
    def mask(self, d):
        m = d.state_mask(getattr(d, "plan_conditions", self.states))
        return None if m is None else m & getattr(d, "scope_mask", np.ones(d.n, dtype=bool))
    def need_signal(self):
        if self.signal not in R.CONCEPTS: return _ans("I could not tell which measurement that is about, so I will not guess one.", "plan_clarify", ok=False)
        return None


def _nodata(c, label, what): return _ans(f"There is no recorded {what} {label}, so I cannot answer that.", "temporal_nodata", ok=False)


def _vals(c, d):
    arr = d.vals.get(c.signal)
    if arr is None: return None, None
    m = c.mask(d)
    if m is None: return None, None
    return arr, m & ~np.isnan(arr)


def _pct_ok(n, p):
    tail = min(p, 100 - p) / 100.0
    need = 20 if tail >= 0.5 else math.ceil(10.0 / max(tail, 1e-6))
    return n >= max(need, 20), need


# ------------------------------------------------------------------ threshold family
def run_exists(c):
    d, label = c.load()
    arr, m = _vals(c, d)
    if arr is None or d.n == 0 or not m.any(): return _nodata(c, label, f"{c.sig_label()}{c.state_text()}")
    cm = c.compare(d) & m
    hi = c.th[0] in ("gt", "ge"); vals = np.where(m, arr, np.nan); i = int(np.nanargmax(vals) if hi else np.nanargmin(vals)); v = float(arr[i]); unit = R.CONCEPTS[c.signal].unit
    word = "highest" if hi else "lowest"
    if cm.any():
        return _ans(f"Yes. The {word} {c.sig_label()}{c.state_text()} {label} was {v:.1f} {unit}, on {stamp(d.ts[i], c.tz)} (of one-minute averages); it was {c.th_text()} in {int(cm.sum())} recorded samples.", "composed_exists", verdict=True, value=v, target=c.signal)
    return _ans(f"No - it was never {c.th_text()}{c.state_text()}. The {word} {c.sig_label()} {label} was {v:.1f} {unit}, on {stamp(d.ts[i], c.tz)} (of one-minute averages).", "composed_exists", verdict=False, value=v, target=c.signal)


def run_threshold_test(c):
    d, label = c.load(); arr, m = _vals(c, d)
    if arr is None or d.n == 0 or not m.any(): return _nodata(c, label, f"{c.sig_label()}{c.state_text()}")
    agg = c.agg or "mean"; v = float({"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}[agg](arr[m])); op = c.th[0]
    ok = {"gt": v > c.th[1], "ge": v >= c.th[1], "lt": v < c.th[1], "le": v <= c.th[1], "eq": math.isclose(v, c.th[1], rel_tol=0.0, abs_tol=1e-9)}[op]; unit = R.CONCEPTS[c.signal].unit
    name = {"mean": "average", "max": "highest", "min": "lowest", "median": "median"}[agg]
    return _ans(f"{'Yes' if ok else 'No'}. The {name} {c.sig_label()}{c.state_text()} {label} was {v:.1f} {unit}, which is {'' if ok else 'not '}{c.th_text()} (from {int(m.sum()):,} recorded one-minute samples).", "temporal_avg_vs_threshold", value=v, verdict=bool(ok))


def run_duration(c):
    d, label = c.load(([c.signal] if c.th and c.signal else []))
    if d.n == 0: return _nodata(c, label, "samples")
    sm = c.mask(d)
    if sm is None: return _ans(f"The drive state is not recorded for {label}, so I cannot apply the condition ({describe_states(c.states)}).", "temporal_duration", ok=False)
    m = sm.copy()
    if c.th:
        cm = c.compare(d)
        if cm is None: return _nodata(c, label, c.sig_label())
        m &= cm
    n = int(m.sum()); sec = n * STEP
    pre = [x for x in c.states if x.predicate]
    if c.th: subj = f"{c.sig_label().capitalize()} was {c.th_text()}" + (f" while {describe_states(pre)}" if pre else "")
    else: subj = f"The car was {describe_states(pre)}"
    unknown = int(np.isnan(d.mode).sum()) if d.mode is not None else d.n
    extra = f" Samples with no recorded drive state: {unknown}." if pre and unknown else ""
    overlap = ""
    if len(pre) >= 2 and all(x.op != "not" for x in pre) and len({x.predicate for x in pre if x.predicate in MODE}) >= 2 and n == 0:
        overlap = " This vehicle records one drive state at a time (parked, city, highway or charging), so two of those states cannot overlap in its data."
    return _ans(f"{subj} for {fmt_dur(sec)} ({sec / 3600.0:.1f} hours) {label}. Each recorded one-minute sample counts as 60 s; {n:,} of {d.n:,} recorded samples qualified.{extra}{overlap}", "temporal_duration", seconds=sec, samples=n)


def run_share(c):
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, "samples")
    den = c.mask(d)
    if den is None: return _ans(f"The drive state is not recorded for {label}.", "temporal_share", ok=False)
    dn = c.params.get("den")                                    # an explicitly named denominator state, or None / 'all'
    if dn in (None, ""):
        named = [x.predicate for x in c.states if x.op != "not" and x.predicate in MODE]
        dn = named[0] if named else "driving"                    # the default denominator: the one drive state the question names, else driving time
    if dn != "all":
        inner = d.state_mask([Condition(predicate="driving" if dn in ("moving", "driving") else dn)])
        if inner is None: return _ans("The drive state is not recorded, so that denominator cannot be built.", "temporal_share", ok=False)
        den = den & inner
    den_note = "all recorded samples" if dn == "all" else f"{dn if dn != 'driving' else 'driving'} time"
    cmp_ = c.compare(d)
    if cmp_ is None: return _nodata(c, label, c.sig_label())                      # the measured signal has no recorded series: an explicit "no recorded <signal>", never an exception
    num = den & cmp_; nd = int(den.sum())
    if nd == 0: return _ans(f"There are no recorded samples of {den_note}{c.state_text()} {label}.", "temporal_share", ok=False)
    pct = 100.0 * int(num.sum()) / nd
    shown = [x for x in c.states if x.predicate and not (x.predicate == dn and x.op != "not")]
    scope = ""
    if shown: scope = ", " + " and ".join((("excluding the time " + describe_states([Condition(predicate=x.predicate)])) if x.op == "not" else ("only while " + describe_states([x]))) for x in shown)
    return _ans(f"{pct:.1f}% of the {den_note}{scope} {label} was spent with {c.sig_label()} {c.th_text()} ({int(num.sum()):,} of {nd:,} recorded one-minute samples, about {int(num.sum())} minutes).", "temporal_share", pct=pct, num=int(num.sum()), den=nd)


def run_episodes(c):
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = c.mask(d)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    rs = runs(d.ts, cm & sm); n = len(rs); longest = max((run_seconds(d.ts, a, b) for a, b in rs), default=0)
    first = f" The first began at {stamp(d.ts[rs[0][0]], c.tz)}." if rs else ""
    return _ans(f"{c.sig_label().capitalize()} went {c.th_text()} on {n} separate occasion{'' if n == 1 else 's'} {label}{c.state_text()}; the longest lasted {fmt_dur(longest)}.{first} An occasion is a stretch of consecutive recorded samples (no gap over {BREAK // 60} min), counted once however long it lasts.",
                "temporal_episodes", n=n, longest=longest)


def run_continuous(c):
    need = c.min_dur or 0
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = c.mask(d)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    rs = runs(d.ts, cm & sm)
    if not rs: return _ans(f"No. {c.sig_label().capitalize()} was never {c.th_text()}{c.state_text()} {label} in the recorded samples.", "temporal_continuous", verdict=False)
    best = max(rs, key=lambda r: run_seconds(d.ts, r[0], r[1])); bs = run_seconds(d.ts, *best)
    good = [r for r in rs if run_seconds(d.ts, r[0], r[1]) >= need]
    if need and not good:
        return _ans(f"No. The longest stretch with {c.sig_label()} {c.th_text()}{c.state_text()} {label} lasted {fmt_dur(bs)} ({stamp(d.ts[best[0]], c.tz)}), shorter than the {fmt_dur(need)} asked about. A single qualifying sample is one minute and does not count as continuous.", "temporal_continuous", verdict=False, longest=bs)
    if need:
        return _ans(f"Yes. {c.sig_label().capitalize()} stayed {c.th_text()}{c.state_text()} for at least {fmt_dur(need)} on {len(good)} occasion{'' if len(good) == 1 else 's'} {label}; the longest was {fmt_dur(bs)} starting {stamp(d.ts[best[0]], c.tz)}. Continuity requires recorded samples no more than {BREAK // 60} min apart.", "temporal_continuous", verdict=True, longest=bs, n=len(good))
    return _ans(f"The longest unbroken stretch with {c.sig_label()} {c.th_text()}{c.state_text()} {label} lasted {fmt_dur(bs)}, starting {stamp(d.ts[best[0]], c.tz)}.", "temporal_continuous", longest=bs)


def run_days_with(c):
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    cm = c.compare(d); sm = c.mask(d)
    if cm is None or sm is None: return _nodata(c, label, c.sig_label())
    need = c.min_dur or 0
    rs = [r for r in runs(d.ts, cm & sm) if run_seconds(d.ts, r[0], r[1]) >= max(need, STEP)]
    days = {}
    for a, b in rs:
        start, end = int(d.ts[a]), int(d.ts[b]) + STEP
        day = local_midnight(start, c.tz)
        while day < end:
            overlap = max(0, min(end, day + DAY) - max(start, day))
            if overlap: days[day] = max(days.get(day, 0), overlap)
            day += DAY
    dur = f" for at least {fmt_dur(need)} without a break" if need else ""
    listing = str(c.params.get("list", "0")) == "1"
    nd = len({int(x) for x in ((d.ts + c.tz) // DAY)})
    if not days: return _ans(f"On none of the {nd} calendar days {label} was {c.sig_label()} {c.th_text()}{dur}.", "temporal_days", n=0)
    if listing:
        txt = "; ".join(f"{day_label(k, c.tz)} (longest {fmt_dur(v)})" for k, v in sorted(days.items()))
        return _ans(f"{c.sig_label().capitalize()} was {c.th_text()}{dur} on {len(days)} day{'' if len(days) == 1 else 's'} {label}: {txt}.", "temporal_days", n=len(days))
    return _ans(f"{c.sig_label().capitalize()} was {c.th_text()}{dur} on {len(days)} of {nd} calendar days {label}.", "temporal_days", n=len(days), of=nd)


# ------------------------------------------------------------------ distributions
def _percentile_values(c, arr_mask_pairs, ps):
    out = {}
    for p in ps:
        ok, need = _pct_ok(len(arr_mask_pairs), p)
        out[p] = (float(np.percentile(arr_mask_pairs, p, method="linear")) if ok else None, need)
    return out


def run_percentile(c):
    try: ps = [float(x) for x in str(c.params.get("pct", "")).split(",") if x != ""]
    except ValueError: ps = []
    if not ps: return _ans("Which percentile do you mean?", "plan_clarify", ok=False)
    if any(not math.isfinite(p) or p < 0 or p > 100 for p in ps): return _ans("A percentile must be between 0 and 100.", "plan_invalid", ok=False)
    d, label = c.load()
    if d.n == 0: return _nodata(c, label, c.sig_label())
    arr, m = _vals(c, d)
    if arr is None or not m.any(): return _nodata(c, label, f"{c.sig_label()}{c.state_text()}")
    if any(x.predicate == "idling" and x.op == "not" for x in c.states) and "speed" in d.vals:
        pass
    vals = arr[m]; n = len(vals); unit = R.CONCEPTS[c.signal].unit; reported = {}; lines = []
    for p in ps:
        ok, need = _pct_ok(n, p); name = "median" if p == 50 else f"{p:g}th percentile"
        if not ok: lines.append(f"the {name} cannot be estimated reliably from {n:,} samples (a {p:g}th percentile needs at least {need:,})")
        else: reported[p] = float(np.percentile(vals, p, method="linear")); lines.append(f"{name} {reported[p]:.1f} {unit}")
    extra = f" The interquartile range is {reported[75.0] - reported[25.0]:.1f} {unit} (p75 {reported[75.0]:.1f} minus p25 {reported[25.0]:.1f})." if str(c.params.get("iqr", "0")) == "1" and 25.0 in reported and 75.0 in reported else ""
    return _ans(f"{c.sig_label().capitalize()}{c.state_text()} {label}: " + "; ".join(lines) + f".{extra} Basis: {n:,} valid recorded one-minute samples, linear-interpolation percentiles.", "temporal_percentiles", values=reported, n=n)


def run_percentile_vs(c):
    ps = [float(x) for x in str(c.params.get("pct", "50")).split(",") if x]
    if len(c.q.groups) < 2: return _ans("Two states are needed to compare.", "plan_clarify", ok=False)
    a, b = c.q.groups[:2]; p = ps[0]
    supported = {"city", "highway", "charging", "parked", "driving"}
    if a not in supported or b not in supported: return _ans("That state is not supported for a percentile comparison.", "plan_invalid", ok=False)
    if not math.isfinite(p) or not 0 <= p <= 100: return _ans("A percentile must be between 0 and 100.", "plan_invalid", ok=False)
    d, label = c.load()
    if d.n == 0 or d.mode is None: return _nodata(c, label, c.sig_label())
    arr = d.vals.get(c.signal)
    if arr is None: return _nodata(c, label, c.sig_label())
    res = {}
    for k in (a, b):
        mk = {"city": d.mode == 1, "highway": d.mode == 2, "charging": d.mode == 3, "parked": d.mode == 0, "driving": np.isin(d.mode, (1, 2))}[k] & ~np.isnan(arr)
        res[k] = arr[mk]
    unit = R.CONCEPTS[c.signal].unit; name = "median" if p == 50 else f"{p:g}th-percentile"
    for k in (a, b):
        ok, need = _pct_ok(len(res[k]), p)
        if not ok: return _ans(f"There are only {len(res[k]):,} recorded {k} samples {label}, too few for a {p:g}th percentile (needs {need:,}).", "temporal_pct_compare", ok=False)
    va, vb = float(np.percentile(res[a], p)), float(np.percentile(res[b], p))
    direction = c.params.get("dir"); lead = ""
    if direction in ("higher", "lower"): lead = "Yes. " if ((va > vb) if direction == "higher" else (va < vb)) else "No. "
    desc = lambda k: ("while " + k) if k in ("charging", "driving", "parked") else ("in " + k + " driving")
    return _ans(lead + f"The {name} {c.sig_label()} {label} was {va:.1f} {unit} {desc(a)} against {vb:.1f} {unit} {desc(b)} (n = {len(res[a]):,} and {len(res[b]):,} samples; the two populations do not overlap).", "temporal_pct_compare", a=va, b=vb)


def run_range(c):
    d, label = c.load(); arr, m = _vals(c, d)
    if arr is None or d.n == 0 or not m.any(): return _nodata(c, label, f"{c.sig_label()}{c.state_text()}")
    v = arr[m]; unit = R.CONCEPTS[c.signal].unit; st = f"While the car was {describe_states([x for x in c.states if x.predicate])} " if any(x.predicate for x in c.states) else "Over "
    return _ans(f"{st}{label}, {c.sig_label()} ranged from {v.min():.1f} to {v.max():.1f} {unit}: a range of {v.max() - v.min():.1f} {unit} ({len(v):,} recorded samples).", "temporal_range_state", range=float(v.max() - v.min()))


# ------------------------------------------------------------------ comparisons between windows / halves / days
def _mean_in(c, spec, agg="mean"):
    d, label = c.load(spec=spec); arr, m = _vals(c, d)
    if arr is None or d.n == 0 or not m.any(): return None, 0, label
    f = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median, "sum": np.sum}[agg]
    return float(f(arr[m])), int(m.sum()), label


def run_window_change(c):
    agg = c.agg or "mean"
    if c.spec2 is None: return _ans("Which two periods should I compare?", "plan_clarify", ok=False)
    va, na, la = _mean_in(c, c.spec, agg); vb, nb, lb = _mean_in(c, c.spec2, agg)
    if va is None or vb is None:
        miss = la if va is None else lb
        return _ans(f"I can't compare the two periods: there is no recorded {c.sig_label()} {miss}, so I won't offer a difference.", "composed_window_change", target=c.signal, missing=miss)
    u = R.CONCEPTS[c.signal].unit; d_ = va - vb; word = {"mean": "average", "max": "highest", "min": "lowest", "median": "median", "sum": "total"}[agg]
    pct = f" ({d_ / vb * 100:+.0f}%)" if vb not in (0, 0.0) and u != "%" else ""
    verdict = ""
    direction = c.params.get("dir")
    if direction in ("higher", "lower"):
        if abs(d_) < max(0.05, abs(vb) * 0.005): verdict = "No, they were essentially the same. "
        else: verdict = "Yes. " if ((d_ > 0) == (direction == "higher")) else "No. "
    return _ans(verdict + f"The {word} {c.sig_label()} was {va:.1f} {u} {la} and {vb:.1f} {u} {lb}: a difference of {abs(d_):.1f} {u}, {'higher' if d_ > 0 else 'lower'} {la}{pct}.", "composed_window_change", target=c.signal, a=va, b=vb)


def run_halves(c):
    lo, hi, label = c.window(); mid = (lo + hi) // 2
    da, _ = c.load([c.signal], lo=lo, hi=mid); db, _ = c.load([c.signal], lo=mid, hi=hi)
    agg = c.agg or "mean"
    def aggregate(d):
        arr, m = _vals(c, d)
        f = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median}.get(agg)
        return (float(f(arr[m])), int(m.sum())) if f is not None and arr is not None and m.any() else (None, 0)
    ma, _na = aggregate(da); mb, _nb = aggregate(db)
    if ma is None or mb is None: return _ans(f"One of the two halves of {label} has no recorded {c.sig_label()}, so they cannot be compared.", "temporal_halves", ok=False)
    part = c.params.get("part", "second"); subj, other = (mb, ma) if part == "second" else (ma, mb)
    direction = c.params.get("dir", "higher"); d_ = subj - other; yes = (d_ > 0) if direction != "lower" else (d_ < 0); unit = R.CONCEPTS[c.signal].unit
    stat = {"mean": "average", "min": "lowest", "max": "highest", "median": "median"}[agg]
    return _ans(f"{'Yes' if yes else 'No'}. The {stat} {c.sig_label()}{c.state_text()} was {mb:.1f} {unit} in the second half of {label} (from {stamp(mid, c.tz)}) and {ma:.1f} {unit} in the first half (until then): a difference of {abs(mb - ma):.1f} {unit}. The period was split at its midpoint.", "temporal_halves", first=ma, second=mb)


def run_by_period(c):
    by = c.params.get("by", "day"); agg = c.agg or "mean"; desc = c.params.get("ord", "desc") == "desc"
    d, label = c.load()
    if d.n == 0 or c.signal not in d.vals or d.vals[c.signal] is None: return _nodata(c, label, c.sig_label())
    arr = d.vals[c.signal]; sm = c.mask(d)
    if sm is None: return _ans(f"The requested state cannot be evaluated {label}.", "temporal_by_period", ok=False)
    ok = ~np.isnan(arr) & sm
    if by == "weekday":
        key = ((d.ts + c.tz) // DAY + 3) % 7; names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        rows = [(int(k), float(arr[(key == k) & ok].mean()), int(((key == k) & ok).sum())) for k in range(7) if ((key == k) & ok).any()]
        if len(rows) < 2: return _ans(f"Fewer than two weekdays have recorded data {label}.", "temporal_weekday", ok=False)
        hi_r = max(rows, key=lambda r: r[1]); lo_r = min(rows, key=lambda r: r[1]); unit = R.CONCEPTS[c.signal].unit
        best, worst = (hi_r, lo_r) if desc else (lo_r, hi_r)
        return _ans(f"{names[best[0]]} had the {'highest' if desc else 'lowest'} average {c.sig_label()} {label}: {best[1]:.1f} {unit}, which is {abs(best[1] - worst[1]):.1f} {unit} {'higher' if desc else 'lower'} than the {'lowest' if desc else 'highest'} weekday ({names[worst[0]]}, {worst[1]:.1f} {unit}). All weekdays: " + "; ".join(f"{names[k]} {v:.1f}" for k, v, _n in rows) + ".", "temporal_weekday", best=names[best[0]], worst=names[worst[0]])
    local = ((d.ts + c.tz) // DAY) * DAY - c.tz; f = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}[agg]
    vals = []; part = set()
    lo, hi, _ = c.window()
    for day in np.unique(local):
        m = (local == day) & ok
        if m.any():
            day_lo, day_hi = max(lo, int(day)), min(hi, int(day) + DAY)
            expected = max(1, math.ceil((day_hi - day_lo) / STEP))
            covered = int(m.sum())
            # Ranking a one-sample partial day beside a full day is not a
            # defensible daily comparison.  Boundary days remain visible but
            # are not eligible unless they meet the same 90% coverage rule.
            if covered < math.ceil(expected * 0.9):
                part.add(int(day)); continue
            vals.append((int(day), float(f(arr[m]))));
            if (day_hi - day_lo) < DAY - 60: part.add(int(day))
    if len(vals) < 2: return _ans(f"Fewer than two days have recorded {c.sig_label()} {label}.", "composed_argday", ok=False)
    k = int(c.params.get("k", 1) or 1); order = sorted(vals, key=lambda r: r[1], reverse=desc); best = order[0]; unit = R.CONCEPTS[c.signal].unit
    word = ("highest" if desc else "lowest") + (" average" if agg == "mean" else "")
    top = order[:k]
    if k > 1:
        return _ans(f"The {k} days with the {word} {c.sig_label()} {label}: " + "; ".join(f"{day_label(d_, c.tz)} {v:.1f} {unit}" for d_, v in top) + ".", "composed_argday", days=[d_ for d_, _ in top])
    return _ans(f"The day with the {word} {c.sig_label()} {label} was {day_label(best[0], c.tz)}, at {best[1]:.1f} {unit}.\nPer day: " + "; ".join(f"{day_label(d_, c.tz)} {v:.1f} {unit}" + (" (part of the day)" if d_ in part else "") for d_, v in sorted(vals)) + ".", "composed_argday", target=c.signal, best_day=best[0], value=best[1], per_day=len(vals))


def run_pair_gap(c):
    a, b = c.signal, c.signal2
    if a not in R.CONCEPTS or b not in R.CONCEPTS: return _ans("Which two measurements should I compare?", "plan_clarify", ok=False)
    d, label = c.load([a, b])
    va, vb = d.vals.get(a), d.vals.get(b)
    if va is None or vb is None or d.n == 0: return _nodata(c, label, f"{R.label_of(a)} and {R.label_of(b)}")
    unit, unit_b = R.CONCEPTS[a].unit, R.CONCEPTS[b].unit
    if R.unit_family(unit) != R.unit_family(unit_b):
        return _ans(f"{R.label_of(a)} ({unit}) and {R.label_of(b)} ({unit_b}) are different physical quantities, so I cannot subtract them.", "plan_invalid", ok=False)
    if unit != unit_b:
        try:
            vb = np.asarray([R.convert(float(x), unit_b, unit) if not np.isnan(x) else np.nan for x in vb], dtype=float)
        except (TypeError, ValueError):
            return _ans("Those measurements cannot be converted into a common unit.", "plan_invalid", ok=False)
    if c.params.get("by") == "day":
        diff = np.abs(va - vb); sm = c.mask(d); ok = ~np.isnan(diff) & (sm if sm is not None else False); local = ((d.ts + c.tz) // DAY) * DAY - c.tz; rows = []
        for day in np.unique(local):
            m = (local == day) & ok
            if m.any():
                i = int(np.argmax(np.where(m, diff, -1))); rows.append((int(day), float(diff[i]), int(d.ts[i]), int(m.sum())))
        if not rows: return _nodata(c, label, "paired readings")
        if c.params.get("ord"):
            best = max(rows, key=lambda r: r[1]) if c.params.get("ord") == "desc" else min(rows, key=lambda r: r[1])
            return _ans(f"The {R.label_of(a)}-to-{R.label_of(b)} difference was {'greatest' if c.params.get('ord') == 'desc' else 'smallest'} on {day_label(best[0], c.tz)}: {best[1]:.1f} {unit} at {stamp(best[2], c.tz)}. Each day's value is the largest absolute difference between the two recorded in the same minute ({best[3]} paired samples that day).", "temporal_pair_day", day=best[0], diff=best[1])
        return _ans(f"Largest {R.label_of(a)}-to-{R.label_of(b)} difference per day {label} (absolute, same-minute pairs): " + "; ".join(f"{day_label(d_, c.tz)} {v:.1f} {unit}" for d_, v, _t, _n in rows) + ".", "temporal_pair_day", days=len(rows))
    sm = c.mask(d); ok = ~np.isnan(va) & ~np.isnan(vb) & (sm if sm is not None else False)
    if not ok.any(): return _nodata(c, label, "paired readings")
    ma, mb = float(va[ok].mean()), float(vb[ok].mean()); dd = ma - mb
    if abs(dd) < 0.05: return _ans(f"The {R.label_of(a)} and the {R.label_of(b)} were the same on average {label} ({ma:.1f} {unit}).", "composed_signal_gap", diff=dd)
    return _ans(f"The {R.label_of(a)} was {abs(dd):.1f} {unit} {'higher' if dd > 0 else 'lower'} than the {R.label_of(b)} on average {label}: {ma:.1f} {unit} against {mb:.1f} {unit}.", "composed_signal_gap", a=a, b=b, diff=dd)


# ------------------------------------------------------------------ parking / runs
def _day_table(c, d):
    days = {}
    if d.mode is None: return days
    local = ((d.ts + c.tz) // DAY) * DAY - c.tz
    for day in np.unique(local):
        idx = np.where(local == day)[0]; md = d.mode[idx]; rec = int((~np.isnan(md)).sum()); parked = int((md == 0).sum()); rs = runs(d.ts[idx], md == 0)
        days[int(day)] = {"recorded": rec, "parked": parked, "longest": max((run_seconds(d.ts[idx], a, b) for a, b in rs), default=0)}
    return days


def run_parked_days(c):
    d, label = c.load([])
    if d.n == 0 or d.mode is None: return _nodata(c, label, "drive-state samples")
    tab = _day_table(c, d); mode = c.params.get("mode", "all_day")
    # The denominator is the requested calendar interval, not merely the days
    # where a row happened to be stored.  Missing days are explicitly
    # unclassified so sparse telemetry cannot masquerade as full coverage.
    lo, hi, _ = c.window(); first = local_midnight(lo, c.tz); last = local_midnight(max(lo, hi - 1), c.tz)
    for day in range(first, last + DAY, DAY): tab.setdefault(day, {"recorded": 0, "parked": 0, "longest": 0})
    rows = [(day, v, v["recorded"] >= FULL_DAY * 1440) for day, v in sorted(tab.items())]
    if mode == "hours":
        need_h = float(c.params.get("hours", 20))
        okd = [(day, v) for day, v, full in rows if v["parked"] * STEP >= need_h * 3600]; unclass = [day for day, v, full in rows if not full and v["parked"] * STEP < need_h * 3600]
        txt = f"{len(okd)} of {len(rows)} calendar days {label} had at least {need_h:g} hours of recorded parked time"
        if okd: txt += ": " + "; ".join(f"{day_label(day, c.tz)} ({v['parked'] * STEP / 3600:.1f} h)" for day, v in okd)
        if unclass: txt += f". {len(unclass)} day{'s' if len(unclass) != 1 else ''} had too little recorded data to reach that, so the true count could be higher"
        return _ans(txt + ".", "temporal_parked_hours", n=len(okd), of=len(rows))
    if mode == "any":
        anyp = [day for day, v, full in rows if v["parked"] > 0]
        return _ans(f"{len(anyp)} of {len(rows)} calendar days {label} had some recorded parked time.", "temporal_parked_days", n=len(anyp), of=len(rows))
    allday = [(day, v) for day, v, full in rows if full and v["parked"] == v["recorded"]]; anypark = [day for day, v, full in rows if v["parked"] > 0]
    unclass = [day for day, v, full in rows if not full]; covmax = max((v["recorded"] for _, v, _ in rows), default=0)
    parts = [f"{len(allday)} of {len(rows)} calendar days {label} were parked for the whole day"]
    if allday: parts[0] += ": " + ", ".join(day_label(day, c.tz) for day, _ in allday)
    parts.append(f"{len(anypark)} had some parked time")
    if unclass: parts.append(f"{len(unclass)} could not be classified because fewer than {int(FULL_DAY * 100)}% of their 1,440 minutes are recorded (the best-covered day has {covmax} recorded minutes)")
    return _ans("; ".join(parts) + f". A day counts as parked all day only when at least {int(FULL_DAY * 100)}% of it is recorded and every recorded sample is parked.", "temporal_parked_days", n=len(allday), of=len(rows), unclassified=len(unclass))


def run_longest_run(c):
    d, label = c.load([])
    if d.n == 0 or d.mode is None: return _nodata(c, label, "drive-state samples")
    pre = [x for x in c.states if x.predicate]
    sm = d.state_mask(pre) if pre else None
    if sm is None: return _ans("Which state should I look for the longest stretch of?", "plan_clarify", ok=False)
    rs = runs(d.ts, sm)
    name = describe_states(pre)
    if not rs: return _ans(f"The car was not recorded as {name} at any time {label}.", "temporal_longest_parked", ok=False)
    best = max(rs, key=lambda r: run_seconds(d.ts, r[0], r[1])); bs = run_seconds(d.ts, *best)
    return _ans(f"The longest continuous stretch {name} {label} lasted {fmt_dur(bs)}, from {stamp(d.ts[best[0]], c.tz)} to {stamp(d.ts[best[1]] + STEP, c.tz)}. Stretches are broken wherever recorded samples are more than {BREAK // 60} min apart, so unrecorded time is never counted.", "temporal_longest_parked", seconds=bs)


# ------------------------------------------------------------------ events, trips, charging sessions (typed fields only)
def _events(c, ty, lo, hi):
    from .temporal_events import fetch_events
    return fetch_events(c.qc, ty, lo, hi)


def run_event_states(c):
    """count / list_events with named states (negation in `not`), or with two or more mutually exclusive states in `groups` (one count per state)."""
    q = c.q
    if q.event is None: return None
    from .temporal_events import modes_at, apply_states
    ty = q.event.type; lo, hi, label = c.window()
    if q.scope.time.type == "all": lo, hi, label = c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history"
    items = _events(c, ty, lo, hi); lab = R.EVENTS[ty][0]
    event_preds = {"hard_braking": "HARSH_BRAKE", "hard_acceleration": "HARSH_ACCEL"}
    for x in q.scope.conditions:
        if x.predicate in event_preds:
            matches = ty == event_preds[x.predicate]
            if (x.op == "not" and matches) or (x.op != "not" and not matches): items = []
    conds = [x for x in q.scope.conditions if x.predicate and x.predicate not in event_preds]
    modes = modes_at(c.qc, [i["ts"] for i in items], lo, hi)
    if len(q.groups) >= 2:
        counts = {}; parts = []
        for g in q.groups:
            n_ = sum(1 for m_ in modes if not np.isnan(m_) and apply_states(m_, [Condition(predicate=g)] + conds)); counts[g] = n_
            parts.append(f"{n_} {lab}{'' if n_ == 1 else 's'} {describe_states([Condition(predicate=g)])}")
        return _ans(f"{label.capitalize() if not label.startswith(('the ', 'over')) else 'Over ' + label.replace('over ', '', 1)}: " + "; ".join(parts) + (f", with the car {describe_states(conds)}" if conds else "") + ".", "temporal_event_groups", counts=counts)
    keep = []; unknown = 0
    for it, m_ in zip(items, modes):
        if not conds: keep.append(it)
        elif np.isnan(m_): unknown += 1
        elif apply_states(m_, conds): keep.append(it)
    n = len(keep); cond_txt = describe_states(conds).replace("not driving", "not moving")
    nt = f" Note: {unknown} event{'s' if unknown != 1 else ''} fell in a minute with no recorded drive state and could not be classified." if unknown else ""
    if len({x.predicate for x in conds if x.predicate in MODE and x.op != "not"}) >= 2: nt += " Note: this vehicle records one drive state at a time, so two of those states cannot hold together."
    where = f" while the car was {cond_txt}" if conds else ""
    if q.op == "list_events":
        e = q.event
        if e.selector == "first": keep = keep[:1]
        elif e.selector == "last": keep = keep[-1:]
        elif e.selector == "nth": keep = keep[e.n - 1:e.n]
        elif e.selector == "page":
            start = (e.n - 1) * 25; keep = list(reversed(keep))[start:start + 25]
        n = len(keep)
        if not keep: return _ans(f"No {lab} was recorded {label}{where}.{nt}", "temporal_events", n=0)
        rows = "; ".join(f"{stamp(i['ts'], c.tz)} {i.get('detail', '')}".strip() for i in keep[:12])
        return _ans(f"{n} {lab}{'s' if n != 1 else ''} {label}{where}: {rows}{' ...' if n > 12 else ''}.{nt}", "temporal_events", n=n)
    return _ans(f"{n} {lab}{'s' if n != 1 else ''} {label}{where}.{nt}", "temporal_events", n=n)


def run_trip_events(c):
    from .temporal_events import trips_between, trip_class
    q = c.q; ty = q.event.type if q.event else "HARSH_BRAKE"
    lo, hi, label = c.window()
    if q.scope.time.type == "all": lo, hi, label = c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history"
    trips = trips_between(c.qc, lo, hi)
    if not trips: return _ans(f"There are no recorded trips {label}.", "temporal_trip_events", ok=False)
    evs = _events(c, ty, lo, hi + 1); lab = R.EVENTS[ty][0]
    for tr in trips: tr["n"] = sum(1 for e in evs if tr["start"] <= e["ts"] <= tr["end"])
    sel = trips; desc = []
    tt = c.params.get("tripthr")
    if tt:
        op, _, v = str(tt).partition(":"); v = float(v)
        cmpf = {"gt": lambda x: x > v, "ge": lambda x: x >= v, "lt": lambda x: x < v, "le": lambda x: x <= v, "eq": lambda x: math.isclose(x, v)}[op]
        sel = [x for x in sel if x["wh_km"] is not None and cmpf(x["wh_km"])]; desc.append(f"trips matching the {op} {v:g} Wh/km threshold")
    klass = q.scope.select.klass if q.scope.select else None
    if klass: sel = [x for x in sel if trip_class(x) == klass]; desc.append(f"{klass} trips")
    es = q.scope.select
    if es and es.rank_by:
        keys = {"wh_per_km": "wh_km", "distance_driven": "km", "recency": "start"}; key = keys.get(es.rank_by)
        if key:
            sel = [x for x in sel if x.get(key) is not None]
            sel = sorted(sel, key=lambda x: x[key], reverse=es.order == "desc")[:es.k or len(sel)]
    scope = " among ".join(reversed(desc)) if desc else "trips"
    mode = c.params.get("mode", "count")
    if mode == "none":
        none = [x for x in sel if x["n"] == 0]
        return _ans(f"{len(none)} of {len(sel)} {scope} {label} had no {lab} at all. ({sum(1 for x in sel if x['n'] > 0)} had at least one.)", "temporal_trip_events", n=len(none), of=len(sel))
    if mode == "some":
        some = [x for x in sel if x["n"] > 0]
        rows = "; ".join(f"{stamp(x['start'], c.tz)} ({x['n']} {lab}{'s' if x['n'] != 1 else ''}, {x['km']:.0f} km)" for x in some[-10:])
        return _ans(f"After excluding trips that never had a {lab}, {len(some)} of {len(sel)} {scope} {label} remain" + (f"; the most recent: {rows}." if some else "."), "temporal_trip_events", n=len(some), of=len(sel))
    total = sum(x["n"] for x in sel)
    return _ans(f"{total} {lab}{'s' if total != 1 else ''} occurred on {len(sel)} {scope} {label} (an event belongs to the trip whose start-to-end interval contains it).", "temporal_trip_events", n=total, trips=len(sel))


def run_trip_extreme(c):
    from .temporal_events import trips_between
    lo, hi, label = c.window()
    if c.q.scope.time.type == "all": lo, hi, label = c.qc.ctx.lo, c.qc.ctx.now + 1, "over the whole recorded history"
    trips = trips_between(c.qc, lo, hi)
    skip = str(c.params.get("skiplatest", "0")) == "1"
    if skip and len(trips) < 2: return _ans("There are fewer than two recorded trips, so nothing remains after excluding the most recent.", "temporal_trip_extreme", ok=False)
    rest = trips[:-1] if skip else trips
    worst = (c.agg or "max") == "max"
    eligible = [x for x in rest if x["wh_km"] is not None]
    if not eligible: return _ans(f"There are no trips with recorded energy consumption {label}.", "temporal_trip_extreme", ok=False)
    best = (max if worst else min)(eligible, key=lambda x: x["wh_km"])
    ex = f"Excluding the most recent trip ({stamp(trips[-1]['start'], c.tz)}), the" if skip else "The"
    return _ans(f"{ex} trip with the {'worst' if worst else 'best'} energy consumption {label} was on {stamp(best['start'], c.tz)}: {best['wh_km']:.0f} Wh/km over {best['km']:.1f} km ({len(rest)} trips considered, each at least 0.2 km).", "temporal_trip_extreme", wh_km=best["wh_km"])


def run_session_attr(c):
    conn = c.qc.conn
    if "charges" not in c.qc.profile.tables: return _ans("No charging sessions are recorded for this vehicle.", "plan_nodata", ok=False)
    vs = c.params.get("vs") == "prev"
    rows = conn.execute("SELECT start, end, kind, kw, kwh, soc0, soc1 FROM charges WHERE end <= ? ORDER BY start DESC LIMIT 2", (c.qc.ctx.now + 1,)).fetchall()
    if not rows or (vs and len(rows) < 2): return _ans("There are not enough recorded charging sessions for that.", "temporal_session", ok=False)
    sess = rows[:2] if vs else rows[:1]; attr = c.params.get("attr", "start"); tz = c.tz
    names = ["the last charge", "the one before"]
    def show_ts(ts): d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); return f"{d.day} {d.strftime('%b')}, {d:%H:%M}"
    vals = []; label = None
    for start, end, kind, kw, kwh, soc0, soc1 in sess:
        if attr == "duration": vals.append(end - start); label = ("duration", "s")
        elif attr == "energy": vals.append(kwh); label = ("energy added to the battery", "kWh")
        elif attr == "power": vals.append(kw); label = ("peak charging power", "kW")
        elif attr == "soc": vals.append(soc1 - soc0); label = ("state-of-charge gain", "%")
        elif attr == "temperature":
            sig = c.signal or "battery_temperature"
            q2 = Query(op="aggregate", target=sig, agg="mean"); q2.scope.time = TimeSpec("range", start=int(start), end=int(end) + 1)
            v, _ = C._value(c.qc, q2)
            if v is None: return _ans(f"There is no recorded {R.label_of(sig)} during that charge.", "temporal_session", ok=False)
            vals.append(v); label = (f"average {R.label_of(sig)}", R.CONCEPTS[sig].unit)
        elif attr == "start": vals.append(start); label = ("start", "ts")
        else: return _ans(f"The charging-session attribute '{attr}' is not supported, so I will not substitute a different field.", "plan_invalid", ok=False)
    what, unit = label
    def sh(v):
        if unit == "s": return C._fmt_dur(v)
        if unit == "ts": return show_ts(v)
        return f"{v:.0f} percentage points" if what == "state-of-charge gain" else f"{v:.1f} {unit}"
    if what == "start":
        def one(sx): st, en, kind_, kw_, kwh_, s0_, s1_ = sx; return f"started on {show_ts(st)} and ended at {show_ts(en)[-5:]} ({C._fmt_dur(en - st)}): {s0_:.0f}% to {s1_:.0f}%, +{kwh_:.1f} kWh at up to {kw_:.0f} kW"
        return _ans("; ".join(f"{n.capitalize()} {one(sx)}" for n, sx in zip(names, sess)) + ".", "last_charge", n=len(vals))
    if len(sess) == 1: return _ans(f"The {what} during {names[0]} ({show_ts(sess[0][0])} to {show_ts(sess[0][1])[-5:]}) was {sh(vals[0])}.", "composed_session", value=vals[0], what=what)
    d_ = vals[0] - vals[1]; diff = C._fmt_dur(abs(d_)) if unit == "s" else (f"{abs(d_):.1f} {unit}" if unit != "%" else f"{abs(d_):.0f} percentage points")
    cmpw = ("longer" if d_ > 0 else "shorter") if unit == "s" else ("higher" if d_ > 0 else "lower")
    return _ans(f"The {what} was {sh(vals[0])} during {names[0]} ({show_ts(sess[0][0])}) and {sh(vals[1])} during {names[1]} ({show_ts(sess[1][0])}): {diff} {cmpw}.", "composed_session", a=vals[0], b=vals[1], what=what)


def run_normality(c):
    lo, hi, label = c.window(); agg = c.agg or "mean"
    cur, _n, _l = _mean_in(c, c.spec, agg)
    if cur is None: return _nodata(c, label, c.sig_label())
    base = []; d0 = local_midnight(lo, c.tz) - DAY
    for _ in range(14):
        v, _ = C._value(c.qc, C._aggregate_query(c.signal, agg, TimeSpec("range", start=d0, end=d0 + DAY), [x for x in c.states]))
        if v is not None: base.append(v)
        d0 -= DAY
    if len(base) < 5: return _ans(f"I can't judge whether that was normal: fewer than 5 of the 14 days before {label} have recorded {c.sig_label()}.", "composed_vs_normal", ok=False)
    mean = sum(base) / len(base); sd = (sum((x - mean) ** 2 for x in base) / (len(base) - 1)) ** 0.5
    n_days = max(1, round((hi - lo) / DAY)); se = sd / (n_days ** 0.5) if agg == "mean" else sd; z = (cur - mean) / se if se > 1e-9 else 0.0
    kind = "unusually high" if z > 1.5 else "unusually low" if z < -1.5 else "within its usual range"
    u = R.CONCEPTS[c.signal].unit; word = {"mean": "average", "max": "highest", "min": "lowest"}.get(agg, agg)
    d = c.params.get("dir"); lead = ""
    if d in ("higher", "lower"): lead = "Yes. " if ((z > 1.5) if d == "higher" else (z < -1.5)) else "No. "
    return _ans(lead + f"The {word} {c.sig_label()} {label} was {cur:.1f} {u}, {kind} for this vehicle: in the {len(base)} days before it the daily {word} was {mean:.1f} {u} (typical day-to-day spread +/-{sd:.1f} {u}).", "composed_vs_normal", value=cur, mean=mean, sd=sd, z=z)


def run_clock_slot(c):
    slot = c.params.get("slot"); agg = c.agg or "mean"
    if slot:
        s, e = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21)}.get(slot, (None, None))
        slots = [(s * 60, e * 60)] if s is not None else [(0, 6 * 60), (21 * 60, 24 * 60)]
        label = f"in the {slot} ({s:02d}:00 to {e:02d}:00)" if s is not None else "at night (21:00 to 06:00)"
    else:
        try:
            h1, m1 = [int(x) for x in str(c.params["from"]).split(":")]; h2, m2 = [int(x) for x in str(c.params["to"]).split(":")]
        except Exception: return _ans("I could not read the clock times of that request.", "plan_clarify", ok=False)
        a, b = h1 * 60 + m1, h2 * 60 + m2
        if b <= a: slots = [(a, 1440), (0, b)]
        else: slots = [(a, b)]
        label = f"between {a // 60 % 24:02d}:{a % 60:02d} and {b // 60 % 24:02d}:{b % 60:02d}"
    days, wlabel = C._days(c.qc, c.spec); parts = []
    for day, spec in days:
        for a, b in slots:
            lo, hi = max(spec.start, day + a * 60), min(spec.end, day + b * 60)
            if hi <= lo: continue
            q = C._aggregate_query(c.signal, agg if agg in ("mean", "max", "min", "sum") else "mean", TimeSpec("range", start=lo, end=hi), [x for x in c.states])
            try: res = c.qc.executor.run(q.normalized())
            except Exception: continue
            if res.status == "ok" and isinstance(res.value, (int, float)): parts.append((float(res.value), int(res.samples or 1)))
    if not parts: return _ans(f"There is no recorded {c.sig_label()} {label} {wlabel}.", "composed_clock", ok=False)
    if agg == "mean": val = sum(v * n for v, n in parts) / sum(n for _, n in parts)
    elif agg == "max": val = max(v for v, _ in parts)
    elif agg == "min": val = min(v for v, _ in parts)
    else: val = sum(v for v, _ in parts)
    word = {"mean": "average", "max": "highest", "min": "lowest", "sum": "total"}[agg if agg in ("mean", "max", "min", "sum") else "mean"]
    return _ans(f"The {word} {c.sig_label()} {label} {wlabel} was {C._fmt(c.signal, val)} (from {sum(n for _, n in parts):,} samples).", "composed_clock", target=c.signal, value=val)


def run_epistemic(c):
    from . import temporal_misc as M
    from .temporal import Data as D
    kind = c.params.get("kind"); q = c.q
    sc = type("S", (), {})()                                                  # a minimal context with the typed fields only (no question text)
    sc.qc, sc.brain, sc.tz, sc.t = c.qc, c.brain, c.tz, ""; sc.signal = c.signal; sc.signals = [c.signal] if c.signal else []; sc.states = []; sc.th = c.th; sc.spec = c.spec; sc.min_dur = None
    sc.window = c.window; sc.load = c.load; sc.sig_label = c.sig_label
    now = c.qc.ctx.now
    if kind == "certainty": return M.run_fault_certainty_plan(sc, q.event.type if q.event else "COOLANT_FLOW_DROP")
    if kind == "proof": return M.run_prove_zero_degradation_plan(sc)
    if kind == "causation": return M.run_causation_plan(sc)
    if kind == "criteria": return M.run_fault_criteria_plan(sc)
    if kind == "regen": return M.run_regen_reconcile_plan(sc)
    if kind == "tail": return M.run_tail_percentile_plan(sc, float((c.params.get("pct") or "99.9").split(",")[0]), c.signal or "battery_temperature")
    if kind == "exact": return M.run_exact_instant_plan(sc, str(c.params.get("at", "")), c.signal)
    if kind == "conflict": return M.run_conflicting_value_plan(sc, c.signal or "coolant_flow", [c.params.get("v1"), c.params.get("v2")])
    return _ans("I cannot tell what kind of check that is.", "plan_clarify", ok=False)


RUNNERS = {"exists": run_exists, "threshold_test": run_threshold_test, "duration": run_duration, "share": run_share, "episodes": run_episodes, "continuous": run_continuous, "days_with": run_days_with,
           "percentile": run_percentile, "percentile_vs": run_percentile_vs, "range": run_range, "window_change": run_window_change, "halves": run_halves, "by_period": run_by_period, "pair_gap": run_pair_gap,
           "parked_days": run_parked_days, "longest_run": run_longest_run, "trip_events": run_trip_events, "trip_extreme": run_trip_extreme, "session_attr": run_session_attr,
           "normality": run_normality, "clock_slot": run_clock_slot, "epistemic": run_epistemic}
THRESHOLD_OPS = ("exists", "threshold_test", "share", "episodes", "continuous", "days_with")


def run_topic(c):
    """op=topic: the model named a whole-question intent from the closed topic set; run its route handler on the canonical request built from the plan's typed arguments."""
    from . import topics, learned_front
    q = c.q; p = c.params; topic = p.get("topic")
    phrase = topics.canonical_request(topic, q.target, q.target2, p.get("v1"), p.get("v2"), p.get("n"))
    if phrase is None: return _ans("I recognised what kind of question this is, but not every detail it needs (a signal, a number or a period), so I will not guess.", "plan_clarify", ok=False)
    br = c.brain                                                                             # a topic request is self-contained: no legacy conversation state may leak into it
    br.qcs.last = None; br.sem_topic = None; br.qcs.choices = []; br.qcs.pending = None; br.qcs.options = []
    try: br.qa.ctx.update(topic=None, diag=None, event_row=None)
    except Exception: pass
    with learned_front.legacy_allowed():
        r = br._ask_routes(phrase, use_slm=False)
    f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
    r = dict(r); r["facts"] = dict(f, topic=topic, canonical_request=phrase, frontend="learned_topic")
    return r


RUNNERS["topic"] = run_topic


def execute(brain, qc, q):
    """-> result dict (answer, brief, intent, facts) or None when this module does not own the operation."""
    fn = RUNNERS.get(q.op)
    if fn is None and q.op in OPS_WITH_EVENT and q.event is not None and (q.groups or any(x.predicate for x in q.scope.conditions)): fn = run_event_states
    if fn is None: return None
    c = PlanCtx(brain, qc, q)
    if q.op in THRESHOLD_OPS and c.th is None: return _ans("I could not tell which threshold to compare against, so I will not guess.", "plan_clarify", ok=False)
    if q.op in ("exists", "threshold_test", "duration", "share", "episodes", "continuous", "days_with", "percentile", "range", "by_period", "clock_slot", "normality", "window_change", "halves") and q.op != "duration":
        miss = c.need_signal()
        if miss: return miss
    return fn(c)
