"""Evidence contracts: what an answer is ALLOWED TO CLAIM, given what was asked and what the stored data can still support.

A valid question can still be answered with more certainty than the data allows. Examples this module catches, each of which the earlier pipeline stated without qualification:
  * 'highest speed last month' is computed from one-minute averages once the 1 Hz window has passed; a brief peak inside a minute can be higher. The claim is narrowed to 'highest one-minute average'.
  * 'show all driver-assist disengagements' returned the newest ten; the answer now says how many exist and how many are shown.
  * events cannot be recorded while the recorder had a gap; the answer states the share of active time that was not recorded.
  * a request for 30 days when only 19 are on record is answered for 19, and says so.
  * an estimate (remaining range) is an estimate; an association is not a cause.
Every answer therefore carries a Contract: evidence class (observed | derived | estimated | associated | unresolved), the resolution of the stored data it used, completeness (complete | partial | truncated), data coverage, and the
limits as sentences. `assess` is deterministic and uses only the query, the executed result and the history span. `apply` narrows the wording; `check` is the verifier's rule that an answer which says 'all' must be complete."""
import re
from dataclasses import dataclass, field

from . import signal_registry as R

DAY = 86400
UNIT_S = {"minute": 60, "hour": 3600, "day": DAY, "week": 7 * DAY, "month": 30 * DAY}
RESOLUTION = {"1 Hz": "1-second samples", "per-minute": "1-minute averages", "daily": "daily values", "trip ledger": "one record per trip", "events": "recorded events", "registry": "catalog", "1 Hz event context": "1-second samples around the event"}
ORDER_SENSITIVE = ("max", "min", "p95", "std", "median", "percent_time")
CLASS_WORD = {"observed": "measured", "derived": "computed from measurements", "estimated": "an estimate", "associated": "an association (not proof of cause)", "unresolved": "not resolved"}


@dataclass
class Contract:
    result_class: str = "observed"
    resolution: str = ""
    completeness: str = "complete"        # complete | partial | truncated
    coverage: float = None                 # share of the requested period that is on record (None = complete / not applicable)
    signal_coverage: dict = field(default_factory=dict)      # signal[:period] -> share of the ACTIVE days it covers (only gaps are listed); None = could not be checked (unknown, never assumed complete)
    signal_days: dict = field(default_factory=dict)          # signal[:period] -> {covered, partial, missing, expected}
    recorded_share: float = None           # share of the ACTIVE time the recorder actually captured (None = unknown)
    shown: int = None
    total: int = None
    limits: list = field(default_factory=list)
    page: dict = field(default_factory=dict)          # event list paging: first, last, size, total
    brief_note: str = ""                   # short parenthetical added to the one-sentence answer when the claim had to be narrowed
    refuse: str = None

    def to_dict(self): return {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}


MIN_DAY_FRACTION = 0.5          # a day counts as covered when the signal has readings for at least half of that day's recorded activity minutes; fewer is "partly", none is "missing"


def _span(sec):
    """A length of time in the unit a person would say it: minutes, hours or days."""
    sec = float(sec)
    if sec < 5400: return f"{max(1, round(sec / 60))} min"
    if sec < 2 * DAY: return f"{sec / 3600:.0f} h" if sec >= 7200 else f"{sec / 3600:.1f} h"
    return f"{sec / DAY:.1f} days" if sec < 10 * DAY else f"{sec / DAY:.0f} days"


def _active_days(conn, tz, lo, hi):
    """-> (activity, known). activity = {local day: active minutes} from the trip ledger AND the charge ledger (a start inside [lo, hi) owns the whole record). known is False when neither ledger can be read: coverage is then UNKNOWN, not complete.
    An empty but readable ledger gives ({}, True): there is no recorded activity in the interval and no day-level claim is made."""
    act = {}; readable = False
    for table in ("trips", "charges"):
        try: rows = conn.execute(f"SELECT start, end FROM {table} WHERE start < ? AND COALESCE(end,start) >= ?", (hi, lo)).fetchall(); readable = True
        except Exception: continue
        for st, en in rows:
            st, en = max(st, lo), min(en or st, hi)
            if en < st: continue
            # Attribute minutes to every local day traversed, rather than
            # assigning a cross-midnight trip entirely to its start day.
            cur = st
            while cur < max(en, st + 1):
                day_end = (((cur + tz) // DAY) + 1) * DAY - tz
                take = min(max(en, st + 1), day_end) - cur
                d = (cur + tz) // DAY; act[d] = act.get(d, 0.0) + max(0.0, take / 60.0)
                cur += max(take, 1)
    return act, readable


def _signal_days(conn, ctx, q, lo, hi, profile=None):
    """{signal: {covered, partial, missing, expected} | None} for each MEASURED signal the answer depends on (derived quantities are followed to the signals they are computed from), over local days with recorded activity.
    None = unknown (no ledger to compare with). A day with a single reading is not a covered day: it needs at least MIN_DAY_FRACTION of the day's activity minutes."""
    out = {}
    names = [x for x in (q.target, q.target2) if x] + [c.signal for sc in (q.scope, q.scope_b) if sc for c in sc.conditions if c.signal]
    measured = []
    for n in dict.fromkeys(names):
        if profile is not None:
            try:
                from .query_planner import measured_needs
                measured += sorted(measured_needs(n, profile))
            except Exception: measured.append(n)
        else: measured.append(n)
    act, known = _active_days(conn, ctx.tz, lo, hi)
    for n in dict.fromkeys(measured):
        store = R.STORE.get(n, {})
        ent = store.get("1m") or store.get("1s")
        if not ent:
            # The signal is legitimate but its retained tier cannot support a
            # day-coverage claim.  Preserve that unknown rather than silently
            # omitting it from the evidence contract.
            out[n] = None
            continue
        if not known: out[n] = None; continue
        if not act: continue
        table, col = ent[0], ent[1]
        try: rows = conn.execute(f"SELECT (ts + ?) / 86400, COUNT(*) FROM {table} WHERE ts >= ? AND ts < ? AND {col} IS NOT NULL GROUP BY 1", (ctx.tz, lo, hi)).fetchall()
        except Exception: out[n] = None; continue
        cnt = dict(rows); cov = part = miss = 0
        for d, minutes in act.items():
            k = cnt.get(d, 0)
            if k >= MIN_DAY_FRACTION * minutes: cov += 1
            elif k > 0: part += 1
            else: miss += 1
        out[n] = {"covered": cov, "partial": part, "missing": miss, "expected": len(act)}
    return out


def _requested_seconds(q):
    t = q.scope.time
    if t.type == "relative" and t.value and t.unit in UNIT_S: return t.value * UNIT_S[t.unit]
    if t.type == "calendar" and t.name in ("today", "this_morning", "last_night", "yesterday", "day_before_yesterday"): return DAY
    return None


def _recorded_share(res):
    for f in res.flags or []:
        if f.get("code") == "DROPOUT":
            m = re.search(r"\((\d+)% of the active time\)", f.get("detail", ""))
            if m: return 1.0 - int(m.group(1)) / 100.0
    return None


def assess(q, res, ctx=None, conn=None, profile=None):
    """-> Contract for an executed result (res.status == 'ok')."""
    c = Contract(); op = q.op; tier = res.tier or ""; c.resolution = RESOLUTION.get(tier, tier)
    names = [x for x in (q.target, q.target2) if x] + [x.signal for sc in (q.scope, q.scope_b) if sc for x in sc.conditions if x.signal]
    kinds = {R.CONCEPTS[n].kind for n in names if n in R.CONCEPTS}
    if res.estimated or q.target == "range_estimate" or getattr(res, "quality_kind", "") == "estimated": c.result_class = "estimated"
    elif op in ("correlation", "explain_event", "explain_change", "event_context"): c.result_class = "associated" if op != "event_context" else "observed"
    elif op in ("trend", "detect_change") or "derived" in kinds or getattr(res, "quality_kind", "") == "derived": c.result_class = "derived"
    else: c.result_class = "observed"
    # ---- resolution limits: statistics that depend on the extremes of a signal are not what one-minute averages can give
    if tier == "per-minute" and op in ("extreme", "aggregate", "percent_time") and (q.agg in ORDER_SENSITIVE or op in ("extreme", "percent_time")):
        what = {"max": "highest", "min": "lowest", "p95": "95th percentile", "std": "variability", "median": "median", "percent_time": "share of time"}.get(q.agg if op != "percent_time" else "percent_time", "extreme")
        c.limits.append(f"Over this period only one-minute averages are stored, so the {what} is taken over averages; a brief peak inside a minute can be higher or lower.")
        c.brief_note = "of one-minute averages" if q.agg in ("max", "min", "p95") else ""
    # ---- completeness of event lists
    if op in ("list_events", "count") and q.event is not None:
        total = int(res.value) if isinstance(res.value, (int, float)) else None; shown = len(res.items or []); c.total = total; c.shown = shown
        pg = (res.extra or {}).get("page")
        if op == "list_events" and q.event.selector in ("all", "page") and pg and pg["total"] > pg["last"] - (pg["first"] - 1 if pg["first"] else 0) or (op == "list_events" and pg and pg["first"] > 1):
            c.completeness = "truncated" if pg["last"] < pg["total"] else c.completeness; c.page = dict(pg)
            c.limits.append(f"Showing events {pg['first']} to {pg['last']} of {pg['total']}, newest first." + (f" Say 'next' to see events {pg['last'] + 1} to {min(pg['last'] + pg['size'], pg['total'])}." if pg["last"] < pg["total"] else " That is the last page."))
        c.recorded_share = _recorded_share(res)
        if c.recorded_share is not None and c.recorded_share < 0.995:
            c.completeness = "partial" if c.completeness == "complete" else c.completeness
            c.limits.append(f"The recorder missed about {100 * (1 - c.recorded_share):.0f}% of the active time in this period, so events during those gaps cannot appear; the true count may be higher.")
    elif res.flags and any(f.get("code") == "DROPOUT" for f in res.flags):
        c.recorded_share = _recorded_share(res)
        if c.recorded_share is not None and c.recorded_share < 0.9: c.limits.append(f"About {100 * (1 - c.recorded_share):.0f}% of the active time in this period was not recorded; statistics describe the recorded part.")
    # ---- coverage: the requested period vs the recorded history, and each requested signal vs the days it actually has readings (a gap anywhere below 99% is stated)
    if ctx is not None and res.period and res.period[1] > res.period[0] and q.scope.time.type in ("relative", "calendar", "date") and ctx.hi > ctx.lo:
        lo, hi = int(res.period[0]), int(min(res.period[1], ctx.now + 1)); asked = max(1, hi - lo); have = max(0, min(hi, ctx.hi + 1) - max(lo, ctx.lo))
        if have < 0.95 * asked:
            c.coverage = round(have / asked, 3); c.completeness = "partial" if c.completeness == "complete" else c.completeness
            c.limits.append(f"Only {_span(have)} of the {_span(asked)} you asked about are on record.")
    # Day-level coverage describes a historical interval.  A current-value
    # query is an instant backed by the canonical live snapshot; comparing its
    # one-second interval with a whole day's activity ledger creates a false
    # "0 of 1 active days" warning even when that snapshot is valid.
    if (conn is not None and ctx is not None and res.period and res.period[1] > res.period[0]
            and op not in ("capability",) and q.scope.time.type in ("relative", "calendar", "date")):
        from .query_planner import resolve_time
        periods = [("", int(res.period[0]), int(min(res.period[1], ctx.now + 1)))]
        if q.scope_b is not None and q.scope_b.time is not None:                                           # a comparison has two periods: both are inspected
            try: lo2, hi2, lab2 = resolve_time(q.scope_b.time, ctx); periods = [("in the first period", *periods[0][1:]), ("in the second period", int(lo2), int(min(hi2, ctx.now + 1)))]
            except Exception: pass
        for where, plo, phi in periods:
            for sig, d in _signal_days(conn, ctx, q, plo, phi, profile).items():
                key = sig + (":" + where if where else ""); lab = R.label_of(sig).capitalize() + (" " + where if where else "")
                if d is None:
                    c.signal_coverage[key] = None; c.completeness = "partial" if c.completeness == "complete" else c.completeness
                    c.limits.append(f"{lab}: day-by-day coverage could not be checked, because no trip or charge ledger is available to compare with."); continue
                c.signal_days[key] = d
                if d["covered"] < d["expected"]:
                    c.signal_coverage[key] = round(d["covered"] / d["expected"], 3); c.completeness = "partial" if c.completeness == "complete" else c.completeness
                    c.limits.append(f"{lab} has readings covering {d['covered']} of {d['expected']} active days" + (f" ({d['partial']} only partly, {d['missing']} not at all)" if d["partial"] or d["missing"] else "") + (f" {where}" if False else "") + ".")
    if tier == "per-minute" and op == "percent_time": c.limits.append("The share of time is counted over one-minute averages, so excursions shorter than a minute are not counted exactly.")
    if c.result_class == "estimated": c.limits.append("This is a model-based estimate, not a measurement.")
    if c.result_class == "associated": c.limits.append("Signals that move together are associated; this alone does not show which one drives the other.")
    return c


def numbers(c):
    """The numbers a contract's sentences state, so that the answer verifier can find every one of them in the evidence."""
    out = {}
    if c.shown is not None: out["contract_shown"] = float(c.shown)
    if c.total is not None: out["contract_total"] = float(c.total)
    if c.page:
        out["contract_first"] = float(c.page["first"]); out["contract_last"] = float(c.page["last"]); out["contract_next_first"] = float(c.page["last"] + 1); out["contract_next_last"] = float(min(c.page["last"] + c.page["size"], c.page["total"]))
    if c.coverage is not None: out["contract_coverage_pct"] = float(round(100 * c.coverage))
    for lim in c.limits:
        for m in re.finditer(r"Only (\d+(?:\.\d+)?) of the (\d+(?:\.\d+)?) days", lim): out["contract_have_days"] = float(m.group(1)); out["contract_asked_days"] = float(m.group(2))
    for i, lim in enumerate(c.limits):                                                                  # every number a limit sentence states (minutes, hours or days) is evidence: the sentence is made from the contract itself
        for j, m in enumerate(re.findall(r"\d+(?:\.\d+)?", lim)): out[f"contract_limit_{i}_{j}"] = float(m)
    for key, d in c.signal_days.items():                                                              # evidence is keyed by signal (and period): one signal's coverage never overwrites another's
        for f in ("covered", "partial", "missing", "expected"): out[f"contract_{f}:{key}"] = float(d[f])
    if c.recorded_share is not None: out["contract_missed_pct"] = float(round(100 * (1 - c.recorded_share)))
    for lim in c.limits:
        for m in re.finditer(r"only (\d+(?:\.\d+)?) days", lim): out["contract_days"] = float(m.group(1))
        for m in re.finditer(r"asked for (\d+) ", lim): out["contract_asked"] = float(m.group(1))
    return out


def apply(answer, c):
    """Narrow the wording of an Answer in place: a parenthetical on the one-sentence claim when needed, the evidence class and the limits under it."""
    if c is None: return answer
    if c.brief_note and c.brief_note not in answer.brief: answer.brief = answer.brief.rstrip(".") + f" ({c.brief_note})."
    line = f"Evidence: {CLASS_WORD[c.result_class]}" + (f", {c.resolution}" if c.resolution else "") + (f", coverage {100 * c.coverage:.0f}%" if c.coverage is not None else "") + "."
    answer.details = list(answer.details) + [line] + c.limits
    return answer


def check(answer_text, q, c):
    """Verifier rule: wording that claims completeness requires a complete contract. Returns a list of issues."""
    issues = []
    if c is None: return issues
    t = answer_text.lower()
    if c.completeness != "complete" and re.search(r"\b(all|every|no other|none other|only these)\b", t) and q.op in ("list_events", "count") and "showing" not in t and "of " not in t:
        issues.append("claims completeness although the evidence is " + c.completeness)
    if c.result_class == "estimated" and not re.search(r"about|estimate|approximately", t + " " + " ".join(c.limits).lower()): issues.append("an estimate is stated as a measurement")
    return issues
