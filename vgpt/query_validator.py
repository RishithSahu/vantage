"""Query validator: rejects structurally invalid plans BEFORE anything is executed. It checks the shape and the vocabulary of a Query against the registry; whether this vehicle HAS the data is a
separate, softer question answered by capability.py (a valid query about a missing signal is not an error - it gets an explanation).

Everything a Query can contain is either a number, a name from an enumerated registry list, or an ISO date; there is no free-text field that is ever executed, so the plan is read-only by construction.
The string scan below is belt and braces: if a model (or a log line the model was shown) smuggled SQL, shell, path or control characters into any field, the plan is refused here."""
import math
import re
from dataclasses import dataclass, field

from . import signal_registry as R
from .capability import CapabilityReport, assess, OK
from .query_ir import is_calendar_name, OPS, AGGS, PLAN_PARAMS, TIME_TYPES, TIME_UNITS, CALENDAR, COND_OPS, SELECTORS, ENTITIES, ORDERS, REFS, MAX_K, Query

DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NAME_RX = re.compile(r"^[A-Za-z0-9_]{1,48}$")
OPS_NEEDING_TARGET = {"current_value", "aggregate", "acceleration_band", "extreme", "trend", "compare", "correlation", "explain_change", "detect_change",
                      "exists", "threshold_test", "share", "episodes", "continuous", "days_with", "percentile", "percentile_vs", "range", "window_change",
                      "halves", "by_period", "pair_gap", "normality", "clock_slot"}

# The executor's actual typed signatures.  Keeping this table at the validation
# boundary prevents a globally legal token from reaching an operation that
# cannot implement it.
OP_AGGS = {
    "threshold_test": {"mean", "min", "max", "median"},
    "window_change": {"mean", "min", "max", "median", "sum"},
    "by_period": {"mean", "min", "max", "sum"},
    "normality": {"mean", "min", "max"},
    "clock_slot": {"mean", "min", "max", "sum"},
    "halves": {"mean", "min", "max", "median"},
    "trip_extreme": {"min", "max"},
}
OP_PARAMS = {
    "exists": {"thr"}, "threshold_test": {"thr"}, "duration": {"thr"},
    "share": {"thr", "den"}, "episodes": {"thr"}, "continuous": {"thr", "mind"},
    "days_with": {"thr", "mind", "list"}, "percentile": {"pct", "iqr"},
    "percentile_vs": {"pct", "dir"}, "window_change": {"dir"},
    "halves": {"part", "dir"}, "by_period": {"by", "ord", "k"},
    "pair_gap": {"by", "ord"}, "parked_days": {"mode", "hours"},
    "trip_events": {"tripthr", "mode"}, "trip_extreme": {"skiplatest"},
    "session_attr": {"attr", "vs"}, "normality": {"dir"},
    "clock_slot": {"slot", "from", "to"}, "epistemic": {"kind", "pct", "at", "v1", "v2"},
    "topic": {"topic", "v1", "v2", "n", "span"},
}


def _number(value, where, issues, *, lo=None, hi=None, integer=False):
    try:
        n = float(value)
    except (TypeError, ValueError):
        issues.append(Issue("BAD_PARAM", where, "must be a number")); return None
    if not math.isfinite(n) or (integer and not n.is_integer()) or (lo is not None and n < lo) or (hi is not None and n > hi):
        bounds = f" in {lo}..{hi}" if lo is not None and hi is not None else ""
        issues.append(Issue("BAD_PARAM", where, f"must be a finite {'integer' if integer else 'number'}{bounds}")); return None
    return int(n) if integer else n


def _check_clock(value, where, issues, seconds=False):
    pat = r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$" if seconds else r"^(?:[01]\d|2[0-3]):[0-5]\d$"
    if not isinstance(value, str) or not re.match(pat, value): issues.append(Issue("BAD_PARAM", where, "must be a valid clock time"))


def _check_params(q, issues):
    allowed = OP_PARAMS.get(q.op, set())
    for k in q.params:
        if k not in PLAN_PARAMS or k not in allowed:
            issues.append(Issue("BAD_PARAM", f"params.{k}", f"'{k}' is not valid for '{q.op}'"))
    p = q.params
    for k, v in p.items():
        w = f"params.{k}"
        if k in ("thr", "tripthr"):
            op, sep, raw = str(v).partition(":")
            if not sep or op not in COND_OPS: issues.append(Issue("BAD_PARAM", w, "must be comparator:number"))
            else: _number(raw, w, issues)
        elif k == "pct":
            vals = str(v).split(",")
            if not vals or any(x == "" for x in vals): issues.append(Issue("BAD_PARAM", w, "percentile list is empty or malformed"))
            else:
                for x in vals: _number(x, w, issues, lo=0, hi=100)
        elif k in ("mind", "hours", "v1", "v2"): _number(v, w, issues, lo=0)
        elif k in ("k", "n", "span"): _number(v, w, issues, lo=1, hi=MAX_K if k == "k" else None, integer=True)
        elif k in ("list", "iqr", "skiplatest", "gap") and str(v) not in ("0", "1"): issues.append(Issue("BAD_PARAM", w, "must be 0 or 1"))
        elif k == "den" and v not in set(R.PREDICATES) | {"all", "driving", "moving"}: issues.append(Issue("BAD_PARAM", w, "unsupported denominator"))
        elif k == "by" and v not in ("day", "weekday"): issues.append(Issue("BAD_PARAM", w, "must be day or weekday"))
        elif k == "ord" and v not in ORDERS: issues.append(Issue("BAD_PARAM", w, "must be asc or desc"))
        elif k == "part" and v not in ("first", "second"): issues.append(Issue("BAD_PARAM", w, "must be first or second"))
        elif k == "mode" and v not in ("count", "none", "some", "all_day", "hours", "any"): issues.append(Issue("BAD_PARAM", w, "unsupported mode"))
        elif k == "attr" and v not in ("start", "duration", "energy", "power", "soc", "temperature"): issues.append(Issue("BAD_PARAM", w, "unsupported charging-session attribute"))
        elif k == "vs" and v != "prev": issues.append(Issue("BAD_PARAM", w, "must be prev"))
        elif k == "dir" and v not in ("higher", "lower"): issues.append(Issue("BAD_PARAM", w, "must be higher or lower"))
        elif k == "slot" and v not in ("morning", "afternoon", "evening", "night"): issues.append(Issue("BAD_PARAM", w, "unsupported clock slot"))
        elif k in ("from", "to"): _check_clock(v, w, issues)
        elif k == "at": _check_clock(v, w, issues, seconds=True)


@dataclass
class Issue:
    code: str
    field: str
    message: str

    def __str__(self): return f"{self.code} ({self.field}): {self.message}"


@dataclass
class ValidationResult:
    issues: list = field(default_factory=list)
    capability: CapabilityReport = None

    @property
    def valid(self): return not self.issues                                    # structurally valid
    @property
    def answerable(self): return self.valid and self.capability is not None and self.capability.ok

    def codes(self): return [i.code for i in self.issues]


def _walk_strings(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from _walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(x, str): yield path, x


def _check_time(t, issues, where):
    if t.type not in TIME_TYPES: issues.append(Issue("BAD_TIME", where, f"unknown time type '{t.type}'")); return
    if t.start is not None or t.end is not None: issues.append(Issue("BAD_TIME", where, "absolute timestamps cannot be supplied in a query"))
    if t.type in ("relative", "previous"):
        if t.unit not in TIME_UNITS: issues.append(Issue("BAD_TIME", where, f"unknown unit '{t.unit}'"))
        if not isinstance(t.value, int) or isinstance(t.value, bool) or not (1 <= t.value <= 3650): issues.append(Issue("BAD_TIME", where, "the number of units must be 1..3650"))
    elif t.type == "between":
        if t.unit not in ("minute", "hour", "day"): issues.append(Issue("BAD_TIME", where, f"unknown unit '{t.unit}'"))
        ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (t.value, t.value2))
        if not ok or not (0 <= t.value2 < t.value <= 3650): issues.append(Issue("BAD_TIME", where, "an interval is given as how many units ago it begins and ends (the far end larger than the near end, at most 3650)"))
    elif t.type == "calendar" and not is_calendar_name(t.name): issues.append(Issue("BAD_TIME", where, f"unknown calendar name '{t.name}'"))
    elif t.type == "date":
        if not t.date or not DATE_RX.match(t.date): issues.append(Issue("BAD_TIME", where, "dates must be YYYY-MM-DD"))
        else:
            import datetime
            try: datetime.date.fromisoformat(t.date)
            except ValueError: issues.append(Issue("BAD_TIME", where, f"'{t.date}' is not a real calendar date"))


def _check_condition(c, issues, where):
    if (c.predicate is None) == (c.signal is None): issues.append(Issue("BAD_CONDITION", where, "a condition is either a named predicate or a signal threshold")); return
    if c.predicate is not None:
        if c.predicate not in R.PREDICATES: issues.append(Issue("BAD_CONDITION", where, f"unknown predicate '{c.predicate}'"))
        if c.op not in (None, "not"): issues.append(Issue("BAD_CONDITION", where, "a named predicate only supports explicit negation"))
        return
    cc = R.CONCEPTS.get(c.signal)
    if cc is None: issues.append(Issue("UNKNOWN_SIGNAL", where, f"unknown signal '{c.signal}'")); return
    if cc.dtype == "categorical": issues.append(Issue("BAD_CONDITION", where, f"'{c.signal}' is categorical; use a named predicate"))
    if c.op not in COND_OPS: issues.append(Issue("BAD_CONDITION", where, f"unknown comparison '{c.op}'"))
    if not isinstance(c.value, (int, float)) or isinstance(c.value, bool) or not math.isfinite(c.value): issues.append(Issue("BAD_CONDITION", where, "the threshold must be a finite number"))
    if c.unit and R.unit_family(c.unit) != R.unit_family(cc.unit): issues.append(Issue("BAD_UNIT", where, f"{c.unit} does not measure {cc.label}"))


def _check_scope(s, issues, where):
    _check_time(s.time, issues, where + ".time")
    for i, c in enumerate(s.conditions): _check_condition(c, issues, f"{where}.conditions[{i}]")
    if s.select:
        e = s.select
        if e.entity not in ENTITIES: issues.append(Issue("BAD_SELECTION", where, f"unknown entity '{e.entity}'")); return
        if e.klass is not None and (e.entity != "trip" or e.klass not in R.TRIP_CLASSES): issues.append(Issue("BAD_SELECTION", where, f"unknown {e.entity} class '{e.klass}'"))
        if e.rank_by is not None and e.rank_by not in R.ENTITY_METRICS[e.entity]: issues.append(Issue("BAD_RANKING", where, f"cannot rank {e.entity}s by '{e.rank_by}'"))
        if e.order not in ORDERS: issues.append(Issue("BAD_RANKING", where, f"order must be asc or desc"))
        if e.k is not None and (not isinstance(e.k, int) or not 1 <= e.k <= MAX_K): issues.append(Issue("BAD_RANKING", where, f"k must be 1..{MAX_K}"))
        if e.rank_by is None and e.k is not None: issues.append(Issue("BAD_RANKING", where, "k needs a ranking metric"))


def validate(q, profile=None):
    """-> ValidationResult. With a profile, also runs the capability analysis (valid query + missing data = an explained answer, not an error)."""
    issues = []
    if not isinstance(q, Query): return ValidationResult([Issue("NOT_A_QUERY", "", "expected a Query")])
    if q.op not in OPS: issues.append(Issue("UNKNOWN_OP", "op", f"unknown operation '{q.op}'")); return ValidationResult(issues)
    d = q.to_dict()
    for path, s in _walk_strings(d):                                            # read-only / injection guard: names and dates only, no punctuation that could mean anything to SQL, a shell or a path
        if path.startswith("params."): continue                                # typed below; payload punctuation is part of its grammar
        leaf = path.split(".")[-1].split("[")[0]
        if leaf == "date":
            if not DATE_RX.match(s): issues.append(Issue("BAD_TEXT", path, "not a date"))
        elif leaf == "unit":
            if not re.match(r"^[A-Za-z0-9/%]{1,12}$", s): issues.append(Issue("BAD_TEXT", path, "not a unit"))
        elif not NAME_RX.match(s): issues.append(Issue("BAD_TEXT", path, "only plain names are allowed here"))
    if q.agg is not None and q.agg not in AGGS: issues.append(Issue("BAD_AGG", "agg", f"unknown aggregation '{q.agg}'"))
    if q.agg is not None and q.op in OP_AGGS and q.agg not in OP_AGGS[q.op]: issues.append(Issue("BAD_AGG", "agg", f"'{q.op}' does not support '{q.agg}'"))
    _check_params(q, issues)
    for r in q.refs:
        if r not in REFS: issues.append(Issue("BAD_REF", "refs", f"unknown reference '{r}'"))
    # target(s)
    cc = None
    if q.op in OPS_NEEDING_TARGET and not q.target and "target" not in q.refs: issues.append(Issue("MISSING_TARGET", "target", f"'{q.op}' needs a target signal"))
    for fld in ("target", "target2"):
        v = getattr(q, fld)
        if v is None: continue
        c = R.CONCEPTS.get(v)
        if c is None: issues.append(Issue("UNKNOWN_SIGNAL", fld, f"'{v}' is not a known vehicle signal or metric"))
        elif fld == "target": cc = c
    if q.op == "correlation" and (not q.target2 or q.target == q.target2): issues.append(Issue("MISSING_TARGET", "target2", "a correlation needs two different signals"))
    if q.op == "pair_gap" and q.target and q.target2:
        a, b = R.CONCEPTS.get(q.target), R.CONCEPTS.get(q.target2)
        if a and b and R.unit_family(a.unit) != R.unit_family(b.unit): issues.append(Issue("BAD_UNIT", "target2", "pair-gap signals must have compatible dimensions"))
    # aggregation vs type
    if cc is not None and q.agg is not None and q.agg in AGGS:
        allowed = R.AGG_BINARY if cc.dtype == "binary" else R.AGG_CATEGORICAL if cc.dtype == "categorical" else R.AGG_NUMERIC
        if q.agg not in allowed: issues.append(Issue("BAD_AGG", "agg", f"cannot take the {q.agg} of {cc.label} ({cc.dtype})"))
        if q.agg == "sum" and cc.name not in R.EXTENSIVE: issues.append(Issue("BAD_AGG", "agg", f"summing {cc.label} is meaningless - ask for the energy or the average"))
        if cc.name in R.EXTENSIVE and q.agg in ("mean", "median", "std", "p95") and q.op == "aggregate": issues.append(Issue("BAD_AGG", "agg", f"{cc.label} accumulates over a period; ask for the total (sum)"))
    if q.op == "aggregate" and q.agg is None: issues.append(Issue("MISSING_AGG", "agg", "an aggregate needs an aggregation"))
    if q.op == "aggregate" and cc is not None and cc.dtype == "categorical": issues.append(Issue("BAD_AGG", "agg", "categorical summaries need a dedicated count or time-share operation"))
    if q.op == "current_value" and q.scope.time.type != "current": issues.append(Issue("BAD_TIME", "scope.time", "a current-value query must use the current time scope"))
    if q.op == "current_value" and (q.scope.conditions or q.scope.select is not None): issues.append(Issue("BAD_SCOPE", "scope", "a current-value query cannot silently ignore historical filters or entity selections"))
    if q.op == "acceleration_band":
        bounds = [c for c in q.scope.conditions if c.signal == "speed" and c.op in ("ge", "gt", "le", "lt")]
        lower = [c for c in bounds if c.op in ("ge", "gt")]
        upper = [c for c in bounds if c.op in ("le", "lt")]
        if q.target != "acceleration" or len(lower) != 1 or len(upper) != 1 or float(lower[0].value) >= float(upper[0].value): issues.append(Issue("BAD_ACCELERATION_BAND", "scope.conditions", "a speed-band acceleration needs one lower and one higher speed bound"))
    # unit
    if q.unit:
        fam = R.unit_family(q.unit)
        if fam is None: issues.append(Issue("BAD_UNIT", "unit", f"unknown unit '{q.unit}'"))
        elif q.op == "percent_time" and q.unit != "%": issues.append(Issue("BAD_UNIT", "unit", "a time share is reported as a percentage"))
        elif cc is not None and fam != R.unit_family(cc.unit) and not (cc.unit == "none" or q.op == "count"): issues.append(Issue("BAD_UNIT", "unit", f"{q.unit} does not measure {cc.label} ({cc.unit})"))
    # scopes
    _check_scope(q.scope, issues, "scope")
    if q.scope_b: _check_scope(q.scope_b, issues, "scope_b")
    # compare
    if q.op == "compare":
        if not q.scope_b and len(q.groups) < 2: issues.append(Issue("BAD_COMPARE", "compare", "a comparison needs two periods or at least two groups"))
        if q.groups and len(q.groups) != 2: issues.append(Issue("BAD_COMPARE", "groups", "a comparison is pairwise and needs exactly two groups"))
        if q.scope_b and q.groups: issues.append(Issue("BAD_COMPARE", "compare", "compare periods or groups, not both"))
    for g in q.groups:
        if (g[4:] if g.startswith("not_") else g) not in R.PREDICATES: issues.append(Issue("BAD_CONDITION", "groups", f"unknown group '{g}'"))
    if q.op == "percentile_vs":
        supported = {"city", "highway", "charging", "parked", "driving"}
        if len(q.groups) != 2: issues.append(Issue("BAD_COMPARE", "groups", "a percentile comparison needs exactly two supported groups"))
        for g in q.groups:
            if g not in supported: issues.append(Issue("BAD_COMPARE", "groups", f"percentile comparison does not support '{g}'"))
    if q.groups and q.op != "compare": issues.append(Issue("BAD_COMPARE", "groups", "groups only apply to a comparison"))
    # events
    if q.op in ("event_context", "list_events", "explain_event") and q.event is None: issues.append(Issue("MISSING_EVENT", "event", f"'{q.op}' needs an event"))
    if q.event:
        e = q.event
        if e.type not in R.EVENTS: issues.append(Issue("UNKNOWN_EVENT", "event", f"unknown event '{e.type}'"))
        if e.selector not in SELECTORS: issues.append(Issue("BAD_EVENT", "event", f"unknown selector '{e.selector}'"))
        if e.selector in ("nth", "page") and (not e.n or e.n < 1): issues.append(Issue("BAD_EVENT", "event", f"{e.selector} needs a positive number"))
        for nm, v in (("before", e.before), ("after", e.after)):
            if not isinstance(v, int) or not 0 <= v <= 6 * 3600: issues.append(Issue("BAD_EVENT", "event", f"{nm} must be 0..21600 seconds"))
        if q.op not in ("event_context", "list_events", "explain_event", "count", "trip_events", "epistemic"):
            issues.append(Issue("BAD_EVENT", "event", f"'{q.op}' does not implement event anchoring"))
    if q.op == "count" and q.event is None and q.scope.select is None and not q.target: issues.append(Issue("MISSING_TARGET", "count", "count what? name an event, a selection or a signal condition"))
    if q.op == "count" and q.target and not q.scope.conditions: issues.append(Issue("MISSING_CONDITION", "scope.conditions", "counting a signal requires the condition whose episodes should be counted"))
    if q.op == "rank" and (q.scope.select is None or q.scope.select.rank_by is None): issues.append(Issue("BAD_RANKING", "scope.select", "a ranking needs an entity and a metric"))
    if q.op == "rank" and q.scope.conditions: issues.append(Issue("BAD_SCOPE", "scope.conditions", "ranking cannot silently ignore sample-level conditions"))
    if q.op == "percent_time" and not q.scope.conditions and (cc is None or cc.dtype == "numeric"): issues.append(Issue("MISSING_CONDITION", "scope.conditions", "the share of time needs a condition"))
    res = ValidationResult(issues)
    if profile is not None and not issues:
        try: res.capability = assess(q, profile)
        except Exception as e:                                                  # never let the analysis of a plan crash the assistant
            res.issues.append(Issue("INTERNAL", "capability", f"{type(e).__name__}"))
    return res
