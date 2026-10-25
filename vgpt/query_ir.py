"""Vehicle Query IR: the typed, compositional description of WHAT is being asked - independent of how the question was worded and of how it will be computed.

    natural language  --(parser: rules first, small language model as fallback)-->  Query  --(validator)-->  --(planner)-->  --(executor)-->  evidence  --(composer / verifier)-->  answer

A Query is data, never code: it names registry concepts (signals, derived metrics, events, named predicates) and carries numbers. It cannot express SQL, file paths, vehicle commands or free text,
which is the security boundary between the language model and the vehicle data. Two representations, one meaning:
  * dict / JSON  (to_dict / from_dict)  - for tests, logs, the UI and the planner
  * compact text (to_text / from_text)  - what the small language model writes (a handful of tokens, enumerated vocabulary, generated into a GBNF grammar by query_grammar.py)

Compositionality: a scope = time window  x  optional entity selection (e.g. "the three least efficient highway trips")  x  sample-level conditions (e.g. "while charging").
A query asks for something (target, aggregation) over a scope, or compares two scopes / groups. "Average battery temperature during the three least efficient highway trips last month" is therefore
ONE query: aggregate(battery_temperature, mean) over Scope(time=last_month, select=trip[highway] ranked by wh_per_km desc, k=3) - no intent had to be invented for it."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import re

OPS = ("current_value", "aggregate", "acceleration_band", "extreme", "trend", "compare", "count", "list_events", "event_context", "rank", "correlation", "percent_time", "explain_event", "capability", "explain_change", "detect_change")
# Temporal / threshold / distribution / epistemic operations of the query plan.  Their arguments are the ordinary fields (target, agg, scope.time, scope.conditions, groups, event, selection)
# plus the typed parameters in Query.params; they are executed by vgpt/plan_exec.py from those fields alone - never from the question text.
PLAN_OPS = ("exists", "threshold_test", "duration", "share", "episodes", "continuous", "days_with", "percentile", "percentile_vs", "window_change", "halves", "by_period", "pair_gap", "range", "parked_days",
            "longest_run", "trip_events", "trip_extreme", "session_attr", "normality", "clock_slot", "epistemic", "topic")
PLAN_PARAMS = {"thr": "gt|lt|ge|le:number", "mind": "seconds", "pct": "numbers", "den": "state", "by": "day|weekday", "ord": "asc|desc", "k": "int", "part": "first|second", "list": "0|1", "mode": "word", "hours": "number",
               "iqr": "0|1", "tripthr": "gt|lt:number", "skiplatest": "0|1", "attr": "word", "vs": "prev", "dir": "higher|lower", "slot": "word", "from": "HH:MM", "to": "HH:MM", "kind": "word", "why": "word", "gap": "0|1", "v1": "number", "v2": "number", "at": "HH:MM:SS", "n": "int", "span": "int", "topic": "id"}
OPS = OPS + PLAN_OPS
AGGS = ("mean", "min", "max", "median", "std", "first", "last", "delta", "count", "p95", "sum", "percent_time", "mode")
TIME_TYPES = ("current", "relative", "calendar", "date", "all", "previous", "between")      # previous: the window of the same length that ends where the "last N" window begins
TIME_UNITS = ("minute", "hour", "day", "week", "month")
import re as _re
CAL_DYNAMIC = _re.compile(r"^(?:y\d{4}|m\d{4}_(?:0[1-9]|1[0-2])|q\d{4}_[1-4]|h\d{4}_[12]|s\d{4}_(?:0[1-9]|1[0-2])_\d{4}_(?:0[1-9]|1[0-2])|f\d{4}_(?:0[1-9]|1[0-2])|last_year|this_year)$")      # a named calendar year (y2025), a named month (m2025-03), last / this calendar year: the person typed the value, the planner turns it into seconds


def is_calendar_name(n): return bool(n) and (n in CALENDAR or bool(CAL_DYNAMIC.match(n)))


CALENDAR = ("today", "yesterday", "this_week", "last_week", "this_month", "last_month", "last_weekend", "this_morning", "last_night", "day_before_yesterday")
COND_OPS = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "=="}
SELECTORS = ("last", "first", "all", "nth", "most_severe", "page")          # page: evs=pageK = events 25(K-1)+1 .. 25K, newest first (how a long list is retrieved in full)
EVENT_PAGE = 25
ENTITIES = ("trip", "charge_session")
ORDERS = ("asc", "desc")
REFS = ("target", "time", "conditions", "selection", "event")
MAX_K = 50


class IRError(ValueError):
    pass


@dataclass
class TimeSpec:
    type: str = "relative"
    value: Optional[int] = None            # relative: how many units
    unit: Optional[str] = None             # relative: minute | hour | day | week | month
    name: Optional[str] = None             # calendar name
    date: Optional[str] = None             # YYYY-MM-DD
    start: Optional[int] = None            # epoch seconds - only set by the executor / tests, never accepted from text
    end: Optional[int] = None
    value2: Optional[int] = None           # between: `value` and `value2` are how many units ago the interval begins and ends (value2 = 0: up to now); whole days when unit is day

    def to_text(self):
        if self.type == "current": return "now"
        if self.type == "relative": return f"last:{self.value}:{self.unit}"
        if self.type == "previous": return f"prev:{self.value}:{self.unit}"
        if self.type == "calendar": return f"cal:{self.name}"
        if self.type == "date": return f"date:{self.date}"
        if self.type == "range": return f"range:{self.start}:{self.end}"
        if self.type == "between": return f"between:{self.value}:{self.value2}:{self.unit}"
        return self.type

    @staticmethod
    def from_text(s):
        if s in ("now", "current"): return TimeSpec("current")
        if s == "all": return TimeSpec("all")
        if s == "since_event": return TimeSpec("since_event")
        p = s.split(":")
        if p[0] == "last" and len(p) == 3:
            try: return TimeSpec("relative", int(p[1]), p[2])
            except ValueError: raise IRError(f"bad number in time '{s}'")
        if p[0] == "prev" and len(p) == 3:
            try: return TimeSpec("previous", int(p[1]), p[2])
            except ValueError: raise IRError(f"bad number in time '{s}'")
        if p[0] == "cal" and len(p) >= 2: return TimeSpec("calendar", name=":".join(p[1:]))
        if p[0] == "between" and len(p) == 4:
            try: return TimeSpec("between", int(p[1]), p[3], value2=int(p[2]))
            except ValueError: raise IRError(f"bad number in time '{s}'")
        if p[0] == "date" and len(p) == 2: return TimeSpec("date", date=p[1])
        raise IRError(f"bad time '{s}'")


@dataclass
class Condition:
    """Either a named predicate ("highway", "charging", "raining", "hard_braking" ...) or a threshold on a signal (signal + op + value)."""
    predicate: Optional[str] = None
    signal: Optional[str] = None
    op: Optional[str] = None               # gt | lt | ge | le | eq
    value: Optional[float] = None
    unit: Optional[str] = None

    def to_text(self):
        if self.predicate: return ("not:" if self.op == "not" else "") + self.predicate
        value = format(float(self.value), ".15g")
        return f"sig:{self.signal}:{self.op}:{value}" + (f":{self.unit.replace('/', '_per_')}" if self.unit else "")     # canonical, round-trip-safe numeric text; conversion is the engine's job, never the model's

    @staticmethod
    def from_text(s):
        if s.startswith("sig:"):
            p = s.split(":")
            if len(p) not in (4, 5): raise IRError(f"bad condition '{s}'")
            try: return Condition(signal=p[1], op=p[2], value=float(p[3]), unit=p[4].replace("_per_", "/") if len(p) == 5 else None)
            except ValueError: raise IRError(f"bad number in condition '{s}'")
        if s.startswith("not:"): return Condition(predicate=s[4:], op="not")                 # the complement of a named state ("excluding charging")
        return Condition(predicate=s)


@dataclass
class EntitySelect:
    """'the three least efficient highway trips': entity=trip, klass=highway, rank_by=wh_per_km, order=desc (least efficient = highest consumption), k=3."""
    entity: str = "trip"
    klass: Optional[str] = None            # trip class from the registry (highway | city | rainy | long | short)
    rank_by: Optional[str] = None          # an entity metric from the registry
    order: str = "desc"
    k: Optional[int] = None

    def to_text(self): return f"{self.entity}:{self.rank_by or '-'}:{self.order}:{self.k or '-'}:{self.klass or '-'}"

    @staticmethod
    def from_text(s):
        p = s.split(":")
        if len(p) != 5: raise IRError(f"bad selection '{s}'")
        try: k = None if p[3] == "-" else int(p[3])
        except ValueError: raise IRError(f"bad number in selection '{s}'")
        return EntitySelect(p[0], None if p[4] == "-" else p[4], None if p[1] == "-" else p[1], p[2], k)


@dataclass
class Scope:
    time: TimeSpec = field(default_factory=lambda: TimeSpec("relative", 7, "day"))
    select: Optional[EntitySelect] = None
    conditions: list = field(default_factory=list)


@dataclass
class EventRef:
    type: str = "ANY_WARNING"
    selector: str = "last"
    n: Optional[int] = None
    before: int = 0                        # seconds of context before the event
    after: int = 0


@dataclass
class Query:
    op: str = "aggregate"
    target: Optional[str] = None
    target2: Optional[str] = None          # correlation partner
    agg: Optional[str] = None
    scope: Scope = field(default_factory=Scope)
    scope_b: Optional[Scope] = None        # compare: the second period / selection
    groups: list = field(default_factory=list)       # compare: named predicates (e.g. ["city", "highway"])
    event: Optional[EventRef] = None
    unit: Optional[str] = None             # unit the person asked for
    params: dict = field(default_factory=dict)       # typed parameters of the plan operations (PLAN_PARAMS): threshold, minimum duration, percentiles, ...
    refs: list = field(default_factory=list)         # parts inherited from the previous query (follow-ups)
    text: str = ""                         # the original question (for logs / the verifier only - never executed)
    confidence: float = 1.0
    source: str = "rules"                  # rules | slm | followup | tests
    notes: list = field(default_factory=list)       # e.g. "no period given, used the last 7 days" - shown by the composer, never executed

    # ---- dict form
    def to_dict(self):
        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items() if v not in (None, [], {}, "")}
            if isinstance(x, list): return [clean(v) for v in x]
            return x
        d = clean(asdict(self)); d.pop("text", None); d.pop("confidence", None); d.pop("source", None); d.pop("notes", None); return d

    @staticmethod
    def from_dict(d):
        if not isinstance(d, dict): raise IRError("a query must be an object")
        allowed = {"op", "target", "target2", "agg", "scope", "scope_b", "groups", "event", "unit", "refs", "text", "confidence", "source", "params"}
        extra = set(d) - allowed
        if extra: raise IRError(f"unknown field(s): {sorted(extra)}")
        q = Query(op=d.get("op", ""), target=d.get("target"), target2=d.get("target2"), agg=d.get("agg"), groups=list(d.get("groups") or []), unit=d.get("unit"), refs=list(d.get("refs") or []), params=dict(d.get("params") or {}),
                  text=str(d.get("text", ""))[:300], confidence=float(d.get("confidence", 1.0)), source=str(d.get("source", "rules")))
        q.scope = _scope(d.get("scope")) if d.get("scope") else Scope()
        q.scope_b = _scope(d["scope_b"]) if d.get("scope_b") else None
        if d.get("event"):
            e = d["event"]; extra = set(e) - {"type", "selector", "n", "before", "after"}
            if extra: raise IRError(f"unknown event field(s): {sorted(extra)}")
            q.event = EventRef(e.get("type", "ANY_WARNING"), e.get("selector", "last"), e.get("n"), int(e.get("before", 0)), int(e.get("after", 0)))
        return q

    # ---- compact text form (what the model writes)
    def to_text(self):
        parts = [f"op={self.op}"]
        if self.target: parts.append(f"tgt={self.target}")
        if self.target2: parts.append(f"tgt2={self.target2}")
        if self.agg: parts.append(f"agg={self.agg}")
        parts.append(f"t={self.scope.time.to_text()}")
        if self.scope_b: parts.append(f"t2={self.scope_b.time.to_text()}")
        if self.scope.conditions: parts.append("c=" + ",".join(c.to_text() for c in self.scope.conditions))
        if self.scope_b and self.scope_b.conditions != self.scope.conditions:
            parts.append("c2=" + (",".join(c.to_text() for c in self.scope_b.conditions) or "-"))
        if self.scope.select: parts.append("sel=" + self.scope.select.to_text())
        if self.scope_b and self.scope_b.select != self.scope.select: parts.append("sel2=" + (self.scope_b.select.to_text() if self.scope_b.select else "-"))
        if self.groups: parts.append("grp=" + ",".join(self.groups))
        if self.event:
            parts.append(f"ev={self.event.type}")
            if self.event.selector != "last" or self.event.n: parts.append(f"evs={self.event.selector}{self.event.n or ''}")        # defaults (last, 0 s) are not written: fewer tokens for the model
            if self.event.before: parts.append(f"bef={self.event.before}")
            if self.event.after: parts.append(f"aft={self.event.after}") if self.event.after else []
        for k in sorted(self.params):
            if self.params[k] not in (None, ""): parts.append(f"p.{k}={self.params[k]}")
        if self.unit: parts.append(f"unit={self.unit.replace('/', '_per_')}")
        if self.refs: parts.append("ref=" + ",".join(self.refs))
        return " ".join(parts)

    @staticmethod
    def from_text(s):
        kv = {}
        for tok in s.strip().split():
            if "=" not in tok: raise IRError(f"bad token '{tok}'")
            k, v = tok.split("=", 1)
            if k in kv: raise IRError(f"duplicate key '{k}'")
            kv[k] = v
        if "op" not in kv: raise IRError("missing op")
        known = {"op", "tgt", "tgt2", "agg", "t", "t2", "c", "c2", "sel", "sel2", "grp", "ev", "evs", "bef", "aft", "unit", "ref"}
        extra_keys = {k for k in kv if k.startswith("p.")}
        bad = [k for k in extra_keys if k[2:] not in PLAN_PARAMS]
        if (set(kv) - known - extra_keys) or bad: raise IRError(f"unknown key(s): {sorted((set(kv) - known - extra_keys) | set(bad))}")
        q = Query(op=kv["op"], target=kv.get("tgt"), target2=kv.get("tgt2"), agg=kv.get("agg"), source="slm")
        q.scope = Scope(TimeSpec.from_text(kv["t"]) if "t" in kv else TimeSpec("relative", 7, "day"), EntitySelect.from_text(kv["sel"]) if "sel" in kv else None, [Condition.from_text(x) for x in kv["c"].split(",")] if kv.get("c") else [])
        if "t2" in kv:
            bconds = list(q.scope.conditions) if "c2" not in kv else ([] if kv["c2"] == "-" else [Condition.from_text(x) for x in kv["c2"].split(",")])
            sel2 = q.scope.select if "sel2" not in kv else (None if kv["sel2"] == "-" else EntitySelect.from_text(kv["sel2"]))
            q.scope_b = Scope(TimeSpec.from_text(kv["t2"]), sel2, bconds)
        if kv.get("grp"): q.groups = kv["grp"].split(",")
        if "ev" in kv:
            sel = kv.get("evs", "last"); n = None
            match = re.fullmatch(r"(all|most_severe|first|last|nth|page)(\d*)", sel)
            if not match: raise IRError("bad event selector")
            sel, suffix = match.groups()
            if suffix:
                if sel not in ("nth", "page"): raise IRError("bad event selector")
                n = int(suffix)
                if n < 1: raise IRError("bad event selector")
            try: q.event = EventRef(kv["ev"], sel, n, int(kv.get("bef", 0)), int(kv.get("aft", 0)))
            except ValueError: raise IRError("bad event window")
        if kv.get("unit"): q.unit = kv["unit"].replace("_per_", "/")
        if kv.get("ref"): q.refs = kv["ref"].split(",")
        for k in extra_keys: q.params[k[2:]] = kv[k]
        return q

    def normalized(self):
        """A copy with every threshold expressed in the registry unit of its signal (60 mph -> 96.56 km/h) and the unit field cleared: what the planner and executor work with.
        Deterministic arithmetic (signal_registry.convert); an unknown or mismatching unit is left for the validator to reject."""
        from . import signal_registry as R
        q = self.clone()
        for sc in (q.scope, q.scope_b):
            if sc is None: continue
            for c in sc.conditions:
                if c.signal and c.unit and c.signal in R.CONCEPTS:
                    cu = R.CONCEPTS[c.signal].unit
                    try: c.value = round(R.convert(float(c.value), c.unit, cu), 4); c.unit = None
                    except ValueError: pass
        return q

    # ---- composition helpers used by the follow-up resolver
    def clone(self):
        import copy
        return copy.deepcopy(self)


def _scope(d):
    if not isinstance(d, dict): raise IRError("scope must be an object")
    extra = set(d) - {"time", "select", "conditions"}
    if extra: raise IRError(f"unknown scope field(s): {sorted(extra)}")
    t = d.get("time") or {"type": "relative", "value": 7, "unit": "day"}
    extra = set(t) - {"type", "value", "unit", "name", "date", "start", "end"}
    if extra: raise IRError(f"unknown time field(s): {sorted(extra)}")
    sel = None
    if d.get("select"):
        s = d["select"]; extra = set(s) - {"entity", "klass", "rank_by", "order", "k"}
        if extra: raise IRError(f"unknown selection field(s): {sorted(extra)}")
        sel = EntitySelect(s.get("entity", "trip"), s.get("klass"), s.get("rank_by"), s.get("order", "desc"), s.get("k"))
    conds = []
    for c in d.get("conditions") or []:
        extra = set(c) - {"predicate", "signal", "op", "value", "unit"}
        if extra: raise IRError(f"unknown condition field(s): {sorted(extra)}")
        conds.append(Condition(c.get("predicate"), c.get("signal"), c.get("op"), c.get("value"), c.get("unit")))
    return Scope(TimeSpec(**{k: t.get(k) for k in ("type", "value", "unit", "name", "date", "start", "end")}), sel, conds)
