"""Plan -> typed Query, preserving the plan's meaning exactly.

Jobs, none of which reads language:
  * merge(): the explicit state edit. A follow-up / correction / clarification reply states what changes; `drop` removes parts; conditions are added, removed or replaced as the plan SAYS (never "every threshold on that signal is replaced"). Every field keeps its PROVENANCE: current message | previous accepted state | the original message under a clarification.
  * ground(): optional arguments a small model can invent (period, condition, unit) must cite words, and the quote is checked against the source it claims (the current message, or the original message a clarification reply answers; an inherited field is trusted from the state that accepted it). A quote that does not exist - or, with a `support` check, words that do not support the chosen value - makes the interpretation UNRESOLVED: nothing is dropped, nothing is defaulted in its place, no broader query is executed; the proposed fields stay in the diagnostic record and the caller repairs the plan or asks a specific question.
  * validate(): STRUCTURE only - well-formed fields, registered identifiers, operation/type/unit compatibility, required arguments present (derived from toolspec's own table), a state reference that exists, a supported calculation. It never judges whether the plan matches the question. Any exception is a failed validation (fail closed).
  * compile(): builds the Query. Nothing the plan states is changed: the statistic is the stated statistic, a signal condition stays a signal condition, a stated period is never replaced, no quantity is converted into another. Only OMITTED fields get a documented default, and every default is disclosed in `notes`. fidelity() then re-reads the finished Query and checks that every explicit plan field is carried by it.
Outcomes (kind): query | tool | ambiguous (one specific question) | unresolved (no registered concept) | ungrounded (a proposed argument could not be grounded) | refuse | chat | invalid (not well formed / cannot be executed / a required argument is missing) | unsupported (a calculation or a period the engine does not support)."""
import copy
import datetime as dt
from dataclasses import dataclass, field

from . import signal_registry as R
from . import toolspec as T
from .query_ir import Condition, EntitySelect, EventRef, Query, Scope, TimeSpec
from .toolspec import FN_OP, GROUNDED, NO_TURN

# the plan fields each function consumes (the rest of a carried-over plan is not meaningful for it)
FN_FIELDS = {"current_value": ("tgt", "unit"), "statistic": ("tgt", "stat", "per", "cond", "ent", "evt", "unit"), "extreme": ("tgt", "stat", "per", "cond", "ent", "evt", "unit"), "count_events": ("evt", "per"), "count_entities": ("ent", "per"),
             "list_events": ("evt", "ev", "per", "cond"), "did_event_happen": ("evt", "per"), "did_condition_hold": ("tgt", "cond", "per"), "time_share": ("cond", "tgt", "per"), "compare": ("tgt", "stat", "per", "per2", "grp"),
             "when_changed": ("tgt", "per", "cond"), "event_context": ("evt", "tgt", "ev", "bef", "aft", "per"), "explain_change": ("tgt", "per", "cond"), "explain_event": ("evt", "ev", "per"), "trend": ("tgt", "per", "cond"),
             "rank_trips": ("ent", "per"), "correlate": ("tgt", "tgt2", "per"), "is_recorded": ("tgt",), "use_tool": ("tool", "focus", "per")}
FN_IR = {"current_value": "current_value", "statistic": "aggregate", "extreme": "extreme", "count_events": "count", "count_entities": "count", "list_events": "list_events", "did_event_happen": "count", "did_condition_hold": "count", "time_share": "percent_time",
         "compare": "compare", "when_changed": "detect_change", "event_context": "event_context", "explain_change": "explain_change", "explain_event": "explain_event", "trend": "trend", "rank_trips": "rank", "correlate": "correlation", "is_recorded": "capability"}
DROP_FIELDS = {"period": ("per", "per2"), "conditions": ("cond",), "statistic": ("stat",), "signal": ("tgt", "tgt2"), "event": ("evt", "ev", "bef", "aft"), "groups": ("grp",), "selection": ("ent",), "unit": ("unit",)}
KEEP = ("tgt", "tgt2", "stat", "per", "per2", "cond", "evt", "ev", "bef", "aft", "grp", "ent", "unit", "tool", "focus")
TOOL_WINDOWS = {"all": "a", "cal:today": "t", "cal:yesterday": "y", "cal:this_week": "tw", "cal:last_weekend": "we", "last:7:day": "7", "last:1:week": "7", "last:14:day": "14", "last:2:week": "14", "last:28:day": "28", "last:4:week": "28"}
TOOL_DEFAULT_WINDOW = {"service": "a", "tyres": "14", "weather": "7", "driving": "7", "health": "14", "places": "y"}
TOOL_SUPPORTED = "7, 14 or 28 days, today, yesterday, this week, last weekend, a date, or all recorded time"
GROUND_LABEL = {"per": "period", "per2": "second period", "cond": "restriction", "unit": "unit"}


@dataclass
class Outcome:
    kind: str
    query: Query = None
    tool: str = ""
    focus: str = ""
    window: str = ""
    say: str = ""
    what: str = ""
    notes: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)          # the resolved plan (explicit fields only): what the conversation state stores
    pending_plan: dict = None                         # the partial plan to carry when a question is asked about it
    issues: list = field(default_factory=list)        # [(code, message)]
    reason: str = ""
    existence: bool = False
    tool_period: str = ""
    failures: list = field(default_factory=list)      # grounding failures: [{"key", "label", "value", "words", "why"}]
    proposed: dict = None                             # the model's own plan, kept for the diagnostic record when grounding failed
    provenance: dict = field(default_factory=dict)    # plan key -> current | state | original ; "default:<key>" -> documented default


# ------------------------------------------------------------------------------------------------ state edit
def _cond_key(c): return ("n", c) if isinstance(c, str) else ("t", c.get("signal"), c.get("op"))


def _norm_cond(c):
    if isinstance(c, dict): return ("t", c.get("signal"), c.get("op"), float(c.get("value")), c.get("unit") or None)
    return ("n", c)


def edit_conditions(old, new, mode):
    old = list(old or []); new = list(new or [])
    if mode == "replace": return new
    if mode == "remove": return [c for c in old if not any(_cond_key(c) == _cond_key(n) for n in new)]
    out = list(old)
    for c in new:
        if all(_norm_cond(c) != _norm_cond(x) for x in out): out.append(c)
    return out


def merge(base, plan):
    """The resolved plan = the base (previous resolved plan, or the partial plan of a pending clarification) with the plan's explicit edits applied. A `new` turn, or a turn without any base, stands alone. Only fields the plan's function consumes survive. The quoted words that ground each field (`src`) and its provenance (`prov`) travel with it."""
    fn = plan.get("fn"); fields = FN_FIELDS.get(fn, ())
    own = {k: copy.deepcopy(v) for k, v in plan.items() if k in KEEP}; osrc = dict(plan.get("src") or {}); oprov = dict(plan.get("prov") or {})
    if plan.get("turn", "new") == "new" or not base:
        out = own; src = osrc; prov = {k: oprov.get(k, "current") for k in out}
    else:
        out = {k: copy.deepcopy(v) for k, v in base.items() if k in KEEP}; src = dict(base.get("src") or {}); prov = {k: "state" for k in out}
        for d in plan.get("drop") or []:
            for k in DROP_FIELDS.get(d, ()): out.pop(k, None); src.pop(k, None); prov.pop(k, None)
        for k, v in own.items():
            if k == "cond":
                out["cond"] = edit_conditions(out.get("cond"), v, plan.get("cedit") or "add")
                if osrc.get("cond"): src["cond"] = (src["cond"] + " / " + osrc["cond"]) if (src.get("cond") and (plan.get("cedit") or "add") == "add") else osrc["cond"]
            elif k == "ent": out["ent"] = {**(out.get("ent") or {}), **v}
            else:
                out[k] = v
                if k in osrc: src[k] = osrc[k]
                else: src.pop(k, None)
            prov[k] = oprov.get(k, "current")
        if "cond" in out and not out["cond"]: out.pop("cond")
    out = {k: v for k, v in out.items() if k in fields}
    src = {k: v for k, v in src.items() if k in out}; prov = {k: v for k, v in prov.items() if k in out}
    out["fn"] = fn; out["op"] = FN_OP.get(fn)
    if src: out["src"] = src
    if prov: out["prov"] = prov
    if plan.get("exists_condition"): out["exists_condition"] = True
    return out


# ------------------------------------------------------------------------------------------------ grounding of optional arguments
def _norm_words(t):
    t = str(t or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(t.split()).strip(" .,?!;:\"'")


def _shown(v):
    return ", ".join(c if isinstance(c, str) else f"{c.get('signal')} {c.get('op')} {c.get('value')}" for c in v) if isinstance(v, list) else str(v)


def _without_redundant_native_unit(plan, message=None, original=None):
    """Return a copy with an *unmentioned* native output unit removed.

    A current-value plan for speed means the same thing with or without the
    registry's native ``km/h`` output unit.  Small models sometimes emit that
    harmless schema default even when the user did not say it.  Treating the
    default as a claimed user restriction caused a needless clarification.
    Non-native conversions and explicitly mentioned units remain grounded and
    checked in the normal way.
    """
    unit = plan.get("unit"); con = R.CONCEPTS.get(plan.get("tgt"))
    if not unit or con is None or unit != con.unit: return plan
    words = _norm_words((plan.get("src") or {}).get("unit"))
    sources = (_norm_words(message), _norm_words(original))
    if words and any(words in source for source in sources if source): return plan
    out = copy.deepcopy(plan); out.pop("unit", None)
    src = dict(out.get("src") or {}); src.pop("unit", None)
    if src: out["src"] = src
    else: out.pop("src", None)
    return out


def ground(plan, message, original=None, support=None):
    """Check the provenance of the arguments THIS call proposes (period, restriction, unit). -> (prov, failures). Sources: the current message; the original message of a clarification (`original`). `support(key, words, value)` -> True / False / None (None = cannot judge) is an optional second check that the quoted words support the chosen value. message=None: no check."""
    prov = {}; fails = []
    if message is None: return prov, fails
    msg = _norm_words(message); orig = _norm_words(original) if original else ""
    for pn, key in GROUNDED.items():
        if plan.get(key) in (None, "", [], {}): continue
        words = _norm_words((plan.get("src") or {}).get(key)); rec = {"key": key, "label": GROUND_LABEL[key], "value": _shown(plan[key]), "words": words}
        if len(words) < 2: fails.append(dict(rec, why="no words quoted")); continue
        if words in msg: where = "current"
        elif orig and words in orig: where = "original"
        else: fails.append(dict(rec, why="the quoted words do not occur in the message")); continue
        if support is not None and support(key, words, plan[key]) is False: fails.append(dict(rec, why="the quoted words do not support that value")); continue
        prov[key] = where
    return prov, fails


# ------------------------------------------------------------------------------------------------ required arguments (derived from toolspec) and validation (structure only)
def _present(p, key):
    if key.startswith("ent."): return (p.get("ent") or {}).get(key[4:]) not in (None, "")
    if key == "stat": return p.get("stat") in ("min", "max")
    return p.get(key) not in (None, "", [], {})


def _need(p):
    """Required arguments of the resolved plan: [(field, question to ask)] - from the same table that defines the schema and the grammar."""
    fn = p.get("fn"); miss = [(k, q) for k, q in T.required_keys(fn) if not _present(p, k)]
    for group in T.ONE_OF.get(fn, ()):
        keys = [T.PARAM_KEY[g] for g in group]
        if not any(_present(p, k) and (k != "grp" or len(p.get("grp") or []) >= 2) for k in keys): miss.append((keys[0], T.ASK[group[0]]))
    return miss


def validate(p, registry, now=None, tz=0):
    """-> list of (code, message). Structure and registered identifiers only."""
    out = []; by = registry.by_id
    def sig(i, what):
        if i not in by or by[i].kind != "signal": out.append(("unregistered", f"{what} {i!r} is not a registered signal"))
    for k in ("tgt", "tgt2"):
        if p.get(k) is not None: sig(p[k], "target")
    if p.get("evt") is not None and (p["evt"] not in by or by[p["evt"]].kind != "event"): out.append(("unregistered", f"event {p['evt']!r} is not registered"))
    if p.get("tool") is not None and "tool:" + str(p["tool"]) not in by: out.append(("unregistered", f"tool {p['tool']!r} is not registered"))
    modes = []
    for c in (p.get("cond") or []) + [{"n": g} for g in (p.get("grp") or [])]:
        if isinstance(c, dict) and "n" in c: c = c["n"]
        if isinstance(c, str):
            if c not in by or by[c].kind != "condition": out.append(("unregistered", f"condition {c!r} is not registered"))
            elif c in R.MODE_PREDICATES and c not in modes: modes.append(c)
        elif isinstance(c, dict):
            s = c.get("signal"); sig(s, "condition signal")
            if c.get("op") not in T.CMPOPS: out.append(("malformed", f"bad comparison {c.get('op')!r}"))
            try: float(c.get("value"))
            except (TypeError, ValueError): out.append(("malformed", "threshold value is not a number"))
            con = R.CONCEPTS.get(s)
            if con is not None:
                if con.dtype == "categorical": out.append(("type", f"{s} is categorical: a numeric threshold does not apply"))
                u = c.get("unit")
                if u and con.unit and R.unit_family(u) != R.unit_family(con.unit): out.append(("unit", f"unit {u} does not match {s} ({con.unit})"))
        else: out.append(("malformed", "a condition must be a name or a threshold"))
    if len([m for m in modes]) > 1 and not p.get("grp"): out.append(("conflict", "conditions " + " and ".join(modes) + " cannot hold together (a vehicle is in one driving mode at a time)"))
    if p.get("unit"):
        con = R.CONCEPTS.get(p.get("tgt")); u = p["unit"]
        if con is not None and con.unit and R.unit_family(u) != R.unit_family(con.unit): out.append(("unit", f"unit {u} does not apply to {p['tgt']} ({con.unit})"))
    for k in ("per", "per2"):
        v = p.get(k)
        if v is None: continue
        try: t = TimeSpec.from_text(v)
        except Exception: out.append(("malformed", f"period {v!r} is not a valid period")); continue
        if t.type == "date":
            try: d = dt.date.fromisoformat(t.date)
            except ValueError: out.append(("malformed", f"{t.date} is not a date")); continue
            if now is not None and d > dt.datetime.fromtimestamp(now + (tz or 0), dt.timezone.utc).date(): out.append(("malformed", f"{t.date} is in the future"))
        if t.type == "relative" and (t.unit not in ("minute", "hour", "day", "week", "month") or not t.value or t.value < 1): out.append(("malformed", f"bad relative period {v!r}"))
    ent = p.get("ent")
    if ent:
        e = ent.get("entity"); m = ent.get("rank_by")
        if e and e not in R.ENTITY_METRICS: out.append(("malformed", f"unknown entity {e!r}"))
        elif e and m and m not in R.ENTITY_METRICS[e]: out.append(("type", f"{m} is not a ranking metric of a {e}"))
        if ent.get("klass") and ent["klass"] not in R.TRIP_CLASSES: out.append(("malformed", f"unknown trip class {ent['klass']!r}"))
        if ent.get("order") and ent["order"] not in T.ORDERS: out.append(("malformed", "order must be asc or desc"))
        if ent.get("k") is not None and not (1 <= int(ent["k"]) <= 50): out.append(("malformed", "count out of range"))
    if p.get("fn") == "extreme" and p.get("stat") not in (None, "min", "max"): out.append(("malformed", "extreme needs min or max"))
    if p.get("fn") == "statistic" and p.get("stat") is not None and p["stat"] not in T.STATS: out.append(("unsupported", f"statistic {p['stat']!r} is not a summary statistic (use the highest / lowest operation)"))
    if p.get("fn") == "compare" and p.get("stat") is not None and p["stat"] not in T.CSTATS: out.append(("unsupported", f"statistic {p['stat']!r} is not supported"))
    if p.get("fn") == "statistic" and p.get("stat") and p.get("tgt") in R.CONCEPTS and R.CONCEPTS[p["tgt"]].dtype == "categorical" and p["stat"] not in R.AGG_CATEGORICAL:
        out.append(("unsupported", f"{R.CONCEPTS[p['tgt']].label} is a category: a {p['stat']} is not defined for it"))
    return out


# ------------------------------------------------------------------------------------------------ time
def to_timespec(per):
    return None if per is None else TimeSpec.from_text(per)


def tool_window(per):
    """Domain tools accept a fixed set of windows; a stated period outside it is reported, never replaced."""
    if per is None: return None
    if per in TOOL_WINDOWS: return TOOL_WINDOWS[per]
    if per.startswith("date:"): return "d:" + per[5:]
    return ""


# ------------------------------------------------------------------------------------------------ compile
def to_condition(c):
    if isinstance(c, str): return Condition(predicate=c)
    return Condition(signal=c["signal"], op=c["op"], value=float(c["value"]), unit=c.get("unit") or None)


def from_condition(c):
    if c.predicate: return c.predicate
    d = {"signal": c.signal, "op": c.op, "value": float(c.value)}
    if c.unit: d["unit"] = c.unit
    return d


def compile_plan(plan, state_plan, pending, registry, now=None, tz=0, message=None, original=None, support=None):
    """plan: canonical plan of the model's call; state_plan: the previous resolved plan; pending: {"plan": partial plan or None} of an open clarification; message / original: the current message and, for a clarification reply, the original question (the sources a quotation may come from). -> Outcome. Never raises: an exception is an invalid plan."""
    try: return _compile(plan, state_plan, pending, registry, now, tz, message, original, support)
    except Exception as e: return Outcome("invalid", reason=f"{type(e).__name__}: {e}", issues=[("exception", f"{type(e).__name__}")], plan=state_plan or {})


def _compile(plan, state_plan, pending, registry, now, tz, message=None, original=None, support=None):
    plan = _without_redundant_native_unit(plan, message, original)
    fn = plan.get("fn")
    if fn == "ask_user": return Outcome("ambiguous", say=plan.get("ask") or "", plan=state_plan or {}, reason="model asked")
    if fn == "unresolved": return Outcome("unresolved", what=plan.get("what") or "", plan=state_plan or {})
    if fn == "decline": return Outcome("refuse", say=plan.get("say") or "offtopic", plan=state_plan or {})
    if fn == "say": return Outcome("chat", say=plan.get("say") or "help", plan=state_plan or {})
    if fn not in FN_FIELDS: return Outcome("invalid", reason=f"unknown function {fn!r}", plan=state_plan or {})
    prov, fails = ground(plan, message, original, support); turn = plan.get("turn", "new")
    base = (pending or {}).get("plan") if turn == "clarification_reply" else state_plan
    if fails:                                                                                           # unresolved: keep what the model proposed, resolve the rest, and let the caller repair or ask - never execute a broader query
        failed = {f["key"] for f in fails}; rest = {k: v for k, v in plan.items() if k not in failed}; rest["src"] = {k: v for k, v in (plan.get("src") or {}).items() if k not in failed}
        partial = merge(base if turn != "new" else None, dict(rest, prov=prov))
        return Outcome("ungrounded", failures=fails, proposed=plan, pending_plan=partial, plan=state_plan or {}, reason="; ".join(f"{f['label']} {f['value']}: {f['why']}" for f in fails),
                       say="I could not find where you asked for " + " and for ".join(f"the {f['label']} “{f['value']}”" for f in fails) + ". Which did you mean - or is there none?")
    p = merge(base if turn != "new" else None, dict(plan, prov=prov))
    issues = validate(p, registry, now, tz)
    if issues: return Outcome("unsupported" if all(i[0] == "unsupported" for i in issues) else "invalid", issues=issues, reason="; ".join(m for _, m in issues), plan=state_plan or {})
    miss = _need(p)
    if miss: return Outcome("invalid", say=miss[0][1], issues=[("missing", f"required argument missing: {miss[0][0]}")], reason=f"missing {miss[0][0]}", plan=state_plan or {}, pending_plan=p)
    pv = dict(p.get("prov") or {})
    if fn == "use_tool":
        o = _tool(p, state_plan); o.provenance = pv; return o
    notes = []; q = _build(p, notes, registry)
    bad = fidelity(p, q)
    if bad: return Outcome("invalid", issues=[("fidelity", f"the executable query does not carry: {', '.join(bad)}")], reason="query does not preserve the plan", plan=state_plan or {})
    q.notes = notes
    return Outcome("query", query=q, notes=notes, plan=p, existence=fn in ("did_event_happen", "did_condition_hold"), provenance=pv)


def _tool(p, state_plan):
    tool = p["tool"]; per = p.get("per"); focus = p.get("focus") or ""
    if per == "now":                                                                                   # "now" is not a window: the tool reports the present state (the weather tool has an explicit live mode)
        per = None; focus = focus or ("now" if tool == "weather" else "")
    w = tool_window(per)
    if w == "": return Outcome("unsupported", reason=f"The {tool} check works over {TOOL_SUPPORTED}; it cannot take the period you gave, and I have not used another.", plan=state_plan or {}, tool=tool, tool_period=per)
    notes = []
    if w is None: w = TOOL_DEFAULT_WINDOW.get(tool, "14"); notes.append(f"No period given, so the {tool} check used its standard window.")
    return Outcome("tool", tool=tool, focus=focus, window=w, notes=notes, plan=p, tool_period=per or "")


def _build(p, notes, registry):
    fn = p["fn"]; ir = FN_IR[fn]; q = Query(op=ir, source="slm", target=p.get("tgt"), target2=p.get("tgt2")); conds = [to_condition(c) for c in p.get("cond") or []]
    t = to_timespec(p.get("per")); sel = None
    if p.get("ent"):
        e = p["ent"]; k = e.get("k"); order = e.get("order")
        if fn == "rank_trips":                                                                         # a ranking has documented defaults; a selection used inside another operation states exactly what it selects
            if k is None: k = 5; notes.append("No count was given, so I ranked the top 5.")
            if order is None: order = "desc"; notes.append("No order was given, so I took the highest first.")
        sel = EntitySelect(e["entity"], e.get("klass"), e.get("rank_by"), order or "desc", None if k is None else int(k))
    con = R.CONCEPTS.get(q.target)
    for c in conds:
        if c.signal and c.unit is None and R.CONCEPTS.get(c.signal) is not None and R.CONCEPTS[c.signal].unit not in ("none", "category", ""): notes.append(f"No unit was given for the {c.signal} threshold, so I used {R.CONCEPTS[c.signal].unit}.")
    stat = p.get("stat")
    if fn == "statistic":
        if stat is None:
            if q.target in R.EXTENSIVE: stat = "sum"; notes.append("No statistic was given and this quantity accumulates, so I used the total.")
            elif con is not None and con.dtype == "categorical": stat = "percent_time"
            else: stat = "mean"; notes.append("No statistic was given, so I used the average.")
        q.agg = stat
    elif fn == "extreme": q.agg = stat
    elif fn == "compare":
        q.agg = stat or "mean"
        if not stat: notes.append("No statistic was given, so I used the average.")
        if p.get("grp"): q.groups = list(p["grp"])
    elif fn in ("count_events", "did_event_happen"): q.event = EventRef(type=p["evt"], selector="all")
    elif fn == "list_events":
        sel_name = p.get("ev")
        if sel_name is None: sel_name = "all"; notes.append("No selection was given, so I listed every occurrence.")
        q.event = EventRef(type=p["evt"], selector=sel_name)
    elif fn in ("event_context", "explain_event"):
        sel_name = p.get("ev")
        if sel_name is None: sel_name = "last"; notes.append("No occurrence was named, so I used the most recent one.")
        q.event = EventRef(type=p["evt"], selector=sel_name, before=int(p.get("bef") or 0), after=int(p.get("aft") or 0))
        if fn == "event_context" and not q.event.before and not q.event.after: q.event.before = 600; notes.append("No window was given, so I looked 600 s before the event.")
    if fn in ("statistic", "extreme") and p.get("evt"):                                                 # an event-anchored statistic: computed inside the window of that event
        q.event = EventRef(type=p["evt"], selector="last"); notes.append("No occurrence was named, so I used the most recent one.")
    # ---- scope and time defaults (documented; disclosed)
    q.scope = Scope(t or TimeSpec("relative", 7, "day"), sel, conds)
    if t is None:
        if fn == "current_value": q.scope.time = TimeSpec("current")
        elif sel is not None and fn != "count_entities": q.scope.time = TimeSpec("all"); notes.append("No period given, so I looked at the whole recorded history.")
        elif fn in ("event_context", "explain_event", "is_recorded", "when_changed") or (fn == "list_events" and (p.get("ev") or "all") in ("last", "first")) or (fn in ("statistic", "extreme") and p.get("evt")): q.scope.time = TimeSpec("all"); notes.append("No period given, so I looked at the whole recorded history.")
        elif fn == "explain_change": q.scope.time = TimeSpec("relative", 14, "day"); notes.append("No period given, so I compared the last 14 days with the 14 days before.")
        elif fn == "trend": q.scope.time = TimeSpec("relative", 14, "day"); notes.append("No period given, so I used the last 14 days.")
        else: notes.append("No period given, so I used the last 7 days.")
    if fn in ("statistic", "extreme", "trend") and q.scope.time.type == "current": q.scope.time = TimeSpec("relative", 24, "hour"); notes.append("For a statistic, 'now' means the last 24 hours.")
    if p.get("per2"): q.scope_b = Scope(to_timespec(p["per2"]), sel, list(conds))
    if p.get("unit"): q.unit = p["unit"]
    return q


# ------------------------------------------------------------------------------------------------ fidelity: the finished Query carries every explicit plan field
def query_fields(q):
    f = {"tgt": q.target, "tgt2": q.target2}
    if q.agg: f["stat"] = q.agg
    if q.scope.time is not None: f["per"] = q.scope.time.to_text()
    if q.scope_b is not None: f["per2"] = q.scope_b.time.to_text()
    f["cond"] = [from_condition(c) for c in q.scope.conditions]
    if q.groups: f["grp"] = list(q.groups)
    if q.event is not None:
        f["evt"] = q.event.type; f["ev"] = q.event.selector
        if q.event.before: f["bef"] = q.event.before
        if q.event.after: f["aft"] = q.event.after
    if q.scope.select is not None: s = q.scope.select; f["ent"] = {"entity": s.entity, "rank_by": s.rank_by, "order": s.order, "k": s.k, "klass": s.klass}
    if q.unit: f["unit"] = q.unit
    return f


def fidelity(p, q):
    """Names of explicit plan fields the Query does not carry (empty = faithful)."""
    qf = query_fields(q); bad = []
    for k in FN_FIELDS[p["fn"]]:
        v = p.get(k)
        if v in (None, "", [], {}): continue
        if p["fn"] in ("count_events", "did_event_happen") and k == "ev": continue
        if k == "cond":
            if sorted(map(_norm_cond, v)) != sorted(map(_norm_cond, qf.get("cond", []))): bad.append(k)
        elif k == "ent":
            if any(qf.get("ent", {}).get(a) != b for a, b in v.items()): bad.append(k)
        elif k == "grp":
            if list(v) != qf.get("grp", []): bad.append(k)
        elif k == "stat":
            if p["fn"] in ("statistic", "extreme", "compare") and qf.get("stat") != v: bad.append(k)
        elif qf.get(k) != v: bad.append(k)
    return bad


def query_to_plan(q, fn=None):
    """The resolved plan equivalent of a typed Query (so a confirmed or otherwise-resolved query can be continued)."""
    if q is None: return None
    rev = {"current_value": "current_value", "aggregate": "statistic", "extreme": "extreme", "list_events": "list_events", "percent_time": "time_share", "compare": "compare", "detect_change": "when_changed", "event_context": "event_context",
           "explain_change": "explain_change", "explain_event": "explain_event", "trend": "trend", "rank": "rank_trips", "correlation": "correlate", "capability": "is_recorded"}
    if fn is None and q.op == "count": fn = "did_event_happen" if q.event is not None else ("count_entities" if q.scope.select is not None else "did_condition_hold")
    fn = fn or rev.get(q.op)
    if fn is None: return None
    qf = query_fields(q); p = {"fn": fn, "op": FN_OP[fn]}
    for k in FN_FIELDS[fn]:
        v = qf.get(k)
        if v in (None, "", [], {}): continue
        p[k] = v
    return p
