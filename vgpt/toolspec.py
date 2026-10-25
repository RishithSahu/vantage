"""The plan language: ONE authoritative definition from which the tool schemas shown to the model (vgpt/schema.py), the generation grammar (plangrammar.py) and the compiler's required-argument checks (plancompile.py) are all derived, so the three cannot disagree.

A plan is ONE call: the function name is the operation, the arguments are its signals, events, period, restrictions, selection and the explicit edit of the conversation state.
Representation is structural, not lexical. A period is a typed string (relative amount + unit, a calendar name, a date); an event window is an amount + unit; a threshold is an object {signal, op, value, unit}; an entity selection is {entity, metric, order, k, class}; a state edit says whether conditions are added, removed or replaced and which fields to drop. Deterministic code resolves dates and converts units.
This module converts a call to / from the canonical plan dict that plancompile.py turns into the typed Query. The call TEXT (Pythonic, JSON, frame) is the business of dialects.py."""
import ast
import json
import re

CALL_OPEN, CALL_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"

# ---- closed vocabularies of the language (registry-dependent ones - signals, events, conditions, tools, classes, metrics, units - come from vgpt/schema.enums)
STATS = ("mean", "median", "std", "sum", "p95", "delta")                                                  # the summary statistics of `statistic`; the highest / lowest value is the `extreme` operation
CSTATS = STATS + ("min", "max")                                                                          # `compare` may compare any of them between two scopes
MINMAX = ("min", "max")
EVSEL = ("last", "first", "all", "most_severe")
CMPOPS = ("gt", "lt", "ge", "le", "eq")
TURNS = ("new", "follow_up", "correction", "clarification_reply")
CEDITS = ("add", "remove", "replace")
DROPS = ("period", "conditions", "statistic", "signal", "event", "groups", "selection", "unit")
ENTITIES = ("trip", "charge_session")
ORDERS = ("asc", "desc")
CALENDAR = ("today", "yesterday", "day before yesterday", "this week", "previous week", "this month", "previous month", "last weekend", "this morning", "last night")
TOOLS_FOCI = ("odometer", "service", "model", "brakes", "now", "period", "style", "eco", "adas", "duration", "all", "battery", "tyres", "camera", "electrical", "cooling", "where", "distance", "eta", "heading", "history")
DECLINES = ("control", "offtopic")
SAYS = ("greeting", "thanks", "help")

# ---- the operations. name: (meaning, label, [(param, type, required-for-a-new-request)]).
# `label` says in plain words what the person wants (used by the frame format and by documentation); `meaning` is the one-line definition shown to the model.
FUNCS = {
    "current_value": ("the latest recorded value of a signal", "latest value", [("signal", "signal", True), ("unit", "unit", False)]),
    "statistic": ("mean, median, std, sum, p95 or delta of a signal over a period (NOT the highest or lowest: that is extreme), optionally only while conditions hold, over selected trips or sessions, or inside an event's window",
                  "numerical summary over an interval", [("signal", "signal", True), ("stat", "stat", False), ("period", "period", False), ("conditions", "conds", False), ("selection", "sel", False), ("event", "event", False), ("unit", "unit", False)]),
    "extreme": ("the highest or lowest value of a signal in a period and when it occurred", "highest or lowest value", [("signal", "signal", True), ("which", "minmax", True), ("period", "period", False), ("conditions", "conds", False), ("selection", "sel", False), ("event", "event", False), ("unit", "unit", False)]),
    "count_events": ("how many times an event occurred in a period", "number of occurrences", [("event", "event", True), ("period", "period", False)]),
    "count_entities": ("how many trips or charging sessions there were in a period", "number of trips or sessions", [("entity", "entity", True), ("trip_class", "tclass", False), ("period", "period", False)]),
    "list_events": ("the individual occurrences of an event (all, first, last or most severe), optionally only while conditions hold", "individual event records", [("event", "event", True), ("which", "evsel", False), ("period", "period", False), ("conditions", "conds", False)]),
    "did_event_happen": ("yes/no: did an event occur in a period", "did the event occur", [("event", "event", True), ("period", "period", False)]),
    "did_condition_hold": ("yes/no: did a threshold or named condition on a signal hold at any time in a period", "did the condition hold", [("signal", "signal", True), ("conditions", "conds", True), ("period", "period", False)]),
    "time_share": ("the percent of time a condition held in a period", "time spent in a state", [("conditions", "conds", True), ("signal", "signal", False), ("period", "period", False)]),
    "compare": ("one signal's statistic in two periods, or under two named conditions, side by side", "comparison between scopes", [("signal", "signal", True), ("stat", "cstat", False), ("period", "period", False), ("period2", "period", False), ("groups", "grps", False)]),
    "when_changed": ("WHEN a signal changed the most (the time of the largest change)", "timing of a change", [("signal", "signal", True), ("period", "period", False), ("conditions", "conds", False)]),
    "event_context": ("what a signal or the signals did in a window before and after one event", "what happened around an event", [("event", "event", True), ("signal", "signal", False), ("which", "evsel", False), ("before", "dur", False), ("after", "dur", False), ("period", "period", False)]),
    "explain_change": ("WHY a signal went up or down: compares a recent period with the one before and lists associated factors", "why a signal changed", [("signal", "signal", True), ("period", "period", False), ("conditions", "conds", False)]),
    "explain_event": ("WHY one event happened: the conditions around it", "why an event happened", [("event", "event", True), ("which", "evsel", False), ("period", "period", False)]),
    "trend": ("is a signal drifting up or down over a period, and by how much", "trend over a period", [("signal", "signal", True), ("period", "period", False), ("conditions", "conds", False)]),
    "rank_trips": ("the top trips or charging sessions ranked by a metric", "ranking of trips or sessions", [("entity", "entity", True), ("metric", "metric", True), ("order", "order", False), ("k", "int", False), ("trip_class", "tclass", False), ("period", "period", False)]),
    "correlate": ("how strongly two signals move together", "relationship between signals", [("signal", "signal", True), ("signal2", "signal", True), ("period", "period", False)]),
    "is_recorded": ("whether this vehicle records a signal at all", "is it recorded", [("signal", "signal", True)]),
    "use_tool": ("a ready-made domain check: service and wear, tyres, weather, driving style, overall health, places", "ready-made check", [("tool", "tool", True), ("focus", "focus", False), ("period", "period", False)]),
    "ask_user": ("ask ONE specific question when something essential is ambiguous", "ask the person", [("question", "text", True)]),
    "unresolved": ("no registered concept is equivalent to what is asked; name the missing quantity", "no matching quantity", [("what", "text", True)]),
    "decline": ("the request is not about recorded vehicle data, or asks to operate the vehicle", "decline", [("reason", "decl", True)]),
    "say": ("greeting, thanks or a request for help", "small talk", [("text", "saykind", True)]),
}
NO_TURN = {"ask_user", "unresolved", "decline", "say"}
ONE_OF = {"compare": [("period2", "groups")]}                                                  # at least one of these arguments is required (a comparison needs a second period or two conditions)
ASK = {"signal": "Which signal do you mean?", "event": "Which event or warning do you mean?", "which": "Do you want the highest or the lowest value?", "conditions": "Which condition should I check?", "entity": "Do you mean trips or charging sessions?",
       "metric": "What should I rank them by?", "tool": "Which check do you want - service, tyres, weather, driving, health or places?", "signal2": "Which second signal should I relate it to?", "period2": "What should I compare it against - another period or another condition?"}


def func_label(name): return FUNCS[name][1]
def func_meaning(name): return FUNCS[name][0]
def func_params(name): return FUNCS[name][2]


# Optional arguments that can be INVENTED by a small model carry their evidence: the words of the message that state them are copied first (`period_words`, `conditions_words`, `unit_words`), then the value is written. plancompile.ground checks where the quoted words come from.
GROUNDED = {"period": "per", "period2": "per2", "conditions": "cond", "unit": "unit"}


def words_name(pn): return pn + "_words"


def all_params(name, grounded=True):
    """The parameters of a function in the order the call writes them: `turn` first (so the model states what kind of message this is before it writes the content), then the function's own (each grounded optional one preceded by its quoted words), then the edit parameters."""
    params = func_params(name)
    if name in NO_TURN: return list(params)
    own = []
    for n, t, r in params:
        if grounded and n in GROUNDED: own.append((words_name(n), "span", False))
        own.append((n, t, r))
    edits = ([("conditions_edit", "cedit", False)] if any(n == "conditions" for n, _, _ in params) else []) + [("drop", "dropl", False)]            # a condition edit exists only where conditions do
    return [("turn", "turn", False)] + own + edits


FN_OP = {"current_value": "current", "statistic": "aggregate", "extreme": "extreme", "count_events": "count", "count_entities": "count", "list_events": "list", "did_event_happen": "exists", "did_condition_hold": "exists", "time_share": "share", "compare": "compare",
         "when_changed": "change", "event_context": "context", "explain_change": "explain", "explain_event": "explain", "trend": "trend", "rank_trips": "rank", "correlate": "correlate", "is_recorded": "capability", "use_tool": "tool",
         "ask_user": "clarify", "unresolved": "unresolved", "decline": "refuse", "say": "chat"}
PARAM_KEY = {"signal": "tgt", "signal2": "tgt2", "stat": "stat", "which": None, "period": "per", "period2": "per2", "conditions": "cond", "unit": "unit", "event": "evt", "groups": "grp", "before": "bef", "after": "aft", "selection": "ent",
             "tool": "tool", "focus": "focus", "question": "ask", "reason": "say", "text": "say", "what": "what", "turn": "turn", "conditions_edit": "cedit", "drop": "drop"}
FLAT_ENT = (("entity", "entity"), ("metric", "rank_by"), ("order", "order"), ("k", "k"), ("trip_class", "klass"))              # flat selection arguments of rank_trips / count_entities -> plan["ent"]
SEL_KEYS = (("entity", "entity"), ("metric", "rank_by"), ("order", "order"), ("k", "k"), ("class", "klass"))                    # keys of a `selection` object -> plan["ent"]


def required_keys(name):
    """The plan keys a NEW request must carry for this function (derived from the same table as the schema and the grammar), as [(plan key, ask question)]; a follow-up gets them from the state it continues."""
    out = []
    for pn, pt, req in func_params(name):
        if not req: continue
        if pn == "which": out.append(("stat" if name == "extreme" else "ev", ASK["which"]))
        elif pn in ("entity", "metric") and name == "rank_trips": out.append(("ent." + dict(FLAT_ENT)[pn], ASK[pn]))
        elif pn == "entity": out.append(("ent.entity", ASK[pn]))
        elif PARAM_KEY.get(pn): out.append((PARAM_KEY[pn], ASK.get(pn, "Could you say more about " + pn + "?")))
    return out


# ------------------------------------------------------------------------------------------------ parsing a call
class CallError(ValueError):
    pass


def parse_call(text):
    """-> (function name, {param: python value}). Accepts exactly one call, optionally wrapped in the tool-call markers; anything else (a truncated call, prose) is an error."""
    t = text.strip()
    if t.startswith(CALL_OPEN): t = t[len(CALL_OPEN):]
    if t.endswith(CALL_CLOSE): t = t[: -len(CALL_CLOSE)]
    t = t.strip()
    if not (t.startswith("[") and t.endswith("]")): raise CallError("not a bracketed call (truncated or prose)")
    try: tree = ast.parse(t[1:-1].strip(), mode="eval")
    except SyntaxError as e: raise CallError(f"not a well-formed call: {e.msg}")
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args: raise CallError("not a single keyword call")
    if call.func.id not in FUNCS: raise CallError(f"unknown function {call.func.id!r}")
    try: args = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except (ValueError, SyntaxError): raise CallError("non-literal argument")
    check_args(call.func.id, args)
    return call.func.id, args


def check_args(fn, args):
    extra = set(args) - {p[0] for p in all_params(fn)}
    if extra: raise CallError(f"unknown argument(s) {sorted(extra)}")


# ------------------------------------------------------------------------------------------------ periods and durations: plain, typed phrases in the call, canonical in the plan
CAL_NAT = {"today": "today", "yesterday": "yesterday", "day_before_yesterday": "day before yesterday", "this_week": "this week", "last_week": "previous week", "this_month": "this month", "last_month": "previous month",
           "last_weekend": "last weekend", "this_morning": "this morning", "last_night": "last night"}
_NAT_CAL = {v: k for k, v in CAL_NAT.items()}


def period_canonical(text):
    """'last 2 weeks' -> 'last:2:week', 'yesterday' -> 'cal:yesterday', 'previous week' -> 'cal:last_week', 'all time' -> 'all', 'on 2026-03-03' -> 'date:2026-03-03'. None when it is not one of the allowed forms."""
    t = (text or "").strip().lower()
    m = re.fullmatch(r"last (\d+) (minute|hour|day|week|month)s?", t)
    if m: return f"last:{int(m.group(1))}:{m.group(2)}"
    if t in ("now", "all"): return t
    if t == "all time": return "all"
    if t == "last week": return "last:7:day"                                                       # the product's documented convention: "last week" = the last 7 days, "last month" = the last 30 days
    if t == "last month": return "last:30:day"
    m = re.fullmatch(r"on (\d{4}-\d{2}-\d{2})", t)
    if m: return "date:" + m.group(1)
    return ("cal:" + _NAT_CAL[t]) if t in _NAT_CAL else None


def period_natural(canon):
    if not canon: return None
    if canon == "all": return "all time"
    if canon == "now": return "now"
    p = canon.split(":", 1)
    if p[0] == "last": a, b = p[1].split(":"); return f"last {a} {b}{'' if a == '1' else 's'}"
    if p[0] == "cal": return CAL_NAT.get(p[1], p[1])
    if p[0] == "date": return "on " + p[1]
    return canon


_DUR = {"second": 1, "minute": 60, "hour": 3600}


def duration_seconds(text):
    """'10 minutes' -> 600 (None when it is not an amount and a unit)."""
    m = re.fullmatch(r"(\d+) (second|minute|hour)s?", (text or "").strip().lower())
    return int(m.group(1)) * _DUR[m.group(2)] if m else None


def duration_natural(sec):
    if not sec: return None
    for unit, n in (("hour", 3600), ("minute", 60), ("second", 1)):
        if sec % n == 0 and sec // n >= 1: k = sec // n; return f"{k} {unit}{'' if k == 1 else 's'}"
    return f"{sec} seconds"


def call_to_plan(name, args):
    """Function call -> canonical plan dict (keys: op fn turn tgt tgt2 stat per per2 cond cedit evt ev grp ent bef aft unit tool focus ask say what drop src)."""
    plan = {"op": FN_OP[name], "fn": name}
    for k, v in args.items():
        if k == "which":
            if name == "extreme": plan["stat"] = v
            else: plan["ev"] = v
        elif k in ("entity", "metric", "order", "k", "trip_class"): continue
        elif k == "selection": plan["ent"] = {dst: v[src] for src, dst in SEL_KEYS if isinstance(v, dict) and v.get(src) is not None}
        elif k.endswith("_words") and k[:-6] in GROUNDED: plan.setdefault("src", {})[GROUNDED[k[:-6]]] = v
        elif k in ("period", "period2"): plan[PARAM_KEY[k]] = period_canonical(v) or v
        elif k in ("before", "after"): plan[PARAM_KEY[k]] = duration_seconds(v)
        elif k in PARAM_KEY and PARAM_KEY[k]: plan[PARAM_KEY[k]] = v
    if name in ("rank_trips", "count_entities"):                                                       # only what the call states: a follow-up edit ("the top 3 instead") changes one part and keeps the rest
        ent = {dst: args[src] for src, dst in FLAT_ENT if args.get(src) is not None}
        if ent: plan["ent"] = ent
    if name == "did_condition_hold": plan["exists_condition"] = True
    if name == "use_tool" and "per" in plan: plan["tool_period"] = plan["per"]
    plan.setdefault("turn", "new")
    return plan


FN_BY_OP = {"current": "current_value", "aggregate": "statistic", "extreme": "extreme", "count": "count_events", "list": "list_events", "share": "time_share", "compare": "compare", "change": "when_changed", "context": "event_context",
            "trend": "trend", "rank": "rank_trips", "correlate": "correlate", "capability": "is_recorded", "tool": "use_tool"}


def plan_to_fn_args(plan):
    """Canonical (resolved) plan -> (function name, ordered argument dict) as the model would write them (None when the plan has no call form). Dialect-neutral."""
    fn = plan.get("fn")
    if fn is None:
        op = plan.get("op")
        fn = ("did_condition_hold" if plan.get("exists_condition") else "did_event_happen") if op == "exists" else ("explain_event" if plan.get("evt") and not plan.get("tgt") else "explain_change") if op == "explain" else FN_BY_OP.get(op)
    if fn is None or fn not in FUNCS or fn in NO_TURN: return None
    args = {"turn": "new"}
    for pn, pt, _ in func_params(fn):
        key = PARAM_KEY.get(pn)
        if pn == "which": v = plan.get("stat") if fn == "extreme" else plan.get("ev")
        elif pn in ("entity", "metric", "order", "k", "trip_class") and fn in ("rank_trips", "count_entities"): v = (plan.get("ent") or {}).get(dict(FLAT_ENT)[pn])
        elif pn == "selection":
            e = plan.get("ent") or {}; v = {src: e[dst] for src, dst in SEL_KEYS if e.get(dst) is not None} or None
        elif pn in ("period", "period2"): v = period_natural(plan.get(key))
        elif pn in ("before", "after"): v = duration_natural(plan.get(key))
        else: v = plan.get(key) if key else None
        if v not in (None, "", [], {}):
            if pn in GROUNDED and (plan.get("src") or {}).get(GROUNDED[pn]): args[words_name(pn)] = plan["src"][GROUNDED[pn]]
            args[pn] = v
    return fn, args


def args_to_pythonic(fn, args):
    q = lambda v: json.dumps(v, ensure_ascii=False)                                                  # ", " and ": " exactly as the grammar writes them
    return f"{fn}(" + ", ".join(f"{k}={q(v)}" for k, v in args.items()) + ")"


def plan_to_call(plan):
    """Canonical (resolved) plan -> the call text (Pythonic form), '' when the plan has none."""
    fa = plan_to_fn_args(plan)
    return args_to_pythonic(*fa) if fa else ""


def plan_to_frame(plan):
    """A compact, NON-call rendering of a resolved plan for the conversation-state lines of the prompt (a copyable call invites the model to repeat it): the operation in plain words, then its arguments, e.g. `numerical summary over an interval | signal=speed | stat=mean | period=last 7 days`. Quoted words and edit parameters are left out."""
    fa = plan_to_fn_args(plan)
    if not fa: return ""
    fn, args = fa
    return " | ".join([func_label(fn)] + [f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}" for k, v in args.items() if k != "turn" and not k.endswith("_words")])


def frame_fields(fn):
    """The fields of the frame format for a function, in order: its own parameters (no quoted words), then conditions_edit where conditions exist. [(name, type)]"""
    out = [(pn, pt) for pn, pt, _ in func_params(fn)] if True else []
    if fn not in NO_TURN and any(n == "conditions" for n, _ in out): out.append(("conditions_edit", "cedit"))
    return out


FRAME_DROP = {"target": "signal", "period": "period", "period2": "period", "conditions": "conditions", "stat": "statistic", "signal": "signal", "event": "event", "groups": "groups", "selection": "selection", "unit": "unit"}     # frame field -> the `drop` component a `none` in a follow-up removes


# Descriptive, semantic labels of the operations for a pretrained schema-conditioned EXTRACTOR (classification by what the person wants, not by function name). A fixed vocabulary of operations, not a list of user phrasings.
FUNC_NL = {"current_value": "asks what a measurement is right now", "statistic": "asks for an average, total, median or other statistic of a measurement over a period of time", "extreme": "asks for the highest, lowest, maximum or minimum value reached",
           "count_events": "asks how many times something happened", "count_entities": "asks how many trips or charging sessions there were", "list_events": "asks to list or show the individual events or warnings",
           "did_event_happen": "asks yes or no whether an event or warning happened", "did_condition_hold": "asks yes or no whether a measurement ever went above or below a level", "time_share": "asks what percentage or share of the time something was true",
           "compare": "asks to compare two time periods or two driving situations", "when_changed": "asks when a measurement changed the most", "event_context": "asks what was happening just before or after an event or warning",
           "explain_change": "asks why a measurement went up or down or got worse", "explain_event": "asks why an event or warning happened", "trend": "asks whether a measurement is trending up or down over time",
           "rank_trips": "asks for the top trips or charging sessions ordered by something, such as the longest or most efficient", "correlate": "asks how two measurements relate to each other", "is_recorded": "asks whether the vehicle records or has data for something",
           "use_tool": "asks for a service, tyre, weather, health or driving-style check", "decline": "an instruction to operate the vehicle or a request unrelated to the vehicle's data", "say": "a greeting or thanks",
           "ask_user": "too vague to tell what is being asked", "unresolved": "about a quantity this vehicle does not record"}


# ---- the REQUESTED RESULT: what the person wants back, independent of how the engine implements it. Typed software derives the executor from the result and the kind of the grounded target (signal / event / trips or sessions); a model is never asked to pick an internal function name.
RESULT_NL = {"latest_value": "asks what a measurement is right now", "summary": "asks for an average, total, median, variability or net change of a measurement over a period", "extreme": "asks for the highest or lowest value reached",
             "count": "asks how many times something happened, or how many trips or sessions there were", "existence": "asks yes or no whether something happened", "list": "asks to list or show individual events or warnings, or the first or last one",
             "time_share": "asks what share, percentage or amount of time something was true", "comparison": "asks to compare two periods or two situations", "change_timing": "asks when a measurement changed or started changing",
             "explanation": "asks why something went up, went down, got worse or happened", "context": "asks what was happening around an event or warning", "trend": "asks whether a measurement is trending up or down",
             "ranking": "asks for the top trips or charging sessions by some measure", "relationship": "asks how two measurements relate to each other", "capability": "asks whether the vehicle records or measures something",
             "domain_check": "asks for a service, tyre, weather, health or driving-style check", "refuse": "tells the assistant to do something, or is not about the vehicle", "unsupported": "asks about the future or something that cannot be known from records"}
STAT_NL = {"mean": "the average or typical value", "sum": "the total amount", "median": "the median value", "std": "how variable or spread out the values are", "p95": "the 95th percentile value", "delta": "how much it rose or fell, the net change",
           "min": "the lowest value", "max": "the highest value"}
RESULT_OF_FN = {"current_value": "latest_value", "statistic": "summary", "extreme": "extreme", "count_events": "count", "count_entities": "count", "list_events": "list", "did_event_happen": "existence", "did_condition_hold": "existence",
                "time_share": "time_share", "compare": "comparison", "when_changed": "change_timing", "event_context": "context", "explain_change": "explanation", "explain_event": "explanation", "trend": "trend", "rank_trips": "ranking",
                "correlate": "relationship", "is_recorded": "capability", "use_tool": "domain_check", "decline": "refuse"}


# ---- the INTENT surface: the model states WHAT RESULT is wanted (a closed, meaningful vocabulary); typed code derives the executor from the result and the kind of the grounded target.
RESULT_OF_FN.update({"ask_user": "ask", "unresolved": "unresolved", "say": "chat"})
RESULT_NL.update({"ask": "too vague to tell what is being asked", "unresolved": "about a quantity that no registered capability means", "chat": "a greeting or thanks"})
TARGET_PARAM = {"current_value": "signal", "statistic": "signal", "extreme": "signal", "count_events": "event", "list_events": "event", "did_event_happen": "event", "did_condition_hold": "signal", "time_share": "signal", "compare": "signal", "when_changed": "signal",
                "event_context": "event", "explain_change": "signal", "explain_event": "event", "trend": "signal", "correlate": "signal", "is_recorded": "signal"}
INTENT_ORDER = ["latest_value", "summary", "extreme", "count", "existence", "list", "time_share", "comparison", "change_timing", "explanation", "context", "trend", "ranking", "relationship", "capability", "domain_check", "refuse", "ask", "unresolved", "chat"]


def fns_of(result): return [fn for fn, r in RESULT_OF_FN.items() if r == result]


def _sname(fn, pn):
    if pn == TARGET_PARAM.get(fn): return "target"
    if pn in ("stat", "which") and fn in ("statistic", "extreme", "compare"): return "statistic"
    return pn


def surface_fields(result):
    """The fields written after `result=` / `turn=` for a result, in order: [(surface name, type)]. `target` is the one signal or event the request is about (its kind selects the executor); `statistic` is the summary statistic, min / max or a comparison statistic according to the result."""
    out = {}; order = []
    for fn in fns_of(result):
        for pn, pt, req in func_params(fn):
            sn = _sname(fn, pn); ty = {"statistic": {"statistic": "stat", "extreme": "minmax", "compare": "cstat"}.get(fn)}.get(sn) if sn == "statistic" else ("target_" + pt if sn == "target" else pt)
            if sn not in out: out[sn] = ty; order.append(sn)
            elif out[sn] != ty and sn == "target": out[sn] = "target"
    if result not in ("ask", "unresolved", "chat", "refuse") and any(n == "conditions" for n in order): order.append("conditions_edit"); out["conditions_edit"] = "cedit"
    return [(n, out[n]) for n in order]


def route(result, sargs, kind_of):
    """Typed routing: (result, surface arguments, the kind of the target id) -> function. kind_of(id) -> 'signal' | 'event' | None."""
    fns = fns_of(result)
    if len(fns) == 1: return fns[0]
    k = kind_of(sargs.get("target")) if sargs.get("target") else None
    if result == "count": return "count_events" if (k == "event" or sargs.get("target")) else "count_entities"
    if result == "existence": return "did_event_happen" if k == "event" else "did_condition_hold"
    if result == "explanation": return "explain_event" if k == "event" else "explain_change"
    return fns[0]


def decode_intent(result, sargs, kind_of):
    """(result, surface arguments) -> (function, canonical arguments)."""
    fn = route(result, sargs, kind_of); args = {}
    for sn, v in sargs.items():
        if sn == "target": args[TARGET_PARAM[fn]] = v
        elif sn == "statistic": args["which" if fn == "extreme" else "stat"] = v
        else: args[sn] = v
    check_args(fn, {k: v for k, v in args.items() if k not in ("turn",)} if False else args)
    return fn, args


def encode_intent(fn, args):
    """(function, canonical arguments) -> (result, surface arguments)."""
    s = {}
    for pn, v in args.items():
        if pn == TARGET_PARAM.get(fn): s["target"] = v
        elif pn in ("stat", "which") and fn in ("statistic", "extreme", "compare"): s["statistic"] = v
        else: s[pn] = v
    return RESULT_OF_FN[fn], s


def surface_required(result):
    """Surface fields a NEW request must carry for this result (only decided for results that map to one function; for a result that maps to several, typed routing + validation decide)."""
    fns = fns_of(result)
    if len(fns) != 1: return {"target"} if result in ("count", "existence", "explanation") and False else set()
    fn = fns[0]; return {_sname(fn, pn) for pn, pt, r in func_params(fn) if r and fn not in NO_TURN}
