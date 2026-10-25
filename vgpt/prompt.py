"""The model's input, in two parts.

STATIC PREFIX (instruction + tool declarations + the registry's named conditions): identical for every message, evaluated once and kept as a saved model state (slm_backend.PrefixLlama).
PER-MESSAGE TEXT (what changes): today's date, the retrieved capability CARDS (identifier, kind, unit, what it means, availability, relationships), the previous resolved plan, the pending clarification, and the message. The model sees the question, the capabilities and the conversation state together and writes ONE tool call.

No worked examples of questions are carried: the instruction defines the operations and the argument forms, the cards carry the meaning of the concepts. Nothing here is a phrase rule."""
import datetime as dt

from . import signal_registry as R
from .dialects import Pythonic
from . import toolspec as T
from .toolspec import call_to_plan, parse_call, plan_to_frame

HEAD = "You are the language front end of a read-only analytics engine over ONE vehicle's recorded data. Turn the person's message into exactly ONE {what}. You never answer from memory and never write numbers; the engine computes them."
CHOOSE = """Choosing the operation and its arguments:
- Pick the operation by what is asked: a current value, a statistic, an extreme, a count, a list of events, did-it-happen, share of time, a comparison, when something changed, what happened around an event, why, a trend, a ranking, a relationship, is-it-recorded, or a domain tool.
- current_value is for the present reading of one signal, including requests phrased as how fast, hot, full, high or low it is now. time_share is ONLY for a percentage, share or duration for which a condition held; never use it to return a signal's present value.
- When a retrieved capability is a tool and its meaning directly answers the request, use use_tool with that tool identifier and an available focus. The tool computes the answer. Do not ask the person's question back.
- Every signal and event identifier must be copied from the Capabilities cards of the message. Read what each card means. If what is asked is a registered concept, use its identifier, even when the card says NOT RECORDED (the engine then explains that the vehicle does not record it). If NONE of the cards means the quantity asked about, use unresolved and name the missing quantity; if a card does mean it, never use unresolved. Never choose a merely related quantity instead.
- A condition is a named condition (listed below; prefer one whenever it fits) or a threshold {"signal": ID, "op": "gt|lt|ge|le|eq", "value": NUMBER, "unit": UNIT} with the number and unit as the person gave them. Two bounds are two thresholds.
- A selection {"entity": "trip|charge_session", "metric": ..., "order": "asc|desc", "k": N, "class": ...} picks which trips or charging sessions a statistic covers (the longest, the latest, the two worst ...). An event on a statistic restricts it to the window of that event.
- period is "now", "all time", "last N minutes|hours|days|weeks|months", "last week", "last month", "today", "yesterday", "day before yesterday", "this week", "previous week", "this month", "previous month", "last weekend", "this morning", "last night" or "on YYYY-MM-DD" (use Today to complete a date). "last week" and "last month" are the product's own convention (7 and 30 days); any other span is counted in the unit the person used. before and after are amounts such as "10 minutes"."""
TOOLCALL_TAIL = """- turn: "new" is a standalone request (give every required argument). "follow_up", "correction" or "clarification_reply" continue the previous request or the pending question shown in the state lines: write only what changes. conditions_edit says how the conditions change: "add", "remove" or "replace" (replace = these conditions instead of the earlier ones). drop lists parts of the earlier request to remove (period, conditions, statistic, signal, event, groups, selection, unit).
- Earlier turns of the conversation are context only. A message about something else is a new, complete call; never repeat an earlier call unless the message asks for it again.
- Before a period, a condition or a unit, copy EXACTLY the words of the message that state it (period_words, conditions_words, unit_words), then write its value. If the message has no such words, leave that argument out: never add a period, condition or unit of your own, and never copy arguments from the examples.
- Give any other optional argument only when the message states it.
- ask_user only when an ESSENTIAL argument has two or more reasonable readings. Ask ONE specific question naming the missing distinction; never repeat or merely rephrase the person's message. decline is for a request to operate the vehicle or something unrelated to the vehicle data. say is for a greeting, thanks or a request for help."""
PLAIN_TAIL = """- turn: "new" is a standalone request (give every required argument). "follow_up", "correction" or "clarification_reply" continue the previous request or the pending question shown in the state lines: write only what changes. conditions_edit says how the conditions change: "add", "remove" or "replace" (replace = these conditions instead of the earlier ones). drop lists parts of the earlier request that no longer apply.
- Earlier turns of the conversation are context only. A message about something else is a new, complete call; never repeat an earlier call unless the message asks for it again.
- Give an optional argument (period, conditions, unit, ...) only when the message states it: if it does not, leave the argument out. Never add a period, condition or unit of your own.
- ask_user only when the message has no reasonable reading; ask ONE specific question. decline is for a request to operate the vehicle or something unrelated to the vehicle data. say is for a greeting, thanks or a request for help.
- Write exactly one call, as a JSON array with one element: [{"name": <operation>, "arguments": {...}}]."""
FRAME_TAIL = """- Write ONE line: `operation | turn="new" | field=value | field=value ...`. Write EVERY field of that operation, in the order listed, and write none for a field the message does not state. Strings are in double quotes. Never add a period, condition or unit the message does not state, and never copy values from the examples.
- turn is "new" for a standalone request. "follow_up", "correction" or "clarification_reply" continue the previous request or the pending question shown in the state lines: then write keep for every field that stays as it was, none to remove a field, or the new value. conditions_edit says how new conditions combine with the earlier ones: "add", "remove" or "replace".
- ask_user only when the message has no reasonable reading; ask ONE specific question. decline is for a request to operate the vehicle or something unrelated to the vehicle data. say is for a greeting, thanks or a request for help."""


EVIDENCE_BULLET = '- When lines from a reading aid precede the message, treat them as evidence: a period, restriction, number or selection that the aid found is probably stated, and one it reports as \\"none found\\" is probably not stated - but the message itself decides.'


INTENT_TAIL = """- Write ONE line: `result=<what the person wants back> | turn="new" | field=value | field=value ...`. Choose the result by what is WANTED (a number now, a statistic, the highest value, a count, yes/no, a list, a share of time, a comparison, when it changed, why, what happened around an event, a trend, a ranking, a relationship, whether it is recorded, a domain check), not by how it will be computed. `target` is the one signal or event the request is about. Write EVERY field of that result in the order listed, and write none for a field the message does not state. Strings are in double quotes. Never add a period, condition or unit the message does not state, and never copy values from the examples.
- turn is "new" for a standalone request. "follow_up", "correction" or "clarification_reply" continue the previous request or the pending question shown in the state lines: then write keep for every field that stays as it was, none to remove a field, or the new value. conditions_edit says how new conditions combine with the earlier ones: "add", "remove" or "replace".
- ask only when the message has no reasonable reading; ask ONE specific question. refuse is for a request to operate the vehicle or something unrelated to the vehicle data. chat is for a greeting or thanks."""


def intent_spec(registry):
    """The results of the intent surface and their fields, generated from the one plan-language definition: result (what is wanted; fields * = required for a new request)."""
    out = ["Results (each field is written as name=value, in this order):"]
    for r in T.INTENT_ORDER:
        req = T.surface_required(r); out.append(f"- {r} ({T.RESULT_NL[r]}): " + ", ".join(sn + ("*" if sn in req else "") for sn, st in T.surface_fields(r)))
    return "\n".join(out)


def frame_spec(registry):
    """The operations of the frame format, generated from the one plan-language definition: name (what the person wants): fields in order (* = required for a new request)."""
    out = ["Operations (each field is written as name=value, in this order):"]
    for fn in T.FUNCS:
        required = {pn for pn, pt, r in T.func_params(fn) if r and fn not in T.NO_TURN}; fields = [pn + ("*" if pn in required else "") for pn, pt in T.frame_fields(fn)]
        out.append(f"- {fn} ({T.func_label(fn)}; {T.func_meaning(fn)}): " + ", ".join(fields))
    return "\n".join(out)


def instruction(dialect=None, evidence=False):
    framed = dialect is not None and dialect.name in ("frame", "intent")
    choose = CHOOSE.replace("- Every signal and event identifier must be copied", EVIDENCE_BULLET + "\n- Every signal and event identifier must be copied", 1) if evidence else CHOOSE
    return HEAD.format(what="line" if framed else "tool call") + "\n\n" + choose + "\n" + (INTENT_TAIL if (dialect is not None and dialect.name == "intent") else FRAME_TAIL if framed else PLAIN_TAIL.replace("Write exactly one call, as a JSON array with one element: [{\"name\": <operation>, \"arguments\": {...}}].", "Write exactly one tool call.") if (dialect is not None and dialect.name == "qwen") else PLAIN_TAIL if (dialect is not None and not dialect.grounded) else TOOLCALL_TAIL)


def system_text(registry, dialect=None, evidence=False):
    conds = "; ".join(f"{c.id} = {c.title}" for c in registry.items if c.kind == "condition")
    classes = ", ".join(c.id.split(":", 1)[1] for c in registry.items if c.kind == "tripclass")
    metrics = ", ".join(sorted({c.id.split(".", 1)[1] for c in registry.items if c.kind == "metric"}))
    framed = dialect is not None and dialect.name in ("frame", "intent")
    return instruction(dialect, evidence) + f"\n\nNamed conditions: {conds}.\nTrip classes: {classes}. Ranking metrics: {metrics}." + (("\n\n" + (intent_spec(registry) if dialect.name == "intent" else frame_spec(registry))) if framed else "")


def tools(registry, dialect=None, level="rich"):
    """The tool declarations of the model's own chat template, generated from the one authoritative plan-language definition (toolspec + schema)."""
    return (dialect or Pythonic()).declarations(registry, level)


def today_line(now, tz=0):
    d = dt.datetime.fromtimestamp((now or 0) + (tz or 0), dt.timezone.utc)
    return f"Today: {d.date().isoformat()} ({d.strftime('%A')})"


def card_ids(hits, registry, state_ids=(), max_hits=6):
    """Which capabilities the model is shown: the retrieved ones (signals, events, tools, trip classes and metrics; named conditions are already in the static prefix), the capabilities of the previous plan (a follow-up keeps them), and what is related to them. Order = retrieval order."""
    ids = []; direct = [h for h in hits if h.sem_rank != -1 or h.lex_rank != -1]; related = [h for h in hits if h.sem_rank == -1 and h.lex_rank == -1]
    for h in direct[:max_hits] + related:
        c = registry.by_id.get(h.id)
        if c is not None and c.kind in ("signal", "event", "tool") and h.id not in ids: ids.append(h.id)      # ranking metrics, trip classes and named conditions are listed once in the static prefix
    for i in state_ids:
        if i in registry.by_id and i not in ids and registry.by_id[i].kind in ("signal", "event", "tool"): ids.append(i)
    return ids


def plan_ids(plan):
    """Capability identifiers a resolved plan refers to."""
    out = []
    if not plan: return out
    for k in ("tgt", "tgt2", "evt"):
        if plan.get(k): out.append(plan[k])
    for c in plan.get("cond") or []:
        if isinstance(c, dict) and c.get("signal"): out.append(c["signal"])
    if plan.get("tool"): out.append("tool:" + plan["tool"])
    return list(dict.fromkeys(out))


def state_lines(prev_plan=None, pending=None):
    """The conversation state, ALWAYS written out compactly (never left to be inferred from the earlier chat turns): the previous resolved request; and, when a clarification is open, the question that was asked, the question it is about and the plan so far."""
    out = []
    f = plan_to_frame(prev_plan) if prev_plan else ""
    if f: out.append("Previous request: " + f)
    if pending:
        out.append(f"Awaiting your answer to: “{pending['ask']}” (about: “{pending['original']}”)")
        pf = plan_to_frame(pending["plan"]) if pending.get("plan") else ""
        if pf: out.append("Plan so far: " + pf)
    return out


def message_text(question, cards, now, tz=0, prev_plan=None, pending=None, feedback=None, evidence=None):
    """The user turn: today, the capability cards, the explicit conversation state, an optional repair note, and the message."""
    lines = [today_line(now, tz), "Capabilities (id | kind | unit | meaning):", cards or "(none matched)"] + state_lines(prev_plan, pending)
    for f in feedback or []: lines.append("Check: " + f)
    lines += list(evidence or [])
    lines.append("Message: " + question)
    return "\n".join(lines)


def call_turn(call, dialect=None):
    """The assistant turn for a call written in the Pythonic form (as the examples are): re-written in the model's own dialect."""
    fn, args = parse_call("[" + call + "]")
    return (dialect or Pythonic()).assistant_text(fn, args)


# ------------------------------------------------------------------------------------------------ format examples
# A few worked examples as real chat turns, to show the call format and the argument forms (periods, thresholds, state edits, the unresolved / decline / ask outcomes). They are not a bank of questions: one per KIND of call. Cards are generated from the registry, so the prompt stays consistent with it.
EXAMPLE_NOW = 1774000000                      # a fixed example date; only the date line is used
_EX = [   # a short conversation: (message, card ids, the call). Every function appears about once (state edits use other functions than the plain statistic), in mixed order. A period, condition or unit is preceded by the words of the message that state it.
    ("What is the engine oil pressure?", ["brake_pressure"], 'unresolved(what="engine oil pressure")'),
    ("How hot did the motor run on average over the last 5 days while driving?", ["motor_temperature"],
     'statistic(turn="new", signal="motor_temperature", stat="mean", period_words="over the last 5 days", period="last 5 days", conditions_words="while driving", conditions=["driving"])'),
    ("Restrict that to motorway driving.", ["motor_temperature"], 'statistic(turn="follow_up", conditions_words="motorway driving", conditions=["highway"], conditions_edit="replace")'),
    ("Why did the cabin temperature go up in the last 10 days?", ["cabin_temperature", "hvac_power"], 'explain_change(turn="new", signal="cabin_temperature", period_words="in the last 10 days", period="last 10 days")'),
    ("Switch on the seat heating", [], 'decline(reason="control")'),
    ("Show every time lane keeping switched off in the last 3 days", ["ADAS_DISENGAGE", "adas_state"], 'list_events(turn="new", event="ADAS_DISENGAGE", which="all", period_words="in the last 3 days", period="last 3 days")'),
    ("How much of last week was the cabin warmer than 26 degrees?", ["cabin_temperature", "ambient_temperature"],
     'time_share(turn="new", conditions_words="warmer than 26 degrees", conditions=[{"signal": "cabin_temperature", "op": "gt", "value": 26, "unit": "C"}], period_words="last week", period="last week")'),
    ("When did the battery temperature jump the most?", ["battery_temperature", "thermal_rise_rate"], 'when_changed(turn="new", signal="battery_temperature")'),
    ("Is the battery health drifting down over the last 3 months?", ["battery_soh", "battery_soc"], 'trend(turn="new", signal="battery_soh", period_words="over the last 3 months", period="last 3 months")'),
    ("How many times did I corner hard last night?", ["HARSH_CORNER", "lateral_acceleration"], 'count_events(turn="new", event="HARSH_CORNER", period_words="last night", period="last night")'),
    ("And on the week before?", ["HARSH_CORNER", "lateral_acceleration"], 'count_events(turn="follow_up", period_words="the week before", period="previous week")'),
    ("Is the motor hotter on the highway than in the city?", ["motor_temperature", "motor_power"], 'compare(turn="new", signal="motor_temperature", stat="mean", groups=["highway","city"])'),
    ("Do you record wheel speed on the front left?", ["wheel_speed_fl"], 'is_recorded(turn="new", signal="wheel_speed_fl")'),
    ("The four shortest charging sessions", ["charging_power", "CHARGE_START"], 'rank_trips(turn="new", entity="charge_session", metric="duration", order="asc", k=4)'),
    ("Does the cabin temperature follow the outside temperature?", ["cabin_temperature", "ambient_temperature"], 'correlate(turn="new", signal="cabin_temperature", signal2="ambient_temperature")'),
    ("What was happening in the 10 minutes before the last tyre warning?", ["TPMS_LOW", "tyre_pressure"], 'event_context(turn="new", event="TPMS_LOW", which="last", before="10 minutes")'),
    ("I meant the 12 volt battery, not the tyres", ["LOW_12V", "TPMS_LOW"], 'event_context(turn="correction", event="LOW_12V")'),
    ("Did the 12 volt battery warn me last night?", ["LOW_12V", "aux_battery_voltage"], 'did_event_happen(turn="new", event="LOW_12V", period_words="last night", period="last night")'),
    ("Why did the 12 volt warning come on?", ["LOW_12V", "aux_battery_voltage"], 'explain_event(turn="new", event="LOW_12V", which="last")'),
    ("How was it?", [], 'ask_user(question="Which signal or event do you mean, and for which period?")'),
    ("The battery temperature, yesterday", ["battery_temperature", "cell_temperature_max"], 'statistic(turn="clarification_reply", signal="battery_temperature", period_words="yesterday", period="yesterday")'),
    ("How do my brakes look?", ["brake_pad_life", "tool:service"], 'use_tool(turn="new", tool="service", focus="brakes")'),
    ("Thanks, that helps", [], 'say(text="thanks")'),
    ("Average speed on my two longest trips", ["speed", "distance_driven"], 'statistic(turn="new", signal="speed", stat="mean", selection={"entity": "trip", "metric": "distance_driven", "order": "desc", "k": 2})'),
    ("What was the lowest charge level yesterday?", ["battery_soc", "battery_soh"], 'extreme(turn="new", signal="battery_soc", which="min", period_words="yesterday", period="yesterday")'),
    ("What is the battery charge right now?", ["battery_soc", "battery_soh"], 'current_value(turn="new", signal="battery_soc")'),
]


CORE = {"unresolved", "statistic", "explain_change", "decline", "list_events", "time_share", "count_events", "compare", "rank_trips", "event_context", "ask_user", "extreme", "current_value", "use_tool"}       # the operations a short example conversation still shows (the others are defined by the schema / frame spec)


def shots(registry, dialect=None, core=False):
    """[(user text, assistant text)]: the examples whose capabilities exist in this registry, as one short conversation. A continuation (follow-up, correction, clarification reply) stands right after the turn it continues and carries the same explicit state lines the live prompt carries."""
    out = []; prev = None; asked = None; last_q = None
    if core == "none": return []                                   # no worked examples at all: the definitions and the cards only
    for q, ids, call in _EX:
        if core and call.split("(")[0] not in CORE: prev = asked = None; continue
        ids = [i[5:] if i.startswith("tool:") else i for i in ids]
        if any((i not in registry.by_id and "tool:" + i not in registry.by_id) for i in ids): prev = asked = None; continue
        ids = [("tool:" + i if "tool:" + i in registry.by_id else i) for i in ids]
        fn, args = parse_call("[" + call + "]"); plan = call_to_plan(fn, args); turn = args.get("turn", "new")
        pend = {"ask": asked, "original": last_q, "plan": None} if (turn == "clarification_reply" and asked) else None
        out.append((message_text(q, registry.card_text(ids), EXAMPLE_NOW, 0, prev if turn in ("follow_up", "correction") else None, pend), (dialect or Pythonic()).assistant_text(fn, args)))
        if fn == "ask_user": asked = args["question"]; last_q = q; prev = None
        else:
            asked = None
            if fn in ("ask_user", "unresolved", "decline", "say"): prev = None
            else: prev = plan
    return out
