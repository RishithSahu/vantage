"""Typed dialogue state and typed follow-up edits.

The conversation is not a string history: after every answered question the assistant holds the TYPED query it answered (target concept, operation, statistic, window, groups, event reference).  A follow-up utterance is analysed into a small edit of that query - a different entity of the same physical family, a different window, a different statistic, an assessment of the value just given, a second concept to set beside it, a status question about the fault just listed - and the edited query is executed by the same typed engine that answers every other question (so units, coverage, evidence and verification are the engine's, not this module's).

Nothing here recognises a particular sentence.  The inputs are: the concept mentions found by the ontology lexicon, the time expressions found by the typed time parser, and a few closed-class function words that mark the speech act (pronouns, comparison, assessment, status, change).  If the typed state cannot support the edit the module declines and the ordinary pipeline answers or asks."""
import os
import re

from . import analytics as A
from . import nl2ir
from . import signal_registry as R
from .query_ir import Query, TimeSpec

FAMILY_UNITS = {"C": "temperature", "%": "percent"}


def _result(*a, **k):
    from .semantic_ops import _result as r
    return r(*a, **k)


def topic(brain):
    """The typed query of the last answered turn, or None."""
    st = getattr(brain, "sem_topic", None); turn = getattr(brain, "turn", 0)
    if st is not None and st[0] >= turn - 1: return st[1]
    q = getattr(getattr(brain, "qcs", None), "last", None)
    return q if q is not None else (st[1] if st is not None else None)


def remember(brain, q):
    """Record the typed query this turn answered, so that the next turn can edit it."""
    brain.sem_topic = (getattr(brain, "turn", 0), q)


def _window(q):
    t = q.scope.time
    return t


def _run(brain, q, route, **facts):
    qc = brain.qc()
    if qc is False: return None
    r = qc.execute(q)
    if r.status not in ("answered", "no_data"): return None
    remember(brain, q)
    return _result(r.answer.text, route, "ANALYSIS", query=q.to_text(), **facts)


def _concept_set(cs, exclude=()):
    return [c for c in cs if c in R.CONCEPTS and c not in exclude]


def _mentions(t, family=None):
    """Every ontology concept the message names; an ambiguous phrase ("the battery") yields ALL its candidate concepts, so that the typed state can disambiguate it by physical family instead of an arbitrary first pick."""
    low = " " + re.sub(r"[^a-z0-9%/.-]+", " ", t.lower()) + " "; out = []
    for phrase, names in R.LEXICON.items():
        p = phrase.lower().replace("-", " ")
        if re.search(r"(?<![a-z0-9])" + re.escape(p).replace(chr(92) + " ", "[ -]") + r"(?![a-z0-9])", low): out += [n for n in sorted(names) if n in R.CONCEPTS and n not in out]
    return out


def try_resolve(brain, raw, normalised):
    """Entry point from the brain: None unless a typed state exists and the message edits it.  Without any state, a message that supplies a restriction but no measure is an incomplete typed query: the missing slot is asked for."""
    if os.environ.get("VGPT_NO_DIALOGUE") == "1" or topic(brain) is None: return None
    from . import semantic_ops
    t = str(raw).lower().replace(chr(8217), "'")
    return resolve(brain, raw, t, _mentions(t + " " + normalised.lower()))


def _family(c):
    return R.CONCEPTS[c].unit


def _times(brain, t):
    from .nl2ir import _S, _times
    try: return _times(_S(nl2ir.normalise(t)), brain.now, brain.tz)
    except Exception: return []


ACTS = {
    "new_question": "a different question about something else: it asks for a value, a list, advice or a priority that the previous answer did not cover",
    "swap_entity": "the same question again but about another sensor instead; the previous quantity is no longer wanted",
    "change_window": "how much the same quantity changed over a stated period",
    "compare_with": "wants the previous quantity together with another quantity, side by side or contrasted; both are wanted",
    "assess_value": "whether the value just given is normal",
    "fault_status": "whether the very fault or event just discussed is still active or present now",
    "evidence": "the data or readings behind the change just reported",
    "switch_metric": "the same groups and period compared on another measure",
    "same_window": "whether the comparisons use the same dates",
}


def _summary(q):
    from .paraphrase import describe
    try: return describe(q)
    except Exception: return q.to_text()


def _tokens(text): return set(re.findall(r"[a-z0-9]+", text.lower()))


FUNCTION_WORDS = set("a an the of my me is are was were do does did what how about and or it its that this those them these on in at to for with from by be can could would should i you your we our there their then now also so just only please tell show give".split())


def _siblings(target, t):
    """Concepts of the same physical family as the topic whose names carry a content word of the message (the ontology, not a phrase list, decides: 'the battery' next to a motor temperature names the battery temperature)."""
    if target not in R.CONCEPTS: return []
    fam = _family(target); words = _tokens(t) - FUNCTION_WORDS - _tokens(R.CONCEPTS[target].label); out = []
    for name, c in R.CONCEPTS.items():
        if name == target or _family(name) != fam: continue
        names = _tokens(" ".join((c.label,) + tuple(c.synonyms)))
        if words & names: out.append((len(words & names), name))
    return [n for _, n in sorted(out, reverse=True)]


def feasible_acts(brain, q, t, concepts):
    """The speech acts the typed state can actually support for this message (type constraints): an act whose required argument is absent, or whose presupposition the message contradicts, is not a candidate."""
    target = q.target; ok = ["new_question"]; numeric = target in R.CONCEPTS and R.CONCEPTS[target].dtype == "numeric"
    others = [c for c in concepts if c != target]; sib = _siblings(target, t) if target in R.CONCEPTS else []; tms = _times(brain, t)
    foreign = [c for c in others if target not in R.CONCEPTS or _family(c) != _family(target)]          # the message names a quantity of another kind: a question of its own unless it asks to set the two side by side
    if q.op == "compare" and q.groups:
        if concepts: ok.append("switch_metric")                                                          # a measure named, groups and period inherited
        else: ok.append("same_window")
        return ok
    if foreign:
        return ok + (["compare_with"] if target in R.CONCEPTS else [])
    if sib: ok.append("swap_entity")
    if tms and numeric: ok.append("change_window")
    if (others or sib) and target in R.CONCEPTS: ok.append("compare_with")
    if numeric and not concepts: ok.append("assess_value")
    if q.event is not None or q.op in ("list_events", "event_context"): ok.append("fault_status")
    if q.agg == "delta" or q.op in ("trend", "detect_change"): ok.append("evidence")
    return ok


LETTERS = "ABCDEFGHI"
ARGLESS = {"fault_status", "evidence", "assess_value", "same_window", "change_window"}
MARGIN = 3.0
ORDERS = int(__import__("os").environ.get("VGPT_ACT_ORDERS", "2"))        # option orders averaged (2 = measured configuration; 1 halves the cost)
_PRIORS = {}


def _letter_logprobs(llm, system, body, n):
    import math
    import numpy as np
    import llama_cpp
    prompt = "<|im_start|>system" + chr(10) + system + "<|im_end|>" + chr(10) + "<|im_start|>user" + chr(10) + body + "<|im_end|>" + chr(10) + "<|im_start|>assistant" + chr(10) + "Answer:"
    ids = llm.tokenize(prompt.encode(), add_bos=False, special=True); llm.reset(); llm.eval(ids)
    z = np.ctypeslib.as_array(llama_cpp.llama_get_logits(llm.ctx), shape=(llm.n_vocab(),)).astype(np.float64); z = z - z.max(); lp = z - math.log(np.exp(z).sum()); out = []
    for i in range(n):
        toks = {llm.tokenize((" " + LETTERS[i]).encode(), add_bos=False)[-1], llm.tokenize(LETTERS[i].encode(), add_bos=False)[-1]}
        out.append(max(lp[t] for t in toks))
    return np.array(out)


def decide_act(brain, raw, q, options, args=""):
    """The dialogue act, chosen by the permitted frozen language model among the acts the typed state supports.  Decision procedure (measured in bench/dialogue_acts2.py: 13/17 against 3/17 for plain constrained generation with this model): letter-labelled multiple choice, every option scored by its next-token log-probability, contextual calibration against a content-free message, averaged over two option orders.  None when no model is available - the module then declines and the ordinary pipeline answers."""
    import math
    import numpy as np
    slm = getattr(brain, "slm", None)
    if slm is None or not hasattr(slm, "_ensure") or len(options) < 2: return None
    system = "You decide how a vehicle-data assistant should read a message in a conversation. Answer with the letter of the best option."
    summary = _summary(q); named = args or "no sensor, no event and no period of its own"
    def body(order, m):
        lines = chr(10).join(f"{LETTERS[i]}. {ACTS[o]}" for i, o in enumerate(order))
        return f"Previous answer was about: {summary}{chr(10)}New message: {m}{chr(10)}The message names: {named}{chr(10)}Options:{chr(10)}{lines}{chr(10)}Answer with one letter."
    try:
        with slm._lock:
            slm._ensure(); llm = slm.slm.llm; tot = {o: 0.0 for o in options}
            for order in ((list(options), list(options)[::-1]) if ORDERS >= 2 else (list(options),)):
                key = (summary, tuple(order)); prior = _PRIORS.get(key)                       # the content-free prior depends only on the previous answer and the option order: computed once
                if prior is None: prior = _PRIORS[key] = _letter_logprobs(llm, system, body(order, "N/A"), len(order))
                s = _letter_logprobs(llm, system, body(order, raw), len(order))
                z = s - prior; z = z - z.max(); p = np.exp(z) / np.exp(z).sum()
                for i, o in enumerate(order): tot[o] += math.log(p[i] + 1e-9)
            slm.last = slm.clock(); slm._arm()
    except Exception: return None
    best = max(tot, key=tot.get)
    # acts that carry no argument of their own (they only point back at the previous answer) must beat a new question by a clear margin on the calibrated scores; acts that bring an explicit argument (a sensor, a metric) are not held to it
    if best in ARGLESS and tot[best] - tot.get("new_question", tot[best]) < MARGIN: return "new_question"
    return best


def resolve(brain, raw, t, cs):
    """-> a result dict, or None when the message is not an edit of the typed state.  The speech act comes from the model among the feasible ones; the arguments come from the ontology and the typed time parser; the edited query is executed by the typed engine."""
    q = topic(brain)
    if q is None or (q.target is None and q.event is None and not q.groups): return None
    concepts = _concept_set(cs); target = q.target
    named = [R.label_of(c) for c in concepts]; tms0 = [x for x in _times(brain, t) if x.type != "current"]          # "now" is a discourse marker here, not a period of its own
    args = "; ".join(filter(None, ["the measure " + ", ".join(named) if named else "", "a period of its own" if tms0 else "no period of its own", "no group of its own" if q.groups else ""]))
    feas = feasible_acts(brain, q, t, concepts)
    try: alone = nl2ir.parse(raw, now=brain.now, tz=brain.tz, profile=brain.qc().profile, state=None)
    except Exception: alone = None
    try: alone = nl2ir.parse(raw, now=brain.now, tz=brain.tz, profile=brain.qc().profile, state=None)
    except Exception: alone = None
    _meas_tokens = set().union(*[_tokens(" ".join((R.CONCEPTS[c].label,) + tuple(R.CONCEPTS[c].synonyms))) for c in concepts if c in R.CONCEPTS]) if concepts else set()
    _elliptical = bool(concepts) and (_tokens(t) - FUNCTION_WORDS - {"what", "about", "how", "and", "also", "same", "now", "then", "for", "the", "a", "an", "of", "my", "it", "too", "instead", "rather", "show", "me", "give", "tell", "is", "are", "was", "does", "do", "its", "their", "temp", "level"}) <= _meas_tokens                                  # nothing in the message but the measure: a fragment, not a question of its own
    if q.op == "compare" and q.groups and concepts and not tms0 and _elliptical and alone is not None and not alone.ok:
        # frame unification: the typed parser cannot make an executable query from the message alone, and the message supplies exactly the one slot the compare frame lacks (the measure): the groups and the window are inherited
        metric = alone.query.target if (alone.query is not None and alone.query.target in concepts) else concepts[0]
        q2 = q.clone(); q2.target = metric; q2.agg = q.agg or "mean"
        return _run(brain, q2, "group_compare_edit", groups=list(q.groups), concept=metric)
    sib = _siblings(target, t) if target in R.CONCEPTS else []
    content = _tokens(t) - FUNCTION_WORDS - _tokens(R.CONCEPTS[target].label if target in R.CONCEPTS else "")
    covered = bool(sib) and content <= _tokens(" ".join((R.CONCEPTS[sib[0]].label,) + tuple(R.CONCEPTS[sib[0]].synonyms)))      # every content word of the message is accounted for by the sibling entity: the message is nothing but the entity
    if "swap_entity" in feas and sib and covered and not concepts and not tms0 and q.op != "compare":
        # frame unification: the message names an entity of the same physical family and nothing else (no measure, no period): the question frame of the topic is kept and the entity replaced
        q2 = q.clone(); q2.target = sib[0]; q2.source = "followup"
        return _run(brain, q2, "entity_swap", concept=sib[0], previous=target)
    act = feas[0] if len(feas) == 1 else decide_act(brain, raw, q, feas, args)
    if act in (None, "new_question"): return None
    if act == "switch_metric" and q.op == "compare" and q.groups and concepts:
        _m = alone.query.target if (alone is not None and alone.query is not None and alone.query.target in concepts) else concepts[0]
        q2 = q.clone(); q2.target = _m; q2.agg = q.agg or "mean"
        return _run(brain, q2, "group_compare_edit", groups=list(q.groups), concept=_m)
    if act == "same_window" and q.op == "compare" and q.groups:
        tm = q.scope.time; lo = brain.now - (tm.value or 14) * 86400 if tm.type == "relative" and tm.unit == "day" else None
        span = f"{A.stamp(lo)} to {A.stamp(brain.now)}" if lo else tm.to_text()
        return _result(f"Yes. The comparisons in this conversation use the same groups ({' vs '.join(q.groups)}) over the same window ({span}); only the metric differs.", "comparison_window", "STATUS")
    if act == "fault_status": return _fault_status(brain, q)
    if act == "assess_value": return _normality(brain, q)
    if act == "compare_with":
        others = [c for c in concepts if c != target] or _siblings(target, t)
        if others: return _side_by_side(brain, q, others[0])
    if act == "swap_entity":
        sib = _siblings(target, t)
        if sib:
            q2 = q.clone(); q2.target = sib[0]; q2.source = "followup"
            return _run(brain, q2, "entity_swap", concept=sib[0], previous=target)
    tms = _times(brain, t)
    if act == "change_window" and tms:
        q2 = q.clone(); q2.op = "aggregate"; q2.agg = "delta"; q2.scope.time = tms[0]; q2.source = "followup"
        return _run(brain, q2, "change_over_window", concept=target)
    if act == "evidence": return _evidence(brain, q)
    return None


def observe(brain, text):
    """After an answered turn, make sure the typed state exists: the typed reading of the question that was just answered (whichever route answered it) is what the next message edits.  A question that names two groups but no metric ("compare city and highway") is recorded as a group comparison with the default 14-day window."""
    if os.environ.get("VGPT_NO_DIALOGUE") == "1": return
    st = getattr(brain, "sem_topic", None)
    if st is not None and st[0] == getattr(brain, "turn", 0): return
    try: p = nl2ir.parse(text, now=brain.now, tz=brain.tz, profile=brain.qc().profile, state=None)
    except Exception: return
    q = p.query
    if q is not None and (q.target or q.event is not None or q.groups): remember(brain, q); return
    try:
        groups = [g for _, g in nl2ir._predicates(nl2ir._S(nl2ir.normalise(text)), [])]
        if len(groups) == 2: remember(brain, Query(op="compare", groups=groups, agg="mean", scope=Query().scope.__class__(time=TimeSpec("relative", 14, "day"))))
    except Exception: pass


def _fault_status(brain, q):
    """Whether the event just discussed is still active: an event record is history; activity needs a current signal."""
    et = q.event.type if q.event is not None else None
    sel = et and et != "ANY_WARNING"
    if sel:
        row = brain.c.execute("SELECT ts,type,detail FROM events WHERE ts<=? AND type=? ORDER BY ts DESC LIMIT 1", (brain.now, et)).fetchone()
    else:
        from . import signal_registry as _R
        warning_types = []
        for group in ("ANY_WARNING", "DTC_SET"):
            warning_types += list(getattr(_R, "EVENT_GROUPS", {}).get(group, (group,)))
        warning_types = sorted(set(warning_types))
        marks = ",".join("?" * len(warning_types))
        row = brain.c.execute(f"SELECT ts,type,detail FROM events WHERE ts<=? AND type IN ({marks}) ORDER BY ts DESC LIMIT 1", (brain.now, *warning_types)).fetchone()
    if not row: return _result("No such event is recorded, so there is nothing to be active.", "fault_status", "STATUS", available=False)
    ts, typ, detail = row; low = (detail or "").lower(); cur = []
    from . import live_snapshot
    snap = live_snapshot.snapshot(brain.c, brain.now); vals = snap.get("values", {}) if snap.get("available") else {}
    if "coolant" in low and vals.get("coolant_flow") is not None: cur.append(f"current coolant flow is {vals['coolant_flow']:.1f} L/min")
    if typ == "WARN_BATT_OVERTEMP" and vals.get("battery_temperature") is not None: cur.append(f"current pack temperature is {vals['battery_temperature']:.1f} C (the warning level is 40 C)")
    ev = f"The latest relevant record is {typ} at {A.stamp(ts)}: {detail}."
    tail = " This is a stored historical record; it does not by itself show whether the condition is active now." + (" Current evidence: " + "; ".join(cur) + "." if cur else " No current status signal for it is available.")
    simulated = " The record is labelled illustrative/simulated, so it is not confirmation of a real fault." if "illustrative" in low or "simulated" in low else ""
    return _result(ev + tail + simulated, "fault_status", "STATUS", event_ts=ts, event_type=typ)


def _normality(brain, q):
    """The value just reported against this vehicle's own recent history (14 days): inside the typical band or not."""
    qc = brain.qc(); tgt = q.target
    def agg(a):
        x = Query(op="aggregate", target=tgt, agg=a, scope=q.scope.__class__(time=TimeSpec("relative", 14, "day")))
        r = qc.execute(x); return float(r.result.value) if r.status == "answered" and r.result is not None and isinstance(r.result.value, (int, float)) else None
    cr = qc.execute(Query(op="current_value", target=tgt))
    if cr.status != "answered" or not isinstance(getattr(cr.result, "value", None), (int, float)): return None
    v = float(cr.result.value); mean, sd, lo, hi = agg("mean"), agg("std"), agg("min"), agg("max"); unit = R.CONCEPTS[tgt].unit
    if mean is None or sd is None or lo is None or hi is None: return None
    label = R.label_of(tgt)
    if sd == 0:
        z = None
        verdict = "at its only recorded value" if v == mean else ("outside its observed range and unusually high" if v > mean else "outside its observed range and unusually low")
    else:
        z = (v - mean) / sd
        verdict = "within its usual range" if abs(z) <= 1.5 else ("unusually high" if z > 0 else "unusually low")
    remember(brain, q)
    return _result(f"{label.capitalize()} is {v:.1f} {unit} now, {verdict} for this vehicle: over the last 14 days the average was {mean:.1f} {unit} (typical spread +/-{sd:.1f}), range {lo:.1f} to {hi:.1f} {unit}. This compares the reading with the vehicle's own recorded history, not with a manufacturer limit.", "normality_check", "ANALYSIS", concept=tgt, z=round(z, 2) if z is not None else None)


def _side_by_side(brain, q, other):
    from . import live_snapshot
    snap = live_snapshot.snapshot(brain.c, brain.now); vals = snap.get("values", {}) if snap.get("available") else {}
    a, b = q.target, other
    if vals.get(a) is None or vals.get(b) is None: return None
    ua, ub = R.CONCEPTS[a].unit, R.CONCEPTS[b].unit; la, lb = R.label_of(a), R.label_of(b)
    diff = f" Difference: {abs(vals[a] - vals[b]):.1f} {ua}." if ua == ub else " They are different quantities (" + ua + " vs " + ub + "), so no difference is computed."
    remember(brain, q)
    return _result(f"From one snapshot at {A.stamp(snap['sample_ts'])}: {la} is {vals[a]:.1f} {ua}; {lb} is {vals[b]:.1f} {ub}.{diff}", "side_by_side", "COMPARE", concepts=[a, b], sample_ts=snap["sample_ts"])


def _evidence(brain, q):
    """The observations behind a reported change: dated daily values across the same window and their difference."""
    tm = q.scope.time
    if tm.type != "relative" or tm.unit not in ("minute", "hour", "day"):
        return None
    seconds = tm.value * {"minute": 60, "hour": 3600, "day": 86400}[tm.unit]
    lo = brain.now - seconds
    st = R.STORE.get(q.target) or {}; day = st.get("day")
    if day:
        table, col, scale = day
        rows = brain.c.execute(f"SELECT ts,{col} FROM {table} WHERE ts>=? AND ts<=? AND {col} IS NOT NULL ORDER BY ts", (lo, brain.now)).fetchall(); pts = [(ts, v / scale) for ts, v in rows]
    else:
        series = A.daily_signal(brain.c, {"battery_temperature": "batt_temp"}.get(q.target, q.target), lo, brain.now) or []; pts = [(x.get("ts"), x["v"]) for x in series]
    if len(pts) < 2: return None
    unit = R.CONCEPTS[q.target].unit; first, last = pts[0], pts[-1]
    shown = pts if len(pts) <= 8 else [pts[0], pts[len(pts) // 4], pts[len(pts) // 2], pts[3 * len(pts) // 4], pts[-1]]
    lines = "; ".join(f"{A.stamp(ts) if ts else 'day'}: {v:.2f} {unit}" for ts, v in shown)
    return _result(f"The reported change in {R.label_of(q.target)} over the requested {tm.value:g} {tm.unit}(s) comes from {len(pts)} recorded values. First valid value {first[1]:.2f} {unit} ({A.stamp(first[0])}), last {last[1]:.2f} {unit} ({A.stamp(last[0])}): difference {last[1] - first[1]:+.2f} {unit}. Sample of the observations: {lines}.", "change_evidence", "ANALYSIS", points=len(pts))


def _missing_measure(brain, raw, normalised):
    try:
        t = nl2ir.normalise(raw); S = nl2ir._S(t); groups = [g for _, g in nl2ir._predicates(S, [])]
        if len(groups) != 1 or _mentions(t) or _times(brain, t): return None          # one restriction and nothing else: a fragment (two groups would be a comparison)
        p = nl2ir.parse(raw, now=brain.now, tz=brain.tz, profile=brain.qc().profile, state=None)
        if p.query is not None and (p.query.target or p.query.event is not None): return None
    except Exception: return None
    names = " and ".join(R.PREDICATES[g][0].replace("while ", "").replace("in ", "").replace("on the ", "") if g in R.PREDICATES else g for g in groups)
    return _result(f"Which measure should I look at for {names} - energy consumption, average speed, or distance driven? Nothing earlier in this conversation says what you are comparing.", "missing_measure", "CLARIFY", available=False)
