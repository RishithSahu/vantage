"""The interpretation contract (version 2): provenance for every field that affects execution, a SUPPORT check and a COVERAGE check.

Fields covered: operation; target(s); statistic; primary and secondary period; conditions (named conditions and thresholds: comparator, number, unit, referent); groups and their order; entity selection (entity, ranking metric, order, size); event type, selector and window;
requested unit; references to the previous turn.

Origins
  question    a phrase of this question states it (the phrase is recorded)
  parts       the question describes it in other words (every word of one phrase for it is present; unique best candidate)
  cue         implied by a wording or unit that conventionally names it ("how fast" -> speed, "km/h" -> vehicle speed); the cue is recorded
  prior_turn  the question continues the previous turn (explicit continuation cue, or a bare fragment of slot values) and that turn had it
  default     nothing says it; the engine's documented default for this operation applies (disclosed with the ACTUAL value)
  vague       a vague time word ("lately") justifies exactly the operation's documented default window, nothing broader (disclosed)
  removed     a follow-up deliberately removed it
  ambiguous   more than one reading is defensible; the alternatives are listed and the answer must be disclosed or clarified
  ungrounded  nothing supports it

Two checks
  SUPPORT   every proposed field has an origin other than ungrounded / ambiguous;
  COVERAGE  every explicit constraint of the question (period phrases, number+unit thresholds, named conditions, negations, group names, selection words) is represented by the query, deliberately removed by a follow-up, or reported as unresolved.
The check re-reads the question; it does not trust the parser or the model that produced the query. The lexical vocabulary tables are shared with the parser (a limitation, stated in the report); operations, units, comparators, referents, orders and defaults are checked by their own logic."""
import re
from dataclasses import dataclass, field

from . import nl2ir, signal_registry as R
from .vocab_parts import _tokens, _phrases_of, _described_by_parts, _signal_hits, signal_by_parts, event_by_parts      # noqa: F401  (re-exported for callers)
from .query_ir import TimeSpec

# ---------------------------------------------------------------------------------------------------------------- documented defaults (the same table the parser applies)
def documented_default_period(q, text):
    """The period the engine applies when the question gives none, per operation (mirrors nl2ir)."""
    sel = q.scope.select
    if q.op == "current_value": return "now"
    if sel is not None: return "all"
    if q.op in ("event_context", "explain_event") or (q.op == "list_events" and q.event is not None and (q.event.selector in ("last", "first", "nth", "most_severe") or (q.event.selector == "all" and q.event.n))): return "all"
    if q.op in ("capability", "detect_change"): return "all"
    if q.op in ("explain_change", "trend"): return "last:14:day"
    return "last:7:day"


def label_of_period(text):
    if text in ("all", "now"): return {"all": "the whole recorded history", "now": "the current moment"}[text]
    if text.startswith("cal:"): return text[4:].replace("_", " ")
    if text.startswith("last:"):
        _, n, u = text.split(":"); return f"the last {n} {u}s" if n != "1" else f"the last {u}"
    if text.startswith("date:"): return text[5:]
    return text


OP_CUES = {
    "acceleration_band": r"\b(?:between|from)\b.+\b(?:and|to)\b",
    "count": r"how many|number of|\bcount|how often|how frequently|\btimes\b|\bany\b|was there|were there|\bat all\b|\bever\b|how much .* (?:happen|occur)|^(?:was|were|did|has|have|had) ",
    "list_events": r"\blist|\bshow|display|\bwhich\b|\bwhen\b|\bevery\b|all the|\beach\b|\bfirst\b|last time|\blatest|most recent|\bany\b|what .* (?:warning|alert|fault|code|diagnostic)s?|\b(?:critical|most serious|most severe|worst)\b.*\b(?:diagnostic|fault|code|warning|event)s?",
    "event_context": r"before|ahead of|leading|led (?:up )?to|around|prior to|what (?:was|were) .* (?:doing|going on)|at the (?:time|moment)|when .* (?:happened|occurred|came on|appeared|turned|switched|got)|while|during",
    "explain_event": r"\bwhy\b|what caused|\breason|how come|\bcause",
    "explain_change": r"\bwhy\b|how come|what (?:is |was )?(?:making|caus\w+|explains?|made|changed)|\breason|explain|account for|what happened to|behind",
    "detect_change": r"when did|which day|what day|biggest|largest|sharpest|greatest|\bjump|\bchang|shift|worst|most",
    "trend": r"trend|getting|going (?:up|down)|increas|decreas|rising|falling|drift|over time|week by week|day by day|better or worse|\bworse\b|\bbetter\b|dropping|growing|slipp|improv|declin|losing|creeping|climbing",
    "compare": r"compar|versus|\bvs\b|\bthan\b|differ|\bbetter\b|\bworse\b| or |between|which uses more|against",
    "rank": r"\brank|\btop\b|\bmost\b|\bleast\b|longest|shortest|farthest|furthest|highest|lowest|\bbest\b|\bworst\b|biggest|fastest|slowest|which (?:trip|session|charge)|coldest|hottest|\d+ (?:longest|shortest)|\b(?:last|latest|most recent|previous|first|earliest) (?:trip|journey|drive|charge|charging session|session)\b",
    "extreme": r"highest|lowest|\bmax|\bmin|\bpeak|hottest|coldest|fastest|slowest|strongest|hardest|worst|\bbest\b|\btop\b|\bat its\b|how (?:hot|cold|fast|high|low|warm|hard|strong|big|slow)|biggest|largest|smallest|\bleast\b|\bmost\b|closest",
    "percent_time": r"percent|\bshare\b|fraction|how (?:long|much of|often|much time)|proportion|% of|\bminutes?\b|time (?:above|below|over|under|with)|\bstay|\bspent|how many hours",
    "current_value": r"\bnow\b|current|right now|at the moment|at present|\bwhat is\b|what's|\bwhats\b|how (?:is|are|much|charged|full|hot|warm|cold)|\bis (?:the|my|it)\b|\bare (?:the|my)\b",
    "capability": r"do you|does (?:it|the car|this)|can you|how fast can .{0,24}\b(?:go|travel)\b|capable of|rated|specification|\brecord|\btrack|measure|\blog\b|\bhave\b|know|available|support|\bis there\b|\bare there\b",
    "correlation": r"correlat|relationship|related|track each other|depend|affect|influence|\blink|\bconnect",
    "explain_event_alt": r"",
}
GENERIC_OPS = {"aggregate", "list_events"}                                                                                   # a bare event name conventionally asks to list/show its recorded occurrences

COMP = [(r"(?:greater|more|higher|bigger|larger|faster|quicker|hotter|warmer|harder|stronger) than|above|over|exceed(?:s|ed|ing)?|beyond", "gt"), (r"at least|no less than|minimum of", "ge"), (r"(?:less|lower|smaller|slower|colder|cooler|weaker) than|below|under|beneath", "lt"), (r"at most|no more than|up to|maximum of", "le")]
UNITS = [(r"miles? per hour|mph", "mph"), (r"kilomet(?:re|er)s? per hour|km/h|kmph|kph", "km/h"), (r"degrees? celsius|celsius|°\s*c\b|degrees? c\b|deg c\b|(?<=in )c\b|(?<=to )c\b", "C"), (r"degrees? fahrenheit|fahrenheit|°\s*f\b|degrees? f\b|(?<=in )f\b|(?<=to )f\b", "F"), (r"per ?cent|percent|%", "%"),
         (r"psi", "psi"), (r"bar", "bar"), (r"kpa|kilopascals?", "kPa"), (r"kilowatts?|kw", "kW"), (r"watts?", "W"), (r"volts?", "V"), (r"rpm", "rpm"), (r"l/min|lit(?:re|er)s? per minute", "L/min"), (r"g-?force|\bg\b", "g"), (r"degrees?", "degree")]
UNIT_RX = "|".join(f"(?P<u{i}>{rx})" for i, (rx, _) in enumerate(UNITS))
COMP_RX = "|".join(f"(?P<c{i}>{rx})" for i, (rx, _) in enumerate(COMP))
THRESH_RX = re.compile(r"(?:" + COMP_RX + r")\s+(?P<num>-?\d+(?:\.\d+)?|" + "|".join(sorted(nl2ir.NUMW, key=len, reverse=True)) + r")\s*(?:(?:" + UNIT_RX + r")(?![a-z]))?")
OP_COMPAT = {"gt": ("gt", "ge"), "ge": ("ge", "gt"), "lt": ("lt", "le"), "le": ("le", "lt"), "eq": ("eq",)}
REFERENT = {"outside": {"ambient_temperature"}, "ambient": {"ambient_temperature"}, "weather": {"ambient_temperature"}, "battery": {"battery_temperature", "battery_soc"}, "pack": {"battery_temperature"}, "motor": {"motor_temperature", "motor_power"}, "engine": {"motor_temperature"},
            "cabin": {"cabin_temperature"}, "inside": {"cabin_temperature"}, "interior": {"cabin_temperature"}, "tyre": {"tyre_temperature", "tyre_pressure"}, "tire": {"tyre_temperature", "tyre_pressure"}, "coolant": {"coolant_temperature"},
            "charge": {"battery_soc"}, "charging": {"charging_power"}, "soc": {"battery_soc"}, "hvac": {"hvac_power"}, "climate": {"hvac_power"}, "heater": {"hvac_power"}, "braking": {"braking_intensity", "brake_pressure"}, "brake": {"brake_pressure", "braking_intensity"}}
CANDIDATE_SIGNALS = ("ambient_temperature", "battery_temperature", "motor_temperature", "cabin_temperature", "tyre_temperature", "tyre_pressure", "hvac_power", "charging_power", "motor_power", "battery_soc")
UNIT_CUE_SIGNAL = {"km/h": "speed", "mph": "speed"}                                                                           # a speed unit conventionally names vehicle speed (recorded as a cue)
TARGET_CUES = {"speed": r"how fast|\bspeed|\bfast\b|\bkph|\bmph|km/h", "rated_max_speed": r"how fast can .{0,24}\b(?:go|travel)\b|rated (?:maximum|top) speed|top speed (?:rating|specification)|maximum speed specification", "distance_driven": r"how far|\bdistance|\bkms?\b|kilomet|\bmiles\b|driven", "wh_per_km": r"per ?km|per kilomet|efficien|consumption|economy|economical|thirst|energy (?:use|usage)",
              "energy_used": r"energy|electricity|kwh|kw hours?|kilowatt hours?|juice|power used|burn|consum", "battery_soc": r"charge|battery (?:level|percent|left)|state of charge|\bsoc\b|how full|battery is left|juice left", "drive_mode": r"driving state|drive mode|mode|parked|city|highway", "braking_intensity": r"(?:strongest|hardest|harshest|most intense).*brak|brak.*(?:strongest|hardest|harshest)|how hard .*brak"}
VAGUE = re.compile(r"\b(lately|recently|these days|nowadays|used to|than before|than it was|than it used|a (?:month|week|while|few \w+) back|ago|of late|worse|better)\b")
CONT_RX = re.compile(r"^(?:and |but |so |ok |okay |then |also |now )*(?:what about|how about|and for|and in|and on|and during|and while|and when|same(?: for| but| thing)?|only|just|actually|instead|without|include|now for|and now|and the|and)\b")
REF_RX = re.compile(r"\b(those|them|these|that|it|then)\b")
ORDER_DESC = r"longest|most|highest|biggest|largest|greatest|fastest|hottest|warmest|top|worst|maximum"
ORDER_ASC = r"shortest|least|lowest|smallest|slowest|coldest|coolest|fewest|bottom|minimum|best"
RANK_METRIC = {"distance_driven": r"long|short|distance|far|km|kilomet", "energy_used": r"energy|kwh|power|electricity", "wh_per_km": r"efficien|economical|consum|per ?km|energy per|economy", "avg_speed": r"speed|fast|slow|quick", "ambient_temperature": r"cold|hot|warm|cool|temperature|freezing",
               "recency": r"recent|last|latest|previous|newest|first|oldest", "duration": r"long|short|duration|time|minutes|hours"}


@dataclass
class FieldOrigin:
    name: str
    value: str
    origin: str
    phrase: str = ""
    alternatives: list = field(default_factory=list)
    alt_values: list = field(default_factory=list)         # machine-readable alternatives (signal names) behind the readable ones

    def to_dict(self): return {"field": self.name, "value": self.value, "origin": self.origin, "phrase": self.phrase, "alternatives": list(self.alternatives)}


@dataclass
class Interpretation:
    fields: list = field(default_factory=list)
    ambiguities: list = field(default_factory=list)       # conventions applied ("last week = last 7 days")
    unresolved: list = field(default_factory=list)        # coverage gaps: explicit constraints of the question that the query does not represent
    source: str = "rules"

    def of(self, name): return [f for f in self.fields if f.name == name]
    def ungrounded(self): return [f for f in self.fields if f.origin == "ungrounded"]
    def ambiguous(self): return [f for f in self.fields if f.origin == "ambiguous"]
    def to_dict(self): return {"source": self.source, "fields": [f.to_dict() for f in self.fields], "ambiguities": list(self.ambiguities), "unresolved": list(self.unresolved)}

    def issues(self):
        """Why this query should not be answered as it stands (support failures, ambiguous fields, coverage gaps)."""
        out = [f"{f.name} '{f.value}' is not supported by the question or the conversation" for f in self.ungrounded()]
        out += [f"{f.name} '{f.value}' is ambiguous: {' or '.join(f.alternatives[:4])}" for f in self.ambiguous()] + list(self.unresolved)
        return list(dict.fromkeys(out))

    def disclosures(self):
        """What the person should be told about how the wording was read, with the ACTUAL values selected."""
        out = []
        for f in self.fields:
            if f.origin == "default" and f.name == "time": out.append(f"No period was given, so I used {label_of_period(f.value)}.")
            elif f.origin == "default" and f.name == "statistic": out.append(f"No statistic was named, so I used the {f.value}.")
            elif f.origin == "default" and f.name == "event_selector": out.append(f"No selection was given, so I used {f.value}.")
            elif f.origin == "default" and f.name == "event_window": out.append(f"No window was given, so I looked {f.value} before the event.")
            elif f.origin == "vague" and f.name == "time": out.append(f"'{f.phrase}' was read as {label_of_period(f.value)}.")
            elif f.origin == "cue" and f.name == "condition" and f.phrase in ("km/h", "mph"): out.append("A speed threshold was read as the vehicle's speed.")
        out += [a[0].upper() + a[1:] + "." for a in self.ambiguities]
        return list(dict.fromkeys(out))

    def clarifications(self):
        """Concrete questions for the ambiguous fields: [{field, question, alternatives}]."""
        qs = []
        for f in self.ambiguous():
            alts = f.alternatives[:4]
            if alts: qs.append({"field": f.name, "alternatives": alts, "question": f"Which {f.name.replace('_', ' ')} do you mean - " + ", ".join(alts[:-1]) + (" or " if len(alts) > 1 else "") + alts[-1] + "?"})
        for u in self.unresolved: qs.append({"field": "coverage", "alternatives": [], "question": u})
        return qs


AMBIGUOUS_PERIODS = [(r"\b(?:the )?(?:last|past|previous) week\b(?! ?\d)", "'last week' can mean the previous calendar week or the last 7 days; it was read as the last 7 days"),
                     (r"\b(?:the )?(?:last|past|previous) month\b(?! ?\d)", "'last month' can mean the previous calendar month or the last 30 days; it was read as the last 30 days")]


def thresholds_units(t): return [u for rx, u in UNITS if re.search(r"(?<![a-z])(?:" + rx + r")(?![a-z])", t)]


def _words(t): return re.findall(r"[a-z0-9']+", t)


def _num(x):
    x = str(x)
    return float(nl2ir.NUMW[x]) if x in nl2ir.NUMW else float(x)


def thresholds(t):
    """Compositional extraction of (comparator, number, unit, span) from the question: each part is recognised separately."""
    out = []
    # A numeric interval is two closed bounds. Keeping it in this independent
    # support checker prevents a correct compiled interval from being rejected
    # merely because it used "between/from" rather than two comparators.
    numrx = r"-?\d+(?:\.\d+)?|" + "|".join(sorted(nl2ir.NUMW, key=len, reverse=True))
    unitrx = r"miles? per hour|mph|kilomet(?:re|er)s? per hour|km/h|kmph|kph"
    mb = re.search(r"\b(?:between|from)\s+(?P<a>" + numrx + r")\s*(?P<ua>" + unitrx + r")?\s+(?:and|to)\s+(?P<b>" + numrx + r")\s*(?P<ub>" + unitrx + r")?\b", t)
    if mb:
        a, b = _num(mb.group("a")), _num(mb.group("b"))
        unit = "mph" if "mph" in ((mb.group("ua") or mb.group("ub") or "").lower()) else "km/h"
        if a > b: a, b = b, a
        out.extend(({"cmp": "ge", "num": a, "unit": unit, "span": (mb.start(), mb.end())}, {"cmp": "le", "num": b, "unit": unit, "span": (mb.start(), mb.end())}))
    for m in THRESH_RX.finditer(t):
        cmp = next(COMP[i][1] for i in range(len(COMP)) if m.group(f"c{i}")); unit = next((UNITS[i][1] for i in range(len(UNITS)) if m.group(f"u{i}")), None)
        out.append({"cmp": cmp, "num": _num(m.group("num")), "unit": unit, "span": (m.start(), m.end())})
    return out


def _to_signal_unit(num, unit, sig):
    """number spoken in `unit` -> the signal's registered unit (or None when the dimensions differ)."""
    reg = R.CONCEPTS[sig].unit
    if unit in (None, "degree"): return num if unit is None or R.unit_family(reg) == "temperature" else None
    if unit == reg: return num
    try: return R.convert(num, unit, reg)
    except Exception: return None


def _near_words(t, span, k=8):
    ws = [(m.start(), m.group(0)) for m in re.finditer(r"[a-z]+", t)]; before = [w for s, w in ws if s < span[0]][-k:]; after = [w for s, w in ws if s >= span[1]][:k]; return before + after


def _period_vs_text(q, text, found, out, fu, prev):
    """Primary and secondary period: support and coverage."""
    spec = q.scope.time.to_text(); texts = [s.to_text() for s in found]; default = documented_default_period(q, text); vague = VAGUE.search(text)
    if texts:
        if spec in texts or (spec == "all" and "all" in texts): out.fields.append(FieldOrigin("time", spec, "question", ", ".join(texts)))
        else: out.fields.append(FieldOrigin("time", spec, "ungrounded", "the question's period is " + ", ".join(texts)))
    elif q.op == "capability": pass
    elif prev is not None and fu and spec == prev.scope.time.to_text(): out.fields.append(FieldOrigin("time", spec, "prior_turn"))
    elif prev is not None and fu and prev.scope.time.type == "calendar" and re.search(r"\b(?:the )?(?:day|week|month) before\b|\b(?:the )?(?:previous|prior|preceding) (?:day|week|month)\b|\bthe one before\b", text) and spec == {"yesterday": "cal:day_before_yesterday", "today": "cal:yesterday", "this_week": "cal:last_week", "this_month": "cal:last_month"}.get(prev.scope.time.name): out.fields.append(FieldOrigin("time", spec, "prior_turn", "the period before the previous one"))
    elif re.search(r"\b(?:this morning|last night|tonight|yesterday|today|this week|this month|weekend|overnight)\b", text) and q.scope.time.type == "calendar": out.fields.append(FieldOrigin("time", spec, "question", "calendar word"))
    elif q.op in ("aggregate", "extreme", "trend") and spec == "last:24:hour" and re.search(r"\b(now|right now|at the moment|currently)\b", text): out.fields.append(FieldOrigin("time", spec, "default", "'now' for a statistic"))
    elif spec == default and vague: out.fields.append(FieldOrigin("time", spec, "vague", vague.group(0)))
    elif spec == default: out.fields.append(FieldOrigin("time", spec, "default"))
    else: out.fields.append(FieldOrigin("time", spec, "ungrounded", f"nothing in the question gives {spec}; the documented default for {q.op} is {default}"))
    for rx, msg in AMBIGUOUS_PERIODS:
        if re.search(rx, text) and q.scope.time.type == "relative" and q.scope.time.unit == "day": out.ambiguities.append(msg)
    # secondary period
    if q.scope_b is not None and q.scope_b.time is not None:
        sb = q.scope_b.time.to_text()
        if sb in texts and (len(texts) >= 2 or sb != spec): out.fields.append(FieldOrigin("time_b", sb, "question", ", ".join(texts)))
        elif re.search(r"\b(?:before|previous|prior|preceding|earlier|than usual|than normal|than before)\b", text) and sb != spec: out.fields.append(FieldOrigin("time_b", sb, "cue", "relative wording"))
        else: out.fields.append(FieldOrigin("time_b", sb, "ungrounded", "no second period in the question"))
        if len(texts) >= 2 and texts[0] == sb and texts[1] == spec and sb != spec: out.unresolved.append("the two periods are mentioned in the opposite order to the comparison the query makes")
    # coverage: every explicit period phrase must be represented
    represented = {spec} | ({q.scope_b.time.to_text()} if (q.scope_b is not None and q.scope_b.time is not None) else set())
    for tx in texts:
        if tx not in represented and not (tx == "now" and q.op == "current_value"): out.unresolved.append(f"the question names the period '{label_of_period(tx)}' but the query does not use it")


def interpret(text, q, prev=None, source="rules", now=None, tz=0):
    """-> Interpretation of query `q` for `text` (support + coverage). `prev` is the previous accepted Query, the only thing a continuation may inherit from."""
    try:
        from . import nlu
        text = nlu.shorthand(text); fixed, fixes = nlu.normalize(str(text), repair_typos=True); text = fixed if fixes else text
    except Exception: pass
    text = re.sub(r"^(?:now|ok|okay|so|then)[, ]+(?=(?:the )?same|for\b|what about|how about|and\b|only\b|just\b)", "", str(text).strip(), flags=re.I)             # a leading discourse word is not a requested period
    t = nl2ir.normalise(text); out = Interpretation(source=source); S = nl2ir._S(t)
    nwords = len(_words(t)); op_cue_anywhere = any(re.search(rx, t) for k, rx in OP_CUES.items() if rx and k not in ("aggregate",))
    cont = bool(CONT_RX.match(t)); fragment = (not op_cue_anywhere) and nwords <= 6 and bool(nl2ir._times(nl2ir._S(t), now, tz) or _signal_hits(t) or nl2ir._predicates(nl2ir._S(t), []) or nl2ir.WHEEL_RX.search(t))
    unit_frag = bool(re.match(r"^(?:and |now |ok )*(?:in|as|using) (?:\w+ ?){1,3}$", t)) and bool(thresholds_units(t))
    ref_word = REF_RX.search(t) is not None and prev is not None and not re.search(r"\bthen\b", t)
    elliptical = prev is not None and not (_signal_hits(t) or [e for _, e in nl2ir._events(nl2ir._S(t)) if e] or nl2ir._predicates(nl2ir._S(t), []) or nl2ir._selection(nl2ir._S(t), t) is not None or event_by_parts(t) or signal_by_parts(t))     # a question that names no signal, event, condition or selection at all must take its subject from somewhere: the previous turn
    fu = prev is not None and (cont or fragment or unit_frag or ref_word or elliptical)
    seen = _signal_hits(t); tok = _tokens(t); tok_free = tok - set().union(*[_tokens(ph) for ph in seen.values()]) if seen else tok
    # ---- operation
    cue = OP_CUES.get(q.op)
    if q.op in GENERIC_OPS or q.op == "none": out.fields.append(FieldOrigin("operation", q.op, "default" if not op_cue_anywhere else "question", ""))
    elif cue and re.search(cue, t): out.fields.append(FieldOrigin("operation", q.op, "question", re.search(cue, t).group(0)))
    elif fu and prev is not None and prev.op == q.op: out.fields.append(FieldOrigin("operation", q.op, "prior_turn"))
    else: out.fields.append(FieldOrigin("operation", q.op, "ungrounded", "no wording of the question supports this operation"))
    # ---- targets
    pt = [x for x in ((prev.target, prev.target2) if prev else ()) if x]
    for slot, n in (("target", q.target), ("target2", q.target2)):
        if not n: continue
        base = re.sub(r"_(fl|fr|rl|rr)$", "", n); mwp = nl2ir.WHEEL_RX.search(t)
        if n in seen: out.fields.append(FieldOrigin(slot, n, "question", seen[n]))
        elif base != n and mwp and nl2ir._wheel_variant(base, (mwp.group(1), mwp.group(2))) == n and (base in seen or any(x.startswith(base + "_") for x in seen) or _described_by_parts(tok, _phrases_of(base)) or _described_by_parts(tok, _phrases_of(n))): out.fields.append(FieldOrigin(slot, n, "question", f"{mwp.group(0)} + signal"))
        elif _described_by_parts(tok_free, _phrases_of(n)): out.fields.append(FieldOrigin(slot, n, "parts", _described_by_parts(tok_free, _phrases_of(n))))
        elif n in TARGET_CUES and re.search(TARGET_CUES[n], t) and not (seen and n in ("speed",) and False): out.fields.append(FieldOrigin(slot, n, "cue", re.search(TARGET_CUES[n], t).group(0)))
        elif n in pt and fu: out.fields.append(FieldOrigin(slot, n, "prior_turn"))
        elif fu and prev is not None and (tok & _tokens(R.label_of(n))): out.fields.append(FieldOrigin(slot, n, "prior_turn", "part of the name"))
        else: out.fields.append(FieldOrigin(slot, n, "ungrounded"))
    # wheel / component position: a specific sensor must come from a stated position (or the previous turn), a general one must not be made specific and vice versa
    mw = nl2ir.WHEEL_RX.search(t) if hasattr(nl2ir, "WHEEL_RX") else None
    for slot, n in (("target", q.target), ("target2", q.target2)):
        if n and re.search(r"_(fl|fr|rl|rr)$", n):
            ok = mw and (nl2ir._wheel_variant(n, (mw.group(1), mw.group(2))) == n)
            if not ok and not (fu and n in pt): out.fields.append(FieldOrigin("component_position", n, "ungrounded", "no wheel position in the question"))
        elif n and mw and nl2ir._wheel_variant(n, (mw.group(1), mw.group(2))) and nl2ir._wheel_variant(n, (mw.group(1), mw.group(2))) in R.CONCEPTS:
            out.unresolved.append(f"the question names the {mw.group(1)} {mw.group(2)} position but the query uses the general signal {n}")
    # ---- statistic
    if q.op in ("aggregate", "extreme") and q.agg:
        words = [a for rx, a in nl2ir.AGG_WORDS if re.search(rx, t)]
        extreme_phr = re.search(r"\b(how (?:fast|hot|cold|high|low|warm|hard|strong)|fastest|slowest|worst|best|strongest|hardest|harshest|closest|lowest|least|highest|most|at its (?:worst|best|peak))\b", t)
        if q.agg in words or (q.agg in ("max", "min") and q.op == "extreme" and extreme_phr): out.fields.append(FieldOrigin("statistic", q.agg, "question", ",".join(words) or (extreme_phr.group(0) if extreme_phr else "")))
        elif q.agg == "std" and re.search(r"\bvariab\w*|\bsteady\b|\bconsistent\b|\bstable\b", t): out.fields.append(FieldOrigin("statistic", q.agg, "question", "variability"))
        elif q.agg == "p95" and re.search(r"\b95(?:th)?\b", t): out.fields.append(FieldOrigin("statistic", q.agg, "question", "95th"))
        elif q.agg == "delta" and re.search(r"\b(fall|fell|loose|lose|lost|drop|rise|rose|gain|change|differ\w*)\b", t): out.fields.append(FieldOrigin("statistic", q.agg, "question", "change"))
        elif prev is not None and prev.agg == q.agg and fu: out.fields.append(FieldOrigin("statistic", q.agg, "prior_turn"))
        elif q.agg in ("mean", "sum") and not words and not extreme_phr: out.fields.append(FieldOrigin("statistic", {"mean": "average", "sum": "total"}[q.agg], "default"))
        else: out.fields.append(FieldOrigin("statistic", q.agg, "ungrounded", ",".join(words)))
    # ---- periods
    found = nl2ir._times(nl2ir._S(t), now, tz)
    _period_vs_text(q, t, found, out, fu, prev)
    # ---- conditions: thresholds (compositional) and named conditions
    trip = thresholds(t); used_trip = set(); pconds = {c.to_text() for sc in ((prev.scope,) if prev else ()) for c in sc.conditions}
    for scn, sc in (("scope", q.scope), ("scope_b", q.scope_b)):
        for c in (sc.conditions if sc is not None else []):
            name = c.to_text()
            if c.predicate: continue
            if not c.signal: continue
            if name in pconds and fu and not trip: out.fields.append(FieldOrigin("condition", name, "prior_turn")); continue
            best = None
            for i, tr in enumerate(trip):
                if tr["cmp"] not in OP_COMPAT.get(c.op, ()) and not (c.op in ("gt", "ge") and tr["cmp"] in ("gt", "ge")) and not (c.op in ("lt", "le") and tr["cmp"] in ("lt", "le")): continue
                spoken = c.unit if c.unit and not R.unit_family(c.unit) is None else None; v_reg = c.value
                cands = [_to_signal_unit(tr["num"], tr["unit"], c.signal)]
                if c.unit and tr["unit"] in (None, c.unit): cands.append(tr["num"])
                if any(x is not None and (abs(x - c.value) <= max(0.01, abs(x) * 0.006) or (c.unit and abs(tr["num"] - c.value) <= 0.01)) for x in cands): best = (i, tr); break
            if best is None:
                out.fields.append(FieldOrigin("condition", name, "ungrounded", "no comparator + number + unit in the question gives this threshold")); continue
            i, tr = best; used_trip.add(i); near = _near_words(t, tr["span"]); dim = R.unit_family(R.CONCEPTS[c.signal].unit); same_dim = [n for n, cc in R.CONCEPTS.items() if R.unit_family(cc.unit) == dim and cc.dtype != "categorical"]
            if tr["unit"] not in (None, "degree") and R.unit_family(tr["unit"]) != dim: out.fields.append(FieldOrigin("condition", name, "ungrounded", f"the unit {tr['unit']} is not a {dim} unit")); continue
            refs = set().union(*[REFERENT[w] for w in near + _words(t) if w in REFERENT]) if any(w in REFERENT for w in near + _words(t)) else set(); ref = next(iter(refs), None)
            if c.signal in seen: out.fields.append(FieldOrigin("condition", name, "question", f"{tr['cmp']} {tr['num']:g} {tr['unit'] or ''}".strip()))
            elif c.signal in refs: out.fields.append(FieldOrigin("condition", name, "question", "referent word"))
            elif tr["unit"] in UNIT_CUE_SIGNAL and UNIT_CUE_SIGNAL[tr["unit"]] == c.signal: out.fields.append(FieldOrigin("condition", name, "cue", tr["unit"]))
            elif fu and prev is not None and (c.signal in pt or c.signal in {x.signal for sc_ in (prev.scope, prev.scope_b) if sc_ is not None for x in sc_.conditions if x.signal}): out.fields.append(FieldOrigin("condition", name, "prior_turn", "signal of the previous turn"))
            elif refs and c.signal not in refs: out.fields.append(FieldOrigin("condition", name, "ungrounded", f"the question's referent is {ref}, not {c.signal}"))
            else: out.fields.append(FieldOrigin("condition", name, "ambiguous", "the unit gives a dimension, not the sensor", [R.label_of(n) for n in same_dim if n in CANDIDATE_SIGNALS][:4], [n for n in same_dim if n in CANDIDATE_SIGNALS][:4]))
    for i, tr in enumerate(trip):                                                                      # coverage: every stated threshold must be represented
        if i not in used_trip and not any(c.signal and abs(tr["num"] - (c.value or 0)) < 1e-9 for sc in (q.scope, q.scope_b) if sc is not None for c in sc.conditions):
            out.unresolved.append(f"the question states the threshold '{tr['cmp']} {tr['num']:g}{(' ' + tr['unit']) if tr['unit'] else ''}' but the query has no such condition")
    # named conditions: support, coverage, negation
    pred_hits = {p for _, p in nl2ir._predicates(nl2ir._S(t), [])}
    neg = {m.group(2) for m in re.finditer(r"\b(without|not|except|excluding|ignoring|ignore|other than|no longer|forget|drop|remove|skip|leave out|disregard|never mind)\s+(?:the |any |a )?([a-z]+)", t)}
    q_preds = {c.predicate for sc in (q.scope, q.scope_b) if sc is not None for c in sc.conditions if c.predicate} | set(q.groups) | {sc.select.klass for sc in (q.scope, q.scope_b) if sc is not None and sc.select is not None and sc.select.klass}
    for sc in (q.scope, q.scope_b):
        for c in (sc.conditions if sc is not None else []):
            if not c.predicate: continue
            name = c.to_text()
            if c.predicate in pred_hits or _tokens(c.predicate.replace("_", " ")) <= tok: out.fields.append(FieldOrigin("condition", name, "question", c.predicate))
            elif name in pconds and fu: out.fields.append(FieldOrigin("condition", name, "prior_turn"))
            else: out.fields.append(FieldOrigin("condition", name, "ungrounded"))
            if c.op != "not" and (any(w in neg for w in re.findall(r"[a-z]+", c.predicate.replace("_", " "))) or any(nl2ir._PRED_PHRASES.get(w) == c.predicate for w in neg) or (c.predicate == "raining" and "rain" in neg)): out.fields.append(FieldOrigin("condition", name, "ungrounded", "the question negates this condition"))
    if re.search(r"charg\w* sessions?", t): pred_hits = pred_hits - {"charging"}                                      # "charging session" names the entity, not a condition
    for p in pred_hits - {"driving"}:                                                                  # coverage of stated conditions (the generic word 'driving' is excluded)
        if q.event is not None and R.EVENT_AS_PREDICATE.get(q.event.type) == p: continue                                      # the event itself is the named thing ("hard brakes" is counted, not a condition)
        if p not in q_preds and not any(w in neg for w in p.replace("_", " ").split()) and not any(nl2ir._PRED_PHRASES.get(w) == p for w in neg): out.unresolved.append(f"the question names the condition '{p.replace('_', ' ')}' but the query does not apply it")
    # ---- groups and their order
    if q.groups:
        pos = {}
        for g in q.groups:
            m = None
            for ph, name in nl2ir._PRED_PHRASES.items():
                if name == g:
                    mm = re.search(r"\b" + re.escape(ph) + r"\b", t)
                    if mm and (m is None or mm.start() < m): m = mm.start()
            pos[g] = m
        for g in q.groups: out.fields.append(FieldOrigin("group", g, "question" if pos[g] is not None else ("parts" if _tokens(g.replace("_", " ")) <= tok else "ungrounded")))
        ps = [pos[g] for g in q.groups if pos[g] is not None]
        if len(ps) == len(q.groups) and ps != sorted(ps): out.unresolved.append("the groups are mentioned in a different order than the comparison the query makes")
    # ---- selection
    sel = q.scope.select
    if sel is not None:
        ent = {"trip": r"\btrips?\b|journeys?|drives?", "charge_session": r"charg\w+ sessions?|charges?\b|sessions?"}.get(sel.entity)
        out.fields.append(FieldOrigin("selection_entity", sel.entity, "question" if (ent and re.search(ent, t)) else ("prior_turn" if fu and prev is not None and prev.scope.select is not None and prev.scope.select.entity == sel.entity else "ungrounded")))
        if sel.rank_by:
            rx = RANK_METRIC.get(sel.rank_by); out.fields.append(FieldOrigin("selection_metric", sel.rank_by, "question" if (rx and re.search(rx, t)) else ("prior_turn" if fu and prev is not None and prev.scope.select is not None and prev.scope.select.rank_by == sel.rank_by else "ungrounded")))
            eff = re.search(r"efficien|economical", t) and sel.rank_by == "wh_per_km"
            d, a = re.search(r"\b(" + ORDER_DESC + r")\b", t), re.search(r"\b(" + ORDER_ASC + r")\b", t)
            want = None
            if eff: want = "asc" if re.search(r"\bmost\b|\bbest\b", t) else "desc" if re.search(r"\bleast\b|\bworst\b", t) else None
            elif d and not a: want = "desc"
            elif a and not d: want = "asc"
            elif d and a: want = None
            if want and sel.order != want: out.fields.append(FieldOrigin("selection_order", sel.order, "ungrounded", f"the wording asks for {want}ending order"))
            elif want: out.fields.append(FieldOrigin("selection_order", sel.order, "question", (d or a).group(0) if (d or a) else "efficiency wording"))
        if sel.k:
            nums = {float(x) for x in re.findall(r"\d+", t)} | {float(nl2ir.NUMW[w]) for w in _words(t) if w in nl2ir.NUMW}
            if float(sel.k) in nums or (sel.k == 1 and re.search(r"\b(the|a|which|what|longest|shortest|best|worst|most|least|highest|lowest|last|latest|first|previous|recent|fastest|coldest|hottest)\b", t)): out.fields.append(FieldOrigin("selection_size", str(sel.k), "question"))
            elif fu and prev is not None and prev.scope.select is not None and prev.scope.select.k == sel.k: out.fields.append(FieldOrigin("selection_size", str(sel.k), "prior_turn"))
            elif sel.k == 3 and q.op == "rank": out.fields.append(FieldOrigin("selection_size", str(sel.k), "default", "3"))
            else: out.fields.append(FieldOrigin("selection_size", str(sel.k), "ungrounded"))
    # ---- event: type, selector, window
    if q.event is not None:
        ev = {e for _, e in nl2ir._events(nl2ir._S(t)) if e}; evn = q.event.type; phr = [ph for ph, e in nl2ir._EVENT_PHRASES.items() if e == evn] + [evn.replace("_", " ").lower()]
        if evn in ev: out.fields.append(FieldOrigin("event", evn, "question"))
        elif evn == "ANY_WARNING" and re.search(r"\b(warning|alert|alarm|fault|problem|issue)s?\b", t): out.fields.append(FieldOrigin("event", evn, "question", "warning wording"))
        elif _described_by_parts(tok_free, phr): out.fields.append(FieldOrigin("event", evn, "parts", _described_by_parts(tok_free, phr)))
        elif prev is not None and prev.event is not None and prev.event.type == evn and fu: out.fields.append(FieldOrigin("event", evn, "prior_turn"))
        else: out.fields.append(FieldOrigin("event", evn, "ungrounded"))
        selw = {"last": r"\blast\b|latest|most recent|previous", "first": r"\bfirst\b|earliest", "all": r"\ball\b|every|\blist\b|\bshow\b|\bdisplay\b|each|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty) (?:latest|most recent|newest|last|recent)\b", "nth": r"\b\d+(?:st|nd|rd|th)\b|second|third", "most_severe": r"\bcritical\b|most serious|most severe|\bworst\b"}
        if q.op == "list_events":
            if re.search(selw.get(q.event.selector, r"$^"), t): out.fields.append(FieldOrigin("event_selector", q.event.selector, "question"))
            elif q.event.selector == "last": out.fields.append(FieldOrigin("event_selector", "the most recent occurrence", "default"))
            elif prev is not None and prev.event is not None and prev.event.selector == q.event.selector and fu: out.fields.append(FieldOrigin("event_selector", q.event.selector, "prior_turn"))
            else: out.fields.append(FieldOrigin("event_selector", q.event.selector, "ungrounded"))
        if q.op == "event_context" and (q.event.before or q.event.after):
            secs = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}; m = re.search(r"(\d+(?:\.\d+)?|" + "|".join(nl2ir.NUMW) + r")\s*(second|sec|minute|min|hour|hr)s?\b", t)
            if m and abs(_num(m.group(1)) * secs[m.group(2)] - (q.event.before or q.event.after)) < 1: out.fields.append(FieldOrigin("event_window", str(q.event.before or q.event.after), "question", m.group(0)))
            elif not m: out.fields.append(FieldOrigin("event_window", f"{(q.event.before or q.event.after) // 60} minutes", "default"))
            else: out.fields.append(FieldOrigin("event_window", str(q.event.before or q.event.after), "ungrounded", m.group(0)))
    # ---- requested unit
    if q.unit:
        spoken = {u for rx, u in UNITS if re.search(r"(?<![a-z])(?:" + rx + r")(?![a-z])", t)}
        out.fields.append(FieldOrigin("unit", q.unit, "question" if q.unit in spoken or q.unit.lower() in spoken else ("prior_turn" if fu and prev is not None and prev.unit == q.unit else "ungrounded")))
    # ---- references to the previous turn: never inherited silently
    if prev is not None and REF_RX.search(t) and not cont:
        w = REF_RX.search(t).group(1)
        if w == "then" and prev.event is not None and prev.scope.time is not None: out.fields.append(FieldOrigin("reference", "then", "ambiguous", "'then' could mean the time of the event or a comparison period", ["the moment of the previous event", "the period of the previous question"]))
    return out


def summary(it):
    return "; ".join(f"{f.name}={f.value} [{f.origin}]" for f in it.fields) + (" | " + "; ".join(it.ambiguities) if it.ambiguities else "") + (" | UNRESOLVED: " + "; ".join(it.unresolved) if it.unresolved else "")
