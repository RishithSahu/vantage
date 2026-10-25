"""Fast deterministic parser: natural language -> Query IR, with no model involved.

It runs BEFORE any language-model call. It recognises, by longest phrase match against the registry vocabulary, the building blocks of a question (time, signal, aggregation, condition, entity selection, event,
comparison groups), assembles them into a Query, and reports how much of the sentence it actually understood. A parse that explains the whole sentence is trusted and answered immediately (milliseconds);
a parse that leaves content words unexplained is returned with a low confidence so the caller can try the language model or the legacy engine - nothing is guessed silently.

Also here: follow-up questions ("what about last week?", "and for highway?") merged with the previous Query, and clarification questions for genuinely ambiguous wording ("temperature" -> which one?)."""
import datetime as dt
import os
import copy
import re
from dataclasses import dataclass, field

from . import signal_registry as R
from .query_ir import Condition, EntitySelect, EventRef, Query, Scope, TimeSpec

ACCEPT = 0.72                                  # minimum share of content words explained for the parse to be trusted

NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100}
NUM = r"(?:\d+(?:\.\d+)?|" + "|".join(NUMW) + r")"
MONTHS = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
WEEKDAYS = {d: i for i, d in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
STOP = set("""same actually only just instead also again the a an of in on at to for and or is was were are be been being did do does done i me my mine we our you your it its this that these those what whats which how when where who why show tell give get please can could would will
 with by from than then there their had have has having use used using drive drove driven take took taken event events data record recorded happen happened happening occur occurs occurred occurring list ignore between compared compare current currently latest live during while over about any all vehicle car ev much many time times as so just really also up out into per any some there's that's is it's like want need know see look let
if so then too very quite ever again still already yet am pm last latest first
# conversational filler: politeness, hedges and openers carry no query content
would could should might may please kindly mind telling thanks thank check quick one okay ok hey hi hello assistant sorry bother forget silly probably curious wondering know like um uh well right maybe possibly sure just friend asked couldn answer thinking earlier before immediately directly shortly spin spins spinning spun rotate rotates rotating run runs running display displays present make makes made making recover recovers recovered recovering gone went got risen fallen spend spends spending bhai yaar arre boss bro dude mate silly question probably but
""".split())
CMP = {"above": "gt", "over": "gt", "exceeding": "gt", "exceeded": "gt", "exceeds": "gt", "exceed": "gt", "greater than": "gt", "more than": "gt", "higher than": "gt", "bigger than": "gt", "faster than": "gt", "quicker than": "gt", "slower than": "lt", "hotter than": "gt", "warmer than": "gt", "colder than": "lt", "cooler than": "lt", "below": "lt", "under": "lt",
       "less than": "lt", "lower than": "lt", "beneath": "lt", "smaller than": "lt", "at least": "ge", "no less than": "ge", "at most": "le", "no more than": "le", "up to": "le", "exactly": "eq", "equal to": "eq", ">=": "ge", "<=": "le", ">": "gt", "<": "lt"}
UNIT_ALIASES = {"km/h": "km/h", "kph": "km/h", "kmph": "km/h", "mph": "mph", "%": "%", "c": "C", "f": "F", "kw": "kW", "w": "W", "kwh": "kWh", "wh/km": "Wh/km", "km": "km", "mi": "mi", "miles": "mi", "kpa": "kPa", "psi": "psi", "bar": "bar",
                "v": "V", "a": "A", "nm": "Nm", "rpm": "rpm", "m/s": "m/s", "hp": "hp", "deg": "deg", "m": "m", "min": "min", "l/min": "L/min"}
UNIT_RX = r"(degrees? fahrenheit|degrees? celsius|fahrenheit|celsius|degrees?|km/h|kmph|kph|mph|%|kwh|wh/km|kw|kpa|psi|bar|rpm|m/s|l/min|nm|hp|km|miles|mi|deg|c|f|v|a|w|m)"
HEAD_BY_CONTEXT = {   # bare "temperature" / "power" ... + a context word -> the concept
    "temperature": {"battery": "battery_temperature", "pack": "battery_temperature", "motor": "motor_temperature", "cabin": "cabin_temperature", "inside": "cabin_temperature", "interior": "cabin_temperature", "outside": "ambient_temperature", "ambient": "ambient_temperature",
                    "weather": "ambient_temperature", "external": "ambient_temperature", "tyre": "tyre_temperature", "tire": "tyre_temperature", "gearbox": "transmission_temperature", "transmission": "transmission_temperature", "cell": "cell_temperature_max"},
    "power": {"battery": "battery_power", "pack": "battery_power", "motor": "motor_power", "charging": "charging_power", "charger": "charging_power", "climate": "hvac_power", "hvac": "hvac_power", "regen": "regen_power"},
    "voltage": {"battery": "battery_voltage", "pack": "battery_voltage", "12v": "aux_battery_voltage", "auxiliary": "aux_battery_voltage", "cell": "cell_voltage_min"},
    "current": {"battery": "battery_current", "pack": "battery_current"}, "pressure": {"tyre": "tyre_pressure", "tire": "tyre_pressure", "brake": "brake_pressure", "braking": "brake_pressure"},
    "health": {"battery": "battery_soh", "pack": "battery_soh"}, "temp": None, "flow": {"coolant": "coolant_flow", "cooling": "coolant_flow"},
    "speed": {"wheel": "wheel_speed_fl", "motor": "motor_rpm", "charging": "charging_power", "charger": "charging_power"}}
HEAD_BY_CONTEXT["temp"] = HEAD_BY_CONTEXT["temperature"]
BARE_OPTIONS = {"temperature": ["battery", "motor", "cabin", "outside"], "temp": ["battery", "motor", "cabin", "outside"], "power": ["battery", "motor", "charging", "climate-control"], "voltage": ["battery", "12 V battery", "cell"], "pressure": ["tyre", "brake"]}
EXTENSIVE_DEFAULT = {"energy_used": "sum", "distance_driven": "sum", "regen_energy": "sum"}
LEGACY_TARGETS = {"battery_temperature", "battery_soc", "speed", "ambient_temperature", "hvac_power", "coolant_flow", "energy_used", "wh_per_km", "range_estimate", "tyre_pressure"}
RANK_HINTS = [   # (regex over the whole sentence, entity metric, order)
    (r"\b(?:least|less) (?:efficient|economical|frugal)\b|\bworst(?: efficiency| consumption| economy)?\b|\bmost (?:wasteful|energy[- ]hungry|power[- ]hungry|consuming)\b|\b(?:highest|worst|biggest) (?:consumption|energy consumption|wh/km|energy use)\b", "wh_per_km", "desc"),
    (r"\bmost (?:efficient|economical|frugal)\b|\bbest(?: efficiency| consumption| economy)?\b|\b(?:lowest|best|smallest) (?:consumption|energy consumption|wh/km|energy use)\b", "wh_per_km", "asc"),
    (r"\blongest\b|\bfarthest\b|\bfurthest\b|\b(?:most|highest|largest) (?:distance|km|kilometres|kilometers)\b|\bcovered the most\b", "distance_driven", "desc"),
    (r"\bshortest\b|\b(?:least|lowest|smallest) (?:distance|km)\b", "distance_driven", "asc"),
    (r"\bfastest\b|\b(?:highest|top) average speed\b", "avg_speed", "desc"), (r"\bslowest\b|\blowest average speed\b", "avg_speed", "asc"),
    (r"\b(?:used|consumed|took|drew|burned|burnt|had) the most (?:energy|power|electricity|kwh)\b|\bmost (?:energy|electricity|kwh)\b|\bhighest energy\b", "energy_used", "desc"),
    (r"\b(?:used|consumed|took) the least (?:energy|power|electricity|kwh)\b|\bleast (?:energy|electricity|kwh)\b|\blowest energy\b", "energy_used", "asc"),
    (r"\bmost (?:climate|hvac|heater|heating|ac) (?:power|energy|use)\b", "hvac_power", "desc"),
]
ENTITY_RX = re.compile(r"\b(trips?|drives?|journeys?|rides?|charging sessions?|charge sessions?|charge cycles?|charges|(?:(?<=\bmy )|(?<=\bthe )|(?<=\blast )|(?<=\bthat )|(?<=\bthis )|(?<=\blatest )|(?<=\bprevious )|(?<=\bfirst )|(?<=\ba ))charge(?! (?:level|percent\w*|state|status|power|rate|current|port|cable|limit|speed|curve|point|station|time|number)))\b")      # "my last charge" is a session; "state of charge" and "charge level" are not
CLASS_WORDS = {"highway": "highway", "motorway": "highway", "freeway": "highway", "city": "city", "urban": "city", "rainy": "rainy", "wet": "rainy", "long": "long", "short": "short"}
NOUN_FOR_CHARGE = ("charging session", "charge session", "charge cycle", "charges", "charging sessions", "charge sessions", "charge cycles")


def num(x):
    x = str(x); return NUMW[x] if x in NUMW else (int(x) if re.fullmatch(r"\d+", x) else float(x))


@dataclass
class Parse:
    query: Query = None
    confidence: float = 0.0
    clarify: str = ""                           # a question to ask the person instead of answering
    options: list = field(default_factory=list)
    pending: Query = None                       # the query waiting for the answer to the clarification
    unexplained: list = field(default_factory=list)
    followup: bool = False
    reason: str = ""
    ms: float = 0.0

    @property
    def ok(self): return self.query is not None and self.confidence >= ACCEPT and not self.clarify


@dataclass
class ConvState:
    """What the conversation remembers between questions (follow-ups, clarifications). Plain data; owned by the caller."""
    last: Query = None
    pending: Query = None
    options: list = field(default_factory=list)
    slot: str = ""
    choices: list = field(default_factory=list)          # readings offered by a clarification (Query objects); a reply such as "2" or "the first" selects one


# ------------------------------------------------------------------------------------------------ vocabulary tables
def _flex(p): return re.escape(p).replace(r"\ ", r"[ -]").replace("\\-", "[ -]?")


def _alt(phrases): return "|".join(_flex(p) for p in sorted(set(phrases), key=lambda x: (-len(x), x)))


_CONCEPT_PHRASES = sorted(R.LEXICON, key=lambda p: (-len(p), p))
_CONCEPT_RX = re.compile(r"(?<![\w/])(" + _alt(_CONCEPT_PHRASES) + r")(?:'s|s|es)?(?![\w/])")
_EVENT_PHRASES = {}
for _k, (_l, _syn) in R.EVENTS.items():
    for _p in _syn: _EVENT_PHRASES.setdefault(_p, _k)
_EVENT_RX = re.compile(r"(?<![\w/])(" + _alt(_EVENT_PHRASES) + r")(?:s)?(?![\w/])")
_PRED_PHRASES = {}
for _k, _syn in R.PRED_SYN.items():
    for _p in _syn: _PRED_PHRASES.setdefault(_p, _k)
_PRED_RX = re.compile(r"(?<![\w/])(?:(?:while|when|during|in|on|at|under|whilst|whenever|with)\s+(?:the\s+|my\s+|a\s+)?)?(" + _alt(_PRED_PHRASES) + r")(?![\w/])")
_CALENDAR = [(r"day before yesterday", "day_before_yesterday"), (r"yesterday", "yesterday"), (r"today|so far today", "today"), (r"this week|so far this week", "this_week"), (r"last week|previous week", "last_week"), (r"this month|so far this month", "this_month"),
             (r"last month|previous month", "last_month"), (r"last weekend|the weekend", "last_weekend"), (r"this morning", "this_morning"), (r"last night|overnight|tonight", "last_night")]


def normalise(text):
    t = " " + str(text).lower().replace("’", "'").replace("`", "'") + " "
    # Compact durations are common in typed and dictated requests.  They are
    # the same temporal structure as their spaced forms, not a new wording.
    t = re.sub(r"\b(\d+(?:\.\d+)?)(secs?|mins?|hrs?|hours?|days?|weeks?|months?)\b", r"\1 \2", t)
    t = re.sub(r"\b(what|how|where|when|who|that|it|there)'s\b", r"\1 is", t)                    # contractions are one closed class
    t = re.sub(r"\b(?:over |during |in |for |across )?the same (\d+|one|two|three|four|five|six|seven|ten|fourteen|thirty) (days?|weeks?|months?)\b", r" over the last \1 \2 ", t)      # "the same 14 days": the stated span, anchored on now
    t = re.sub(r"\bhow come\b", "why", t)                                                                      # "how come X is zero" asks for the reason, exactly as "why" does
    t = re.sub(r"°\s*([cf])\b|degrees?\s+(?:celsius|centigrade|c)\b|\bcelsius\b", lambda m: " c " if (m.group(1) or "c") == "c" else " f ", t)
    t = re.sub(r"degrees?\s+(?:fahrenheit|f)\b|\bfahrenheit\b", " f ", t)
    t = re.sub(r"\b(?:km ?/ ?h|kmph|kph|km per hour|kilomet(?:re|er)s? per hour)\b", "km/h", t)
    t = re.sub(r"\b(?:miles per hour|mi ?/ ?h)\b", "mph", t)
    t = re.sub(r"\bbreak (?=(?:pads?|fluid|wear)\b)", "brake ", t)
    t = re.sub(r"\bnewton[- ]?met(?:re|er)s?\b", "nm", t); t = re.sub(r"\b(?:revs|revolutions|rotations) per minute\b", "rpm", t); t = re.sub(r"\b(?:litres|liters) per minute\b", "l/min", t)
    t = re.sub(r"\bkilowatts?\b", "kw", t); t = re.sub(r"\bwatts?\b", "w", t); t = re.sub(r"\bvolts?\b", "v", t); t = re.sub(r"\bkilopascals?\b", "kpa", t)
    t = re.sub(r"\b(?:per ?cent|percent|percentage)\b(?! of)", "%", t)
    t = re.sub(r"(\d)\s*%", r"\1 %", t); t = re.sub(r"(\d)(km/h|mph|kw|kwh|kpa|psi|bar|rpm|c|f|v|km)\b", r"\1 \2", t)
    comp = r"(battery|pack|motor|coolant|inverter|charger|cabin)(?: pack)?"
    t = re.sub(r"\b" + comp + r" (hottest|warmest)\b", r" \1 temperature highest ", t)                  # a superlative adjective after a component names that component's temperature
    t = re.sub(r"\b" + comp + r" (coldest|coolest)\b", r" \1 temperature lowest ", t)
    t = re.sub(r"\bhow (hot|warm|cold|cool) (did|does|do|was|is|has|have|will) (the |my )?" + comp + r"\b(?= (?:get|got|go|gets|reach\w*|become|run|climb\w*|was|were|yesterday|today|last|this|over|in|during|on)\b| ?$)", lambda m: "how " + m.group(1) + " " + m.group(2) + " " + (m.group(3) or "") + m.group(4) + " temperature", t)
    t = re.sub(r"\b(\d{1,2}) ?(?:st|nd|rd|th) percentile\b", r" p\1 ", t)
    t = re.sub(r"\b(?:how much )?(?:energy|electricity|power|battery energy|juice)(?: (?:did|do|does|will|would|have|has) (?:i|we|the car|my car|it))?(?: (?:use|used|consume|consumed|spend|spent|burn|burned|draw|drew))? per (?:km|kilomet(?:re|er))\b", " wh per km ", t)
    t = re.sub(r"\b\d{1,2}(?::\d{2})? ?(?:am|pm)\b|\b\d{1,2}:\d{2}\b", " clocktime ", t)
    t = re.sub(r"\b(\d+) ?(?:st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"[?!,;:\"()\[\]{}]", " ", t); t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)
    t = re.sub(r"\b(?:soc)\b", "soc", t)
    t = re.sub(r"\b(temperature|temp|power|voltage|pressure|health|current) of (?:the |my )?([a-z0-9]+(?: [a-z0-9]+)?)\b(?! (?:was|is|were|at|in|during|over|for|last))", lambda m: m.group(2) + " " + m.group(1), t)
    t = re.sub(r"\b(battery|motor|cabin|tyre|tire|coolant)'s\b", r"\1", t)
    t = re.sub(r"\b(?:vs\.?|versus|against)\b", " vs ", t)
    return " ".join(t.split())


class _S:
    """The sentence plus a mask of which characters have been explained."""
    def __init__(self, t): self.t = t; self.used = [False] * len(t)

    def free(self, a, b): return not any(self.used[a:b])

    def take(self, a, b):
        for i in range(a, b): self.used[i] = True

    def find(self, rx, take=True, flags=0):
        for m in re.finditer(rx, self.t, flags) if isinstance(rx, str) else rx.finditer(self.t):
            if self.free(m.start(), m.end()):
                if take: self.take(m.start(), m.end())
                return m
        return None

    def findall(self, rx, take=True):
        out = []
        for m in (re.finditer(rx, self.t) if isinstance(rx, str) else rx.finditer(self.t)):
            if self.free(m.start(), m.end()):
                if take: self.take(m.start(), m.end())
                out.append(m)
            else:                                                                  # the match may be free except for its leading preposition ("plugged in yesterday": "in" belongs to "plugged in")
                pm = re.match(r"(?:on|for|during|in|over|within|across) ", m.group(0))
                if pm and self.free(m.start() + pm.end(), m.end()):
                    sub = (re.compile(rx) if isinstance(rx, str) else rx).match(self.t, m.start() + pm.end())
                    if sub is not None and sub.end() == m.end():
                        if take: self.take(sub.start(), sub.end())
                        out.append(sub)
        return out

    def leftover(self):
        words = []
        for m in re.finditer(r"[a-z0-9'/%.\-]+", self.t):
            if not any(self.used[m.start():m.end()]):
                w = m.group(0).strip("'.-")
                if w and w not in STOP and not re.fullmatch(r"\d+(?:\.\d+)?", w) and not re.fullmatch(r"[a-z]+(?:n't|'(?:s|re|ve|ll|d|m))", w): words.append(w)   # an English contraction is a function word
        return words

    def content_words(self):
        return [m.group(0) for m in re.finditer(r"[a-z0-9'/%.\-]+", self.t) if m.group(0).strip("'.-") and m.group(0).strip("'.-") not in STOP and not re.fullmatch(r"\d+(?:\.\d+)?", m.group(0))]


# ------------------------------------------------------------------------------------------------ extraction of the building blocks
def _local(now, tz):
    now = (now or 0) - 1 if now and ((now + (tz or 0)) % 86400 == 0) else (now or 0)                       # a history that ends exactly at midnight ends the day before
    return dt.datetime.fromtimestamp(now + (tz or 0), dt.timezone.utc)


def _times(S, now, tz):
    """-> list of TimeSpec in the order mentioned."""
    found = []
    def add(m, spec): found.append((m.start(), spec))
    for m in S.findall(r"(?:in |over |during |for |within |across |throughout )?(?:the |these |those )?(?:last|past|previous) (" + NUM + r") (minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b"):
        u = m.group(2)[0:3]; unit = {"min": "minute", "hou": "hour", "hrs": "hour", "day": "day", "wee": "week", "mon": "month"}[u]; v = int(num(m.group(1)))
        add(m, TimeSpec("relative", v, unit))
    for m in S.findall(r"(?<![\w])(?:over |in |during |for |within )?(?:the )?(?:past |last )?(fortnight|couple of days|couple of weeks)\b"):                       # idiomatic spans
        add(m, TimeSpec("relative", 14, "day") if m.group(1) == "fortnight" else TimeSpec("relative", 2, "day") if m.group(1).endswith("days") else TimeSpec("relative", 14, "day"))

    for m in S.findall(r"(?<![\w])(?:over|across|within|throughout) (?:the )?(" + NUM + r") (minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b"):         # "over 30 days" is a span of time, not a threshold
        u = m.group(2)[0:3]; unit = {"min": "minute", "hou": "hour", "hrs": "hour", "day": "day", "wee": "week", "mon": "month"}[u]; add(m, TimeSpec("relative", int(num(m.group(1))), unit))
    for m in S.findall(r"(?:in |over |during |for |within )?(?:the )?(?:last|past) (hour|day|week|month|24 hours)\b"):
        w = m.group(1); bare = "last" in m.group(0).split() and "the" not in m.group(0).split() and not re.match(r"(?:in|over|during|within)\b", m.group(0))              # "last week" = the week before this one; "in the last week", "the past week" = the 7 days up to now
        add(m, TimeSpec("calendar", name="last_week") if (w == "week" and bare) else TimeSpec("calendar", name="last_month") if (w == "month" and bare) else
                            TimeSpec("relative", 7, "day") if w == "week" else TimeSpec("relative", 30, "day") if w == "month" else TimeSpec("relative", 24, "hour") if w == "24 hours" else TimeSpec("relative", 1, w))
    for pat, name in _CALENDAR:
        for m in S.findall(r"(?<![\w])(?:on |for |during |in |over )?(?:" + pat + r")(?![\w])"): add(m, TimeSpec("calendar", name=name))
    for m in S.findall(r"(?<![\w])(?:till|until|up to|up till|as of|thus far|to)\s+(?:now|date|today|this day)(?![\w])|(?<![\w])to date(?![\w])"): add(m, TimeSpec("all"))
    for m in S.findall(r"(?<![\w])(?:on |for |in |over )?(?:all time|ever|overall|so far|altogether|since (?:the )?(?:start|beginning)|(?:whole|entire|full) (?:history|record)|lifetime|since i (?:got|bought) it)(?![\w])"): add(m, TimeSpec("all"))
    for m in S.findall(r"(?<![\w])(?:on |for |in )?(\d{1,2}) (?:of )?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?: (\d{4}))?(?![\w])"): add(m, _date(int(m.group(1)), MONTHS[m.group(2)], m.group(3), now, tz))
    for m in S.findall(r"(?<![\w])(?:on |for |in )?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{1,2})(?: (\d{4}))?(?![\w])"): add(m, _date(int(m.group(2)), MONTHS[m.group(1)], m.group(3), now, tz))
    for m in S.findall(r"(?<![\w])(\d{4})-(\d{2})-(\d{2})(?![\w])"): add(m, TimeSpec("date", date=f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    nth_ago = []
    for m in S.findall(r"(?<![\w])(" + NUM + r"|a|an) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)s? (?:ago|back)(?![\w])"):
        if now is None: continue
        n = 1 if m.group(1) in ("a", "an") else max(1, int(num(m.group(1)))); today = _local(now, tz).date(); back = (today.weekday() - WEEKDAYS[m.group(2)]) % 7 or 7
        add(m, TimeSpec("date", date=(today - dt.timedelta(days=back + 7 * (n - 1))).isoformat())); nth_ago.append((m.start(), m.end()))
    for m in S.findall(r"(?<![\w])(?:on |last |this |for )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?![\w])"):
        if now is not None and not any(a <= m.start() < b for a, b in nth_ago):
            today = _local(now, tz).date(); back = (today.weekday() - WEEKDAYS[m.group(1)]) % 7 or 7
            add(m, TimeSpec("date", date=(today - dt.timedelta(days=back)).isoformat()))
    for m in S.findall(r"(?<![\w])(?:right now|currently|at the moment|at present|just now|now)(?![\w])"):
        if re.search(r"(?:till|until|up to|up till|as of|since|from|before|by|to) $", S.t[:m.start()]) and m.group(0) == "now": continue          # "till now" is a span that ends now, not the moment
        add(m, TimeSpec("current"))
    out = [s for _, s in sorted(found, key=lambda x: x[0])]
    return out


def _date(d, mo, year, now, tz):
    y = int(year) if year else (_local(now, tz).year if now else dt.date.today().year)
    try: date = dt.date(y, mo, d)
    except ValueError: return TimeSpec("date", date="0000-00-00")                      # fails validation -> explained, not guessed
    if not year and now and date > _local(now, tz).date(): date = dt.date(y - 1, mo, d)
    return TimeSpec("date", date=date.isoformat())


TEMP_REFERENT = {"outside": "ambient_temperature", "ambient": "ambient_temperature", "weather": "ambient_temperature", "battery": "battery_temperature", "pack": "battery_temperature", "motor": "motor_temperature", "cabin": "cabin_temperature", "inside": "cabin_temperature", "interior": "cabin_temperature", "tyre": "tyre_temperature", "tire": "tyre_temperature"}


def _to_unit(v, unit_in, concept):
    """A number the person gave, in the unit they gave, -> the registry unit of the concept (or None when the units do not fit)."""
    c = R.CONCEPTS[concept]
    if not unit_in: return float(v), None
    u = UNIT_ALIASES.get(unit_in.lower(), unit_in)
    if u == c.unit: return float(v), None
    if R.unit_family(u) is not None and R.unit_family(u) == R.unit_family(c.unit): return round(R.convert(float(v), u, c.unit), 3), u
    return None, u


def _conditions(S, profile, target_hint=None):
    out = []
    sig_rx = _CONCEPT_RX.pattern
    pat = re.compile(r"(?:(?:when|while|whenever|with|at|if|where|during|and)\s+)?(?:(?P<sig>" + sig_rx[len("(?<![\\w/])("):sig_rx.rindex(")(?:'s|s|es)?")] + r")(?:'s|s|es)?\s+)?(?:(?:was|is|were|are|went|got|goes|gets|reached|hit|ever|stayed|remained)\s+)?"
                     r"(?P<cmp>" + "|".join(sorted(map(re.escape, CMP), key=len, reverse=True)) + r")\s*(?P<num>-?\d+(?:\.\d+)?)(?:\s*(?P<unit>" + UNIT_RX[1:-1] + r")(?![a-z0-9/]))?")
    for m in pat.finditer(S.t):
        if not S.free(m.start(), m.end()): continue
        sig = None
        if m.group("sig"):
            names = R.LEXICON.get(m.group("sig").replace("-", " "), None) or R.LEXICON.get(m.group("sig"), None)
            if names: sig = _pick(sorted(names), profile)
        unit = m.group("unit"); generic_deg = False
        if unit:
            ul = unit.lower()
            if ul in ("degrees", "degree"): unit = None; generic_deg = True
            elif "fahrenheit" in ul: unit = "F"
            elif "celsius" in ul: unit = "C"
        if sig is None and (generic_deg or (unit in ("C", "F"))):    # a temperature unit gives a DIMENSION: the sensor comes from a referent word near the threshold ("outside", "the pack", "cabin")
            ctx = S.t[max(0, m.start() - 45): m.end() + 30]; ref = next((TEMP_REFERENT[w] for w in re.findall(r"[a-z]+", ctx) if w in TEMP_REFERENT), None)
            if ref is not None:
                sig = ref
                for mm in re.finditer(r"[a-z]+", S.t[m.end(): m.end() + 30]):
                    if mm.group(0) in TEMP_REFERENT: S.take(m.end() + mm.start(), m.end() + mm.end()); break
        if sig is None:                                              # "above 100 km/h" -> the unit says which signal
            u = UNIT_ALIASES.get((unit or "").lower())
            by_unit = {"km/h": "speed", "mph": "speed", "kPa": "tyre_pressure", "psi": "tyre_pressure", "kW": target_hint if target_hint and R.CONCEPTS[target_hint].unit == "kW" else None, "rpm": "motor_rpm"}
            sig = by_unit.get(u)
            if sig is None and target_hint and u and R.unit_family(u) == R.unit_family(R.CONCEPTS[target_hint].unit): sig = target_hint
            if sig is None and u == "%" and target_hint and R.CONCEPTS[target_hint].unit == "%": sig = target_hint
        if sig is None: continue
        v, conv = _to_unit(m.group("num"), unit, sig)
        if v is None: continue
        S.take(m.start(), m.end()); out.append(Condition(signal=sig, op=CMP[m.group("cmp")], value=float(m.group("num")) if conv else v, unit=conv))
    return out


def _pick(names, profile):
    names = list(names)
    if len(names) == 1 or profile is None: return names[0]
    ok = [n for n in names if R.CONCEPTS[n].dtype != "categorical" and profile.resolve(n).available]
    return (ok or names)[0]


_NEG_LEAD = re.compile(r"\b(?:excluding|exclude|excludes|without|except(?: for)?|ignoring|other than|not counting|not including|apart from|besides|leaving out|minus|but not|outside(?: of)?|not|never|wasn't|weren't|isn't|aren't|am not|no)\b(?:\s+(?:the|any|all|time|times|period|periods|minutes?|samples?|readings?|events?|warnings?|faults?|brakes?|trips?|which|happened|happen|occurred|occur|took|place|when|while|during|whenever|of|i|was|were|am|is|are|it|car|vehicle|being|that|spent|spend|in|on|at|my))*\s*$")


def _predicates(S, op_events):
    """Named conditions, skipping words that are part of a signal name ("charging power" is a signal, not the condition "charging")."""
    spans = [(m.start(), m.end()) for m in _CONCEPT_RX.finditer(S.t) if R.LEXICON.get(m.group(1).replace("-", " ")) or R.LEXICON.get(m.group(1))]
    out = []
    for m in _PRED_RX.finditer(S.t):
        if not S.free(m.start(), m.end()): continue
        a, b = m.start(1), m.end(1)
        if any(a < y and b > x for x, y in spans): continue
        S.take(m.start(), m.end()); k = m.group(1)
        name = _PRED_PHRASES.get(k) or _PRED_PHRASES[k.replace("-", " ")]
        ml = _NEG_LEAD.search(S.t[max(0, m.start() - 70):m.start()])
        if ml:
            S.negated = getattr(S, "negated", set()) | {name}; off = max(0, m.start() - 70); S.take(off + ml.start(), off + ml.end())      # "excluding the time I was charging": the complement of the state
        out.append((k, name))
    return out


def _events(S):
    out = []
    for m in S.findall(_EVENT_RX):
        k = m.group(1).replace("-", " "); out.append((m.group(1), _EVENT_PHRASES.get(k) or _EVENT_PHRASES.get(m.group(1))))
    return out


def _selection(S, t):
    """'the three least efficient highway trips' -> EntitySelect. Looks at the entity noun and the few words before it, plus whole-sentence ranking phrases."""
    nm = S.find(ENTITY_RX, take=False)
    if not nm: return None
    noun = nm.group(1); entity = "charge_session" if noun in NOUN_FOR_CHARGE or noun.startswith("charge") or noun.startswith("charging") else "trip"
    start = nm.start(); words = list(re.finditer(r"[a-z0-9/%.\-']+", t[:start]))[-6:]; prefix_start = words[0].start() if words else start
    prefix = t[prefix_start:start]
    rank_by, order = None, "desc"; span = None
    for rx, metric, od in RANK_HINTS:
        m = re.search(rx, t)
        if m and metric in R.ENTITY_METRICS[entity] and S.free(m.start(), m.end()): rank_by, order, span = metric, od, m; break
    k = None; klass = None; recency = None
    for m in re.finditer(r"\b(?:top |bottom |worst |best )?(" + NUM + r")\b", prefix):
        if re.search(r"(?:^|\s)(?:last|past|first|top|bottom|worst|best|next)?\s*$", prefix[:m.start()]) or True:
            try: k = int(num(m.group(1))); break
            except Exception: pass
    for w in re.findall(r"[a-z]+", re.sub(r"\bhow long\b", " ", prefix)):                                    # "how long was my last trip" asks for a duration; it does not select long trips
        if w in CLASS_WORDS and not (w == "long" and rank_by == "distance_driven"): klass = CLASS_WORDS[w]
    if re.search(r"\b(last|latest|most recent|previous|final|recent)\b", prefix) and not rank_by: recency = "desc"
    elif re.search(r"\b(first|earliest|oldest)\b", prefix) and not rank_by: recency = "asc"
    if entity == "charge_session": klass = None
    if klass and klass not in R.TRIP_CLASSES: klass = None
    pl = noun.endswith("s") and not noun.endswith("ss") and noun not in ("charges",) or noun == "charges"
    if rank_by is None and recency: rank_by, order = "recency", recency; k = k or 1
    elif rank_by is not None and k is None and not pl: k = 1
    elif rank_by is not None and k is None and pl: k = 5 if re.search(r"\b(which|what|show|list|rank|top)\b", t) else None
    if rank_by is None and klass is None and k is None:                                     # a bare "trip" is not a selection ...
        if re.search(r"\b(?:how many|number of|count of)\b", t):                            # ... unless the trips themselves are being counted
            S.take(nm.start(), nm.end()); return EntitySelect(entity, None, None, "desc", None)
        return None
    if span: S.take(span.start(), span.end())
    MOD = set(CLASS_WORDS) | {"last", "latest", "most", "recent", "previous", "final", "first", "earliest", "oldest", "top", "bottom", "worst", "best", "my", "all", "each", "every", "the", "of", "our", "least", "efficient", "economical"} | set(NUMW)
    S.take(nm.start(), nm.end())
    for w in re.finditer(r"[a-z0-9/%.\-']+", prefix):                       # only the words that describe the selection are explained by it
        if w.group(0) in MOD or re.fullmatch(r"\d+", w.group(0)): S.take(prefix_start + w.start(), prefix_start + w.end())
    if rank_by is None: k = None
    return EntitySelect(entity, klass, rank_by, order, k)


CUE_PATTERNS = {
    "count": r"\b(how many|number of|count of|how often|how frequently|were there any|was there any|any |did i have any|has there been|have there been)\b",
    "list": r"\b(list|show|which|what) .*?(events?|warnings?|faults?|alerts?|codes?|dtcs?|disengagements?|brakings?)\b|\bwhen (did|was|were|has)\b|\bwhat (warnings?|faults?|alerts?|events?)\b",
    "ctx": r"\b(what happened|what was going on|what led|led up|leading up|lead up|before the|before it|prior to|just before|right before|preceded|around the|around when|surrounding|after the)\b",
    "why": r"\b(explain|why did|why was|why is|why has|why does|what caused|what triggered|what made|reason for|what's behind|what is behind|how come)\b",
    "cap": r"\b(do you have|does it have|does the (?:vehicle|car) have|can you measure|can it measure|do you record|does it record|does the (?:vehicle|car) (?:record|measure|log|track)|is .* (?:available|recorded|measured|supported)|do you (?:measure|track|log|know|support)|are you able to (?:measure|read)|do we (?:record|have|measure)|why (?:do|does) (?:you|we|it|the (?:vehicle|car)) (?:not|n't) (?:have|record|measure|track|know)|why is .* not (?:available|recorded|measured)|available data|what (?:signals|sensors) )\b",
    "compare": r"\b(compare|compared|comparison|vs|differ|differs|difference|differences|higher|lower|more|less|better|worse|bigger|smaller|than)\b",
    "trend": r"\bhow (?:has|have|did|does|do) (?:the |my )?(?:[a-z-]+ ){1,5}?(?:changed|change)\b|\b(trend|trending|rising|falling|getting (?:worse|better|higher|lower|hotter|colder|slower|faster|weaker)|increasing|decreasing|going (?:up|down)|over time|improving|degrading|drifting|drift|creeping)\b",
    "corr": r"\b(correlat\w*|linked to|how strongly|relationship between|relation between|link between|related to|depend on|depends on|depend upon|affect|affects|affected by|influence[sd]? by|influence)\b",
    "pct": r"\b(what (?:percent|percentage|%|fraction|share|proportion) of|how much of the time|how much of (?:my |this |that |last |yesterday|today|tonight|every |each )|how much time|how long|how many (?:minutes|mins|hours|hrs|seconds)|what part of|how often)\b|% of the time|\b(?:share|fraction|proportion) of (?:the |this |that )?(?:time|week|day|month|drive|driving|trip)",
    "now": r"\b(what is|what's|whats|how is|how's|is the|current|currently|right now|now|at the moment|latest|live|show me the)\b",
}


AGG_WORDS = [(r"\bhow (?:hot|warm|fast|high|strong|hard|big)\b.*\b(?:get|got|go|went|reach\w*|climb\w*)\b", "max"), (r"\bhow (?:cold|low|slow)\b.*\b(?:get|got|go|went|reach\w*|drop\w*|fall\w*)\b", "min"),
(r"\b(?:average|avg|mean|typical|typically|on average|usual)\b", "mean"), (r"\b(?:maximum|max|highest|peak|hottest|top|biggest|largest|greatest|most extreme|fastest|strongest|hardest|harshest|at its (?:worst|peak|highest))\b", "max"), (r"\b(?:minimum|min|lowest|coldest|smallest)\b", "min"),
             (r"\b(?:total|sum|in total|altogether|combined)\b", "sum"), (r"\bmedian\b", "median"), (r"\b(?:variation|variability|variable|std|standard deviation|spread|fluctuat\w*|vary|varies|varied|stability|stable)\b", "std"), (r"\b(?:95th percentile|p95|95th)\b", "p95"), (r"\bp50\b", "median"), (r"\b(?:first|earliest|oldest)\b(?: [a-z/]+){0,3}? (?:reading|value|sample|measurement)s?\b", "first"),
             (r"\b(?:change|changed|drop|dropped|fell|rise|rose|increase|increased|decrease|decreased|gain|gained|lost|loss|(?:go|goes|going|went|gone) (?:up|down)|climb\w*|grow|grew)\b", "delta")]


WHEEL_BASES = {"tyre_pressure": "tyre_pressure", "wheel_speed": "wheel_speed"}
WHEEL_RX = re.compile(r"(?<![a-z])(front|rear|back)[ -]*(left|right)(?![a-z])")
REMOVE_RX = re.compile(r"^(?:and |ok |okay |actually |now )*(?:(?:without|no longer|drop|remove|ignore|forget)(?: the| that| about)? .*|(?:include|add) .*(?:too|as well|also)|(?:all|any|every) (?:trips|driving|the trips|journeys)|all of it)$")


def _wheel_variant(target, pos):
    base = target
    for suf in ("_fl", "_fr", "_rl", "_rr"):
        if target.endswith(suf): base = target[: -len(suf)]
    code = {"front": "f", "rear": "r", "back": "r"}[pos[0]] + pos[1][0]; name = base + "_" + code
    return name if base in WHEEL_BASES and name in R.CONCEPTS else None


def _revise(text, p, state):
    """Generic composition steps on top of the vocabulary parse: a wheel position ("front left", "rear right") makes a tyre / wheel signal that specific one (never silently the general one); a follow-up that removes a restriction
    ("without the rain restriction", "actually all trips") clears the matching conditions of the previous query."""
    t = normalise(text); m = WHEEL_RX.search(t)
    if m:
        pos = (m.group(1), m.group(2)); last = state.last if state is not None else None
        if p.query is not None and p.query.target and _wheel_variant(p.query.target, pos): p.query.target = _wheel_variant(p.query.target, pos)
        elif p.query is None and last is not None and last.target and _wheel_variant(last.target, pos) and len(t.split()) <= 6:
            nq = last.clone() if hasattr(last, "clone") else copy.deepcopy(last); nq.target = _wheel_variant(last.target, pos); nq.source = "followup"; nq.notes = []; return Parse(nq, 0.9, followup=True, reason="follow-up: another wheel")
    if state is not None and state.last is not None and len(t.split()) <= 8 and re.search(r"\b(?:the )?(?:day|week|month) before\b|\b(?:the )?(?:previous|prior|preceding) (?:day|week|month)\b|\bthe one before\b", t):
        last = state.last; SHIFT = {"yesterday": "day_before_yesterday", "today": "yesterday", "this_week": "last_week", "this_month": "last_month"}
        if last.scope.time.type == "calendar" and last.scope.time.name in SHIFT:
            nq = last.clone() if hasattr(last, "clone") else copy.deepcopy(last); nq.scope.time = TimeSpec("calendar", name=SHIFT[last.scope.time.name]); nq.source = "followup"; nq.notes = []
            return Parse(nq, 0.9, followup=True, reason="follow-up: the period before the previous one")
    last = state.last if state is not None else None
    if last is None and re.search(r"\b(those|them|these)\b", t) and not re.search(r"\b(?:these|those) (?:last|past|previous)\b", t) and re.search(r"\b(how many|count|number of|list|show|which|when|first|last)\b", t) and len(t.split()) <= 8:
        return Parse(None, 0.0, clarify="I don't have a previous set of events or trips to refer to. Which events or trips do you mean?", reason="reference without antecedent")
    if last is not None and len(t.split()) <= 9:
        m = re.match(r"^(?:and |actually |ok |now )*(?:make|change) (?:that|it) (?:to )?(.+)$", t)                       # "make that colder than 0": replace the threshold of the previous query, keep everything else
        if m:
            from . import interpretation as _IN
            trs = _IN.thresholds(m.group(1)); sigc = [c for c in last.scope.conditions if c.signal and c.value is not None]
            if len(trs) == 1 and len(sigc) == 1:
                tr = trs[0]; c0 = sigc[0]; v, conv = _to_unit(str(tr["num"]), tr["unit"], c0.signal) if tr["unit"] not in (None, "degree") else (tr["num"], None)
                if v is not None:
                    nq = last.clone() if hasattr(last, "clone") else copy.deepcopy(last); nq.scope.conditions = [x for x in nq.scope.conditions if x is not c0 and x.to_text() != c0.to_text()] + [Condition(signal=c0.signal, op={"gt": "gt", "ge": "ge", "lt": "lt", "le": "le"}[tr["cmp"]], value=float(tr["num"]) if conv else v, unit=conv)]
                    nq.source = "followup"; nq.notes = []; return Parse(nq, 0.9, followup=True, reason="follow-up: threshold replaced")
            if len(trs) == 1 and len(sigc) > 1: return Parse(None, 0.0, clarify="Which threshold should I change - " + " or ".join(f"the {R.label_of(c.signal)} one" for c in sigc[:3]) + "?", options=[R.label_of(c.signal) for c in sigc[:3]], reason="several thresholds in the previous question")
        if re.search(r"\bthen\b", t) and not re.match(r"^(?:and |so |ok )*then\b.*\b(?:what about|how about)\b", t) and (last.event is not None or last.scope.select is not None) and last.scope.time is not None and not _times(_S(t), now=None, tz=0):
            alts = ["the moment of the previous event" if last.event is not None else "the time of the previous trip", "the period of the previous question"]
            return Parse(None, 0.0, clarify="By 'then' do you mean " + alts[0] + " or " + alts[1] + "?", options=alts, reason="'then' has two plausible antecedents")
        if last.op == "list_events" and last.event is not None and re.match(r"^(?:and |ok |okay |please |show |give me |what about )*(?:the )?(?:next|more|rest|remaining|following|other)\b", t):
            nq = last.clone() if hasattr(last, "clone") else copy.deepcopy(last); k = (last.event.n or 1) if last.event.selector == "page" else 1; nq.event.selector = "page"; nq.event.n = k + 1; nq.source = "followup"; nq.notes = []
            return Parse(nq, 0.9, followup=True, reason="follow-up: the next page of events")
        if re.search(r"\b(those|them|these)\b", t) and re.search(r"\b(how many|count|number of|list|show|which|when|first|last)\b", t) and last.event is not None:
            nq = last.clone() if hasattr(last, "clone") else copy.deepcopy(last); cnt = bool(re.search(r"\b(how many|count|number of)\b", t)); nq.op = "count" if cnt else "list_events"
            if not cnt: nq.event.selector = "first" if "first" in t else "last" if re.search(r"\blast\b", t) else "all"
            nq.source = "followup"; nq.notes = []; return Parse(nq, 0.9, followup=True, reason="follow-up: the previous event set")
        if re.search(r"\b(those|them|these)\b", t) and last.event is None and last.scope.select is None and len(t.split()) <= 7:
            return Parse(None, 0.0, clarify="I don't have a previous set of events or trips to refer to. Which events or trips do you mean?", reason="reference without antecedent")
    if state is not None and state.last is not None and len(t.split()) <= 9 and REMOVE_RX.match(t) and state.last.scope.conditions:
        S = _S(t); named = {k for _, k in _predicates(S, [])}; nq = state.last.clone() if hasattr(state.last, "clone") else copy.deepcopy(state.last)
        keep = [c for c in nq.scope.conditions if named and not (c.predicate in named or (c.predicate == "raining" and "rain" in t) or (c.predicate and c.predicate.replace("_", " ") in t))]
        if len(keep) != len(nq.scope.conditions) or not named:
            nq.scope.conditions = keep if named else []; nq.source = "followup"; nq.notes = []; return Parse(nq, 0.9, followup=True, reason="follow-up: restriction removed")
    return p


def parse(text, now=None, tz=0, profile=None, state=None):
    """Text -> Parse. `now` / `tz` resolve weekdays and dates without a year; `profile` breaks ties between concepts the vehicle does and does not record."""
    import time as _time
    t0 = _time.perf_counter()
    try:
        from . import nlu as _nlu
        text = _nlu.shorthand(text)
    except Exception: pass
    text = re.sub(r"^(?:now|ok|okay|so|then)[, ]+(?=(?:the )?same|for\b|what about|how about|and\b|only\b|just\b)", "", re.sub(r"^(?:no|nope|sorry)[, ]+(?:i )?(?:meant|mean)\b", "actually", str(text).strip(), flags=re.I), flags=re.I)             # "now the same for last month": the word 'now' is a discourse marker here, not a time
    p = _parse(text, now, tz, profile, state)
    try:                                                                            # words that are in no vocabulary are misspellings: the repaired sentence is read as well, and wins when it is executable
        from . import nlu as _nlu2
        fixed0, fixes0 = _nlu2.normalize(str(text), repair_typos=True)
        if fixes0:
            p0 = _parse(fixed0, now, tz, profile, state)
            if p0.ok and (not p.ok or len(p0.unexplained or []) <= len(p.unexplained or [])) and p0.query is not None: p = p0; p.reason = "typos repaired: " + ", ".join(f"{a}->{b}" for a, b in fixes0[:3])
    except Exception: pass
    p = _revise(text, p, state)
    if not p.ok and not p.clarify and not p.followup:                                 # second pass: the same sentence with misspelt words repaired; a typo-created ambiguity must not suppress repair
        try:
            from . import nlu
            fixed, fixes = nlu.normalize(str(text), repair_typos=True)
            if fixes:
                p2 = _parse(fixed, now, tz, profile, state)
                if p2.confidence > p.confidence or (p2.ok and len(p2.unexplained or []) < len(p.unexplained or [])): p = p2; p.reason = "typos repaired: " + ", ".join(f"{a}->{b}" if isinstance(f, tuple) else str(f) for f in fixes[:3] for a, b in [f if isinstance(f, tuple) else (f, f)])
        except Exception: pass
    if p.query is not None: p.query.text = str(text)[:300]
    p.ms = (_time.perf_counter() - t0) * 1000
    return p


def _parse(text, now, tz, profile, state):
    t = normalise(text)
    if state is not None and state.pending is not None:
        r = _resolve_pending(t, state, profile)
        if r is not None: return r
    S = _S(t); notes = []
    # --- vocabulary spans, longest first so "battery temperature" is never read as "battery" + "temperature"
    sel = _selection(S, t)
    ev_hits = _events(S)
    events_first = ev_hits
    times = _times(S, now, tz)                                                      # after the event vocabulary: "plugged in" must keep its preposition
    conds_sig = _conditions(S, profile, None)
    pred_hits = _predicates(S, ev_hits)
    # --- operation cues
    low = t
    cue = {k: bool(re.search(v, low)) for k, v in CUE_PATTERNS.items()}
    causal_raw = causal = bool(re.search(r"\b(why|how come|what makes|what is making|what's making|whats making|what causes|what is causing|what's causing|reason|explain|what explains|account for|help me understand|what is behind|what's behind|whats behind)\b", t)) and not events_first
    change_words = bool(re.search(r"\b(lately|recently|dropped|decreased|reduced|shrunk|shortened|increased|rose|risen|fell|fallen|gone (?:up|down)|got (?:worse|better|shorter|hotter|colder)|changed|different|than (?:before|usual|it used to|last)|compared|since|used to|over the (?:last|past|previous)|in the (?:last|past)|this week|this month|last (?:week|month))\b", t)) or bool(times)
    causal = causal and change_words
    if ((causal_raw and not causal) and not cue["cap"]) or (causal and os.environ.get("VGPT_CHANGE_OPS", "1") in ("0", "off")):
        return Parse(None, 0.0, reason="causal question about a signal: the older analysis engines handle it", unexplained=[])
    when_changed = bool(re.search(r"\bwhen (?:did|was|has|have|do)\b.*\b(?:chang\w*|jump\w*|drop\w*|start\w* to|began|begin|fall\w*|fell|rise|rose|worst|most|shift\w*)\b|\b(?:biggest|largest|greatest|sharpest|largest|steepest)\b.*\b(?:change|shift|jump|drop|rise|increase|decrease|fall)\b|\b(?:which|what) (?:day|date)\b.*\b(?:jump\w*|drop\w*|worst|chang\w*|fell|rose)\b|\bpinpoint\b", t)) and not events_first and not causal
    if re.search(r"^(does|do|will|would|can|could)\b.*\b(affect|impact|influence|hurt|reduce|improve)\b", t) and not re.search(r"\b(correlat\w*|relationship|related)\b", t):
        return Parse(None, 0.0, reason="'does X affect Y' asks for an explanation: the older analysis engines handle it", unexplained=[])
    if len(t.split()) > 40: return Parse(None, 0.0, reason="too long for the rules")
    if re.search(r"\b(best|worst)\b.*\b(and|than|versus|vs|against)\b.*\b(best|worst)\b", t) or re.search(r"\b(worst|best|least|most|longest|shortest)\b.*\bthan\b.*\b(worst|best|least|most|longest|shortest)\b", t) or re.search(r"^(how (can|do|should|could) i|what should i|any (tips|advice)|tips|advice)\b", t):
        return Parse(None, 0.0, reason="comparison of two selections or an advice question: not expressible as one query")
    # --- targets
    tg_hits = []
    for m in _CONCEPT_RX.finditer(t):
        if not S.free(m.start(), m.end()): continue
        names = R.LEXICON.get(m.group(1).replace("-", " ")) or R.LEXICON.get(m.group(1))
        if not names: continue
        tg_hits.append((m, _pick(sorted(names), profile)))
    # bare heads ("temperature") resolved by a context word, or turned into a clarification
    bare = None
    for m in re.finditer(r"(?<![\w/])(temperature|temp|power|voltage|current|pressure|health|flow)(?![\w/])", t):
        if not S.free(m.start(), m.end()) or any(m.start() < h[0].end() and m.end() > h[0].start() for h in tg_hits): continue
        ctx = HEAD_BY_CONTEXT.get(m.group(1)) or {}; word = next((w for w in re.findall(r"[a-z0-9]+", t) if w in ctx), None)
        if m.group(1) == "current" and re.search(r"\bcurrent (speed|value|reading|level|state|status)\b|\bthe current\b", t): continue
        if word: tg_hits.append((m, ctx[word])); S.take(m.start(), m.end()); S.take(*next(((x.start(), x.end()) for x in re.finditer(re.escape(word), t) if S.free(x.start(), x.end())), (0, 0)))
        elif m.group(1) in BARE_OPTIONS: bare = m.group(1)
    tg_hits.sort(key=lambda h: h[0].start())
    if not tg_hits and not events_first and os.environ.get("VGPT_PARTS_MATCH", "1") not in ("0", "off"):          # nothing matched strictly: describe-by-parts (all words of one phrase, unique best candidate)
        from . import vocab_parts as _I
        hit = _I.event_by_parts(t) if re.search(r"\b(?:how many|how often|how frequently|times|count|number of|list|show|when|any|was there|were there|has there|have there|at all|ever|first|last time)\b", t) else None
        if hit:
            events_first.append((t, hit[0]))
            for a, b in hit[1]: S.take(a, b)
        else:
            hit = _I.signal_by_parts(t)
            if hit:
                class _M:
                    def __init__(self, a, b): self.a, self.b = a, b
                    def start(self): return self.a
                    def end(self): return self.b
                    def group(self, *_): return t[self.a:self.b]
                _mm = _M(min(a for a, _ in hit[1]), max(b for _, b in hit[1])); _mm.parts = list(hit[1])                 # only the words of the phrase are explained, not what lies between them
                tg_hits.append((_mm, hit[0]))
                for a, b in hit[1]: S.take(a, b)
    taken_by_targets = [(m.start(), m.end()) for m, _ in tg_hits]
    for k, v in CUE_PATTERNS.items():                         # the words that carry the operation are part of the understood sentence (but never words that name a signal)
        if cue[k] and k != "now":
            for m in re.finditer(v, t):
                if S.free(m.start(), m.end()) and not any(m.start() < y and m.end() > x for x, y in taken_by_targets):
                    # The trend pattern deliberately spans its grammatical
                    # subject (``how has <measure> changed``).  Marking that
                    # entire span as understood hid an unknown requested
                    # measure and let a later follow-up inherit the previous
                    # signal.  Consume only the change predicate; the subject
                    # must be claimed separately by schema linking.
                    if k == "trend":
                        op = re.search(r"\b(?:changed?|trend(?:ing)?|rising|falling|getting|increasing|decreasing|going|improving|degrading|drifting?|creeping)\b", m.group(0))
                        if op is not None:
                            S.take(m.start() + op.start(), m.start() + op.end())
                            continue
                    S.take(m.start(), m.end())
    targets = []
    for m, c in tg_hits:
        if c not in targets: targets.append(c)
        for a_, b_ in getattr(m, "parts", [(m.start(), m.end())]): S.take(a_, b_)
    SPECIFIC = {"regen_energy": ("energy_used",), "charging_power": ("battery_power",)}                  # a named phenomenon already carries its generic quantity ("energy ... regenerative braking recover")
    for spec_, gen_ in SPECIFIC.items():
        if spec_ in targets: targets = [x for x in targets if x not in gen_]
    # A modal vehicle-ability construction asks for a specification, not an
    # observed maximum. The mapping is grammatical and ontology-level.
    rated_match = re.search(r"\bhow fast can\s+(?:this |the |my |our )?(?:car|vehicle|ev|it)\s+(?:go|travel)\b|\b(?:rated|specified)\s+(?:maximum|top) speed\b|\btop speed specification\b", t)
    rated_speed = bool(rated_match)
    if rated_speed:
        targets = ["rated_max_speed" if x == "speed" else x for x in targets]
        if "rated_max_speed" not in targets: targets.insert(0, "rated_max_speed")
        cue["cap"] = True
        S.take(rated_match.start(), rated_match.end())
    # A bounded transition is a first-class physical operation, not a generic
    # threshold filter.  This is a grammatical construction over arbitrary
    # numeric bounds, rather than a route for a particular transcript sentence.
    band_bounds = None
    if "acceleration" in targets:
        mb = re.search(r"\b(?:between|from)\s+(" + NUM + r")\s*(mph|km/h|kmph|kph)?\s+(?:and|to)\s+(" + NUM + r")\s*(mph|km/h|kmph|kph)?\b", t)
        if mb:
            a, b = float(num(mb.group(1))), float(num(mb.group(3)))
            ua, ub = UNIT_ALIASES.get((mb.group(2) or mb.group(4) or "km/h").lower(), "km/h"), UNIT_ALIASES.get((mb.group(4) or mb.group(2) or "km/h").lower(), "km/h")
            try: a, b = R.convert(a, ua, "km/h"), R.convert(b, ub, "km/h")
            except ValueError: pass
            band_bounds = tuple(sorted((a, b)))
            S.take(mb.start(), mb.end())
    # --- conditions that need a target to know their unit ("above 40" for a temperature question)
    if not conds_sig:
        conds_sig = _conditions(S, profile, targets[0] if targets else None)
    # --- hard braking is an event when counted / listed and a condition when something else is measured "during" it
    count_like = cue["count"] or cue["list"] or cue["ctx"] or cue["why"]
    events = []; preds = [name for _, name in pred_hits]
    _bin = [x for x in targets if x == "rain" and R.CONCEPTS[x].dtype == "binary"]
    if _bin and len(targets) > len(_bin):
        targets = [x for x in targets if x not in _bin]; preds.append("raining")
    if "driving" in preds and ("city" in preds or "highway" in preds): preds = [x for x in preds if x != "driving"]
    if not targets and {"city", "highway"}.issubset(set(preds)):
        return Parse(None, 0.0, clarify="What would you like to compare between city and highway driving: energy consumption, average speed, distance, or another recorded signal?", options=["energy consumption", "average speed", "distance"], reason="comparison groups were clear but the requested quantity was omitted")
    for phrase, ev in events_first:
        if ev in R.EVENT_AS_PREDICATE and not count_like and (targets or cue["pct"]): preds.append(R.EVENT_AS_PREDICATE[ev])
        else: events.append(ev)
    # --- aggregation
    agg = None; agg_word = None
    for rx, a in AGG_WORDS:
        m = S.find(rx)
        if m and agg is None: agg, agg_word = a, m.group(0)
    if agg is None:                                                                          # a structural match may have consumed the aggregation word; recover its semantic role from the complete sentence
        for rx, a in AGG_WORDS:
            m = re.search(rx, low)
            if m: agg, agg_word = a, m.group(0); break
    unit = None
    mu = S.find(r"\b(?:in|using|as|into) (mph|km/h|psi|bar|kpa|f|c|miles|km|kwh|wh/km|hp|kw|w)\b")
    if mu: unit = UNIT_ALIASES.get(mu.group(1).lower(), mu.group(1))
    # --- assemble
    q = Query(); q.scope = Scope(); conf_cap = 1.0
    q.scope.select = sel
    neg_preds = getattr(S, "negated", set())
    q.scope.conditions = [Condition(predicate=x, op="not" if x in neg_preds else None) for x in dict.fromkeys(preds)] + conds_sig
    if band_bounds is not None:
        q.scope.conditions = [c for c in q.scope.conditions if c.signal != "speed"] + [Condition(signal="speed", op="ge", value=band_bounds[0], unit="km/h"), Condition(signal="speed", op="le", value=band_bounds[1], unit="km/h")]
    if events and len(set(events)) > 1: events = events[:1]
    if len(targets) > 2: targets = targets[:2]
    # op
    if band_bounds is not None:
        q.op, q.target = "acceleration_band", "acceleration"
    elif cue["cap"] and targets and not times:
        q.op, q.target = "capability", targets[0]
    elif causal and targets and not events:
        q.op = "explain_change"
    elif when_changed and targets and not events:
        q.op = "detect_change"
    elif cue["why"] and events:
        q.op = "explain_event"
    elif events and cue["ctx"]:
        q.op = "event_context"
    elif events and (cue["count"] and not re.search(r"\b(when|list|show|which|what)\b", low) or re.search(r"\bhow (?:many|often|frequently)\b", low) or re.search(r"^(?:did|was there|were there|has there|have there|is there) ", low) and not cue["list"] or re.search(r"^(?:was|were|did|has|have|had) ", low) and not cue["list"] and not re.search(r"\b(?:last|first|latest|previous|most recent|when|which|what)\b", low) or re.search(r"\b(?:any|number of)\b", low) and not cue["list"]): q.op = "count"
    elif events: q.op = "list_events"
    elif sel is not None and not targets and (sel.rank_by or cue["count"]): q.op = "count" if (cue["count"] and not sel.rank_by) else "rank"
    elif len(targets) == 2 and cue["corr"] and not agg_word: q.op = "correlation"
    elif cue["compare"] and (len(set(preds)) >= 2 or len(times) >= 2) and not (cue["trend"] and len(times) < 2):
        q.op = "compare"
    elif cue["trend"] and targets: q.op = "trend"
    elif re.search(r"\b(?:how many times|number of times|how often)\b", t) and conds_sig and not events and not preds and not agg_word:
        q.op = "count"; targets = [conds_sig[0].signal] if not targets else targets                         # episodes in which the threshold was crossed
    elif cue["pct"] and (conds_sig or preds or (targets and R.CONCEPTS[targets[0]].dtype != "numeric")): q.op = "percent_time"
    elif sel is not None and sel.rank_by and re.search(r"\b(?:which|what) (?:trip|journey|drive|charge|charging session|session)\b", t): q.op = "rank"
    elif targets and agg in ("max", "min") and agg_word and not re.search(r"\b(average|avg|mean)\b", t): q.op = "extreme"
    elif targets and sel is not None: q.op = "extreme" if agg in ("max", "min") and not re.search(r"\b(average|avg|mean)\b", t) else "aggregate"
    elif targets and (times and times[0].type != "current" or agg): q.op = "aggregate"
    elif targets and not times and cue["now"]: q.op = "current_value"
    elif targets and (times and times[0].type == "current") and not (conds_sig or preds): q.op = "current_value"
    elif targets and not times: q.op = "current_value" if not (conds_sig or preds) else "aggregate"
    else: q.op = ""
    if q.op == "current_value" and (conds_sig or preds): q.op = "aggregate"
    # --- fill
    if q.op in ("rank", "count") and sel is not None and q.op == "rank" and sel.k is None: sel.k = 5
    if q.op != "capability" and targets: q.target = targets[0]
    if q.op == "correlation": q.target2 = targets[1]
    if q.op == "compare":
        if len(set(preds)) >= 2:
            q.groups = list(dict.fromkeys(preds))[:2]; q.scope.conditions = [c for c in q.scope.conditions if not (c.predicate in q.groups)]
        else: q.scope_b = Scope(times[1], sel, list(q.scope.conditions))
        q.agg = agg if agg in ("mean", "max", "min", "median", "sum") else ("sum" if q.target in R.EXTENSIVE else "mean")
        if q.target is None and "consumption" not in t: q.op = ""
    if q.op == "percent_time" and not q.target and q.scope.conditions and conds_sig: q.target = None
    if q.op == "percent_time":                                                                                  # "how many hours / minutes ..." is the asked-for duration: the words are understood, not leftover
        for m_ in re.finditer(r"\bhow many (minutes|mins|hours|hrs|seconds)\b|\bhow much (time)\b|\bhow (long)\b", S.t):
            g_ = next(i_ for i_ in (1, 2, 3) if m_.group(i_)); S.take(m_.start(g_), m_.end(g_))
    if q.op == "percent_time" and q.target == "distance_driven" and conds_sig and re.search(r"\bhow (?:many|much) (?:minutes|mins|hours|hrs|seconds|time)\b", low): q.target = None   # time spent above a threshold, not a distance
    if q.op in ("extreme",): q.agg = agg
    elif q.op == "aggregate":
        c = R.CONCEPTS.get(q.target) if q.target else None
        q.agg = agg if agg else (EXTENSIVE_DEFAULT.get(q.target) or "mean")
        if c is not None and q.target in R.EXTENSIVE and q.agg == "mean" and not agg: q.agg = "sum"
        if agg is None and q.target not in EXTENSIVE_DEFAULT: notes.append("You didn't say which statistic, so I used the average.")
        if c is not None and c.dtype == "binary": q.agg = "count"
        if c is not None and c.dtype == "categorical" and q.agg not in R.AGG_CATEGORICAL: q.agg = "percent_time"
    elif q.op in ("trend", "explain_change", "detect_change"): q.agg = None
    if q.op == "current_value": q.agg = None
    if events: q.event = _event_ref(events[0], t, low)
    if q.op == "list_events" and q.event is not None and q.event.selector == "last":                       # "the ten latest codes": the n newest, not just the newest one
        mk = re.search(r"\b(" + NUM + r") (?:latest|most recent|newest|last|recent)\b", t)
        if mk:
            try:
                k_ = int(num(mk.group(1)))
                if 2 <= k_ <= 100: q.event.selector = "all"; q.event.n = k_; S.take(mk.start(), mk.end()); q.scope.time = q.scope.time if re.search(r"\b(?:days?|weeks?|months?|hours?|minutes?|yesterday|today|tonight|week|month|since|ago)\b", t) else TimeSpec("all")
            except Exception: pass
    if q.op == "list_events" and q.event is not None and q.event.selector == "last" and re.search(r"\b(all|every|each|list|show|display|give me)\b", t) and not re.search(r"\b(last|latest|most recent|previous|newest|when was|when did)\b", re.sub(r"\b(?:the )?(?:last|past|previous) (?:\w+ )?(?:minutes?|hours?|days?|weeks?|months?|nights?|weekends?)\b", " ", t)): q.event.selector = "all"           # "the last 30 days" is a period, not "the last event"
    if q.op in ("explain_change", "detect_change") and q.target == "energy_used": q.target = "wh_per_km"                   # a change in energy use is judged per distance: total energy also changes with how far you drove
    if q.op == "event_context" and q.event: _context_window(q.event, t, S)
    # time
    if times: q.scope.time = times[0]
    else:
        if q.op == "current_value": q.scope.time = TimeSpec("current")
        elif sel is not None:
            q.scope.time = TimeSpec("all"); notes.append("No period given, so I looked at the whole recorded history.")
        elif q.op in ("event_context", "explain_event") or (q.op == "list_events" and q.event and (q.event.selector in ("last", "first", "nth", "most_severe") or (q.event.selector == "all" and q.event.n))): q.scope.time = TimeSpec("all")       # a singled-out occurrence is searched over history unless a period is given
        elif q.op == "capability": q.scope.time = TimeSpec("all")
        elif q.op == "acceleration_band": q.scope.time = TimeSpec("relative", 7, "day"); notes.append("No period given, so I used the last 7 days of retained high-rate telemetry.")
        elif q.op == "detect_change": q.scope.time = TimeSpec("all")
        elif q.op == "explain_change": q.scope.time = TimeSpec("relative", 14, "day"); notes.append("No period given, so I compared the last 14 days with the 14 days before.")
        else:
            q.scope.time = TimeSpec("relative", 7, "day"); notes.append("No period given, so I used the last 7 days.")
    if q.op == "current_value" and times and times[0].type != "current": q.scope.time = times[0]
    if q.op in ("aggregate", "extreme", "trend", "acceleration_band") and q.scope.time.type == "current": q.scope.time = TimeSpec("relative", 24, "hour"); notes.append("For a statistic, 'now' means the last 24 hours.")
    if q.op == "trend" and q.scope.time.type == "relative" and q.scope.time.unit in ("minute", "hour") : notes.append("A trend needs a longer period; the result may be weak.")
    if q.op == "trend" and not times: q.scope.time = TimeSpec("relative", 14, "day"); notes[-1:] = ["No period given, so I used the last 14 days."]
    if unit and q.target and R.unit_family(unit) == R.unit_family(R.CONCEPTS[q.target].unit): q.unit = unit
    q.notes = notes
    # --- clarification / confidence
    mw = WHEEL_RX.search(t)
    if mw and any(_wheel_variant(x, (mw.group(1), mw.group(2))) for x in targets): S.take(mw.start(), mw.end())          # a wheel position that specialises the signal is understood, not leftover
    if q.event and q.event.selector == "most_severe":
        for m in re.finditer(r"\b(?:critical|most serious|most severe|worst)\b", t): S.take(m.start(), m.end())
    left = S.leftover(); cw = S.content_words()
    kinds = (1 if targets else 0) + (1 if events else 0) + (1 if sel is not None else 0)
    if q.op in ("list_events", "count", "event_context", "explain_event") and targets and not (conds_sig or preds): left = left + ["(signal and event together)"]
    if len(targets) > 1 and q.op not in ("correlation", "compare"): left = left + ["(several signals)"]
    if kinds >= 3 or (events and sel is not None and q.op != "count" and targets): left = left + ["(mixed topics)"]
    ex = [w for w in left]
    conf = 1.0 - (len(ex) / max(1, len(cw))) if cw else 0.0
    if not q.op: conf = min(conf, 0.3)
    out = Parse(q if q.op else None, round(max(0.0, conf), 2), unexplained=ex)
    if bare and not targets and q.op == "" and not events:
        out.clarify = f"Which {bare} do you mean - " + ", ".join(BARE_OPTIONS[bare][:-1]) + f" or {BARE_OPTIONS[bare][-1]}?"; out.options = BARE_OPTIONS[bare]; out.confidence = 0.0
        pend = Query(op="aggregate", agg="mean"); pend.scope = Scope(times[0] if times else TimeSpec("relative", 7, "day"), None, q.scope.conditions); out.pending = pend
    elif bare and (not targets) and q.op in ("", "current_value", "aggregate", "extreme", "trend") and q.target is None:
        out.clarify = f"Which {bare} do you mean - " + ", ".join(BARE_OPTIONS[bare][:-1]) + f" or {BARE_OPTIONS[bare][-1]}?"; out.options = BARE_OPTIONS[bare]; out.pending = q
    if len(targets) > 1 and q.op not in ("correlation", "compare") and out.query is not None and not set(targets) <= {"energy_used", "wh_per_km"}:                       # "energy efficiency" is one concept (energy per distance), not two signals
        q.notes.append(f"You named more than one signal; I answered for {R.label_of(q.target or targets[0])}.")
    if out.clarify: return out
    if state is not None and state.last is not None:
        f = _followup(t, q, targets, times, agg, preds, conds_sig, sel, events, state.last, S, cue, unit)
        if f is not None: return f
    return out


def _event_ref(ev, t, low):
    sel = "most_severe" if re.search(r"\b(?:critical|most serious|most severe|worst)\b", t) else "first" if re.search(r"\bfirst\b", t) else "last"
    r = EventRef(type=ev, selector=sel)
    return r


def _context_window(e, t, S):
    m = re.search(r"(" + NUM + r") ?(seconds?|secs?|minutes?|mins?|hours?|hrs?) (?:before|prior|ahead|leading|earlier)", t)
    if m:
        v = int(num(m.group(1))); mult = 1 if m.group(2).startswith("sec") else 60 if m.group(2).startswith("min") else 3600; e.before = min(6 * 3600, v * mult)
    elif re.search(r"\b(before|leading up|led up|prior|preceded|lead up|what led)\b", t): e.before = 600
    m = re.search(r"(" + NUM + r") ?(seconds?|secs?|minutes?|mins?|hours?|hrs?) (?:after|following|later)", t)
    if m:
        v = int(num(m.group(1))); mult = 1 if m.group(2).startswith("sec") else 60 if m.group(2).startswith("min") else 3600; e.after = min(6 * 3600, v * mult)
    elif re.search(r"\b(after|around|surrounding|following)\b", t): e.after = 600
    if re.search(r"\b(around|surrounding|what happened)\b", t) and not e.before: e.before = 300
    if not e.before and not e.after: e.before = 600


# ------------------------------------------------------------------------------------------------ follow-ups and clarifications
FOLLOW_RX = re.compile(r"^(?:and |but |so |ok |okay |then |also )*(?:what about|how about|and for|and in|and during|same for|same but|what if|and what about|now for|and now|how was it|what was it|and while|and on|and when|only|just|actually|instead|make that|make it|change that to|change it to|switch to|swap to|for|in|during|on|at|and)\b")


STRONG_FOLLOW_RX = re.compile(r"^(?:and |but |so |ok |okay |then |also )*(?:what about|how about|and for|same for|same but|what if|and what about|now for|and now|how was it|and while|and when|only|just|actually|instead|make that|make it|change that to|change it to|switch to)\b")


def _followup(t, q, targets, times, agg, preds, conds, sel, events, prior, S, cue, unit):
    components = bool(targets or times or agg or preds or conds or sel or events)
    cw = S.content_words()
    short = len(cw) <= 6
    self_contained = q.op not in ("", "aggregate", "current_value", "extreme", "trend") or (bool(q.target) and q.op != "")
    strong = bool(STRONG_FOLLOW_RX.match(t)) or bool(re.match(r"^(?:and|also|but)\b", t) and len(t.split()) <= 6 and not any(cue.get(k) for k in ("count", "list", "ctx", "why", "corr", "compare", "trend", "pct", "cap")))        # a short "and <slot>" fragment continues the previous question
    if not components or not (strong or (not self_contained and (FOLLOW_RX.match(t) or short))): return None
    if self_contained and not strong: return None
    if S.leftover(): return None                                                       # a fragment with words we did not understand is not a safe follow-up
    nq = prior.clone() if hasattr(prior, "clone") else prior; refs = []
    nq.source = "followup"; nq.notes = []; changed = False
    if targets == ["rain"] and prior.target and prior.target != "rain" and not re.search(r"\b(?:rain (?:sensor|level|intensity)|how (?:much|hard) (?:is it )?rain)", t):       # "raining" after a question about something else is a restriction, not a new signal
        targets = []; preds = list(preds) + ["raining"]; components = True
    if targets:
        if nq.op == "capability": nq.op = "current_value"
        nq.target = targets[0]; changed = True
        if not agg and nq.op in ("aggregate", "extreme") and nq.agg:                                                      # the statistic follows the new signal (a total of energy is not a total of consumption) unless the person named one
            c_ = R.CONCEPTS[nq.target]
            if c_.dtype == "binary": nq.agg = "count"
            elif c_.dtype == "categorical": nq.agg = nq.agg if nq.agg in R.AGG_CATEGORICAL else "percent_time"
            elif nq.op == "aggregate" and nq.agg in ("sum", "mean"): nq.agg = "sum" if nq.target in R.EXTENSIVE else "mean"
    else: refs.append("target")
    if times:
        nq.scope.time = times[0]; changed = True
        if nq.op == "current_value" and times[0].type != "current": nq.op = "aggregate"; nq.agg = nq.agg or "mean"
        if times[0].type == "current" and nq.op == "aggregate": nq.op = "current_value"; nq.agg = None
    else: refs.append("time")
    if agg == "delta" and re.match(r"^(?:actually |ok |now )*(?:change|make|switch|swap)\b", t): agg = None                  # "change that to ..." is the replacement verb, not a request for the change of a signal
    if agg and nq.op in ("aggregate", "extreme", "current_value"):
        nq.agg = agg; changed = True
        nq.op = "extreme" if agg in ("max", "min") and nq.op != "aggregate" or (agg in ("max", "min") and any(w in t for w in ("highest", "lowest", "peak", "max", "min"))) else "aggregate"
    if preds or conds:
        new = [Condition(predicate=x) for x in dict.fromkeys(preds)] + conds
        keep = [c for c in nq.scope.conditions if not ((c.predicate in R.MODE_PREDICATES and any(n.predicate in R.MODE_PREDICATES for n in new)) or (c.signal and any(n.signal == c.signal for n in new)) or (c.predicate and any(n.predicate == c.predicate for n in new)))]
        replace_all = bool(re.search(r"\binstead\b|^(?:actually |ok |now )*(?:change|make) (?:that|it)\b|^(?:actually |ok |now )*(?:switch|swap) (?:it |that )?(?:to|for)\b", t))        # an explicit replacement discards the previous restriction(s), not only one of the same kind
        nq.scope.conditions = (new if replace_all else keep + new); changed = True
    else: refs.append("conditions")
    if sel is not None: nq.scope.select = sel; changed = True
    elif nq.scope.select is not None: refs.append("selection")
    if cue.get("count") and nq.op == "list_events" and nq.event is not None and not events: nq.op = "count"; changed = True                    # "how many this month" after a list of events counts them
    if events:
        nq.event = _event_ref(events[0], t, t); changed = True
        if nq.op not in ("count", "list_events", "event_context", "explain_event"): nq.op = "count"
    if unit and nq.target and R.unit_family(unit) == R.unit_family(R.CONCEPTS[nq.target].unit): nq.unit = unit; changed = True
    if not changed: return None
    nq.refs = [r for r in refs if r in ("target", "time", "conditions", "selection")]
    if nq.op == "compare":
        if len(set(preds)) >= 1 and nq.groups:
            pass
    return Parse(nq, 0.9, followup=True, reason="follow-up of the previous question")


def _resolve_pending(t, state, profile):
    """The person answered a clarification question: 'battery' -> the pending query with that target."""
    pend, opts = state.pending, state.options
    if not t or len(t.split()) > 6: return None
    tg = None
    for m in _CONCEPT_RX.finditer(t):
        names = R.LEXICON.get(m.group(1).replace("-", " "))
        if names: tg = _pick(sorted(names), profile); break
    if tg is None:
        for o in opts:
            if o.lower().replace(" ", "") in t.replace(" ", ""):
                head = state.slot or ""
                tg = None
        word = next((w for w in re.findall(r"[a-z0-9]+", t) if any(w == o.split()[0].lower() for o in opts)), None)
        if word:
            for head, ctx in HEAD_BY_CONTEXT.items():
                if ctx and word in ctx and pend is not None and pend.op != "capability": tg = ctx[word]; break
    if tg is None: return None
    q = pend.clone() if hasattr(pend, "clone") else pend; q.target = tg; q.source = "followup"
    if not q.op or q.op == "": q.op = "aggregate"
    if q.op == "aggregate" and not q.agg: q.agg = "mean"
    state.pending = None; state.options = []
    return Parse(q, 0.9, followup=True, reason="answer to the clarification question")


LEGACY_TOPICS = {"battery_temperature", "battery_soc", "speed", "ambient_temperature", "hvac_power", "coolant_flow", "energy_used", "wh_per_km", "distance_driven", "range_estimate", "tyre_pressure", "tyre_temperature", "cabin_temperature",
                 "motor_temperature", "aux_battery_voltage", "battery_soh", "regen_energy", "regen_power", "motor_rpm", "fuel_level", "battery_power", "charging_power", "charging_state", "odometer", "brake_pad_life", "light", "rain", "humidity",
                 "wind_speed", "camera_blockage", "camera_sharpness", "camera_brightness", "lane_confidence", "adas_state", "drive_mode", "heading", "gps_speed"}


ACCUMULATED_TARGETS = {"wh_per_km", "energy_used", "distance_driven", "regen_energy"}


def _legacy_equivalent(q):
    """Shapes the older routes already answer correctly (and that are covered by their tests) - the new engine stays out of their way."""
    sel = q.scope.select; conds = q.scope.conditions
    if q.target is not None and q.target not in LEGACY_TOPICS: return False
    if sel is not None and sel.rank_by == "recency" and not conds and q.op in ("aggregate", "extreme", "current_value"): return True                  # "average speed on my last drive"
    if q.op == "compare" and q.scope_b is not None and not conds and sel is None: return True                                                       # this week against last week
    if q.op == "percent_time" and sel is None and conds and all((c.predicate in ("driving",)) or (c.signal == "speed" and c.op in ("gt", "ge")) for c in conds): return True   # share of time above a speed
    if q.op == "aggregate" and q.agg in ("mean", "sum") and sel is None and len(conds) == 1 and conds[0].predicate in R.MODE_PREDICATES: return True            # "mean battery temperature on the motorway"
    return False


def takes_over(q, report=None):
    """Should the new engine answer this query rather than the older routes? Only for what the older intents cannot express: sample-level conditions, entity selections, period / group comparisons, correlations,
    shares of time, other statistics, signals the older engine never knew - and for signals this vehicle does not record (an explained answer instead of a guess). Everything the older routes already answer well stays there."""
    if q is None: return False
    sel = q.scope.select
    if sel is not None and sel.rank_by == "recency" and not q.target: return False                  # "describe my last trip" is the older trip summary
    if q.op == "capability": return True
    if q.target == "tyre_pressure" and any(e[1] != "tp_*" for e in R.STORE.get("tyre_pressure", {}).values()): return True      # this vehicle stores an aggregate pressure, not four wheels: the older wheel-by-wheel engine cannot answer
    if q.op == "explain_change" and q.target in ("range_estimate", "wh_per_km"): return False                 # the older range / efficiency dialogue (evidence chain, alternatives, missing data) stays
    if q.op in ("explain_change", "detect_change"): return True
    if _legacy_equivalent(q): return False
    if q.op in ("explain_event", "event_context", "count", "list_events"): return bool(q.scope.conditions or sel is not None)
    if q.op == "correlation" and (q.target in ACCUMULATED_TARGETS or q.target2 in ACCUMULATED_TARGETS): return False                 # per-trip quantities: the older correlation engine
    if q.op == "compare" and q.groups and set(q.groups) == {"city", "highway"} and q.target in ("wh_per_km", "energy_used", None): return False   # the older mode comparison
    if q.scope.conditions or sel or q.scope_b or q.groups or q.op in ("percent_time", "correlation", "rank", "compare", "acceleration_band"): return True
    if q.target and q.target not in LEGACY_TOPICS: return True
    if q.target2: return True
    if q.agg in ("median", "std", "p95") : return True
    if q.op in ("aggregate", "extreme") and q.agg == "delta" and q.target in ("battery_soc", "battery_temperature", "speed"): return True
    if report is not None and not report.ok and q.target not in LEGACY_TOPICS: return True
    return False
