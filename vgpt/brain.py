"""Query brain: question -> normalise (typos, synonyms) -> intent (fast rules, else SLM slot filler) -> deterministic analytics -> answer text.

Design rules (they are what keeps the system accurate AND small):
  * The SLM only fills a 3-word form (intent metric event). It never writes numbers or prose, so it cannot hallucinate vehicle facts.
  * The SLM is consulted only when the rules are unsure, and its answer is accepted only if it is CONFIDENT (token probabilities) and GROUNDED
    (the slots it picked must be supported by the question). Otherwise the assistant asks a clarifying question instead of guessing.
  * Topics the car has no data for (tyre pressure, driver score, navigation ...) get an honest 'I don't have that', never a wrong answer.
  * Time windows come from deterministic rules (yesterday, last weekend, since last charge ...), never from the model.
"""
import math, os, re, time
from . import analytics as A
from . import analytics_x as AX
from . import dialogue
from . import nlu
from . import nl2ir
from . import qa_clf
from .store import open_ro

LEX_METRIC = [("lane_q", ["lane detection", "lane quality", "lane confidence", "lane visibility", "lane marking quality"]), ("batt_temp", ["battery temp", "pack temp", "battery heat", "battery hot", "battery temperature", "pack temperature"]), ("coolant_flow", ["coolant", "cooling", "pump"]),
              ("soc", ["state of charge", "soc", "battery level", "charge level", "battery percentage", "how much charge", "battery left", "charge left", "charge do i have"]),
              ("amb", ["outside temp", "ambient", "weather", "temperature outside", "outdoor", "outside temperature"]), ("hvac", ["hvac", "heater", "heating", "climate", "air con", "cabin"]),
              ("energy_use", ["energy", "consumption", "efficiency", "mileage", "fuel", "economy", "thirst", "range"]), ("speed", ["speed"])]
LEX_EVENT = [("CAN_ANOMALY", ["can bus", "bus anomal", "intrusion", "cyber", "injection", "flooding", "fuzz", "spoof", "bus attack", "attack"]),
             ("WARN_BATT_OVERTEMP", ["overheat", "over temp", "overtemp", "temperature warning", "battery warning", "battery overheating", "too hot", "thermal", "heat warning", "temperature alert", "temperature alarm", "heat alert", "overheating warning", "pack alert", "pack alarm"]),
             ("ADAS_DISENGAGE", ["adas", "disengage", "lane keep", "lane assist", "driver assist", "autopilot", "assist quit", "cruise", "lane departure", "drop out"]),
             ("DTC_SET", ["fault", "dtc", "error code", "diagnostic", "trouble code"]), ("CAMERA_DEGRADED", ["blurry", "blurred", "dirty lens", "smudge", "glare", "camera degrad", "camera quality", "fogged", "dirty camera"]), ("CAMERA_BLOCKED", ["camera", "lens", "windscreen sensor", "blockage", "obstruct"]),
             ("CHARGE_START", ["charging session", "charge session", "fast charge", "charged", "charging"])]
ANCHOR_ROUTES = {"ctx", "why_event", "events", "root_cause", "causal_graph", "intervention", "serious", "proximity", "similar", "attr3", "lag", "first_abnormal", "evidence", "history", "incidents", "c1", "c3", "c5", "nuclear1", "diag", "conditions", "standstill", "alt", "conf", "trace"}
NON_ANSWER_INTENTS = {"SAFETY", "NONE", "HELP", "ABOUT", "FEEDBACK", "SMALLTALK", "FOREIGN", "GLOSSARY", "UNITS", "CLARIFY"}
INTENTS = ["WHY_RANGE", "BEFORE_EVENT", "EVENTS", "COMPARE", "LARGEST", "TREND", "STATUS", "USAGE", "SUMMARY", "HEALTH", "TYRES", "WEATHER", "SERVICE", "DRIVING", "PLACES", "NONE"]
DOMAIN_INTENTS = ("HEALTH", "TYRES", "WEATHER", "SERVICE", "DRIVING", "PLACES")
DEFAULT_W = {"EVENTS": "y", "WHY_RANGE": "14", "TREND": "14", "LARGEST": "a", "COMPARE": "a", "STATUS": "a", "BEFORE_EVENT": "a", "NONE": "a", "USAGE": "7", "SUMMARY": "7", "HEALTH": "14", "TYRES": "14", "WEATHER": "7", "SERVICE": "a", "DRIVING": "7", "PLACES": "y"}
CAPS = ("range and efficiency, energy used and distance driven, battery temperature and cooling, tyre pressures, rain and light, service and wear, driving style, ADAS and the camera, places, fault codes, "
        "charging sessions and CAN-bus security")
EXAMPLES = {
    "WHY_RANGE": "Why has my range dropped lately?", "USAGE": "How much energy did I use yesterday?", "USAGE_D": "How far did I drive since last weekend?",
    "SUMMARY": "What was the outside temperature yesterday?", "BEFORE_EVENT": "What happened before the battery overheating warning?", "EVENTS": "Show all ADAS disengagements from yesterday",
    "COMPARE": "Compare city and highway consumption", "LARGEST": "When did efficiency drop the most?", "TREND": "How has the battery temperature changed over two weeks?",
    "STATUS": "What is my estimated range now?", "CAN": "Show CAN bus attacks", "TYRES": "How are my tyre pressures?", "HEALTH": "Is my battery healthy?", "WEATHER": "Is it raining?",
    "SERVICE": "When is my next service?", "DRIVING": "How many times did I brake hard last week?", "PLACES": "Where is my car parked?", "CAMERA": "Is the camera blocked right now?"}


def _first(text, lex):
    for key, words in lex:
        if any(w in text for w in words): return key
    return None


def explicit_when(t): return nlu.parse_when(t, None)


# ------------------------------------------------------------------------------------------------ rules (fast path)
RANGE_NOW = re.compile(r"what(?:.s| is) (?:my|the) (?:current |remaining |estimated |driving )?range|(?:how.s|how is) (?:my|the) (?:driving )?range|show (?:me )?(?:my )?range|expected range|remaining range|estimated range|current range|range left|range now|range right now|distance to empty|km left|km remaining|how much range|"
                       r"range (do|can) (i|we)|how far (can|could|will) (i|we|you|it)|how many km (are )?(left|remaining)|range is left|how far (can|could) (i|we) (still )?(travel|drive|go)")
CAUSE = r"what caused|what led|led up|lead up|leading up|prior to|preceded|preceding|before|ahead of|what happened|what was going on|sequence of events|why did"
MODES = r"cit(?:y|ies)|highway|urban|town|motorway|freeway|streets"
PERIOD = r"since|yesterday|today|last|past|this week|weekend|month|week|days|so far"
CONSUME = r"\buse\b|\bused\b|\busing\b|consum|spent|spend|burn|drain|\btook\b|\btake\b|needed"
CURRENT = r"right now|currently|\bcurrent\b|at the moment|\bnow\b|\bleft\b|remaining"


# ------------------------------------------------------------------------------------------------ extended domains (tyres, weather, service, health, driving, places)
D_TYRE = re.compile(r"\btires?\b|\btpms\b|\bpressures?\b|puncture|inflat\w*|\bflat\b|\bwheels?\b|\bvalve\b|losing air")
D_WEATHER = re.compile(r"\brain\w*|drizzl\w*|\bwipers?\b|\bwet\b|\bcloud\w*|overcast|\bsunn\w*|sunshine|\bsun\b|\bdark\b|\bbright\w*|daylight|\bfog\w*|\bstorm\w*|\bsnow\w*|\bweather\b|\blux\b|\bdusk\b|\bhumid\w*|\bsky\b")
D_SERVICE = re.compile(r"how much (has|have) .{0,15}(run|driven|done|covered|travell?ed)|which car|what car|which vehicle|what vehicle|which model|car name|car model|what model|vehicle (name|model)|\bvin\b|what is this car|(?:info|information|details|facts|specs?|specifications?|anything|something) about (my|this|the) (car|vehicle)|tell me about (the|my|this) (car|vehicle)|how many km\b.{0,40}\b(in total|total|altogether|overall|on the clock)|\bpads?\b|\bworn\b|\bwear\b|\bservice\b|\bservicing\b|maintenance|maintain\w*|odometer|\bodo\b|total km|km (on|in) (the|my) (car|odometer)|brake pads?|\bbrakes?\b|brake wear|\b12 ?v\b|12 volt|auxiliary|\bsoftware\b|firmware|\bwarranty\b|inspection|\boil\b|\bmot\b")
HEALTH_Q = re.compile(r"\b(is|are|was|were|how.s|hows|how is|how are|how was)\b.{0,40}\b(ok|okay|good|fine|healthy|alright|bad|wrong|working|worn|condition|state|status|health|performing)\b|\bhealth\b|\bany (problems?|issues?|faults?|warnings?|troubles?)\b|"
                      r"anything (wrong|bad|odd|strange|to worry)|something wrong|\bwhat.s wrong\b|not (so |very )?good|not working|\bproblems?\b|\bissues?\b|\bdegrad\w*|\bsoh\b|\bcondition\b|everything (ok|okay|fine)|all (ok|okay|fine)")
HEALTH_NOUN = [("tyres", re.compile(r"\btires?\b")), ("brakes", re.compile(r"\bbrakes?\b|brake pads?")), ("camera", re.compile(r"camera|lens")), ("electrical", re.compile(r"\b12 ?v\b|12 volt|auxiliary|electrical")),
               ("motor", re.compile(r"engine|motor|drive ?unit|powertrain")), ("cooling", re.compile(r"coolant|cooling|\bpump\b")), ("battery", re.compile(r"battery|\bpack\b|\bsoh\b|state of health|charging health"))]
D_DRIVE = re.compile(r"how many (hours|minutes)\b.{0,25}\bdriv\w*|hard(ly)? brak\w*|brak\w* (too|hard|harsh\w*|a lot|often|much|gently|smooth\w*)|too (hard|harsh\w*|aggressive\w*)|harsh\w*|sudden\w*|aggressive\w*|\bsmooth\w*|driving (style|habits?)|\bhabits?\b|rate my driving|driving (score|skill)|how well (do|did) i drive|good driver|brak(e|es|ing) (hard|a lot)|"
                     r"hard accelerat\w*|\bcornering\b|\bregen\w*|how long (did|was|have) (i|we)|time (did )?(i|we) (spend|spent)|\bdrive time\b|hours? (of )?driv\w*|minutes? (of )?driv\w*|"
                     r"(lane|adas|assist|autopilot|cruise)\b.{0,25}\b(time|usage|used|use|engaged|active|percent\w*|share|long|much)|\b(time|usage|how long|how much|percent\w*|share)\b.{0,25}\b(lane|adas|assist|autopilot|cruise)")
D_PLACE = re.compile(r"which (spot|place|street|space|lot).{0,25}(park|left|leave)|which trips?\b|\btrips?\b.{0,20}\b(did i|have i|i) (make|made|take|took|do|did)\b|^where are we$|^where are we (now|right now)$|how many time(s)?\b.{0,20}\b(park|stop|drive|go|leave|left)|how many trips|where (am i|is (my|the) (car|vehicle)|did i (park|go|drive|leave)|was i|have i been|were we|is it parked|do i (usually )?go|to next)|(?:\b(?:is|was|am|are|where)\b.{0,20}\bparked\b|\bparked (?:at|in|near|where|now)\b)|\blocation\b|\bgps\b|\bposition\b|"
                     r"destination|where.s my car|which places|\btrips? to\b|\bgo(ing)? to (work|home)\b|\bheading\b|how often (do|did) i (go|visit|drive)|where to (go|next)|where am i going|where are we (going|now|parked)")


def parse_domain(t):
    """Rules for the extended domains -> (intent, params, confidence) or None. Checked before the older metric / event rules."""
    w = nlu.parse_when(t, None); p = {"metric": None, "event": None, "w": w, "assumed": w is None, "focus": None}
    if re.search(r"camera|lens", t) and re.search(r"cloud\w*|overcast|\bsunn\w*|\brain\w*|\bfog\w*|\bdark\b|\bbright\w*|weather|\bsky\b", t) and not re.search(r"blocked|blockage|dirty|obstruct|not working|working", t):
        p.update(w=w or "7", focus="now", note="camera", ask="cloud"); return "WEATHER", p, 0.9
    if re.search(r"camera|lens|blockage|\bblocked\b", t) and not re.search(r"hard brak|rate my|lane (assist|keep)", t): return None
    current = bool(re.search(CURRENT + r"|^is it\b|^are we\b|^is the\b|what.s it like|how.s the|at the minute|^what (is|.s) the|^how (is|.s) the|^what.s", t))
    past = bool(re.search(r"\b(was|were|did|has|have|had|been)\b|since|last|yesterday|today|week|month|ago|recent\w*|lately", t))
    PLACE_WORD = r"\b(home|house|work|office|workplace|school|gym|mall|supermarket|grocery|shop|store|clinic|cinema|restaurant|park|garden|friends?|airport|out of town|outstation|car|here|there|destination)\b"
    if re.search(r"\b(how far|distance|how long|how much time|eta|how close|how near|which way|direction|bearing|how many (km|miles) (is|to|away))\b", t) and re.search(PLACE_WORD, t) \
            and not re.search(r"\b(did|have|drove|driven|can|could|will|left|remaining|range|travel\w*|go on|per|battery|was|were|spent|spend|charge|charging|parked|since)\b", t):
        p.update(focus="eta" if re.search(r"how long|how much time|\beta\b|time to|get (to |there|home)|arrive|reach", t) else "distance", q=t, w="y"); return "PLACES", p, 0.93
    if re.search(r"where (is|are) (my|the) (home|house|work|office|school|gym|mall|supermarket|clinic|cinema|restaurant|park|garden|friends?|airport)|coordinates|latitude|longitude|lat ?long|\bgps\b|current location|my location", t):
        p.update(focus="coords" if re.search(r"coordinates|latitude|longitude|lat ?long|\bgps\b", t) else "distance", q=t, w="y"); return "PLACES", p, 0.93
    if re.search(r"how fast (am i|are we|is (it|the car)|was i)|speedometer|am i speeding|current speed", t):
        p.update(metric="speed", focus=None); return "STATUS", p, 0.93
    if re.search(r"which (direction|way)|(my|the car.s|our) (heading|course|direction)|\bcompass\b|am i (going|heading|driving|facing) (north|south|east|west)|what direction", t):
        p.update(focus="heading", w="y", q=t); return "PLACES", p, 0.93
    if D_PLACE.search(t) and not re.search(r"\bhome (charg|wallbox)|charging station", t):
        focus = "going" if re.search(r"going|destination|heading|where to|next", t) else "history" if re.search(r"did i (go|drive|park|leave)|was i|have i been|were we|trips? to|which places|how often|how many", t) else "here"
        p.update(focus=focus, q=t, w=w or ("7" if "how many" in t else "y")); return "PLACES", p, 0.92
    if D_TYRE.search(t) and not re.search(r"\bwheel (speed|drive)", t):
        p.update(w=w or "14"); return "TYRES", p, 0.93
    if re.search(r"(what|which|best|ideal|optimal|right)\b.{0,25}\bspeed\b.{0,45}\b(mileage|efficien\w*|range|economy|save|most|best|least energy)|best speed|economical speed|eco(nomy)? speed|speed for (best|max\w*) (range|mileage|efficiency)", t):
        p.update(w=w or "28", focus="eco"); return "DRIVING", p, 0.92
    if D_DRIVE.search(t) and not re.search(r"\bwhy\b.{0,20}\brange\b", t):
        p.update(w=w or "7", focus="adas" if re.search(r"lane|adas|assist|autopilot|cruise", t) else "duration" if re.search(r"how long|time (did )?(i|we)|hours?|minutes?|drive time", t) else "style"); return "DRIVING", p, 0.92
    if D_WEATHER.search(t) and not re.search(r"temperature|\btemp\b|how (hot|cold|warm)", t) and not re.search(r"\bwhy\b|range|consum|efficien", t):
        live = current and not (past and w) and not re.search(r"\b(was|were|did|has|have|had|been)\b", t)
        p.update(w=w or "7", focus="now" if live else "period", ask="wipers" if re.search(r"wipers?", t) else "cloud" if re.search(r"cloud\w*|overcast|sunn\w*|\bsky\b|sunshine|\bsun\b", t) else "dark" if re.search(r"dark|bright\w*|daylight|dusk", t) else "rain"); return "WEATHER", p, 0.92
    if re.search(r"maintenance|\bservice\b|servicing|odometer|\bodo\b", t) and not re.search(r"\bwhy\b|range|consum|efficien", t):
        p.update(w="a", focus="odo" if re.search(r"odometer|\bodo\b", t) else "service"); return "SERVICE", p, 0.93
    hq = HEALTH_Q.search(t); ev0 = _first(t, LEX_EVENT)
    generic = re.search(r"\b(car|vehicle|everything|anything|all|ev|problems?|issues?|faults?|wrong|health|condition|status|systems?)\b", t)
    if hq and (generic or any(rx.search(t) for _, rx in HEALTH_NOUN)) and not (ev0 in ("DTC_SET", "WARN_BATT_OVERTEMP", "ADAS_DISENGAGE", "CAN_ANOMALY", "CHARGE_START") and (w or re.search(r"how many|show|list|count", t))) and not re.search(r"range|consum|efficien|distance|drain|economy|mileage|charg(e|ing) (session|level)|state of charge", t):
        focus = next((k for k, rx in HEALTH_NOUN if rx.search(t)), "all")
        if focus == "tyres": p.update(w=w or "14"); return "TYRES", p, 0.93
        if focus == "brakes": p.update(focus="brakes"); return "SERVICE", p, 0.9
        p.update(w=w or "14", focus=focus, note="engine" if re.search(r"\bengine\b", t) else None); return "HEALTH", p, 0.91
    if D_SERVICE.search(t) and not re.search(r"\bwhy\b|range|consum|efficien|software update.*camera", t):
        p.update(w="a", focus="odo" if re.search(r"how much (has|have)", t) else "model" if re.search(r"which car|what car|which vehicle|what vehicle|which model|car name|car model|what model|vehicle (name|model)|\bvin\b|what is this car|about (my|this|the) (car|vehicle)|tell me about", t) else "brakes" if re.search(r"brake|\bpads?\b|\bworn\b", t) else "odo" if re.search(r"odometer|\bodo\b|total km|km (on|in)", t) else "oil" if re.search(r"\boil\b", t) else "service"); return "SERVICE", p, 0.92
    if re.search(r"\bengine\b|\bmotor\b", t) and re.search(r"good|ok|okay|fine|wrong|bad|working|health|problem|issue", t): p.update(w=w or "14", focus="motor", note="engine"); return "HEALTH", p, 0.88
    return None


def parse_fast(text):
    """-> (intent, params, confidence). Accepts raw text (normalises it) - use parse_clean() to skip normalisation."""
    return parse_clean(nlu.normalize(text)[0])


def parse_clean(t):
    dom = parse_domain(t)
    if dom: return dom
    metric, event = _first(t, LEX_METRIC), _first(t, LEX_EVENT)
    if metric is None and "temperature" in t and not re.search(r"battery|pack|coolant|motor|cabin", t): metric = "amb"
    p = {"metric": metric, "event": event, "w": None, "assumed": False}
    sc = {}
    cause_ev = event in ("WARN_BATT_OVERTEMP", "DTC_SET")
    if RANGE_NOW.search(t) and not re.search(r"\bwhy\b|reason|explain|dropped|reduced", t): sc["STATUS"] = 0.93; p["metric"] = "range"
    if re.search(r"camera|lens", t) and re.search(CURRENT + r"|is the (front )?camera|how is the (front )?camera|camera (ok|okay|working|status|clear|clean|not working|broken|faulty|failed|dirty|blocked)|(which|what) camera|"
                                                 r"(is|are) (not )?(working|ok|okay|clean|clear|dirty|blocked|broken)|not working", t) and not re.search(r"since|yesterday|today|last|week|how many|history|did |was ", t):
        sc["STATUS"] = 0.93; p["metric"] = "camera"
    if re.search(r"what (can|do|am|an|are) (you|i|we) see|what (a|i)m i seeing|what an i seeing", t): sc["STATUS"] = 0.9; p["metric"] = "camera"
    if re.search(r"\b(has|have|did|is|does|was|had)\b.{0,30}\b(range|efficien\w*|consumption|mileage)\b.{0,30}\b(drop\w*|fall\w*|fell|reduc\w*|decreas\w*|worse\w*|declin\w*|shrunk|lower\w*|gone down|dip\w*|got shorter)", t) and "WHY_RANGE" not in sc and not re.search(r"largest|biggest|worst|\bwhen\b|\bmost\b|which day|what day|\bthan\b|versus|\bvs\b|compare", t):
        sc["WHY_RANGE"] = 0.9
    if p["metric"] == "camera": p["ask"] = "which" if re.search(r"(which|what) camera", t) else "see" if re.search(r"\bsee\b|seeing", t) else None
    if p["metric"] == "lane_q" and re.search(r"how|what|average|quality|confidence|during|yesterday|today|last|week|drive|trip", t) and not re.search(CURRENT, t): sc["SUMMARY"] = 0.92
    if re.search(r"how much\b.{0,20}\b(did|have|has)\b.{0,12}\b(charge|charged|recharge|recharged|plug\w*)\b|how (often|many times) (did|have|do) (i|we) (charge|recharge|plug)|charging (history|sessions?)", t) and not re.search(r"state of charge|charge (left|level)|battery (left|level)|range", t):
        sc["EVENTS"] = 0.93; p["event"] = "CHARGE_START"; event = "CHARGE_START"
    if re.search(CAUSE, t) and cause_ev and not re.search(r"how many|count|list|show all", t): sc["BEFORE_EVENT"] = 0.95
    elif re.search(r"before|leading up|prior to|preceded|what happened|led to|cause of|what caused", t) and event: sc["BEFORE_EVENT"] = 0.9
    if len(re.findall(MODES, t)) >= 2 and re.search(r"compare|versus|\bvs\b|difference|differ|which|more|less|worse|better|cheaper|than", t): sc["COMPARE"] = 0.93
    if (re.search(r"largest|biggest|worst|highest|steepest|sharpest|lowest|fell off|day over day|compared to the day before|spike|jump", t) or
            (re.search(r"\bwhen\b", t) and re.search(r"drop|fall|decline|worsen|deteriorat|increase|spike|jump|worst|lowest", t))) \
            and not re.search(r"\bwhy\b|what changed", t) and metric in ("energy_use", None) and not RANGE_NOW.search(t): sc["LARGEST"] = 0.9
    if re.search(r"\bwhy\b|reason|what.s causing|explain|what changed|what.s behind|what is behind|what.s going on|how come", t) and metric in ("energy_use", None) \
            and re.search(r"range|consum|efficien|reduc|drop|drain|lower|worse|shorter|thirst|mileage|distance|power per|tanked|fewer|less|increase", t): sc["WHY_RANGE"] = 0.93
    if re.search(r"show|list|how many|count|any|all|did|were there|events|has the|have there|\bwas\b|\bwere\b|\bever\b|\bbeen\b", t) and event and "BEFORE_EVENT" not in sc and "STATUS" not in sc: sc["EVENTS"] = 0.9
    if re.search(r"trend|changed|over the last|compared to|how has|how did|has the|been running|hotter|colder", t) and metric and not ({"WHY_RANGE", "LARGEST", "COMPARE"} & set(sc)): sc["TREND"] = 0.85
    no_change = not re.search(r"change|trend|compared|versus|\bwhy\b", t)
    if (re.search(r"how much|how many|total|what (was|is|were)|what.s|tell me|my ", t) and re.search(r"fuel|energy|power|kwh|electric|charge|battery", t) and re.search(CONSUME, t)
            and not re.search(CURRENT, t) and no_change and not ({"WHY_RANGE", "LARGEST", "COMPARE", "BEFORE_EVENT"} & set(sc))): sc["USAGE"] = 0.92
    if re.search(r"how far (did|have|was)|how many km|distance|driven|\bdrove\b|travell?ed|mileage|\bkm\b", t) and re.search(PERIOD, t) and no_change \
            and not ({"WHY_RANGE", "LARGEST", "COMPARE", "STATUS"} & set(sc)): sc["USAGE"] = max(sc.get("USAGE", 0), 0.9)
    if re.search(r"(average|avg|overall) (consumption|efficiency|mileage|energy)|my (consumption|efficiency|mileage)", t) and not re.search(PERIOD + r"|why|compare|change", t) and not sc: sc["USAGE"] = 0.85
    if re.search(r"how was|how were|what was|what were|what.s|whats|average|avg|maximum|\bmax\b|minimum|\bmin\b|highest|lowest|hottest|coldest|how hot|how cold|how warm", t) \
            and metric in ("amb", "speed", "batt_temp", "coolant_flow", "hvac") and not re.search(CURRENT, t) and not re.search(CAUSE, t) and not ({"LARGEST", "COMPARE", "TREND", "WHY_RANGE", "BEFORE_EVENT"} & set(sc)): sc["SUMMARY"] = 0.9
    if re.search(r"current|right now|\bnow\b|at the moment|what is|what.s my|how much|how fast", t) and metric and not sc: sc["STATUS"] = 0.85
    if not sc: return "NONE", p, 0.2
    intent = max(sc, key=sc.get); conf = sc[intent]
    if len(sc) > 1 and sorted(sc.values())[-1] - sorted(sc.values())[-2] < 0.04: conf -= 0.2          # ambiguous between intents
    if intent in ("BEFORE_EVENT", "EVENTS") and not event: conf -= 0.4
    if event == "CHARGE_START" and metric in ("batt_temp", "coolant_flow") or re.search(r"strange|odd|unusual|weird|wrong", t) and intent == "EVENTS": conf -= 0.3     # probably a cause question
    w = nlu.parse_when(t, None)
    p["assumed"] = w is None
    p["w"] = w or ("a" if event == "CAN_ANOMALY" else "7" if (event or "").startswith("CAMERA") and intent == "EVENTS" else DEFAULT_W.get(intent, "14"))
    if intent == "WHY_RANGE": p["metric"] = "energy_use"
    if intent == "LARGEST" and not p["metric"]: p["metric"] = "energy_use"
    if intent == "SUMMARY" and not p["metric"]: p["metric"] = "amb"
    if intent == "SUMMARY": p["emph"] = "max" if re.search(r"\b(max|maximum|highest|top|fastest|hottest|peak)\b", t) else "min" if re.search(r"\b(min|minimum|lowest|slowest|coldest)\b", t) else None
    if intent == "SUMMARY" and w is None and re.search(r"\b(what.s|whats|what is|how (hot|cold|warm) is)\b", t) and not re.search(r"average|avg|\bmax|\bmin|highest|lowest|hottest|coldest|\bwas\b|\bwere\b", t): intent = "STATUS"
    return intent, p, conf


def domain_focus(intent, t):
    if intent == "HEALTH": return next((k for k, rx in HEALTH_NOUN if rx.search(t)), "all")
    if intent == "WEATHER": return "now" if re.search(CURRENT + r"|^is it\b", t) else "period"
    if intent == "PLACES": return "going" if re.search(r"going|destination|heading|next", t) else "history" if re.search(r"did i|was i|been|went|trips?", t) else "here"
    if intent == "DRIVING": return "adas" if re.search(r"lane|adas|assist", t) else "style"
    return None


FAST_MIN = 0.8
DOMAIN = re.compile(r"can bus|intrusion|attack|inject|flood|spoof|fuzz|car\b|vehicle|battery|pack|range|charg|energy|consum|efficien|mileage|fuel|economy|adas|lane|assist|autopilot|cruise|fault|dtc|error code|diagnos|"
                    r"coolant|cooling|pump|temp|thermal|overheat|speed|highway|motorway|freeway|city|urban|town|driv|drove|trip|warning|alert|alarm|hvac|heater|heating|climate|camera|lens|windscreen|blurr|glare|blockage|"
                    r"\bkm\b|\bwh\b|kwh|disengage|thirst|\bsoc\b|weather|distance|how far|travel|outside|recharge|"
                    r"tire|tyre|pressure|puncture|inflat|\bflat\b|rain|wiper|cloud|overcast|sunny|\bdark\b|\bwet\b|\bsee\b|seeing|wheel|trouble|\bworn\b|\bwear\b|service|maint|odometer|brake|\bpads?\b|health|problem|issue|wrong|where|\bpark|school|\bgym\b|\bmall\b|supermarket|grocery|\boffice\b|clinic|cinema|restaurant|airport|\beta\b|how long|direction|heading|compass|which way|how fast|speedometer|\bgps\b|coordinates|latitude|longitude|location|\bhome\b|\bwork\b|gps|harsh|smooth|aggress|regen|engine|motor|12 ?v|software|\boil\b|\bcondition|shape")


def in_domain_clean(t): return bool(DOMAIN.search(t))


def in_domain(text): return in_domain_clean(nlu.normalize(text)[0])


# ------------------------------------------------------------------------------------------------ SLM slot filler
WORDS_GRAMMAR = r"""root ::= task " " intent " " metric " " event
task ::= "data" | "define" | "estimate" | "capability" | "advice" | "other"
intent ::= "why" | "before" | "events" | "compare" | "largest" | "trend" | "now" | "usage" | "average" | "maximum" | "minimum" | "duration" | "other"
metric ::= "energy" | "battery" | "motor" | "coolant" | "speed" | "charge" | "chargepower" | "torque" | "outside" | "climate" | "range" | "tyre" | "weather" | "service" | "health" | "driving" | "place" | "vehicle" | "none"
event ::= "adas" | "overheat" | "fault" | "charging" | "camera" | "none" """
from .semantic_frame import SemanticFrame, TASK as W_TASK, OPERATION as W_OPERATION, CONCEPT as W_METRIC, EVENT as W_EVENT, LEGACY_CONCEPT
W_INTENT = {token: SemanticFrame.from_tokens("data", token, "none", "none").legacy_intent for token in W_OPERATION}
SYSTEM_W = """Map a vehicle request to exactly four ontology words: task intent metric event. Do not answer it.
task: data (read recorded/live values) | define (explain a term) | estimate (derive a future quantity) | capability (ask whether a value exists or why it is unavailable) | advice (recommend, buy or change something) | other
intent: why | before | events | compare | largest | trend | now | usage | average | maximum | minimum | duration | other
metric: energy | battery (battery temperature) | motor | coolant | speed | charge (state of charge) | chargepower (charging rate) | torque | outside | climate | range | tyre | weather | service | health | driving | place | vehicle | none
event: adas | overheat | fault | charging | camera | none
Q: Explain regenerative braking.
define other driving none
Q: Give the meaning of adaptive cruise control.
define other driving adas
Q: Give me the mean cabin temperature for yesterday.
data average climate none
Q: What peak road speed was recorded this week?
data maximum speed none
Q: Estimate the remaining time until the battery is full.
estimate duration charge none
Q: How many minutes would a refill from the present battery level require?
estimate duration charge none
Q: Why is wheel torque not available?
capability other torque none
Q: Can this data source measure shaft torque?
capability other torque none
Q: Recommend replacement wheels.
advice other tyre none
Q: Where should I purchase winter tyres?
advice other tyre none
Q: List diagnostic alerts from last month.
data events none fault
Q: What is the traction motor status?
data now motor none
Q: Report the electrical power entering the battery right now.
data now chargepower charging
Q: How did ambient temperature change?
data trend outside none
Q: Is urban driving more efficient than motorway driving?
data compare energy none
Q: Contrast urban and highway operation.
data compare none none
Q: Tell me a joke.
other other none none"""


class Slm:
    """Slot-filling SLM: 3 grammar-forced words, ~4 output tokens. Returns per-field probabilities so callers can refuse low-confidence parses."""
    def __init__(self, path, threads=2, n_ctx=896, mmap=True, fmt="words", state_path=None, lora_path=None, n_batch=128):
        from llama_cpp import Llama, LlamaGrammar
        t = time.perf_counter()
        # A 128-token batch keeps the complete process comfortably below 2 GB
        # while avoiding dozens of tiny ARM prompt-evaluation batches.  The
        # persistent state below prevents re-evaluating the shared ontology
        # prompt after an idle unload or process restart.
        self.n_batch = int(n_batch)
        kw = dict(model_path=path, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads, n_batch=self.n_batch, n_ubatch=self.n_batch, use_mmap=mmap, verbose=False)
        if lora_path: kw["lora_path"] = lora_path
        self.llm = Llama(**kw)
        self.fmt = "words"; self.grammar = LlamaGrammar.from_string(WORDS_GRAMMAR.strip(), verbose=False)
        if lora_path and "semantic-lora" in os.path.basename(lora_path):
            from training.semantic_data import SYSTEM
            self.system = SYSTEM
        else: self.system = SYSTEM_W
        self.load_s = time.perf_counter() - t; self.path = path; self.lora_path = lora_path; self.warm = False; self.state_path = state_path; self.state_source = None

    def _state_key(self):
        import hashlib
        def identity(path):
            if not path: return None
            h = hashlib.sha256(); size = os.path.getsize(path); h.update(str(size).encode())
            with open(path, "rb") as fh:
                h.update(fh.read(1 << 20))
                if size > 2 << 20: fh.seek(-(1 << 20), 2); h.update(fh.read(1 << 20))
            return h.hexdigest()
        return {"model": identity(self.path), "lora": identity(self.lora_path), "prompt": hashlib.sha256(self.system.encode()).hexdigest(), "n_ctx": self.llm.n_ctx(), "n_batch": self.n_batch, "llama_cpp": __import__("llama_cpp").__version__}

    def warmup(self):
        """Prefill the shared system prompt. With a state file this is a ~30 MB load instead of re-processing ~800 prompt tokens on every boot:
        the state is built ONCE on the target itself (first boot), persisted to flash, and reloaded afterwards (guarded by a model/prompt/version key)."""
        import pickle
        if self.state_path and os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f: d = pickle.load(f)
                if d["key"] == self._state_key(): self.llm.load_state(d["state"]); self.warm = True; self.state_source = "loaded"; return
            except Exception: pass                                   # corrupt / stale file -> rebuild below
        self.parse("warmup question about my car"); self.warm = True; self.state_source = "built"
        if self.state_path:
            try:
                tmp = self.state_path + ".tmp"
                with open(tmp, "wb") as f: pickle.dump({"key": self._state_key(), "state": self.llm.save_state()}, f, protocol=4)
                os.replace(tmp, self.state_path)
            except OSError: pass

    def close(self):
        import gc
        self.llm = None; gc.collect()

    def parse(self, text):
        """-> (intent, slots{metric,event}, info{ms, out_tokens, raw, pmax}). pmax = lowest top-token probability over the generation steps (cheap confidence signal;
        read through a logits callback, so no vocabulary-sized logprob buffer is allocated)."""
        from llama_cpp import LogitsProcessorList
        import numpy as np
        steps = []
        def watch(input_ids, scores):
            m = float(np.max(scores)); z = float(np.sum(np.exp(scores - m))); steps.append(1.0 / z); return scores      # softmax probability of the argmax token
        t = time.perf_counter()
        r = self.llm.create_chat_completion(messages=[{"role": "system", "content": self.system}, {"role": "user", "content": "Q: " + text}], grammar=self.grammar,
                                            max_tokens=10, temperature=0.0, logits_processor=LogitsProcessorList([watch]))
        ch = r["choices"][0]; out = ch["message"]["content"].strip(); u = r.get("usage", {})
        # The grammar emits exactly four ontology words.  llama.cpp exposes a
        # score for every emitted token; retain the conservative minimum for
        # each semantic slot rather than marking metric/event confidence as
        # unknown and then silently preventing the model from contributing
        # those slots at all.
        pmin = min(steps) if steps else None
        info = {"ms": (time.perf_counter() - t) * 1000, "out_tokens": u.get("completion_tokens"), "raw": out, "pmax": pmin,
                "p": {"intent": steps[0] if steps else None, "metric": pmin, "event": pmin}}
        words = out.split()
        if len(words) != 4 or words[0] not in W_TASK or words[1] not in W_OPERATION or words[2] not in W_METRIC or words[3] not in W_EVENT:
            return "NONE", {"task": "other", "metric": None, "event": None, "operator": None}, info
        frame = SemanticFrame.from_tokens(*words)
        return frame.legacy_intent, frame.as_dict(), info


class LazySlm:
    """Loads the model on the first question that needs it and releases it after `idle_s` of silence, so the ~0.5 GB goes back to the vehicle OS.
    Drop-in for Slm (parse / warmup). A daemon timer performs the idle release;
    tick() remains as a cheap opportunistic check after questions."""
    def __init__(self, path, idle_s=120.0, clock=time.monotonic, **kw):
        import threading
        self.path, self.idle_s, self.kw, self.clock = path, idle_s, kw, clock
        self.slm, self.last, self.loads, self.load_s, self.warm_s = None, 0.0, 0, 0.0, 0.0
        self._lock, self._timer = threading.RLock(), None

    @property
    def loaded(self): return self.slm is not None

    def _ensure(self):
        if self.slm is None:
            t = time.perf_counter(); self.slm = Slm(self.path, **self.kw); self.load_s = time.perf_counter() - t
            t = time.perf_counter(); self.slm.warmup(); self.warm_s = time.perf_counter() - t; self.loads += 1

    def warmup(self): self._ensure()

    def _arm(self):
        import threading
        if self._timer is not None: self._timer.cancel()
        self._timer = threading.Timer(self.idle_s, self.tick); self._timer.daemon = True; self._timer.start()

    def parse(self, text):
        with self._lock:
            self._ensure(); self.last = self.clock()
            i, p, info = self.slm.parse(text); info["cold_load_s"] = round(self.load_s + self.warm_s, 2) if self.loads and self.last and info.get("first") else None
            self._arm()
            return i, p, info

    def choose(self, system, user, options, max_tokens=8):
        """One grammar-constrained choice among `options` (the answer can only be one of them). -> (option, seconds). The same loaded model, the same idle release."""
        from llama_cpp import LlamaGrammar
        with self._lock:
            self._ensure(); self.last = self.clock(); t = time.perf_counter()
            g = LlamaGrammar.from_string("root ::= " + " | ".join('"%s"' % o for o in options), verbose=False)
            prompt = "<|im_start|>system" + chr(10) + system + "<|im_end|>" + chr(10) + "<|im_start|>user" + chr(10) + user + "<|im_end|>" + chr(10) + "<|im_start|>assistant" + chr(10)
            out = self.slm.llm.create_completion(prompt, grammar=g, max_tokens=max_tokens, temperature=0.0)
            self._arm(); return out["choices"][0]["text"].strip(), time.perf_counter() - t

    def tick(self):
        with self._lock:
            if self.slm is not None and self.clock() - self.last >= self.idle_s: self.unload()

    def unload(self):
        with self._lock:
            if self._timer is not None: self._timer.cancel(); self._timer = None
            if self.slm is not None: self.slm.close(); self.slm = None


SLM_MIN_INTENT, SLM_MIN_SLOT = 0.55, 0.55
TOPIC = re.compile(r"range|consum|efficien|energy|fuel|mileage|economy|thirst|distance|\bkm\b|city|highway|town|motorway|urban|freeway|drove|driv|power|kwh|charge|battery")


DOM_GROUND = {"TYRES": re.compile(r"air|wheel|tire|tyre|pressure|flat|puncture|rubber"), "WEATHER": re.compile(r"weather|rain|sun|dark|light|cloud|wet|outside|sky|bright|fog|dry|storm|wiper"),
              "SERVICE": re.compile(r"servic|maint|odometer|brake|pad|battery|12|update|software|due|km|wear|oil"), "HEALTH": re.compile(r"car|vehicle|battery|pack|brake|motor|engine|condition|shape|ok|good|wrong|fine|health|problem|issue|fault|bad"),
              "DRIVING": re.compile(r"driv|brak|accel|smooth|style|habit|lane|regen|time|hard|gentle|aggress|assist"), "PLACES": re.compile(r"where|park|location|home|work|trip|go|went|leave|left|spot|place|been")}


def accept_slm(intent, sp, info, rules_p, t):
    """Grounding + confidence gate for the SLM's parse. Returns (ok, reason)."""
    if intent == "NONE": return True, "other"
    if intent in DOMAIN_INTENTS:
        pr = info.get("p") or {}
        if pr.get("intent") is not None and pr["intent"] < SLM_MIN_INTENT: return False, "model unsure of intent"
        return (True, "ok") if DOM_GROUND[intent].search(t) else (False, "no vehicle topic in the question")
    pr = info.get("p") or {}
    if pr.get("intent") is not None and pr["intent"] < SLM_MIN_INTENT: return False, "model unsure of intent"
    ev = rules_p["event"] or (sp["event"] if (pr.get("event") or 0) >= SLM_MIN_SLOT else None)
    me = rules_p["metric"] or (sp["metric"] if (pr.get("metric") or 0) >= SLM_MIN_SLOT else None)
    if intent in ("EVENTS", "BEFORE_EVENT") and not ev: return False, "no event grounded"
    if intent in ("TREND", "STATUS", "SUMMARY") and not me: return False, "no measurement grounded"
    if intent in ("WHY_RANGE", "COMPARE", "USAGE", "LARGEST") and not TOPIC.search(t): return False, "no vehicle topic in the question"
    return True, "ok"


def suggestions(t, n=3):
    from .qa_fuzzy import related
    return related(t, n)


def suggestions_old(t, n=2):
    toks = set(re.findall(r"[a-z]{3,}", t)); scored = []
    for k, ex in EXAMPLES.items():
        et = set(re.findall(r"[a-z]{3,}", ex.lower())); scored.append((len(toks & et) / max(1, len(toks | et)), ex))
    scored.sort(reverse=True); return [ex for sc, ex in scored[:n]]


# ------------------------------------------------------------------------------------------------ compose
def _fmt_items(items, n=6, dated=False):
    return "; ".join(f"{i['time'][5:] if dated else i['time'][11:]} {i['detail']}" for i in items[:n]) + (f"; +{len(items) - n} more" if len(items) > n else "")



# ------------------------------------------------------------------------------------------------ compose: extended domains
def _gps_tail(h):
    t = ", live fix" if h.get("fix_age_s", 999) <= 3 and h.get("src") == "gps" else f", {h['time'][11:]}"
    if h.get("moving") and h.get("heading_dir"): t += f", heading {h['heading_dir']} at {h['speed']:.0f} km/h"
    return t


def _plural(n, w): return f"{n} {w}{'' if n == 1 else 's'}"
def _hm(minutes): h, m = divmod(int(minutes), 60); return f"{h} h {m:02d} min" if h else f"{m} min"
RAIN_TXT = {0: "no rain", 1: "light rain", 2: "moderate rain", 3: "heavy rain"}


def _sky(lg): return {"dark": "it was dark (night)", "dusk or dawn": "it was dusk or dawn", "dim": "dim light"}.get(lg, f"{lg} skies")


NO_DATA = {"PLACES": "I couldn't answer that - this vehicle's data has no GPS positions or saved places, so I can't say where it is or where it has been.", "WEATHER": "I couldn't answer that - this vehicle has no rain or light sensor readings, so I can't say what the weather was.",
           "SERVICE": "I couldn't answer that - this vehicle has no service, odometer or brake-wear records, so I can't say when a service is due.", "DRIVING": "I couldn't answer that - hard braking, acceleration and cornering are not recorded for this vehicle, so I can't describe the driving style.",
           "TYRES": "I couldn't answer that - this vehicle does not report tyre pressures, so I can't check the tyres."}


def no_data(intent, f):
    """A plain sentence for 'this car's data set does not have that' instead of a database message."""
    why = str(f.get("why", "no result"))
    if "no such data" in why and intent in NO_DATA: return NO_DATA[intent]
    return f"I couldn't answer that from the vehicle data: {why}."


def compose_domain(intent, p, f, persona="driver"):
    eng = persona == "engineer"
    if not f.get("ok"): return no_data(intent, f)
    wt = A.WHEN_TEXT.get(f.get("when") or p.get("w"), "")
    if intent == "TYRES":
        ws = f["wheels"]; bad = [w for w in ws if w["name"] in f["active_warnings"]]; soft = [w for w in ws if w["status"] != "ok" and w["name"] not in f["active_warnings"]]
        if bad: s = "Tyre-pressure warning: " + "; ".join(f"the {w['name']} tyre is low at {w['kpa']} kPa ({w['kpa'] / 100:.1f} bar, spec about {w['spec']} kPa)" for w in bad) + ". Please check it for a puncture and inflate it."
        elif soft: s = "Pressures are a little low on the " + " and ".join(w["name"] for w in soft) + " tyre" + ("s" if len(soft) > 1 else "") + "."
        else: s = "All four tyres look fine." if len(ws) == 4 else f"The {len(ws)} tyre{'s' if len(ws) != 1 else ''} with a recorded pressure look fine."
        U = p.get("units"); pv = lambda k: dialogue.pressure(k, U); pu = dialogue.pressure_unit(U)
        s += f" Last reading {f['time'][11:]}: " + ", ".join(f"{w['name']} {pv(w['kpa'])}" for w in ws) + f" {pu} at 20 C (spec {(pv(ws[0]['spec']) + ' front / ' + pv(ws[2]['spec']) + ' rear') if len(ws) == 4 else ', '.join(w['name'] + ' ' + pv(w['spec']) for w in ws)}" + ("; raw readings " + ", ".join(pv(w['raw_kpa']) for w in ws) + f" {pu} at the tyres' own temperature)." if eng else ").")
        if f.get("leak"):
            lk = f["leak"]; s += f" The {lk['wheel']} tyre has been losing about {-lk['kpa_per_day']:.1f} kPa a day this week" + (f"; at that rate it reaches the warning level in " + ("less than a day." if lk["days_to_warning"] < 1 else f"about {lk['days_to_warning']:.0f} days.") if lk["days_to_warning"] else " and is already at the warning level.")
        if f["events"] and (eng or not bad): s += f" {wt.capitalize()}: " + "; ".join(f"{e['time'][5:]} {e['detail']}" for e in f["events"][-4:]) + "."
        if f.get("not_recorded"): s += " The " + " and ".join(f["not_recorded"]) + " tyre pressure is not recorded for this vehicle, so it is not covered."
        return s
    if intent == "WEATHER":
        pre = "The camera is only monitored for health, not scene understanding, so I use the light sensor instead. " if p.get("note") == "camera" else ""
        if f["mode"] == "now":
            lg = f.get("light"); ask = p.get("ask"); ratio = f" (about {round(100 * f['sky_ratio'])}% of the light a clear sky would give)" if f.get("sky_ratio") is not None else ""
            sky_s = f"Sky: {lg}{ratio}." if lg else ""; rain_s = f"Rain: {RAIN_TXT.get(f['rain'], 'rain')}."
            stale = "" if f["fresh"] else f"The car only senses rain and light while it is in use and it is not being driven now; its last reading was at {f['time'][11:]} ({_hm(f['age_min'])} ago). "
            if ask == "wipers": pre += "I can't test the wiper motor itself, only what the rain sensor and wiper state report. "
            body = " ".join(x for x in ((sky_s, rain_s) if ask in ("cloud", "dark") else (rain_s, sky_s)) if x)
            return pre + stale + body + (f" Outside temperature {f['amb_c']} C." if f.get("amb_c") is not None else "")
        if f["rain_min"] == 0:
            s = f"No rain was detected while the car was in use {wt} ({_hm(f['active_min'])} of driving and charging on {_plural(f['days_with_use'], 'day')})."
        else:
            s = (f"{wt.capitalize()} the wipers/rain sensor registered rain for {_hm(f['rain_min'])} of the {_hm(f['active_min'])} the car was in use ({100 * f['rain_min'] / f['active_min']:.0f}%), on {f['rain_days']} of {_plural(f['days_with_use'], 'day')}"
                 + (f"; {_hm(f['heavy_min'])} of it heavy" if f["heavy_min"] else "") + ".")
        if f.get("sky_ratio_avg") is not None:
            r = f["sky_ratio_avg"]; s += f" In daylight the sky was on average {'sunny' if r >= 0.7 else 'partly cloudy' if r >= 0.35 else 'overcast' if r >= 0.12 else 'heavily overcast'} (about {round(100 * r)}% of clear-sky light)."
        if f.get("amb_avg") is not None: s += f" Average outside temperature {f['amb_avg']} C."
        return s + " (The car can only sense rain and light while it is switched on.)"
    if intent == "SERVICE":
        fo = p.get("focus"); parts = []
        if fo == "model":
            if f.get("legacy"): return f"This is a {f['model']}."
            bits = [f"battery {f['cap_kwh']:.0f} kWh" + (f" ({f['soh']}% health)" if f.get("soh") else "")]
            if f.get("home_kw"): bits.append(f"home charger {f['home_kw']:.1f} kW")
            if f.get("tyre_front"): bits.append(f"tyres {f['tyre_front']:.0f} kPa front / {f['tyre_rear']:.0f} kPa rear")
            if f.get("sw"): bits.append(f"software {f['sw']}")
            return f"This is a {f['model']}" + (f", based in {f['city']}" if f.get("city") else "") + ": " + ", ".join(bits) + "."
        if fo == "oil": parts.append("This is an electric car, so there is no engine oil to change.")
        if fo == "odo": return f"The odometer reads {f['odo_km']:,} km (as of {f['time'][11:]})."
        if fo == "brakes":
            if f.get("pad_pct") is None: return "I don't have brake-wear data for this vehicle."
            return (f"The brake pads have about {f['pad_pct']}% left" + (" - plan a replacement soon" if f["pad_pct"] < 20 else " - plenty of life") + ". Because the car recovers energy with its motor when slowing, the friction brakes wear slowly.")
        parts.append(f"Odometer {f['odo_km']:,} km.")
        if "km_to_service" in f:
            if f["due"]: parts.append(f"Your service is due now: it is {f['km_since_service']:,} km since the last one (interval {f['service_interval_km']:,} km).")
            else: parts.append(f"Next service in about {f['km_to_service']:,} km (interval {f['service_interval_km']:,} km" + (f", last one {f['months_since']:.0f} months ago, so by date in about {max(0, f['months_to_service']):.0f} months" if f.get("months_to_service") is not None else "") + ").")
        extra = []
        if f.get("pad_pct") is not None: extra.append(f"brake pads about {f['pad_pct']}%")
        if f.get("v12") is not None: extra.append(f"12 V battery {f['v12']} V")
        if f.get("soh") is not None: extra.append(f"traction battery health {f['soh']}%")
        if f.get("sw"): extra.append(f"software {f['sw']}")
        if extra: parts.append("Wear and condition: " + ", ".join(extra) + ".")
        return " ".join(parts)
    if intent == "HEALTH":
        it = f["items"]; pr, at = f["problems"], f["attention"]; fo = f.get("focus"); prim = {"battery": {"battery"}, "motor": {"motor"}}.get(fo)
        if prim:
            P = [x for x in it if x["area"] in prim]; R = [x for x in it if x["area"] not in prim and (eng or x["level"] != "ok")]; nm = "The battery" if fo == "battery" else "The drive motor"
            mk = {"problem": "PROBLEM", "attention": "watch", "ok": "ok"}; ln = lambda x: f"\n  - {x['text']} [{mk[x['level']]}]"
            head = (f"{nm} itself has a problem." if any(x["level"] == "problem" for x in P) else f"{nm} itself looks fine" + (", but something that affects it needs attention." if R else "."))
            pre_ = "This is an electric car, so there is no engine - the traction battery and the drive motor are what I monitor. " if p.get("note") == "engine" else ""
            return pre_ + head + "".join(ln(x) for x in P) + ("\n  Related:" + "".join(ln(x) for x in R) if R else "")
        note = "This is an electric car, so there is no engine - the traction battery, drive motor and cooling are what I monitor. " if p.get("note") == "engine" else ""
        head = ("Everything I can see looks fine." if not pr and not at else (f"{_plural(pr, 'problem')} need{'s' if pr == 1 else ''} attention" if pr else "") + (" and " if pr and at else "") + (f"{_plural(at, 'item')} worth watching" if at else "") + ".")
        if head[0].islower(): head = head[0].upper() + head[1:]
        shown = it if eng else [x for x in it if x["level"] != "ok"][:5] or it[:3]
        mark = {"problem": "PROBLEM", "attention": "watch", "ok": "ok"}
        s = head + "".join(f"\n  - {x['area']}: {x['text']} [{mark[x['level']]}]" for x in shown)
        if not eng and not pr and not at: s = head[:-1] + ": " + "; ".join(x["text"] for x in it[:4]) + "."
        return note + s
    if intent == "DRIVING":
        fo = p.get("focus")
        if fo == "eco":
            b, w_ = f["best"], f["worst"]; band = lambda x: f"{x['lo']} km/h to {x['hi']} km/h" if x["hi"] < 200 else f"above {x['lo']} km/h"
            tab = "; ".join(f"{band(x)} {x['wh_km']} Wh/km" for x in f["bins"])
            return (f"From your car's own data ({f['basis']}), the best speed for mileage is {band(b)}: {b['wh_km']} Wh/km. The most costly band is {band(w_)} at {w_['wh_km']} Wh/km, {f['pct']}% more per km. All bands: {tab}. "
                    + ("(Per-minute averages include stop-and-go driving, so slow bands look better than steady cruising would.) " if "per-minute" in f["basis"] else "") + "Air drag grows with the square of speed, so easing off a little on the motorway saves the most.")
        lead = f"{wt.capitalize()}: {_plural(f['trips'], 'trip')}, {f['km']} km in {_hm(f['minutes'])}"
        if fo == "duration": return lead + f" ({f['hwy_km']} km on the motorway)."
        s = lead + "."
        if fo == "adas":
            if f["adas_share"] is None: return s + " Lane assist was not used on the motorway in that period."
            return s + f" Lane assist was active for {_hm(f['adas_min'])} of {_hm(f['hwy_min'])} of motorway driving ({f['adas_share']}%), with {_plural(f['adas_disengage'], 'disengagement')}."
        if f.get("has_ext"):
            bl = f["brake_100"]; al = f["accel_100"]
            s += f" Hard braking {f['brake']} time{'s' if f['brake'] != 1 else ''}" + (f" ({bl} per 100 km; typical is {AX.TYPICAL_BRAKE_PER_100KM[0]:.0f}-{AX.TYPICAL_BRAKE_PER_100KM[1]:.0f})" if bl is not None else "") + f", hard acceleration {f['accel']}, hard cornering {f['corner']}."
            if f.get("regen_share") is not None: s += f" Regenerative braking recovered {f['regen_kwh']} kWh ({f['regen_share']}% of the energy used)."
            if f.get("adas_share") is not None: s += f" Lane assist covered {f['adas_share']}% of motorway time."
            if bl is not None: s += " I don't give a driving score, but by these numbers your driving was " + ("calm." if bl <= AX.TYPICAL_BRAKE_PER_100KM[0] and (al or 0) <= 4 else "typical." if bl <= AX.TYPICAL_BRAKE_PER_100KM[1] * 1.5 else "on the firm side.")
        else: s += " This vehicle does not record hard-braking or acceleration events."
        return s
    if intent == "PLACES" and f["focus"] in ("distance", "eta", "coords"):
        h = f["here"]; fmt = lambda la, lo: f"{abs(la):.4f} {'N' if la >= 0 else 'S'}, {abs(lo):.4f} {'E' if lo >= 0 else 'W'}"
        where = (f"at {h['near']}" if h["at"] else f"about {h['km']:.1f} km {h['compass']} of {h['near']}") + (f" in the {h['area']} area" if h.get("area") else "")
        lead = ("The car is on the move, " if h["moving"] else "The car is parked ") + where + f" (GPS {fmt(h['lat'], h['lon'])}{_gps_tail(h)})."
        out = []
        if f.get("car_from_me"): out.append("I only know where the car is - I have no way to locate you.")
        if f.get("pair"):
            pr = f["pair"]; out.append(f"{pr['a']} to {pr['b']} is {pr['km']:.1f} km in a straight line, about {pr['road_km']:.1f} km by road" + (f"; your {pr['typical']['n']} earlier trips between them took about {pr['typical']['min']} min." if pr["typical"] else f" - roughly {pr['eta_min']} min at your usual speeds."))
        for t in f.get("targets", []):
            if t["here_is_it"]: out.append(f"The car is at {t['name']} right now."); continue
            s_ = f"{t['name']} is {t['km']:.1f} km {t['compass']} of the car in a straight line (about {t['road_km']:.0f} km by road)"
            s_ += (f"; your {t['typical']['n']} earlier trips from {h['near']} took about {t['typical']['min']} min." if t["typical"] else f" - roughly {t['eta_min']} min at your usual speeds.")
            if f["focus"] == "coords" or re.search(r"\bwhere\b", p.get("q") or ""): s_ += f" Its position is {fmt(t['lat'], t['lon'])}."
            out.append(s_)
        if not f.get("pair") and not f.get("targets"):
            out.append("Nearest saved places: " + ", ".join(f"{n['name']} {n['km']} km {n['compass']}" for n in f["nearest"]) + ".")
            if f["focus"] != "coords": out.append("I couldn't match that to one of the saved places (" + ", ".join(f["saved"][:8]) + ") - name one and I'll give the distance and time.")
        return " ".join([lead] + out)
    if intent == "PLACES":
        fo = f["focus"]; h = f.get("here"); lines = []
        if fo == "heading":
            if not h: return "I don't have a GPS fix yet."
            if h.get("heading_dir") is None: return "I have a position but no course information (only per-minute positions are stored for that moment)."
            s = (f"The car is heading {h['heading_dir']} ({h['heading']} degrees)" + (f" at {h['speed']:.0f} km/h." if h["moving"] else f" - it is parked; that was its last course at {h['time'][11:]}."))
            if f.get("ahead"): s += " Saved places roughly ahead: " + ", ".join(f"{a['name']} ({a['km']} km)" for a in f["ahead"]) + "."
            return s
        if fo == "going":
            s = ("I don't have a navigation destination, " + (f"but the car is heading {h['heading_dir']} at {h['speed']:.0f} km/h." if h and h["moving"] and h.get("heading_dir") else "so I can only guess from your habits."))
            if f.get("ahead"): s += " Saved places roughly ahead: " + ", ".join(f"{a['name']} ({a['km']} km)" for a in f["ahead"]) + "."
            g = f.get("guess")
            if g: s += f" On {'weekdays' if g['weekday'] else 'weekends'} at this time from {g['from']} you have gone to {g['to']} {g['count']} of {g['of']} times - so probably {g['to']}."
            else: s += " There isn't a clear pattern for this time of day yet."
            return s
        if h:
            fmt = lambda la, lo: f"{abs(la):.4f} {'N' if la >= 0 else 'S'}, {abs(lo):.4f} {'E' if lo >= 0 else 'W'}"
            where = (f"at {h['near']}" if h["at"] else f"about {h['km']:.1f} km {h.get('compass') or ''} of {h['near']}".replace("  ", " ")) + (f" in the {h['area']} area" if h.get("area") else "")
            lines.append(("The car is on the move, " if h["moving"] else "The car is parked ") + where + f" (GPS {fmt(h['lat'], h['lon'])}{_gps_tail(h)}).")
        if fo == "history" and f["trips"] and re.search(r"how many", p.get("q") or ""):
            lines.append(f"The car was parked {len(f['trips'])} time{'s' if len(f['trips']) != 1 else ''} {wt} - once after each trip.")
        if fo == "history" and f["trips"]:
            lines.append(f"{wt.capitalize()}: {_plural(len(f['trips']), 'trip')} - " + "; ".join(f"{t['time']} {t['from']} to {t['to']} ({t['km']} km, {t['min']} min)" for t in f["trips"][:8]) + (f"; +{len(f['trips']) - 8} more" if len(f["trips"]) > 8 else "") + ".")
        elif fo == "history": lines.append(f"No trips were recorded {wt}.")
        return " ".join(lines) or "I don't have a location for the car yet."
    return "I can answer questions about " + CAPS + "."


def compose(intent, p, f, persona="driver"):
    if intent in DOMAIN_INTENTS: return compose_domain(intent, p, f, persona)
    eng = persona == "engineer"
    # These are capability/privacy questions, not a request for the current
    # camera state.  A stale telemetry record must not prevent the assistant
    # from explaining that it never interprets image content or from naming
    # the monitored camera.
    if intent == "STATUS" and p.get("ask") == "see":
        return "I don't interpret what the camera sees - I only monitor whether it can see properly." + (f" Its current status is unavailable: {f.get('why')}." if not f.get("ok") else "")
    if intent == "STATUS" and p.get("ask") == "which" and not f.get("ok"):
        return "Only the front camera is monitored. Its current status is unavailable: " + str(f.get("why", "no current camera evidence")) + "."
    if not f.get("ok"): return no_data(intent, f)
    wt = A.WHEN_TEXT.get(f.get("when") or p.get("w"), "")
    if intent == "WHY_RANGE":
        r0, r1, inc = f["range_first_km"], f["range_last_km"], f["increase_pct"]; chg = r1 / r0 - 1; d_amb = f["amb_last"] - f["amb_first"]; d_hv = f["hvac_wh_km_last"] - f["hvac_wh_km_first"]; d_hw = f["hwy_last"] - f["hwy_first"]; why = []
        if chg <= -0.03:
            s = f"Your estimated range fell about {-100 * chg:.0f}% ({r0} -> {r1} km) because energy use rose {inc:.0f}% (from {f['first']} to {f['last']} Wh/km)."
            if d_amb <= -1.5: why.append(f"colder weather ({f['amb_first']} -> {f['amb_last']} C)" + (f", which raised climate-control use by {d_hv:.0f} Wh/km" if d_hv > 2 else ""))
            elif d_amb >= 3 and d_hv > 2: why.append(f"hotter weather ({f['amb_first']} -> {f['amb_last']} C), which raised air-conditioning use by {d_hv:.0f} Wh/km")
            if d_hw >= 4: why.append(f"more highway driving ({f['hwy_first']}% -> {f['hwy_last']}% of distance)")
            if f.get("step_day") and f["step_delta"] > 12: why.append(f"a sudden jump on {f['step_day']} (+{f['step_delta']} Wh/km day-over-day - check for added drag such as a roof box, or low tyre pressure)")
            s += (" Main reasons: " + ", and ".join(why) + ".") if why else " I can't pin it on a single cause in the weather or route mix."
        elif chg >= 0.03:
            s = f"Your range has not dropped - it rose about {100 * chg:.0f}% ({r0} -> {r1} km) because energy use fell {-inc:.0f}% (from {f['first']} to {f['last']} Wh/km)."
            if d_amb >= 1.5: why.append(f"warmer weather ({f['amb_first']} -> {f['amb_last']} C)" + (f", which cut climate-control use by {-d_hv:.0f} Wh/km" if d_hv < -2 else ""))
            if d_hw <= -4: why.append(f"less highway driving ({f['hwy_first']}% -> {f['hwy_last']}% of distance)")
            if why: s += " Probably because of " + " and ".join(why) + "."
            if f.get("step_day") and f["step_delta"] > 12: s += f" There was one expensive day, {f['step_day']} (+{f['step_delta']} Wh/km), but it did not change the trend."
        else:
            s = f"Your range has been stable (about {r1} km; energy use changed {inc:+.0f}%, {f['first']} -> {f['last']} Wh/km)."
            if f.get("step_day") and f["step_delta"] > 12: s += f" The biggest single-day jump was {f['step_day']} (+{f['step_delta']} Wh/km)."
        return s
    if intent == "LARGEST": return f"The largest drop in efficiency was on {f['day']}: energy use rose by {f['delta']} Wh/km compared with the previous day (to {f['value']} Wh/km)."
    if intent == "TREND":
        return f"{A.METRICS.get(f['metric'], ('', 1, '', f['metric']))[3] if f['metric'] != 'energy_use' else 'Energy use'} went from {f['first']} to {f['last']} {f['unit']} ({f['change_pct']:+.1f}%) over {f['days']} days."
    if intent == "STATUS":
        if "state" in f and "range_km" not in f:                                                   # camera status
            s = {"clear": "The front camera is clear.", "blocked": "The front camera is BLOCKED", "degraded": "The front camera image is DEGRADED"}[f["state"]]
            if p.get("ask") == "see": s = "I don't interpret what the camera sees - I only monitor whether it can see properly. " + s
            elif p.get("ask") == "which": s = "Only the front camera is monitored. " + s.replace("is clear.", "is working normally (clear).")
            if f["state"] != "clear": s += f" since {f['since']}: {f['detail']}."
            if f.get("lane_avg") is not None: s += f" Over the latest recorded minute ({f['minute']}) lane-marking confidence averaged {f['lane_avg']:.0%} and the view was {'unobstructed' if f['block_avg'] < 0.1 else str(round(f['block_avg'] * 100)) + '% obstructed on average'}."
            return s
        if f.get("metric") == "range" or "range_km" in f:
            return (f"Estimated range is about {f['range_km']} km (battery {f['soc']}%, at your recent {f['wh_km']} Wh/km"
                    f"{'' if not f['basis_days'] else ', last ' + str(f['basis_days']) + ' days'}; as of {f['time']}).")
        return f"{A.METRICS[f['metric']][3].capitalize()} is {f['value']} {f['unit']} (as of {f['time']})."
    if intent == "COMPARE": return f"City driving uses {f['city']} Wh/km and highway driving {f['highway']} Wh/km - the car needs {abs(f['diff_pct']):.0f}% {'more' if f['diff_pct'] > 0 else 'less'} energy per km on the highway."
    if intent == "USAGE":
        if f["trips"] == 0: return f"No trips were recorded {wt}."
        s = f"{wt.capitalize()} you drove {f['km']} km in {f['trips']} trip{'s' if f['trips'] != 1 else ''}, using {f['kwh']} kWh ({f['wh_km']} Wh/km"
        s += f", about {f['range_basis_km']} km per full charge)." if f.get("range_basis_km") else ")."
        return s
    if intent == "SUMMARY":
        if "lane_avg" in f:
            return (f"Camera {wt}: lane-marking confidence averaged {f['lane_avg']:.0%}, the view was on average {f['block_avg']:.0%} obstructed (worst {f['block_max']:.0%}); "
                    f"{f['blocked_events']} blockage and {f['degraded_events']} image-degradation event(s) over {f['minutes']} minute{'s' if f['minutes'] != 1 else ''} of footage.")
        lab = A.METRICS[f["metric"]][3].capitalize()
        if f.get("min") is None:                                                          # minute tier only: the average is sound; an exact highest/lowest needs the one-second data
            ask = " An exact highest or lowest speed needs the one-second data, which does not cover that period." if p.get("emph") in ("max", "min") else ""
            return f"{lab} {wt}: average {f['avg']} {f['unit']}." + ask
        if p.get("emph") == "max": return f"Highest {lab.lower()} {wt}: {f['max']} {f['unit']} (average {f['avg']}, lowest {f['min']})." + (" That is the highest speed recorded, not the car's rated top speed." if f["metric"] == "speed" else "")
        if p.get("emph") == "min": return f"Lowest {lab.lower()} {wt}: {f['min']} {f['unit']} (average {f['avg']}, highest {f['max']})."
        return f"{lab} {wt}: average {f['avg']} {f['unit']} (low {f['min']}, high {f['max']})."
    if intent == "EVENTS":
        lab = A.EVENT_LABEL.get(f["type"], f["type"]); wh = A.WHEN_TEXT.get(f["when"], "")
        if f["count"] == 0: return f"No {lab} events {wh}."
        if f["type"] == "CAN_ANOMALY" and f["count"] > 6:                       # summarise: by attack type, first/last
            from collections import Counter
            kinds = Counter(i["detail"].split(":")[0] for i in f["items"])
            return (f"{f['count']} CAN bus incidents {wh}: " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common()) +
                    f". First {f['items'][0]['time']} ({f['items'][0]['detail'].split(': ', 1)[1][:60]}), last {f['items'][-1]['time']}.")
        out = f"{f['count']} {lab} event{'s' if f['count'] != 1 else ''} {wh}: " + _fmt_items(f["items"], dated=f["when"] not in ("y", "t")) + "."
        if f.get("kwh") is not None: out += f" Total {f['kwh']} kWh added" + (f" ({f['dc_kwh']} kWh of it DC fast charging)." if f.get("dc_kwh") else ".")
        return out
    if intent == "BEFORE_EVENT":
        d = f["deviations"]; top = [x for x in d if abs(x["z"]) > 2][:2]
        s = f"{A.EVENT_LABEL.get(f['type'], f['type']).capitalize()} at {f['time']}. In the {f['window_min']} minutes before: "
        s += ", ".join(f"{A.METRICS[x['signal']][3]} went {x['start']} -> {x['end']} {x['unit']}" for x in top) if top else "no signal moved unusually"
        s += "."
        if f["dc_fast_charge"]: s += " A DC fast charge was in progress."
        if f["first_log"]: s += f" First low-flow log at {f['first_log'][0]} ({f['first_log'][1]})."
        if f["causes"]: s += " Probable cause: " + f["causes"][0]
        if eng:
            s += "\n  evidence: " + "; ".join(f"{x['signal']} z={x['z']:+.1f}" for x in d)
            s += "\n  timeline: " + "; ".join(f"{t['time']} {t['what']}" for t in f["timeline"][-5:])
            s += f"\n  source: {f['source']}"
        return s
    return f"I can answer questions about {CAPS}."


# ------------------------------------------------------------------------------------------------ orchestrator
class Brain:
    def __init__(self, db_path, slm=None, now=None, persona="driver", slm_ir="auto", policy="legacy", selector=None, qc_mode=None):
        self._qc_mode = qc_mode                                                               # "all": every question the parser can express goes through the typed-query pipeline (the verified one)
        self.policy = policy; self.selector = selector                                      # policy="clarify": answer only supported readings, otherwise ask with concrete readings; selector: frozen-model candidate selection (optional)
        self.c = open_ro(db_path)
        from .vehicle import lake as _lake
        _lake.adopt(self.c, key=os.path.abspath(db_path))                                      # a provisioned vehicle database describes its own signals: register them before anything reads the registries
        self.slm, self.persona = slm, persona; self.units = dict(dialogue.DEFAULT_UNITS); self.last = None
        end = int(self.c.execute("SELECT v FROM meta WHERE k='end'").fetchone()[0]); self.now = now or end + 8 * 3600
        try: self.tz = int(float(self.c.execute("SELECT v FROM meta WHERE k='env.tz_h'").fetchone()[0]) * 3600)
        except Exception: self.tz = 0
        from .qa_final import QA
        self.qa = QA(self); self._text = ""
        from . import convstate
        self.trace = convstate.Trace()                                                           # explicit record of what each turn changed and inherited
        self.qcs = nl2ir.ConvState(); self._qc = None; self.last_qc = None                  # query-compiler path: conversation state, lazily built engine
        if slm_ir == "auto":                                                                 # the language model that compiles questions the rules cannot parse (needs the LoRA adapter file)
            try:
                from .ir_slm import autodetect
                slm_ir = autodetect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
            except Exception: slm_ir = None
        self.slm_ir = slm_ir

    def run_tool(self, intent, p):
        c, now = self.c, self.now; w = p.get("w") or "14"; m = p.get("metric"); e = p.get("event")
        if intent == "WHY_RANGE": return A.why_range(c, now, w if w in ("7", "14", "28", "a") else "14")
        if intent == "LARGEST": return A.largest_change(c, m if m in ("energy_use", "batt_temp", "coolant_flow", "speed", "soc", "amb", "hvac") else "energy_use", now, w if w in ("7", "14", "28", "a") else "a")
        if intent == "TREND": return A.trend(c, m or "energy_use", now, w if w in ("7", "14", "28") else "14")
        if intent == "STATUS": return A.est_range(c, now) if m == "range" else (A.camera_status(c, now) if m == "camera" else A.status(c, m if m in A.METRICS else "soc", now))
        if intent == "COMPARE": return A.compare_modes(c, now, w if w in ("7", "14", "28", "a") else "a")
        if intent == "USAGE": return A.usage(c, now, w)
        if intent == "SUMMARY": return A.camera_summary(c, now, w) if m == "lane_q" else A.summary(c, m if m in A.METRICS else "amb", now, w)
        if intent == "TYRES": return AX.tyres(c, now, w)
        if intent == "WEATHER": return AX.weather(c, now, w, live=(p.get("focus") == "now"))
        if intent == "SERVICE": return AX.vehicle_info(c, now) if p.get("focus") == "model" else AX.service(c, now)
        if intent == "HEALTH": return AX.health(c, now, w, p.get("focus"))
        if intent == "DRIVING": return AX.eco_speed(c, now, w if w in ("7", "14", "28") else "28") if p.get("focus") == "eco" else AX.driving(c, now, w)
        if intent == "PLACES": return AX.places(c, now, w, p.get("focus") or "here", p.get("q") or "")
        if intent == "EVENTS": return A.events(c, e or "ADAS_DISENGAGE", now, w or "y")
        if intent == "BEFORE_EVENT": return A.before_event(c, e or "WARN_BATT_OVERTEMP", now, 60)
        return {"ok": False, "why": "that isn't something I can answer from the vehicle data"}

    QC_MODE = property(lambda self: (self._qc_mode or os.environ.get("VGPT_QC", "auto")).lower())          # off | auto (new engine for what the old intents cannot express) | first (new engine for everything it can parse) | shadow (answer with the old, record the new)

    def qc(self):
        """The query compiler for this database (built on first use; never fails the assistant)."""
        if self._qc is None:
            from .qc import QueryCompiler
            self._qc = QueryCompiler(self.c, now=self.now, legacy=self.qa)
        else:
            self._qc.refresh(self.now)
        return self._qc

    def _pick_choice(self, text):
        """The person's reply to a clarification -> the chosen Query, or None when the reply is not a choice. Consumes the pending choices either way a choice is made."""
        t = text.strip().lower().rstrip(".!? "); ch = self.qcs.choices; words = {"one": 1, "first": 1, "1st": 1, "two": 2, "second": 2, "2nd": 2, "three": 3, "third": 3, "3rd": 3}
        m = re.match(r"^(?:option |number |reading |the |no\.? )?(\d)(?:\b|$)", t) or re.match(r"^(?:option |number |reading |the )?(one|two|three|first|second|third|1st|2nd|3rd)(?: one| reading| option)?$", t)
        k = None
        if m: k = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
        elif t in ("yes", "yeah", "yep", "ok", "okay", "correct", "right", "that one", "yes please", "sure") and len(ch) >= 1: k = 1
        if k is None or not (1 <= k <= len(ch)): return None
        q = ch[k - 1]; self.qcs.choices = []; return q

    def _no_engine(self):
        """True when this vehicle's data holds no engine, fuel or exhaust signal (an electric car), read once from its own signal catalogue."""
        if getattr(self, "_noeng", None) is None:
            try:
                from .vehicle import caps as _caps
                k = _caps.detect(self.c); self._noeng = not (k.get("engine") or k.get("fuel"))
            except Exception: self._noeng = False
        return self._noeng

    def qc_try(self, text, t0, understood):
        """Deterministic parser -> Query IR -> validated, planned, executed, composed and verified answer. Returns a result dict, or None to let the older routes answer."""
        mode = self.QC_MODE
        if mode in ("off", "0", "false", "no") or os.environ.get("VGPT_NO_QC"): return None
        try: qc = self.qc()
        except Exception: self._qc = False; return None
        if qc is False: return None
        if self.policy == "clarify" and self.qcs.choices and re.match(r"^(?:neither|none|none of (?:these|those|them)|no|nope|not (?:these|those|that)|something else|other|neither of these)[.!? ]*$", text.strip().lower()) or (self.policy == "clarify" and self.qcs.choices and re.match(r"^(?:option |number )?%d[.!? ]*$" % (len(self.qcs.choices) + 1), text.strip().lower())):
            self.qcs.choices = []; msg = "Okay - I will not guess. Tell me in other words what you want to know: which signal or event, and for which period?"; self.trace.record(text, None, "clarify", msg)
            return self._result(t0, msg, "CLARIFY", {"metric": None, "event": None, "w": "a"}, "qc", 0.5, {"ok": True, "qc": True, "clarify": True, "neither": True}, understood)
        picked = self._pick_choice(text) if (self.policy == "clarify" and self.qcs.choices) else None                    # the person answered a clarification with "2" / "the first" / "yes"
        if self.policy == "clarify" and self.qcs.choices and picked is None:                                             # anything else is a new question; expire the whole open clarification
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
        try: pr = nl2ir.Parse(picked, 0.95, followup=True, reason="the reading the person chose") if picked is not None else nl2ir.parse(text, now=self.now, tz=self.tz, profile=qc.profile, state=self.qcs)
        except Exception: return None
        none_p = {"metric": None, "event": None, "w": "a"}
        decision = None
        if self.policy == "clarify" and picked is None and mode != "shadow":
            from . import policy as POL
            try:
                if self.selector is not None: decision, _inf = POL.decide_by_selection(text, pr, self.selector, self.qcs.last, qc.profile, self.now, self.tz)
                else: decision = POL.decide(text, pr, None, self.qcs.last, qc.profile, self.now, self.tz)
            except Exception: decision = None
            if decision is not None and decision.kind == "clarify" and decision.candidates and re.match(r"^(?:why|how come)\b", text.strip().lower()) and not os.environ.get("VGPT_NO_QA"):
                try: _qr = self.qa.ask(text, nlu.normalize(text)[0])                                                       # an explanatory "why" about a signal belongs to the analysis engines that hold the reasons; a menu of statistics is not an explanation
                except Exception: _qr = None
                if _qr and ((_qr.get("facts") or {}).get("route") in ("signal_why", "why_event", "root_cause", "why_range", "why_range2", "prove", "effect_ctrl")):
                    return self._result(t0, _qr["answer"], _qr["intent"], none_p, "qa", 1.0, _qr["facts"], understood)
            if decision is not None and decision.kind == "clarify" and decision.candidates:
                self.qcs.choices = list(decision.candidates); self.trace.record(text, None, "clarify", decision.question)
                menu = decision.question + ("" if "neither of these" in decision.question and "(1)" in decision.question and "(2)" in decision.question and len(decision.candidates) == 1 else "\nReply with a number" + (f" (1-{len(decision.candidates)}" if len(decision.candidates) > 1 else " (1") + "; or 'neither' if none fits), or ask again in other words.")
                return self._result(t0, menu, "CLARIFY", none_p, "qc", 0.5, {"ok": True, "qc": True, "clarify": True, "readings": [c.to_text() for c in decision.candidates]}, understood)
            if decision is not None and decision.kind == "answer": pr.query, pr.confidence, pr.followup, pr.clarify = decision.query, 0.9, pr.followup, ""
            elif decision is not None and decision.kind == "clarify" and decision.question and not pr.ok:
                self.trace.record(text, None, "clarify", decision.question); return self._result(t0, decision.question, "CLARIFY", none_p, "qc", 0.5, {"ok": True, "qc": True, "clarify": True}, understood)
            elif decision is not None and decision.kind == "refuse":
                pq = pr.query                                                                                           # the rules recognised part of a data question (a signal, event or condition) but no reading is executable: ask, do not let a fuzzy route answer something else
                if decision.source == "gate-control":                                                                  # read-only: a control request is refused here, never handed to a route that could answer something else
                    self.trace.record(text, None, "refused", decision.question); return self._result(t0, decision.question, "SAFETY", none_p, "gate", 1.0, {"ok": False, "qc": True, "refused": True, "why": "read-only"}, understood)
                if decision.source not in ("gate", "status") and pq is not None and pq.op and (pq.target or pq.event is not None or pq.scope.conditions):
                    self.trace.record(text, None, "clarify", decision.question); return self._result(t0, decision.question, "CLARIFY", none_p, "qc", 0.5, {"ok": True, "qc": True, "clarify": True}, understood)
                self._decline = decision; return None                                                                  # not recognisably about recorded data (or not answerable): the older NON-data routes may still answer (see ask())
        if pr.clarify and mode != "shadow":
            self.qcs.pending, self.qcs.options = pr.pending, list(pr.options); self.trace.record(text, None, "clarify", pr.clarify)
            return self._result(t0, pr.clarify, "CLARIFY", none_p, "qc", 0.5, {"ok": True, "qc": True, "clarify": True, "suggest": list(pr.options[:3])}, understood)
        slm_info = None
        if not pr.ok and not pr.clarify and self.slm_ir is not None and mode != "shadow" and not pr.reason and self.policy != "clarify":      # the rules are unsure: let the language model compile the question (never when the rules recognised a causal / advice question)
            try:
                from .ir_slm import grounding_issues
                mq, slm_info = self.slm_ir.compile(text)
                from . import interpretation
                if mq is not None and not grounding_issues(mq, text) and not interpretation.interpret(text, mq, self.qcs.last, source="model", now=self.now, tz=self.tz).issues(): pr.query, pr.confidence, pr.followup = mq, 0.8, False; pr.query.source = "slm"
                else: return None                                                    # refused (op=none), malformed, or not grounded in the question: the older routes take over
            except Exception: return None
        if not pr.ok: return None
        q = pr.query
        from .query_validator import validate
        v = validate(q, qc.profile)
        if not v.valid: return None
        if mode in ("auto", "") and not nl2ir.takes_over(q, v.capability): return None
        if mode == "all" and picked is None and q.op == "aggregate" and not q.scope.conditions and q.scope.select is None and q.scope_b is None and q.target in nl2ir.LEGACY_TARGETS and not q.unit                 and all(any(m in n for m in ("No period given", "No period was given")) or "No statistic" in n for n in q.notes) and len(q.notes) >= 1 and re.search(r"\b(how(?:'s| is| are)|what(?:'s| is| are)|tell me about|status|state|condition)\b", nl2ir.normalise(text)) and not re.search(r"\b(average|avg|mean|highest|lowest|maximum|minimum|total|typical|median|peak)\b", nl2ir.normalise(text)): return None      # a bare "how is my battery": a whole-component status question, answered by the status engine, not an invented statistic over a default period
        if mode == "shadow":
            try: self.last_qc = qc.execute(q)
            except Exception: self.last_qc = None
            return None
        prev_q = self.qcs.last
        t1 = time.perf_counter()
        r = qc.execute(q, text=text)
        t2 = time.perf_counter()
        if r.status in ("invalid",): return None
        tr = os.environ.get("VGPT_QC_TRACE")
        if tr:
            try:
                with open(tr, "a", encoding="utf-8") as fh: fh.write(f"{text}" + chr(9) + f"{q.to_text()}" + chr(9) + f"{r.status}" + chr(10))
            except OSError: pass
        a = r.answer; self.last_qc = r
        if r.status in ("answered", "no_data"): self.qcs.last = q
        self.trace.record(text, q, {"answered": "answered", "no_data": "no_data"}.get(r.status, "limitation"), "; ".join(self._disclosed(text, q)) if hasattr(self, "_disclosed") else "")
        self.qcs.pending = None; self.qcs.options = []
        self._brief = a.brief
        facts = dict(r.facts(), suggest=a.followups[:3] or None, parse_ms=round(pr.ms, 2), parser=("slm" if slm_info else "rules"), slm=slm_info and {"ms": round(slm_info["ms"], 1), "tokens": slm_info["out_tokens"]})
        try:
            from . import interpretation
            said = [d for d in interpretation.interpret(text, q, prev_q, source="slm" if slm_info else "rules", now=self.now, tz=self.tz).disclosures() if not any(re.sub(r"[^a-z0-9 ]", "", d.lower()).replace(" was ", " ") .strip() in re.sub(r"[^a-z0-9 ]", "", n.lower()).replace(" was ", " ") for n in q.notes)]
        except Exception: said = []
        text_out = a.text + ("\n" + " ".join(q.notes) if q.notes and a.kind in ("value", "analysis") else "") + ("\n" + " ".join(said) if said and r.status == "answered" and a.kind in ("value", "analysis") else "")
        return self._result(t0, text_out, "QC_" + q.op.upper(), {"metric": q.target, "event": q.event.type if q.event else None, "w": q.scope.time.to_text()}, "qc", pr.confidence, facts, understood, None, t1, t2)

    def soft_route(self, text, t_rep):
        """Last resort before 'I did not understand': the classifier (relaxed), then the nearest known question if it is reasonably close - unless the classifier says the question is off-topic."""
        if os.environ.get("VGPT_NO_QA"): return None
        try:
            if self.qa.clf_says_off_topic(text, t_rep): return None
            qc = self.qa.ask_clf(text, t_rep, "last")
            if qc: qc["example"] = "the closest question I know"; return qc
            return self.qa.ask_fuzzy(text, min_score=0.30, t=t_rep, soft=True)
        except Exception: return None

    def _anchor(self, full):
        """The warning / fault moment this answer is about (so the UI can offer an evidence chart): only when the answer itself names that day."""
        import datetime as _dt
        ev = self.qa.ctx.get("event"); cands = [self.qa.ctx.get("anchor_ts"), ev[0] if ev else None]
        for ts in cands:
            if not ts: continue
            try: d = _dt.datetime.fromtimestamp(ts + self.tz, _dt.timezone.utc); label = f"{d.day} {dialogue._MON3[d.month - 1]}"
            except Exception: continue
            if label in full: return int(ts)
        return None

    def _result(self, t0, answer, intent, p, path, conf, facts, understood, slm_info=None, t1=None, t2=None):
        t3 = time.perf_counter(); t1 = t1 or t0; t2 = t2 or t1
        if path not in ("qa", "qc"):
            from .qa import neutralise
            answer = neutralise(answer); self.qa.note(self._text, answer, intent, p, facts)
            if intent not in NON_ANSWER_INTENTS: self.qcs.last = None        # a data answer from the older routes ends the query-compiler conversation (its follow-ups would be stale); a refusal, help text or small talk leaves the accepted state untouched
        elif path == "qa" and intent not in NON_ANSWER_INTENTS: self.qcs.last = None
        if self.slm is not None and hasattr(self.slm, "tick"): self.slm.tick()
        full = dialogue.convert_units(answer, self.units); bf = getattr(self, "_brief", None); self._brief = None
        bf = dialogue.convert_units(bf, self.units) if bf else dialogue.brief(full)
        yr = getattr(self, "year", None)
        if yr is None:
            import datetime as _dt
            try: yr = self.year = _dt.datetime.fromtimestamp(self.now + self.tz, _dt.timezone.utc).year
            except Exception: yr = self.year = None
        full = dialogue.tidy(full, yr); bf = dialogue.plain(dialogue.tidy(bf, yr))
        ok_a = (path == "qa" and self.qa.ctx.get("last_route") in ANCHOR_ROUTES) or (path == "fast" and intent in ("BEFORE_EVENT", "EVENTS", "HEALTH"))
        anchor = self._anchor(full) if ok_a else None
        sug_out = facts.get("suggest") if isinstance(facts, dict) else None
        return {"suggest": sug_out, "anchor": anchor, "answer": full, "brief": bf, "speech": dialogue.speakable(dialogue.plain(dialogue.spoken(full, bf))), "units": dict(self.units), "intent": intent, "params": p, "path": path, "conf": round(conf, 2), "facts": facts, "understood": understood,
                "ms": {"parse": round((t1 - t0) * 1000, 2), "tools": round((t2 - t1) * 1000, 2), "compose": round((t3 - t2) * 1000, 2), "total": round((t3 - t0) * 1000, 2)}, "slm": slm_info}

    def _frontend_mode(self):
        return os.environ.get("VGPT_FRONTEND", "legacy").lower()

    def _learned_model(self):
        if getattr(self, "_lm", None) is None:
            from .ir_slm import LazyIrSlm
            d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            lora = os.environ.get("VGPT_QP_LORA") or os.path.join(d, "qp-lora.gguf")
            self._lm = LazyIrSlm(os.path.join(d, "qwen2.5-0.5b-instruct-q4_k_m.gguf"), lora, threads=int(os.environ.get("VGPT_THREADS", "2")), n_ctx=1536)
        return self._lm

    def _learned_turn(self, text, mode):
        """Learned front end.  Returns a result dict, or None to continue with the legacy pipeline (only modes learned / shadow ever do that)."""
        from .learned_front import LearnedFront
        if getattr(self, "_lf", None) is None: self._lf = LearnedFront(self, self._learned_model())
        if mode == "learned_only" and os.environ.get("VGPT_TRIPWIRE") == "1" and not getattr(Brain, "_armed", False):
            from .learned_front import arm_tripwire; arm_tripwire(); Brain._armed = True
        t0 = time.perf_counter(); none_p = {"metric": None, "event": None, "w": "a"}
        if mode == "shadow":
            try:
                r_, it = self._lf.answer(text)
                self._shadow = {"plan": it.raw, "answer": (r_ or {}).get("answer"), "issues": it.issues, "error": it.error}
            except Exception as e: self._shadow = {"error": str(e)}
            return None
        r, it = self._lf.answer(text)
        if r is not None:
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []; self._brief = r.get("brief")
            q = it.query
            return self._result(t0, r["answer"], r.get("intent", "QC"), {"metric": q.target, "event": q.event.type if q.event else None, "w": q.scope.time.to_text()} if q is not None else none_p, "qc", 0.9, r.get("facts") or {"ok": True}, None)
        if mode == "learned": return None
        msg = self._lf.clarification(it)
        return self._result(t0, msg, "CLARIFY", none_p, "qc", 0.4, {"ok": False, "learned": True, "clarify": True, "plan": it.raw, "error": it.error}, None)

    GENERIC_DATA_INTENTS = {"EVENTS", "BEFORE_EVENT", "TREND", "COMPARE", "LARGEST", "USAGE", "SUMMARY", "WHY_RANGE", "ANALYSIS", "LOOKUP", "HEALTH", "STATUS"}          # statistic / event / comparison intents (and a status guess nobody asked for)
    GATE_ALLOWED = {"SAFETY", "NONE", "FOREIGN", "SMALLTALK", "HELP"}

    def ask(self, text, use_slm=True):
        """Under the product configuration (policy="clarify") the typed pipeline owns every data question: when the policy declined to form an executable reading, an older engine may still answer greetings, help, units, glossary, safety refusals and present-state status / health summaries, but NOT a statistic, event or comparison question (it would answer without the typed checks, and often something else). Those get the policy's own question or limitation instead."""
        self.turn = getattr(self, "turn", 0) + 1; self._decline = None
        from . import safety_policy                                                           # deterministic safety policy: a decision-under-hazard question is never a telemetry statistic
        _si = safety_policy.classify(re.sub(r"<[^>]{0,300}>", " ", str(text))[:600])
        if _si is not None:
            _t0 = time.perf_counter(); _r = safety_policy.respond(_si, self)
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
            return self._result(_t0, _r["answer"], "SAFETY", {"metric": None, "event": None, "w": "a"}, "policy", 1.0, _r["facts"], None)
        fe = self._frontend_mode()
        if fe != "legacy" and not getattr(self, "_in_translate", False):
            lr = self._learned_turn(text, fe)
            if lr is not None: return lr
        r = self._ask_routes(text, use_slm)
        if r.get("intent") not in ("CLARIFY", "NONE", "SAFETY", "HELP", "SMALLTALK", "GLOSSARY", "ABOUT", "FEEDBACK", "FOREIGN", "UNITS") and r.get("path") not in ("gate",): self._reshow_q = text   # the last data question, so a units change can show its answer again
        if fe in ("learned", "shadow") and isinstance(r.get("facts"), dict): r["facts"]["frontend"] = "legacy_fallback" if fe == "learned" else "legacy"; r["facts"]["shadow"] = getattr(self, "_shadow", None) if fe == "shadow" else None
        if r.get("intent") not in ("CLARIFY", "NONE", "SAFETY", "HELP", "SMALLTALK", "GLOSSARY", "ABOUT", "FEEDBACK", "FOREIGN") and r.get("path") not in ("gate",):
            from . import dialogue_state; dialogue_state.observe(self, text)
        d = self._decline
        if d is not None and self.policy == "clarify" and ((d.source == "none" and r.get("intent") in self.GENERIC_DATA_INTENTS) or (d.source == "gate" and r.get("intent") not in self.GATE_ALLOWED)):
            msg = "I can only answer questions about this vehicle's recorded data, and I cannot act on that." if d.source == "gate" else (d.question or "I could not turn that into something I can compute from this vehicle's data. Which signal, event or period do you mean?")
            self.trace.record(text, None, "refused" if d.source == "gate" else "clarify", msg)
            return self._result(time.perf_counter(), msg, "SAFETY" if d.source == "gate" else "CLARIFY", {"metric": None, "event": None, "w": "a"}, "gate" if d.source == "gate" else "qc", 0.5, {"ok": False, "qc": True, "declined_by_policy": True, "source": d.source}, None)
        return r

    def _ask_routes(self, text, use_slm=True):
        text = re.sub(r"<[^>]{0,300}>", " ", str(text)); text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text); text = " ".join(text.split())[:600]      # markup, control characters and absurd lengths never reach the rules
        t0 = time.perf_counter(); A.set_tz(self.tz)
        t, fixes = nlu.normalize(text, repair_typos=False)                       # pass 1: exactly what the user typed
        understood = None
        if not in_domain_clean(t) or parse_clean(t)[2] < FAST_MIN:               # pass 2: only if pass 1 is not already confident, try repairing typos
            t2, fixes2 = nlu.normalize(text, repair_typos=True)
            if fixes2 and (not in_domain_clean(t) or parse_clean(t2)[2] > parse_clean(t)[2]): t, understood = t2, t2
        none_p = {"metric": None, "event": None, "w": "a"}
        self._text = text
        t_rep = nlu.normalize(text, repair_typos=True)[0]
        from . import dialogue_state
        # A complete request with its own signal/event must not be treated as an
        # answer to an earlier clarification.  Conversation edits are allowed
        # only for fragments whose meaning genuinely depends on prior state.
        _direct = None
        try:
            _direct = nl2ir.parse(text, now=self.now, tz=self.tz, profile=self.qc().profile, state=None)
            _domain0 = parse_clean(t)
            _self_contained_domain = bool(_domain0[2] >= FAST_MIN and _domain0[0] in DOMAIN_INTENTS
                                          and re.search(r"\b(?:service|servicing|maintenance|health|healthy|problems?|issues?|tyres?|tires?|weather|driving|location|gps|where)\b", t))
            _fresh = bool((_direct.ok and _direct.query and (_direct.query.target or _direct.query.event or _direct.query.groups) or _self_contained_domain)
                          and not re.match(r"^(?:and|also|but|what about|how about|same|for that|for it)\b", t))
        except Exception:
            _fresh = False
        if _fresh:
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
        _dr = None if _fresh else dialogue_state.try_resolve(self, text, t_rep)
        if _dr:
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []; self._brief = _dr.get("brief")
            return self._result(t0, _dr["answer"], _dr["intent"], none_p, "qa", 0.9, _dr["facts"], understood)
        for tt in (t, nlu.normalize(text, repair_typos=True)[0]):                               # whole-sentence help / about / feedback / small talk beat the keyword routes ("show me some examples")
            if dialogue.HELP_RX.search(tt): return self._result(t0, dialogue.HELP_TEXT, "HELP", none_p, "rules", 1.0, {"ok": True}, understood)
            st_ = dialogue.smalltalk(tt)
            if st_: return self._result(t0, st_, "SMALLTALK", none_p, "rules", 1.0, {"ok": True}, understood)
        hg = dialogue.hinglish(text)
        if hg and hg[0] == "translate" and not getattr(self, "_in_translate", False):             # Hindi-English phrases are unambiguous: translate first, then answer as usual
            self._in_translate = True
            try: r_ = self.ask(hg[1], use_slm=use_slm)
            finally: self._in_translate = False
            r_["understood"] = f"{text} -> {hg[1]}"; return r_
        if hg and hg[0] == "note": return self._result(t0, hg[1], "FOREIGN", none_p, "rules", 1.0, {"ok": False, "why": "language"}, understood)
        if re.search(r"\bweather (in|at|for|of|over) (?!my\b|the (car|vehicle)\b|here\b|this\b|our\b)[a-z]", text.lower()):               # the weather somewhere else is not something a car knows
            return self._result(t0, "Sorry, I can't help with the weather elsewhere: I only know what this car's own rain and light sensors recorded. Ask \"Is it raining?\" or \"How was the weather on my last trip?\".", "NONE", none_p, "unsupported", 1.0, {"ok": False, "why": "weather elsewhere", "unsupported": "weather elsewhere"}, understood)
        from .qa_fuzzy import OOD
        if OOD.search(text.lower()) and not re.search(r"\b(car|vehicle|battery|motor|tyre|tire|brake|brakes|trip|drive|driving|charge|charging|range|ev)\b", text.lower()):     # a recipe, a song, a capital city: not about the car (and never typo-"repaired" into a car word)
            sug_ = suggestions(t)
            return self._result(t0, "That doesn't look like a question about this car, so I'll leave it.\nTry one of these:\n  - " + "\n  - ".join(sug_), "NONE", none_p, "gate", 1.0, {"ok": False, "why": "not about this vehicle", "suggest": sug_}, understood)
        def qa_done(qr, semantic_info=None):
            li, lp, lc = parse_clean(t); intent_q, params_q = qr["intent"], none_p
            if li in ("EVENTS", "BEFORE_EVENT", "TREND", "COMPARE", "LARGEST", "USAGE", "SUMMARY", "STATUS", "WHY_RANGE", "HEALTH") and lc >= FAST_MIN and qr["intent"] not in ("SAFETY", "NONE"): intent_q, params_q = li, lp     # report the slots the rules understood
            if (qr.get("facts") or {}).get("route") in ("city_highway", "city_highway_energy"):
                facts_ = A.compare_modes(self.c, self.now, "a")
                if facts_.get("ok"):
                    intent_q, params_q = "COMPARE", {"metric": "energy_use", "event": None, "w": "a"}
                    self.last = (intent_q, params_q, facts_)
            self._brief = qr.get("brief")
            facts_q = qr.get("facts") or {}
            confidence = 0.9 if facts_q.get("verified") or facts_q.get("engine") == "qc" else 0.85
            # If this route depended on the semantic classifier, do not report
            # more confidence than its weakest constrained decision supports.
            if semantic_info and semantic_info.get("pmax") is not None:
                confidence = min(confidence, float(semantic_info["pmax"]))
            return self._result(t0, qr["answer"], intent_q, params_q, "qc" if facts_q.get("engine") == "qc" else "qa", confidence, facts_q, understood, semantic_info)       # an answer the typed engine computed is reported as such

        # One learned semantic decision is needed only when deterministic
        # parsing has *not* already yielded a complete, valid program.  The
        # former unconditional front door made a simple request such as
        # "current speed" pay a full 0.5B inference before the exact typed
        # program was executed.  A valid program with no unexplained tokens is
        # stronger evidence than a broad four-slot classifier frame, and this
        # keeps the classifier available for the genuinely ambiguous cases.
        direct_complete = bool(_direct and _direct.ok and _direct.query and not _direct.unexplained)

        # Resolve compositional domain semantics before consulting a coarse
        # learned frame.  These operations combine a speech act with ontology
        # entities and execute against verified data; they are strictly more
        # specific than the model's four-field task/concept summary.  Letting
        # that summary go first allowed a plausible but wrong frame (for
        # example distance capability -> maximum speed) to pre-empt the exact
        # range operation, and made basic questions pay avoidable inference
        # latency.  Qwen 0.5B remains the only learned model and is used when
        # this typed semantic layer cannot resolve the request.
        from . import semantic_ops
        semantic_result = semantic_ops.handle(self, text, t_rep)
        if semantic_result:
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
            return qa_done(semantic_result)

        # A clearly named quantity that is absent from the selected vehicle's
        # schema is also an exact semantic result.  Report that limitation
        # before a coarse model frame can map it onto a neighbouring sensor
        # (alignment -> tyre pressure, tread depth -> pressure, and so on).
        try:
            from . import measure_grounding
            explicit_measures, absent_measures = measure_grounding.named_measures(text)
            if absent_measures and not explicit_measures:
                named = list(dict.fromkeys(label for _, label in absent_measures))
                quantity = ", ".join(named)
                msg = f"This vehicle's recorded schema does not include {quantity}, so I cannot calculate it or substitute another signal."
                return self._result(t0, msg, "NONE", none_p, "schema", 1.0,
                                    {"ok": False, "route": "unrecorded_measure", "missing": named}, understood)
        except Exception:
            # Grounding is a protective fast path; the normal typed compiler
            # remains available if catalogue inspection itself fails.
            pass

        # One learned semantic decision precedes compilation when needed.  It separates
        # task type (definition / estimate / capability / advice / telemetry)
        # from the requested operation and vehicle concept.  Dispatch below is
        # by ontology value, never by a phrase from a failing transcript.
        semantic_intent = None; semantic = None; semantic_info = None
        if not direct_complete and use_slm and self.slm is not None and os.environ.get("VGPT_SEMANTIC_FRONTDOOR") == "1":
            try: semantic_intent, semantic, semantic_info = self.slm.parse(t)
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("semantic interpreter unavailable (%s: %s)", type(e).__name__, e)
        if semantic:
            task, concept = semantic.get("task"), semantic.get("metric")
            operator = semantic.get("operator")

            # Compile broad whole-vehicle meanings into a closed typed topic.
            # The model supplies the task/operator frame; this small structural
            # check distinguishes a health request about the whole vehicle from
            # a request about one signal.  It is deliberately not a table of
            # transcript phrases or answers.
            whole_vehicle = bool(re.search(r"\b(?:car|vehicle|ev|automobile)\b", t))
            # ``how`` is only an interrogative.  Treating it as a health
            # predicate made unrelated requests such as seating capacity look
            # like whole-vehicle health checks.  The state predicate must be
            # present independently of the question word.
            overall_state = bool(re.search(r"\b(?:condition|status|state|health|healthy|okay|ok|fine|doing|wellbeing|problems?|issues?|faults?|wrong)\b", t))
            if task == "telemetry" and concept is None and operator in ("other", "now") and whole_vehicle and overall_state:
                from .query_ir import Query
                from . import plan_exec
                q_ = Query(op="topic", params={"topic": "vehicle_health"})
                r_ = plan_exec.execute(self, self.qc(), q_)
                if r_:
                    r_["facts"] = dict(r_.get("facts") or {}, semantic_frame=semantic)
                    return qa_done(r_, semantic_info)

            # A modal maximum-speed request is a vehicle capability, not the
            # present speed and not automatically the highest observed sample.
            # Keep that distinction in the ontology/typed-plan layer.
            modal_capability = bool(re.search(r"\b(?:can|could|capable|capability|rated|specified|specification)\b", t))
            if concept == "speed" and operator == "maximum" and (task == "capability" or modal_capability):
                from .query_ir import Query
                from . import plan_exec
                q_ = Query(op="topic", params={"topic": "capability"})
                r_ = plan_exec.execute(self, self.qc(), q_)
                if r_:
                    r_["facts"] = dict(r_.get("facts") or {}, semantic_frame=semantic)
                    return qa_done(r_, semantic_info)

            # A four-field frame is deliberately not a complete vehicle query.
            # Do not let a broad class such as "advice" or "estimate" pre-empt
            # the typed relation/condition handlers below.  Those handlers can
            # preserve the requested quantity, target and scope; this coarse
            # frame is retained as evidence and used only after they decline.
        # Stable task-level routes precede the analytical compiler.  These are
        # broad capabilities (glossary, vehicle identity, live location), not
        # fixes for individual sentences.  A new self-contained request also
        # closes any unanswered clarification so stale state cannot hijack it.
        bounded_routes = {"soh_rate", "charge_time", "last_charge", "battery_equiv", "drive_modes", "improve_score"}
        # A comparison with a named metric belongs to the typed compiler.  The
        # general domain question has no numeric target, so the explanatory
        # comparison capability handles it without inventing one.
        if re.search(r"\bcity\b", t) and re.search(r"\b(?:highway|motorway)\b", t) and not re.search(r"\b(?:consumption|efficiency|energy|speed|distance|regen(?:eration|erative)?|time|temperature|power)\b", t):
            bounded_routes.add("city_highway")
        if re.search(r"\b(?:what does|what is|mean(?:ing)?|explain|decode|interpret)\b.{0,30}\b(?:code|dtc|fault|error|trouble)\b|\b(?:code|dtc|fault|error|trouble)\b.{0,20}\bmean\b", t): bounded_routes.add("dtc_explain")
        if re.search(r"\b(?:12 ?v|12 volt|auxiliary battery)\b", t) and re.search(r"\b(?:weak|health|healthy|check|status|condition|voltage|ok|okay|fine)\b", t): bounded_routes.add("v12")
        # Only consult legacy handlers for capabilities the typed IR does not
        # express. Its unrestricted classifier previously hijacked unrelated
        # requests (for example diagnostics -> longest-trip analytics).
        # Cross-cutting semantic relations and calculations are composed from
        typed_owns = False
        try:                                                                                          # a complete, executable statistic over a stated window belongs to the typed engine; keyword routes and glossary must not take it
            _p = nl2ir.parse(text, now=self.now, tz=self.tz, profile=self.qc().profile, state=None)
            typed_owns = bool(_p.ok and _p.query is not None and (_p.query.target or (_p.query.op == "percent_time" and _p.query.scope.conditions and not any("No period given" in n_ for n_ in (_p.query.notes or [])))) and _p.query.op in ("aggregate", "extreme", "compare", "percent_time", "trend") and not _p.unexplained
                              and not (_p.query.target and nl2ir.R.CONCEPTS[_p.query.target].dtype == "categorical" and any("No period given" in n for n in (_p.query.notes or []))))   # a categorical state with no stated window is a question about the configuration, not a statistic
        except Exception: pass
        specialist = None if (os.environ.get("VGPT_NO_QA") or typed_owns) else self.qa.ask(text, t_rep, only=bounded_routes)
        specialist_route = ((specialist or {}).get("facts") or {}).get("route")
        if specialist_route in bounded_routes or (specialist_route == "events" and ((specialist or {}).get("facts") or {}).get("type") == "DTC_SET"):
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
            return qa_done(specialist)
        # Only unmistakable definition requests bypass the typed compiler.
        # Diagnostic-code explanations need recorded event context, not a
        # generic glossary definition of “DTC”.
        definition_request = bool(re.search(r"^(?:what (?:is|are)|what does .+? mean|define|meaning of|explain)\b", t)) and not re.search(r"\b(?:code|dtc|fault|error|trouble)\b", t)
        # An explicit definition speech act is not a telemetry aggregate even
        # when the named term also appears in the signal ontology.
        g = dialogue.glossary(t) if definition_request else None
        if g:
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []; self.qcs.last = None
            return self._result(t0, g, "GLOSSARY", none_p, "rules", 1.0, {"ok": True}, understood)
        domain_intent, domain_p, domain_conf = parse_clean(t)
        typed_probe = nl2ir.parse(text, now=self.now, tz=self.tz, profile=self.qc().profile, state=None) if domain_intent in ("SERVICE", "PLACES") else None
        typed_component = bool(typed_probe and typed_probe.ok and typed_probe.query and typed_probe.query.target in ("brake_fluid_condition",))
        stable_location = domain_intent == "PLACES" and domain_p.get("focus") in ("here", "coords", "distance", "eta", "heading", "going")
        stable_identity = domain_intent == "SERVICE" and domain_p.get("focus") == "model"
        if domain_conf >= FAST_MIN and ((stable_location or stable_identity) and not typed_owns and not (typed_probe and typed_probe.ok and typed_probe.query is not None and (typed_probe.query.target or typed_probe.query.scope.select is not None or typed_probe.query.event is not None or (typed_probe.query.op == "percent_time" and typed_probe.query.scope.conditions)) and not typed_probe.unexplained)) and not typed_component and not nlu.unsupported(t):
            self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
            f = self.run_tool(domain_intent, domain_p); domain_p["units"] = self.units
            if f.get("ok"): self.last = (domain_intent, {k: v for k, v in domain_p.items() if k != "units"}, f)
            return self._result(t0, compose(domain_intent, domain_p, f, self.persona), domain_intent, domain_p, "fast", domain_conf, f, understood)
        uns = nlu.unsupported(t)
        if uns:
            if uns[0] == "forecasts" and not os.environ.get("VGPT_NO_QA"):                       # the analysis engines hold the specific refusal ("an exact time would be invented"); the generic one is the fallback
                qr_ = self.qa.ask(text, t_rep)
                if qr_: return qa_done(qr_)
            f = {"ok": False, "why": uns[1], "unsupported": uns[0]}
            return self._result(t0, f"Sorry, I can't help with {uns[0]}: {uns[1]}.", "NONE", none_p, "unsupported", 1.0, f, understood)
        qcr = None if (dialogue.EV_RX.search(t) and self._no_engine()) else self.qc_try(text, t0, understood)      # the query compiler: compositional questions, new signals, missing-data explanations (a question about pistons or gears on an electric car is a glossary question, even though the catalogue now knows a `gear` signal)
        if qcr: return qcr
        qr = None if os.environ.get("VGPT_NO_QA") else self.qa.ask(text, t_rep)                  # the keyword routes first
        if qr: return qa_done(qr)
        um = dialogue.parse_units(text.lower())
        for tt in (t, nlu.normalize(text, repair_typos=True)[0]):                               # conversation layer: follow-ups, help, glossary (deterministic text)
            if dialogue.FOLLOW_RX.search(tt):
                if not self.last: return self._result(t0, "I haven't answered anything yet - ask me something first and I'll explain it.", "FOLLOWUP", none_p, "rules", 1.0, {"ok": True}, understood)
                li, lp, lf = self.last; lp = dict(lp, units=self.units)
                return self._result(t0, dialogue.explain(li, lp, lf, compose, self.units), "FOLLOWUP", none_p, "rules", 1.0, {"ok": True, "about": li}, understood)
            if dialogue.HELP_RX.search(tt): return self._result(t0, dialogue.HELP_TEXT, "HELP", none_p, "rules", 1.0, {"ok": True}, understood)
            if dialogue.ABOUT_RX.search(tt): return self._result(t0, dialogue.ABOUT_TEXT, "ABOUT", none_p, "rules", 1.0, {"ok": True}, understood)
            if dialogue.FEEDBACK_RX.search(tt): return self._result(t0, dialogue.FEEDBACK_TEXT, "FEEDBACK", none_p, "rules", 1.0, {"ok": True}, understood)
            st_ = dialogue.smalltalk(tt)
            if st_: return self._result(t0, st_, "SMALLTALK", none_p, "rules", 1.0, {"ok": True}, understood)
            hg = dialogue.hinglish(tt)
            if hg and hg[0] == "translate" and not getattr(self, "_in_translate", False):
                self._in_translate = True
                try: r_ = self.ask(hg[1], use_slm=use_slm)
                finally: self._in_translate = False
                r_["understood"] = f"{text} -> {hg[1]}"; return r_
            if hg and hg[0] == "note": return self._result(t0, hg[1], "FOREIGN", none_p, "rules", 1.0, {"ok": False, "why": "language"}, understood)
            if dialogue.EV_RX.search(tt) and not um: return self._result(t0, dialogue.EV_TEXT, "GLOSSARY", none_p, "rules", 1.0, {"ok": True}, understood)
            g = None if um else dialogue.glossary(tt)
            if g: return self._result(t0, g, "GLOSSARY", none_p, "rules", 1.0, {"ok": True}, understood)
        if self.last and re.match(r"^(are|is) (they|it|those|these) (working|ok|okay|fine|good|on|running)( right now| now| currently| at the moment)?$", t):          # "are they working right now?" -> the previous topic, live
            li, lp, lf = self.last; lp = dict(lp, units=self.units)
            if li == "WEATHER": lp["focus"] = "now"
            f2 = self.run_tool(li, lp); self.last = (li, {k: v for k, v in lp.items() if k != "units"}, f2)
            return self._result(t0, "Checking again: " + compose(li, lp, f2, self.persona), li, lp, "rules", 1.0, f2, understood)
        if um:
            self.units.update(um)
            if not in_domain_clean(t) or parse_clean(t)[2] < FAST_MIN:                          # "I want it in psi": remember, and show the last answer again
                msg = "Okay - " + dialogue.describe_units(self.units) + "."
                if self.last:
                    li, lp, lf = self.last; lp = dict(lp, units=self.units); msg += "\n" + compose(li, lp, lf, self.persona)
                elif getattr(self, "_reshow_q", None):                                          # answered by the typed engine / qa layer, which keeps no (intent, params): ask it again in the new units
                    try: msg += "\n" + self._ask_routes(self._reshow_q, use_slm=False)["answer"]
                    except Exception: pass
                return self._result(t0, msg, "UNITS", none_p, "rules", 1.0, {"ok": True, "units": dict(self.units)}, understood)
        if not os.environ.get("VGPT_NO_QA"):                                                   # then the conversation layer above, then the classifier, then the nearest-example router
            ct_ = self.qa.clf_top(text, t_rep)
            if ct_ and ct_[0][0][1] >= qa_clf.T["convo"][0] and ct_[1] >= qa_clf.T["convo"][1]:
                if ct_[0][0][0] == "help": return self._result(t0, dialogue.HELP_TEXT, "HELP", none_p, "rules", 1.0, {"ok": True}, understood)
                if ct_[0][0][0] == "foreign": return self._result(t0, dialogue.HINGLISH_NOTE, "FOREIGN", none_p, "rules", 1.0, {"ok": False, "why": "language"}, understood)
            qc = self.qa.ask_clf(text, t_rep, "answer")
            if qc: return qa_done(qc)
            qf = self.qa.ask_fuzzy(text, min_score=0.5, t=t_rep)
            if qf: return qa_done(qf)
        if not in_domain_clean(t) or parse_clean(t)[2] < FAST_MIN:                              # nothing in the keyword routes and the older rules are unsure: try the fuzzy router before giving up
            qf = None if os.environ.get("VGPT_NO_QA") else self.qa.ask_fuzzy(text, min_score=0.55, t=t_rep)
            if qf:
                self._brief = qf.get("brief")
                return self._result(t0, qf["answer"], qf["intent"], none_p, "qa", 0.8, qf["facts"], understood)
        if not in_domain_clean(t):
            qs_ = self.soft_route(text, t_rep)
            if qs_: return self._result(t0, qs_["answer"], qs_["intent"], none_p, "qa", 0.6, qs_["facts"], 'read as "' + qs_["example"] + '"')
            f = {"ok": False, "why": "not about this vehicle"}
            sug_ = suggestions(t_rep)
            return self._result(t0, "That doesn't look like a question about this car, so I'll leave it.\nTry one of these:\n  - " + "\n  - ".join(sug_), "NONE", none_p, "gate", 1.0, dict(f, suggest=sug_), understood)
        intent, p, conf = parse_clean(t); path = "fast"; slm_info = None
        if conf < FAST_MIN:
            if use_slm and self.slm is not None:
                try:
                    # The semantic front door has already classified this
                    # utterance.  Reusing that immutable frame prevents a
                    # second model invocation from producing a different
                    # interpretation of the same question.
                    if semantic_intent is not None and semantic is not None:
                        s_intent, sp, slm_info = semantic_intent, semantic, semantic_info
                    else:
                        s_intent, sp, slm_info = self.slm.parse(t)
                    path = "slm"
                except Exception as e:                                                   # the model could not load (library missing, file damaged, out of memory): carry on with the rules, never fail the question
                    import logging; logging.getLogger("vehiclegpt").warning("language model unavailable (%s: %s); using the rules only", type(e).__name__, e)
                    self.slm = None; qs_ = self.soft_route(text, t_rep)
                    if qs_: return self._result(t0, qs_["answer"], qs_["intent"], none_p, "qa", 0.6, qs_["facts"], 'read as "' + qs_["example"] + '"')
                    sug_ = suggestions(t_rep)
                    return self._result(t0, "I'm not sure I understood that. Did you mean one of these?\n  - " + "\n  - ".join(sug_), "NONE", none_p, "clarify", 0.0, {"ok": False, "why": "no confident reading", "suggest": sug_}, understood)
                dm = sp.get("metric")
                if isinstance(dm, str) and dm.startswith("domain:"): s_intent, sp["metric"] = dm[7:].upper(), None
                ok, why = accept_slm(s_intent, sp, slm_info, p, t)
                if not ok or s_intent == "NONE":
                    qs_ = self.soft_route(text, t_rep)
                    if qs_: return self._result(t0, qs_["answer"], qs_["intent"], none_p, "qa", 0.6, qs_["facts"], 'read as "' + qs_["example"] + '"', slm_info)
                    from . import dialogue_state
                    mm = dialogue_state._missing_measure(self, text, t_rep)                              # a restriction without a measure is an incomplete typed query: ask for the missing slot
                    if mm: return self._result(t0, mm["answer"], "CLARIFY", none_p, "qa", 0.5, mm["facts"], understood, slm_info)
                    sug = suggestions(t); f = {"ok": False, "why": why}
                    msg = ("I'm not sure I understood that. " if not ok else "That isn't something I can answer from this vehicle's data. ") + "Did you mean one of these?\n  - " + "\n  - ".join(sug); f = dict(f, suggest=sug)
                    return self._result(t0, msg, "NONE", none_p, "clarify", 0.0, f, understood, slm_info, time.perf_counter(), time.perf_counter())
                intent = s_intent
                w = nlu.parse_when(t, None)
                p = {"metric": p["metric"] or LEGACY_CONCEPT.get(sp["metric"], sp["metric"]), "event": p["event"] or sp["event"], "w": w or DEFAULT_W.get(intent, "14"), "assumed": w is None}     # windows come from rules, never the model
                if intent == "WHY_RANGE": p["metric"] = "energy_use"
                if intent == "LARGEST" and not p["metric"]: p["metric"] = "energy_use"
                if intent == "SUMMARY" and not p["metric"]: p["metric"] = "amb"
                if intent == "USAGE": p["metric"] = p["metric"] or "energy_use"
                if intent in DOMAIN_INTENTS: p["focus"] = domain_focus(intent, t)
            elif intent == "NONE":
                qs_ = self.soft_route(text, t_rep)
                if qs_: return self._result(t0, qs_["answer"], qs_["intent"], none_p, "qa", 0.6, qs_["facts"], 'read as "' + qs_["example"] + '"')
                from . import dialogue_state
                mm = dialogue_state._missing_measure(self, text, t_rep)                              # last resort before a generic menu: a restriction without a measure is an incomplete typed query - ask for the missing slot
                if mm: return self._result(t0, mm["answer"], "CLARIFY", none_p, "qa", 0.5, mm["facts"], understood)
                sug = suggestions(t)
                return self._result(t0, "I'm not sure I understood that. Did you mean one of these?\n  - " + "\n  - ".join(sug), "NONE", none_p, "clarify", conf, {"ok": False, "why": "no confident reading", "suggest": sug}, understood)
            else: path = "fast-low"
        t1 = time.perf_counter()
        f = self.run_tool(intent, p)
        t2 = time.perf_counter()
        p["units"] = self.units
        ans = compose(intent, p, f, self.persona)
        if intent == "WHY_RANGE" and f.get("ok"):
            try: ans = self.qa.range_answer(p.get("w") if p.get("w") in ("7", "14", "28") else "14") or ans
            except Exception: pass
        if f.get("ok"): self.last = (intent, {k: v for k, v in p.items() if k != "units"}, f)
        notes = []
        if p.get("assumed") and intent in ("USAGE", "SUMMARY") and f.get("ok"): notes.append(f"No period given, so I used the last {p.get('w', '7')} days.")
        if re.search(r"\bfuel\b|\bmileage\b|\befficiency\b", t) and intent in ("USAGE", "COMPARE", "LARGEST", "WHY_RANGE", "TREND") and f.get("ok"):
            notes.append("This is an electric vehicle, so I report energy (kWh, Wh/km) rather than fuel.")
        if notes: ans += " (" + " ".join(notes) + ")"
        return self._result(t0, ans, intent, p, path, conf, f, understood, slm_info, t1, t2)
