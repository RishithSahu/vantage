"""Conversation layer (stdlib only): unit preferences ("I want it in psi"), help, a small glossary ("what is a CAN bus?"), and follow-ups ("what do you mean by more?").
All answers are deterministic text; nothing here talks to the model. The Brain keeps the last answer so a follow-up can explain it and a unit change can re-show it."""
import re

DEFAULT_UNITS = {"p": "kpa", "d": "km", "t": "c"}
KM = 1.609344


# ------------------------------------------------------------------------------------------------ units
def parse_units(raw):
    """Unit changes requested in the raw (lower-cased) text -> {"p": psi|bar|kpa, "d": mi|km, "t": f|c}. The data question itself is not touched ("how many km are left" changes nothing)."""
    u = {}; cue = re.search(r"\b(in|use|using|show|switch|change|set|convert|want|prefer|display|give|make|units?|please)\b", raw)
    if cue and re.search(r"\bmetric\b", raw) and not re.search(r"\b(psi|mph|miles?|fahrenheit)\b", raw): return {"p": "kpa", "d": "km", "t": "c"}                  # "use metric units" resets all three
    if cue and re.search(r"\bimperial\b", raw): return {"p": "psi", "d": "mi", "t": "f"}
    if re.search(r"\bpsi\b", raw): u["p"] = "psi"
    elif cue and re.search(r"\bbar\b|\bbars\b", raw) and not re.search(r"\bbar (graph|chart)\b", raw): u["p"] = "bar"
    elif cue and re.search(r"\bkpa\b|kilopascal", raw): u["p"] = "kpa"
    if re.search(r"\bmph\b|\bmiles\b|\bmile\b|imperial|\bin mi\b", raw) and cue: u["d"] = "mi"
    elif cue and re.search(r"\bin (km|kms|kilomet\w+)\b|\bmetric\b|\bkm/h\b|\bkph\b", raw) and re.search(r"\b(in|use|show|switch|change|set|convert|want|prefer|display|make|units?)\b", raw): u["d"] = "km"
    if re.search(r"fahrenheit|°f|\bin f\b|degrees f\b", raw): u["t"] = "f"
    elif re.search(r"celsius|centigrade|°c|\bin c\b|degrees c\b", raw): u["t"] = "c"
    return u


def describe_units(u):
    return ", ".join(x for x in (
        {"kpa": "tyre pressure in kPa", "bar": "tyre pressure in bar", "psi": "tyre pressure in psi"}[u["p"]],
        "distance in " + ("miles (speed in mph)" if u["d"] == "mi" else "kilometres (speed in km/h)"),
        "temperature in " + ("degrees Fahrenheit" if u["t"] == "f" else "degrees Celsius")))


def _num(s): return float(s.replace(",", ""))


def _fmt(x, dec=0): return f"{x:,.{dec}f}" if abs(x) >= 1000 else f"{x:.{dec}f}"


def _cv(x, unit):
    v = _num(x); return f"{v / KM:.1f}" if unit == "km/h" else f"{v * KM:.1f}"


def _ul(unit): return "mph" if unit == "km/h" else "Wh/mi"


def convert_units(s, u):
    """Re-express the numbers in an answer in the preferred units (the answer text is produced in km, km/h, Wh/km, C and kPa)."""
    if not u or u == DEFAULT_UNITS: return s
    if u.get("d") == "mi":
        s = re.sub(r"(\d[\d,.]*) (km/h|Wh/km) \(low ([\d.]+), high ([\d.]+)\)", lambda m: f"{_cv(m[1], m[2])} {_ul(m[2])} (low {_cv(m[3], m[2])}, high {_cv(m[4], m[2])})", s)
        s = re.sub(r"(\d[\d,.]*)( -> | to )(\d[\d,.]*) (km/h|Wh/km)\b", lambda m: f"{_cv(m[1], m[4])}{m[2]}{_cv(m[3], m[4])} {_ul(m[4])}", s)
        s = re.sub(r"(\d+(?:\.\d+)?) per 100 km; typical is (\d+)-(\d+)", lambda m: f"{_num(m[1]) * KM:.1f} per 100 mi; typical is {round(_num(m[2]) * KM)}-{round(_num(m[3]) * KM)}", s)
        s = s.replace("per 100 km", "per 100 mi")
        s = re.sub(r"(\d[\d,]*(?:\.\d+)?) km/h", lambda m: f"{_fmt(_num(m[1]) / KM, 1)} mph", s)
        s = re.sub(r"(\d+(?:\.\d+)?) Wh/km", lambda m: f"{_num(m[1]) * KM:.1f} Wh/mi", s)
        s = re.sub(r"(\d[\d,]*(?:\.\d+)?) -> (\d[\d,]*(?:\.\d+)?) km\b", lambda m: f"{_fmt(_num(m[1]) / KM)} -> {_fmt(_num(m[2]) / KM)} mi", s)
        s = re.sub(r"(\d[\d,]*(?:\.\d+)?) km\b", lambda m: (f"{_fmt(_num(m[1]) / KM)} mi" if _num(m[1]) >= 20 else f"{_fmt(_num(m[1]) / KM, 1)} mi"), s)
        s = s.replace("km on the motorway", "mi on the motorway").replace("Wh/km", "Wh/mi").replace(" per km", " per mi").replace("kilometres", "miles").replace("kilometre", "mile")
    if u.get("t") == "f":
        f = lambda c: c * 9 / 5 + 32
        s = re.sub(r"(-?\d[\d.]*) C \(low (-?[\d.]+), high (-?[\d.]+)\)", lambda m: f"{f(_num(m[1])):.1f} F (low {f(_num(m[2])):.1f}, high {f(_num(m[3])):.1f})", s)
        s = re.sub(r"(-?\d[\d.]*)( -> | to )(-?\d[\d.]*) C\b", lambda m: f"{f(_num(m[1])):.1f}{m[2]}{f(_num(m[3])):.1f} F", s)
        s = re.sub(r"(-?\d+(?:\.\d+)?) -> (-?\d+(?:\.\d+)?) C\b", lambda m: f"{f(_num(m[1])):.1f} -> {f(_num(m[2])):.1f} F", s)
        s = re.sub(r"(-?\d+(?:\.\d+)?) ?\u00b0C\b", lambda m: f"{f(_num(m[1])):.1f} \u00b0F", s)                  # typed-engine answers print the degree sign
        s = re.sub(r"(-?\d+(?:\.\d+)?) C\b", lambda m: f"{f(_num(m[1])):.1f} F", s)
    if u.get("p") in ("psi", "bar"):
        if u["p"] == "psi": s = re.sub(r"(\d+(?:\.\d+)?) kPa", lambda m: f"{_num(m[1]) * 0.145038:.1f} psi", s)
        else: s = re.sub(r"(\d+(?:\.\d+)?) kPa", lambda m: f"{_num(m[1]) / 100:.2f} bar", s)
    return s


def pressure(kpa, u):
    """Bare number in the preferred pressure unit (for lists that print the unit once)."""
    p = (u or DEFAULT_UNITS).get("p", "kpa")
    return f"{kpa * 0.145038:.1f}" if p == "psi" else f"{kpa / 100:.2f}" if p == "bar" else f"{kpa:.0f}"


def pressure_unit(u): return {"psi": "psi", "bar": "bar"}.get((u or DEFAULT_UNITS).get("p", "kpa"), "kPa")


# ------------------------------------------------------------------------------------------------ help
HELP_TEXT = ("I'm VehicleGPT - I live in this car and work fully offline. Ask me about:\n"
             "  - range, energy and efficiency (\"Why has my range dropped?\", \"How much energy did I use yesterday?\")\n"
             "  - battery, cooling and charging (\"Is my battery healthy?\", \"How much did I charge last week?\")\n"
             "  - tyres, brakes and service (\"How are my tyres?\", \"When is my next service?\")\n"
             "  - rain, light and weather as the car senses it (\"Is it raining?\")\n"
             "  - your driving and lane assist (\"How many times did I brake hard last week?\")\n"
             "  - the front camera (\"Is the camera blocked right now?\")\n"
             "  - GPS and places: where the car is, how far home or work are, how long to get there (\"How far is home?\", \"Where did I go yesterday?\")\n"
             "  - faults and CAN-bus security (\"Any fault codes this month?\", \"Show CAN bus attacks\")\n"
             "You can also say \"I want it in psi / miles / Fahrenheit\", or ask \"what is a CAN bus?\" or \"what do you mean?\".")
HELP_RX = re.compile(r"^(help( me)?|what can( i)?( ask)?( you)?( do| tell| help| about)?( me)?( about)?|what do you (know|do|cover|offer|support)|what are you|who are you|what (questions|topics|things|can) .{0,25}\b(ask|do|help)|what (is|are) (possible|your (capabilities|features|skills))|how (can|do) (i|you) use (you|this)|what can i ask( you)?|show (me )?(the )?(commands|examples|features)|examples?|what (questions|things|topics|kinds? of questions) (can|do) you (answer|handle|cover|know|help( with)?)|what (can|do) you answer"
                     r"|what (kind|sort|type)s? of (questions?|things) (can|may|should) i (ask|say)( you)?|what (is|are) (a|the|some) (question|questions) (about|for|on) (this|the|my) (vehicle|car)|how do i (use|talk to|ask|speak to) (you|this|it)|(show|give) me (some |a few )?(example|sample)s?( questions)?|what (can|should) i (ask|say)|what can you help me with|(list|tell me) (the things|what) you can do|how does this work|what topics (do you|can you) cover|what are your features|what do you know about (my|the) (car|vehicle)|what (is|are) (a|the) question|help me (with|to) (ask|use))[?.! ]*$")


ABOUT_RX = re.compile(r"^(are you|r u|are u)( a| an)? (gpt|chatgpt|ai|a\.i|bot|robot|chatbot|human|person|llm|model|assistant|car|real|offline|online)$|^(who made|who built|who created|who trained) you$|^what model are you$|^which (ai|model) are you$|^who is vehiclegpt$")
ABOUT_TEXT = ("I'm VehicleGPT, the assistant built into this car. I run entirely on the car's own computer with no cloud connection: a rules engine understands most questions, a small language model (Qwen2.5-0.5B) helps with unusual wording, "
              "and every number comes from the car's recorded data. I'm not ChatGPT, I can't chat about the world - ask me about the car (say \"what can I ask you?\").")
FEEDBACK_RX = re.compile(r"^(you are|youre|you re|u r|you r|are you) (not good|stupid|dumb|useless|bad|wrong|rubbish|terrible|silly|an idiot)$|^(that.s|thats|this is) (wrong|not right|bad|useless)$|^(?:bruh+|bro+)$")
FEEDBACK_TEXT = "Sorry about that. Tell me what you were hoping to find out and I'll try again - or ask \"what can I ask you?\" to see what I can answer."
EV_RX = re.compile(r"\b(pistons?|cylinders?|spark ?plugs?|exhaust|clutch|gear ?box|gears?|transmission|fuel tank|carburet\w*|turbo\w*|crank ?shaft|camshaft|engine (oil|size|capacity|cc|power)|horse ?power|bhp)\b|\bev\b|electric (car|vehicle)|(is|are) (my|this|the) (car|vehicle) (an? )?(ev|electric|petrol|diesel|hybrid|gas)|(petrol|diesel|gas) or electric|what (type|kind) of (car|vehicle|fuel|engine)|fuel type")
EV_TEXT = ("This is a battery-electric vehicle: no engine, pistons, exhaust, clutch or multi-speed gearbox (a single-speed reduction gear drives the wheels) and no engine oil. "
           "I monitor the traction battery, drive motor, battery cooling, tyres, brakes, the 12 V system and the camera instead.")


# ------------------------------------------------------------------------------------------------ glossary
GLOSSARY = [
    (r"can ?bus|controller area network|can ?fd|\bcan\b(?= (?:network|messages?|frames?|ids?|protocol|signals?|attacks?)\b|\s*\??$)", "The CAN bus is the car's internal network: the battery, motor, brake and other control units broadcast short messages on it many times a second. I only listen (read-only) and watch for unusual traffic such as flooding or spoofed values. Ask \"Show CAN bus attacks\" to see what I detected."),
    (r"\bsoc\b|state of charge", "State of charge (SoC) is how full the traction battery is, in percent - the electric car's fuel gauge."),
    (r"\bsoh\b|state of health|battery health", "State of health (SoH) is the battery's usable capacity compared with when it was new. 98% means it can still hold 98% of the original energy; it fades slowly with age and cycles."),
    (r"regen\w*|recuperation", "Regenerative braking uses the motor as a generator when you slow down, turning motion back into battery energy. It also means the friction brakes do less work and wear more slowly."),
    (r"\btpms\b|tire pressure monitor\w*|tyre pressure monitor\w*", "TPMS is the tyre-pressure monitoring system: a sensor in each wheel reports pressure (and temperature). It warns when a tyre is about 20% below the specified pressure."),
    (r"\badas\b|driver assist\w*", "ADAS means advanced driver-assistance systems: lane keeping, adaptive cruise control, emergency braking and similar features. I only report how they were used and when they disengaged."),
    (r"lane (assist|keep\w*|centering)", "Lane assist uses the front camera to see road markings and gently steers to keep the car in its lane. It needs a clear camera view and visible markings, and it hands control back to you when it cannot see them."),
    (r"\bdtcs?\b|trouble codes?|fault codes?|error codes?", "A DTC (diagnostic trouble code) is a standard code a control unit stores when it detects a fault, for example P1A00 for low battery-coolant flow. A workshop reads them over the diagnostic port; I read them from the car's diagnostic messages."),
    (r"wh ?/ ?km|watt.?hours? per k\w+", "Wh/km is the energy the car uses per kilometre. Lower is better: a typical electric car uses roughly 130-220 Wh/km, more on fast roads and in the cold."),
    (r"driving efficien\w*|energy efficien\w*|\beconomy\b|\bmileage\b", "Driving efficiency is how much energy the vehicle uses to travel a distance, normally shown as Wh/km. Lower Wh/km means the car travelled farther for the same battery energy."),
    (r"\bkwh\b|kilowatt.?hours?", "A kilowatt-hour (kWh) is the unit of energy that batteries store and chargers deliver. 60 kWh drives a typical electric car roughly 300-400 km."),
    (r"dc fast charg\w*|fast charg\w*|dc charg\w*", "DC fast charging feeds direct current straight into the battery at high power (50-200 kW), adding hundreds of km in about half an hour. It slows down as the battery fills and when the battery is very cold or hot."),
    (r"\bac charg\w*|home charg\w*|wallbox", "AC charging uses the car's on-board charger at 3-11 kW - typical for home or workplace charging, mostly overnight. It is gentler on the battery than fast charging."),
    (r"odometer", "The odometer is the total distance the car has driven in its life."),
    (r"12 ?v\b|12 volt|auxiliary battery", "The 12 V battery powers the lights, locks and control units, and wakes the car up. Even in an electric car it can weaken with age and cause start-up problems, so I watch its resting voltage."),
    (r"heat ?pump", "A heat pump heats the cabin by moving heat instead of making it with a resistive heater, so it uses roughly half to a third of the energy. It works less well in deep cold."),
    (r"\bobd\b|obd ?ii", "OBD-II is the standard diagnostic connector and message set every modern car supports; it exposes values such as speed and coolant temperature and the stored fault codes."),
    (r"\buds\b|unified diagnostic", "UDS (Unified Diagnostic Services) is the diagnostic protocol control units speak over CAN or Ethernet - reading fault codes, software versions and so on. I only read it, I never write to it."),
    (r"some ?/ ?ip|someip", "SOME/IP is a service-oriented message protocol on automotive Ethernet; services like ADAS state or charging publish events that I subscribe to."),
    (r"\bvhal\b|vehicle hal", "The Vehicle HAL is Android Automotive's interface to the car: it publishes properties such as speed, tyre pressure and battery level that apps and I can read."),
    (r"\bslm\b|small language model|\bllm\b|language model", "A language model is the AI that understands wording. Here it is a small one (Qwen 0.5B) that runs on the car itself and only turns your sentence into a request; every number in my answers comes from the car's recorded data."),
    (r"estimated range|driving range|\brange\b", "Estimated range is the battery energy left divided by your recent average consumption (Wh/km), so it changes with weather, speed and how you drive."),
    (r"coolant|thermal management|battery cooling", "The battery cooling system pumps liquid coolant around the pack to keep it at a safe temperature, especially during fast charging. Low coolant flow is a fault I watch for."),
    (r"\bota\b|over.the.air", "OTA (over-the-air) means the car receives software updates wirelessly, like a phone."),
    (r"\bpsi\b|\bkpa\b|\bbar\b", "kPa, bar and psi are all pressure units: 100 kPa = 1 bar = 14.5 psi. Say \"I want it in psi\" and I'll switch."),
    (r"mph|km/h|kph", "km/h is kilometres per hour and mph is miles per hour; 100 km/h is about 62 mph. Say \"I want it in miles\" and I'll switch."),
]
_GLOSS = [(re.compile(rx), txt) for rx, txt in GLOSSARY]
GLOSS_Q = re.compile(r"\b(what (is|are|does|do|s)|whats|what.s|explain|define|meaning of|tell me about|stand for)\b")
PERSONAL = re.compile(r"\b(my|mine|our|current|currently|now|today|yesterday|last|left|this week|week|month|right now|reading|level|value|did i|have i)\b")


def glossary(t):
    if not GLOSS_Q.search(t) or PERSONAL.search(t): return None
    for rx, txt in _GLOSS:
        if rx.search(t): return txt
    return None


# ------------------------------------------------------------------------------------------------ follow-ups
FOLLOW_RX = re.compile(r"^(what do you mean( by .+)?|what does (that|this|it) mean|more what|explain( that| it| this| more| further)?( please)?|why is that|why so|how so|elaborate|tell me more|more details?|can you explain( that| it| this)?|in simple words|simpler|clarify|what is that|huh|i dont understand|i don.t understand|say that again|repeat( that)?|what do you mean)$")


def explain(intent, p, f, compose, units):
    """Plain-language explanation of the previous answer: what the terms mean, how the number was obtained, then the evidence view."""
    base = {
        "COMPARE": lambda: f"Wh/km is the energy the car uses per kilometre (lower is better). In the city it used {f['city']} Wh/km, on the motorway {f['highway']} Wh/km - {abs(f['diff_pct']):.0f}% {'more' if f['diff_pct'] > 0 else 'less'} energy per km on the motorway, mostly because air drag grows with the square of speed, while stop-and-go city driving gets energy back through regenerative braking.",
        "WHY_RANGE": lambda: "Estimated range is the battery energy left divided by your recent Wh/km. When energy per km goes up - heating or cooling the cabin, more motorway driving, extra drag such as a roof box or low tyre pressure - the range goes down, and the reverse.",
        "USAGE": lambda: "Energy is what the car drew from the battery while driving, distance comes from the odometer, and Wh/km is the first divided by the second. 'Per full charge' is the usable battery energy divided by that Wh/km.",
        "TYRES": lambda: "Tyre pressure rises as tyres warm up, so I convert each reading to the equivalent at 20 C using the tyre's own temperature and compare it with the car's specified cold pressure. The official TPMS warning is 20% below that; a falling trend can show a slow leak before then.",
        "WEATHER": lambda: "Rain comes from the car's rain sensor / wiper state, so it is only known while the car is switched on. Cloudiness is inferred by comparing the ambient-light sensor with how bright a clear sky would be at this place and time.",
        "SERVICE": lambda: "Service is due at the earlier of the distance interval or 12 months. Brake-pad life is estimated from how much braking the friction brakes had to do (the motor does most of it), and battery health is today's capacity compared with new.",
        "DRIVING": lambda: "Hard braking or acceleration means a force of at least 0.28 g (about 2.7 m/s2) for a second, hard cornering 0.35 g sideways - thresholds taken from real accelerometer data. Regeneration share is the energy recovered compared with the energy used; lane-assist share is the part of motorway time it was active.",
        "HEALTH": lambda: "I look at stored fault codes, the coolant-flow trend against two weeks ago, tyre pressures, the 12 V battery, brake wear, battery health and the camera, and list anything that is off.",
        "PLACES": lambda: "Places are the saved locations (home, work ...) matched to where the car stopped. The guess for where you are going counts how often you went where from the current place at this time of day.",
        "STATUS": lambda: "State of charge is how full the battery is; the range estimate divides the energy left by your recent Wh/km, so it moves with your driving.",
        "EVENTS": lambda: "These are events the car recorded with their time and detail; I list the ones in the period you asked for.",
        "BEFORE_EVENT": lambda: "I compare the signals in the hour before the event with how they normally behave in the same driving mode (z-scores), and add the recorded log lines and charging state.",
    }.get(intent)
    head = base() if base else "Here is the same answer with the evidence."
    body = compose(intent, p, f, "engineer") if intent in ("WHY_RANGE", "HEALTH", "BEFORE_EVENT", "TYRES") else ""
    return convert_units(head + (("\n" + body) if body else ""), units)


# ------------------------------------------------------------------------------------------------ short answers for a driver, and text that reads well aloud
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def brief(answer, limit=200):
    """The spoken / default-visible version of an answer: its first sentence or clause (every long answer leads with its conclusion), without bullets or measurement chatter."""
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    lines = [l for l in lines if not l.startswith(("Anchor:", "Event:", "Note:"))] or lines
    if not lines: return answer
    first = lines[0].lstrip("-* ").strip()
    if first.endswith(":") and len(lines) > 1: first += " " + lines[1].lstrip("-* ").strip()
    first = re.sub(r"\s*\((?:1 Hz data|per-minute data|1 Hz sample|1 Hz|[0-9.]+ ?ms)\.?\)", "", first)
    if len(first) <= limit: return first
    out = ""
    for sen in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", first):
        if out and len(out) + len(sen) + 1 > limit: break
        out = (out + " " + sen).strip()
        if len(out) >= limit * 0.55: break
    if 40 <= len(out) <= limit * 1.15: return out
    cut = first[:limit]; pos = max(cut.rfind(d) for d in (". ", "; ", ": ", " - ", ", but ", " (", ", and ", ", "))
    if pos >= 70: return cut[:pos].rstrip(" ,;:-(") + ("" if cut[:pos].rstrip().endswith(".") else ".")
    return cut.rsplit(" ", 1)[0] + " ..."


_MON3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_M3RX = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_SPOKEN_EVENT_IDS = {
    # Event identifiers are storage/API values, not prose.  Expand them before
    # markdown cleanup; otherwise the underscore stripper glues the words
    # together and browser speech engines commonly spell the result out.
    "CAMERA_BLOCKED": "camera blocked warning",
    "CAMERA_DEGRADED": "camera image degraded warning",
    "CAMERA_RESTORED": "camera restored event",
    "DTC_SET": "diagnostic trouble code set",
    "TPMS_LOW": "low tyre-pressure warning",
    "WARN_BATT_OVERTEMP": "battery over-temperature warning",
    "LOW_12V": "low twelve-volt battery warning",
    "ADAS_DISENGAGE": "driver-assistance disengagement",
    "CHARGE_START": "charging session start",
    "CHARGE_END": "charging session end",
}


_SPEAK = [(r"\b(\d{1,2}) " + _M3RX + r"(?: \d{4})?, (\d{1,2}):(\d{2})(?::\d{2})?\b", lambda m: f"{m[1]} {_MONTHS[_MON3.index(m[2])]} at {int(m[3])}:{m[4]}"),
          (r"\b(\d{1,2}) " + _M3RX + r"\b(?: (\d{4}))?", lambda m: f"{m[1]} {_MONTHS[_MON3.index(m[2])]}" + (f" {m[3]}" if m[3] else "")),
          (r"\b(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::\d{2})?", lambda m: f"{int(m[3])} {_MONTHS[int(m[2]) - 1]} at {int(m[4])}:{m[5]}"), (r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: f"{int(m[3])} {_MONTHS[int(m[2]) - 1]}"),
          (r"(?<=\d) ?Wh/km", " watt-hours per kilometre"), (r"(?<=\d) ?Wh/mi", " watt-hours per mile"), (r"(?<=\d) ?km/h", " kilometres per hour"), (r"(?<=\d) ?mph", " miles per hour"), (r"(?<=\d) ?kWh", " kilowatt-hours"),
          (r"(?<=\d) ?kW\b", " kilowatts"), (r"(?<=\d) ?L/min", " litres per minute"), (r"(?<=\d) ?rpm\b", " revolutions per minute"), (r"(?<=\d) ?kPa", " kilopascals"), (r"(?<=\d) ?psi\b", " p s i"),
          (r"(?<=\d) ?km\b", " kilometres"), (r"(?<=\d) ?mi\b", " miles"), (r"(?<=\d) ?m/s2", " metres per second squared"), (r"(?<=\d) ?°?C\b", " degrees Celsius"), (r"(?<=\d) ?°?F\b", " degrees Fahrenheit"),
          (r"(?<=\d) ?%", " percent"), (r"\bp ?< ?0\.001", "p-value below 0.001"), (r"\bp ?= ?", "p-value "), (r"\bR2\b", "R squared"), (r"->|=>", " to "), (r"\bDTC\b", "diagnostic code"), (r"\bADAS\b", "A D A S"),
          (r"\bSoC\b", "state of charge"), (r"\bGPS\b", "G P S"), (r"\bEV\b", "E V"), (r"\bCAN\b", "CAN"), (r"[*_`#]", ""), (r"\s{2,}", " ")]


def speakable(text):
    # Use human ontology labels for every exact event token.  Keeping this at
    # the speech boundary leaves the visible, auditable event code untouched.
    text = re.sub(r"\b(?:" + "|".join(map(re.escape, _SPOKEN_EVENT_IDS)) + r")\b",
                  lambda m: _SPOKEN_EVENT_IDS[m.group(0)], text)
    for rx, rep in _SPEAK: text = re.sub(rx, rep, text)
    # SpeechSynthesis voices are inconsistent with a decimal adjacent to a
    # unit: some pause or spell the digits ("3 6 kilometres").  The speech
    # payload is allowed to be phonetic while the visible answer remains 3.6.
    text = re.sub(r"(?<![\w.])(-?\d+)\.(\d+)(?!\d)", lambda m: f"{m[1]} point {' '.join(m[2])}", text)
    return text.strip()


_TAGS = [(r"\s*\[PROBLEM\]", " - a problem"), (r"\s*\[watch\]", " - worth watching"), (r"\s*\[ok\]", " - fine"), (r"\s*\[[A-Za-z ]{2,12}\]", "")]


def spoken(full, brief):
    """What is read aloud: the brief, plus the items of a short bullet list when the brief is only its headline (otherwise a list is read as its first line only)."""
    lines = [l.strip() for l in full.split("\n") if l.strip()]
    items = [l for l in lines[1:] if l.startswith(("-", "*"))]
    if len(lines) < 2 or not items or len(items) > 8 or len(full) > 900 or not brief.strip().startswith(lines[0].lstrip("-* ").strip().rstrip(":")): return brief
    out = [lines[0].rstrip(":.") + ":" if lines[0].endswith(":") else brief.rstrip(".") + "."]
    for it in items:
        t = it.lstrip("-* ").strip()
        for rx, rep in _TAGS: t = re.sub(rx, rep, t)
        if t.rstrip(".")[:60] in brief: continue
        t = re.sub(r"^([A-Za-z ]{3,14}): ", lambda m: m[1].capitalize() + ": ", t)
        out.append(t.rstrip(".") + ".")
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ final polish of any answer text


def _dt(m, year):
    y, mo, d = int(m[1]), int(m[2]), int(m[3]); tail = m[4] or ""
    return f"{d} {_MON3[mo - 1]}" + (f" {y}" if year and y != year else "") + (f", {tail.strip()}" if tail.strip() else "")


def tidy(text, year=None):
    """Reader-friendly final form: human dates (14 Sep, 13:05), degree signs, 'to' instead of arrows between numbers, no double spaces, sentences start with a capital."""
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})( \d{1,2}:\d{2}(?::\d{2})?)?", lambda m: _dt(m, year), text)
    t = re.sub(r"(?<=\d) ?-> ?(?=[-+]?\d)", " to ", t)
    t = re.sub(r"(?<=\d) (C|F)\b(?!\w)", r" °\1", t)
    t = re.sub(r"(?<=\d)°\s?(C|F)\b", r" °\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"(?m)^(\s*)([a-z])", lambda m: m[1] + m[2].upper(), t)
    t = re.sub(r"(?<!e\.g)(?<!i\.e)(?<=[.!?]) ([a-z])", lambda m: " " + m[1].upper(), t)
    return t


_STATS = re.compile(r"\s*\((?:[^()]*?(?:\bp ?[<=]|\br = |\brho|\bR2|\bCI\b|\bt = |\bdf = |\bz = |\bHz\b|bootstrap|resamples|AIC)[^()]*?)\)")


def plain(text):
    """Drop the statistics in brackets from a spoken / short answer (they stay in the details)."""
    t = _STATS.sub("", text)
    t = re.sub(r"\s*\((?:illustrative code|an estimate[^)]*)\)", "", t)
    return re.sub(r"\s+([.,;:])", r"\1", t).strip()


# ------------------------------------------------------------------------------------------------ small talk and Hinglish
SMALL = [
    (re.compile(r"^(hi|hello|hey|hiya|yo|namaste|hola|howdy|hey there|hi there|hello there)( vehiclegpt| car| assistant| buddy)?[!. ]*$"), "Hello! I'm VehicleGPT. Ask me anything about your car - for example \"Brief me\", \"Is my car healthy?\" or \"How's my battery?\"."),
    (re.compile(r"^(thanks|thank you|thank you so much|thank you very much|thank you very much indeed|thanks a lot|thanks so much|thx|cheers|much appreciated|ta|thank u)( vehiclegpt| buddy)?[!. ]*$"), "You're welcome! Ask me anything else about the car."),
    (re.compile(r"^(good job|well done|great|perfect|nice|awesome|excellent|brilliant|wow|amazing|very good|that.s (great|good|helpful|perfect|nice)|helpful|great job|good answer|nice one)[!. ]*$"), "Glad that helped. Ask me anything else about the car."),
    (re.compile(r"^(ok|okay|alright|cool|sure|fine|got it|i see|understood|yes|yeah|yep|no|nope|hmm|hm)[!. ]*$"), "Okay. What would you like to know about the car?"),
    (re.compile(r"^(bye|goodbye|good bye|see you|see ya|good night|take care|later|cya)( vehiclegpt)?[!. ]*$"), "Goodbye - drive safely."),
    (re.compile(r"^(sorry|my bad|oops|excuse me|pardon)[!. ]*$"), "No problem. What would you like to know?"),
    (re.compile(r"^(how are you|how.s it going|how are you doing|what.s up|sup|how do you do)[?!. ]*$"), "I'm running fine - fully offline and ready. How can I help with the car?"),
    (re.compile(r"^(what.s your name|what is your name|what are you called|your name)[?!. ]*$"), None),
]
HING = set("bhai kya hai hain nahi nhi kaise kaisi kaisa gaadi gadi batao bata btao kitna kitni kitne kab kyun kyu theek thik mera meri mere mujhe aap tum hoga hogi karo chal chalo chalega chalegi accha acha haan bolo bol rahe raha rahi ho hum yaar kuch abhi sab bilkul sakta sakti kar karna lagta lag hawa pahiye haal dikkat samajh nahin kidhar kahan kahaan wala wali ka ki ke ko se mein me par".split())
HING_MAP = [
    (r"gaadi (ka )?(kya )?haal|gaadi kaisi|gadi kaisi|car kaisi|kya haal hai", "how is the car doing"),
    (r"(gaadi|gadi|car).{0,12}(theek|thik|sahi|badhiya)|sab theek|koi (problem|dikkat|issue|fault|kharabi)|kuch kharab", "is my car healthy"),
    (r"battery (kitni|kitna|kitne) (bachi|bacha|bachi hai|hai|left|charge)|charge (kitna|kitni)|battery (kaisi|kaisa|ka haal)", "how's my battery"),
    (r"range (kitna|kitni)|kitna (chalega|chalegi|chal sakti|chal sakta)|kitni door (chalegi|chalega|jaa)", "what's my range"),
    (r"(last |pichli |akhri )?charge (kab|kab kiya|kab hua)|charge kab", "when did i last charge"),
    (r"(tyre|tire|tayar|pahiye|tyres).{0,15}(hawa|pressure|theek|kaise|kaisi)|hawa kitni", "how are my tyres"),
    (r"(safe|surakshit).{0,10}(hai|hoga)|chala sakta|chalana safe|ja sakta (hu|hoon)|jaana safe|long trip pe jaa", "is it safe to keep driving"),
    (r"service kab|service (due|baaki)|servicing", "when is my next service"),
    (r"(battery|pack).{0,10}(garam|zyada garam|hot)|overheat", "was the battery ever hot"),
    (r"(kitna|kitne km|kitni door).{0,10}(chala|chali|chale|drive)|aaj kitna|is hafte kitna", "how many km did i drive this week"),
    (r"mausam|barish|baarish", "is it raining"),
    (r"kharcha|kitne paise|paisa", "how much did i spend on charging"),
]


HINGLISH_NOTE = ("Mujhe abhi sirf English mein poochhe gaye sawaal samajh aate hain / I understand questions in English best - and a few Hindi phrases like \"gaadi ka haal kya hai\". "
                 "Try: \"How's my battery?\", \"Is it safe to drive?\" or \"Brief me\".")


def smalltalk(t):
    """-> reply text for greetings / thanks / praise / goodbye, or None."""
    t = t.strip().lower(); t0_ = t
    t = re.sub(r"^((ok|okay|alright|all right|cool|great|nice|fine|well|so|oh|yes|yeah|yep|right|hmm)[ ,!.]+)+(?=\S)", "", t) or t0_
    for rx, rep in SMALL:
        if rx.search(t): return rep if rep is not None else ABOUT_TEXT
    return None


def hinglish(t):
    """-> ('translate', english question) | ('note', reply) | None, for questions written in Hindi-English ('gaadi ka kya haal hai')."""
    toks = re.findall(r"[a-z]+", t.lower())
    if len({w for w in toks if w in HING and w not in ("bhai", "yaar")}) < 2 and not re.search(r"\b(gaadi|gadi|kitna|kitni|batao|haal)\b", t.lower()): return None
    for rx, en in HING_MAP:
        if re.search(rx, t.lower()): return "translate", en
    return "note", HINGLISH_NOTE
