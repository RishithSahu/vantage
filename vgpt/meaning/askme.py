"""Questions that need no language model: the assistant itself, the clock, what lies outside the vehicle's data, the service record, and "how much battery will 50 km use".

`handle(brain, text)` -> {"answer", "intent", "facts", "read"} or None.  `text` is the question after `canon.normalise`.  Every number comes from the vehicle's own tables; what cannot be known says so (docs/CANON.md)."""
import datetime as dt
import math
import os
import re

from .. import analytics as A

NL = chr(10)


def _clock(brain):
    tz = int(getattr(brain, "tz", 0) or 0); return dt.datetime.fromtimestamp(int(brain.now) + tz, dt.timezone.utc)


def _meta(c, *keys):
    try: return dict(c.execute("SELECT k, v FROM meta WHERE k IN (%s)" % ",".join("?" * len(keys)), keys).fetchall())
    except Exception: return {}


def _model_name(brain):
    m = _meta(brain.c, "veh.model"); return m.get("veh.model") or "this vehicle"


# ------------------------------------------------------------------------------------------------------------------ what the assistant is
def identity(brain, m, text):
    return NL.join([f"I am Vantage, the assistant of {_model_name(brain)}. I answer from what this vehicle recorded, I work entirely offline on this device, and I do not send your questions or its data anywhere.",
                    "Ask me about the battery, range and charging, trips and driving, tyres, warnings and fault codes, service and maintenance."])


def how_it_works(brain, m, text):
    ad = os.path.basename(os.environ.get("VGPT_MEANING_ADAPTER", "") or "the meaning adapter")
    return NL.join(["A small language model - Qwen2.5 with 0.5 billion parameters, plus a small adapter trained on vehicle questions (" + ad.replace(".gguf", "") + ") - runs on this device. It does one job: it turns your question into a structured query (what to look up, for which period).",
                    "Ordinary program code then reads this vehicle's recorded data and computes the numbers. The model never makes up a value: when the data is not there, I say so.",
                    "Everything stays on this device; there is no network connection."])


def capabilities(brain, m, text):
    return NL.join(["I answer questions about this vehicle's own recorded data. For example:",
                    " - Battery and range: \"How much battery is left?\", \"How long to charge to 80%?\", \"What is my range?\"",
                    " - Trips and driving: \"How many trips this week?\", \"What was my longest trip?\", \"How am I driving?\", \"Consumption per month\"",
                    " - Health: \"Is my car okay?\", \"How are my tyres?\", \"Any warnings this week?\", \"When was the last service?\"",
                    " - History and trends: \"Is my battery health dropping?\", \"Is my consumption higher on weekends?\", \"Is there a pattern in when the warnings appear?\"",
                    "You can type a question or use the microphone."])


def clock_now(brain, m, text):
    d = _clock(brain); return f"It is {d:%H:%M} on {d:%A} {d.day} {d:%B %Y} on this vehicle's clock."


OUT_OF_SCOPE = {
    "law": "Traffic rules, driving law and licences are outside what I can know: I only read this vehicle's recorded data.",
    "buy": "I cannot tell where to buy or get a car or a part: I only know what this vehicle itself recorded.",
    "price": "Prices and costs of cars, fuel or electricity are not in this vehicle's data, so I cannot say.",
    "weather": "I have no weather forecast: I only know what this vehicle's sensors recorded (for example the outside temperature and rain it measured).",
    "route": "I cannot plan routes or give directions; I can tell you about the trips this vehicle recorded.",
    "other": "That is outside what I can know: I only read this vehicle's recorded data (battery, trips, tyres, warnings, service and so on).",
}


def out_of_scope(kind):
    def f(brain, m, text): return OUT_OF_SCOPE[kind] + NL + "Try asking about its battery, trips, tyres, warnings or service."
    return f


# ------------------------------------------------------------------------------------------------------------------ the service record
def _service(brain):
    from .. import analytics_x as AX
    f = AX.service(brain.c, brain.now); meta = _meta(brain.c, "veh.last_service_km", "veh.last_service_ts", "veh.service_km")
    if not f.get("ok"): return None, meta
    return f, meta


def service_record(brain, m, text):
    """The last service (when and at what odometer), the distance since, the interval, and what is due."""
    t = " " + text.lower() + " "; f, meta = _service(brain)
    if f is None: return "This vehicle has no usable odometer record, so I cannot say anything about its service."
    tz = int(brain.tz or 0); odo = f["odo_km"]
    try: last_ts = float(meta["veh.last_service_ts"]); last_km = float(meta["veh.last_service_km"])
    except (KeyError, ValueError): return f"The odometer is {odo:,} km, but this vehicle has no recorded last service, so I cannot say when it was serviced or when the next service is due."
    when = dt.datetime.fromtimestamp(last_ts + tz, dt.timezone.utc); since = f.get("km_since_service", round(odo - last_km)); iv = f.get("service_interval_km")
    months = f.get("months_since")
    lines = [f"The last recorded service was on {when.day} {when:%B %Y}, at {last_km:,.0f} km on the odometer."]
    lines.append(f"Since then the car has driven {since:,} km" + (f" in about {months:.0f} months." if months is not None else "."))
    if iv:
        left = f.get("km_to_service")
        if f.get("due") or (left is not None and left <= 0): lines.append(f"The service interval is {iv:,} km, so the next service is overdue" + (f" by {-left:,} km" if left is not None and left < 0 else "") + ".")
        else: lines.append(f"The service interval is {iv:,} km, so the next service is due in about {left:,} km" + (f" (or in about {max(0, f['months_to_service']):.0f} months)" if f.get("months_to_service") is not None else "") + ".")
    asked_last = bool(re.search(r"\b(?:last|previous|latest|done|history|was|did|when)\b", t)) and not re.search(r"\b(?:next|due|need|should|overdue)\b", t)
    if asked_last: return NL.join(lines[:2] + lines[2:])
    return NL.join(lines)


# ------------------------------------------------------------------------------------------------------------------ "how much battery will 50 km use"
def trip_energy(brain, m, text):
    t = text.lower(); km = float(m.group("n").replace(",", ".")); unit = m.group("u")
    can_ = bool(re.search(r"\b(?:can|could|will|would|enough|make it|reach)\b", t)) and not re.search(r"\b(?:drop|use|used|consume|lose|take|burn)\b", t)             # "can I drive 400 km": a yes or no, not a quantity
    if unit.startswith("mi"): km *= 1.609344
    try:
        from . import fuelx as FX
        cp = FX.caps_of(brain.resolver); ice = FX.CAPS.is_combustion(cp)
    except Exception: ice = False
    if ice:
        e = FX.recent_economy(brain)
        if not e: return "I cannot estimate fuel for a trip: there is not enough recent driving with fuel data."
        litres = km * e["l100"] / 100.0; tank = cp.get("tank_l")
        lv = FX._latest(brain, "fuel_level")
        if lv is not None and tank and can_:
            have = lv[0] / 100.0 * tank; rng = have / e["l100"] * 100.0
            return NL.join([("Yes" if km <= rng else "No") + f" - {km:.0f} km needs about {litres:.1f} L at your recent {e['l100']:.1f} L/100 km, and the tank holds about {have:.0f} L ({lv[0]:.0f} %), which is about {rng:.0f} km of range.", "It is an estimate from your own recent driving, not a guarantee."])
        note = ["This vehicle has no traction battery; its energy is fuel."] if any(w in t for w in ("battery", "charge", "soc", "kwh", "percentage")) else []
        return NL.join(note + [f"At your recent {e['l100']:.1f} L/100 km, {km:.0f} km would use about {litres:.1f} L" + (f", about {100 * litres / tank:.0f} % of the {tank:.0f} L tank." if tank else "."),
                        "This is an estimate from your own recent driving; speed, load, hills and temperature change it."])
    er = A.est_range(brain.c, brain.now)
    if not er.get("ok"): return "I cannot estimate the energy for a trip: " + str(er.get("why", "there is not enough recent driving data.")) + "."
    cap = A.cap_kwh(brain.c); kwh = km * er["wh_km"] / 1000.0; pct = 100.0 * kwh / cap; soc = er["soc"]; after = soc - pct
    if can_:
        ok = km <= er["range_km"]
        return NL.join([("Yes" if ok else "No") + f" - {km:.0f} km needs about {kwh:.1f} kWh ({pct:.0f} % of the battery) at your recent {er['wh_km']:.0f} Wh/km, and the battery is at {soc:.0f} % (about {er['range_km']:.0f} km of range).",
                        "It is an estimate from your own recent driving, not a guarantee."])
    return NL.join([f"{km:.0f} km would use about {kwh:.1f} kWh, which is about {pct:.0f} percentage points of the battery's {cap:.0f} kWh: from {soc:.0f} % to about {max(after, 0):.0f} %.",
                    f"This uses your recent consumption of {er['wh_km']:.0f} Wh/km; speed, hills, temperature and heating change it, so it is an estimate, not a guarantee."])


# ------------------------------------------------------------------------------------------------------------------ the table
N = r"(?P<n>\d+(?:[.,]\d+)?)\s*(?P<u>km|kms|kilomet(?:re|er)s?|miles?|mi)\b"
def charge_state(brain, m, text):
    """Is the car charging now: the latest reading of the charge state, its age, the power, and what is left to a full battery."""
    c = brain.c; now = int(brain.now); tz = int(brain.tz or 0)
    row = c.execute("SELECT ts, soc, power, mode FROM sig_1s WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
    if row is None or now - row[0] > 900:
        r2 = c.execute("SELECT ts, soc, power, mode FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (now,)).fetchone()
        if r2 is not None and (row is None or r2[0] > row[0]): row = r2
    if row is None: return "This vehicle has no charging or battery reading, so I cannot say whether it is charging."
    try:
        from . import fuelx as FX
        if FX.CAPS.is_combustion(FX.caps_of(brain.resolver)): return None                                          # a fuel vehicle: the typed resolver gives the one standard "no traction battery" answer (fuelx.no_battery, route outcome:no_traction_battery)
    except Exception: pass
    ts, soc, pw, mode = row; soc = soc / 10.0 if soc is not None else None; age = now - ts; kw = abs(pw) / 1000.0 if pw is not None else 0.0
    if abs(pw or 0) < 100 and mode == 3: kw = 0.0
    when = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); stale = age > 600
    lead = (f"The last reading is from {when:%H:%M} on {when.day} {when:%B}" + (f" ({age / 3600:.0f} h ago)" if age >= 5400 else f" ({age / 60:.0f} min ago)") + ": ") if stale else ""
    lastc = c.execute("SELECT start, end, kind, kwh, soc0, soc1 FROM charges WHERE start<=? ORDER BY start DESC LIMIT 1", (now,)).fetchone()
    if mode == 3:
        cap = A.cap_kwh(c); extra = ""
        if soc is not None and kw > 0.5:
            to80 = max(0.0, (80 - soc) / 100 * cap / kw * 60) if soc < 80 else 0.0; to100 = max(0.0, (100 - soc) / 100 * cap / kw * 60)
            extra = (f" At this power it needs about {to80:.0f} min to reach 80 % and about {to100:.0f} min to be full." if soc < 80 else f" At this power it needs about {to100:.0f} min to be full.") if soc < 99.5 else " The battery is full."
        return lead + (f"Yes - the car is charging" + (f" at {kw:.1f} kW" if kw else "") + (f" and the battery is at {soc:.0f} %." if soc is not None else ".")) + extra
    state = {0: "parked", 1: "driving in town", 2: "driving on the highway"}.get(mode, "not charging")
    out = lead + f"No - the car is not charging; it is {state}" + (f" and the battery is at {soc:.0f} %." if soc is not None else ".")
    if lastc: out += f" The last charge was on {dt.datetime.fromtimestamp(lastc[0] + tz, dt.timezone.utc):%d %B at %H:%M} ({lastc[2]}, {lastc[3]:.1f} kWh, {lastc[4]:.0f} to {lastc[5]:.0f} %)."
    return out


def that_date(brain, m, text):
    """A follow-up to an answer about one moment ("which year", "what day was that"): the date of what the last answer was about."""
    ts = getattr(getattr(brain.dialogue, "last", None), "focus_ts", None)
    if not ts: return None                                                                                           # nothing to point back to: the model reads it as a new question
    d = dt.datetime.fromtimestamp(int(ts) + int(brain.tz or 0), dt.timezone.utc); t = text.lower()
    what = "year" if "year" in t else "month" if "month" in t else "day" if " day" in t else "date"
    return {"year": f"That was in {d.year}.", "month": f"That was in {d:%B %Y}.", "day": f"That was on {d:%A} {d.day} {d:%B %Y}.", "date": f"That was on {d.day} {d:%B %Y} at {d:%H:%M}."}[what]


def data_catalog(brain, m, text):
    from . import records as RC
    r = RC.inventory(brain); return r["answer"]


HANDLERS = [
    ("that_date", re.compile(r"^\s*(?:(?:which|what) (?:year|date|day|month)(?: (?:was|is) (?:that|it|this))?|when was (?:that|it)|what year|what year was that)\s*$"), that_date),
    ("charge_state", re.compile(r"\b(?:is|are|was)\b.{0,15}\b(?:car|vehicle|it|ev|battery|we|i)\b.{0,15}\b(?:charging|plugged in|on charge)\b|\bam i charging\b|\bstill charging\b|\bcharging (?:now|right now|currently)\b"), charge_state),
    ("data_catalog", re.compile(r"\b(?:what|which)\b.{0,25}\b(?:data|signals|readings|measurements|information|sensors)\b.{0,40}\b(?:record|records|recorded|report|reports|send|sends|collect|collects|store|stored|have|hold|available|log)\b|\b(?:information|info|details)\b.{0,30}\brecorded data\b|\blist (?:all )?(?:the )?(?:signals|sensors|data)\b|\bwhat (?:can you see|do you (?:know|see) about (?:the |my |this )?(?:car|vehicle))\b"), data_catalog),
    ("how_it_works", re.compile(r"\b(?:what|which) (?:llm|model|ai|language model|software|network|brain|engine|technology)\b.{0,30}\b(?:you|u|this|assistant|it)\b.{0,20}\b(?:run|use|using|based|powered|built|on)\b|\bwhat are you (?:running|built|based|powered)\b|\bhow do you work\b|\bwhat (?:llm|model) (?:are|do) you\b"), how_it_works),
    ("identity", re.compile(r"\b(?:who are you|what is your name|what are you called|who made you|who built you|who created you|are you (?:vantage|vehiclegpt|an? (?:ai|bot|robot|human|chatbot|person|assistant))|introduce yourself)\b"), identity),
    ("capabilities", re.compile(r"\b(?:what can you do|how can you help|what can i ask|what do you do|what are you for|help me|what can you tell|what questions)\b"), capabilities),
    ("clock", re.compile(r"^\s*(?:what(?:'s| is| was)?|tell me|give me|do you know) (?:the )?(?:current |exact )?(?:time|date|day)(?: right now| now| today| is it| please)?\s*$|^\s*what time is it(?: now| right now)?\s*$|^\s*what day is (?:it|today)\s*$|^\s*what(?:'s| is) today\s*$|^\s*(?:what|which) (?:is )?(?:today's )?date\s*$"), clock_now),
    ("trip_energy", re.compile(r"\b(?:battery|charge|soc|percentage|percent|energy|kwh|power|juice|fuel|petrol|diesel|gas|gasoline|litres?|liters?|tank)\b.{0,60}\b(?:drop|use|used|consume|need|lose|take|go down|fall|cost)\b.{0,40}?" + N), trip_energy),
    ("trip_energy2", re.compile(r"(?:\b(?:travel|drive|go|ride|cover|do)\b.{0,12}?|\b(?:how much|how many)\b.{0,25}\b(?:battery|charge|soc|percentage|percent|energy|kwh|power|juice|fuel|petrol|diesel|gas|gasoline|litres?|liters?|tank)\b.{0,30}?)" + N + r"(?:.{0,50}\b(?:battery|charge|percent|percentage|kwh|energy|juice|fuel|petrol|diesel|gas|gasoline|litres?|liters?|tank)\b|.{0,30}\b(?:use|need|take|consume|cost|burn)\b|\s*\??\s*$)"), trip_energy),
    ("trip_can", re.compile(r"\b(?:can|could|will|would|do i have enough)\b.{0,50}\b(?:drive|go|travel|reach|make it|get)\b.{0,25}?" + N), trip_energy),
    ("service", re.compile(r"\b(?:servic(?:e|ed|es|ing)|maintenance|maintained)\b"), service_record),
    ("law", re.compile(r"\b(?:side of the road|which side|traffic (?:rule|law|light rule)s?|driving (?:law|rule)s?|speed limit (?:law|here|in )|legal(?:ly)?|allowed to|driving licen[cs]e|driving test|road rules?|fine for|parking (?:rule|fine|ticket))\b"), out_of_scope("law")),
    ("buy", re.compile(r"\bwhere\b.{0,15}\b(?:get|buy|purchase|find|order)\b.{0,15}\b(?:this|the|a|my|an?)\b.{0,10}\b(?:car|vehicle|ev|model)\b|\b(?:car|vehicle) (?:dealer|dealership|showroom)\b|\bwhere to buy\b"), out_of_scope("buy")),
    ("price", re.compile(r"\b(?:how much (?:does|do|is|would|will) (?:this|the|a) (?:car|vehicle|ev) (?:cost|worth)|price of (?:this|the|a) (?:car|vehicle|ev)|price of (?:petrol|diesel|fuel|electricity)|(?:car|vehicle) price)\b"), out_of_scope("price")),
    ("weather", re.compile(r"\b(?:weather (?:forecast|tomorrow|today|report)|will it rain (?:tomorrow|later|today)|forecast)\b"), out_of_scope("weather")),
    ("route", re.compile(r"\b(?:directions? to|route to|navigate to|how do i get to|nearest (?:petrol|gas|charging|charger|station|hospital|restaurant)|where is the nearest)\b"), out_of_scope("route")),
]
OTHER_SUBJECT = re.compile(r"\b(?:compare|comparison|versus|vs|before and after|average|mean|total|trend|consumption|efficiency|economy|range|temperature|speed|energy|fuel|battery|charge|charging|tyre|tyres|brake|brakes|distance|trips?|warnings?|power|pressure|noise|cost|spent|drove|driven|driving)\b")
SERVICE_WORDS = re.compile(r"\b(?:last|previous|latest|history|done|when|how is|how's|status|due|next|overdue|need|needs|should|record|recorded|serviced|up to date|condition|interval|book|schedule|time for|go for|get|state of)\b")


def handle(brain, text):
    t = " " + " ".join(str(text).lower().split()).strip(" ?!.") + " "
    for name, rx, fn in HANDLERS:
        m = rx.search(t)
        if not m: continue
        if name == "service" and OTHER_SUBJECT.search(t): continue                                                  # "compare fuel consumption before and after my last service" is about consumption: the model reads it and the period is the service date
        if name == "service" and (not SERVICE_WORDS.search(t) or re.search(r"\b(?:service|maintenance) (?:log|logs|message|messages|line|lines)\b", t) or re.search(r"\b(?:brake|tyre|battery|cooling|motor)\b.{0,12}\b(?:service|maintenance)\b.{0,6}$", t) and False): continue
        try: ans = fn(brain, m, t)
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("askme %s failed: %s: %s", name, type(e).__name__, e); continue
        if ans: return {"answer": ans, "intent": "ASKME", "facts": {"ok": True, "route": "askme:" + name, "executor": "askme"}, "read": name.replace("_", " ")}
    return None
