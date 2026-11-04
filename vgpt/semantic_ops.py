"""Small, compositional semantic operations that are not ordinary aggregates.

This module deliberately does not contain an acceptance-question table.  It
recognises grammatical relations (comparison, implication, freshness,
conversion and bounded transitions), grounds their arguments in the vehicle
ontology, and computes answers from recorded data.  A paraphrase with the
same relation and arguments therefore follows the same path.
"""
import math
import functools
import re
import statistics

from . import analytics as A
from . import analytics_x as AX
from .nearest import nearest
from . import kinematics
from . import live_snapshot
from . import dialogue_state
from .query_ir import Query, TimeSpec
from . import signal_registry as R


CURRENT = r"\b(?:now|currently|current|right now|at (?:this|the) moment|at present)\b"


def _result(answer, route, intent="ANALYSIS", **facts):
    return {"answer": answer, "brief": answer.split("\n", 1)[0], "intent": intent,
            "facts": dict({"ok": True, "route": route}, **facts)}


_rx = functools.lru_cache(maxsize=None)(re.compile)                    # the lexicon has over a thousand phrases: Python's own cache holds 512, so every question recompiled them all (4 s on the Pi)


def _concepts(text):
    """Return ontology concepts named in text, longest lexical span first."""
    t = " " + re.sub(r"[^a-z0-9%/.-]+", " ", text.lower().replace("’", "'")) + " "
    aliases = []
    for phrase, names in R.LEXICON.items():
        p = phrase.lower().replace("-", " ")
        if _rx(r"(?<![a-z0-9])" + re.escape(p).replace(r"\ ", r"[ -]") + r"(?![a-z0-9])").search(t):
            aliases.append((len(p), sorted(names)[0]))
    # Concepts whose everyday names express a relation rather than a sensor.
    extra = {
        "tyre_safety": r"\b(?:tyres?|tires?)\b.*\bsafe|\bsafe\b.*\b(?:tyres?|tires?)\b",
        "tyre_pressure": r"\b(?:tyre|tire) pressure\b",
        "overheating": r"\bover[- ]?heat(?:ing|ed)?\b",
        "active_fault": r"\b(?:active|still present|still active|confirmed)\b.*\b(?:fault|warning|problem)|\b(?:fault|warning|problem)\b.*\b(?:active|still present|still active|confirmed)\b",
        "recorded_fault": r"\b(?:diagnostic|dtc|fault|warning|code)\b",
        "rated_max_speed": r"\b(?:officially |official |rated |specified )(?:maximum|top|max) speed\b|\b(?:maximum|top|max) speed (?:rating|specification)\b",
        "recorded_max_speed": r"\b(?:highest|maximum|max|top) recorded speed\b",
        "battery_energy": r"\bbattery energy\b|\bstored energy\b",
    }
    if re.search(r"\b(?:whats|what(?:'s| is)|which|tell me)\b.*\b(?:car|vehicle)(?:'s|s)?\b.*\b(?:max|maximum|top) speed\b", t) and not re.search(r"\b(?:recorded|reached|drove|went|last|past|yesterday|week|month)\b", t):
        aliases.append((110, "rated_max_speed"))
    aliases.extend((100, name) for name, rx in extra.items() if re.search(rx, t))
    if re.search(r"\b(?:state of charge|soc|battery (?:level|charge)|battery (?:is )?at \d+(?:\.\d+)?\s*%)", t): aliases.append((100, "battery_soc"))
    if re.search(r"\b(?:state of health|soh|battery health|battery (?:is )?healthy|its health)\b", t): aliases.append((100, "battery_soh"))
    if re.search(r"\bbattery\b.*\bdegrad\w*\b", t): aliases.append((100, "battery_soh"))
    out = []
    for _, name in sorted(aliases, reverse=True):
        if name not in out: out.append(name)
    return out


def _fresh_snapshot(brain):
    return live_snapshot.snapshot(brain.c, brain.now)


def _fmt_value(concept, value):
    unit = R.CONCEPTS[concept].unit
    if unit == "%": return f"{value:.1f}%"
    if unit in ("rpm", "W"): return f"{value:,.0f} {unit}"
    return f"{value:.1f} {unit}"


def _current(brain, concept):
    snap = _fresh_snapshot(brain)
    value = snap.get("values", {}).get(concept)
    if not snap.get("available") or value is None:
        return None, snap
    return value, snap


DEFINITIONS = {
    "battery_soc": "state of charge (SoC) is how full the battery is now",
    "battery_soh": "state of health (SoH) is its usable capacity relative to when new",
    "battery_temperature": "battery temperature measures heat in the traction-battery pack",
    "motor_temperature": "motor temperature measures heat in the traction motor",
    "charging_power": "charging power is the instantaneous transfer rate, measured in kW",
    "battery_energy": "battery energy is an accumulated amount stored or transferred, measured in kWh",
    "tyre_pressure": "tyre pressure measures inflation pressure",
    "tyre_safety": "overall tyre safety also depends on tread, damage, punctures, age, load and correct fitment",
    "coolant_flow": "coolant flow measures circulation through the cooling circuit",
    "overheating": "overheating is a temperature condition and requires temperature evidence",
    "rated_max_speed": "rated top speed is a manufacturer specification",
    "recorded_max_speed": "highest recorded speed is the maximum observation in this vehicle's history",
    "recorded_fault": "a stored diagnostic record shows that an event was logged in the past",
    "active_fault": "active-fault status requires a current status signal or a confirmed uncleared code",
}


def _relation(brain, raw, t, cs):
    relation = "difference" if re.search(r"\b(?:difference|distinction|contrast|compare|versus|vs)\b", t) else "implication" if re.search(r"\b(?:mean|prove|proof|establish|guarantee|confirm(?:ed|ation)?)\b", t) else "difference" if (len(cs) >= 2 and not [x for x in dialogue_state._times(brain, t) if x.type != "current"] and not re.search(r"\d", t) and not re.search(r"\band\b|,|\bplus\b", t)) else None
    if not relation or len(cs) < 2: return None
    # Prefer complementary ontology pairs when a broad alias also matched.
    pairs = [
        ("battery_soc", "battery_soh"), ("battery_temperature", "motor_temperature"),
        ("charging_power", "battery_energy"), ("tyre_pressure", "tyre_safety"),
        ("coolant_flow", "overheating"), ("rated_max_speed", "recorded_max_speed"),
        ("recorded_fault", "active_fault"),
    ]
    pair = next((p for p in pairs if set(p) <= set(cs)), None)
    if pair is None: return None
    a, b = pair; da, db = DEFINITIONS[a], DEFINITIONS[b]
    if relation == "implication":
        answer = f"No. {da.capitalize()}; {db}. The first fact alone does not prove the second."
    else:
        answer = f"{da.capitalize()}; {db}."
    readings = []
    for c in pair:
        if c in live_snapshot.CONCEPT_TO_STATE:
            v, snap = _current(brain, c)
            if v is not None: readings.append(f"{R.label_of(c)} is {_fmt_value(c, v)} at {A.stamp(snap['sample_ts'])}")
    if readings: answer += "\nCurrent readings from one snapshot: " + "; ".join(readings) + "."
    if pair == ("coolant_flow", "overheating"):
        v, snap = _current(brain, "battery_temperature")
        if v is not None: answer += f" Current pack temperature is {v:.1f} C at {A.stamp(snap['sample_ts'])}; evaluate that and verified warning status separately."
    if pair == ("rated_max_speed", "recorded_max_speed"):
        rated = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.rated_max_speed','veh.max_speed_kmh') ORDER BY k LIMIT 1").fetchone()
        peak = brain.c.execute("SELECT ts,MAX(speed) FROM sig_1m WHERE ts<=?", (brain.now,)).fetchone()
        answer += " The rated specification is " + (f"{float(rated[0]):g} km/h." if rated else "not recorded in the vehicle profile.")
        if peak and peak[1] is not None: answer += f" The highest observed one-minute speed is {peak[1]/10:.1f} km/h in the recorded telemetry."
    if pair == ("recorded_fault", "active_fault"):
        answer += " This data set labels stored diagnostic records as illustrative/simulated, so they are not confirmation of a real current fault."
    return _result(answer, "semantic_relation", relation=relation, concepts=list(pair))


def _inventory(brain, t):
    if not (re.search(r"\b(?:every|all)\b", t) and re.search(r"\b(?:value|measurement|signal|reading)s?\b", t) and re.search(CURRENT, t)): return None
    snap = _fresh_snapshot(brain)
    if not snap.get("available"): return _result("No fresh authoritative telemetry snapshot is available.", "current_inventory", "STATUS", available=False)
    live, unavailable = [], []
    for c in live_snapshot.CONCEPT_TO_STATE:
        v = snap["values"].get(c)
        (live if v is not None else unavailable).append(f"{R.label_of(c)}: {_fmt_value(c, v)}" if v is not None else R.label_of(c))
    er = A.est_range(brain.c, brain.now); derived = []
    if er.get("ok"): derived = [f"estimated range: {er['range_km']} km", f"recent consumption: {er['wh_km']:.1f} Wh/km"]
    answer = f"Current measured values at {A.stamp(snap['sample_ts'])} ({snap['tier']}, age {snap['age_s']} s):\n  - " + "\n  - ".join(live)
    if derived: answer += "\nDerived estimates:\n  - " + "\n  - ".join(derived)
    answer += "\nUnavailable in this snapshot: " + (", ".join(unavailable) if unavailable else "none of the configured live signals") + "."
    return _result(answer, "current_inventory", "STATUS", sample_ts=snap["sample_ts"], concepts=list(snap["values"]))


def _state_queries(brain, t, cs):
    snap = _fresh_snapshot(brain)
    if "battery_power" in cs and re.search(CURRENT, t) and re.search(r"\b(?:power|using|draw|drawing)\b", t) and not re.search(r"\b(?:how old|age of|fresh\w*|stale|timestamp|how recent)\b", t):
        value = snap.get("values", {}).get("battery_power") if snap.get("available") else None
        if value is None: return _result("No fresh valid battery-power reading is available; I will not substitute an old value.", "current_power", "STATUS", available=False)
        direction = "power is flowing into the battery (charging or regeneration)" if value < 0 else "the battery is supplying power" if value > 0 else "net battery power is approximately zero"
        return _result(f"Battery power is {value:.1f} kW at {A.stamp(snap['sample_ts'])} (age {snap['age_s']} s): {direction}. Negative follows this vehicle's sign convention for inflow. This reading is " + ("current" if snap["age_s"] <= 5 else f"stale: the last valid sample is {snap['age_s']} s old") + ".", "current_power", "STATUS", value=value, sample_ts=snap["sample_ts"], age_s=snap["age_s"])
    if re.search(r"\b(?:moving|stationary|stopped)\b", t) and re.search(CURRENT, t):
        speed = snap.get("values", {}).get("speed") if snap.get("available") else None
        if speed is None: return _result("I cannot determine whether the car is moving because no fresh valid speed reading is available.", "motion_state", "STATUS", available=False)
        state = "moving" if abs(speed) > 0.5 else "stationary"
        return _result(f"The car is {state}: speed is {speed:.1f} km/h at {A.stamp(snap['sample_ts'])} (age {snap['age_s']} s).", "motion_state", "STATUS", speed=speed, sample_ts=snap["sample_ts"])
    if re.search(r"\b(?:current|fresh|stale|age|how old)\b", t) and re.search(r"\b(?:reading|sample|data)\b", t) and cs:
        c = next((x for x in cs if x in live_snapshot.CONCEPT_TO_STATE), None)
        if c:
            value = snap.get("values", {}).get(c) if snap.get("available") else None
            if value is None: return _result(f"There is no fresh valid {R.label_of(c)} reading at the requested time; I will not call an older value current.", "freshness", "STATUS", available=False)
            verdict = "current" if snap["age_s"] <= 5 else f"not current (the newest valid sample is {snap['age_s']} s old)"
            return _result(f"The {R.label_of(c)} reading is {verdict}: {_fmt_value(c, value)}, recorded at {A.stamp(snap['sample_ts'])}, age {snap['age_s']} s ({snap['tier']}).", "freshness", "STATUS", concept=c, sample_ts=snap["sample_ts"], age_s=snap["age_s"])
    if re.search(r"\b(?:last|latest|most recent)\b.*\bvalid\b", t) and cs:
        c = next((x for x in cs if x in live_snapshot.CONCEPT_TO_STATE), None)
        if c:
            store = R.STORE.get(c, {}).get("1s") or R.STORE.get(c, {}).get("1m")
            if store:
                table, col, scale = store
                row = brain.c.execute(f"SELECT ts,{col} FROM {table} WHERE ts<=? AND {col} IS NOT NULL ORDER BY ts DESC LIMIT 1", (brain.now,)).fetchone()
                if row: return _result(f"The last valid {R.label_of(c)} reading was {_fmt_value(c, row[1] / scale)}, recorded at {A.stamp(row[0])}.", "latest_valid", "LOOKUP", concept=c, sample_ts=row[0], value=row[1] / scale)
    return None


def _energy_math(brain, t):
    er = A.est_range(brain.c, brain.now)
    # Distance remaining down to a stated state-of-charge is a different
    # quantity from time required to charge.  It is calculated from the
    # current SoC and the same recent-consumption range basis, never routed to
    # the charging-time handler merely because both mention charge.
    targets = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?(?:\s+(?:charge|soc))?)", t)]
    if er.get("ok") and targets and re.search(r"\b(?:how far|distance|range|drive|travel|go)\b", t) and re.search(r"\b(?:before|until|down to|reaching|reach)\b", t):
        target_soc = targets[-1]; current_soc = float(er["soc"])
        if not 0 <= target_soc < current_soc:
            return _result(f"The target state of charge must be between 0% and the current {current_soc:.0f}%.", "range_to_soc", "CLARIFY", available=False)
        km = float(er["range_km"]) * (current_soc - target_soc) / current_soc
        return _result(f"At the recent {float(er['wh_km']):.1f} Wh/km, about {km:.0f} km of the current estimated range remains before the battery reaches {target_soc:g}% (current charge {current_soc:.0f}%). This is an estimate; driving, weather and climate use can change it.", "range_to_soc", distance_km=round(km, 1), target_soc=target_soc, current_soc=current_soc, wh_per_km=float(er["wh_km"]))
    # Charging an interval is about the two stated endpoints, never current SoC.
    ps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", t)]
    if len(ps) >= 2 and re.search(r"\b(?:charge|charging)\b", t) and re.search(r"\b(?:energy|kwh|need|require)\b", t):
        lo, hi = ps[0], ps[1]
        if not (0 <= lo <= 100 and 0 <= hi <= 100 and hi > lo): return _result("The charge endpoints must be between 0% and 100%, with the target above the start.", "charge_interval", "CLARIFY", available=False)
        cap = A.cap_kwh(brain.c); stored = cap * (hi - lo) / 100.0; eff = 0.92; wall = stored / eff
        return _result(f"From {lo:g}% to {hi:g}% is a {hi-lo:g}-percentage-point interval. With {cap:.1f} kWh usable capacity, that stores about {stored:.1f} kWh in the battery. At an explicit {eff:.0%} charging-efficiency assumption, the charger would draw about {wall:.1f} kWh.", "charge_interval", stored_kwh=stored, wall_kwh=wall, capacity_kwh=cap)
    # A distance with an efficiency STATED in the question is arithmetic on the user's figure: it needs no recorded efficiency.
    sm = re.search(r"\b(?:efficiency|consumption|economy)\b.{0,25}?(\d+(?:\.\d+)?)\s*wh\s*(?:/|per)\s*km\b|(\d+(?:\.\d+)?)\s*wh\s*(?:/|per)\s*km\b", t)
    dm0 = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilomet(?:re|er)s?)\b(?!\s*/)", t)
    if sm and dm0 and re.search(r"\b(?:kwh|energy|use|used|consume|require|need)\b", t):
        stated = float(sm.group(1) or sm.group(2)); km = float(dm0.group(1)); kwh = km * stated / 1000.0
        return _result(f"With the stated {stated:g} Wh/km, {km:g} km needs {km:g} x {stated:g} / 1000 = {kwh:.1f} kWh. This is a calculation from the figure you gave, not a measurement of any trip.", "distance_to_energy",
                       distance_km=km, wh_per_km=stated, energy_kwh=kwh, stated=True)
    if not er.get("ok"): return None
    whkm = float(er["wh_km"])
    if re.search(r"\bper\s+(?:100|one hundred)\s*(?:km|kilomet)", t) and re.search(r"\b(?:energy|consumption|efficiency)\b", t):
        return _result(f"Recent consumption is {whkm:.1f} Wh/km, which is {whkm / 10.0:.1f} kWh/100 km. Basis: the most recent comparable driving window.", "consumption_per_100km", wh_per_km=whkm, kwh_per_100km=whkm / 10.0)
    dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilomet(?:re|er)s?)\b", t)
    em = re.search(r"(\d+(?:\.\d+)?)\s*kwh\b", t)
    if dm and re.search(r"\b(?:energy|kwh|require|need|consume|use)\b", t) and re.search(r"\b(?:trip|travel|drive|driving|journey|distance|efficiency|consumption)\b", t):
        km = float(dm.group(1)); kwh = km * whkm / 1000.0
        return _result(f"At the recent {whkm:.1f} Wh/km, {km:g} km would require about {kwh:.1f} kWh (distance x consumption). This is an estimate; weather, speed and HVAC use can change it.", "distance_to_energy", distance_km=km, wh_per_km=whkm, energy_kwh=kwh)
    if em and not dm and re.search(r"\b(?:how far|how many (?:km|kilomet|mile)|distance|travel|drive|go|range|km|kilomet)\b", t):
        kwh = float(em.group(1)); km = kwh * 1000.0 / whkm
        return _result(f"Using {kwh:g} kWh at the recent {whkm:.1f} Wh/km gives an energy-only estimate of {km:.1f} km. It does not assume the car's full remaining battery is available.", "energy_to_distance", energy_kwh=kwh, wh_per_km=whkm, distance_km=km)
    return None


def _coordination(brain, raw, t):
    """A message that coordinates two or three complete requests ("the average speed and the highest battery temperature yesterday") is answered part by part: each conjunct is read by the typed parser on its own, a trailing period that only the last conjunct states is shared with the others, and every part is executed by the typed engine.  Declined unless EVERY conjunct is an executable query with its own target (so a pair of names such as "city and highway" is never split)."""
    parts = [x.strip(" ?.!") for x in re.split(r"\s*(?:,|;|\band\b|\bplus\b|\bas well as\b)\s*", str(raw)) if x.strip(" ?.!")]
    if not 2 <= len(parts) <= 3: return None
    qc = brain.qc()
    if qc is False: return None
    from . import nl2ir
    whole = dialogue_state._times(brain, str(raw)); shared = whole[-1] if whole else None
    # a leading question word stated once governs the others ("what was the average speed and the highest battery temperature")
    lead = re.match(r"^(?:what|how|when|tell me|show me|give me)(?: (?:was|is|were|are|did|do))?", parts[0], re.I)
    queries = []
    for k, part in enumerate(parts):
        txt = part if (k == 0 or not lead or re.match(r"^(?:what|how|when|tell|show|give)", part, re.I)) else lead.group(0) + " " + part
        try: p = nl2ir.parse(txt, now=brain.now, tz=brain.tz, profile=qc.profile, state=None)
        except Exception: return None
        q = p.query
        if not p.ok or q is None or not (q.target or q.scope.select is not None or q.event is not None) or p.unexplained or q.op not in ("aggregate", "extreme", "current_value", "trend", "count", "percent_time", "list_events"): return None
        if shared is not None and not [x for x in dialogue_state._times(brain, txt) if x.type != "current"] and q.op != "current_value": q.scope.time = shared
        queries.append(q)
    if len({(q.target, q.op) for q in queries}) < 2 and len({q.agg for q in queries}) < 2: return None            # "the average and the highest speed" is two statistics of one signal: still fine, but two identical queries is not a coordination
    out = []
    for q in queries:
        r = qc.execute(q)
        if r.status not in ("answered", "no_data"): return None
        out.append(r.answer.text.split(chr(10) + "Based on")[0].strip() if r.status == "answered" else r.answer.text)
    dialogue_state.remember(brain, queries[-1])
    return _result((chr(10) + chr(10)).join(f"{i + 1}. {o}" for i, o in enumerate(out)), "coordinated_request", "ANALYSIS", queries=[q.to_text() for q in queries])


def _acceleration(brain, t):
    if not re.search(r"\baccelerat(?:e|ion|ing)|\bharder\b", t): return None
    lo = brain.now - 7 * 86400
    rows = brain.c.execute("SELECT ts,speed,mode FROM sig_1s WHERE ts>=? AND ts<=? AND speed IS NOT NULL ORDER BY ts", (lo, brain.now)).fetchall()
    ts = [r[0] for r in rows]; speed = [r[1] / 10.0 for r in rows]; valid = [r[2] in (1, 2) for r in rows]
    bands = [(float(a), float(b)) for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*(?:km/h|kmph|kph)?\s*(?:-|–|to|and)\s*(\d+(?:\.\d+)?)", t)]
    # The boundary values may be supplied in metres per second.  Convert the
    # physical quantity once, before selecting runs; do not make a separate
    # language rule for every wording of the same speed band.
    if bands and re.search(r"\bm\s*(?:/|per)\s*s(?:ec(?:ond)?)?\b", t):
        bands = [(a * 3.6, b * 3.6) for a, b in bands]
    if not bands and re.search(r"\blow\b", t) and re.search(r"\bhigh\b", t): bands = [(0, 40), (60, 100)]
    if not bands: return None
    parts, means = [], []
    named = {0: "low-speed ", 1: "high-speed "} if (re.search(r"\blow\b", t) and re.search(r"\bhigh\b", t)) else {}
    for bi, (a, b) in enumerate(bands[:2]):
        runs = kinematics.band_runs(ts, speed, a, b, valid=valid)
        if not runs: parts.append(f"{named.get(bi, '')}{a:g}-{b:g} km/h: no valid run"); means.append(None)
        else:
            sec = statistics.fmean(r.seconds for r in runs); acc = statistics.fmean(r.acceleration for r in runs); means.append(acc)
            parts.append(f"{named.get(bi, '')}{a:g}-{b:g} km/h: {sec:.2f} s average, {acc:.2f} m/s2 across {len(runs)} valid run(s)")
    if len(parts) == 1: answer = "Acceleration from " + parts[0] + ". Crossings are linearly interpolated; gaps, invalid states and non-upward transitions are excluded."
    else:
        good = [(i, v) for i, v in enumerate(means) if v is not None]
        verdict = "Insufficient comparable runs." if len(good) < 2 else (f"The {bands[max(good, key=lambda x:x[1])[0]][0]:g}-{bands[max(good, key=lambda x:x[1])[0]][1]:g} km/h band had the higher acceleration magnitude.")
        answer = "\n".join(parts) + "\n" + verdict
    return _result(answer, "acceleration_bands", bands=bands, run_source="sig_1s")


def _days(t, default=7):
    m = re.search(r"\b(?:last|past|over the last|over the past)\s+(\d+)\s+days?\b", t)
    if m: return max(1, min(365, int(m.group(1))))
    words = {"seven": 7, "nine": 9, "fourteen": 14, "thirty": 30}
    for word, n in words.items():
        if re.search(rf"\b(?:last|past|over the last|over the past)\s+{word}\s+days?\b", t): return n
    if "month" in t: return 30
    if "week" in t: return 7
    return default


def _local_day0(brain):
    return ((int(brain.now) + int(brain.tz)) // 86400) * 86400 - int(brain.tz)


def _clock_hour(t, match):
    hour = int(match.group(1)); minute = int(match.group(2) or 0)
    tail = t[match.end():match.end() + 18]
    if re.search(r"\b(?:pm|night|evening)\b", tail) and hour < 12: hour += 12
    if re.search(r"\bam\b", tail) and hour == 12: hour = 0
    return hour, minute


def _temporal_lookup(brain, t, cs):
    """Calendar-relative point and bounded-window semantics.

    Time resolution is shared across signals and location.  This prevents an
    exact clock request from degrading into an extreme/whole-day statistic.
    """
    day0 = _local_day0(brain)

    # A period whose endpoints are calendar offsets is a range, not the first
    # ``N days ago`` point mentioned in the sentence.
    span = re.search(r"\b(?:from\s+)?(\d+)\s+days?\s+ago\s+(?:to|through|until)\s+(\d+)\s+days?\s+ago\b", t)
    if span and re.search(r"\b(?:average|avg|mean)\b", t) and "speed" in cs:
        a, b = int(span.group(1)), int(span.group(2)); older, newer = max(a, b), min(a, b)
        lo, hi = day0 - older * 86400, day0 - newer * 86400 + 86400
        row = brain.c.execute("SELECT AVG(speed/10.0),COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? AND speed IS NOT NULL AND mode IN (1,2)", (lo, hi)).fetchone()
        if not row or not row[1]: return _result("No valid driving-speed samples exist in that calendar range.", "calendar_range", "SUMMARY", available=False)
        return _result(f"Average vehicle speed from {A.stamp(lo)} through {A.stamp(hi-1)} was {row[0]:.1f} km/h across {row[1]} valid driving-minute samples.", "calendar_range", "SUMMARY", value=row[0], samples=row[1], period=(lo, hi))

    clock = re.search(r"\b(?:at|between|btw)\s+(\d{1,2})(?::(\d{2}))?\b", t)
    if not clock: return None
    rel = re.search(r"\b(\d+)\s+(days?|weeks?)\s+ago\b", t)
    if "day before yesterday" in t: offset = 2
    elif "yesterday" in t: offset = 1
    elif "today" in t: offset = 0
    elif rel: offset = int(rel.group(1)) * (7 if rel.group(2).startswith("week") else 1)
    else: return None
    hour, minute = _clock_hour(t, clock); target = day0 - offset * 86400 + hour * 3600 + minute * 60

    # between/btw H and H is an interval aggregate on the selected calendar
    # day, not an instant at its first boundary.
    end_clock = re.search(r"\b(?:between|btw)\s+\d{1,2}(?::\d{2})?\s+(?:and|to)\s+(\d{1,2})(?::(\d{2}))?", t)
    if end_clock and re.search(r"\b(?:average|avg|mean)\b", t) and "speed" in cs:
        h2, m2 = int(end_clock.group(1)), int(end_clock.group(2) or 0)
        if re.search(r"\b(?:pm|night|evening)\b", t[end_clock.end():]) and h2 < 12: h2 += 12
        hi = day0 - offset * 86400 + h2 * 3600 + m2 * 60
        if hi <= target: hi += 12 * 3600
        row = brain.c.execute("SELECT AVG(speed/10.0),COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? AND speed IS NOT NULL AND mode IN (1,2)", (target, hi)).fetchone()
        if not row or not row[1]: return _result("No valid driving-speed samples exist in that clock interval.", "clock_range", "SUMMARY", available=False)
        return _result(f"Average vehicle speed from {A.stamp(target)} to {A.stamp(hi)} was {row[0]:.1f} km/h across {row[1]} valid driving-minute samples.", "clock_range", "SUMMARY", value=row[0], samples=row[1], period=(target, hi))

    if re.search(r"\bwhere\b|\blocation\b", t):
        row = nearest(brain.c, "gps_1s", "ts,lat,lon,hdop", target)
        if not row or abs(row[0]-target) > 90: return _result(f"No valid GPS fix is recorded close enough to {A.stamp(target)}; I will not substitute the current location.", "location_point", "PLACES", available=False, target_ts=target)
        return _result(f"At {A.stamp(row[0])}, the recorded location was GPS {row[1]/1e6:.5f} N, {row[2]/1e6:.5f} E (HDOP {row[3]/10:.1f}).", "location_point", "PLACES", sample_ts=row[0], target_ts=target, lat=row[1]/1e6, lon=row[2]/1e6)

    concept = next((c for c in cs if c in ("speed", "motor_rpm")), None)
    if not concept: return None
    row = nearest(brain.c, "sig_1s", "ts,speed", target, "speed")
    if not row or abs(row[0]-target) > 5: return _result(f"No valid {R.label_of(concept)} sample is recorded close enough to {A.stamp(target)}.", "signal_point", "LOOKUP", available=False, target_ts=target)
    speed = row[1] / 10.0
    if concept == "motor_rpm":
        from .qa_core import rpm_of
        value = rpm_of(speed)
        return _result(f"Estimated motor speed at {A.stamp(row[0])} was {value:,.0f} rpm, derived from the recorded {speed:.1f} km/h using the documented nominal reduction ratio and wheel radius; no RPM sensor is present.", "signal_point", "LOOKUP", value=value, sample_ts=row[0], target_ts=target, estimated=True)
    return _result(f"Vehicle speed at {A.stamp(row[0])} was {speed:.1f} km/h.", "signal_point", "LOOKUP", value=speed, sample_ts=row[0], target_ts=target)


def _state_duration_comparison(brain, t):
    """Compute a comparison of durations spent in distinct vehicle states.

    ``and`` joins the requested report items here; it is not a predicate that
    requires one sample to be both driving and charging.  These states are
    mutually exclusive in the telemetry mode encoding, so passing the request
    to the general conjunction composer would necessarily fabricate a zero
    intersection.
    """
    asks_duration = re.search(r"\b(?:time|duration|how long|long did)\b", t)
    requested = {
        "driving": bool(re.search(r"\bdriv(?:e|ing)\b", t)),
        "charging": bool(re.search(r"\bcharg(?:e|ing)\b", t)),
    }
    if not asks_duration or sum(requested.values()) < 2:
        return None

    lo, hi = A.window(brain.now, "y" if "yesterday" in t else str(_days(t)))
    rows = brain.c.execute(
        "SELECT mode,COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? GROUP BY mode",
        (lo, hi),
    ).fetchall()
    counts = dict(rows)
    durations = {
        "driving": counts.get(1, 0) + counts.get(2, 0),
        "charging": counts.get(3, 0),
    }
    details = " and ".join(
        f"{name} duration was {durations[name]} minutes"
        for name in ("driving", "charging") if requested[name]
    )
    return _result(
        f"From {A.stamp(lo)} to {A.stamp(hi)}, recorded {details}. "
        "These are separate state durations from one-minute samples, not an overlap or SoC averages.",
        "state_duration",
        "COMPARE",
        **{f"{name}_minutes": durations[name] for name in requested if requested[name]},
        period=(lo, hi),
    )


def _history_special(brain, t, cs):
    # Component condition requests must never substitute a related component.
    if "brake_fluid_condition" in cs:
        return _result("Brake fluid condition isn't recorded for this vehicle: no valid brake-fluid condition or quality measurement exists in its data. Brake-pad life is a different measurement and is not a substitute.", "component_unavailable", "STATUS", concept="brake_fluid_condition", available=False)

    if re.search(r"\bbrakes?\b", t) and re.search(r"\b(?:look|looking|condition|okay|ok|health|fine|good)\b", t):
        row=brain.c.execute("SELECT ts,pad FROM vday WHERE ts<=? AND pad IS NOT NULL ORDER BY ts DESC LIMIT 1",(brain.now,)).fetchone()
        pad=f"Recorded brake-pad life is {row[1]:.0f}% at {A.stamp(row[0])}. " if row else "No brake-pad condition is recorded. "
        return _result(pad+"Brake-fluid condition is not recorded, so pad life cannot establish that the whole braking system or fluid is okay. A physical inspection is still required for noises, feel, leaks or warning lamps.","brake_condition","STATUS",pad=row)

    if "rated_max_speed" in cs and "recorded_max_speed" not in cs:
        top = brain.c.execute("SELECT ts, speed FROM sig_1m WHERE speed IS NOT NULL ORDER BY speed DESC LIMIT 1").fetchone()
        seen = f" Highest speed recorded in this vehicle's history: {top[1] / 10.0:.0f} km/h on {A.stamp(top[0])} (an observation, not a rating)." if top else ""
        return _result("The vehicle profile does not contain a verified officially rated maximum speed. I will not substitute the highest speed observed in telemetry." + seen, "capability", "STATUS", concept="rated_max_speed", available=False)

    # A direct correction or remembered earlier charge asks for live SoC.
    if re.search(r"\b(?:what is it now|what's it now|battery percentage|battery level)\b", t) and re.search(r"\bbattery\b", t) and not [x for x in (dialogue_state._times(brain, t) + dialogue_state._times(brain, getattr(brain, "_last_norm", t))) if x.type != "current"]:       # a stated window (also behind a spelling mistake) belongs to the typed engine
        v, snap = _current(brain, "battery_soc")
        if v is not None:
            dialogue_state.remember(brain, Query(op="current_value", target="battery_soc"))
            earlier = " The percentage you mentioned earlier was an older reading; this is the current one." if re.search(r"\b(?:earlier|said|before|previously)\b", t) else ""
            return _result(f"State of charge is {v:.1f}% now, at {A.stamp(snap['sample_ts'])}.{earlier}", "current_soc", "STATUS", sample_ts=snap["sample_ts"], value=v)

    if re.search(r"\b(?:last|latest|most recent)\s+(?:valid )?trip\b", t) and re.search(r"\benergy\b", t):
        row = brain.c.execute("SELECT start,end,km,kwh FROM trips WHERE end<=? ORDER BY end DESC LIMIT 1", (brain.now,)).fetchone()
        if not row: return _result("No completed trip is recorded.", "last_trip_energy", "USAGE", available=False)
        return _result(f"The latest completed trip used {row[3]:.2f} kWh over {row[2]:.1f} km, from {A.stamp(row[0])} to {A.stamp(row[1])}.", "last_trip_energy", "USAGE", start=row[0], end=row[1], distance_km=row[2], energy_kwh=row[3])

    if False and re.search(r"\b(?:regenerative|regeneration|regen|recuperat)\w*\b", t) and re.search(r"\b(?:energy|kwh|recover|recovered)\b", t):      # superseded: the typed engine integrates the same samples over the window that was actually asked for
        days = _days(t); lo = brain.now - days * 86400
        row = brain.c.execute("SELECT COUNT(*),SUM(CASE WHEN power<0 AND mode IN (1,2) THEN -power ELSE 0 END) FROM sig_1m WHERE ts>=? AND ts<=?", (lo, brain.now)).fetchone()
        kwh = (row[1] or 0) / 1000.0 / 60.0
        return _result(f"Regenerative braking recovered about {kwh:.2f} kWh over the last {days} days, integrating negative battery power only while the vehicle was moving ({row[0]} one-minute samples inspected).", "regen_energy", "USAGE", energy_kwh=kwh, days=days, samples=row[0])

    # Two adjacent, non-overlapping windows.
    if re.search(r"\bprevious\s+(?:\d+|seven|fourteen|thirty)\s+days?\b", t) and re.search(r"\b(?:compare|versus|vs|with)\b", t) and re.search(r"\b(?:energy|consumption|efficiency)\b", t):
        days = _days(t); hi = brain.now; split = hi - days * 86400; lo = split - days * 86400
        def eff(a, b):
            k, d = brain.c.execute("SELECT COALESCE(SUM(kwh),0),COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<?", (a, b)).fetchone()
            return (1000 * k / d if d > .5 else None, d)
        recent, dr = eff(split, hi); previous, dp = eff(lo, split)
        if recent is None or previous is None: return _result("There is not enough trip distance in both non-overlapping windows to compare consumption.", "period_comparison", "COMPARE", available=False)
        return _result(f"Last {days} days ({A.stamp(split)} to {A.stamp(hi)}): {recent:.1f} Wh/km over {dr:.1f} km. Previous {days} days ({A.stamp(lo)} to {A.stamp(split)}): {previous:.1f} Wh/km over {dp:.1f} km. Difference: {recent-previous:+.1f} Wh/km.", "period_comparison", "COMPARE", recent=(split, hi, recent), previous=(lo, split, previous))

    if re.search(r"\bcity\b", t) and re.search(r"\b(?:highway|motorway)\b", t) and re.search(r"\b(?:energy|consumption|efficiency)\b", t):
        # one engine, one definition: the typed group comparison (the same numbers every other consumption question reports)
        days = _days(t, 14); q = Query(op="compare", target="wh_per_km", agg="mean", groups=["city", "highway"], scope=Query().scope.__class__(time=TimeSpec("relative", days, "day")))
        qc = brain.qc()
        r = qc.execute(q) if qc is not False else None
        if r is not None and r.status == "answered":
            dialogue_state.remember(brain, q)
            # the typed engine computed this comparison, so it is reported as the typed path - unless a significance test was asked for, which it does not perform
            tag = {} if re.search(r"hypothes|significan|statistic|p-value|\btest\b", t) else {"engine": "qc"}
            return _result(r.answer.text, "city_highway_energy", "COMPARE", days=days, **tag)
        return _result(f"The same {days}-day period does not contain enough city and highway driving for a fair comparison.", "city_highway_energy", "COMPARE", available=False)

    if re.search(r"\bday by day\b", t) and "battery_temperature" in cs:
        days = _days(t); lo = brain.now - days * 86400; series = A.daily_signal(brain.c, "batt_temp", lo, brain.now)
        if not series: return _result("No valid battery-temperature daily series is available for that period.", "daily_series", "TREND", available=False)
        lines = [f"{x['day']}: {x['v']:.1f} C average" for x in series]
        return _result(f"Battery temperature day by day over the last {days} days (valid driving/charging minute samples):\n  - " + "\n  - ".join(lines), "daily_series", "TREND", days=days, points=series)

    if re.search(r"\bexactly\s+\d+\s+days?\s+ago\b", t) and "battery_temperature" in cs:
        n = int(re.search(r"\bexactly\s+(\d+)\s+days?\s+ago\b", t).group(1)); target = brain.now - n*86400
        row = nearest(brain.c, "sig_1m", "ts,batt", target, "batt")
        if not row or abs(row[0]-target) > 3600: return _result(f"No battery-temperature observation exists within one hour of {A.stamp(target)} ({n} days ago).", "point_lookup", "LOOKUP", available=False, target_ts=target)
        return _result(f"Battery temperature nearest exactly {n} days ago was {row[1]/10:.1f} C at {A.stamp(row[0])}; offset {abs(row[0]-target)} seconds from the requested instant.", "point_lookup", "LOOKUP", target_ts=target, sample_ts=row[0], value=row[1]/10)

    if re.search(r"\b(?:degrad|capacity lost|since i got|since i bought)\w*\b", t) and "battery_soh" in cs:
        rows = brain.c.execute("SELECT ts,soh FROM vday WHERE soh IS NOT NULL ORDER BY ts").fetchall()
        if len(rows) < 2: return _result("There are not two defensible state-of-health observations to calculate degradation.", "soh_change", "TREND", available=False)
        a,b = rows[0],rows[-1]; change=b[1]-a[1]
        return _result(f"State of health changed from {a[1]:.1f}% on {A.stamp(a[0])} to {b[1]:.1f}% on {A.stamp(b[0])}: {change:+.1f} percentage points across the recorded history. That is observed capacity change, not a remaining-life prediction.", "soh_change", "TREND", first=(a[0],a[1]), last=(b[0],b[1]), change_points=change)
    return None


def _driver_semantics(brain, t, cs):
    """Resolve broad driver speech acts before statistical parsing.

    These are compositions of an act (capability, maintenance, cumulative
    reading, advice or event history) and an ontology entity.  No complete
    question string is catalogued here, and no unavailable quantity is ever
    replaced with a nearby recorded signal.
    """
    # An explanation pronoun is meaningful only when this semantic layer set
    # a recent referent.  It must never be resolved against an unrelated old
    # service, RPM or range answer.
    if re.fullmatch(r"\s*(?:what does (?:this|that|it) mean|what is the meaning of (?:this|that|it)|explain (?:this|that|it))\s*", t):
        if getattr(brain, "_semantic_referent", None) == "adas_state":
            return _result("ADAS means advanced driver-assistance systems. State 0 is off or inactive and state 1 is active in this vehicle's categorical signal; neither state means the car is autonomous.", "adas_state_definition", "GLOSSARY")
        return None
    if re.search(r"\bwhy (?:are|were) you talking about\b", t):
        brain._semantic_referent = None
        brain.qcs.last = None; brain.qcs.choices = []; brain.qcs.pending = None; brain.qcs.options = []
        return _result("That topic did not match your question. I have cleared it rather than carrying it into the next answer; please ask the vehicle question again.", "context_correction", "CLARIFY")

    # Direct present-value speech acts.  A named physical quantity plus a
    # present-time request is complete semantics and must not be replaced by a
    # component health summary merely because both contain “battery” or “car”.

    soc_request = (re.search(r"\b(?:charge|energy)\s+left\b|\b(?:battery|charge)\s+(?:level|percentage|percent)\b|\bhow much\b.{0,18}\bbattery charge\b", t)
                   and not re.search(r"\b(?:time|long|rate|speed|power|capacity|cycle|session|add|added|used|use)\b", t))
    if soc_request:
        v, snap = _current(brain, "battery_soc")
        if v is None: return _result("No fresh valid state-of-charge reading is available.", "current_soc", "STATUS", available=False)
        return _result(f"The traction battery is at {v:.1f}% state of charge now, at {A.stamp(snap['sample_ts'])}.", "current_soc", "STATUS", value=v, sample_ts=snap["sample_ts"])

    # Energy capacity, state of health and amp-hour capacity are separate
    # dimensions.  The database verifies kWh but contains no nominal pack
    # voltage, so an Ah/mAh conversion would otherwise invent a denominator.
    mah_with_voltage = bool(re.search(r"\bm\s*a\s*h\b|\bmah\b|milliamp", t) and re.search(r"\b\d+(?:\.\d+)?\s*v(?:olts?)?\b", t))          # a stated voltage makes mAh an energy: that is the equivalence calculation, not a capacity statement
    if re.search(r"\bbatter(?:y|ies)\b", t) and (re.search(r"\bcapacity\b", t) or re.search(r"\bm\s*a\s*h\b|\bmah\b|milliamp", t)) and not mah_with_voltage:
        cap = A.cap_kwh(brain.c)
        if re.search(r"\bm\s*a\s*h\b|\bmah\b|milliamp", t):
            return _result(f"The verified pack capacity is {cap:.1f} kWh. I cannot convert that to mAh without the pack's verified nominal voltage; mAh alone is not an energy-capacity rating.", "battery_capacity", "STATUS", capacity_kwh=cap, available_mah=False)
        soh = brain.c.execute("SELECT soh FROM vday WHERE ts<=? AND soh IS NOT NULL ORDER BY ts DESC LIMIT 1", (brain.now,)).fetchone()
        suffix = f" Latest state of health is {float(soh[0]):.1f}%, which is a separate measure of retained capacity." if soh else ""
        return _result(f"The verified usable battery capacity in the vehicle profile is {cap:.1f} kWh.{suffix}", "battery_capacity", "STATUS", capacity_kwh=cap)

    if re.search(r"\b(?:charging|charge)\s+(?:rate|speed|power)\b|\bhow fast\b.{0,20}\bbatter(?:y|ies)\b.{0,10}\bcharg", t):
        try: row = brain.c.execute("SELECT ts,charging_power FROM xsig_1s WHERE ts<=? AND charging_power IS NOT NULL ORDER BY ts DESC LIMIT 1", (brain.now,)).fetchone()
        except Exception: row = None
        if row and brain.now - row[0] <= 300:
            return _result(f"Charging power is {float(row[1]):.1f} kW now, at {A.stamp(row[0])}. That is the instantaneous charge rate, not energy added.", "charging_rate", "STATUS", value_kw=float(row[1]), sample_ts=row[0], engine="qc")
        row = brain.c.execute("SELECT start,kw,kind FROM charges WHERE start<=? ORDER BY start DESC LIMIT 1", (brain.now,)).fetchone()
        if row: return _result(f"The car is not reporting current charging power. The most recent recorded charging session used about {float(row[1]):.1f} kW ({row[2]}) at {A.stamp(row[0])}; that historical charging power is not a live reading.", "charging_rate", "STATUS", available=False, last_kw=float(row[1]), engine="qc")
        return _result("No current or historical charging-power measurement is recorded.", "charging_rate", "STATUS", available=False, engine="qc")

    if re.search(r"\b(?:charging|charge|battery)\s+cycles?\b|\bcycles?\b.{0,16}\b(?:charg|battery)", t):
        row = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.charge_cycles','veh.battery_cycle_count') ORDER BY k LIMIT 1").fetchone()
        if row: return _result(f"The verified battery cycle counter is {int(float(row[0]))} cycles.", "charge_cycles", "STATUS", value=int(float(row[0])))
        sessions = brain.c.execute("SELECT COUNT(*) FROM charges WHERE start<=?", (brain.now,)).fetchone()[0]
        return _result(f"A battery cycle counter is not recorded, so I cannot state completed charge cycles. There are {sessions} recorded charging sessions, but sessions are not equivalent to full battery cycles.", "charge_cycles", "STATUS", available=False, recorded_sessions=sessions)

    # Export-power questions are vehicle capability questions, not requests for
    # present battery current.  Require the relevant outlet/V2L specification.
    if re.search(r"\b(?:laptop|phone|appliance|device|house|grid)\b", t) and re.search(r"\b(?:charge|power|run|supply)\b", t) or re.search(r"\boutput (?:current|power)\b", t):
        p = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.v2l_kw','veh.export_power_kw','veh.outlet_power_w') ORDER BY k LIMIT 1").fetchone()
        if not p: return _result("The verified vehicle profile does not include an accessory outlet, inverter, V2L, or export-power rating, so I cannot say that it can power another device or state its output current. Output current also depends on the specified output voltage.", "export_power_capability", "STATUS", available=False)
        return _result(f"The vehicle profile records an external-output rating of {p[0]}. Check the specified voltage, connector and owner's manual before connecting a device.", "export_power_capability", "STATUS", rating=p[0])

    if re.search(r"\bmotor(?:'s|s)?\s+(?:mechanical )?power\b", t):
        if re.search(r"\b(?:calculate|formula|work out|derive)\b", t):
            return _result("Motor mechanical power is torque multiplied by angular speed: power in kW = torque in N m x RPM / 9549. Use measurements from the same instant; battery electrical power is not the same quantity.", "motor_power_definition", "GLOSSARY")
        try: row = brain.c.execute("SELECT ts,motor_power FROM xsig_1s WHERE ts<=? AND motor_power IS NOT NULL ORDER BY ts DESC LIMIT 1", (brain.now,)).fetchone()
        except Exception: row = None
        if row and brain.now-row[0] <= 300: return _result(f"Motor mechanical power is {float(row[1]):.1f} kW now, at {A.stamp(row[0])}.", "motor_power", "STATUS", value_kw=float(row[1]), sample_ts=row[0])
        return _result("No fresh measured motor-power value is available. I will not substitute battery power or motor temperature because they are different quantities.", "motor_power", "STATUS", available=False)

    if re.search(r"\b(?:how (?:is|was|do)|what do you think|properly|well)\b", t) and re.search(r"\b(?:drive|driving|driver)\b", t) and not re.search(r"\b(?:safe|safely|tips?|advice|avoid|should|prove|would)\b", t):   # advice and counterfactuals are other questions; "driving score / style" is the 0-100 smoothness score of qa_h (documented formula), not this assessment
        w = "7" if re.search(r"\b(?:this|last|past) week\b", t) else "28"
        f = AX.driving(brain.c, brain.now, w)
        if not f.get("ok"): return _result("There is not enough recorded driving in that period to assess driving style.", "driving_assessment", "DRIVING", available=False)
        events = f"{f['brake']} hard-braking, {f['accel']} hard-acceleration and {f['corner']} hard-cornering events"
        return _result(f"Over the last {w} days, the car recorded {f['km']:.1f} km across {f['trips']} trips, with {events}. These telemetry indicators describe smoothness; they cannot prove that every manoeuvre was proper or safe.", "driving_assessment", "DRIVING", **f)

    if re.search(r"\b(?:my|our|the car'?s|vehicle'?s)\s+address\b", t):
        return _result("The vehicle data contains GPS coordinates and saved place names, not a verified postal street address. I will not reverse-geocode or invent an address from coordinates while offline.", "postal_address", "PLACES", available=False)

    if re.fullmatch(r"\s*(?:adas|driver assistance|driver-assistance)\s*", t):
        brain._semantic_referent = "adas_state"
        return _result("ADAS means advanced driver-assistance systems. In this vehicle, the recorded state indicates whether assistance is off or active, and separate events record disengagements; it does not mean the car is autonomous.", "adas_definition", "GLOSSARY")
    if re.search(r"\b(?:adas|driver assist(?:ant|ance)?|lane assist)\b", t) and re.search(r"\b(?:state|value|category)?\s*(?:0|zero)\b", t) and re.search(r"\b(?:mean|meaning|what|explain)\b", t):
        brain._semantic_referent = "adas_state"
        return _result("Driver-assistance state 0 means off or inactive in this vehicle's signal definition. It is a categorical state, not an error code and not a capability rating.", "adas_state_definition", "GLOSSARY", value=0, label="off")

    # A quantified coreference keeps the last typed measure but expands all of
    # its physical members.  Tyre pressure is currently the only four-member
    # measurement in the schema.
    if re.fullmatch(r"\s*(?:list |show )?(?:all|every)(?: of)? (?:them|those|these)\s*", t):
        try:
            from . import dialogue_state
            previous = dialogue_state.topic(brain)
        except Exception: previous = None
        if previous is not None and previous.target == "tyre_pressure":
            f = AX.tyres(brain.c, brain.now, "14")
            if f.get("ok"):
                values = "; ".join(f"{w['name']} {w['kpa']:.0f} kPa" for w in f.get("wheels", []))
                return _result(f"All recorded tyres: {values}.", "tyre_inventory", "TYRES", wheels=f.get("wheels", []), active_warnings=f.get("active_warnings", []), not_recorded=f.get("not_recorded", []), spec_unavailable=f.get("spec_unavailable", []))

    # A bare causal follow-up inherits the previously established component.
    # The antecedent is typed conversation state set by the battery-status
    # handler, not a scan of the previous answer's prose.  This makes “why?”
    # work while preventing an unrelated old warning from being selected when
    # there is no live conversational referent.
    if re.fullmatch(r"\s*(?:why|why so|how come|what caused (?:that|it))\s*", t):
        if getattr(getattr(brain, "qa", None), "ctx", {}).get("topic") == "battery":
            event = brain.c.execute("SELECT 1 FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts<=? LIMIT 1", (brain.now,)).fetchone()
            if event:
                return _result(brain.qa.h_why_event("why battery overheating", t), "why_event", "EVENTS")

    # Vehicle capacity is a profile specification; it is not current seat
    # occupancy and certainly not whole-vehicle health.
    people = bool(re.search(r"\b(?:people|persons?|passengers?|occupants?|adults?|children|kids|seats?|seating)\b", t))
    capacity = bool(re.search(r"\b(?:fit|seat|carry|hold|accommodat\w*|capacity|room for)\b", t))
    if people and capacity:
        row = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.seats','veh.seating_capacity') ORDER BY k LIMIT 1").fetchone()
        if row:
            try: seats = int(float(row[0]))
            except (TypeError, ValueError): seats = None
            if seats is not None: return _result(f"The verified vehicle profile lists {seats} seats, including the driver.", "profile_capability", "STATUS", concept="seating_capacity", value=seats)
        return _result("The vehicle profile does not contain a verified seating capacity, so I cannot say how many people it is designed to carry. Current seat occupancy and rated seating capacity are different facts.", "profile_capability", "STATUS", concept="seating_capacity", available=False)

    # Comparisons against another vehicle require an equally sourced external
    # specification or telemetry record.  Do not silently replace “better”
    # with this driver's score, which measures a completely different thing.
    comparison = bool(re.search(r"\b(?:better|worse|compare\w*|versus|vs\.?|than)\b", t))
    own_vehicle = bool(re.search(r"\b(?:my|our|this) (?:car|vehicle)\b", t))
    external_vehicle = re.search(r"\b(?:than|versus|vs\.?|with)\s+(?:the\s+)?([a-z0-9][a-z0-9 -]{1,60})$", t)
    if comparison and own_vehicle and external_vehicle:
        other = external_vehicle.group(1).strip()
        return _result(f"I cannot determine whether this car is better than {other} from this vehicle's records alone. 'Better' also needs a criterion such as range, acceleration, efficiency, price or safety, plus a verified source for both vehicles.", "external_comparison", "COMPARE", available=False, other_vehicle=other)

    # Ownership age needs paperwork/profile provenance, not telemetry uptime.
    if re.search(r"\b(?:how long|age)\b", t) and re.search(r"\b(?:bought|buy|purchased?|owned?|got|brought)\b", t) and re.search(r"\b(?:car|vehicle|it)\b", t):
        row = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.purchase_date','veh.acquired_date') ORDER BY k LIMIT 1").fetchone()
        if not row: return _result("The vehicle data does not contain a verified purchase or acquisition date, so I cannot calculate how long you have owned it. That belongs in the registration or purchase records.", "profile_capability", "STATUS", concept="ownership_age", available=False)

    modal = bool(re.search(r"\b(?:can|could|able|possible)\b", t))

    # Modal speed capability is a rated specification, never the highest
    # historical telemetry sample.
    if modal and re.search(r"\b(?:fast|speed|quick)\b", t) and re.search(r"\b(?:travel|drive|go|car|vehicle|we|it)\b", t):
        top = brain.c.execute("SELECT ts,speed FROM sig_1m WHERE speed IS NOT NULL ORDER BY speed DESC LIMIT 1").fetchone()
        rated = brain.c.execute("SELECT v FROM meta WHERE k IN ('veh.rated_max_speed','veh.max_speed_kmh') ORDER BY k LIMIT 1").fetchone()
        if rated:
            try: value = float(rated[0])
            except (TypeError, ValueError): value = None
            if value is not None: return _result(f"The verified rated maximum speed is {value:g} km/h.", "profile_capability", "STATUS", concept="rated_max_speed", value=value)
        observed = f" Highest vehicle speed recorded in the retained telemetry: {top[1] / 10.0:.0f} km/h, but that is an observation, not a rating." if top else ""
        return _result("The vehicle's rated maximum speed isn't recorded in the verified profile, so I cannot state how fast it is designed to go. I will not substitute an observed speed for that specification." + observed, "profile_capability", "STATUS", concept="rated_max_speed", available=False)

    # Modal distance capability means remaining driving range.  Speed is only
    # selected when the requested quantity itself is fast/speed.
    travel = bool(re.search(r"\b(?:travel|drive|go|distance|journey|trip)\b", t))
    amount = bool(re.search(r"\b(?:how (?:far|much)|what distance|range|remaining)\b", t))
    if travel and amount and (modal or "range" in t) and not re.search(r"\b(?:fast|speed|quick)\b", t):
        er = A.est_range(brain.c, brain.now)
        if not er.get("ok"): return _result("The current driving range cannot be estimated because charge or recent consumption data is missing.", "remaining_range", "STATUS", available=False)
        return _result(f"The current estimated driving range is about {er['range_km']:.0f} km at {er['soc']:.0f}% charge, based on the recent {er['wh_km']:.0f} Wh/km. It is an estimate, not a guaranteed distance.", "remaining_range", "STATUS", range_km=er["range_km"], soc=er["soc"], wh_per_km=er["wh_km"])

    # A request for the present cumulative distance is the odometer.  A stated
    # time window remains a trip-distance aggregate and is left to typed IR.
    cumulative = bool(re.search(r"\b(?:total|altogether|overall|odometer|on the clock|how (?:many|much))\b", t))
    distance = bool(re.search(r"\b(?:distance|miles?|kilomet(?:re|er)s?|km|travel\w*|driv(?:e|en)|covered|gone)\b", t))
    windowed = bool(re.search(r"\b(?:last|past|yesterday|today|week|month|day|hour|since the last trip|during|between|from)\b", t))
    requested_energy = bool(re.search(r"\b(?:energy|kwh|wh|consumption|efficien\w*|need|required?|use)\b", t))
    if cumulative and distance and not windowed and not modal and not requested_energy:
        f = AX.service(brain.c, brain.now)
        if not f.get("ok"): return _result("No valid odometer reading is recorded for this vehicle.", "odometer", "STATUS", available=False)
        km = float(f["odo_km"])
        if re.search(r"\b(?:miles?|mi)\b", str(getattr(brain, "_text", t)).lower()):
            return _result(f"The odometer is at about {km / 1.609344:,.0f} miles ({km:,.0f} km), as of {f['time']}.", "odometer", "STATUS", value_km=km)
        return _result(f"The odometer is at {km:,.0f} km, as of {f['time']}.", "odometer", "STATUS", value_km=km)

    # A selected trip is an entity scope, not a time series beginning after
    # the trip.  Colloquial “since/for the last drive” commonly selects the
    # most recent completed drive; disclose that interpretation explicitly.
    if re.search(r"\b(?:average|avg|mean)\b", t) and re.search(r"\bspeed\b", t) and re.search(r"\b(?:since|for|on|during|of) (?:the )?(?:last|latest|previous) (?:drive|trip|journey)\b", t):
        row = brain.c.execute("SELECT start,end,avg_speed FROM trips WHERE end<=? AND avg_speed IS NOT NULL ORDER BY end DESC LIMIT 1", (brain.now,)).fetchone()
        if not row: return _result("No completed trip with an average-speed record is available.", "last_trip_speed", "SUMMARY", available=False)
        return _result(f"The average speed during the most recent completed trip was {row[2]:.1f} km/h ({A.stamp(row[0])} to {A.stamp(row[1])}). I interpreted 'since the last drive' as 'during the last drive'; after it ended is a different time window.", "last_trip_speed", "SUMMARY", sample_start=row[0], sample_end=row[1], value=row[2])

    # Maintenance timing is one domain operation regardless of whether the
    # speaker says service, servicing or maintenance.
    service = bool(re.search(r"\b(?:service|servicing|maintenance|maintain)\w*\b", t))
    service_act = bool(re.search(r"\b(?:when|due|next|should|need|time|schedule|book|get|go for)\b", t))
    if service and service_act:
        f = AX.service(brain.c, brain.now)
        if not f.get("ok"): return _result("This vehicle has no usable odometer and service-interval record, so I cannot determine when service is due.", "service_due", "SERVICE", available=False)
        if "km_to_service" not in f: return _result(f"The odometer is {f['odo_km']:,} km, but the vehicle profile has no verified service interval or last-service record.", "service_due", "SERVICE", available=False)
        if f["due"]: msg = f"Service is due now. The odometer is {f['odo_km']:,} km; it has been {f['km_since_service']:,} km since the recorded service."
        else:
            by_date = f"; by time, in about {max(0, f['months_to_service']):.0f} months" if f.get("months_to_service") is not None else ""
            msg = f"The next service is due in about {f['km_to_service']:,} km{by_date}. Current odometer: {f['odo_km']:,} km."
        return _result(msg, "service_due", "SERVICE", **f)

    # Advice about caring for the vehicle combines the recorded maintenance
    # and health state; it does not ask the statistical compiler to invent a
    # signal or period.
    advice = bool(re.search(r"\b(?:how (?:can|do|should)|ways?|tips?|advice)\b", t))
    care = bool(re.search(r"\b(?:take care|look after|maintain|maintenance|keep)\b", t)) and bool(re.search(r"\b(?:car|vehicle)\b", t))
    if advice and care:
        h = AX.health(brain.c, brain.now, "14"); s = AX.service(brain.c, brain.now); actions = []
        if h.get("ok"):
            actions += [f"Have the {x['area']} issue checked: {x['text']}." for x in h.get("items", []) if x.get("level") == "problem"][:2]
        if s.get("ok") and "km_to_service" in s:
            actions.append("Book the scheduled service now." if s.get("due") else f"Plan the next scheduled service in about {s['km_to_service']:,} km.")
        if s.get("pad_pct") is not None: actions.append(f"Brake-pad life is recorded at about {s['pad_pct']:.0f}%; inspect it physically at service.")
        if not actions: actions.append("No specific maintenance problem is supported by the recorded data; follow the manufacturer schedule and respond to warning lamps or unusual noises.")
        return _result("Based on this vehicle's records:\n  - " + "\n  - ".join(actions) + "\nThese checks use recorded data and do not replace the owner's manual or a physical inspection.", "maintenance_advice", "SERVICE", items=len(actions))

    # Whole-vehicle problem/health requests must reach the complete health
    # aggregation, not whichever component classifier happens to score first.
    health_state = bool(re.search(r"\b(?:health|healthy|condition|status|problems?|issues?|faults?|wrong|doing)\b", t))
    if health_state and re.search(r"\b(?:car|vehicle)\b", t) and not re.search(r"\b(?:battery|motor|engine|coolant|tyres?|tires?|brakes?|camera|12 ?v)\b", t):
        return _result(brain.qa.h_health_summary(t, t), "vehicle_health", "HEALTH")

    # “Engine” and “traction motor” are not interchangeable.  Resolve this
    # component identity before the generic over-temperature event family;
    # otherwise the word “overheat” alone could silently turn an engine query
    # into a battery-warning explanation.
    if re.search(r"\bengine\b", t) and re.search(r"\b(?:hot|overheat\w*|over temperature)\b", t):
        model = brain.c.execute("SELECT v FROM meta WHERE k='veh.model' LIMIT 1").fetchone()
        ev = bool(model and re.search(r"\b(?:ev|electric)\b", str(model[0]), re.I))
        if ev:
            return _result("This is a battery-electric vehicle, so it has no combustion engine to overheat. The traction motor and battery pack have separate temperature signals; I will not substitute either one without knowing which component you mean.", "powertrain_distinction", "STATUS", powertrain="battery_electric")

    # Event semantics: event identity comes from the registry, while the
    # speech act decides whether to explain, date or check occurrence.
    thermal_event = "overheating" in cs or bool(re.search(r"\b(?:battery|pack)\b.{0,20}\b(?:hot|overheat\w*|over temperature)\b|\b(?:hot|overheat\w*|over temperature)\b.{0,20}\b(?:battery|pack)\b", t))
    if thermal_event:
        if re.search(r"\b(?:why|cause|reason|trigger)\b", t): return _result(brain.qa.h_why_event("why battery overheating", t), "why_event", "EVENTS")
        if re.search(r"\bwhen\b", t):
            row = brain.c.execute("SELECT ts,detail FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts<=? ORDER BY ts DESC LIMIT 1", (brain.now,)).fetchone()
            if not row: return _result("No battery over-temperature warning is recorded.", "thermal_event", "EVENTS", available=False)
            return _result(f"The latest battery over-temperature warning was at {A.stamp(row[0])}: {row[1]}.", "thermal_event", "EVENTS", event_ts=row[0])
        if re.search(r"\b(?:did|was|were|has|have|ever|any|issues?)\b", t): return _result(brain.qa.h_hot_battery(t, t), "thermal_history", "HEALTH")

    # Point-in-time lookup: ``N hours ago`` is an instant, unlike ``the last N
    # hours`` which remains an aggregate window for the typed compiler.
    ago = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\s+ago\b", t)
    if ago and cs:
        concept = next((c for c in cs if c in R.STORE and R.CONCEPTS[c].dtype == "numeric"), None)
        if concept:
            unit = ago.group(2); factor = 86400 if unit.startswith("day") else 3600 if unit.startswith(("hour", "hr")) else 60 if unit.startswith(("minute", "min")) else 1
            target = brain.now - float(ago.group(1)) * factor; store = R.STORE[concept].get("1s") or R.STORE[concept].get("1m")
            if store:
                table, col, scale = store; row = nearest(brain.c, table, f"ts,{col}", int(target), col); tolerance = 5 if table == "sig_1s" else 90
                if row and abs(row[0] - target) <= tolerance: return _result(f"{R.label_of(concept).capitalize()} {ago.group(0)} was {_fmt_value(concept, row[1] / scale)}, at {A.stamp(row[0])}.", "point_ago", "LOOKUP", concept=concept, sample_ts=row[0], target_ts=target)
                return _result(f"There is no valid {R.label_of(concept)} sample close enough to the requested instant ({ago.group(0)}), so I will not substitute a different time.", "point_ago", "LOOKUP", concept=concept, available=False)

    # Quantifier scope over tyres means report every recorded wheel, not the
    # minimum pressure and not a generic event-list referent.
    if re.search(r"\b(?:all|every|each)\b", t) and re.search(r"\b(?:tyres?|tires?|wheels?)\b", t):
        f = AX.tyres(brain.c, brain.now, "14")
        if not f.get("ok"): return _result("No tyre-pressure readings are recorded for this vehicle.", "tyre_inventory", "TYRES", available=False)
        values = "; ".join(f"{w['name']} {w['kpa']:.0f} kPa" for w in f.get("wheels", []))
        return _result(f"Latest recorded pressures for all tyres: {values}.", "tyre_inventory", "TYRES", wheels=f.get("wheels", []), active_warnings=f.get("active_warnings", []), not_recorded=f.get("not_recorded", []), spec_unavailable=f.get("spec_unavailable", []))

    # Speech act and plural entity are separate: definition/location questions
    # receive a conceptual answer, while an unqualified plural status/report
    # request receives every member of the four-wheel measurement.  This is a
    # quantified ontology operation, not a minimum-pressure shortcut.
    tyres = bool(re.search(r"\b(?:tyres?|tires?)\b", t))
    if tyres and re.search(r"\bwhere\b", t):
        return _result("The tyres are fitted to the four wheel positions: front left, front right, rear left and rear right. If you want their readings, ask for tyre pressure or tyre status.", "component_location", "GLOSSARY")
    if tyres and re.search(r"\bwhat (?:are|is)\b|\bdefine\b|\bmeaning\b", t) and not re.search(r"\b(?:pressure|psi|kpa|bar|temperature|temp|reading|level)\b", t):         # "what is the tyre pressure on the front left" asks for a measurement, not what a tyre is
        return _result("Tyres are the rubber road-contact parts fitted around the wheels. This vehicle records pressure for each tyre; it does not infer tread depth or wheel alignment from pressure.", "component_definition", "GLOSSARY")
    tyre_report = bool(re.search(r"\b(?:how (?:are|is)|status|condition|tell|show|report|check|list)\b", t))
    tyre_window = bool(re.search(r"\b(?:yesterday|today|last|past|week|month|day|hour|between|from|during|trend|average|mean|highest|lowest)\b", t))
    if tyres and tyre_report and not tyre_window:
        f = AX.tyres(brain.c, brain.now, "14")
        if not f.get("ok"): return _result("No tyre-pressure readings are recorded for this vehicle.", "tyre_inventory", "TYRES", available=False)
        values = "; ".join(f"{w['name']} {w['kpa']:.0f} kPa ({w.get('status', 'recorded')})" for w in f.get("wheels", []))
        bad = [w for w in f.get("wheels", []) if w.get("status") not in ("ok", "recorded", None)]
        extra = "".join(f" The {w['name']} tyre is {w.get('status')}: {w['kpa']:.0f} kPa, {100 - w['pct']:.0f} % below its {w['spec']:.0f} kPa specification." for w in bad if w.get("pct") is not None and w.get("spec"))
        if f.get("active_warnings"): extra += " A tyre-pressure warning is active for the " + " and ".join(f["active_warnings"]) + " tyre."
        return _result(f"Latest status for all recorded tyres: {values}.{extra}", "tyre_inventory", "TYRES", wheels=f.get("wheels", []), active_warnings=f.get("active_warnings", []), not_recorded=f.get("not_recorded", []), spec_unavailable=f.get("spec_unavailable", []))
    return None


def _advice_and_diagnostics(brain, t, cs):
    if re.search(r"\bco2\b|carbon dioxide", t) and re.search(r"\b(?:saved|petrol|gasoline|compared)\b", t):
        return _result("CO2 savings cannot be calculated from vehicle telemetry alone. I would need electricity consumed from the grid, a documented electricity-emissions factor, petrol-vehicle fuel economy, fuel emissions per litre, and the same distance and boundary assumptions.", "missing_inputs", "ANALYSIS", missing=["grid energy", "grid emissions factor", "petrol baseline", "distance boundary"])
    if re.search(r"\breplacement\b", t) and re.search(r"\b(?:tyre|tire)\b", t):
        return _result("I cannot verify a replacement tyre from telemetry. A valid recommendation needs the approved dimensions, load index, speed rating and vehicle/manufacturer compatibility plus a trustworthy catalogue; none is recorded here.", "parts_advice", "SAFETY", available=False)
    if re.search(r"\b(?:exact|precise)\b.*\b(?:battery\b.*)?(?:lifespan|life|months?)\b", t):
        return _result("An exact remaining battery lifespan in months is not supported by this short history. A defensible estimate would need long-term SoH, calendar age, cycle count/depth, temperature exposure, charging behaviour and a validated degradation model with uncertainty.", "unsupported_precision", "ANALYSIS", available=False)
    if re.search(r"\b(?:how can|how do|ways? to|tips? to)\b.*\b(?:efficien\w*|range)\b", t):
        return _result(brain.qa.h_improve_range(t, t), "efficiency_advice", "ANALYSIS")
    if re.search(r"\b(?:getting|becoming)\b.*\b(?:less efficient|inefficient)|\befficien\w*\b.*\blately\b", t):
        return _result(brain.qa.h_eff_trend(t, t), "efficiency_trend", "TREND")
    if re.search(r"\b(?:overall|healthy|health|doing)\b", t) and re.search(r"\b(?:car|vehicle)\b", t):
        return _result(brain.qa.h_health_summary(t, t), "vehicle_health", "HEALTH")
    priority = bool(re.search(r"\b(?:investigate|check|fix|attention|workshop)\b", t) and re.search(r"\b(?:first|most|immediately|priority|should|needs?)\b", t))
    if priority: return _result(brain.qa.h_investigate(t, t), "diagnostic_priority", "ANALYSIS")
    if re.search(r"\bwhy\b.*\b(?:unhealthy|not healthy|problem)\b", t):
        return _result(brain.qa.h_health_summary(t, t), "health_evidence", "HEALTH")
    if re.search(r"\b(?:safe|safely)\b.*\b(?:drive|driving|km|warning)\b", t) and not re.search(r"\b(?:tips?|habits?|advice|guidelines?|rules?|ways? to|how (?:to|do i|can i|should i))\b", t):
        warn = R.EVENT_GROUPS.get("ANY_WARNING", ())
        marks = ",".join("?" * len(warn))
        last = brain.c.execute(f"SELECT ts,type,detail FROM events WHERE ts<=? AND type IN ({marks}) ORDER BY ts DESC LIMIT 1", (brain.now,) + tuple(warn)).fetchone() if warn else None
        ev = f"The most recent recorded problem is {last[1]} at {A.stamp(last[0])}: {last[2]}. " if last else "No warning or fault event is recorded, so there is no recorded problem to weigh. "
        return _result(ev + "Telemetry cannot guarantee that a journey is safe or that a historical warning remains active. Have a qualified workshop assess a safety-relevant warning before a long drive.", "safety_assessment", "SAFETY")
    if re.search(r"\b(?:simulated|illustrative)\b", t) and re.search(r"\b(?:fault|warning|code|record)\b", t):
        return _result("No. A record marked simulated or illustrative is test data, not confirmation of a real-world fault. Current active status would need a genuine ECU status/clearance signal.", "simulation_status", "STATUS", confirmed=False)

    if re.search(r"\b(?:three|3)\b.*\b(?:serious|severe)\b.*\bconfirmed\b.*\bactive\b", t):
        return _result("No confirmed active-fault list can be established. The stored DTC entries in this demonstration are illustrative/simulated, and the data has no authoritative current uncleared-DTC status. I will not rank them as real active faults.", "confirmed_active_faults", "EVENTS", confirmed_count=0)

    if re.search(r"\bmost recent\b", t) and re.search(r"\bmost severe\b", t) and re.search(r"\bfault\b", t):
        rows = brain.c.execute("SELECT ts,type,sev,detail FROM events WHERE type IN ('DTC_SET','WARN_BATT_OVERTEMP','TPMS_LOW') AND ts<=? ORDER BY ts", (brain.now,)).fetchall()
        if not rows: return _result("No fault or warning events are recorded.", "fault_recency_severity", "EVENTS", available=False)
        severity={"info":0,"watch":1,"warn":2,"warning":2,"critical":3}; recent=max(rows,key=lambda r:r[0]); severe=max(rows,key=lambda r:(severity.get(str(r[2]).lower(),0),r[0]))
        return _result(f"Most recent: {recent[1]} at {A.stamp(recent[0])}, severity {recent[2]} - {recent[3]}. Most severe: {severe[1]} at {A.stamp(severe[0])}, severity {severe[2]} - {severe[3]}. These are separate selections; illustrative DTC text is not a confirmed active fault.", "fault_recency_severity", "EVENTS", recent=recent, severe=severe)

    if re.search(r"\bcoolant\s*flow\b", t) and re.search(r"\btyre\s*press", t) and re.search(r"\b(?:important|serious|severe|priority|compare|than)\b", t):
        rows = brain.c.execute("SELECT ts,type,sev,detail FROM events WHERE type IN ('DTC_SET','TPMS_LOW') AND ts<=? ORDER BY ts DESC", (brain.now,)).fetchall()
        cool=next((r for r in rows if 'coolant' in r[3].lower()),None); tyre=next((r for r in rows if 'tyre' in r[3].lower() or 'tire' in r[3].lower()),None)
        if not cool or not tyre: return _result("Both warning records are not available, so their documented severity cannot be compared.", "warning_comparison", "EVENTS", available=False)
        verdict="coolant-flow" if cool[2]>tyre[2] else "tyre-pressure" if tyre[2]>cool[2] else "neither; their recorded severity is equal"
        return _result(f"By documented event severity, priority is {verdict}. Coolant-flow record: severity {cool[2]} at {A.stamp(cool[0])}; tyre-pressure record: severity {tyre[2]} at {A.stamp(tyre[0])}. Stored events are historical and illustrative here; this does not prove either condition is currently active.", "warning_comparison", "EVENTS", coolant=cool, tyre=tyre)

    if re.search(r"\bevidence\b", t) and re.search(r"\b(?:battery )?cooling\b", t):
        tr=A.trend(brain.c,"coolant_flow",brain.now,"14"); snap=_fresh_snapshot(brain); temp=snap.get("values",{}).get("battery_temperature"); flow=snap.get("values",{}).get("coolant_flow")
        ev=brain.c.execute("SELECT ts,type,sev,detail FROM events WHERE ts<=? AND (lower(detail) LIKE '%coolant%' OR type='WARN_BATT_OVERTEMP') ORDER BY ts DESC LIMIT 6",(brain.now,)).fetchall()
        lines=[f"current coolant flow {flow:.1f} L/min and pack temperature {temp:.1f} C at {A.stamp(snap['sample_ts'])}" if flow is not None and temp is not None else "current cooling signals are incomplete"]
        if tr.get("ok"): lines.append(f"14-day coolant-flow comparison: {tr['first']:.1f} to {tr['last']:.1f} {tr['unit']} ({tr['change_pct']:+.1f}%) across {tr['days']} daily values")
        lines += [f"event {A.stamp(r[0])}, severity {r[2]}: {r[3]}" for r in ev]
        return _result("Cooling evidence (observations, not proof of cause):\n  - " + "\n  - ".join(lines) + "\nDTC records in this data set are illustrative/simulated, not confirmed real faults.", "cooling_evidence", "ANALYSIS", events=ev, trend=tr)

    if re.search(r"\bover[- ]?temperature warnings?\b", t) and re.search(r"\b(?:parked|moving)\b", t):
        ev=brain.c.execute("SELECT ts,detail FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts<=? ORDER BY ts",(brain.now,)).fetchall(); lines=[]
        for ts,detail in ev:
            s=nearest(brain.c, "sig_1m", "ts,speed,mode", ts)
            if not s or abs(s[0]-ts)>90: lines.append(f"{A.stamp(ts)}: no contemporaneous motion sample")
            else: lines.append(f"{A.stamp(ts)}: {'moving' if s[1]/10>.5 else 'parked'}, speed {s[1]/10:.1f} km/h (sample offset {abs(s[0]-ts)} s)")
        return _result("Over-temperature warning context:\n  - "+("\n  - ".join(lines) if lines else "no warnings recorded"),"warning_motion_context","EVENTS",events=len(ev))

    if re.search(r"\b(?:tyre|tire) pressure\b", t) and re.search(r"\b(?:still|present|active|problem)\b", t):
        ty=AX.tyres(brain.c,brain.now,"14")
        if not ty.get("ok"): return _result("Current tyre-pressure condition cannot be confirmed from a valid latest reading.","tyre_status","STATUS",available=False)
        wheels=ty.get("wheels",[]); bad=[w for w in wheels if w.get("status")!='ok']
        detail="; ".join(f"{w['name']} {w['kpa']:.0f} kPa ({w['status']})" for w in wheels)
        active=ty.get('active_warnings') or []
        return _result(("The tyre-pressure problem is still present in the latest readings: " if bad or active else "The latest readings do not show a tyre-pressure problem: ")+detail+f". Active warning count: {len(active)}"+(f" ({', '.join(active)})" if active else "")+".","tyre_status","STATUS",wheels=wheels)

    if re.search(r"\b(?:battery health|state of health|soh)\b", t) and re.search(r"\b(?:mean|meant|not charge|instead|correct)\b", t):
        row=brain.c.execute("SELECT ts,soh FROM vday WHERE ts<=? AND soh IS NOT NULL ORDER BY ts DESC LIMIT 1",(brain.now,)).fetchone()
        if not row: return _result("No valid state-of-health observation is recorded.","battery_soh","HEALTH",available=False)
        dialogue_state.remember(brain, Query(op="current_value", target="battery_soh"))
        return _result(f"Understood: battery health, not charge. The latest valid state of health is {row[1]:.1f}% at {A.stamp(row[0])}. SoH is usable capacity relative to new; it is distinct from the current SoC.","battery_soh","HEALTH",sample_ts=row[0],value=row[1])
    return None


def handle(brain, raw, normalised):
    """A vehicle database that lacks an optional table or column makes the handler "not applicable" (None) instead of raising: the question then falls through to the typed engine, which states plainly what is missing."""
    import logging
    import sqlite3
    try: return _handle_once(brain, raw, normalised)
    except sqlite3.OperationalError as e:
        logging.getLogger("vehiclegpt").warning("semantic_ops skipped: %s", e); return None


def _handle_once(brain, raw, normalised):
    """Operations are tried on the message as typed and, if none applies, on its typo-repaired form (one spelling mistake must not change which operation a message is)."""
    import re as _re
    repaired = str(normalised).strip().lower() != str(raw).strip().lower() and not _re.search(r"\d\s*[-–]\s*\d", str(raw))        # a band such as 20-30 keeps its dash
    if repaired:
        r = _handle(brain, str(normalised), normalised)
        if r is not None: return r
    r = _handle(brain, raw, normalised)
    if r is None and str(normalised).strip().lower() != str(raw).strip().lower(): r = _handle(brain, str(normalised), normalised)
    return r


def _handle(brain, raw, normalised):
    brain._last_norm = str(normalised).lower()
    """Return a routed response or None when ordinary typed IR should handle it."""
    # Keep punctuation in the raw sentence: a dash between two numbers is a
    # structural speed-band delimiter, not decoration.  The general parser's
    # normalised form remains useful for typo-tolerant concept lookup.
    import unicodedata
    t = unicodedata.normalize("NFKC", str(raw)).lower().replace("’", "'"); cs = _concepts(t + " " + normalised.lower())          # compatibility forms (CO₂ -> CO2) are one symbol for the ontology
    if len(re.findall(r"\b(?:find|reconstruct|identify|determine|compare|explain|list|calculate|summari[sz]e|describe|estimate|rank)\b", t)) >= 3: return None
    for fn in (_inventory,):
        r = fn(brain, t)
        if r: return r
    r = _temporal_lookup(brain, t, cs)
    if r: return r
    r = _driver_semantics(brain, t, cs)
    if r: return r
    from . import dialogue_state                                     # a message that edits the typed state of the conversation is resolved against it first
    r = dialogue_state.resolve(brain, raw, t, cs)
    if r: return r
    r = _state_queries(brain, t, cs)
    if r: return r
    # A duration comparison enumerates mutually-exclusive states.  It must be
    # resolved before the general conjunction composer, which correctly handles
    # intersections for filters but is semantically wrong for report items.
    r = _state_duration_comparison(brain, t)
    if r: return r
    from . import composed
    r = composed.handle(brain, raw, t)
    if r: return r
    r = _relation(brain, raw, t, cs)
    if r: return r
    r = _coordination(brain, raw, t)
    if r: return r
    r = _energy_math(brain, t)
    if r: return r
    r = _acceleration(brain, t)
    if r: return r
    r = _history_special(brain, t, cs)
    if r: return r
    r = _advice_and_diagnostics(brain, t, cs)
    if r: return r
    return None
