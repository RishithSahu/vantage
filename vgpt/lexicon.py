"""Central, auditable domain lexicon: which words may be mapped to which canonical vehicle entities, units and abbreviations - and nothing else.

What this module IS:   a lookup of surface forms -> canonical identifiers (signals, events, states, units) with provenance and an ambiguity flag.
What it is NOT:        a router.  It never selects an operation, a window, a condition or an answer; it never rewrites the user's text (`mentions` returns spans over the ORIGINAL string).
Ambiguity is preserved: a surface form that maps to more than one canonical id is returned with all of them and `ambiguous=True`; resolving it is the interpreter's / context's job,
never a silent choice here.

Sources (single source of truth, no duplicated word lists):
    signals     vgpt.signal_registry.LEXICON (Concept.synonyms, labels, EXTRA_SYNONYMS)
    units       UNIT_ALIASES below (checked against signal_registry.UNIT_FAMILIES by the tests)
    abbreviations ABBREVIATIONS below (expansions are glossary terms, not signals)
Every table row carries a rationale; tests/test_lexicon.py covers each unit alias, each abbreviation, ambiguity, non-matches and span preservation."""
import re
from dataclasses import dataclass

from . import signal_registry as R

UNIT_ALIASES = {      # surface form -> canonical unit token of signal_registry.UNIT_FAMILIES
    "kph": "km/h", "kmh": "km/h", "km/hr": "km/h", "kilometres per hour": "km/h", "kilometers per hour": "km/h", "kilometres an hour": "km/h",
    "mph": "mph", "miles per hour": "mph", "metres per second": "m/s", "meters per second": "m/s",
    "celsius": "C", "degrees celsius": "C", "centigrade": "C", "°c": "C", "deg c": "C", "fahrenheit": "F", "degrees fahrenheit": "F", "°f": "F", "kelvin": "K",
    "kilowatt": "kW", "kilowatts": "kW", "kw": "kW", "watt": "W", "watts": "W", "horsepower": "hp", "bhp": "hp",
    "kilowatt hour": "kWh", "kilowatt hours": "kWh", "kilowatt-hours": "kWh", "kwh": "kWh", "watt hour": "Wh", "watt hours": "Wh", "wh": "Wh",
    "kilopascal": "kPa", "kilopascals": "kPa", "kpa": "kPa", "bar": "bar", "psi": "psi", "pounds per square inch": "psi",
    "volt": "V", "volts": "V", "amp": "A", "amps": "A", "ampere": "A", "amperes": "A", "percent": "%", "per cent": "%", "pct": "%",
    "litres per minute": "L/min", "liters per minute": "L/min", "l/min": "L/min", "lpm": "L/min", "revs per minute": "rpm", "revolutions per minute": "rpm",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km", "km": "km", "mile": "mi", "miles": "mi", "metre": "m", "metres": "m",
    "newton metres": "Nm", "newton meters": "Nm", "g": "g", "m/s2": "m/s2", "metres per second squared": "m/s2",
}
PARAPHRASES = {   # reviewed paraphrases (source: the project's own hand-written generator vocabularies, none from any benchmark); each maps to ONE concept
    'speed': ['ground speed', 'how fast i was going', 'km/h reading', 'my speed', "the car's pace", "the car's speed", 'the kph reading', 'the speedo reading', 'the speedometer figure', 'travel speed', 'vehicle velocity'],
    'battery_temperature': ['battery cell heat', 'battery pack heat', 'cell block temperature', 'high-voltage battery temp', 'how hot the battery was', 'how many degrees the pack read', 'hv battery heat', 'temperature of the battery', "the battery's heat", "the battery's temperature", "the pack's temperature", 'the temperature of the traction battery', 'traction pack temperature'],
    'motor_temperature': ['drive unit temperature', 'e-motor heat', 'how hot the motor was', 'motor core temp', 'motor degrees', 'motor winding temp', 'motor winding temperature', 'temperature of the motor', "the e-motor's temperature", 'the heat of the motor', "the motor's heat level", 'the temperature of the drive motor', 'traction motor heat'],
    'coolant_flow': ['circulation rate of the coolant', 'coolant circulation', 'coolant throughput', 'coolant volume per minute', 'cooling water flow', 'flow of coolant', 'flow through the cooling circuit', 'pump output flow', "the cooling loop's flow", 'the flow rate of the coolant'],
    'ambient_temperature': ['air temperature around the car', 'ambient heat', 'external air temperature', 'outdoor temperature', 'outside air heat', 'outside heat', 'temperature outside the car', 'the air temperature around the car', 'the outdoor air temp'],
    'battery_soc': ['battery fill level', 'battery level percentage', 'charge state', 'how charged the pack was', 'how full the battery was', 'state-of-charge figure', 'the charge level', 'the charge percentage'],
    'wh_per_km': ['efficiency in wh per km', 'energy consumption per kilometre', 'energy per kilometre', 'energy use per km', 'energy used per km'],
    'battery_power': ['electrical power draw', 'the power drawn from the battery', 'traction power'],
    'regen_power': ['power recovered when braking'],
    'charging_power': ['power delivered while charging'],
    'hvac_power': ['heater and ac power draw'],
    'cabin_temperature': ['the temperature inside the car'],
    'aux_battery_voltage': ['12 volt battery voltage', "the small battery's voltage"],
    'tyre_pressure': ['the pressure in the tyres'],
    'energy_used': ['kilowatt-hours drawn', 'the electricity consumed'],
    'distance_driven': ['distance travelled', 'kilometres covered', 'mileage covered'],
    'regen_energy': ['energy fed back by braking'],
    'brake_pressure': ['hydraulic brake pressure', 'pressure in the brake circuit'],
    'range_estimate': ['predicted remaining range'],
}

ABBREVIATIONS = {     # abbreviation -> (expansion, kind).  kind "glossary": a term to be defined; kind "signal": names a signal (see signal_registry)
    "soc": ("state of charge", "signal"), "soh": ("state of health", "signal"), "dtc": ("diagnostic trouble code", "glossary"), "tpms": ("tyre pressure monitoring system", "glossary"),
    "adas": ("advanced driver assistance system", "glossary"), "hv": ("high voltage", "modifier"), "lv": ("low voltage", "modifier"), "ev": ("electric vehicle", "glossary"),
    "regen": ("regenerative braking", "glossary"), "hvac": ("heating, ventilation and air conditioning", "signal"), "obd": ("on-board diagnostics", "glossary"),
    "can": ("controller area network", "glossary"), "ota": ("over-the-air update", "glossary"), "dc": ("direct current", "modifier"), "ac": ("alternating current / air conditioning", "ambiguous"),
    "kwh": ("kilowatt hour", "unit"), "wh/km": ("watt hours per kilometre", "unit"),
}
_RATIONALE = {"UNIT": "unit spelling variants map to one canonical unit of the same physical family (values are converted by signal_registry.convert, never here)",
              "ABBR": "abbreviations are expanded for display / grounding only; the expansion is not an interpretation of the request",
              "SIGNAL": "registry synonyms: each surface form maps to the concept(s) whose label or synonym list contains it"}


@dataclass(frozen=True)
class Mention:
    surface: str
    start: int
    end: int
    kind: str                 # signal | unit | abbreviation
    ids: tuple                # canonical identifiers (more than one => ambiguous)
    ambiguous: bool
    source: str               # which table produced it


def _merged_signals():
    t = {f: set(ids) for f, ids in R.LEXICON.items()}
    for cid, ps in PARAPHRASES.items():
        for p in ps: t.setdefault(p.lower(), set()).add(cid)
    return t


SIGNAL_FORMS = _merged_signals()
CANONICAL_UNIT_FORMS = {u: u for us in R.UNIT_FAMILIES.values() for u in us if len(u) >= 2 and u not in ("none", "category", "ratio", "count")}


def _word_boundary(text, a, b):
    return (a == 0 or not text[a - 1].isalnum()) and (b == len(text) or not text[b].isalnum())


def _scan(text, table, kind, source, getids):
    low = text.lower(); out = []
    for form in sorted(table, key=len, reverse=True):
        start = 0
        while True:
            i = low.find(form, start)
            if i < 0: break
            j = i + len(form)
            if _word_boundary(low, i, j): out.append(Mention(text[i:j], i, j, kind, tuple(sorted(getids(form))), len(set(getids(form))) > 1, source))
            start = j
    return out


def mentions(text):
    """-> list[Mention] over the ORIGINAL text, longest match first, overlapping shorter matches removed.  Never modifies or normalises the input."""
    found = []
    found += _scan(text, SIGNAL_FORMS, "signal", "signal_registry.LEXICON + lexicon.PARAPHRASES", lambda f: SIGNAL_FORMS[f])
    found += _scan(text, UNIT_ALIASES, "unit", "lexicon.UNIT_ALIASES", lambda f: {UNIT_ALIASES[f]})
    found += _scan(text, {k.lower(): v for k, v in CANONICAL_UNIT_FORMS.items()}, "unit", "signal_registry.UNIT_FAMILIES", lambda f: {CANONICAL_UNIT_FORMS[next(u for u in CANONICAL_UNIT_FORMS if u.lower() == f)]})
    found += _scan(text, ABBREVIATIONS, "abbreviation", "lexicon.ABBREVIATIONS", lambda f: {ABBREVIATIONS[f][0]})
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    out = []; last_end = -1
    for m in found:
        if m.start >= last_end: out.append(m); last_end = m.end
    return out


def canonical_unit(surface):
    """-> canonical unit or None (a non-match is None, never a guess)."""
    s = re.sub(r"\s+", " ", str(surface).strip().lower())
    if s in UNIT_ALIASES: return UNIT_ALIASES[s]
    for u in (u for us in R.UNIT_FAMILIES.values() for u in us):
        if s == u.lower(): return u
    return None


def signal_candidates(phrase):
    """-> sorted tuple of canonical signal ids for an exact surface form (may hold several: ambiguity is kept), or ()."""
    return tuple(sorted(SIGNAL_FORMS.get(re.sub(r"\s+", " ", phrase.strip().lower()), ())))


def audit_rows():
    """Machine-readable table of every mapping with its rationale (for results/semantic_recovery/semantic_data_audit.md)."""
    rows = [{"surface": f, "maps_to": ";".join(sorted(ids)), "table": "SIGNAL", "ambiguous": len(ids) > 1, "rationale": _RATIONALE["SIGNAL"]} for f, ids in sorted(SIGNAL_FORMS.items())]
    rows += [{"surface": f, "maps_to": u, "table": "UNIT", "ambiguous": False, "rationale": _RATIONALE["UNIT"]} for f, u in sorted(UNIT_ALIASES.items())]
    rows += [{"surface": f, "maps_to": v[0], "table": "ABBR", "ambiguous": v[1] == "ambiguous", "rationale": _RATIONALE["ABBR"]} for f, v in sorted(ABBREVIATIONS.items())]
    return rows
