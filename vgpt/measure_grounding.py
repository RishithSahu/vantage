"""Measure grounding: is the measure a plan selected actually supported by the request?  (validation evidence, never an answer generator)

A plan can be schema-valid and name an existing signal yet mean something else ("cabin CO2" -> mean coolant flow).  `trace_plan` returns one `Trace` per selected measure:

    explicit      a phrase of the question maps (central lexicon) to the selected measure, or the question uses its discriminative words,
                  or it names the measure's quantity kind (units / "temperature" / "fast") together with its component
    context       the question names no measure and the TRUSTED previous plan used it (follow-up); the quantity kind may also be inherited
    derived       the operation needs no measure
    contradicted  the question names a DIFFERENT registry measure, or a different component of the same quantity kind, and nothing supporting the selected one
    ungrounded    nothing in the question supports it (including the case where the question names a quantity the schema does not model)

`outcome_issues` grounds a `none` outcome in the telemetry schema: "unrecorded" needs POSITIVE evidence (a phrase naming a registry measure the database lacks, or a catalogued
absent quantity); a recorded measure named in the question contradicts it; no named measure at all means "I do not know which signal you mean" = ambiguous.

The module does not look at operations, windows or conditions and never produces an answer; it only supplies validation evidence."""
import difflib
import re
from dataclasses import dataclass

from . import lexicon as LX
from . import signal_registry as R

ABSENT_TERMS = {      # quantities the product's schema does not model at all: naming one is positive evidence for "not recorded"
    "cabin co2": "cabin carbon dioxide", "co2 level": "carbon dioxide", "co2": "carbon dioxide", "carbon dioxide": "carbon dioxide", "cabin air quality": "cabin air quality", "air quality": "air quality",
    "passenger": "occupancy", "passengers": "occupancy", "occupants": "occupancy", "seat occupancy": "occupancy", "seating capacity": "seating capacity", "seat capacity": "seating capacity", "seat belt": "seat belt use", "seatbelt": "seat belt use",
    "luggage": "load weight", "boot load": "load weight", "payload": "load weight", "roof load": "load weight", "oil level": "oil level", "engine oil": "oil", "transmission oil": "oil",
    "washer fluid": "washer fluid", "windscreen washer": "washer fluid", "heart rate": "driver physiology", "music": "audio", "radio": "audio", "song": "audio",
    "cabin humidity": "cabin humidity", "door opening": "door state", "door openings": "door state", "window position": "window state",
    "traffic light": "road environment", "speed camera": "road environment", "pothole": "road environment", "road surface": "road environment", "road sign": "road environment",
    "differential": "driveline temperature", "internal resistance": "cell internal resistance", "cell balancing": "cell balancing", "suspension": "suspension", "wheel alignment": "wheel alignment", "alignment": "wheel alignment",
}
_ABSENT_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in ABSENT_TERMS), key=len, reverse=True)) + r")\b", re.I)
_CONDITION_CONCEPTS = {"rain", "light"}          # registry concepts that a question uses as a CONDITION word ("in the rain"), not as the measure

_STOP = {"the", "a", "an", "of", "my", "our", "in", "on", "at", "for", "to", "is", "was", "what", "how", "level", "reading", "value", "current", "rate", "state", "status", "average", "mean", "total", "amount"}
QUANTITY = {      # WHAT KIND of quantity is meant (words and units)
    "temperature": {"temperature", "temp", "degrees", "degree", "celsius", "fahrenheit", "hot", "hotter", "hottest", "warm", "warmer", "warmest", "cold", "colder", "coldest", "heat", "heated", "heating", "overheat", "cool", "cooler"},
    "speed": {"speed", "fast", "faster", "fastest", "kph", "kmh", "velocity", "mph", "pace", "speedo", "speedometer", "slowest", "slow", "slower"},
    "power": {"power", "kw", "kilowatt", "kilowatts", "watt", "watts", "draw", "drawn"},
    "pressure": {"pressure", "kpa", "psi", "bar", "inflated", "inflation"},
    "flow": {"flow", "litres", "liters", "circulation", "circulating", "throughput"},
    "percent": {"charge", "soc", "percent", "percentage", "full", "empty", "remaining", "charged"},
    "energy": {"energy", "kwh", "electricity", "consumption", "consumed", "efficiency", "economy"},
    "distance": {"distance", "kilometres", "kilometers", "km", "miles", "far", "mileage", "driven", "odometer", "travelled"},
    "voltage": {"volt", "volts", "voltage"},
    "acceleration": {"acceleration", "accelerate", "braking", "deceleration"},
}
_ALL_QUANTITY_WORDS = set().union(*QUANTITY.values())
_WEAK_COMPONENTS = {"car", "vehicle", "air", "road", "charging", "control", "used", "unit"}   # too generic to identify a measure on their own
_FAMILY_OF_UNIT = {"km/h": "speed", "mph": "speed", "m/s": "speed", "C": "temperature", "F": "temperature", "K": "temperature", "kPa": "pressure", "bar": "pressure", "psi": "pressure", "kW": "power", "W": "power",
                   "hp": "power", "kWh": "energy", "Wh": "energy", "Wh/km": "energy", "L/min": "flow", "%": "percent", "V": "voltage", "km": "distance", "mi": "distance", "m/s2": "acceleration"}
COMPONENTS = {"battery", "pack", "motor", "coolant", "cabin", "outside", "outdoor", "ambient", "tyre", "tire", "tyres", "tires", "brake", "brakes", "wheel", "charger", "charging", "engine", "transmission",
              "aux", "auxiliary", "cell", "inverter", "hvac", "regen", "regenerative", "air", "road", "vehicle", "car"}


@dataclass
class Trace:
    measure: str
    status: str
    phrase: str = ""
    others: tuple = ()
    provenance: str = ""


def _concept_words(name):
    c = R.CONCEPTS.get(name)
    if c is None: return set()
    words = set()
    for s in (c.label,) + tuple(c.synonyms) + tuple(R.EXTRA_SYNONYMS.get(name, ())) + tuple(LX.PARAPHRASES.get(name, ())):
        words |= {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}
    return words


def _concept_family(name):
    c = R.CONCEPTS.get(name)
    return _FAMILY_OF_UNIT.get(c.unit) if c else None


def _components(name):
    w = _concept_words(name)
    return (w & COMPONENTS) | {x for x in w if x not in _ALL_QUANTITY_WORDS and x not in COMPONENTS and len(x) > 3}


_KNOWN = None
_COMPACT = None


def _known_words():
    """Every word of every registry concept (labels, synonyms, paraphrases): a correctly spelt word is never 'corrected'."""
    global _KNOWN
    if _KNOWN is None: _KNOWN = set().union(*(_concept_words(n) for n in R.CONCEPTS)) | set().union(*(set(re.findall(r"[a-z0-9]+", k)) for k in LX.SIGNAL_FORMS))
    return _KNOWN


def _compact_concept_words():
    """Joined spellings generated from the registry (``coolantflow`` etc.).

    This is deliberately schema-derived rather than a list of question-specific
    exceptions.  Only an unambiguous, complete compact spelling is expanded, so
    an unrelated compound such as ``headwind`` cannot become ``heading``.
    """
    global _COMPACT
    if _COMPACT is None:
        found = {}
        for name, c in R.CONCEPTS.items():
            surfaces = (c.label,) + tuple(c.synonyms) + tuple(R.EXTRA_SYNONYMS.get(name, ())) + tuple(LX.PARAPHRASES.get(name, ()))
            for surface in surfaces:
                parts = [w for w in re.findall(r"[a-z0-9]+", surface.lower()) if w not in _STOP]
                if len(parts) < 2: continue
                compact = "".join(parts)
                found.setdefault(compact, set()).add(tuple(parts))
        _COMPACT = {k: next(iter(v)) for k, v in found.items() if len(v) == 1}
    return _COMPACT


def _norm_words(text):
    """Words of the question; a misspelt component/quantity word is mapped to the nearest known one (same first letter, ratio >= 0.8)."""
    vocab = COMPONENTS | _ALL_QUANTITY_WORDS
    known = _known_words()
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in vocab or w in known or len(w) < 4: out.append(w); continue
        compact = _compact_concept_words().get(w)
        if compact:
            out.extend(compact); continue
        m = difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)
        out.append(m[0] if m and m[0][0] == w[0] else w)
    return out


def _question_families(text, words):
    fams = {f for f, ws in QUANTITY.items() if set(words) & ws}
    for m in LX.mentions(text):
        if m.kind == "unit" and m.ids and m.ids[0] in _FAMILY_OF_UNIT: fams.add(_FAMILY_OF_UNIT[m.ids[0]])
    return fams


def named_measures(text):
    """-> (explicit {concept: surface}, absent [(surface, quantity)]) over the original text."""
    explicit = {}
    for m in LX.mentions(text):
        if m.kind != "signal": continue
        for cid in m.ids:
            if cid in R.CONCEPTS and cid not in _CONDITION_CONCEPTS: explicit.setdefault(cid, m.surface)
    absent = [(m.group(0), ABSENT_TERMS[m.group(1).lower()]) for m in _ABSENT_RE.finditer(text)]
    return explicit, absent


def _context_targets(context_plan):
    if not context_plan: return set()
    from .query_ir import Query
    try: q = Query.from_text(context_plan)
    except Exception: return set()
    return {t for t in (q.target, q.target2) if t}


def trace_plan(q, text, context_plan=None):
    selected = [t for t in (q.target, q.target2) if t]
    if not selected: return [Trace("", "derived", provenance="operation names no measure")]
    explicit, absent = named_measures(text)
    words = _norm_words(text); wset = set(words); ctx = _context_targets(context_plan)
    qf = _question_families(text, words)
    ctx_fams = {_concept_family(c) for c in ctx} - {None}
    text_comps = wset & COMPONENTS
    out = []
    for m in selected:
        if m in explicit:
            out.append(Trace(m, "explicit", explicit[m], tuple(sorted(set(explicit) - {m})), "lexicon")); continue
        comps = _components(m); fam = _concept_family(m)
        covered = {w for surf in explicit.values() for w in re.findall(r"[a-z0-9]+", surf.lower())}        # words already used by a DIFFERENT named measure do not support this one
        comp_hit = sorted((comps & wset) - _ALL_QUANTITY_WORDS - _WEAK_COMPONENTS - covered)
        fam_ok = fam in qf or (not qf and fam in ctx_fams)
        if comp_hit and (fam_ok or (not qf and not ctx_fams)) and not absent:
            out.append(Trace(m, "explicit", " ".join(comp_hit) + (" + " + fam if fam_ok else ""), tuple(sorted(explicit)), "component + quantity kind")); continue
        if fam in qf and not text_comps and not explicit and not absent:
            out.append(Trace(m, "explicit", "quantity kind " + fam, (), "quantity kind only")); continue
        if not explicit and not absent and m in ctx:
            out.append(Trace(m, "context", "", (), "trusted previous plan")); continue
        if explicit: out.append(Trace(m, "contradicted", "; ".join(explicit.values()), tuple(sorted(explicit)), "question names other measures"))
        elif text_comps and not (text_comps & comps) and fam in qf:
            out.append(Trace(m, "contradicted", " ".join(sorted(text_comps)), (), "question names a different component of the same quantity kind"))
        else: out.append(Trace(m, "ungrounded", absent[0][0] if absent else "", tuple(a[1] for a in absent), "no phrase of the question supports this measure"))
    return out


def plan_issues(q, text, context_plan=None):
    """Validation issues from measure grounding ([] = every selected measure is supported)."""
    if q.op == "none": return []
    issues = []
    for t in trace_plan(q, text, context_plan):
        if t.status == "contradicted": issues.append(f"the plan measures {R.label_of(t.measure)} but the question names {t.phrase}")
        elif t.status == "ungrounded": issues.append(f"no part of the question supports measuring {R.label_of(t.measure)}" + (f" (it names {t.phrase})" if t.phrase else ""))
    return issues


def outcome_issues(why, text, profile=None, context_plan=None):
    """-> (issues, suggested_why).  `unrecorded` needs positive evidence; a recorded measure in the question contradicts it; nothing identifiable named -> ambiguous."""
    if why != "unrecorded": return [], why
    explicit, absent = named_measures(text)
    recorded = []
    for cid, surface in explicit.items():
        try: ok = profile.concept_available(cid) if profile is not None else None
        except Exception: ok = None
        if ok is True: recorded.append((cid, surface))
    if recorded: return [f"the outcome says {recorded[0][1]} is not recorded but this vehicle records it"], "contradicted"
    if explicit or absent: return [], "unrecorded"
    if _context_targets(context_plan): return [], "unrecorded"
    return [], "ambiguous"
