"""Vocabulary helpers shared by the rule parser (describe-by-parts fallback) and the interpretation checker: synonym folding, stemming, phrase sets for signals and events, unique best-candidate matching.
These are LEXICAL tools. The interpretation checker uses them only as one kind of support evidence, recorded with the phrase that matched."""
import re

from . import nl2ir, signal_registry as R


SYNONYMS = {"hot": "temperature", "hotter": "temperature", "hottest": "temperature", "warm": "temperature", "warmer": "temperature", "warmest": "temperature", "cold": "temperature", "colder": "temperature", "coldest": "temperature", "heat": "temperature", "heats": "temperature",
            "overheat": "temperature", "overheating": "temperature", "overheated": "temperature", "temp": "temperature", "temps": "temperature", "pack": "battery", "tire": "tyre", "tires": "tyre", "tyres": "tyre", "psi": "pressure", "bar": "pressure", "kpa": "pressure",
            "climate": "hvac", "ac": "hvac", "aircon": "hvac", "heater": "hvac", "braking": "brake", "brakes": "brake", "braked": "brake", "hard": "harsh", "hardest": "harsh", "harshest": "harsh", "volt": "v", "volts": "v", "12v": "12", "disengagement": "disengage",
            "disengaged": "disengage", "disengagements": "disengage", "switched": "disengage", "cut": "disengage", "kilometres": "km", "kilometers": "km", "distance": "range", "far": "range", "charged": "charge", "charging": "charge", "charges": "charge", "driver": "adas",
            "lane": "adas", "assist": "adas", "assistance": "adas", "blocked": "block", "blocking": "block", "faults": "fault", "warnings": "warning", "warn": "warning", "warned": "warning", "minimum": "min", "maximum": "max", "lowest": "min", "highest": "max"}
FILLER = {"control", "estimated", "level", "rate", "the", "of", "a", "an"}
VAGUE_TIME = re.compile(r"\b(lately|recently|these days|used to|than before|than it was|than it used|ago|a (?:month|week|while|few \w+) back|fortnight|week by week|day by day|over time|so far|this (?:week|month|year)|since|earlier|nowadays|worse|better|now)\b")


def _stem(w):
    w = SYNONYMS.get(w, w)
    return w[:5] if len(w) > 5 else w


def _tokens(text): return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in FILLER}


def _phrases_of(sig):
    out = [R.label_of(sig)] if sig in R.CONCEPTS else []
    out += [p for p, names in R.LEXICON.items() if sig in names]
    return out


def _described_by_parts(text_tokens, phrases):
    """True when every (stemmed, synonym-folded) word of at least one phrase for the signal or event appears in the question: the person described it in other words than the lexicon holds."""
    for ph in phrases:
        need = _tokens(ph)
        if need and need <= text_tokens: return ph
    return None


def _signal_hits(t):
    seen = {}
    for m in nl2ir._CONCEPT_RX.finditer(t):
        for n in R.LEXICON.get(m.group(1).replace("-", " "), ()) or R.LEXICON.get(m.group(1), ()): seen.setdefault(n, m.group(0))
    return seen




# ------------------------------------------------------------------------------------------------ unique parts-matching (used by the rule parser as a fallback when the strict lexicon finds nothing)
def _word_spans(t): return [(m.start(), m.end(), _stem(m.group(0))) for m in re.finditer(r"[a-z0-9]+", t) if m.group(0) not in FILLER]


def _unique_best(t, named):
    """named: {name: [phrases]} -> (name, spans) when exactly one name has a phrase (of the greatest size) whose every stemmed word occurs in the text, else None."""
    words = _word_spans(t); stems = {s for _, _, s in words}; best = {}
    for name, phrases in named.items():
        for ph in phrases:
            need = _tokens(ph)
            if need and need <= stems: best[name] = max(best.get(name, 0), len(need))
    if not best: return None
    top = max(best.values()); winners = [n for n, k in best.items() if k == top]
    if len(winners) != 1: return None
    name = winners[0]; need = set().union(*[_tokens(ph) for ph in named[name] if _tokens(ph) and _tokens(ph) <= stems and len(_tokens(ph)) == top])
    return name, top, [(a, b) for a, b, s in words if s in need]


def event_by_parts(t):
    """-> (event, spans) when the question describes one event type in other words than the lexicon holds (at least two of its words)."""
    named = {}
    for ph, e in nl2ir._EVENT_PHRASES.items():
        if e and e != "ANY_WARNING": named.setdefault(e, []).append(ph)
    r = _unique_best(t, named)
    return (r[0], r[2]) if r and r[1] >= 2 else None


def signal_by_parts(t):
    """-> (signal, spans) when the question describes one signal in other words than the lexicon holds (every word of one of its phrases present, and more than one word)."""
    named = {n: _phrases_of(n) for n in R.CONCEPTS}
    r = _unique_best(t, named)
    return (r[0], r[2]) if r and r[1] >= 2 else None
