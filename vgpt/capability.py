"""Capability analysis: can THIS vehicle history answer THIS (already structurally valid) query - and if not, exactly why not and what would make it possible?

Four outcomes, never conflated (the old behaviour said "that doesn't look like a question about this car" for B and D):
    OK            known concept(s), enough data                         -> execute
    MISSING_DATA  known concept, no data and no way to derive it        -> "I understand you're asking for X; X is not recorded. ..."
    INSUFFICIENT  known concept, derivable in principle, but the signals/parameters needed for every derivation are missing
                  -> "X could be calculated from A + B or from C + D; you currently record neither."
    UNKNOWN       not a vehicle concept in the registry                  -> handled by the caller (out of scope / legacy path)
Estimates are allowed only where the registry declares the assumption (and the answer then says "estimated"); an electrical quantity is never silently used as a mechanical one."""
from dataclasses import dataclass, field

from . import signal_registry as R

OK, MISSING, INSUFFICIENT, UNKNOWN = "OK", "MISSING_DATA", "INSUFFICIENT", "UNKNOWN"
_ART = {"a": "an", "e": "an", "i": "an", "o": "an", "u": "an"}


@dataclass
class CapabilityReport:
    status: str = OK
    concept: str = ""                                    # the concept that blocked the query (or the main one)
    message: str = ""                                    # one plain sentence for the driver
    details: list = field(default_factory=list)         # the longer explanation (what would be needed, what is available instead)
    needs: list = field(default_factory=list)            # alternative requirement sets, each a list of concept / parameter names
    estimated: bool = False                              # the answer will rest on a declared assumption
    assumptions: list = field(default_factory=list)
    resolutions: dict = field(default_factory=dict)      # concept -> Resolution for every concept the query touches

    @property
    def ok(self): return self.status == OK


def _label(n):
    if n in R.PARAM_META:
        return {"battery_capacity_kwh": "the battery capacity", "wheelbase_m": "the wheelbase", "wheel_radius_m": "the wheel radius", "gear_ratio": "the motor-to-wheel gear ratio", "steering_ratio": "the steering ratio", "drivetrain_efficiency": "the drivetrain efficiency"}.get(n, n)
    c = R.CONCEPTS.get(n)
    return c.label if c else n


def concepts_of(q):
    """Every registry concept a query touches (targets, condition signals, concepts behind named predicates, the ranking metric)."""
    out = []
    def add(x):
        if x and x not in out: out.append(x)
    add(q.target); add(q.target2)
    for sc in (q.scope, q.scope_b):
        if sc is None: continue
        for c in sc.conditions:
            if c.signal: add(c.signal)
            elif c.predicate in R.PREDICATES:
                rule = R.PREDICATES[c.predicate][1]
                if rule[0] in ("signal", "derived"): add(rule[1])
                if rule[0] in ("mode", "mode_in"): add("drive_mode")
        if sc.select and sc.select.rank_by and sc.select.rank_by in R.CONCEPTS: add(sc.select.rank_by)
    for g in q.groups:
        if g in R.PREDICATES:
            rule = R.PREDICATES[g][1]
            if rule[0] in ("signal", "derived"): add(rule[1])
            if rule[0] in ("mode", "mode_in"): add("drive_mode")
    return out


def _need_sets(res):
    sets = []
    for alt in res.alternatives:
        req, params = alt[0], alt[1]; sets.append(list(req) + list(params))
    return sets


def _explain(concept, res, profile):
    """-> (one sentence for the driver, detail lines, status)."""
    label = _label(concept); sets = _need_sets(res); direct = concept in R.STORE
    def have(n): return (profile.param(n)[0] is not None) if n in R.PARAM_META else profile.concept_available(n)
    def part(n): return _label(n)
    if not sets and concept == "tyre_pressure":                                                              # the overall figure is the lowest of the four wheels: say which wheels exist instead of "not recorded"
        names = {"fl": "front-left", "fr": "front-right", "rl": "rear-left", "rr": "rear-right"}; have_w = [names[w] for w in names if profile.concept_available(f"tyre_pressure_{w}")]; miss_w = [names[w] for w in names if names[w] not in have_w]
        if have_w:
            _and = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
            return (f"I can't give the overall tyre pressure: it is the lowest of all four tyres and the {_and(miss_w)} tyre {'is' if len(miss_w) == 1 else 'are'} not recorded on this vehicle.",
                    [f"Recorded individually: {_and(have_w)}. Not recorded: {_and(miss_w)}.", "Ask for one of the recorded tyres to get a value; nothing was estimated or substituted."], MISSING)
    if not sets:
        return (f"I understand you're asking for {label}. That isn't recorded for this vehicle, so I can't give a value - and I won't estimate one.",
                [f"{label.capitalize()} is a known vehicle signal, but this vehicle's data contains no values for it, and nothing else that is recorded determines it."], MISSING)
    options = ([f"{label} directly"] if direct else []) + [" plus ".join(part(n) for n in s) for s in sets]
    first = f"I understand you're asking for {label}. I can't {'determine' if direct else 'calculate'} it from the signals currently recorded. I would need {_join_or(options[:3])}."
    det = []
    for s in sets:
        miss = [n for n in s if not have(n)]
        det.append(" plus ".join(part(n) for n in s) + (f": missing {', '.join(part(m) for m in miss)}" if miss else ": available"))
    formulas = [a[3] for a in res.alternatives if len(a) > 3]
    det = [f"{label.capitalize()} can be obtained from {'its own sensor' if direct else 'other signals'}" + (" or from:" if direct else ":")] + ["  - " + d for d in det]
    if formulas: det.append("Method: " + "; or ".join(formulas) + ".")
    related = [R.CONCEPTS[n].label for n in ("speed", "battery_power", "battery_soc") if profile.concept_available(n)]
    if related: det.append("What is recorded and related: " + ", ".join(related) + ".")
    det.append("Nothing was estimated or substituted: an electrical quantity is never used in place of a mechanical one, and no value is invented.")
    return first, det, INSUFFICIENT if not direct else MISSING


def _join_or(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + ", or " + items[-1] if len(items) > 2 else " or ".join(items)


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1] if items else "the required signals"


def assess(q, profile):
    """-> CapabilityReport for a structurally valid Query."""
    rep = CapabilityReport(); res_all = {}
    names = concepts_of(q)
    for n in names:
        if n not in R.CONCEPTS: rep.status, rep.concept, rep.message = UNKNOWN, n, f"'{n}' is not a signal I know."; return rep
        res_all[n] = profile.resolve(n)
    rep.resolutions = res_all
    if q.event and q.event.type in R.DETECTED_EVENTS and not profile.concept_available("coolant_flow"):
        rep.status, rep.concept, rep.message = MISSING, "coolant_flow", "I can't find when the coolant flow started to drop because coolant flow is not recorded."; return rep
    for n in names:                                        # the target first, then conditions
        r = res_all[n]
        if not r.available:
            msg, det, status = _explain(n, r, profile)
            rep.status, rep.concept, rep.message, rep.details, rep.needs = status, n, msg, det, _need_sets(r)
            return rep
    for n, r in res_all.items():
        if r.status == "estimated": rep.estimated = True; rep.assumptions += [a for a in r.assumptions if a not in rep.assumptions]
    return rep


def describe_availability(profile, concept):
    """For the 'do you record X?' question: one sentence per state."""
    r = profile.resolve(concept); lab = _label(concept)
    if r.status == "measured": return f"Yes - {lab} is recorded."
    if r.status in ("derived", "estimated"): return f"{lab.capitalize()} is not recorded directly, but I can {'estimate' if r.status == 'estimated' else 'calculate'} it from {', '.join(_label(x) for x in r.requires)} ({r.formula})."
    msg, _, _ = _explain(concept, r, profile); return msg
