"""Condition and negation grounding: do the plan's state conditions agree with what the request says about those states?  (validation evidence only)

Checks, per state predicate the plan lists in `c=`:
    polarity    the question mentions the state inside a negation scope ("not on the highway", "excluding charging", "ignoring parked time", "other than city")
                -> the plan must carry `not:<state>`;  the plan carries `not:<state>` but the question mentions the state with no negation cue -> issue
    invented    the plan lists a state the question (and the trusted previous plan) never mentions
It deliberately does NOT infer a missing condition from the absence of one: "driving time above 90" may mean the driving denominator or the whole period; only an explicit
scope cue ("while", "when", "during", "only", "on the <state>") followed by a state makes a dropped positive condition reportable.
State vocabulary is closed (the registry predicates and their words); it never selects an operation or builds an answer."""
import re

STATE_WORDS = {
    "parked": {"parked", "parking", "stationary", "standing"},
    "charging": {"charging", "charge", "plugged", "charger"},
    "city": {"city", "urban", "town", "citys"},
    "highway": {"highway", "motorway", "freeway", "expressway", "motorways", "highways"},
    "driving": {"driving", "moving", "motion", "underway", "driven"},
    "raining": {"rain", "raining", "rainy", "wet"},
    "night": {"night", "dark", "nighttime", "overnight"},
    "cold": {"cold", "freezing", "frost"},
    "idling": {"idling", "idle"},
    "fast_charging": {"fast-charging", "dc"},
    "regenerating": {"regenerating", "regen", "recuperating"},
    "hard_braking": {"hard-braking"}, "hard_acceleration": {"hard-acceleration"},
}
_NEG_CUE = {"not", "no", "never", "without", "excluding", "exclude", "except", "ignoring", "ignore", "apart", "outside", "off", "other", "leaving", "dry", "skip", "skipping", "omitting", "aside", "unless", "isnt", "wasnt", "dont", "doesnt"}
_NEG_PHRASE = [re.compile(p, re.I) for p in (r"\bother\s+than\b", r"\bapart\s+from\b", r"\baside\s+from\b", r"\bleaving\s+out\b", r"\bnot\s+counting\b", r"\bnot\s+including\b", r"\bdoes\s*n.?t\b|\bwas\s*n.?t\b|\bis\s*n.?t\b|\bwere\s*n.?t\b")]
_SCOPE_CUE = re.compile(r"\b(?:while|when|whenever|during|only|just|on\s+the|in\s+the|at|in)\s+(?:the\s+)?$", re.I)
_DRY = re.compile(r"\b(?:dry|not\s+raining|no\s+rain|without\s+rain)\b", re.I)


def _tokens(text):
    return [(m.group(0).lower(), m.start()) for m in re.finditer(r"[a-z][a-z'\-]*", text, re.I)]


def mentions(text):
    """-> {state: [(index, negated)]} for every closed-vocabulary state word, with a negation-scope flag (cue within the 5 preceding tokens, not crossing , ; . or 'but')."""
    toks = _tokens(text); out = {}
    for i, (w, pos) in enumerate(toks):
        w0 = w.replace("'", "")
        for state, ws in STATE_WORDS.items():
            if w0 in ws or w in ws:
                neg = False; j = i - 1; steps = 0
                while j >= 0 and steps < 5:
                    seg = text[toks[j][1] + len(toks[j][0]): toks[j + 1][1]] if j + 1 < len(toks) else ""
                    if re.search(r"[,;.:]|\bbut\b", seg) and j < i - 1: break
                    if toks[j][0].replace("'", "") in _NEG_CUE: neg = True; break
                    j -= 1; steps += 1
                if not neg:
                    neg = any(p.search(text[max(0, pos - 24):pos + len(w)]) for p in _NEG_PHRASE)
                if state == "raining" and w in ("dry",): neg = True
                out.setdefault(state, []).append((i, neg))
    if _DRY.search(text): out.setdefault("raining", []).append((-1, True))
    return out


_DIRECT = re.compile(r"\b(?:not|excluding|exclude|without|except|ignoring|never)\s+(?:counting\s+|including\s+|when\s+|while\s+|during\s+|the\s+time\s+(?:it\s+was\s+)?|any\s+|all\s+|on\s+the\s+|in\s+the\s+|on\s+|in\s+|the\s+|a\s+|my\s+|of\s+)*", re.I)


def direct_negations(text):
    """States that a DIRECT negation cue ('not on the highway', 'excluding charging', 'without parked time') is immediately attached to."""
    out = set()
    for m in _DIRECT.finditer(text):
        tail = re.findall(r"[a-z][a-z'\-]*", text[m.end():m.end() + 28].lower())[:2]
        for w in tail:
            w0 = w.replace("'", "")
            for state, ws in STATE_WORDS.items():
                if w0 in ws or w in ws: out.add(state)
    return out


def plan_issues(q, text, context_plan=None):
    """Conservative polarity check: a state under a DIRECT negation cue must not appear as a positive condition (the 'not on the highway' -> c=highway inversion).
    Everything else (dropped conditions, extra conditions, post-posed 'left out') is left to the interpreter and the end-to-end evaluation: lexical scope rules for those
    produced 5-9 % false rejections on gold plans (see msr/condition_on_gold.py) and are deliberately NOT enforced."""
    if q.op == "none": return []
    neg = direct_negations(text)
    pos = {c.predicate for c in q.scope.conditions if c.predicate and c.op != "not"}
    return [f"the question excludes {s} but the plan includes only {s}" for s in sorted(neg & pos)]
