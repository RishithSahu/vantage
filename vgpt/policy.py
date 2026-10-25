"""Clarification-first interpretation policy.

Three outcomes (the engineering thresholds are choices of this project, not requirements of the case study; coverage is always reported next to errors):
  ANSWER   the interpretation is supported by the question or the conversation (support check) and represents every explicit constraint (coverage check), and the query validates -> answer with evidence and disclose defaults
  CLARIFY  an ambiguity or an unrepresented constraint remains, or only candidate readings exist -> ask ONE targeted question that offers the concrete readings (never a generic "please rephrase")
  REFUSE   nothing executable can be formed (no candidate validates) -> say so, naming what is missing, instead of guessing
(A required signal, history or operation that is unavailable is not a refusal here: it is answered by the executor with an explained limitation, and an unavailable sensor is never offered as an option.)

Candidates come from: the rule parser's answer, the rule parser's partial parse (the "draft"), and the model's proposal. The model never answers by itself: its proposal is accepted only when it passes the same support + coverage checks, otherwise it becomes one of the readings offered.
`decide()` is deterministic given its inputs and is the same function used by the dialogue and by the evaluation harness."""
from dataclasses import dataclass, field
import copy
import re

from . import interpretation as I, nl2ir, signal_registry as R
from .paraphrase import describe
from .query_validator import validate


POLITE = r"(?:please |can you |could you |would you |will you |now |ok |okay |hey |just )*"
CONTROL_RX = re.compile(r"^" + POLITE + r"(?:(?:disable|enable|activate|deactivate|engage|disengage|override|bypass|unlock|raise|lower|increase|decrease|adjust|honk|flash|toggle|shut)(?=\s|$)|(?:turn|switch) |(?:set|start|stop|open|close|lock|reset) (?:the |my |this |that |it|all |both |everything|me |up|down|on|off|to |charging|driving|cruise|heat|ac |air))")   # a request to operate the vehicle, in any word order after the verb
COMMAND_RX = re.compile(r"^" + POLITE + r"(?:clear |erase|delete |remove |reveal |ignore (?:all |any )?(?:the )?(?:previous|prior|above|your)|forget (?:your|the) (?:instructions|rules)|print (?:your|the) system|write (?:me )?(?:a |an )?(?:poem|story|song))")                 # not a vehicle-control verb, but never answered as a data question


@dataclass
class Decision:
    kind: str                                   # answer | clarify | refuse
    query: object = None                        # the query to execute (answer)
    candidates: list = field(default_factory=list)       # readings offered (clarify)
    question: str = ""
    reasons: list = field(default_factory=list)
    source: str = ""                            # rules | model | draft
    disclosures: list = field(default_factory=list)


def _valid(q, profile):
    try: return profile is None or not validate(q, profile).codes()
    except Exception: return False


def variants(q, it):
    """Alternative readings of an ambiguous query, derived from the interpretation (not from the sentence)."""
    out = []
    for f in it.ambiguous():
        if f.name == "condition" and f.alt_values:
            for sig in f.alt_values:
                v = copy.deepcopy(q)
                for sc in (v.scope, v.scope_b):
                    for c in (sc.conditions if sc is not None else []):
                        if c.signal and c.to_text() == f.value: c.signal = sig
                out.append(v)
    if q.event is not None and q.op in ("count", "list_events") and any(x.name == "operation" and x.origin == "ungrounded" for x in it.fields):
        v = copy.deepcopy(q); v.op = "list_events" if q.op == "count" else "count"
        if v.op == "list_events": v.event.selector = "all"
        out.append(v)
    return out


def _dedupe(qs):
    seen = set(); out = []
    for q in qs:
        # Users choose between meanings, not serialized implementation details.
        # Two IRs that render as the same reading are one option even if a
        # hidden/default field makes their wire forms differ.
        k = re.sub(r"\s+", " ", describe(q).strip().lower())
        if k not in seen: seen.add(k); out.append(q)
    return out


def _ask(cands, why):
    opts = "; ".join(f"({i}) {describe(q)}" for i, q in enumerate(cands, 1))
    return f"I want to be sure I answer the right question. Do you mean {opts}; or ({len(cands) + 1}) neither of these?" + (f" ({why})" if why else "")


STATUS_RX = re.compile(r"^(?:so |ok |okay |hey |please )*(?:how(?:'s| is| are| about)|what(?:'s| is| are) (?:the )?(?:state|status|condition|health)|tell me about|check|give me (?:a |the )?(?:status|summary|overview)|status of|brief me|summary of)\b")
FUEL_RX = re.compile(r"\b(fuel|petrol|diesel|gasoline|gas mileage|mpg|miles per gallon|litres? per 100|liters? per 100|l/100 ?km)\b")


NOT_STATUS_RX = re.compile(r"\b(?:was|were|did|had|has|have|been|ago|last|previous|earlier|before|after|since|when|why|happen\w*|track|record\w*|support\w*|measure\w*|available|declin\w*|drop\w*|worse|better|chang\w*|increas\w*|decreas\w*|trend\w*|getting|compar\w*|than|how (?:often|many|long))\b")


def _status_question(text, now, tz):
    """A question about the CURRENT state of a component (\"how's my battery\", \"is the camera blocked right now\") with no period and no statistic. The status/health engines answer it from the live state; offering invented statistics as readings would be worse."""
    t = nl2ir.normalise(text)
    if any(ts.type != "current" for ts in nl2ir._times(nl2ir._S(t), now, tz)) or any(re.search(rx, t) for rx, _ in nl2ir.AGG_WORDS): return False                       # "now" is the status engines' own time; any other period is a data question
    if NOT_STATUS_RX.search(t): return False                                                                                       # past tense, an event reference, a trend, a comparison or a question about what is recorded is a data question
    if STATUS_RX.match(t): return True
    try:
        from .brain import parse_fast
        intent, _p, conf = parse_fast(text)
    except Exception: return False
    return intent in ("STATUS", "HEALTH") and conf >= 0.88


def _fuel_gate(text, profile, prev, now, tz):
    """A question about FUEL on a vehicle that records none. The honest answer is the limitation plus the electric equivalent, offered as a reading the person can accept or refuse: nothing is estimated and nothing is silently substituted."""
    if not FUEL_RX.search(text): return None
    try:
        if profile is not None and any(profile.concept_available(n) for n in ("fuel_level", "fuel_rate")): return None
    except Exception: pass
    # On a known EV, a request for fuel remaining has one useful, honest
    # vehicle equivalent: remaining electric range (which itself discloses SoC
    # and the consumption basis).  Energy already consumed is the opposite
    # quantity and must never be offered as "fuel left".
    remaining = bool(re.search(r"\b(?:level|left|remain\w*|available|do i have)\b", text)) or (bool(re.search(r"\bhow much\b", text)) and not re.search(r"\b(?:did|used?|consum\w*|burn\w*|spent?|cost)\b", text))
    if remaining:
        from .query_ir import Query
        q = Query.from_text("op=current_value tgt=range_estimate t=now")
        q.notes = ["This vehicle is electric and has no petrol or fuel gauge; the reported equivalent is battery charge and estimated remaining range."]
        return Decision("answer", query=q, reasons=["fuel is unavailable on this EV; remaining electric range is the capability-aware equivalent"], source="fuel-gate",
                        disclosures=list(q.notes))
    alt = re.sub(r"\b(?:fuel (?:efficiency|economy|consumption|use)|gas mileage|mpg|miles per gallon|(?:litres?|liters?) per 100(?: ?km)?|l/100 ?km)\b", "average energy consumption", text); alt = re.sub(FUEL_RX, "energy", alt); pr = nl2ir.parse(alt, now=now, tz=tz, profile=profile)
    cands = [q for q in ([pr.query] if pr.query is not None and pr.query.op else []) if _valid(q, profile)]
    if not cands: return Decision("refuse", question="This vehicle is electric and records no fuel data, so I cannot answer a question about fuel. I can report energy use, charge level, range and efficiency in Wh/km - which do you mean?", reasons=["fuel is not recorded"], source="fuel-gate")
    return Decision("clarify", candidates=cands, question="This vehicle is electric: it records no fuel data, and I will not estimate any. The closest electric equivalent I can compute is: " + describe(cands[0]) + ". Is that what you want?  (1) yes; (2) neither of these.", reasons=["fuel is not recorded; the electric equivalent is offered, not assumed"], source="fuel-gate")


def _qualifier_refusal(text, rules, profile):
    """A known component named together with a qualifier that no recorded quantity carries ("engine OIL temperature", "tyre TREAD depth"): the quantity asked for is not the one that was recognised.  The recognised signal must not be offered as if it were the answer: say what is not recorded and name the nearest recorded ones.  A concept that the vehicle does not record at all is left to the executor's missing-data explanation."""
    if not rules.unexplained: return None
    if rules.query is not None and rules.query.op in ("explain_change", "detect_change", "trend", "explain_event"): return None      # "why has my range reduced": the unexplained word is the verb being asked about, not a qualifier
    toks = re.findall(r"[a-z0-9]+", nl2ir.normalise(text)); unexpl = [w for w in rules.unexplained if w.isalpha() and w in toks and w not in nl2ir.STOP and not w.endswith(("ed", "ing"))]
    if not unexpl: return None
    q = rules.query
    if q is not None and q.target in R.CONCEPTS:
        lab = R.CONCEPTS[q.target].label.split(); syn = {x for s_ in R.CONCEPTS[q.target].synonyms for x in s_.split()}
        idx = [i for i, w in enumerate(toks) if w in lab or w in syn]; near = [w for w in unexpl if any(abs(toks.index(w) - i) <= 1 for i in idx)]
        names = [R.label_of(q.target)]
    else:                                                                                              # no reading at all: the component noun that the ontology knows ("tyre", "battery") and its recorded quantities
        comp = [w for w in toks if w in I.REFERENT]
        if not comp: return None
        idx = [toks.index(comp[0])]; near = [w for w in unexpl if abs(toks.index(w) - idx[0]) <= 2]
        names = [R.label_of(c) for c in sorted(I.REFERENT[comp[0]]) if c in R.CONCEPTS][:3]
    if not near: return None
    lo = min(idx + [toks.index(w) for w in near]); hi = max(idx + [toks.index(w) for w in near]); phrase = " ".join(toks[lo:hi + 1])
    return Decision("refuse", question=f"I do not record “{phrase}”. The nearest recorded quantit{'y is ' if len(names) == 1 else 'ies are '}{' and '.join(names)}, but I will not answer with " + ("it" if len(names) == 1 else "them") + " unless you ask for it by name.", reasons=["an unrecorded qualifier of a known quantity"], source="qualifier")


def decide(text, rules, model_query=None, prev=None, profile=None, now=None, tz=0, check_rules=True):
    """rules: the nl2ir.Parse of the question; model_query: the model's proposal (Query) or None. -> Decision."""
    if CONTROL_RX.match(nl2ir.normalise(text)): return Decision("refuse", question="That is a request to change or control something. I only read recorded vehicle data and cannot operate the vehicle, so I have not done anything.", reasons=["a vehicle-control request, not a question about recorded data"], source="gate-control")
    if COMMAND_RX.match(nl2ir.normalise(text)): return Decision("refuse", question="That is a request to change or control something. I only read recorded vehicle data and cannot operate the vehicle.", reasons=["a command, not a question about recorded data"], source="gate")      # read-only assistant: never a clarification, never an action
    fg = _fuel_gate(text, profile, prev, now, tz)
    if fg is not None: return fg
    qr = _qualifier_refusal(text, rules, profile)
    if qr is not None: return qr
    if rules.query is not None and rules.query.target and profile is not None and rules.query.target in R.CONCEPTS and not profile.concept_available(rules.query.target) and not rules.clarify:
        return Decision("answer", rules.query, source="rules", reasons=["the quantity is not recorded: the executor explains what is missing"])           # an operation ambiguity is moot for a quantity the vehicle does not record
    if rules.clarify:
        cands = []
        if rules.pending is not None:                                                                  # "which temperature?": each option is a concrete reading
            for o in rules.options:
                w = o.split()[0].lower()
                for ctx in nl2ir.HEAD_BY_CONTEXT.values():
                    if ctx and w in ctx:
                        q = copy.deepcopy(rules.pending); q.target = ctx[w]; q.op = q.op or "aggregate"; q.agg = q.agg or ("mean" if q.op == "aggregate" else None); cands.append(q); break
        return Decision("clarify", candidates=[c for c in _dedupe(cands) if _valid(c, profile)], question=rules.clarify, reasons=["the question names a quantity that has several sensors or a reference with no antecedent"], source="rules")
    if rules.ok:
        q = rules.query; it = I.interpret(text, q, prev, source="rules", now=now, tz=tz); iss = it.issues()
        if not iss or not check_rules: return Decision("answer", q, source="rules", disclosures=it.disclosures(), reasons=iss)
        if _status_question(text, now, tz): return Decision("refuse", question="", reasons=["a present-state question about a component: answered by the older status engines (not typed-pipeline verified), not by an invented statistic"], source="status")
        cands = _dedupe([q] + variants(q, it)); cands = [c for c in cands if _valid(c, profile)]
        if it.ambiguous() or it.unresolved: return Decision("clarify", candidates=cands, question=(it.clarifications()[0]["question"] if len(cands) < 2 and it.clarifications() else _ask(cands, "")), reasons=iss, source="rules")
        return Decision("clarify", candidates=cands, question=_ask(cands, "some part of your question did not match what I understood"), reasons=iss, source="rules")
    cands = []; draft = rules.query if (rules.query is not None and rules.query.op) else None
    for src, q in (("model", model_query), ("draft", draft)):
        if q is None or q.op == "none": continue
        if not _valid(q, profile): continue
        it = I.interpret(text, q, prev, source=src, now=now, tz=tz)
        if src == "model" and not it.issues(): return Decision("answer", q, source=src, disclosures=it.disclosures())    # a model proposal that passes support + coverage (experimental configurations only)
        cands += [q] + variants(q, it)                                                                            # the rules' partial parse is never answered without confirmation
    if _status_question(text, now, tz):
        return Decision("refuse", question="", reasons=["a present-state question about a component: answered by the older status engines (not typed-pipeline verified), not by an invented statistic"], source="status")
    from . import candidates as CD                                                                       # the deterministic generator adds every reading the question supports (no model needed)
    gen = CD.generate(text, prev, profile, now, tz, rules_draft=draft)
    # a reading proposed by the generator or by the rules' partial parse is CONFIRMED with the person, never answered on structural checks alone; readings the checks flag are dropped when supported ones exist
    pool = _dedupe(cands + [q for q, _ in gen]); pool = [c for c in pool if _valid(c, profile)]
    scored = sorted(((len(I.interpret(text, c, prev, source="candidate", now=now, tz=tz).issues()), i, c) for i, c in enumerate(pool)), key=lambda x: (x[0], x[1]))
    cands = [c for n, _, c in scored if n == 0][:3] or [c for _, _, c in scored][:3]
    if cands: return Decision("clarify", candidates=cands, question=_ask(cands, ""), reasons=["no reading was fully supported by the question"], source="model" if model_query is not None else "draft")
    return Decision("refuse", question="I could not turn that into something I can compute from this vehicle's data. Which signal, event or period do you mean?", reasons=["no valid reading"], source="none")


def decide_by_selection(text, rules, selector, prev=None, profile=None, now=None, tz=0, check_rules=True, topk=6):
    """The model-role variant: when the rules cannot answer, a deterministic generator proposes readings and the (frozen) model only chooses among them. -> (Decision, info)
    A single fully supported reading is answered without the model; several are put to the model; the chosen reading is answered only if it passes support + coverage, otherwise it is offered (with the runner-up) as a clarification."""
    from . import candidates as CD
    info = {"candidates": None, "selected": None, "model_ms": 0.0}
    if CONTROL_RX.match(nl2ir.normalise(text)) or COMMAND_RX.match(nl2ir.normalise(text)): return decide(text, rules, None, prev, profile, now, tz, check_rules), info          # the read-only gate applies to every configuration (a safety rule, not a tuning change)
    if rules.clarify or rules.ok: return decide(text, rules, None, prev, profile, now, tz, check_rules), info
    draft = rules.query if (rules.query is not None and rules.query.op) else None
    cands = CD.generate(text, prev, profile, now, tz, rules_draft=draft); info["candidates"] = [q.to_text() for q, _ in cands]
    if not cands: return Decision("refuse", question="I could not turn that into something I can compute from this vehicle's data. Which signal, event or period do you mean?", reasons=["no candidate reading"], source="none"), info
    clean = [(q, it) for q, it in cands if not it.issues()]
    if len(clean) == 1 and len(cands) == 1: return Decision("answer", clean[0][0], source="candidate", disclosures=clean[0][1].disclosures()), info
    top = cands[:topk]; idx, mi = selector.select(text, prev, [q for q, _ in top]); info["model_ms"] = mi.get("ms", 0.0); info["model_ptok"] = mi.get("ptok"); info["selected"] = idx
    if idx is not None:
        q, it = top[idx]
        if not it.issues(): return Decision("answer", q, source="candidate+model", disclosures=it.disclosures()), info
        order = [q] + [c for c, _ in top if c is not q][:1]
    else: order = [c for c, _ in top[:2]]
    order = _dedupe(order)
    return Decision("clarify", candidates=order, question=_ask(order, ""), reasons=["the chosen reading is not fully supported by the question" if idx is not None else "the model found no matching reading"], source="candidate+model"), info
