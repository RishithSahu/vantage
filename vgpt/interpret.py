"""The editable INTERPRETATION PREVIEW: a frozen model proposes a typed interpretation, the person inspects and may change any part of it, deterministic code runs only what was confirmed.

    question -> model proposal (typed call, resolved against the conversation state) -> preview (fields generated from the plan language + the vehicle's capability catalogue) -> person edits / confirms -> typed validation -> execution

Nothing here interprets wording and nothing decides that a proposal is 'safe': the model's own scores do not separate right from wrong proposals (bench/diag_selector.py), so the person always sees the proposal. Controls are generated from the plan language (vgpt.toolspec) and the registry, never from a list of questions. The person may change the operation itself and any argument - including to a capability the model never proposed. Provenance is kept per field: model (proposed this turn), inherited (from the previous confirmed query), user (typed or chosen), and the original question is stored unchanged."""
from . import plancompile as PC, signal_registry as R, toolspec as T

SIMPLE = {"signal": "signal", "signal2": "signal", "event": "event", "tool": "tool", "stat": "stat", "which": "which", "unit": "unit", "entity": "entity", "metric": "metric", "order": "order", "k": "int", "trip_class": "tclass", "focus": "focus", "reason": "decl", "text": "text", "what": "text", "question": "text"}
NEVER = ("turn", "drop", "conditions_edit")                                    # the preview edits one RESOLVED query; follow-up bookkeeping stays out of the form


def catalogue(reg):
    """Everything the controls offer, from the registry and the plan language: [{id, title, unit, available}] for signals / events / tools, the named conditions, trip classes, ranking metrics, units, statistics, the period forms, and the operations."""
    cards = [c for c in reg.items]
    sig = [{"id": c.id, "title": c.title, "unit": c.unit, "available": bool(c.available)} for c in cards if c.kind == "signal"]
    evt = [{"id": c.id, "title": c.title, "available": bool(c.available)} for c in cards if c.kind == "event"]
    tools = [{"id": c.id[5:] if c.id.startswith("tool:") else c.id, "title": c.title} for c in cards if c.kind == "tool"]
    return {"signals": sig, "events": evt, "tools": tools, "conditions": [{"id": k, "title": v[0]} for k, v in R.PREDICATES.items()], "trip_classes": sorted({c.id.split(":", 1)[1] for c in cards if c.kind == "tripclass"}),
            "metrics": sorted({c.id.split(".", 1)[1] for c in cards if c.kind == "metric"}), "units": sorted({c.unit for c in cards if c.kind == "signal" and c.unit}), "stats": list(T.STATS), "cstats": list(T.CSTATS), "minmax": list(T.MINMAX), "evsel": list(T.EVSEL), "orders": list(T.ORDERS),
            "entities": list(T.ENTITIES), "foci": list(T.TOOLS_FOCI), "declines": list(T.DECLINES), "periods": ["now", "all time", "today", "yesterday", "day before yesterday", "this week", "last week", "previous week", "this month", "last month", "previous month", "last weekend", "this morning", "last night"],
            "period_help": "or: last N minutes|hours|days|weeks|months, or on YYYY-MM-DD", "cmp_ops": list(T.CMPOPS),
            "functions": [{"name": n, "label": T.func_label(n), "meaning": T.func_meaning(n)} for n in T.FUNCS if n not in ("say",)], "field_specs": {n: fields_of(n) for n in T.FUNCS if n != "say"}, "saykinds": list(T.SAYS)}


def fields_of(fn):
    """The editable fields of an operation, from the plan language: [{name, type, required}]"""
    out = []
    for pn, pt, req in T.func_params(fn):
        if pn in NEVER: continue
        out.append({"name": pn, "type": pt, "required": bool(req), "kind": pt})
    return out


def _source(plan, pn):
    key = T.PARAM_KEY.get(pn); prov = (plan.get("prov") or {}).get(key) if key else None
    return {"state": "inherited", "original": "inherited"}.get(prov, "model")


def preview_of(plan, text, error=""):
    """-> the preview for a (resolved) plan: {fn, label, meaning, fields:[{name, kind, required, value, source}], ...}. plan None = the model produced nothing usable: an empty form, the person chooses the operation."""
    if not plan or not T.plan_to_fn_args(plan):
        return {"ok": False, "fn": None, "fields": [], "error": error or "no usable proposal", "text": text}
    fn, args = T.plan_to_fn_args(plan); fields = []
    for f in fields_of(fn):
        v = args.get(f["name"]); fields.append(dict(f, value=v, source=_source(plan, f["name"]) if v not in (None, "", [], {}) else None))
    return {"ok": True, "fn": fn, "label": T.func_label(fn), "meaning": T.func_meaning(fn), "args": {k: v for k, v in args.items() if k not in ("turn",) and not k.endswith("_words")}, "fields": fields, "text": text}


def propose(brain, text, follow_up=True):
    """Ask the model for a proposal (no execution, no state change) and turn it into a preview. A follow-up proposal is resolved against the previous CONFIRMED query, so the person always sees a complete query."""
    pr = brain.propose(text, follow_up); err = pr.error or ""
    if pr.plan is None:
        out = preview_of(None, text, err or "the model produced no valid proposal")
    elif pr.fn in T.NO_TURN:
        out = {"ok": True, "fn": pr.fn, "label": T.func_label(pr.fn), "meaning": T.func_meaning(pr.fn), "args": dict(pr.args or {}), "text": text, "fields": [dict(f, value=(pr.args or {}).get(f["name"]), source="model" if (pr.args or {}).get(f["name"]) not in (None, "") else None) for f in fields_of(pr.fn)], "note": "the model did not propose a query"}
    else:
        base = (brain.state.prev_plan if follow_up else None) if pr.plan.get("turn", "new") in ("follow_up", "correction") else (brain.state.pending or {}).get("plan") if pr.plan.get("turn") == "clarification_reply" else None
        resolved = PC.merge(base, pr.plan) if base else PC.merge(None, pr.plan)
        out = preview_of(resolved, text, err)
    out.update(raw=pr.raw, timing=pr.timing, card_ids=pr.card_ids)
    return out


def diff(proposal_args, final_fn, final_args, proposal_fn):
    """The edits the person made: [{field, from, to}] (+ an operation change). Provenance of the final query: every field not listed here is the model's (or inherited) proposal."""
    edits = []
    if proposal_fn != final_fn: edits.append({"field": "operation", "from": proposal_fn, "to": final_fn})
    keys = [k for k in dict.fromkeys(list((proposal_args or {}).keys()) + list(final_args.keys())) if k != "turn"]
    for k in keys:
        a, b = (proposal_args or {}).get(k), final_args.get(k)
        if a != b: edits.append({"field": k, "from": a, "to": b})
    return edits


def clean_args(fn, args):
    """The confirmed fields as a call: empty values left out, the turn marked new (the person confirmed a complete resolved query)."""
    out = {} if fn in T.NO_TURN else {"turn": "new"}
    for k, v in (args or {}).items():
        if k == "turn" or v in (None, "", [], {}): continue
        out[k] = v
    return out
