"""Schema of the plan language, generated from the ONE authoritative definition (vgpt/toolspec.py) and the capability registry.
`enums(registry)` is the single source of every closed value set (registered signals, events, named conditions, tools, trip classes, ranking metrics, units); the generation grammar (plangrammar.py) and the tool schemas shown to the model (`function_schemas`) both read it, and the compiler's required-argument checks read toolspec's own table, so the prompt, the grammar and the compiler cannot disagree.
A schema carries: parameter descriptions, valid enum values, array item schemas, the properties of the threshold and selection objects, and which arguments a NEW request requires (a follow-up, correction or clarification reply states only what changes, so nothing but `turn` is schema-required; each argument a new request needs says so in its description)."""
from . import signal_registry as R
from . import toolspec as T


def enums(registry):
    return {"signals": registry.signal_ids(), "events": registry.event_ids(), "conditions": registry.condition_ids(), "tools": [c.id.split(":", 1)[1] for c in registry.items if c.kind == "tool"],
            "classes": [c.id.split(":", 1)[1] for c in registry.items if c.kind == "tripclass"], "metrics": sorted({c.id.split(".", 1)[1] for c in registry.items if c.kind == "metric"}),
            "units": sorted({u for us in R.UNIT_FAMILIES.values() for u in us if u not in ("none", "category")})}


PERIOD_DOC = "period phrase (forms listed in the instructions)"


def _type_schema(pt, E):
    """The JSON Schema of one argument type. Compact on purpose: enums where the set is small and closed; the forms of a period and the named conditions are listed ONCE in the instruction rather than repeated in every function."""
    def s(d=None, **k):                                                                                  # a string; an enum needs no type (it is self-describing)
        o = {} if "enum" in k else {"type": "string"}; o.update(k)
        if d: o["description"] = d
        return o
    threshold = {"type": "object", "properties": {"signal": s(), "op": s(enum=list(T.CMPOPS)), "value": {"type": "number"}, "unit": s()}, "required": ["signal", "op", "value"]}
    return {"signal": s("signal id from the cards"), "event": s("event id from the cards"), "stat": s(enum=list(T.STATS)), "cstat": s(enum=list(T.CSTATS)), "minmax": s(enum=list(T.MINMAX)), "evsel": s(enum=list(T.EVSEL)), "unit": s("unit asked for"),
            "conds": {"type": "array", "description": "named conditions or thresholds", "items": {"anyOf": [s(), threshold]}}, "grps": {"type": "array", "items": s()}, "period": s(PERIOD_DOC), "dur": s("amount and unit, e.g. 10 minutes"),
            "int": {"type": "integer"}, "text": s(), "tool": s(enum=E["tools"]), "focus": s(enum=list(T.TOOLS_FOCI)), "decl": s(enum=list(T.DECLINES)), "saykind": s(enum=list(T.SAYS)), "entity": s(enum=list(T.ENTITIES)), "metric": s(enum=E["metrics"]),
            "order": s(enum=list(T.ORDERS)), "tclass": s(enum=E["classes"]),
            "sel": {"type": "object", "description": "which trips or sessions", "properties": {"entity": s(enum=list(T.ENTITIES)), "metric": s(enum=E["metrics"]), "order": s(enum=list(T.ORDERS)), "k": {"type": "integer"}, "class": s(enum=E["classes"])}, "required": ["entity"]},
            "span": s("exact words from the message"), "turn": s(enum=list(T.TURNS)), "cedit": s(enum=list(T.CEDITS)), "dropl": {"type": "array", "items": s(enum=list(T.DROPS))}}[pt]


def function_schemas(registry, level="rich", grounded=True):
    """[{name, description, parameters}] for every function, in definition order. level="rich": a complete JSON Schema (descriptions, enums, array items, threshold / selection properties, required). level="compact": the name, the definition and the parameter names with simple types only (an ablation: how much does schema richness buy per prompt token?)."""
    E = enums(registry); out = []
    if level == "compact":
        short = {"conds": "array", "grps": "array", "dropl": "array", "int": "integer", "sel": "object"}
        return [{"name": name, "description": T.func_meaning(name), "parameters": {pn: short.get(pt, "string") for pn, pt, _ in T.all_params(name, grounded)}} for name in T.FUNCS]
    for name in T.FUNCS:
        props = {}; req = []
        for pn, pt, r in T.all_params(name, grounded):
            sch = dict(_type_schema(pt, E))
            if pt == "span": sch["description"] = f"exact words stating the {pn[:-6]}"
            if r and name not in T.NO_TURN: sch["description"] = ("Required for a new request. " + sch["description"]).strip() if sch.get("description") else "Required for a new request."
            if r and name in T.NO_TURN: req.append(pn)
            props[pn] = sch
        for group in T.ONE_OF.get(name, ()):
            for pn in group: props[pn]["description"] = "A new request needs " + " or ".join(group) + ". " + props[pn].get("description", "")
        if name not in T.NO_TURN: req.append("turn")
        out.append({"name": name, "description": T.func_meaning(name), "parameters": {"type": "object", "properties": props, "required": req}})
    return out
