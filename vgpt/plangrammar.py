"""GBNF grammar of the plan call, generated from the same definition as the tool schemas (vgpt/toolspec.py + vgpt/schema.py).

It enforces SYNTAX and TYPES only: a supported function, the typed argument forms (a period is an amount and a unit or a named calendar period or a date; a threshold is {signal, op, value, unit}; a selection is {entity, metric, order, k, class}; a duration is an amount and a unit), registered identifiers (every signal, event, condition, tool of the registry - including a known concept this vehicle does not record, so the model can name it and the answer can say it is not recorded) and the explicit state-edit parameters.
It is built once per registry. It does NOT depend on the question: there is no word-presence gate (a period, a number, a unit or a removal is available whatever the question says), and no shortlist of functions chosen by similarity to examples. Whether the model chose the RIGHT identifier is the model's job; whether the plan is meaningful is checked by plancompile.validate.

A call is one of two shapes: `turn="new"` with the function's required arguments, or a turn that depends on the conversation (follow_up / correction / clarification_reply) where every argument is optional because it states only what changes."""
from . import schema as S
from . import toolspec as T

CALENDAR = T.CALENDAR


def _alt(items): return " | ".join('"' + str(i).replace("\\", "").replace('"', "") + '"' for i in items)


def _q(items): return '"\\"" (' + _alt(items) + ') "\\""'


def value_rules(registry, max_text=160):
    """{type: GBNF body of the VALUE}: shared by every style (the wrapper differs, the typed values do not)."""
    E = S.enums(registry); v = {}
    v["signal"] = _q(E["signals"]); v["event"] = _q(E["events"]); v["stat"] = _q(T.STATS); v["cstat"] = _q(T.CSTATS); v["minmax"] = _q(T.MINMAX); v["evsel"] = _q(T.EVSEL); v["unit"] = _q(E["units"])
    v["turn"] = _q(T.TURNS); v["cedit"] = _q(T.CEDITS); v["tool"] = _q(E["tools"]); v["focus"] = _q(sorted(T.TOOLS_FOCI)); v["decl"] = _q(T.DECLINES); v["saykind"] = _q(T.SAYS)
    v["entity"] = _q(T.ENTITIES); v["metric"] = _q(E["metrics"]); v["order"] = _q(T.ORDERS); v["tclass"] = _q(E["classes"])
    v["conds"] = '"[" (cond-name | thr) ("," " "? (cond-name | thr))* "]"'; v["grps"] = '"[" cond-name ("," " "? cond-name)* "]"'
    v["dropl"] = '"[" ' + _q(T.DROPS) + ' ("," " "? ' + _q(T.DROPS) + ')* "]"'
    v["sel"] = '"{\\"entity\\": " val-entity ( ", \\"metric\\": " val-metric )? ( ", \\"order\\": " val-order )? ( ", \\"k\\": " val-int )? ( ", \\"class\\": " val-tclass )? "}"'
    v["period"] = ('"\\"" ("now" | "all time" | "last week" | "last month" | "last " [1-9] [0-9]? [0-9]? " " ("minute" | "hour" | "day" | "week" | "month") "s"? | ' + _alt(CALENDAR) + ' | "on " [0-9] [0-9] [0-9] [0-9] "-" [0-9] [0-9] "-" [0-9] [0-9]) "\\""')
    v["dur"] = '"\\"" [1-9] [0-9]? [0-9]? " " ("second" | "minute" | "hour") "s"? "\\""'
    v["int"] = "[1-9] [0-9]? [0-9]?"
    v["span"] = '"\\"" [^"\\\\\\n]{1,80} "\\""'
    v["text"] = '"\\"" [^"\\\\\\n]{1,%d} "\\""' % max_text
    return v, E


def common_rules(registry, max_text=160):
    v, E = value_rules(registry, max_text)
    g = ['num ::= "-"? [0-9]+ ("." [0-9]+)?', "cond-name ::= " + _q(E["conditions"]),
         'thr ::= "{\\"signal\\": " val-signal ", \\"op\\": " ' + _q(T.CMPOPS) + ' ", \\"value\\": " num ( ", \\"unit\\": " val-unit )? "}"']
    g += [f"val-{t} ::= {body}" for t, body in v.items()]
    return g


def build_frame(registry, max_text=160, with_state=True):
    """The frame format: `operation | turn="new" | field=value | field=none ...` - every field of the operation is written, an absent one explicitly as `none` (a follow-up writes `keep` for a field that stays). The typed values are the same rules as the tool-call styles."""
    g = common_rules(registry, max_text); funcs = []
    def fld(pn, pt, mode):                                                                                  # mode: req (a value), new (value | none), edit (value | keep | none)
        alts = {"req": f"val-{pt}", "new": f'(val-{pt} | "none")', "edit": f'(val-{pt} | "keep" | "none")'}[mode]
        return f'" | {pn}=" {alts}'
    for name in T.FUNCS:
        rn = "f-" + name.replace("_", "-"); fields = T.frame_fields(name); req = {pn for pn, pt, r in T.func_params(name) if r}
        if name in T.NO_TURN: body = f'"{name}"' + "".join(fld(pn, pt, "req") for pn, pt in fields)
        else:
            new = '" | turn=\\"new\\""' + "".join(fld(pn, pt, "req" if pn in req else "new") for pn, pt in fields if pn != "conditions_edit") + ('" | conditions_edit=" (val-cedit | "none")' if any(pn == "conditions_edit" for pn, _ in fields) else "")
            follow = '" | turn=" ' + _q(T.TURNS[1:]) + "".join(fld(pn, pt, "edit") for pn, pt in fields)
            body = f'"{name}" ( {new} | {follow} )' if with_state else f'"{name}" {new}'
        g.append(f"{rn} ::= {body}"); funcs.append(rn)
    g.insert(0, f'root ::= ({" | ".join(funcs)})')
    return "\n".join(g)


def build_intent(registry, max_text=160, with_state=True, only=None):
    """The intent surface: `result=<what is wanted> | turn="new" | field=value | field=none ...` - the model states the RESULT (a closed, meaningful vocabulary) and the fields that apply to it; the executor is derived by typed routing from the result and the kind of the target. Absent fields are explicit `none`; a follow-up writes `keep` for a field that stays."""
    g = common_rules(registry, max_text); funcs = []; ty = {"target_signal": "signal", "target_event": "event", "target": "tgt"}
    g.append("val-tgt ::= val-signal | val-event")
    def fld(sn, st, mode):
        t = ty.get(st, st); alts = {"req": f"val-{t}", "new": f'(val-{t} | "none")', "edit": f'(val-{t} | "keep" | "none")'}[mode]
        return f'" | {sn}=" {alts}'
    for result in T.INTENT_ORDER:
        rn = "r-" + result.replace("_", "-"); fields = T.surface_fields(result); req = T.surface_required(result); meta = result in ("ask", "unresolved", "chat", "refuse")
        if meta: body = f'"result={result}"' + "".join(fld(sn, st, "req") for sn, st in fields)
        else:
            new = '" | turn=\\"new\\""' + "".join(fld(sn, st, "req" if sn in req else "new") for sn, st in fields if sn != "conditions_edit") + ('" | conditions_edit=" (val-cedit | "none")' if any(sn == "conditions_edit" for sn, _ in fields) else "")
            follow = '" | turn=" ' + _q(T.TURNS[1:]) + "".join(fld(sn, st, "edit") for sn, st in fields)
            body = f'"result={result}" ( {new} | {follow} )' if with_state else f'"result={result}" {new}'
        g.append(f"{rn} ::= {body}"); funcs.append(rn)
    g.insert(0, f'root ::= ({" | ".join(funcs)})' if only is None else f'root ::= r-{only.replace("_", "-")}')       # only=<result>: the result was decided by an earlier stage; the fields that belong to it are generated
    return "\n".join(g)


def build(registry, max_text=160, with_state=True, style="py", grounded=True, only=None, exclude=()):
    """style="py": `fn(arg=value, ...)` (the tool-call text of LFM2.x, whose assistant turn is opened with `<|tool_call_start|>[`); style="json": `{"name": "fn", "arguments": {"arg": value, ...}}` (the tool-call text of Granite 4 and Qwen2.5, opened with `<tool_call>`). The typed values are identical; only the wrapper differs.
    with_state=False: there is no previous plan and no pending question, so a call cannot refer to one: only the `new` shape is admitted (a state reference must refer to something)."""
    if style == "frame": return build_frame(registry, max_text, with_state)
    if style == "intent": return build_intent(registry, max_text, with_state)
    g = common_rules(registry, max_text); funcs = []; arr = style == "jsonarray"; js = style in ("json", "jsonarray"); exclude = set(exclude or ())
    def pr(pn, pt): return ('"\\"%s\\": " val-%s' % (pn, pt)) if js else f'"{pn}=" val-{pt}'
    turn_new = '"\\"turn\\": \\"new\\""' if js else '"turn=\\"new\\""'
    turn_follow = '"\\"turn\\": " ' if js else '"turn=" '
    for name in T.FUNCS:
        params = T.func_params(name); rn = "f-" + name.replace("_", "-")
        head = ('"{\\"name\\": \\"%s\\", \\"arguments\\": {"' % name) if js else f'"{name}("'; tail = '"}"' if js else '")"'
        if name in T.NO_TURN:
            body = f'{head} ' + pr(params[0][0], params[0][1]) + f' {tail}'
        else:
            def item(pn, pt, req):                                                                    # one argument group: ", name=value" - a grounded one is preceded by its quoted words, as an inseparable pair
                if pn in exclude and not req: return ""
                core = (pr(T.words_name(pn), "span") + ' ", " ' + pr(pn, pt)) if (grounded and pn in T.GROUNDED) else pr(pn, pt)
                return (' ", " ' + core) if req else (' ( ", " ' + core + " )?")
            new = turn_new + "".join(item(pn, pt, req) for pn, pt, req in params)
            follow = turn_follow + _q(T.TURNS[1:]) + "".join(item(pn, pt, False) for pn, pt, _ in params) + (' ( ", " ' + pr("conditions_edit", "cedit") + ' )?' if any(n == "conditions" for n, _, _ in params) else "") + ' ( ", " ' + pr("drop", "dropl") + " )?"
            body = f'{head} ( {new} | {follow} ) {tail}' if with_state else f'{head} {new} {tail}'
        if js: body += ' "}"'
        g.append(f"{rn} ::= {body}"); funcs.append(rn)
    chosen = ["f-" + n.replace("_", "-") for n in ([only] if isinstance(only, str) else (only or T.FUNCS))]
    g.insert(0, (f'root ::= "[" ({" | ".join(chosen)}) "]"' if arr else f'root ::= ({" | ".join(chosen)})' + ("" if js else ' "]"')))                                  # the assistant turn is opened with the template's own call marker (written into the prompt); the grammar covers the call itself
    return "\n".join(g)
