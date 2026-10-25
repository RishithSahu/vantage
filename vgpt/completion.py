"""Constrained completion of a model-written query.

A small model asked to write a whole query will sometimes replace what the question already states: drop a condition, widen a specific sensor, swap the period for the default, turn a count into a list. This module takes the model's query as a proposal
and repairs it against what is known without the model:
  * LOCK - a field of the rule parser's partial parse (the draft) that the interpretation contract traces to words of the question (origin 'question') replaces the model's value for that field. Heuristic fields of the draft (defaults, inferred) are NOT locked:
    the rules make mistakes too, and the model may revise those.
  * OMISSION - a condition or event the question names (draft, origin 'question') but the model left out is added back.
  * ADDITION - a field of the model's query that nothing in the question or the previous turn supports (origin 'ungrounded') is replaced by the previous turn's value when the question reads as a follow-up and that turn had one; otherwise it is left in place for the guard to refuse.
The operation is never overwritten here (the model's job is to resolve it). Every change is returned as a note, so the effect of completion can be counted. The result is still passed through the interpretation guard.
"""
import copy

from . import interpretation as I


def _cond_key(c): return c.to_text()


def complete(text, model_q, draft_q=None, prev_q=None, now=None, tz=0):
    """-> (Query, [notes]). `model_q` is not modified."""
    q = copy.deepcopy(model_q); notes = []
    if draft_q is not None:
        di = I.interpret(text, draft_q, prev_q, source="rules", now=now, tz=tz); locked = {(f.name): f for f in di.fields if f.origin == "question"}
        if "target" in locked and draft_q.target and q.target != draft_q.target: notes.append(f"target {q.target}->{draft_q.target}"); q.target = draft_q.target
        if "target2" in locked and draft_q.target2 and q.target2 != draft_q.target2: notes.append(f"target2 {q.target2}->{draft_q.target2}"); q.target2 = draft_q.target2
        if "time" in locked and q.scope.time.to_text() != draft_q.scope.time.to_text() and q.op not in ("compare",): notes.append(f"time {q.scope.time.to_text()}->{draft_q.scope.time.to_text()}"); q.scope.time = copy.deepcopy(draft_q.scope.time)
        if "statistic" in locked and draft_q.agg and q.op in ("aggregate", "extreme") and q.agg != draft_q.agg: notes.append(f"statistic {q.agg}->{draft_q.agg}"); q.agg = draft_q.agg
        if "event" in locked and draft_q.event is not None and q.event is not None and q.event.type != draft_q.event.type: notes.append(f"event {q.event.type}->{draft_q.event.type}"); q.event.type = draft_q.event.type
        if draft_q.unit and not q.unit and any(f.name == "target" for f in di.fields): q.unit = draft_q.unit; notes.append(f"unit ->{q.unit}")
        have = {_cond_key(c) for c in q.scope.conditions}; named = {_cond_key(c) for c in q.scope.conditions} if False else set()
        cond_locked = {f.value for f in di.fields if f.name == "condition" and f.origin == "question"}
        for c in draft_q.scope.conditions:
            if _cond_key(c) in cond_locked and _cond_key(c) not in have: q.scope.conditions.append(copy.deepcopy(c)); notes.append(f"condition +{_cond_key(c)}")
        if draft_q.groups and not q.groups and q.op == "compare": q.groups = list(draft_q.groups); notes.append("groups from the question")
    mi = I.interpret(text, q, prev_q, source="model", now=now, tz=tz)
    if prev_q is not None:
        fu = {f.name: f for f in mi.fields}
        if "target" in fu and fu["target"].origin == "ungrounded" and prev_q.target: notes.append(f"target {q.target}->{prev_q.target} (previous turn)"); q.target = prev_q.target
        if "time" in fu and fu["time"].origin == "ungrounded": notes.append(f"time {q.scope.time.to_text()}->{prev_q.scope.time.to_text()} (previous turn)"); q.scope.time = copy.deepcopy(prev_q.scope.time)
        if "statistic" in fu and fu["statistic"].origin == "ungrounded" and prev_q.agg: notes.append(f"statistic {q.agg}->{prev_q.agg} (previous turn)"); q.agg = prev_q.agg
        bad = {f.value for f in mi.fields if f.name == "condition" and f.origin == "ungrounded"}
        if bad:
            q.scope.conditions = [c for c in q.scope.conditions if _cond_key(c) not in bad] + [copy.deepcopy(c) for c in prev_q.scope.conditions if _cond_key(c) not in {_cond_key(x) for x in q.scope.conditions}]
            notes.append("unsupported condition replaced by the previous turn's")
    return q, notes
