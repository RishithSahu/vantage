"""Conversation state as explicit updates.

A follow-up is an UPDATE to the previous accepted interpretation, not a fresh question. `diff_queries(prev, new)` states exactly what changed and what was inherited:
  replace_period / replace_secondary_period / replace_target / replace_position / change_statistic / change_operation / change_event / change_unit
  add_condition / remove_condition / replace_condition
  inherited: every field of the new query that is identical to the previous one (and therefore came from the previous turn unless the utterance also states it)
`Trace` records, per turn, the utterance, the resulting query, the outcome (answered / clarify / refused / limitation), the updates and the inherited fields, so that a complete conversation shows precisely what changed and what stayed. The state used is the system's OWN accepted
state (a clarification or a refusal does not replace it), never gold history."""
import re
from dataclasses import dataclass, field

POS = re.compile(r"_(fl|fr|rl|rr)$")


def _conds(q, scope="scope"):
    sc = getattr(q, scope)
    return sorted((c.to_text() for c in sc.conditions), key=str) if sc is not None else []


def diff_queries(prev, new):
    """-> (updates, inherited) between two Query objects (prev may be None: a new topic)."""
    if prev is None: return [{"kind": "new_question"}], []
    ups = []; inh = []
    def cmp(name, a, b, kind):
        if a != b: ups.append({"kind": kind, "from": a, "to": b})
        else: inh.append(name)
    if new.op != prev.op: ups.append({"kind": "change_operation", "from": prev.op, "to": new.op})
    else: inh.append("operation")
    pa, pb = (prev.target or ""), (new.target or "")
    if pa != pb:
        if POS.sub("", pa) == POS.sub("", pb) and pa and pb: ups.append({"kind": "replace_position", "from": pa, "to": pb})
        else: ups.append({"kind": "replace_target", "from": pa, "to": pb})
    else: inh.append("target")
    cmp("target2", prev.target2 or "", new.target2 or "", "replace_target")
    cmp("statistic", prev.agg or "", new.agg or "", "change_statistic")
    cmp("period", prev.scope.time.to_text(), new.scope.time.to_text(), "replace_period")
    if (prev.scope_b is not None) or (new.scope_b is not None): cmp("secondary_period", prev.scope_b.time.to_text() if prev.scope_b else "", new.scope_b.time.to_text() if new.scope_b else "", "replace_secondary_period")
    a, b = _conds(prev), _conds(new); removed = [c for c in a if c not in b]; added = [c for c in b if c not in a]
    if removed and added: ups.append({"kind": "replace_condition", "from": removed, "to": added})
    else:
        for c in removed: ups.append({"kind": "remove_condition", "condition": c})
        for c in added: ups.append({"kind": "add_condition", "condition": c})
    inh += [f"condition {c}" for c in a if c in b]
    ea, eb = (prev.event.type if prev.event else ""), (new.event.type if new.event else "")
    cmp("event", ea, eb, "change_event")
    if (prev.unit or "") != (new.unit or ""): ups.append({"kind": "change_unit", "from": prev.unit or "", "to": new.unit or ""})
    if sorted(prev.groups) != sorted(new.groups) or list(prev.groups) != list(new.groups): ups.append({"kind": "change_groups", "from": list(prev.groups), "to": list(new.groups)})
    return ups, inh


@dataclass
class TurnRecord:
    n: int
    text: str
    outcome: str                       # answered | clarify | refused | limitation | no_data
    ir: str = ""
    updates: list = field(default_factory=list)
    inherited: list = field(default_factory=list)
    note: str = ""

    def line(self):
        ch = "; ".join(f"{u['kind']}" + (f" {u.get('from')}->{u.get('to')}" if "from" in u else (f" {u.get('condition')}" if "condition" in u else "")) for u in self.updates) or "-"
        return f"{self.n}. {self.text!r}: {self.outcome}  {self.ir or '-'}\n     changed: {ch}\n     inherited: {', '.join(self.inherited) or '-'}" + (f"\n     note: {self.note}" if self.note else "")


class Trace:
    """The accepted state and the log of what each turn did to it."""
    def __init__(self): self.accepted = None; self.records = []

    def record(self, text, query, outcome, note=""):
        """Update the trace. Only an ANSWERED (or explained-limitation) turn becomes the accepted state; clarifications and refusals leave it untouched."""
        ups, inh = (diff_queries(self.accepted, query) if query is not None else ([], []))
        rec = TurnRecord(len(self.records) + 1, text, outcome, query.to_text() if query is not None else "", ups if outcome in ("answered", "limitation", "no_data") else [], inh if outcome in ("answered", "limitation", "no_data") else [], note)
        self.records.append(rec)
        if query is not None and outcome in ("answered", "limitation", "no_data"): self.accepted = query
        return rec

    def describe(self): return "\n".join(r.line() for r in self.records)
