"""Scoring core of the evaluator (separate from the harness so that it can be tested with deliberately wrong predictions: tests/test_evaluator.py).

Outcome classes of a predicted query, per question (over the evaluation databases):
  answer           at least one database returns status 'answered'
  explained        no database answers, and every one explains why (missing_data / no_data / unsupported): a non-answer with a reason
  invalid          the query does not parse / validate / execute, or the answer failed verification (status unverified, invalid, exception)
  refusal          the planner declined (op=none)
Scoring:
  gold = op=none            correct iff the prediction is a refusal; anything else is a false action
  gold is an answer         correct iff the prediction's signature equals the gold signature on EVERY database (statuses, values, ALL items, event identity, requested semantics)
  gold is an explained non-answer
                            correct iff the statuses match AND the exact plan matches (operation, signal, event, period, ...): a matching status alone never establishes the right interpretation,
                            because "unsupported" for the wrong signal is a wrong answer to a different question
Reported separately: successful answers, explained non-answers, refusals, clarifications (clarification is its own outcome in the dialogue harness); exact-plan accuracy and execution accuracy are different columns."""
import json, os
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate

NON_ANSWER = ("missing_data", "no_data", "unsupported")
BAD = ("invalid", "unparsed", "exception", "unverified")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FixtureError(RuntimeError): pass


class Engines:
    """One query compiler per evaluation database. A missing database is an error, never a silent reduction of the experiment."""
    def __init__(self, dbs, root=None):
        from vgpt.brain import Brain
        root = root or os.path.join(HERE, "data"); missing = [n for n, _ in dbs if not os.path.exists(os.path.join(root, n + ".db"))]
        if missing: raise FixtureError(f"evaluation databases missing: {missing} (looked in {root})")
        self.qcs = [(n, Brain(os.path.join(root, n + ".db"), slm_ir=None, now=now).qc()) for n, now in dbs]; self.cache = {}

    def run(self, ir):
        """-> list of signatures, one per database."""
        if ir in self.cache: return self.cache[ir]
        out = []
        for name, qc in self.qcs:
            try: q = Query.from_text(ir); v = validate(q, qc.profile)
            except IRError: out.append({"status": "unparsed"}); continue
            if v.codes(): out.append({"status": "invalid"}); continue
            try: out.append(signature(qc.execute(q), q))
            except Exception as e: out.append({"status": "exception", "why": type(e).__name__})
        self.cache[ir] = out; return out


def _num(x, nd=3):
    return round(float(x), nd) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _ident(i):
    return [i.get("ts", i.get("start", i.get("day", i.get("group")))), i.get("type", i.get("event")), _num(i.get("value", i.get("wh_per_km")))]


def signature(r, q=None):
    """Logical result of an executed query: status, target/unit/period, value, ALL items (identity and value), event identity, requested semantics and the evidence qualifications."""
    s = {"status": r.status}; res = r.result
    if q is not None:                                                                                      # what was ASKED is part of the answer's identity (same number for another event or unit is not the same answer)
        s["asked"] = {"event": [q.event.type, q.event.selector, q.event.n, q.event.before, q.event.after] if q.event else None, "unit": q.unit or "", "target": [q.target, q.target2] if q.op != "correlation" else sorted([q.target or "", q.target2 or ""]),
                      "groups": list(q.groups), "agg": q.agg or ""}
    if res is None: return s
    s.update(op=res.op, target=res.target, unit=res.unit or "", period=[int(res.period[0]), int(res.period[1])] if res.period else None, when=res.when)
    v = res.value; s["value"] = _num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (str(v)[:80] if isinstance(v, str) else None)
    s["items"] = [_ident(i) for i in (res.items or []) if isinstance(i, dict)]; s["n"] = len(res.items or []); s["total"] = _num(res.value) if res.op in ("list_events", "count") else None
    x = res.extra or {}; ct = x.get("contract") or {}
    s["qual"] = {"class": ct.get("result_class"), "completeness": ct.get("completeness"), "resolution": ct.get("resolution"), "n_limits": len(ct.get("limits", []))}
    if res.op == "explain_change":
        o = x.get("obs", {}); cb = x.get("contrib") or {}
        s["explain"] = {"before": _num(o.get("before")), "after": _num(o.get("after")), "windows": x.get("windows"), "factors": [[f.get("name"), _num(f.get("contribution"), 1)] for f in cb.get("factors", [])], "residual": _num(cb.get("residual"), 1), "gap": _num(cb.get("gap"), 1), "open": sorted(str(a) for a in (x.get("open") or []))[:6]}
    if res.op == "detect_change":
        sh = x.get("shift") or {}; jp = x.get("jump") or {}; s["detect"] = {"shift_day": sh.get("day"), "jump_day": jp.get("day"), "before": _num(sh.get("before")), "after": _num(sh.get("after")), "shape": sh.get("shape"), "clear": sh.get("clear")}
    if res.op == "event_context": s["event"] = [res.target, res.when]
    return s


def _close(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False                                                            # missing versus numeric is a difference, never a pass
    return abs(a - b) <= 1e-3 + 1e-3 * abs(b)


def same(a, b):
    if a.get("status") != b.get("status"): return False
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if k == "value":
            if isinstance(va, (int, float)) or isinstance(vb, (int, float)) or va is None or vb is None:
                if not _close(va if isinstance(va, (int, float)) else None, vb if isinstance(vb, (int, float)) else None) and not (va is None and vb is None): return False
            elif va != vb: return False
        elif k == "items":
            va, vb = va or [], vb or []
            if len(va) != len(vb): return False
            for x, y in zip(va, vb):
                if x[0] != y[0] or x[1] != y[1] or not _close(x[2], y[2]): return False
        elif k == "explain":
            if (va is None) != (vb is None): return False
            if va is not None:
                if va["windows"] != vb["windows"] or not all(_close(va[f], vb[f]) for f in ("before", "after", "residual", "gap")): return False
                if len(va["factors"]) != len(vb["factors"]) or any(p[0] != q[0] or not _close(p[1], q[1]) for p, q in zip(va["factors"], vb["factors"])): return False
        elif k == "total":
            if not _close(va, vb): return False
        elif va != vb: return False
    return True


def parts(q):
    """The planned query, field by field, with ROLES preserved: ordered operands (targets, groups, the two scopes) are not sorted, so swapped operands differ. Correlation is symmetric and is the only operation whose targets are compared as a set."""
    d = q.to_dict(); sc = d.get("scope", {}); sb = d.get("scope_b") or {}
    def cond(s): return sorted(json.dumps(c, sort_keys=True) for c in (s or {}).get("conditions", []))
    tg = sorted([q.target or "", q.target2 or ""]) if q.op == "correlation" else [q.target or "", q.target2 or ""]
    return {"operation": q.op, "signal": tg, "statistic": q.agg or "", "period": json.dumps([sc.get("time"), sb.get("time")], sort_keys=True), "conditions": json.dumps([cond(sc), cond(sb)]),
            "event": json.dumps([q.event.type, q.event.selector, q.event.n, q.event.before, q.event.after] if q.event else None), "group/selection": json.dumps([list(q.groups), sc.get("select"), sb.get("select")], sort_keys=True), "unit": q.unit or ""}


def error_fields(gold, pred):
    gp, pp = parts(Query.from_text(gold)), parts(Query.from_text(pred)); return [k for k in gp if gp[k] != pp[k]]


def kind_of(sigs, refused):
    if refused: return "refusal"
    st = [s["status"] for s in sigs]
    if any(x in BAD for x in st): return "invalid" if not any(x == "answered" for x in st) else "answer"
    if any(x == "answered" for x in st): return "answer"
    return "explained" if all(x in NON_ANSWER for x in st) else "invalid"


def score_clarification(eng, gold, candidates):
    """The system asked a question instead of answering. Quality = does one of the offered readings contain the intended meaning (executes to the gold result), and how many readings were offered.
    Asking is NOT answering: `correct` stays False; `resolved_correct` is what an oracle user picking the right reading would end up with (after one extra turn)."""
    gnone = gold == "op=none"; hits = [] if gnone else [i for i, c in enumerate(candidates) if score(eng, gold, c)["correct"]]
    return {"kind": "clarify", "gold_kind": "refusal" if gnone else kind_of(eng.run(gold), False), "correct": False, "plan_exact": False, "wrong_answer": False, "false_action": False, "false_refusal": False, "fields": ["clarified"], "scored_by": "clarification",
            "clarify_n": len(candidates), "clarify_hit": bool(hits), "clarify_rank": (hits[0] + 1) if hits else None, "unneeded_clarification": gnone, "resolved_correct": bool(hits), "extra_turns": 1}


def score(eng, gold, pred, candidates=None):
    """-> dict with kind (of the prediction), gold_kind, correct, plan_exact, wrong_answer, false_action, false_refusal, fields, scored_by."""
    if candidates: return score_clarification(eng, gold, candidates)
    gnone = gold == "op=none"; pnone = pred == "op=none"
    try: pq = None if pnone else Query.from_text(pred)
    except IRError: pq = None; pnone = True
    sp = None if pnone else eng.run(pred); sg = None if gnone else eng.run(gold); rec = {"kind": kind_of(sp or [], pnone)}
    if gnone:
        rec.update(gold_kind="refusal", correct=pnone, plan_exact=pnone, wrong_answer=False, false_action=not pnone, false_refusal=False, fields=[], scored_by="refusal"); return rec
    gk = kind_of(sg, False); rec["gold_kind"] = gk
    gq = Query.from_text(gold); rec["plan_exact"] = bool(pq is not None and parts(pq) == parts(gq))
    if pnone or pq is None: rec.update(correct=False, wrong_answer=False, false_action=False, false_refusal=True, fields=["refused"], scored_by="refusal"); return rec
    if gk == "explained" or gk == "invalid":
        ok = bool(rec["plan_exact"] and [s["status"] for s in sp] == [s["status"] for s in sg]); by = "plan+status"
    else:
        ok = len(sp) == len(sg) and all(same(a, b) for a, b in zip(sp, sg)); by = "execution"
    rec.update(correct=ok, wrong_answer=rec["kind"] == "answer" and not ok, false_action=False, false_refusal=False, fields=[] if ok else (error_fields(gold, pred) or ["result"]), scored_by=by)
    rec["wrong_non_answer"] = rec["kind"] == "explained" and not ok                                           # explained a limitation that gold does not have (or for the wrong thing)
    return rec
