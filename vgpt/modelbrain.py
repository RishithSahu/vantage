"""ModelBrain: the product. ONE route for understanding a question: the frozen language model, supported by retrieval of capability descriptions.

    question + conversation -> capability retrieval -> ONE frozen-model tool call -> structural validation -> execution (typed pipeline) -> evidence-backed answer

There is no rule parser, no rule-generated alternative and no lexical check anywhere on this path (nothing in nl2ir / nlu / policy / interpretation is called). If the model is unavailable or cannot form a valid plan the outcome is an explicit failure or a specific question; the route is never changed silently. Deterministic parts stay deterministic: schema validation, unit conversion, database execution (read-only), arithmetic, the evidence contract and the answer verifier.
The model never writes numbers or answers. It writes a plan; the only free text it can produce is the one question of ask_user (and the name of a missing quantity), both shown as the model's, never as data.

Outcomes (facts["outcome"]): answered | no_data | missing_data | unsupported_calculation | invalid_plan | ambiguous | unresolved_target | refused | chat | failed_plan | tool | execution_error.
State changes only on an answered turn (and a question that is open is remembered as pending, with the plan so far); a refusal or failure never erases what the person was asking about. A plan that fails validation is never executed, and showing the person a question about it does not make it executable."""
import re
import time

import json

from . import plancompile as PC
from .brain import Brain, compose as legacy_compose
from .modelplan import Planner, State
from .paraphrase import describe
from . import toolspec as T
from .toolspec import plan_to_fn_args

REPLY_FAILED = "I could not form a valid reading of that, so I have not guessed. Could you say it another way - which signal or event, and which period?"
HELP = "I answer questions about this vehicle's recorded data: values, statistics, events and warnings, comparisons, changes and the associations behind them, service, tyres, weather and driving summaries. I only read data and cannot operate the vehicle."
CANNED = {"control": "That is a request to change or control something. I only read recorded vehicle data and cannot operate the vehicle, so I have not done anything.",
          "offtopic": "That is not a question about this vehicle's recorded data, so I will leave it.",
          "greeting": "Hello! Ask me about your car - for example its state of charge, recent warnings, or how its consumption has changed.",
          "thanks": "You're welcome! Ask me anything else about the car.", "help": HELP}                    # the model chooses a KIND of reply from a closed list; it never writes the reply text
TOOL_INTENT = {"service": "SERVICE", "tyres": "TYRES", "weather": "WEATHER", "driving": "DRIVING", "health": "HEALTH", "places": "PLACES"}
STATUS_OUTCOME = {"answered": "answered", "no_data": "no_data", "missing_data": "missing_data", "unsupported": "unsupported_calculation", "invalid": "invalid_plan", "unverified": "unverified"}


def _clean_text(s, n=300):
    s = re.sub(r"<[^>]{0,300}>", " ", str(s or "")); s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", s)
    return " ".join(s.split())[:n]


def render_existence(text, r, what):
    """Existence answers: 'Yes' only on a valid positive observation. A zero count is 'No recorded occurrence' only when the data contract says the period's records are complete and the recorder's coverage is known; otherwise 'Cannot establish absence', saying why. Never a bare 'No'."""
    n = int(round(r.result.value or 0)); k = r.result.extra.get("contract", {}) or {}
    if n > 0: return "Yes. " + text
    complete = k.get("completeness", "complete") == "complete" and k.get("recorded_share") is not None
    if complete: return f"No recorded occurrence of {what} in that period (the records for it are complete). " + text
    why = []
    if k.get("coverage") is not None and k["coverage"] < 1: why.append(f"only {k['coverage'] * 100:.0f}% of the period is on record")
    if k.get("recorded_share") is not None and k["recorded_share"] < 1: why.append(f"about {(1 - k['recorded_share']) * 100:.0f}% of the active time was not recorded")
    if k.get("recorded_share") is None: why.append("the recorder's coverage of that period could not be checked")
    if k.get("completeness", "complete") != "complete" and not why: why.append("the records are " + str(k.get("completeness")))
    return f"Cannot establish absence: no occurrence of {what} is recorded, but " + "; ".join(why) + ". " + text


class ModelBrain(Brain):
    def __init__(self, db_path, planner=None, persona="driver", now=None, repair=False, expand=False, support_authority=False, **kw):
        super().__init__(db_path, slm=None, persona=persona, now=now, slm_ir=None, policy="clarify", selector=None, qc_mode="all")
        self._P = planner; self.repair = None; self.recovery = None; self.expansion = None; self.support_diag = None; self.use_repair = repair; self.use_expand = expand; self.support_authority = support_authority; self.state = State(); self.last_plan_result = None; self.last_outcome = None; self.outcome = None; self.timing = {}

    def planner(self):
        if self._P is None: self._P = Planner(self.c)
        return self._P

    def reset(self): self.state = State(); self.trace.__init__(); self.qcs = type(self.qcs)(); self.last_qc = None; self.last_plan_result = None; self.last_outcome = None; self.outcome = None; self.recovery = None

    @staticmethod
    def _echoed_question(text, plan):
        """Whether ask_user merely repeats the request instead of clarifying it."""
        if not plan or plan.get("fn") != "ask_user": return False
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
        return bool(norm(text)) and norm(text) == norm(plan.get("ask"))

    # ------------------------------------------------------------------------------------------------
    def ask(self, text, use_slm=True):
        text = _clean_text(text, 600); t0 = time.perf_counter(); self._text = text; none_p = {"metric": None, "event": None, "w": "a"}; self.last_qc = None; self.outcome = None
        if not text: return self._finish(t0, "Ask me something about the car.", "NONE", none_p, "empty", "chat", {"ok": True}, {})
        err = ""; self.recovery = None
        try:
            P = self.planner()
            pr = P.plan(text, self.state, self.now, self.tz)
            # A pending clarification is context, not ownership of every later
            # message.  If it makes the model merely ask the complete new
            # request back, try exactly once with clean conversation state.
            # The same recovery also catches a context-free echoed question.
            if pr and not pr.error and self._echoed_question(text, pr.plan):
                old_raw = pr.raw
                clean = State() if self.state.pending else self.state
                pr2 = P.plan(text, clean, self.now, self.tz, feedback=["Interpret the complete request. Do not repeat it as a clarification question; ask only for a genuinely missing required value."])
                self.recovery = {"first_call": old_raw, "second_call": pr2.raw if pr2 else None, "ignored_pending": bool(self.state.pending)}
                if pr2 is not None and not pr2.error and pr2.plan is not None and not self._echoed_question(text, pr2.plan): pr = pr2
        except Exception as e: pr = None; err = f"{type(e).__name__}"
        self.last_plan_result = pr
        return self._after_plan(t0, text, pr, err, none_p)

    def propose(self, text, follow_up=True):
        """ONLY the language model's proposal for this message and the conversation state: nothing is compiled, executed or remembered (the interpretation preview shows it and the person may edit it). follow_up=False: the PERSON says this is a new topic, so the model sees no earlier request (a small model otherwise copies it); the confirmed query stays remembered for a later follow-up."""
        return self.planner().plan(_clean_text(text, 600), self.state if follow_up else State(), self.now, self.tz)

    def ask_confirmed(self, text, fn, args):
        """Execute an interpretation that the PERSON confirmed (possibly after editing it): the typed call goes through the same validation, compilation and execution as a model-written one - nothing is guessed, defaulted or repaired - and the model is not consulted. The call is a complete one (turn new): the person confirmed a resolved query, not a delta."""
        from .modelplan import PlanResult
        text = _clean_text(text, 600); t0 = time.perf_counter(); self._text = text; none_p = {"metric": None, "event": None, "w": "a"}; self.last_qc = None; self.outcome = None
        try: P = self.planner()
        except Exception as e:
            return self._finish(t0, REPLY_FAILED, "CLARIFY", none_p, "plan", "failed_plan", {"ok": False, "why": type(e).__name__, "failed": True}, {})
        try:
            T.check_args(fn, args); plan = T.call_to_plan(fn, args)
        except T.CallError as e:
            return self._finish(t0, "That interpretation is not valid (" + _clean_text(str(e), 200) + "), so I have not run anything.", "CLARIFY", none_p, "plan", "invalid_plan", {"ok": False, "why": str(e), "failed": True}, {}, None)
        pr = PlanResult(raw=P.dialect.assistant_text(fn, args), fn=fn, args=args, plan=plan, message=text, timing={}); pr.confirmed = True; self.last_plan_result = pr
        return self._after_plan(t0, text, pr, "", none_p)

    def _after_plan(self, t0, text, pr, err, none_p):
        if pr is None or pr.error or pr.plan is None:
            why = err if pr is None else pr.error; self.trace.record(text, None, "failed", why)
            return self._finish(t0, REPLY_FAILED, "CLARIFY", none_p, "plan", "failed_plan", {"ok": False, "why": why, "failed": True}, pr.timing if pr else {}, pr)
        try: P = self.planner()
        except Exception as e:
            why = type(e).__name__; self.trace.record(text, None, "failed", why)
            return self._finish(t0, REPLY_FAILED, "CLARIFY", none_p, "plan", "failed_plan", {"ok": False, "why": why, "failed": True}, pr.timing, pr)
        msg_g = text if (getattr(P.dialect, "grounded", True) and not getattr(pr, "confirmed", False)) else None
        reply = bool(self.state.pending and pr.plan.get("turn") == "clarification_reply"); orig = self.state.pending["original"] if reply else text
        support = getattr(P, "support", None) if self.support_authority else None                       # the embedding support check is DIAGNOSTIC ONLY unless explicitly enabled: it never changes, certifies or rejects a plan
        self.support_diag = self._support_diag(P, pr.plan, text)
        tv = time.perf_counter(); out = PC.compile_plan(pr.plan, self.state.prev_plan, self.state.pending, P.reg, self.now, self.tz, message=msg_g, original=orig if reply else None, support=support); self.last_outcome = out
        self.repair = None
        if out.kind == "ungrounded" and self.use_repair:                                                    # a proposed argument could not be grounded: ONE repair pass - the model is told which quotation failed and rewrites its call (it may quote exactly or leave the argument out itself)
            fb = [f"your {f['label']} words “{f['words']}” are not usable ({f['why']}); copy the exact words from the message, or leave the {f['label']} out." for f in out.failures]
            failed = {f["key"] for f in out.failures}; exclude = [pn for pn, key in T.GROUNDED.items() if key in failed]
            try: pr2 = P.plan(text, self.state, self.now, self.tz, feedback=fb, exclude_fields=exclude)
            except Exception: pr2 = None
            self.repair = {"first_call": pr.raw, "failures": out.failures, "second_call": pr2.raw if pr2 else None}
            if pr2 is not None and not pr2.error and pr2.plan is not None:
                tm1 = pr.timing; pr = pr2; self.last_plan_result = pr
                reply = bool(self.state.pending and pr.plan.get("turn") == "clarification_reply"); orig = self.state.pending["original"] if reply else text
                out = PC.compile_plan(pr.plan, self.state.prev_plan, self.state.pending, P.reg, self.now, self.tz, message=msg_g, original=orig if reply else None, support=support); self.last_outcome = out
                pr.timing = dict(pr.timing, repair_ms=pr.timing.get("plan_ms"), first_plan_ms=tm1.get("plan_ms"))
        # A reply may actually be a complete new request.  Conversation state
        # must not turn one failed clarification into a chain of failures.  If
        # the stateful interpretation still cannot compile, make one clean-state
        # proposal and adopt it only when it is independently executable.  This
        # is a structural recovery rule, not a phrase or intent rule.
        if self.state.pending and out.kind in ("ungrounded", "ambiguous", "invalid"):
            try: pr3 = P.plan(text, State(), self.now, self.tz, feedback=["Treat this message as a standalone request. Preserve only meaning stated in this message."])
            except Exception: pr3 = None
            if pr3 is not None and not pr3.error and pr3.plan is not None:
                out3 = PC.compile_plan(pr3.plan, None, None, P.reg, self.now, self.tz, message=msg_g, support=support)
                if out3.kind in ("query", "tool", "refuse", "chat"):
                    earlier = self.recovery or {}; self.recovery = dict(earlier, stateful_call=pr.raw, standalone_call=pr3.raw, ignored_pending=True)
                    pr, out, orig, reply = pr3, out3, text, False; self.last_plan_result = pr; self.last_outcome = out
        tm = dict(pr.timing, compile_ms=round((time.perf_counter() - tv) * 1000, 2))
        k = out.kind
        if k == "ungrounded":                                                                              # still ungrounded after the repair: a specific question; the failed fields stay in the record, nothing broader is executed
            q = _clean_text(out.say, 300); self.state.pending = {"ask": q, "original": orig, "plan": out.pending_plan}; self.state.history = [(pr.message, self._turn("ask_user", {"question": q}))]; self.trace.record(text, None, "clarify", q)
            return self._finish(t0, q, "CLARIFY", none_p, "plan", "ambiguous", {"ok": True, "clarify": True, "plan": pr.plan, "ungrounded": out.failures, "proposed": out.proposed}, tm, pr)
        if k == "ambiguous":
            q = _clean_text(out.say, 240) or "Which signal or event do you mean, and for which period?"
            self.state.pending = {"ask": q, "original": orig, "plan": None}; self.state.history = [(pr.message, self._turn("ask_user", {"question": q}))]; self.trace.record(text, None, "clarify", q)
            return self._finish(t0, q, "CLARIFY", none_p, "plan", "ambiguous", {"ok": True, "clarify": True, "plan": pr.plan}, tm, pr)
        if k == "invalid" and out.say:                                                                    # a required argument is missing: ask for it, remembering the plan so far
            self.state.pending = {"ask": out.say, "original": orig, "plan": out.pending_plan}; self.state.history = [(pr.message, self._turn("ask_user", {"question": out.say}))]; self.trace.record(text, None, "clarify", out.say)
            return self._finish(t0, out.say, "CLARIFY", none_p, "plan", "ambiguous", {"ok": True, "clarify": True, "plan": pr.plan, "missing": out.reason}, tm, pr)
        if k == "invalid":
            self.trace.record(text, None, "failed", out.reason)
            return self._finish(t0, "I could not turn that into a valid request (" + _clean_text(out.reason, 200) + "), so I have not run anything. Could you say it another way?", "CLARIFY", none_p, "plan", "invalid_plan", {"ok": False, "why": out.reason, "failed": True, "issues": out.issues, "plan": pr.plan}, tm, pr)
        if k == "unsupported":
            self.trace.record(text, None, "limitation", out.reason)
            return self._finish(t0, "I can't compute that as asked: " + _clean_text(out.reason, 300) + (". " if not out.reason.endswith(".") else " ") + "I have not substituted anything else.", "NONE", none_p, "plan", "unsupported_calculation", {"ok": False, "why": out.reason, "plan": pr.plan}, tm, pr)
        if k == "unresolved" and self.use_expand:                                                          # ONE bounded expansion before concluding: search again with what the model named, relaxed floor; re-plan only if that finds capabilities that were not shown
            added = P.expand_ids(out.what, text, pr.card_ids) if hasattr(P, "expand_ids") else []
            self.expansion = {"what": out.what, "added": added, "second_call": None}
            if added:
                try: pr2 = P.plan(text, self.state, self.now, self.tz, extra_ids=added)
                except Exception: pr2 = None
                if pr2 is not None and not pr2.error and pr2.plan is not None:
                    self.expansion["second_call"] = pr2.raw; pr = pr2; self.last_plan_result = pr; tm = dict(pr.timing, compile_ms=tm.get("compile_ms"))
                    out = PC.compile_plan(pr.plan, self.state.prev_plan, self.state.pending, P.reg, self.now, self.tz, message=msg_g, support=support); self.last_outcome = out; k = out.kind
        if k == "unresolved":
            what = _clean_text(out.what, 120) or "that"; by = P.reg.by_id
            known_missing = [i for i in pr.card_ids if by[i].kind == "signal" and not by[i].available][:3]; near = [i for i in pr.card_ids if by[i].kind == "signal" and by[i].available][:3]
            if known_missing: msg = f"I know of {', '.join(known_missing)}, but this vehicle does not record it, and I will not answer with a different quantity."
            else: msg = f"I could not match \u201c{what}\u201d to any quantity I know about, so I cannot tell you whether this vehicle records it; I have not substituted another quantity."
            if near: msg += " Nearby quantities it does record: " + ", ".join(near) + " - ask about one of those by name if that is what you meant."
            self.trace.record(text, None, "limitation", msg)
            return self._finish(t0, msg, "NONE", none_p, "plan", "unresolved_target", {"ok": False, "unresolved": what, "plan": pr.plan, "known_but_not_recorded": known_missing, "expansion": self.expansion}, tm, pr)
        if k == "refuse":
            msg = CANNED.get(out.say, CANNED["offtopic"]); self.trace.record(text, None, "refused", msg)
            return self._finish(t0, msg, "SAFETY", none_p, "gate", "refused", {"ok": False, "refused": True, "plan": pr.plan}, tm, pr)
        if k == "chat": return self._finish(t0, CANNED.get(out.say, HELP), "SMALLTALK", none_p, "plan", "chat", {"ok": True}, tm, pr)
        if k == "tool": return self._tool(text, out, t0, tm, pr)
        return self._query(text, out, t0, tm, pr)

    @staticmethod
    def _support_diag(P, plan, text):
        """Diagnostic record of what the embedding support check WOULD say about each quoted argument (never acted on)."""
        sc = getattr(P, "support", None)
        if sc is None or not plan: return None
        from . import toolspec as T
        out = {}
        for pn, key in T.GROUNDED.items():
            if plan.get(key) in (None, "", [], {}): continue
            w = PC._norm_words((plan.get("src") or {}).get(key))
            if w and w in PC._norm_words(text): out[key] = sc(key, w, plan[key])
        return out or None

    def _remember(self, plan, pr):
        """An answered turn becomes the conversation state: the resolved plan, and the turn itself (message + the plan as a call) shown to the model as an earlier chat turn."""
        self.state.prev_plan = plan; fa = plan_to_fn_args(plan)
        self.state.history = [(pr.message, self._turn(*fa))] if fa else []

    def _turn(self, fn, args): return self.planner().dialect.assistant_text(fn, args)

    def _finish(self, t0, answer, intent, p, path, outcome, facts, timing, pr=None):
        self.outcome = outcome; timing = dict(timing or {}); timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 1); self.timing = timing
        return self._result(t0, answer, intent, p, path, 0.9 if outcome in ("answered", "tool") else 0.5, dict(facts, outcome=outcome, call=(pr.raw if pr else None), retrieved=(pr.card_ids if pr else None)), None,
                            {"ms": timing.get("generate_ms"), "tokens": timing.get("out_tokens"), "retrieve_ms": timing.get("retrieve_ms"), "timing": timing})

    def _query(self, text, out, t0, tm, pr):
        q = out.query; t1 = time.perf_counter()
        try: r = self.qc().execute(q)
        except Exception as e:                                                                              # the engine must never take the conversation down: report, keep the state
            self.trace.record(text, q, "failed", type(e).__name__)
            return self._finish(t0, "I understood the question but could not compute it safely (" + type(e).__name__ + "), so I am not stating a number.", "NONE", {"metric": q.target, "event": None, "w": "a"}, "qc", "execution_error", {"ok": False, "failed": True, "why": type(e).__name__}, tm, pr)
        t2 = time.perf_counter(); self.last_qc = r; a = r.answer; label = STATUS_OUTCOME.get(r.status, "unsupported_calculation")
        if r.status in ("answered", "no_data"):
            self._remember(out.plan, pr); self.qcs.last = q; self.state.pending = None
        self.trace.record(text, q, {"answered": "answered", "no_data": "no_data"}.get(r.status, "limitation"), "; ".join(q.notes))
        body = a.text
        if out.existence and r.status in ("answered", "no_data") and r.result is not None:
            what = (q.event.type.replace("_", " ").lower() if q.event else describe(q))
            body = render_existence(a.text, r, what)
        text_out = body + ("\n" + " ".join(q.notes) if q.notes and (out.existence or a.kind in ("value", "analysis")) else "")
        if r.status in ("answered", "no_data"):                                                             # the receipt: what was run, so the person can check it (described from the typed query)
            try: text_out += "\nI read this as: " + describe(q) + "."
            except Exception: pass
        self._brief = a.brief; tm = dict(tm, execute_ms=round((t2 - t1) * 1000, 1))
        facts = dict(r.facts(), suggest=a.followups[:3] or None, plan=out.plan)
        self.outcome = label; tm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1); self.timing = tm
        confidence = 0.9 if r.status in ("answered", "no_data") else 0.5
        return self._result(t0, text_out, "QC_" + ("EXISTS" if out.existence else q.op.upper()), {"metric": q.target, "event": q.event.type if q.event else None, "w": q.scope.time.to_text()}, "qc", confidence, dict(facts, outcome=label, call=pr.raw, retrieved=pr.card_ids),
                            None, {"ms": tm.get("generate_ms"), "tokens": tm.get("out_tokens"), "retrieve_ms": tm.get("retrieve_ms"), "timing": tm}, t1, t2)

    # ---- domain tools: typed, read-only, with an explicit evidence statement (they have no evidence contract of their own)
    def _tool(self, text, out, t0, tm, pr):
        t1 = time.perf_counter(); intent = TOOL_INTENT.get(out.tool); none_p = {"metric": None, "event": None, "w": "a"}
        if intent is None: return self._finish(t0, REPLY_FAILED, "CLARIFY", none_p, "plan", "invalid_plan", {"ok": False, "why": "no tool"}, tm, pr)
        focus = out.focus or None; p = {"metric": None, "event": None, "w": out.window, "focus": focus, "q": text}
        if intent == "SERVICE": p["focus"] = {"odometer": "odo", "model": "model", "brakes": "brakes"}.get(focus, "service")
        if intent == "WEATHER": p["focus"] = "now" if focus == "now" else "period"
        if intent == "HEALTH" and focus in ("all", None): p["focus"] = None
        try: f = self.run_tool(intent, p); ans = legacy_compose(intent, p, f, self.persona)
        except Exception as e: return self._finish(t0, "I could not run that check (" + type(e).__name__ + ").", "NONE", none_p, "tool", "execution_error", {"ok": False}, tm, pr)
        ans += "\n(Evidence: read-only domain tool '" + out.tool + "' over the recorded data; this tool reports its own checks and has no separate evidence contract.)" + ("\n" + " ".join(out.notes) if out.notes else "")
        self._remember(out.plan, pr); self.state.pending = None; self.outcome = "tool"; tm = dict(tm, execute_ms=round((time.perf_counter() - t1) * 1000, 1), total_ms=round((time.perf_counter() - t0) * 1000, 1)); self.timing = tm
        return self._result(t0, ans, intent, p, "tool", 0.8, dict(f if isinstance(f, dict) else {}, tool=out.tool, outcome="tool", call=pr.raw, retrieved=pr.card_ids), None, {"ms": tm.get("generate_ms"), "tokens": tm.get("out_tokens"), "timing": tm}, t1, time.perf_counter())
