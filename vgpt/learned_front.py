"""The learned semantic front end: raw question (+ the previous plan) -> QueryPlan written by the language model -> validated -> executed deterministically.

    learn the meaning (Qwen2.5-0.5B + query-plan LoRA, grammar-constrained)  ->  validate the structure (query_validator / grounding)  ->  calculate deterministically (qc / plan_exec)

No lexical rule takes part in understanding here: the text goes to the model unchanged (only markup/control characters are stripped).  The deterministic domain policies
(1-minute steps, 120 s continuity, 90 % day coverage, type-7 percentiles, mode mapping, negation scope, trip classes ...) live in the executors and are untouched.

Modes (env VGPT_FRONTEND, read by Brain):
    legacy        the rule pipeline only (default)
    learned       this front end first; when it abstains the legacy pipeline answers and the answer is LABELLED path="legacy_fallback" (never hidden)
    learned_only  this front end only; abstention is an honest clarification / refusal (the ablation configuration: legacy rules are not consulted at all)
    shadow        legacy answers; the learned plan is computed too and attached to facts["shadow_plan"] for comparison
"""
import os
import re
import time

from .query_ir import Query, PLAN_OPS, IRError

from . import outcomes as _outcomes
WHY_TEXT = _outcomes._V1           # legacy name (contract 1 wording); the renderer below reads vgpt.outcomes


_NUMW = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90, "hundred": 100, "half": 0.5, "a": 1, "an": 1, "couple": 2, "dozen": 12}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _dl(a, b):
    """Damerau-Levenshtein distance (insert, delete, substitute, adjacent swap)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): d[i][0] = i
    for j in range(len(b) + 1): d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]: d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _closest_number_word(w):
    """-> [key] or []: the number word nearest to a misspelt token (at most one edit per 4 letters); ties go to the word of the same length (eighyt is eighty, not eight)."""
    best = []
    for k in _NUMW:
        if len(k) < 4: continue
        dist = _dl(w, k)
        if dist <= max(1, len(k) // 4) and dist < len(w) / 2: best.append((dist, abs(len(w) - len(k)), -len(k), k))
    return [min(best)[3]] if best else []


_NOT_NUMBERS = {"tend", "tent", "tens", "text", "next", "nice", "mine", "mind", "line", "lines", "fine", "five's", "seen", "send", "sent", "seem", "ones", "zero's", "fourth", "tenth", "third", "first",
                "second", "thirds", "tends", "teen", "teens", "forty's", "than", "then", "that", "this", "them", "these", "those", "what", "when", "more", "most", "such", "much", "many", "make", "made"}


def _numbers_in(text):
    import difflib
    t = text.lower(); out = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)}
    for m in re.finditer(r"(?<![\w.])(?:-|−)\s?(\d+(?:\.\d+)?)", t): out.add(-float(m.group(1)))        # a written minus sign: "below -3 degrees"
    for m in re.finditer(r"\b(?:minus|negative)\s+(\d+(?:\.\d+)?)", t): out.add(-float(m.group(1)))
    if re.search(r"\bfortnight\b", t): out.add(14.0)                                                         # a fortnight is fourteen days
    words = re.findall(r"[a-z]+", t); prev = None; sign = False
    for w in words:
        v = _NUMW.get(w)
        if v is None and len(w) >= 4 and not (w.endswith("ight") or w in _NOT_NUMBERS):         # a misspelt number word ("fory", "eighyt"); ordinary words that merely look like one ("right", "tent") are not numbers
            m = _closest_number_word(w)
            v = _NUMW[m[0]] if m else None
        if v is not None:
            out.add(float(v))
            if sign: out.add(-float(v))
            if prev in _TENS and 1 <= v <= 9:
                out.add(float(_TENS[prev] + v))
                if sign: out.add(-float(_TENS[prev] + v))
        prev = w if w in _TENS else None
        sign = w in ("minus", "negative") or (sign and w in _TENS)
    return out


def _has(nums, v):
    return any(abs(v - n) <= max(0.01, abs(n) * 0.006) for n in nums) or any(abs(v - n * f) <= max(0.01, abs(v) * 0.006) for n in nums for f in (1.60934, 0.621371, 3.6, 1 / 3.6, 60, 1 / 60, 24, 7, 30, 1000, 0.001))


def _context_numbers(context):
    """Numeric arguments of the previous (trusted) plan text, or an empty set."""
    if not context: return set()
    try: q = Query.from_text(context)
    except Exception: return set()
    out = set()
    for v in (q.params or {}).values():
        for part in re.split(r"[,:]", str(v)):
            try: out.add(float(part))
            except ValueError: pass
    for sc in (q.scope, q.scope_b):
        if sc is None: continue
        for c in sc.conditions:
            if c.value is not None: out.add(float(c.value))
    return out


_UNIT_S = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400, "month": 30 * 86400}      # the executor's own definition (query_planner.resolve_time)
_UNIT_WORDS = {"minute": "minute|minutes|min|mins", "hour": "hour|hours|hr|hrs|h", "day": "day|days|d", "week": "week|weeks|wk|wks", "month": "month|months|mo"}
_IMPLIED = {"week": 7 * 86400, "month": 30 * 86400, "fortnight": 14 * 86400, "day": 86400}          # "last week" / "last month" alone are ROLLING windows (documented policy)


def _word_number(w):
    import difflib
    v = _NUMW.get(w)
    if v is None and len(w) >= 4:
        m = _closest_number_word(w)
        v = _NUMW[m[0]] if m else None
    return v


_UNIT_VOCAB = ("minutes", "minute", "hours", "hour", "days", "day", "weeks", "week", "months", "month", "fortnight", "yesterday", "today")
_STOP_NEAR = {"mount", "mouth", "daily", "dawn", "dash", "hold", "hose", "house", "hair", "wheel", "weeks", "weak", "week", "weeks", "wake", "work", "word", "month", "today", "toady", "yes"}


def _fix_unit_typos(t):
    """A misspelt period word ("moneh", "weke", "yestrday") still names the period the user meant: map it to the nearest unit word (same first letter, ratio >= 0.84)."""
    import difflib
    def fix(m):
        w = m.group(0)
        if len(w) < 4 or w in _UNIT_VOCAB or w in _STOP_NEAR or w in _NUMW: return w
        c = difflib.get_close_matches(w, _UNIT_VOCAB, n=1, cutoff=0.84)
        return c[0] if c and c[0][0] == w[0] else w
    return re.sub(r"[a-z]+", fix, t)


_EQUAL_LEN = re.compile(r"(?i)\bequal[- ]length\b|\bsame (?:length|duration|span)\b|\bimmediately (?:before|preceding)\b|\bpreceding window\b|\bwindow of equal\b")
_CAL_LEN = {"today": 86400, "yesterday": 86400, "day_before_yesterday": 86400, "last_night": 86400, "this_week": 7 * 86400, "last_week": 7 * 86400, "this_month": 30 * 86400, "last_month": 30 * 86400}


def _calendar_len(ts):
    return _CAL_LEN.get(ts.name) if ts is not None and ts.type == "calendar" else None


def _question_windows(text):
    """Window lengths (seconds) the USER asked for: '<number> <unit>', 'a/an <unit>', 'fortnight', and the policy-implied 'last/past/previous <unit>' alone.
    Returns (set of seconds, whether any period is stated at all)."""
    t = _fix_unit_typos(text.lower()); out = set(); stated = False
    for m in re.finditer(r"\b(?:(?P<q>a|one|three)\s+quarters?\s+of\s+an\s+hour|half\s+an\s+hour|a\s+half[\s-]hour|quarter\s+of\s+an\s+hour|a\s+quarter[\s-]hour)\b", t):
        s_ = m.group(0); out.add(1800 if "half" in s_ else (2700 if m.group("q") == "three" else 900)); stated = True
    t = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]+(one|two|three|four|five|six|seven|eight|nine)\b",
               lambda m: str(_TENS[m.group(1)] + _NUMW[m.group(2)]), t)                                   # "twenty-nine hours" is 29 hours
    for m in re.finditer(r"\b(?:(?P<h>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|last|past|previous|trailing|preceding)\s*)?hours?\s*(?:and\s*)?(?:(?P<m>\d+)\s*(?:minutes?|mins?)|(?P<half>a half|half))\b", t):
        hv = 1.0 if m.group("h") in (None, "last", "past", "previous", "trailing", "preceding") else (float(m.group("h")) if m.group("h").isdigit() else _word_number(m.group("h")))
        if hv is None: continue                                                                           # "an hour and thirty-one minutes", "two hours and a half"
        out.add(int(round(hv * 3600 + (int(m.group("m")) * 60 if m.group("m") else 1800)))); stated = True
    full = {u: w for u, w in (("minute", "minutes?"), ("hour", "hours?"), ("day", "days?"), ("week", "weeks?"), ("month", "months?"))}
    unit_full = "|".join(f"(?P<{u}>{w})" for u, w in full.items())
    abbr = "|".join(f"(?P<{u}>{w})" for u, w in _UNIT_WORDS.items())
    for m in re.finditer(r"\b(?P<n>\d+(?:\.\d+)?)\s*-?\s*(?:" + abbr + r")\b", t):        # digits may use abbreviated units (14 d, 2 wks)
        unit = next(u for u in _UNIT_S if m.group(u)); out.add(int(round(float(m.group("n")) * _UNIT_S[unit]))); stated = True
    for m in re.finditer(r"\b(?P<n>[a-z]+)\s*-?\s*(?:" + unit_full + r")\b", t):             # number WORDS need a spelled-out unit ("two weeks", "a day")
        unit = next(u for u in _UNIT_S if m.group(u)); n = _word_number(m.group("n"))
        if n is None: continue
        out.add(int(round(n * _UNIT_S[unit]))); stated = True
    if re.search(r"\bfortnight\b", t): out.add(14 * 86400); stated = True
    for m in re.finditer(r"\b(?:last|past|previous|prior|whole|entire|full|complete)\s+(week|month|day|hour)\b", t):
        out.add(_IMPLIED.get(m.group(1), _UNIT_S[m.group(1)])); stated = True
    for m in re.finditer(r"\b(?:the\s+)?(day|week|month)\s+(?:before|prior)\b|\b(?:the\s+)?(?:previous|preceding|prior)\s+(day|week|month)\b", t):      # "the day before", "the week before that"
        u_ = m.group(1) or m.group(2); out.add(_IMPLIED.get(u_, _UNIT_S[u_])); stated = True
    if re.search(r"\b(?:last|past)\s+24\s*(?:hours?|h)\b", t): out.add(86400); stated = True
    return out, stated


def _window_seconds(ts):
    if ts is None or ts.type not in ("relative", "previous") or ts.value is None: return None
    return int(ts.value) * _UNIT_S[ts.unit]


def _context_windows(context):
    if not context: return set()
    try: q = Query.from_text(context)
    except Exception: return set()
    return {w for w in (_window_seconds(q.scope.time), _window_seconds(q.scope_b.time) if q.scope_b else None) if w}


def _has_period_word(text):
    return bool(re.search(r"(?i)\b(?:day|week|month|hour|fortnight|year)s?\b|\b(?:today|yesterday)\b", text))



GLOSSARY_KEYS = ["can_bus", "soc", "soh", "regen", "tpms", "adas", "lane_assist", "dtc", "wh_per_km", "kwh", "dc_fast", "ac_charging", "odometer", "aux_12v", "heat_pump", "obd", "uds",
                 "someip", "vhal", "language_model", "range", "coolant", "ota", "pressure_units", "speed_units"]      # same order as dialogue.GLOSSARY (the definitions are data, looked up by key)


def definition(key):
    from . import dialogue
    try: return dialogue.GLOSSARY[GLOSSARY_KEYS.index(key)][1]
    except (ValueError, IndexError): return None


class Interp:
    def __init__(self, query=None, raw="", ms=0.0, error=None, issues=None, min_p=None):
        self.query, self.raw, self.ms, self.error, self.issues, self.min_p = query, raw, ms, error, issues or [], min_p

    @property
    def abstain(self): return self.query is None or self.query.op == "none"


class LearnedFront:
    def __init__(self, brain, model=None):
        self.brain, self.model = brain, model          # model: object with .compile(text, context=None) -> (Query|None, info)
        self.last_plan = None                          # plan text of the previous answered turn (dialogue reference is resolved by the model)
        self.contract = _outcomes.contract_for(getattr(model, "lora_path", None) or getattr(getattr(model, "slm", None), "lora_path", None))

    # ---- understanding --------------------------------------------------------------------------------------------------------------------------------
    def interpret(self, text, context="__use_last__"):
        ctx = (self.last_plan if getattr(self.brain.qcs, "last", None) is not None else None) if context == "__use_last__" else context      # context only while a conversation is open
        if self.model is None: return Interp(error="no model")
        try: q, info = self.model.compile(text, context=ctx)
        except Exception as e: return Interp(error=f"{type(e).__name__}: {e}")
        if q is None: return Interp(raw=info.get("raw", ""), ms=info.get("ms", 0.0), error=info.get("error") or "unparseable")
        it = Interp(q, info.get("raw", ""), info.get("ms", 0.0), min_p=info.get("min_p"))
        it.issues = self.check(q, text, context=ctx)
        tau = float(os.environ.get("VGPT_CONF_MIN", "0.8"))                              # the model's own confidence: a plan written with a token it was unsure of is not executed
        safety_topic = q.op == "topic" and (q.params or {}).get("topic") == "safety_assessment"         # a conservative safety answer is the fail-safe direction: not gated
        if q.op != "none" and not safety_topic and it.min_p is not None and it.min_p < tau: it.issues.append(f"low confidence in the reading ({it.min_p:.2f})")
        return it

    def check(self, q, text, context=None):
        """Structural validation of a model-written plan (not understanding).  The only grounding test: a numeric argument the model wrote (threshold, minimum duration,
        percentile, window length) must be traceable to a number in the question - as digits, a number word (also misspelt: within edit distance) or a fixed
        spoken quantity - otherwise the model invented it.  Signals / states / events come from the registry grammar and are not re-checked against words."""
        if q.op == "none": return []
        from .query_validator import validate as _validate
        nums = _numbers_in(text); issues = [str(x) for x in _validate(q).issues]
        inherited = _context_numbers(context)               # numbers of the TRUSTED previous plan; a pure modifier follow-up ("not counting charging") legitimately keeps them
        stated = set(nums)                                  # what the user said in this turn
        nums = nums | inherited
        want = []
        for k in ("thr", "pct", "v1", "v2", "k", "n", "span", "mind"):
            v = (q.params or {}).get(k)
            if v in (None, ""): continue
            for part in re.split(r"[,:]", str(v)):
                try: want.append((k, float(part)))
                except ValueError: pass
        for k, v in want:
            if k == "mind":
                if not any(abs(v - n * m) < 1e-6 for n in nums for m in (1, 60, 3600, 86400)): issues.append(f"the duration {v:g} s is not in the question")
            elif k == "pct" and v in (0, 25, 50, 75, 100): pass            # median / quartiles / min / max are named by a word, not a number
            elif k == "k" and v == 1: pass                                 # "which day was highest": the singular is k=1, there is no number to ground
            elif not _has(nums, v) and not (k == "thr" and v == 0): issues.append(f"the number {v:g} is not in the question")
        if inherited and stated:                            # the user gave a number of their own: a plan that silently keeps a different inherited value contradicts them
            plan_nums = {v for _, v in want} | {float(c.value) for c in q.scope.conditions if c.value is not None}
            used_inherited = [v for v in plan_nums if _has(inherited, v) and not _has(stated, v)]
            window_nums = set()                                 # a number that names the plan's own window ("two weeks", "a fortnight", "30 minutes") is used, not ignored
            for ts_ in (q.scope.time, q.scope_b.time if q.scope_b else None):
                w_ = _window_seconds(ts_)
                if w_: window_nums |= {float(ts_.value), w_ / 60.0, w_ / 3600.0, w_ / 86400.0, w_ / (7 * 86400.0), w_ / (30 * 86400.0)}
            unused_stated = [n for n in stated if not _has(plan_nums, n) and not _has(plan_nums, -n) and not _has(window_nums, n) and n not in (1.0,)]
            if used_inherited and unused_stated:
                issues.append(f"the question gives {unused_stated[0]:g} but the plan keeps the earlier value {used_inherited[0]:g}")
        sc = q.scope
        for c in sc.conditions:
            if c.value is not None and not _has(nums, float(c.value)): issues.append(f"the level {c.value:g} is not in the question")
        asked, stated_period = _question_windows(text); inherited_w = _context_windows(context)
        for ts in (sc.time, q.scope_b.time if q.scope_b else None):
            total = _window_seconds(ts)
            if total is None: continue                                      # calendar, date, current, all: no length to ground
            label = f"{ts.value:g} {ts.unit}{'s' if ts.value != 1 else ''}"
            if total in asked: continue                                     # asked explicitly (equivalent units accepted: 2 weeks == 14 days)
            if q.scope_b is not None and ts is q.scope_b.time and _EQUAL_LEN.search(text):             # "the equal-length period immediately before it": same length as the primary window
                pl = _calendar_len(sc.time) or _window_seconds(sc.time)
                if pl and total == pl: continue
            if stated_period:                                               # the user named a different period: the plan contradicts or invents
                issues.append(f"the plan uses a window of {label} but the question asks for a different period"); continue
            if total in inherited_w: continue                               # nothing stated this turn: legitimately inherited from the trusted context plan
            if total == 7 * 86400: continue                                 # the documented default window ("No period given, so I used the last 7 days")
            if _has(nums, float(ts.value)) and _has_period_word(text): continue   # abbreviated or oddly worded period that still carries the number and a period word
            issues.append(f"the period length {label} is not in the question or the previous plan")
        try:
            from . import measure_grounding as _MG
            issues += _MG.plan_issues(q, text, context_plan=context)                   # a selected measure must be supported by the request, not merely exist in the schema
            from . import condition_grounding as _CG
            issues += _CG.plan_issues(q, text, context_plan=context)                   # a state under a direct negation cue must not become a positive condition
        except Exception as e:                                                             # a broken grounding gate is not permission to execute
            issues.append(f"grounding checks unavailable ({type(e).__name__})")
        return issues

    # ---- execution -----------------------------------------------------------------------------------------------------------------------------------
    def execute(self, q, text):
        """-> dict(answer, intent, facts, brief) or None.  Plan operations go to the text-free plan executor, classic ones to the typed query compiler."""
        qc = self.brain.qc()
        if q.op in PLAN_OPS:
            from . import plan_exec
            r = plan_exec.execute(self.brain, qc, q)
            if r is not None: return r
        r = qc.execute(q, text=text)
        if r.status == "invalid": return None
        facts = r.facts(); a = r.answer
        return {"answer": a.text, "brief": getattr(a, "brief", None), "intent": "QC_" + q.op.upper(), "facts": facts, "status": r.status}

    # ---- the turn ------------------------------------------------------------------------------------------------------------------------------------
    def answer(self, text):
        """-> (result_dict | None, Interp).  None = abstain (the caller decides what to do: honest clarification or labelled legacy fallback)."""
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text)); text = " ".join(text.split())
        if len(text) > 600:
            it = Interp(error="question exceeds the 600-character input limit")
            return {"answer": "That request is too long to interpret safely. Please split it into shorter questions; I will not silently discard any conditions.", "brief": None,
                    "intent": "NONE", "facts": {"ok": False, "why": "input_too_long", "learned": True}}, it
        ctx_before = self.last_plan if getattr(self.brain.qcs, "last", None) is not None else None
        it = self.interpret(text)
        tr = os.environ.get("VGPT_LEARNED_TRACE")
        if tr:                                                                                                   # audit trail: what the model wrote, how sure it was, what the gate did
            import json
            try:
                with open(tr, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"text": text, "context": ctx_before, "raw": it.raw, "min_p": it.min_p, "issues": it.issues, "error": it.error, "ms": round(it.ms, 1)}, ensure_ascii=False) + chr(10))
            except OSError: pass
        if it.query is not None and it.query.op == "topic" and (it.query.params or {}).get("topic") == "safety_assessment":
            from . import safety_policy
            si = safety_policy.classify(text) or safety_policy.SafetyIntent("generic", safety_policy.GENERIC[3], safety_policy.GENERIC[2], "")
            r = safety_policy.respond(si, self.brain if hasattr(self.brain, "c") else None); r["facts"].update({"learned": True, "plan": it.raw})
            return r, it
        if it.query is not None and any("no part of the question supports" in i and "(it names" in i for i in it.issues):
            text_, kind_ = _outcomes.render("unrecorded", self.contract)          # the question names a quantity the schema does not model: a grounded "not recorded"
            return {"answer": text_, "brief": None, "intent": "NONE", "facts": {"ok": False, "why": "unrecorded", "outcome": kind_, "outcome_contract": self.contract, "learned": True, "grounding": "absent quantity", "rejected_plan": it.raw}}, it
        if it.query is not None and it.query.op == "none" and (it.query.params or {}).get("why") == "unrecorded":
            from . import measure_grounding as _MG
            prof = None
            try: prof = self.brain.qc().profile
            except Exception: prof = None
            iss_, why_ = _MG.outcome_issues("unrecorded", text, profile=prof, context_plan=ctx_before)
            if iss_: it.issues += iss_
            elif why_ == "ambiguous": it.query.params["why"] = "ambiguous"
        if it.query is None or it.issues: return None, it
        q = it.query
        if q.op == "none":
            why = (q.params or {}).get("why", "unsupported")
            if why == "definition":
                d = definition((q.params or {}).get("kind"))
                if d: return {"answer": d, "brief": None, "intent": "GLOSSARY", "facts": {"ok": True, "learned": True, "route": "glossary", "kind": q.params.get("kind")}}, it
            text_, kind_ = _outcomes.render(why, self.contract)
            if text_ is None:                                                       # an outcome category the contract does not define is a validator failure, never another category
                it.issues.append(f"unknown outcome category {why!r}"); return None, it
            return {"answer": text_, "brief": None, "intent": "NONE", "facts": {"ok": False, "why": why, "outcome": kind_, "outcome_contract": self.contract, "learned": True}}, it
        try: r = self.execute(q, text)
        except Exception as e:
            it.error = f"execute: {type(e).__name__}: {e}"; return None, it
        if r is None: return None, it
        f = r.setdefault("facts", {}); f["learned"] = True; f["plan"] = it.raw
        if r.get("status") in (None, "answered", "no_data") and (r.get("facts") or {}).get("ok", True): self.last_plan = it.raw; self.brain.qcs.last = q
        return r, it

    def clarification(self, it):
        if it is not None and it.issues:
            return "I read that as a request I could not ground in what you wrote (" + "; ".join(str(i) for i in it.issues[:2]) + "). Could you restate the quantity, the level and the period?"
        return "I could not turn that into a question I can compute from this vehicle's data. Tell me the signal or event and the time span."


_ORIG = {}


class LegacyRuleCalled(RuntimeError):
    """Raised by the ablation tripwire when a legacy lexical understanding entry point is reached in learned_only mode."""


def arm_tripwire():
    """Ablation proof: after this call any legacy understanding entry point (text normaliser/spell repair, rule parser, composed/temporal text fronts, keyword routes)
    raises LegacyRuleCalled instead of running.  Used by tests and by VGPT_TRIPWIRE=1 runs; the deterministic executors and policies stay live."""
    import importlib
    def boom(name):
        def f(*a, **k): raise LegacyRuleCalled(name)
        return f
    targets = [("vgpt.nlu", "normalize"), ("vgpt.nl2ir", "parse"), ("vgpt.composed", "handle"), ("vgpt.temporal_ops", "handle"), ("vgpt.temporal_misc", "handle"),
               ("vgpt.temporal_events", "handle"), ("vgpt.semantic_ops", "handle"), ("vgpt.brain", "parse_clean"), ("vgpt.brain", "parse_domain")]
    armed = []
    for mod, fn in targets:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, fn) and (m, fn) not in _ORIG: _ORIG[(m, fn)] = getattr(m, fn); setattr(m, fn, boom(f"{mod}.{fn}")); armed.append(f"{mod}.{fn}")
        except Exception: pass
    return armed


class legacy_allowed:
    """Context manager: topic routes hand a CANONICAL request (never the user's text) to a legacy route handler; the tripwire is lifted for exactly that call."""
    def __enter__(self):
        self.saved = dict(_ORIG)
        for (m, fn), f in list(_ORIG.items()): setattr(m, fn, f)
        return self

    def __exit__(self, *a):
        def boom(name):
            def f(*a, **k): raise LegacyRuleCalled(name)
            return f
        for (m, fn) in self.saved: setattr(m, fn, boom(f"{m.__name__}.{fn}"))
        return False


def disarm_tripwire():
    for (m, fn), f in list(_ORIG.items()): setattr(m, fn, f)
    _ORIG.clear()
