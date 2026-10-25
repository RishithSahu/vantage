"""Answer verifier: the last gate before anything is shown. It checks the composed text against the evidence it claims to rest on and against the registry, and refuses answers that cannot be justified.

Checks
  NUMBER     every number in the text is an evidence item, a count / threshold / window taken from the query or the result, or part of a recorded string (an event description) that is quoted verbatim
  UNIT       every unit named in the text belongs to a unit family that appears in the evidence (or is time / percent)
  SIGNAL     every signal named in the text is part of the query, the evidence or the declared dependencies - nothing "silently used"
  ESTIMATE   an estimated result says "estimate"; a measured one does not call itself one; assumptions are listed
  UNSUPPORTED  a missing-data / no-data / unsupported answer states no value at all
  CAUSAL     no bare causal claim ("caused", "due to", "proves") - only preceded / associated with / likely contributor
  SAFETY     no vehicle-control instruction (turn off, disable, override, send ...) outside an explicit refusal
  TIME       the answer's timestamp lies inside the recorded history
If a check fails the caller falls back to a conservative evidence-only answer or to the legacy path; a model-written sentence never reaches the screen unverified."""
import re
from dataclasses import dataclass, field

from . import signal_registry as R
from .query_ir import Query

NUM_RX = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
DATE_RX = re.compile(r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?: \d{4})?\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
LISTNUM_RX = re.compile(r"^\s*(?:#|\()?\d+[.)]\s", re.M)
UNIT_RX = re.compile(r"(?<![A-Za-z])(km/h|mph|m/s2|m/s|kWh|Wh/km|Wh/mi|kPa|bar|psi|kW|rpm|Nm|L/min|°C|°F|hp|lux|V|A|W|g|km|mi|%)(?![A-Za-z])")
CAUSAL_RX = re.compile(r"\b(caused by|because of|due to|the cause is|proves?|proved|is the cause|was the cause|resulted from)\b|\bcaused\b", re.I)
SAFETY_RX = re.compile(r"\b(turn(ed)? off|switch(ed)? off|disable|deactivate|override|bypass|send (a )?(can|command|frame)|actuate|unlock the)\b", re.I)
REFUSAL_RX = re.compile(r"\b(can't|cannot|can not|won't|will not|not able|no way|read-only|only (read|analy))", re.I)
REFUSAL_ACTION_RX = re.compile(r"\b(?:can't|cannot|can not|won't|will not|not able|read-only)\b[^.]{0,120}\b(?:turn(?:ed)? off|switch(?:ed)? off|disable|deactivate|override|bypass|send (?:a )?(?:can|command|frame)|actuate|unlock)\b", re.I)
FAMILY_ALWAYS = {"time", "percent", "count", "none", "ratio"}


@dataclass
class Report:
    ok: bool = True
    problems: list = field(default_factory=list)       # (code, detail)
    numbers_checked: int = 0

    def fail(self, code, detail): self.ok = False; self.problems.append((code, detail))

    def __str__(self): return "ok" if self.ok else "; ".join(f"{c}: {d}" for c, d in self.problems)


def _query_numbers(q):
    out = set()
    def add(x):
        if isinstance(x, (int, float)) and not isinstance(x, bool): out.add(float(x))
    if q is None: return out
    t = q.scope.time; add(t.value)
    for sc in (q.scope, q.scope_b):
        if sc is None: continue
        add(sc.time.value)
        if sc.time.type == "calendar" and sc.time.name:                                                   # a year / month the person typed (y2025, m2025_03) is quoted back in the answer's period label
            for g in re.findall(r"\d+", sc.time.name): add(int(g))
        for g in (sc.time.value2,): add(g)
        for c in sc.conditions: add(c.value)
        if sc.select: add(sc.select.k)
    if q.event:
        add(q.event.before); add(q.event.after); add(q.event.n)
        for w in (q.event.before, q.event.after):                                                           # the same window restated in minutes / hours
            if isinstance(w, (int, float)) and w:
                add(w / 60); add(w / 3600)
    return out


def _data_strings(res):
    s = []
    for it in (res.items or []):
        if isinstance(it, dict):
            for k in ("detail", "type"):
                if isinstance(it.get(k), str): s.append(it[k])
    ev = (res.extra or {}).get("event")
    if isinstance(ev, dict) and isinstance(ev.get("detail"), str): s.append(ev["detail"])
    for ne in (res.extra or {}).get("near_events", []) or []:
        if isinstance(ne.get("detail"), str): s.append(ne["detail"])
    for k in (res.reason, res.period_label):                                                              # the engine's own explanation of why there is no value / the window it used ("60 s before the event")
        if k: s.append(k)
    for k in ((res.extra or {}).get("open") or []): s.append(str(k))                                      # measurements that would help are named, not claimed
    for f in (((res.extra or {}).get("contrib") or {}).get("factors") or []): s += [f.get("unit", ""), f.get("name", "")]       # a factor's own description ("degrees below 18 C") is a label, not a measurement
    for a in ((res.extra or {}).get("assoc") or []):
        if isinstance(a, dict) and a.get("mechanism"): s.append(a["mechanism"])
    d = (res.extra or {}).get("diag")
    if isinstance(d, dict):
        for a in d.get("alts") or []: s += [a.get("name", ""), a.get("name", "").split(" (")[0], a.get("support", "")]
    return [x for x in s if x]


def _matches(n, decs, allowed):
    tol = 0.5 * 10 ** (-decs) + 1e-9
    for a in allowed:
        if abs(n - a) <= tol or abs(n - round(a, decs)) <= 1e-9: return True
        if decs == 0 and abs(n - a) < 1.0 and abs(round(a) - n) < 1e-9: return True
    return False


_MAGNITUDE_WORDS = re.compile(
    r"(?:fell|fallen|fall|dropped|drop|decreased|decrease|declined|decline|reduced|reduction|"
    r"lower|less|down|lost|loss|empt(?:ied|ying)|worse|accounts?\s+for|share)\b|\bof\s+the\s+gap",
    re.I,
)


def _matches_in_text(n, decs, allowed, text, start, end):
    """Match a displayed number without erasing its sign.

    A positive *magnitude* is nevertheless legitimate in wording such as
    ``fell by 12%`` or ``12% lower`` when the stored change is -12.  The old
    verifier treated every value as an absolute value, which also licensed the
    materially wrong sentence ``increased by 12%``.  Limit the alternate-sign
    match to an explicit downward/magnitude construction adjacent to the
    number.
    """
    if _matches(n, decs, allowed):
        return True
    if n < 0:
        return False
    nearby = text[max(0, start - 120): min(len(text), end + 48)]
    return bool(_MAGNITUDE_WORDS.search(nearby)) and _matches(-n, decs, allowed)


def _numeric_leaves(value):
    """Yield the numeric leaves of a computed result payload.

    Executor diagnostics are deliberately structured (for example
    ``extra.jump.gap_days``).  Those values are just as much computed evidence
    as top-level fields and must not be rejected merely because they are
    nested.  Booleans are excluded because ``True`` is not evidence for 1.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numeric_leaves(child)


_PERCENT_KEY = re.compile(r"(?:pct|percent|percentage|ratio|share|coverage)", re.I)


def _percentage_leaves(value, key=""):
    """Yield only values whose structured field explicitly denotes a percent.

    A bare numeric result (for example 50 km/h) is never evidence for 50%.
    Some contracts store coverage/ratio values in nested diagnostics rather
    than as EvidenceItems, so retain those deliberately named percentages.
    """
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _percentage_leaves(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _percentage_leaves(child, key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and _PERCENT_KEY.search(key):
        yield float(value)


def verify(answer, ev, q=None, res=None, capability=None, ctx=None):
    rep = Report(); text = answer.text
    # strip quoted recorded strings and dates / times / list markers before reading numbers
    # A contract's resolution is provenance metadata generated from the result
    # (for example, "1-second samples").  It is not a new numeric claim made
    # by the answer.  Preserve the number gate for every other part of the
    # answer; exempting the exact stored contract phrase prevents a verified
    # event-context answer being discarded merely because its evidence label
    # contains the digit 1.
    contract = ((res.extra or {}).get("contract") or {}) if res is not None else {}
    resolution = contract.get("resolution") if isinstance(contract, dict) else ""
    exempt = (_data_strings(res) if res is not None else []) + [f"Data quality: {f['detail']}." for f in ev.flags] + [f["detail"] for f in ev.flags] + [a for a in ev.assumptions] + ["Method: " + ev.method + "." if ev.method else ""] + ([ev.method] if ev.method else []) + [f"({ev.tier})", resolution]
    for s in sorted([x for x in exempt if x], key=len, reverse=True): text = text.replace(s, " ")                 # deterministic strings produced by the engines themselves are quoted, not re-derived
    for lab in sorted({c.label for c in R.CONCEPTS.values() if re.search(r"\d", c.label)}, key=len, reverse=True): text = re.sub(re.escape(lab), " ", text, flags=re.I)        # "12 V battery voltage" names a signal; the 12 is not a measurement
    text = re.sub(r"\b95th[- ]percentile\b", " ", text, flags=re.I)
    for lab in sorted({e[0] for e in R.EVENTS.values() if re.search(r"\d", e[0])}, key=len, reverse=True): text = re.sub(re.escape(lab), " ", text, flags=re.I)
    body = LISTNUM_RX.sub(" ", DATE_RX.sub(" ", text))
    for n in (q.notes if q is not None else []): text = text.replace(n, " ")
    allowed = set(ev.numbers()) | _query_numbers(q)
    allowed |= {float(len(ev.items)), float(ev.samples)}
    if res is not None:
        allowed |= {float(len(res.items or [])), float(res.samples or 0)}
        if isinstance(res.value, (int, float)) and not isinstance(res.value, bool): allowed.add(float(res.value))
        for it in res.items or []:
            if isinstance(it, dict):
                for v in it.values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool): allowed.add(float(v))
        allowed.update(_numeric_leaves(res.extra or {}))
        for k in ("a", "b"):
            if isinstance((res.extra or {}).get(k), (int, float)): allowed.add(float((res.extra or {})[k]))
    if answer.kind in ("missing", "unsupported", "no_data"): allowed = _query_numbers(q) | {0.0}
    for m in NUM_RX.finditer(body):
        raw = m.group(0).replace(",", "");
        try: n = float(raw)
        except ValueError: continue
        decs = len(raw.split(".")[1]) if "." in raw else 0; rep.numbers_checked += 1
        if n in (0.0,) and answer.kind in ("missing", "unsupported", "no_data"): continue
        if not _matches_in_text(n, decs, allowed, body, m.start(), m.end()): rep.fail("NUMBER", f"'{m.group(0)}' is not in the evidence")
    # Bind a displayed measurement to its displayed unit.  Merely sharing a
    # unit family is insufficient: "60 mph" must not be licensed by evidence
    # that says 60 km/h.
    unit_values = []
    for i in ev.items:
        if isinstance(i.value, (int, float)) and not isinstance(i.value, bool) and i.unit: unit_values.append((float(i.value), i.unit))
    if res is not None and isinstance(res.value, (int, float)) and not isinstance(res.value, bool) and res.unit: unit_values.append((float(res.value), res.unit))
    if q is not None:
        for sc in (q.scope, q.scope_b):
            if sc is None: continue
            for c in sc.conditions:
                if isinstance(c.value, (int, float)) and c.unit: unit_values.append((float(c.value), c.unit))
    pair_rx = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*(km/h|mph|m/s2|m/s|kWh|Wh/km|Wh/mi|kPa|bar|psi|kW|rpm|Nm|L/min|°C|°F|hp|lux|V|A|W|g|km|mi|%)(?![A-Za-z])")
    for m in pair_rx.finditer(body):
        n = float(m.group(1).replace(",", "")); u = m.group(2).replace("°C", "C").replace("°F", "F"); decs = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        candidates = []
        for value, source_u in unit_values:
            su = source_u.replace("°C", "C").replace("°F", "F")
            if R.unit_family(su) != R.unit_family(u): continue
            try: candidates.append(R.convert(value, su, u))
            except ValueError: pass
        if u == "%":
            candidates.extend(value for value, source_u in unit_values if source_u.replace("°C", "C").replace("°F", "F") == "%")
            candidates.extend(_percentage_leaves(res.extra or {}) if res is not None else [])
        # A trend slope is stored as a compound unit (for example ``%/day``)
        # but is intentionally written for a driver as ``% per day``.  The
        # numeric value is the same derived result, not an unsupported percent
        # claim.  Keep this narrowly tied to the immediately following time
        # denominator so a speed, energy or arbitrary scalar can never license
        # a percentage merely because it has the same number.
        if (res is not None and isinstance(res.value, (int, float))
                and str(res.unit or "") == f"{u}/day"
                and re.match(r"\s+per\s+day\b", body[m.end():], re.I)):
            candidates.append(float(res.value))
        if not candidates:
            rep.fail("UNIT_VALUE", f"'{m.group(0)}' has no compatible evidence")
        elif not _matches_in_text(n, decs, set(candidates), body, m.start(), m.end()): rep.fail("UNIT_VALUE", f"'{m.group(0)}' does not match evidence expressed in {u}")
    # units
    fams = {R.unit_family(i.unit) for i in ev.items if i.unit} | ({R.unit_family(res.unit)} if res is not None and res.unit else set()) | FAMILY_ALWAYS
    if q is not None and q.unit: fams.add(R.unit_family(q.unit))
    if q is not None:
        for tg in (q.target, q.target2):
            if tg in R.CONCEPTS: fams.add(R.unit_family(R.CONCEPTS[tg].unit))
    for c in (q.scope.conditions if q is not None else []):
        if c.signal: fams.add(R.unit_family(R.CONCEPTS[c.signal].unit))
    if answer.kind not in ("missing", "unsupported"):
        for m in UNIT_RX.finditer(body):
            u = m.group(1).replace("°C", "C").replace("°F", "F"); f = R.unit_family(u)
            if f and f not in fams and u not in ("g", "A", "V", "W", "mi") : rep.fail("UNIT", f"unit '{m.group(1)}' does not belong to the evidence")
    # estimate labelling
    low = text.lower()
    states_value = answer.kind not in ("missing", "unsupported", "no_data")                                  # an answer that states no value has nothing to label
    if states_value and ev.kind == "estimated" and "estimate" not in low and "about" not in low and "approximately" not in low: rep.fail("ESTIMATE", "an estimated value is presented without saying so")
    if ev.kind == "measured" and re.search(r"\bestimated\b", low) and not ev.assumptions: rep.fail("ESTIMATE", "a measured value is called an estimate")
    if states_value and ev.kind == "estimated" and not ev.assumptions and res is not None and res.op != "current_value": rep.fail("ESTIMATE", "an estimate must list its assumptions")
    # signals named in the answer must be part of the question / evidence
    if answer.kind in ("value", "analysis") and q is not None:
        from .capability import concepts_of
        for c in (q.scope.conditions + (q.scope_b.conditions if q.scope_b else [])):                       # the wording of a named condition ("during hard acceleration") is not a signal claim
            if c.predicate in R.PREDICATES: low = low.replace(R.PREDICATES[c.predicate][0].lower(), " ")
        for g in q.groups:
            if g in R.PREDICATES: low = low.replace(R.PREDICATES[g][0].lower(), " ")
        if q.event is not None and q.event.type in R.EVENTS: low = low.replace(R.EVENTS[q.event.type][0].lower(), " ")
        allowed_c = set(concepts_of(q)) | {i.metric.split("_first")[0].split("_last")[0] for i in ev.items} | set(res.via if res is not None else [])
        allowed_labels = {R.CONCEPTS[c].label.lower() for c in allowed_c if c in R.CONCEPTS} | {R.label_of(i.metric).lower() for i in ev.items if i.metric in R.CONCEPTS} | {i.label.lower() for i in ev.items}
        for c in R.CONCEPTS.values():
            lab = c.label.lower()
            if len(lab) > 5 and lab in low and lab not in allowed_labels and c.name not in allowed_c and not any(lab in l for l in allowed_labels) and not any(l in lab for l in allowed_labels if len(l) > 5):
                if c.name in ("speed", "battery_soc") and c.name in (res.extra.get("state", {}) if res else {}): continue
                rep.fail("SIGNAL", f"'{c.label}' is mentioned but was not part of the query or the evidence")
    # causal / safety
    if CAUSAL_RX.search(body): rep.fail("CAUSAL", "an answer may say 'preceded', 'associated with' or 'likely contributor', not 'caused'")
    if SAFETY_RX.search(text) and not REFUSAL_ACTION_RX.search(text): rep.fail("SAFETY", "an answer must never instruct a vehicle-control action")
    # time
    if ctx is not None and res is not None and res.when is not None and not (ctx.lo - 86400 <= res.when <= ctx.hi + 86400): rep.fail("TIME", "the answer's timestamp is outside the recorded history")
    return rep


def safe_fallback(res, ev, q=None):
    """Conservative evidence-only answer used when verification of the composed text fails."""
    from .answer_composer import Answer, vu
    items = [i for i in ev.items if isinstance(i.value, (int, float)) and i.role in ("value", "comparison")][:4]
    if not items: return Answer("I computed a result but couldn't verify the wording of the answer, so I'm not stating it.", [], [], "no_data")
    brief = "Result " + (ev.period + ": " if ev.period else ": ") + "; ".join(f"{i.label or i.metric} {vu(i.value, i.unit)}" for i in items) + "."
    return Answer(brief, [], [], "value")
