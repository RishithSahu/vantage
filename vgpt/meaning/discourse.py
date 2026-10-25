"""Discourse layer of the meaning path: what a turn inherits from the previous answered turn.

The learned frame reads ONE utterance.  A conversation is more than that: "and the week before that", "what about yesterday", "which one was the longest", "why did it happen", "how fast was i
going then" and "how did you get that" are only questions given the turn before.  This module keeps the typed state of the last answered turn (its frame, its resolved period, the event or
instant it was about, the evidence it used) and completes a context-dependent utterance from it:

  * period edit      the frame is `ambiguous` (the model itself says a slot is missing) and the utterance carries a period -> the previous frame is kept, the period is replaced
                     ("for just yesterday", "the week before that", "two days ago"); a period relative to the previous one is computed from the previous RESOLVED window
  * concept edit     an elliptical opener ("and", "what about") + a concept -> the previous act / operator / condition / period are kept, the concept is replaced
  * referent         "why did it happen" with a generic concept -> the concept of the previous answer
  * deixis           "how fast was i going then" -> the instant the previous answer was about
  * ordinal          "which one was the longest" after an event answer -> that event list, ranked by the magnitude recorded in each event
  * provenance       "how did you get that" -> what was read, which query ran, which period, which samples and sources, how complete, and what the limits are

Nothing here decides which quantity a question is about: the concept comes from the frame (or the previous frame), values come from the typed parsers.  The cue lists (discourse.json) are
closed-class function words, kept as data and listed in docs/TRAINING_QUEUE.md as candidates for a trained continuation act; they carry the behaviour until that training is run."""
import datetime as dt
import json
import os
from dataclasses import dataclass, field

from .. import analytics as A
from ..query_ir import TimeSpec
from . import literals as L
from .frame import Frame

HERE = os.path.dirname(os.path.abspath(__file__))
DISC = json.load(open(os.path.join(HERE, "discourse.json"), encoding="utf-8"))
DAY = 86400
UNIT = {"hour": 3600, "day": DAY, "week": 7 * DAY, "month": 30 * DAY}
NONEXEC = ("social", "offtopic", "unsupported", "control", "unsafe")
PERIOD_ACTS = ("stat", "extreme", "trend", "compare", "events", "context", "why")
POSITION_CONCEPTS = ("tyre_pressure",)
FILLER = set("the one ones which was is it and of them those these that this a an for me tell show give what about".split())


def as_timespec(lo, hi, now, tz):
    """An absolute window -> a RELATIVE typed period the validator accepts (the validator refuses absolute timestamps on purpose): one local day -> its date; whole days -> days ago; otherwise minutes ago."""
    lo, hi, now, tz = int(lo), int(hi), int(now), int(tz); today = ((now + tz) // DAY) * DAY - tz
    if (lo + tz) % DAY == 0 and (hi + tz) % DAY == 0 and hi > lo and hi <= today + DAY:
        if hi - lo == DAY: return TimeSpec("date", date=dt.datetime.fromtimestamp(lo + tz, dt.timezone.utc).strftime("%Y-%m-%d"))
        v = (today - lo) // DAY; v2 = (today - hi) // DAY + 1
        if hi == today + DAY: v2 = 0
        if 0 <= v2 < v <= 3650: return TimeSpec("between", v, "day", value2=v2)
    for unit, size in (("minute", 60), ("hour", 3600), ("day", DAY)):                                  # the validator allows at most 3650 units
        v = -(-(now - lo) // size); v2 = max(0, (now - hi) // size)
        if v <= 3650: return TimeSpec("between", v, unit, value2=v2) if v2 < v else None
    return None


def _has(low, key): return any(c in low for c in DISC[key])


@dataclass
class Turn:
    text: str
    frame: Frame
    times: list
    window: tuple = None            # (lo, hi) epoch seconds the answer covered
    focus_ts: int = None            # the instant / event the answer was about
    facts: dict = field(default_factory=dict)
    answer: str = ""


@dataclass
class Completion:
    kind: str                       # period | concept | referent | deixis | ordinal | provenance
    frame: Frame = None
    times: list = None
    focus_ts: int = None
    note: str = ""
    position: str = None


class Dialogue:
    def __init__(self): self.turns = []

    def reset(self): self.turns = []

    @property
    def last(self): return self.turns[-1] if self.turns else None

    # ------------------------------------------------------------------------------------------------ recording
    def record(self, brain, text, frame, times, result, focus_ts=None):
        f = result.get("facts") or {}; window = None
        ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}
        pts = ev.get("period_ts") if ev else None
        if pts and len(pts) == 2: window = (int(pts[0]), int(pts[1]))
        elif times:
            try:
                from ..query_planner import resolve_time
                qc = brain.qc()
                if qc is not False: lo, hi, _ = resolve_time(times[0], qc.ctx); window = (int(lo), int(hi))
            except Exception: window = None
        focus = None
        for k in ("event_ts", "ts", "start", "sample_ts", "target_ts", "day"):
            v = f.get(k)
            if isinstance(v, (int, float)) and v > 1e8: focus = int(v); break
        self.turns.append(Turn(text, frame, list(times or []), window, focus or focus_ts, f, str(result.get("answer", ""))))
        del self.turns[:-8]

    # ------------------------------------------------------------------------------------------------ periods relative to the previous answer
    def shifted_period(self, low, prev, brain):
        """"the week before that" / "the previous 3 days" / "the day after" / "two days ago" -> a typed period computed from the previous RESOLVED window (or from today)."""
        rel = L.relative_period(low)
        if rel is None: return None
        kind, n, unit = rel
        if kind == "ago":
            today = ((int(brain.now) + int(brain.tz)) // DAY) * DAY - int(brain.tz); span = UNIT[unit]; hi = today - (n - 1) * span
            return as_timespec(hi - span, hi, brain.now, brain.tz)
        w = prev.window if prev is not None else None
        if w is None: return None
        lo, hi = w; span = n * UNIT[unit]
        if kind == "before": return as_timespec(lo - span, lo, brain.now, brain.tz)
        return as_timespec(hi, min(hi + span, brain.now + 1), brain.now, brain.tz)

    # ------------------------------------------------------------------------------------------------ completion
    def complete(self, brain, text, frame, lit):
        if brain.resolver.onto.version >= 2: return self._complete_ref(brain, text, frame, lit)                  # frame version 2: the model says what the turn takes from the one before; no cue lists are consulted
        prev = self.last
        low = L.norm_words(text); toks = low.split(); onto = brain.resolver.onto
        if _has(low, "provenance") and len(toks) <= 14: return Completion("provenance")          # also with nothing to explain: that is a statement, not a greeting
        if prev is None: return None
        short = len(toks) <= 9; opener = short and any(low == o or low.startswith(o + " ") for o in DISC["ellipsis_openers"])
        shifted = self.shifted_period(low, prev, brain)
        times = [shifted] if shifted is not None else list(lit.times)
        # an ordinal over the events just listed
        pointer = short and _has(low, "pointer")
        ow = L.ordinal_word(low, DISC["ordinal"]); ordinal = DISC["ordinal"][ow] if ow else None
        bare = bool(toks) and set(toks) <= FILLER | {x for w in DISC["ordinal"] for x in w.split()}                 # nothing but the ordinal: "the longest one", "and the latest?"
        if ordinal and prev.frame.act in ("events", "context", "why") and prev.frame.concept in onto.concepts and onto.concepts[prev.frame.concept].kind == "event" and (pointer or bare) and not lit.times:
            return Completion("ordinal", prev.frame, prev.times, note=ordinal)
        # a fragment: the model itself says a slot is missing; a period (or a relative period) was given -> keep the previous frame
        fragment = frame.act == "ambiguous" or (frame.concept == "none" and frame.act in PERIOD_ACTS + ("at", "now"))
        if fragment and times and prev.frame.act in PERIOD_ACTS + ("now", "at") and prev.frame.concept not in ("none", "unlisted"):
            pf = prev.frame
            if pf.act in ("now", "at"): pf = Frame("stat", pf.concept, "mean" if (onto.concepts.get(pf.concept) is not None and onto.concepts[pf.concept].kind in ("signal", "derived")) else "none", pf.cond, pf.vs, pf.ent)
            return Completion("period", pf, times, note="period replaced")
        # an elliptical opener with a concept of its own: keep what the previous question asked about it
        if opener and frame.concept not in ("none", "unlisted") and frame.act not in NONEXEC + ("ambiguous", "define", "spec", "howto") and prev.frame.act in PERIOD_ACTS and prev.frame.concept not in ("none", "unlisted"):
            pf = prev.frame; cand = Frame(pf.act, frame.concept, pf.op, pf.cond, pf.vs, pf.ent)
            if not onto.validate(cand): return Completion("concept", cand, times or prev.times, note=f"{pf.concept} -> {frame.concept}")
        if frame.act == "why" and frame.concept in ("any_warning", "none") and prev.frame.concept not in ("none", "unlisted", "any_warning") and (_has(low, "referent_why") or low == "why"):
            cand = Frame("why", prev.frame.concept, frame.op, frame.cond, frame.vs, frame.ent)
            if not onto.validate(cand):
                anchor = as_timespec(prev.focus_ts - 120, prev.focus_ts + 120, brain.now, brain.tz) if prev.focus_ts and not times else None       # "it" = the event the last answer picked out
                return Completion("referent", cand, times or ([anchor] if anchor else prev.times), prev.focus_ts, note=f"it = {prev.frame.concept}")
        c0 = onto.concepts.get(frame.concept)
        if prev.focus_ts and _has(low, "deixis_time") and not lit.times and c0 is not None and c0.kind in ("signal", "derived", "declared") and frame.act in ("at", "now", "context", "stat", "trend"):
            return Completion("deixis", Frame("at", frame.concept), None, prev.focus_ts, note="then = the instant of the previous answer")
        if frame.act == "at" and prev.focus_ts and not lit.times:
            return Completion("deixis", frame, None, prev.focus_ts, note="the instant of the previous answer")
        pf = prev.frame; pc = onto.concepts.get(pf.concept)
        if lit.position and short and pc is not None and (pf.concept in POSITION_CONCEPTS) and (frame.concept in ("unlisted", "none") or frame.act == "ambiguous"):
            return Completion("position", pf, times or prev.times, prev.focus_ts, note=f"{lit.position} wheel", position=lit.position)
        return None

    # ------------------------------------------------------------------------------------------------ frame version 2: the model's own `ref`
    def _complete_ref(self, brain, text, frame, lit):
        """The model read the question together with `Previous: <frame of the last answered turn>` and returned a whole frame plus `ref`: what this turn takes from the one before (period, concept, focus, item, answer).
        Nothing here reads a cue word.  What is left is carrying TYPED values across: the resolved window of the previous answer (a relative period such as 'the week before that' is computed from it by the literal
        parser), the instant or event it was about, and its list of events."""
        prev = self.last; onto = brain.resolver.onto
        if frame.act == "explain" or frame.ref == "answer": return Completion("provenance")
        if frame.ref == "none" or prev is None: return None
        pf = prev.frame; f = frame
        if f.act in NONEXEC + ("ambiguous",) or f.concept in ("none", "unlisted"):                       # the model asked to inherit but left a slot empty: the previous frame fills it
            f = Frame(pf.act if f.act in NONEXEC + ("ambiguous",) else f.act, pf.concept if f.concept in ("none", "unlisted") else f.concept, f.op if f.op != "none" else pf.op, f.cond if f.cond != "none" else pf.cond,
                      f.vs if f.vs != "none" else pf.vs, f.ent if f.ent != "none" else pf.ent, f.c2, f.ref)
        if f.act in ("now", "at") and f.ref == "period": f = Frame("stat", f.concept, "mean" if onto.concepts.get(f.concept) is not None and onto.concepts[f.concept].kind in ("signal", "derived") else "none", f.cond, f.vs, f.ent, f.c2, f.ref)
        low = L.norm_words(text); shifted = self.shifted_period(low, prev, brain); times = [shifted] if shifted is not None else list(lit.times)
        if f.ref == "period": return Completion("period", f, times or prev.times, note="period replaced")
        if f.ref == "concept": return Completion("concept", f, times or prev.times, note=f"{pf.concept} -> {f.concept}")
        if f.ref == "focus":
            if f.act == "at": return Completion("deixis", Frame("at", f.concept), None, prev.focus_ts, note="the instant of the previous answer") if prev.focus_ts else None
            anchor = as_timespec(prev.focus_ts - 120, prev.focus_ts + 120, brain.now, brain.tz) if prev.focus_ts and not times else None
            return Completion("referent", f, times or ([anchor] if anchor else prev.times), prev.focus_ts, note=f"it = {pf.concept}")
        if f.ref == "item":
            c = onto.concepts.get(pf.concept); word = f.op if f.op in ("max", "min", "first", "last") else "max"
            if c is not None and c.kind == "event": return Completion("ordinal", pf, prev.times, note=word)
        return None

    # ------------------------------------------------------------------------------------------------ ordinal over an event list
    def rank_events(self, brain, comp):
        prev = self.last; onto = brain.resolver.onto; c = onto.concepts[prev.frame.concept]
        from .. import signal_registry as R
        types = list(R.EVENT_GROUPS.get(c.registry_key, (c.registry_key,))) if c.registry_key else []
        if not types or prev.window is None: return None
        lo, hi = prev.window; marks = ",".join("?" * len(types))
        rows = brain.c.execute(f"SELECT ts,type,detail FROM events WHERE type IN ({marks}) AND ts>=? AND ts<? ORDER BY ts", (*types, lo, min(hi, brain.now + 1))).fetchall()
        A.set_tz(brain.tz)
        if not rows: return {"answer": f"No {c.label} is recorded in that period, so there is no \"one\" to pick.", "intent": "QC_EXTREME", "facts": {"ok": False, "route": "dialogue:ordinal", "executor": "dialogue"}}
        word = {"max": "largest", "min": "smallest", "first": "first", "last": "latest"}[comp.note]
        mag = L.magnitude
        if comp.note in ("first", "last"): pick = rows[0] if comp.note == "first" else rows[-1]; why = f"the {word} of {len(rows)}"
        else:
            scored = [(mag(r[2]), r) for r in rows if mag(r[2]) is not None]
            if not scored:
                lines = "; ".join(f"{A.stamp(r[0])} {r[2]}" for r in rows[:5])
                return {"answer": f"These events are instants and the record carries no magnitude to rank them by, so I cannot say which was the {word}. All {len(rows)}: {lines}.", "intent": "QC_EXTREME", "facts": {"ok": False, "route": "dialogue:ordinal", "executor": "dialogue", "n": len(rows)}}
            pick = (max if comp.note == "max" else min)(scored, key=lambda x: x[0])[1]; why = f"the {word} recorded magnitude of {len(scored)}"
        return {"answer": f"The {c.label} with {why}: {A.stamp(pick[0])} - {pick[2]}.", "intent": "QC_EXTREME", "facts": {"ok": True, "route": "dialogue:ordinal", "executor": "dialogue", "event_ts": int(pick[0]), "n": len(rows), "frame": prev.frame.as_dict()}}

    # ------------------------------------------------------------------------------------------------ provenance
    def explain(self, brain):
        prev = self.last
        if prev is None: return {"answer": "There is no earlier answer in this conversation to explain yet. Ask me something and then ask how I got it.", "intent": "NONE", "facts": {"ok": False, "route": "dialogue:provenance", "executor": "dialogue"}}
        from .resolve import describe
        inv = prev.facts.get("investigation")
        if inv:
            ev = prev.facts.get("evidence") or {}; A.set_tz(brain.tz); parts = [f"events {', '.join(inv['events'][:4])}" if inv.get("events") else "", f"signals {', '.join(inv['concepts'][:5])}" if inv.get("concepts") else "", f"windows {', '.join(f'{k} {v:g} s' for k, v in inv['windows'].items())}" if inv.get("windows") else "", f"counts {inv['counts']}" if inv.get("counts") else ""]
            lines = [f"How I got the answer to \"{prev.text[:110]}{'...' if len(prev.text) > 110 else ''}\":", f"- It is a multi-part request, so I did not squeeze it into one question: I split it into its parts and ran the '{inv['template']}' analysis on the recorded history.",
                     "- Parts I recognised: " + ("; ".join(p for p in parts if p) or "none beyond the analysis wanted") + (f"; anchor: {inv['anchor_kind']}" if inv.get("anchor_kind") else "") + "."]
            lines += [f"- Step: {st}" for st in inv.get("steps", [])[:6]]
            if ev.get("period"): lines.append(f"- Period: {ev['period']}.")
            if ev.get("samples") is not None: lines.append(f"- Data: {ev['samples']} records/samples from {', '.join(ev.get('sources') or ['the store'])} ({ev.get('tier', '')}).")
            if inv.get("premise"): lines.append(f"- Premise check: {inv['premise']}")
            lines.append("- Recorded facts, associations (order, lead time, co-occurrence) and hypotheses are kept apart in the answer; nothing was written to the vehicle data.")
            return {"answer": "\n".join(lines), "intent": "EXPLAIN", "facts": {"ok": True, "route": "dialogue:provenance", "executor": "dialogue", "explains": prev.text, "frame": prev.frame.as_dict()}}
        f = prev.facts; ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}; ct = f.get("contract") if isinstance(f.get("contract"), dict) else {}; A.set_tz(brain.tz)
        lines = [f"How I got the answer to \"{prev.text}\":", f"- I read it as: {describe(prev.frame, brain.resolver.onto)}."]
        if f.get("query"): lines.append(f"- I ran the typed query `{f['query']}` on the local store (read-only; I never write to the vehicle data).")
        elif f.get("canonical_request"): lines.append(f"- I answered it with the {f.get('route', 'dedicated')} routine, using the request \"{f['canonical_request']}\".")
        elif f.get("route"): lines.append(f"- Route: {f['route']}.")
        if prev.window: lines.append(f"- Period: {A.stamp(prev.window[0])} to {A.stamp(prev.window[1])}" + (f" ({ev['period']})" if ev.get("period") else "") + ".")
        if ev.get("samples") is not None or ev.get("sources"):
            lines.append(f"- Data: {ev.get('samples', '?')} {'event' if ev.get('tier') == 'events' else 'sample'}{'' if ev.get('samples') == 1 else 's'}" + (f" from {', '.join(ev['sources'])}" if ev.get("sources") else "") + (f", {ev['tier']} resolution" if ev.get("tier") else "") + ".")
        if ct.get("result_class"): lines.append(f"- Evidence class: {ct['result_class']}" + (f"; resolution: {ct['resolution']}" if ct.get("resolution") else "") + (f"; completeness: {ct['completeness']}" if ct.get("completeness") else "") + ".")
        for lim in (ct.get("limits") or [])[:3]: lines.append(f"- Limit: {lim}")
        if f.get("verified") is not None: lines.append(f"- Answer check: {'verified against the same data' if f['verified'] in (True, 'verified', 'ok') or (isinstance(f['verified'], dict) and f['verified'].get('ok')) else 'not verified'}.")
        if len(lines) <= 2: lines.append("- That answer came from a fixed fact or definition, not from a computation over recorded data.")
        return {"answer": "\n".join(lines), "intent": "EXPLAIN", "facts": {"ok": True, "route": "dialogue:provenance", "executor": "dialogue", "explains": prev.text, "frame": prev.frame.as_dict()}}
