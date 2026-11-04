"""Resolver: a validated meaning frame -> an executed, evidence-backed answer.  Deterministic, read-only, no text rules.

Order (first applicable wins; the choice depends only on the frame):
  1. non-executing acts (social / offtopic / unsupported / control / unsafe / ambiguous) and an unmodelled concept -> the typed outcome text (never executes, never reveals data);
  2. typing: the frame must satisfy the ontology (act x concept kind x operator), otherwise a specific clarification;
  3. a frame the typed query engine expresses (now / stat / extreme / trend / compare / events / context / why) -> compile.to_query -> qc / plan_exec;
  4. instant lookups (at), profile facts (spec), definitions (define), methods (howto): small generic executors over ontology / glossary / bindings data;
  5. topic meanings (health / advice / estimate ...) -> the legacy route handlers through a machine-written canonical request (bindings.json), reported as `legacy_adapter`;
  6. otherwise: say what was understood and that no executor exists for it yet (an explicit coverage gap, never a guess)."""
import json
import os
import sqlite3
import statistics

from .. import analytics as A
from ..nearest import nearest
from .. import kinematics
from .. import outcomes
from .. import plan_exec
from .. import signal_registry as R
from ..query_ir import PLAN_OPS, EventRef, TimeSpec
from ..plan_exec import resolve_time
from ..cause import answer as CA
from . import compile as C
from . import digest as DG
from . import eventx as EX
from . import fuelx as FX
from . import evmap as EVMAP
from . import tripx as TRIPX
from . import thresholdx as THRX
from . import pairs as PR
from . import records as RC
from . import glossary as GL
from . import literals as L
from . import ontology as ON

HERE = os.path.dirname(os.path.abspath(__file__))
NONEXEC = {"social": "greeting", "offtopic": "offtopic", "unsupported": "unsupported", "control": "control", "unsafe": "unsafe", "ambiguous": "ambiguous"}
ACCELERATION = ("acceleration", "acceleration_magnitude", "longitudinal_acceleration")
TYPED_ACTS = ("now", "stat", "extreme", "trend", "compare", "events", "context", "why")
STATE_WORD = {"rain": "detected", "adas_state": "engaged"}
POSITIONAL = {"tyre_pressure": {"fl": "tyre_pressure_fl", "fr": "tyre_pressure_fr", "rl": "tyre_pressure_rl", "rr": "tyre_pressure_rr"},
              "wheel_speed": {"fl": "wheel_speed_fl", "fr": "wheel_speed_fr", "rl": "wheel_speed_rl", "rr": "wheel_speed_rr"}}          # one concept, four sensors: the wheel named in the question selects the registry signal
BIND = json.load(open(os.path.join(HERE, "bindings.json"), encoding="utf-8"))
DOCS = json.load(open(os.path.join(HERE, "concept_docs.json"), encoding="utf-8"))["docs"]


def _result(answer, route, executor, frame, intent="MEANING", **facts):
    return {"answer": answer, "brief": answer.split("\n", 1)[0], "intent": intent, "facts": dict({"ok": True, "route": route, "executor": executor, "frame": frame.as_dict()}, **facts)}


def describe(frame, onto):
    """A short plain-words reading of a frame, for confirmations: 'a question about battery temperature (average)'."""
    c = onto.concepts.get(frame.concept); what = c.label if c else frame.concept
    kind = onto.acts[frame.act]["gloss"]
    extra = [x for x in (frame.op if frame.op != "none" else "", frame.cond if frame.cond != "none" else "") if x]
    return f"{kind}: {what}" + (f" ({', '.join(extra)})" if extra else "")


ECHOED = {"control", "unsupported", "ambiguous", "offtopic"}                                   # the non-answers that should show which question they are about (not `unsafe`: it is never repeated back; not greetings)


def _about(text):
    """'About "unlock the doors": ' - the first words of what was asked, so that a refusal or a request for more detail is visibly about THIS question and not the same sentence for every question.  The text was cleaned
    of control characters and markup before it got here; it is shortened and shown as quoted text only."""
    t = " ".join(str(text or "").replace('"', "'").split())
    if not t: return ""
    return 'About "' + (t if len(t) <= 64 else t[:61].rsplit(" ", 1)[0] + "...") + '": '


def FXX_is_ice(res):
    from . import fuelx as _F
    try: return _F.CAPS.is_combustion(_F.caps_of(res))
    except Exception: return False


class Resolver:
    def __init__(self, brain):
        self.brain = brain; self.onto = ON.load(); self.glossary = GL.load()

    # ------------------------------------------------------------------------------------------------ entry point
    def onto_validate(self, frame):
        """[] when the ontology allows this combination of act, concept and operation."""
        return self.onto.validate(frame)

    def answer(self, text, frame, times=None, focus_ts=None, position=None):
        frame = EVMAP.guard(text, frame, self.onto)                                                    # consumption said, efficiency read: the quantity the words name (evmap.py)
        r = self._answer_inner(text, frame, times, focus_ts, position)
        r = EVMAP.restate(text, frame, r)
        from . import plain as _PL
        return _PL.change_answer(frame, r)

    def _answer_inner(self, text, frame, times=None, focus_ts=None, position=None):
        calx_tail = None
        if frame.act not in NONEXEC and frame.concept not in ("none", "unlisted"):                                              # weekends against weekdays, one line per day / week / month, the pattern in when events happen (vgpt/meaning/calx.py)
            kinds_, per_, pat_ = L.day_grouping(text)
            if kinds_ or per_ or pat_:
                try:
                    from . import calx as CX
                    qc_ = self.brain.qc(); c_ = self.onto.concepts.get(frame.concept)
                    if c_ is not None:
                        r0 = CX.maybe(self, text, frame, self._lit(text, qc_.profile if qc_ is not False else None, c_.id), kinds_, per_, pat_)
                        if r0 is not None and frame.act == "trend" and per_ and not kinds_ and not pat_: calx_tail = r0.get("answer")                  # a trend question: the verdict first, the months after it
                        elif r0 is not None: return r0
                except Exception as e:
                    import logging; logging.getLogger("vehiclegpt").warning("calendar breakdown failed: %s: %s", type(e).__name__, e)
        if frame.act == "trend" and frame.concept == "trip_record" and frame.op in ("rise", "drop", "change"):                      # trips per day, from the trip ledger
            try:
                from . import trendv as TV
                c = self.onto.concepts.get(frame.concept); qc = self.brain.qc(); r0 = TV.trip_trend(self, text, frame, c, self._lit(text, qc.profile if qc is not False else None, c.id))
                if r0 is not None: return r0
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("trip trend failed: %s: %s", type(e).__name__, e)
        if frame.act in ("anomaly", "health", "advice") and frame.c2 not in ("none", frame.concept) and frame.c2 in self.onto.concepts and frame.concept in self.onto.concepts and self.onto.concepts[frame.concept].kind == "topic" and self.onto.concepts[frame.c2].kind == "topic":
            # "anything odd with the tyres or the battery": two components asked in one sentence - each is answered by its own executor and the two answers are given one after the other
            r1 = self._answer_core(text, type(frame)(frame.act, frame.concept, frame.op, frame.cond, frame.vs, frame.ent, "none", frame.ref), times, focus_ts, position)
            r2 = self._answer_core(text, type(frame)(frame.act, frame.c2, frame.op, frame.cond, frame.vs, frame.ent, "none", frame.ref), times, focus_ts, position)
            if isinstance(r1, dict) and isinstance(r2, dict) and isinstance(r1.get("answer"), str) and isinstance(r2.get("answer"), str):
                return dict(r1, answer=r1["answer"] + chr(10) + chr(10) + r2["answer"], facts=dict(r1.get("facts") or {}, second=r2.get("facts")))
        r = self._answer_core(text, frame, times, focus_ts, position)
        if frame.act == "compare" and frame.cond == "cold" and frame.vs == "hot_weather" and isinstance(r, dict) and (r.get("facts") or {}).get("status") == "no_data" and not times:          # no cold weather on record (a hot-climate car): winter months against summer months
            try:
                import datetime as _dt
                from ..query_ir import TimeSpec as _TS
                nd = _dt.datetime.fromtimestamp(self.brain.now + int(self.brain.tz), _dt.timezone.utc); wy = nd.year if nd.month >= 3 else nd.year - 1; sy = nd.year if nd.month >= 9 else nd.year - 1
                r2 = self._answer_core(text, type(frame)("compare", frame.concept, frame.op, "none", "period", frame.ent, frame.c2, frame.ref), [_TS("calendar", name=f"s{wy - 1}_12_{wy}_02"), _TS("calendar", name=f"s{sy}_06_{sy}_08")], focus_ts, position)
                if isinstance(r2, dict) and (r2.get("facts") or {}).get("status") != "no_data" and isinstance(r2.get("answer"), str):
                    r = dict(r2, answer="No recorded time was cold enough to count as cold, so I compared the winter months (December to February) with the summer months (June to August):\n" + r2["answer"])
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("season fallback failed: %s: %s", type(e).__name__, e)
        if frame.act == "trend" and frame.op in ("rise", "drop", "change") and isinstance(r, dict) and str((r.get("facts") or {}).get("route") or "").startswith("qc:"):          # a numeric trend: the asked direction is answered yes / no first (vgpt/meaning/trendv.py)
            try:
                from . import trendv as TV
                c = self.onto.concepts.get(frame.concept); qc = self.brain.qc()
                if c is not None: r = TV.add_verdict(self, text, frame, r, c, self._lit(text, qc.profile if qc is not False else None, c.id))
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("trend verdict failed: %s: %s", type(e).__name__, e)
        if isinstance(r, dict) and isinstance(r.get("answer"), str) and " for from " in r["answer"]: r = dict(r, answer=r["answer"].replace(" for from ", " for "))      # "for from June to August": a season is a period, not a word that follows "for"
        if calx_tail and isinstance(r, dict) and isinstance(r.get("answer"), str): r = dict(r, answer=r["answer"] + chr(10) + chr(10) + calx_tail)
        return r

    def _answer_core(self, text, frame, times=None, focus_ts=None, position=None):
        b, O = self.brain, self.onto
        self._times = list(times) if times else None; self._focus = focus_ts; self._position = position                 # a period / instant inherited from the previous turn (dialogue.py)
        if frame.act in NONEXEC:
            msg, kind = outcomes.render(NONEXEC[frame.act], 2)
            if frame.act in ECHOED: msg = _about(text) + msg
            return _result(msg, f"outcome:{frame.act}", "outcome", frame, intent="NONE" if frame.act != "social" else "SMALLTALK", outcome=kind)
        if frame.concept == "rated_max_speed" and frame.act in ("define", "spec", "now", "at", "stat", "extreme", "howto", "trend", "compare"): return self._max_speed(frame)       # "what is the max speed": both readings of it, said plainly
        if frame.concept in EVMAP.BATTERY_TO_FUEL:                                                           # "state of charge" asked of a petrol / diesel car: the fuel level (evmap.py)
            try: m_ = EVMAP.maybe_ice(self, frame)
            except Exception: m_ = None
            if m_ is not None:
                r_ = self._answer_core(text, m_[0], times, focus_ts, position)
                if isinstance(r_, dict) and isinstance(r_.get("answer"), str): r_ = dict(r_, answer=m_[1] + chr(10) + r_["answer"], brief=(m_[1] + " " + str(r_.get("brief") or "")).strip())
                return r_
        if frame.concept not in ("none", "unlisted") and frame.concept in EVMAP.FUEL_TO_BATTERY:             # "fuel" asked of an electric car: the battery quantity that does the same job (evmap.py)
            try: m_ = EVMAP.maybe(self, frame)
            except Exception: m_ = None
            if m_ is not None:
                r_ = self._answer_core(text, m_[0], times, focus_ts, position)
                if isinstance(r_, dict) and isinstance(r_.get("answer"), str): r_ = dict(r_, answer=m_[1] + chr(10) + r_["answer"], brief=(m_[1] + " " + str(r_.get("brief") or "")).strip())
                return r_
        if frame.concept not in ("none", "unlisted") and frame.act not in NONEXEC and frame.concept in O.concepts:                    # the start and end of a trip, the last N trips (tripx.py)
            try:
                r_t = TRIPX.maybe(self, text, frame, self._lit(text), O.concepts[frame.concept])
                if r_t is not None: return r_t
                r_t = THRX.maybe(self, text, frame, self._lit(text), O.concepts[frame.concept])
                if r_t is not None: return r_t
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("trip executor failed: %s: %s", type(e).__name__, e)
        if frame.concept not in ("none", "unlisted"):                                                    # a combustion vehicle (diesel / petrol, decided by its data): energy left, consumption, fill-ups, engine health in fuel terms
            try: fr = FX.maybe(self, text, frame, self._lit(text))
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("combustion executor failed: %s: %s", type(e).__name__, e); fr = None
            if fr is not None: return fr
        if frame.act == "spec" and frame.concept == "vehicle_model" and frame.concept in O.concepts: return FX.identity(self, self._spec(O.concepts[frame.concept], frame))               # the name, and what the data says the vehicle is
        if frame.act in ("define", "howto") and frame.concept in getattr(O, "absent", ()):             # what a state of health IS does not depend on the vehicle: the definition is given, and so is the fact that this vehicle has none
            return self._general(frame)
        if frame.concept not in ("none", "unlisted") and frame.concept not in O.concepts:           # a concept this vehicle does not have at all (a diesel has no state of charge)
            msg, kind = outcomes.render("unrecorded", 2)
            return _result(_about(text) + msg + " This vehicle does not have that system. Nothing was substituted for it.", "outcome:unrecorded", "outcome", frame, intent="NONE", outcome=kind, ok=False)
        if frame.concept == "unlisted":
            msg, kind = outcomes.render("unrecorded", 2)
            return _result(_about(text) + msg + " Nothing was substituted for it.", "outcome:unrecorded", "outcome", frame, intent="NONE", outcome=kind, ok=False)
        c0 = O.concepts.get(frame.concept)
        if frame.act == "at" and c0 is not None and c0.kind == "spec":                  # "which firmware was installed when ...": a fixed attribute has no history here; give the profile value and say what it is
            r = dict(self._spec(c0, frame)); r["answer"] += " The vehicle profile holds the current value only; it does not record what it was at an earlier time."; r["brief"] = r["answer"].split("\n", 1)[0]
            return r
        if frame.act == "at" and c0 is not None and c0.id in ("wh_per_km", "energy_used", "distance_driven", "regen_energy"):          # a ratio / accumulated quantity has no value at one instant: it is read over the period named ("yesterday's energy per km")
            return self.answer(text, type(frame)("stat", frame.concept, "mean" if c0.id == "wh_per_km" else "sum", frame.cond, frame.vs, frame.ent))
        if frame.act == "trend" and c0 is not None and c0.kind == "topic" and f"trend:{c0.id}" in BIND["legacy"]:
            return self._legacy(BIND["legacy"][f"trend:{c0.id}"], self._lit(text), frame)
        issues = O.validate(frame)
        if issues:
            return _result("I read this as " + describe(frame, O) + ", but I cannot answer that kind of question about that quantity. Tell me which measurement and over what period you mean (for example 'average battery temperature last week').", "clarify:typing", "clarify", frame, intent="CLARIFY", ok=False, issues=issues)
        c = O.concepts[frame.concept]; qc = b.qc(); lit = self._lit(text, qc.profile if qc is not False else None, c.id)
        if frame.act == "why":                                                                                           # the cause of a change or of an event: the cause engine (vgpt/cause), falling back to the older routes below
            parked = c.id in ("battery_soc", "fuel_level") and (frame.cond in ("parked", "night") or any(getattr(t, "name", None) in ("last_night", "tonight") for t in lit.times))          # the energy lost while standing is the digest's parked stretch, not a cause hunt
            r = None if parked else CA.maybe(self, c, frame, lit, text)
            if r is not None:
                if frame.ent == "none" and (r.get("facts") or {}).get("route") == "cause:change":                                   # "why is it so high": also say where the quantity stands against its own usual range
                    try: line = DG.status_line(self, c, frame, lit, text)
                    except Exception: line = None
                    if line and line not in r["answer"]: head, _, rest = r["answer"].partition("\n"); r["answer"] = head + "\n" + line + ("\n" + rest if rest else "")
                return r
        if frame.act in ("summary", "anomaly") or (frame.act == "stat" and frame.op == "usual") or (frame.act == "why" and (frame.ent in ("trip", "day") or (c.id in ("battery_soc", "fuel_level") and (frame.cond in ("parked", "night") or any(getattr(t, "name", None) in ("last_night", "tonight") for t in lit.times))))):                          # digests: a period summary, what changed, what is unusual, how a quantity usually behaves, why one trip stood out
            r = DG.maybe(self, c, frame, lit, text)
            if r is not None: return r
        if frame.act == "compare" and c.id == "trip_record" and frame.vs == "period":                                       # the number of trips in two periods: two counts from the ledger, side by side
            r = self._compare_trips(frame, lit, qc)
            if r is not None: return r
        if frame.c2 != "none" or frame.vs == "typical" or frame.op == "corr":                                              # frame version 2: a second quantity, or the vehicle's own history as the yardstick
            pr = PR.maybe(self, c, frame, lit)
            if pr is not None: return pr
        if frame.act == "extreme" and c.id == "tyre_pressure" and frame.op in ("min", "max"):                           # which wheel: from the per-wheel minute values (vgpt/meaning/careadv.py)
            try:
                from . import careadv as CV
                r = CV.tyre_extreme(self, frame, lit)
                if r is not None: return r
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("tyre extreme failed: %s: %s", type(e).__name__, e)
        if frame.act == "advice" and c.id == "battery_condition":                                                           # battery care from the vehicle's own habits (vgpt/meaning/careadv.py); a combustion vehicle has none
            try:
                from . import careadv as CV, fuelx as FXX
                if not FXX.CAPS.is_combustion(FXX.caps_of(self)):
                    r = CV.battery_care(self, frame, lit)
                    if r is not None: return r
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("battery care failed: %s: %s", type(e).__name__, e)
        key, key2 = f"{frame.act}:{frame.concept}", f"{frame.act}:{frame.concept}:{frame.ent}:{frame.op}"
        if key2 in BIND["legacy"]: return self._legacy(BIND["legacy"][key2], lit, frame)
        if frame.act == "now" and key in BIND["legacy"]: return self._legacy(BIND["legacy"][key], lit, frame)         # a live quantity the typed engine has no reading for but a dedicated routine answers
        if frame.act == "extreme" and frame.ent in ("day", "week") and c.id in ("wh_per_km", "energy_used", "distance_driven") and not FXX_is_ice(self):                              # the trip ledger, per day or week (vgpt/meaning/binx.py)
            from . import binx as BX
            r = BX.extreme_by_bin(self, c, frame, lit)
            if r is not None: return r
        if frame.act == "stat" and frame.ent in ("day", "week") and c.id in ("wh_per_km", "energy_used", "distance_driven") and not FXX_is_ice(self) and frame.cond == "none":              # one line per day / week (binx.series_by_bin)
            from . import binx as BX
            r = BX.series_by_bin(self, c, frame, lit)
            if r is not None: return r
        if frame.act == "extreme" and frame.ent == "day" and c.kind != "event": return self._by_day(c, frame, lit)
        if frame.act == "stat" and frame.cond == "threshold" and c.id in ACCELERATION: return self._accel_band(text, c, lit, frame)
        if frame.act == "stat" and c.id == "trip_record" and not (frame.op == "count" and frame.ent == "trip"):
            r = self._trips(c, frame, lit)
            if r is not None: return r
        if frame.act == "events" and c.id == "trip_record": return self._trip_events(c, frame, lit)
        if c.id == "service_log" and frame.act not in ("define", "howto"): return EX.log_search(self, c, frame, lit, text)                  # the vehicle's text logs: searched by words / source, listed, counted
        if c.id == "diagnostic_status" and frame.act in ("health", "advice"): return RC.dtc_status(self, c, frame, lit)                  # which codes are active, cleared, came back
        if frame.act == "spec" and c.kind in ("signal", "derived", "declared"): return RC.signal_spec(self, c, frame, lit)               # where a measurement comes from
        if frame.act == "stat" and c.kind == "event": return RC.event_stat(self, c, frame, lit)
        if c.id == "charging_session" and (frame.ent == "place" or (frame.act == "events" and frame.ent not in ("day", "week"))): return RC.charge_events(self, c, frame, lit)          # the charging ledger
        if c.kind == "event" and frame.ent == "source" and frame.act in ("events", "extreme"): return RC.by_source(self, c, frame, lit)
        if frame.act == "events" and c.id == "diagnostic_code" and (lit.codes or lit.code_classes): return RC.code_history(self, c, frame, lit)                  # a code typed in the question: its own history
        if c.kind == "event" and c.id != "service_log":                                                                                  # how often: trends, comparisons, per-day / per-week counts of an event
            if frame.act == "trend": return EX.event_trend(self, c, frame, lit)
            if frame.act == "compare": return EX.event_compare(self, c, frame, lit)
            if frame.act == "extreme" or (frame.act == "events" and frame.ent in ("day", "week") and frame.op in ("count", "exists", "list")): return EX.event_by_bin(self, c, frame, lit)
        why = None
        if frame.act in TYPED_ACTS and c.id == "wheel_speed" and c.kind == "signal" and not (getattr(self, "_position", None) or lit.position) and qc is not False and any(qc.profile.concept_available(w) for w in POSITIONAL["wheel_speed"].values()):                  # four sensors, none named: ask which, never pick one
            return _result("Which wheel do you mean - front left, front right, rear left or rear right?", "clarify:wheel", "clarify", frame, intent="CLARIFY", ok=False)
        if frame.act in TYPED_ACTS:
            q, why = C.to_query(frame, lit, O)
            if q is not None:
                pos = getattr(self, "_position", None) or lit.position
                if pos and c.id in POSITIONAL and q.target == c.id: q.target = POSITIONAL[c.id][pos]
                if frame.act == "now" and c.id == "tyre_pressure" and not pos and all(t.type == "current" for t in (self._times or lit.times or [])):             # "what is the tyre pressure": all four wheels, not only the lowest
                    r = self._tyres_now(frame)
                    if r is not None: return r
                r = self._typed(q, text, frame)
                if frame.act == "why" and frame.ent == "none" and r.get("facts", {}).get("ok", True):                                  # "why is it so high": say where the quantity stands against its own usual range
                    try: line = DG.status_line(self, c, frame, lit, text)
                    except Exception: line = None
                    if line and line not in r["answer"]: head, _, rest = r["answer"].partition("\n"); r["answer"] = head + "\n" + line + ("\n" + rest if rest else "")
                return r
        if frame.act == "at": return self._at(text, c, frame)
        if frame.act == "spec": return self._spec(c, frame)
        if frame.act == "define" and lit.codes: return self._legacy(f"what is {lit.codes[0]}", lit, frame)                  # a typed fault code is itself the thing being defined
        if frame.act == "define": return self._define(c, frame)
        if frame.act == "howto": return self._howto(c, frame, text)
        if key in BIND["legacy"]: return self._legacy(BIND["legacy"][key], lit, frame)
        if frame.act in ("health", "advice", "estimate") and c.kind in ("signal", "derived"):
            q, why = C.to_query(type(frame)("now", frame.concept), lit, O)
            if q is not None: return self._typed(q, text, frame)
        why = why or "no executor is wired for this meaning yet"
        return _result(f"I understood this as {describe(frame, O)}, but {why}, so I will not guess an answer.", "gap:" + key, "none", frame, intent="CLARIFY", ok=False, coverage_gap=key)

    def _compare_trips(self, frame, lit, qc):
        from ..query_ir import EntitySelect, Query, Scope
        ts = list(lit.times)
        if len(ts) == 1 and ts[0].type == "calendar" and C.PREVIOUS_CALENDAR.get(ts[0].name): ts = [ts[0], TimeSpec("calendar", name=C.PREVIOUS_CALENDAR[ts[0].name])]
        elif len(ts) == 1 and ts[0].type == "relative": ts = [ts[0], TimeSpec("previous", ts[0].value, ts[0].unit)]
        if len(ts) < 2: return None
        got = []
        for t in ts[:2]:
            res = qc.execute(Query(op="count", scope=Scope(t, EntitySelect("trip", None, None, "desc", None))))
            if res.status != "answered" or not isinstance(getattr(res.result, "value", None), (int, float)): return None
            got.append((int(res.result.value), res.result.period_label))
        bare = lambda x: x[3:] if str(x).startswith("in ") else x
        (n1, l1), (n2, l2) = got; l1, l2 = bare(l1), bare(l2)
        diff = n1 - n2; word = "more" if diff > 0 else "fewer"
        txt = f"{n1} trips for {l1} against {n2} for {l2}" + (f": {abs(diff)} {word} ({abs(100.0 * diff / n2):.1f}%)." if diff and n2 else ".")
        return _result(txt + "\nBased on the trip ledger.\nEvidence: measured, one record per trip.", "compare:trip_record", "trip_ledger", frame, counts=[n1, n2])

    def _lit(self, text, profile=None, target=None):
        lit = L.extract(text, self.brain.now, self.brain.tz, profile, target)
        if getattr(self, "_times", None): lit.times = list(self._times)
        else:
            try:
                from . import anchor as _AN
                a_ = _AN.service_periods(self.brain, text)                                                       # "before and after my last service": the periods are named by the vehicle's own service record
                if a_: lit.times = a_
            except Exception: pass
        return lit

    # ------------------------------------------------------------------------------------------------ executors
    def _typed(self, q, text, frame):
        b = self.brain; qc = b.qc()
        if q.op in PLAN_OPS:
            r = plan_exec.execute(b, qc, q)
            if r is not None:
                f = dict(r.get("facts") or {}); f.update(executor="typed_query", frame=frame.as_dict(), query=q.to_text()); r = dict(r); r["facts"] = f; 
                if q.target in STATE_WORD: r["answer"] = self._state_words(q.target, str(r.get("answer"))); r["brief"] = self._state_words(q.target, str(r.get("brief") or ""))
                return r
        res = qc.execute(q, text=text)
        facts = res.facts() if hasattr(res, "facts") else {}
        facts = dict(facts, executor="typed_query", frame=frame.as_dict(), query=q.to_text(), route="qc:" + q.op)
        if q.op in ("rank", "extreme", "list_events"):                                                           # an answer about ONE moment ("the longest trip was on 17 March"): the next question ("which year?", "what happened then?") points back to it
            try:
                rr = getattr(res, "result", None); its = list(getattr(rr, "items", None) or []); t0 = getattr(rr, "when", None) or (its[0].get("start") or its[0].get("ts") if its and isinstance(its[0], dict) else None)
                if isinstance(t0, (int, float)) and t0 > 1e8 and (len(its) <= 1 or q.op != "list_events"): facts["event_ts"] = int(t0)
            except Exception: pass
        if res.status == "invalid":
            return {"answer": res.answer.text, "brief": res.answer.text, "intent": "CLARIFY", "facts": dict(facts, ok=False)}
        if q.target in STATE_WORD: return {"answer": self._state_words(q.target, res.answer.text), "brief": self._state_words(q.target, getattr(res.answer, "brief", None) or ""), "intent": "QC_" + q.op.upper(), "facts": facts, "status": res.status}
        return {"answer": res.answer.text, "brief": getattr(res.answer, "brief", None), "intent": "QC_" + q.op.upper(), "facts": facts, "status": res.status}

    def _tyres_now(self, frame):
        from ..query_ir import Query, Scope, TimeSpec
        qc = self.brain.qc(); got = []
        for label, key in (("front left", "tyre_pressure_fl"), ("front right", "tyre_pressure_fr"), ("rear left", "tyre_pressure_rl"), ("rear right", "tyre_pressure_rr")):
            res = qc.execute(Query(op="current_value", target=key, scope=Scope(TimeSpec("current"))))
            if res.status == "answered" and isinstance(getattr(res.result, "value", None), (int, float)): got.append((label, float(res.result.value)))
        bad = [(k, v) for k, v in got if not (0.0 < v <= 2000.0)]; got = [(k, v) for k, v in got if (0.0 < v <= 2000.0)]                    # a pressure of zero or less (or of two thousand kPa) is a faulty reading, never a pressure
        if len(got) < 2: return None
        spec = self._meta(["veh.tyre_spec_front"])[1]; low = min(got, key=lambda x: x[1])
        txt = "Tyre pressures right now: " + ", ".join(f"{k} {v:.0f}" for k, v in got) + " kPa" + (f" (the recorded specification is {float(spec):g} kPa cold)" if spec is not None else "") + f".\nLowest: {low[0]}, {low[1]:.0f} kPa."
        if bad: txt += chr(10) + " and ".join(f"The {k} tyre sensor reads {v:.0f} kPa" for k, v in bad) + ", which a tyre cannot have: it is a faulty reading, so that tyre is not covered here."
        try: dev = (low[1] - float(spec)) / float(spec) * 100.0 if spec else 0.0
        except (TypeError, ValueError, ZeroDivisionError): dev = 0.0
        if dev <= -8.0: txt += f"\nThe {low[0]} tyre is low: {-dev:.0f} % below the {float(spec):g} kPa specification" + (" - a critical level, do not keep driving on it." if dev <= -15.0 else ".")
        try:
            from .. import analytics_x as AX
            act = AX.tyres(self.brain.c, self.brain.now, "14").get("active_warnings") or []
            if act: txt += "\nA tyre-pressure warning is active for the " + " and ".join(str(a) for a in act) + " tyre."
        except Exception: pass
        txt += "\nBased on the current reading of each wheel.\nEvidence: measured."
        return _result(txt, "now:tyre_pressure", "instant", frame, wheels={k: v for k, v in got})

    @staticmethod
    def _state_words(target, text):
        """An on / off state is 'above 0.5' only inside the engine: say what the state means."""
        for unit in ("none", "category"): text = text.replace(f"above 0.5 {unit}", STATE_WORD[target])
        return text

    def _legacy(self, phrase, lit, frame):
        from .. import learned_front
        nums = lit.numbers; v = {"pct": (nums[0] if nums else None), "n": (nums[0] if nums else None), "v1": (nums[0] if nums else None), "v2": (nums[1] if len(nums) > 1 else None)}
        for k, d in BIND["legacy_defaults"].items():
            if v.get(k) is None: v[k] = d
        need = [k for k in ("pct", "n", "v1", "v2") if "{" + k + "}" in phrase and v.get(k) is None]
        if need: return _result("I need a number for that (" + ", ".join(need) + "), for example 'a 60 km trip'.", "clarify:literal", "clarify", frame, intent="CLARIFY", ok=False)
        phrase = phrase.format(**{k: (f"{x:g}" if isinstance(x, float) else x) for k, x in v.items()})
        b = self.brain; b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
        try: b.qa.ctx.update(topic=None, diag=None, event_row=None)
        except Exception: pass
        try:
            with learned_front.legacy_allowed():
                r = b._ask_routes(phrase, use_slm=False)
        except Exception as e:                                                         # an executor that cannot run on this vehicle's data says so; it never takes the assistant down
            import logging; logging.getLogger("vehiclegpt").warning("legacy adapter %r failed: %s: %s", phrase, type(e).__name__, e)
            return _result("I understood the question, but the routine that answers it could not run on this vehicle's data (" + type(e).__name__ + "), so I will not guess.", "error:legacy_adapter", "legacy_adapter", frame, intent="CLARIFY", ok=False, error=type(e).__name__)
        r = dict(r); f = dict(r.get("facts") or {}); f.update(executor="legacy_adapter", canonical_request=phrase, frame=frame.as_dict()); r["facts"] = f
        return r

    def _meta(self, keys):
        for k in keys:
            row = self.brain.c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
            if row is not None and str(row[0]).strip() != "": return k, row[0]
        return None, None

    def _spec(self, c, frame):
        label = c.label
        if c.id == "tyre_spec":
            f = self._meta(["veh.tyre_spec_front"])[1]; r = self._meta(["veh.tyre_spec_rear"])[1]
            if f is not None or r is not None: return _result(f"The recorded tyre pressure specification is {f} kPa front and {r} kPa rear.", "spec:tyre_spec", "spec", frame, concept=c.id)
        else:
            k, v = self._meta(c.meta)
            if v is not None:
                unit = "" if c.unit in ("none", "count") else " " + c.unit
                try: v = f"{float(v):g}"
                except (TypeError, ValueError): pass
                note = BIND.get("spec_notes", {}).get(c.id); extra = note["text"] if note and self._meta(note["if_missing_meta"])[1] is None else ""
                return _result(f"The vehicle profile lists {label}: {v}{unit}.{extra}", f"spec:{c.id}", "spec", frame, concept=c.id, value=v, source=k)
        msg = f"The vehicle profile does not contain a verified {label}, so I cannot state it, and I will not substitute a different quantity for it."
        obs = BIND["spec_observed"].get(c.id)
        if obs:
            from ..query_ir import Query, Scope, TimeSpec
            res = self.brain.qc().execute(Query(op="extreme", target=obs["concept"], agg=obs["agg"], scope=Scope(TimeSpec("all"))))
            if res.status == "answered": msg += f" The {obs['label']}, which is {obs['note']}, is: " + res.answer.text.split("\n")[0]
        return _result(msg, f"spec:{c.id}", "spec", frame, ok=False, concept=c.id, available=False)

    def _max_speed(self, frame):
        """Top speed has two meanings and a person usually wants both: what the maker rates the car for (the vehicle profile) and the fastest it was actually recorded going."""
        k, v = self._meta(["veh.rated_max_speed", "veh.max_speed_kmh"]); lines = []; got = None
        if v is not None:
            try: v = f"{float(v):g}"
            except (TypeError, ValueError): pass
            lines.append(f"Rated top speed (vehicle profile): {v} km/h.")
        else: lines.append("The vehicle profile does not list a rated top speed.")
        got = None
        try:
            from ..query_ir import Query, Scope, TimeSpec
            res = self.brain.qc().execute(Query(op="extreme", target="speed", agg="max", scope=Scope(TimeSpec("all"))))
            if res.status == "answered": got = res.answer.text.split(chr(10))[0]
        except Exception: got = None
        lines.append(("Fastest recorded: " + got[0].lower() + got[1:]) if got else "No speed reading is recorded yet.")
        return _result(chr(10).join(lines), "spec:rated_max_speed", "spec", frame, rated=v, observed=got is not None)

    def _reading(self, c, frame):
        """One line with this vehicle's present value of a measured concept (or a count of an event type over 30 days), or None when the vehicle does not record it / has no value."""
        b = self.brain; qc = b.qc()
        if qc is False: return None
        from ..query_ir import Query, Scope, TimeSpec
        if c.kind == "event":
            if not qc.profile.has_events(c.registry_key): return None
            res = qc.execute(Query(op="count", scope=Scope(TimeSpec("relative", 30, "day")), event=EventRef(c.registry_key, "last")))
            return ("Over the last 30 days: " + res.answer.text.split("\n")[0]) if res.status == "answered" else None
        if c.kind not in ("signal", "derived", "declared") or not qc.profile.concept_available(c.id): return None
        if c.id == "tyre_pressure":
            names = (("front left", "tyre_pressure_fl"), ("front right", "tyre_pressure_fr"), ("rear left", "tyre_pressure_rl"), ("rear right", "tyre_pressure_rr")); parts = []
            for label, key in names:
                res = qc.execute(Query(op="current_value", target=key, scope=Scope(TimeSpec("current"))))
                if res.status == "answered" and isinstance(getattr(res.result, "value", None), (int, float)): parts.append(f"{label} {res.result.value:.0f}")
            spec = self._meta(["veh.tyre_spec_front"])[1]
            if parts: return "Right now: " + ", ".join(parts) + " kPa" + (f" (the recorded specification is {float(spec):g} kPa)." if spec is not None else ".")
        q, _ = C.to_query(type(frame)("now", c.id), L.Literals([], [], [], ""), self.onto)
        if q is None: return None
        res = qc.execute(q)
        return res.answer.text.split("\n")[0] if res.status == "answered" else None

    def _general(self, frame):
        """A definition / method of a quantity the open vehicle does not have (a diesel: state of health, charging power, coolant flow): the general text, plus one plain sentence that this vehicle has none."""
        cid = frame.concept; label = cid.replace("_", " "); note = " (This vehicle does not have or record it.)"
        if frame.act == "howto":
            text = BIND["formulas"].get(cid)
            if text is None: return _result(f"I do not have a documented method for {label}, so I will not invent a formula." + note, f"howto:{cid}", "formula", frame, ok=False)
            return _result(text + note, f"howto:{cid}", "formula", frame, concept=cid)
        doc = DOCS.get(cid)
        if not doc: return _result(f"I do not have a definition recorded for {label}." + note, f"define:{cid}", "glossary", frame, ok=False)
        return _result(doc + note, f"define:{cid}", "glossary", frame, concept=cid, recorded=False)

    def _define(self, c, frame):
        doc = DOCS.get(c.id)
        if doc:                                                                           # a written definition of the quantity itself; the glossary's one-liners are word meanings, used only when there is none
            text = doc if doc.lower().startswith(c.label.lower()) else f"{c.label[0].upper() + c.label[1:]}: {doc}"
            extra = [e["meaning"].strip() for e in self.glossary.entries if e.get("concept") == c.id and e.get("kind") in ("term", "slang", "abbrev") and e.get("define", True)]
            if extra:
                have = set(doc.lower().split()); m = extra[0]
                if len(set(m.lower().split()) - have) >= 4: text += " In plain words: " + m[0].lower() + m[1:].rstrip(".") + "."
            qc = self.brain.qc()
            if c.kind in ("signal", "derived", "declared") and qc is not False and not qc.profile.concept_available(c.id):
                return _result(text + (" (This vehicle does not record the signals needed to compute it.)" if c.kind == "derived" else " (This vehicle does not record it.)"), f"define:{c.id}", "glossary", frame, concept=c.id, recorded=False)
            reading = self._reading(c, frame)
            return _result(text + (("\n" + reading) if reading else ""), f"define:{c.id}", "glossary", frame, concept=c.id, recorded=reading is not None)
        rank = {"term": 0, "abbrev": 1, "slang": 2, "unit": 3, "operator": 4}
        ents = sorted([e for e in self.glossary.entries if e.get("concept") == c.id], key=lambda e: rank.get(e.get("kind"), 9))[:1]
        body = (ents[0]["meaning"] if ents else c.gloss or "").strip()
        if not body: return _result(f"I do not have a definition recorded for {c.label}.", f"define:{c.id}", "glossary", frame, ok=False)
        body = body[0].upper() + body[1:] + ("" if body.endswith(".") else ".")
        text = body if ":" in body.split(" ")[0:4].__str__() or ents and ents[0]["term"].lower() in body.lower()[:len(ents[0]["term"]) + 12] else f"{c.label[0].upper() + c.label[1:]}: {body}"
        return _result(text, f"define:{c.id}", "glossary", frame, concept=c.id)

    def _howto(self, c, frame, question=""):
        text = BIND["formulas"].get(c.id)
        if c.id == "battery_capacity":                                                  # a typed pair of values makes the formula a calculation
            ah, volts = L.amp_hours_and_volts(question)
            if ah is not None and volts is not None: return _result(f"{ah * 1000:g} mAh at {volts:g} V is {ah:g} Ah x {volts:g} V = {ah * volts:.1f} Wh ({ah * volts / 1000:.4f} kWh). Energy in Wh = capacity in Ah x nominal voltage in V.", "howto:battery_capacity", "formula", frame, wh=ah * volts)
        if text is None: return _result(f"I do not have a documented method for {c.label}, so I will not invent a formula.", f"howto:{c.id}", "formula", frame, ok=False)
        return _result(text, f"howto:{c.id}", "formula", frame, concept=c.id)

    def _accel_band(self, text, c, lit, frame):
        """Mean acceleration while the speed rises from one stated speed to another ("from 20 to 30 km/h"): interpolated crossings of the 1 Hz speed trace, invalid states and gaps excluded."""
        b = self.brain; f = L.speed_factor(text); nums = [x for x in lit.numbers][:2]
        if len(nums) < 2: return _result("I need the two speeds that bound the acceleration, for example 'from 20 to 30 km/h'.", "clarify:literal", "clarify", frame, intent="CLARIFY", ok=False)
        lo, hi = sorted((nums[0] * f, nums[1] * f)); t1 = b.now; t0 = t1 - 7 * 86400
        rows = b.c.execute("SELECT ts,speed,mode FROM sig_1s WHERE ts>=? AND ts<=? AND speed IS NOT NULL ORDER BY ts", (t0, t1)).fetchall()
        runs = kinematics.band_runs([r[0] for r in rows], [r[1] / 10.0 for r in rows], lo, hi, valid=[r[2] in (1, 2) for r in rows])
        if not runs: return _result(f"No valid acceleration from {lo:g} to {hi:g} km/h is recorded in the last 7 days (needs a continuous 1 Hz rise through both speeds).", "stat:acceleration_band", "kinematics", frame, ok=False, band=[lo, hi])
        sec = statistics.fmean(r.seconds for r in runs); acc = statistics.fmean(r.acceleration for r in runs)
        return _result(f"Acceleration from {lo:g} to {hi:g} km/h: {sec:.2f} s on average, {acc:.2f} m/s2, across {len(runs)} valid run(s) in the last 7 days. Crossings are linearly interpolated; gaps, invalid states and non-upward transitions are excluded.", "stat:acceleration_band", "kinematics", frame, runs=len(runs), seconds=sec, acceleration=acc, band=[lo, hi])

    def _trips(self, c, frame, lit):
        """Statistics over recorded trips: the number of days driven, or the average / total / shortest / longest trip (distance and duration are both given, so either reading of 'length' is answered)."""
        b = self.brain; qc = b.qc(); lo, hi, label = resolve_time(lit.times[0] if lit.times else TimeSpec("relative", 30, "day"), qc.ctx)
        rows = b.c.execute("SELECT start,end,km,kwh FROM trips WHERE start>=? AND start<? AND km>0", (lo, min(hi, b.now + 1))).fetchall()
        if not rows: return _result(f"No trip is recorded {label}.", "stat:trip_record", "trips", frame, ok=False)
        if frame.op == "count" and frame.ent == "day":
            days = {((int(r[0]) + int(b.tz)) // 86400) for r in rows}
            return _result(f"The car was driven on {len(days)} different days {label} ({len(rows)} trips).", "stat:trip_days", "trips", frame, days=len(days), trips=len(rows))
        if frame.op not in ("mean", "sum", "max", "min", "median"): return None
        km = [float(r[2]) for r in rows]; mins = [(float(r[1]) - float(r[0])) / 60.0 for r in rows]
        pick = {"mean": statistics.fmean, "sum": sum, "max": max, "min": min, "median": statistics.median}[frame.op]; word = {"mean": "average", "sum": "total", "max": "longest", "min": "shortest", "median": "median"}[frame.op]
        return _result(f"The {word} trip {label}: {pick(km):.1f} km and {pick(mins):.0f} minutes ({len(rows)} trips).", f"stat:trip_{frame.op}", "trips", frame, trips=len(rows), km=pick(km), minutes=pick(mins))

    def _trip_events(self, c, frame, lit):
        """Trips as dated records: the first / last trip of a period (when it started, how far, how long, how efficient), a short list, or how many."""
        b = self.brain; qc = b.qc(); lo, hi, label = resolve_time(lit.times[0] if lit.times else TimeSpec("all"), qc.ctx)
        rows = b.c.execute("SELECT start,end,km,kwh FROM trips WHERE start>=? AND start<? AND km>0 ORDER BY start", (lo, min(hi, b.now + 1))).fetchall()
        label = "on record" if "whole" in label else label
        if not rows: return _result(f"No trip is recorded {label}.", "events:trip_record", "trips", frame, ok=False)
        def one(r):
            s0, e0, km, kwh = r; wh = f", {1000 * float(kwh) / float(km):.0f} Wh/km" if kwh is not None and float(km) > 0.5 else ""
            return f"{A.stamp(int(s0))}: {float(km):.1f} km in {(float(e0) - float(s0)) / 60:.0f} min{wh}"
        if frame.op in ("count", "exists"): return _result(f"There {'was' if len(rows) == 1 else 'were'} {len(rows)} trip{'' if len(rows) == 1 else 's'} {label}.", "events:trip_record", "trips", frame, trips=len(rows))
        if frame.op == "list": return _result(f"The latest trips {label}:\n" + "\n".join(one(r) for r in rows[-5:][::-1]), "events:trip_record", "trips", frame, trips=len(rows))
        r = rows[0] if frame.op == "first" else rows[-1]
        return _result(f"The {'first' if frame.op == 'first' else 'most recent'} trip {label} started {one(r)}.", "events:trip_record", "trips", frame, start=int(r[0]), km=float(r[2]))

    def _by_day(self, c, frame, lit):
        """Which calendar day had the highest / lowest value of a trip-ledger quantity (energy per km, energy used, distance)."""
        store = R.STORE.get(c.id); tier = (store or {}).get("1m")
        if c.id not in ("wh_per_km", "energy_used", "distance_driven") and tier:
            return self._signal_by_day(c, frame, lit, tier)
        if c.id not in ("wh_per_km", "energy_used", "distance_driven"):
            return _result(f"I understood this as the day with the {'highest' if frame.op == 'max' else 'lowest'} {c.label}, but per-day ranking is not wired for that quantity yet, so I will not guess.", "gap:extreme:day", "none", frame, intent="CLARIFY", ok=False, coverage_gap=f"extreme:{c.id}:day")
        b = self.brain; qc = b.qc(); lo, hi, label = resolve_time(lit.times[0] if lit.times else TimeSpec("relative", 30, "day"), qc.ctx)
        rows = b.c.execute("SELECT start,km,kwh FROM trips WHERE start>=? AND start<? AND km>0", (lo, min(hi, b.now + 1))).fetchall()
        days = {}
        for st, km, kwh in rows:
            d = ((int(st) + int(b.tz)) // 86400) * 86400; acc = days.setdefault(d, [0.0, 0.0]); acc[0] += float(km or 0); acc[1] += float(kwh or 0)
        val = {d: (1000.0 * a[1] / a[0] if c.id == "wh_per_km" else a[1] if c.id == "energy_used" else a[0]) for d, a in days.items() if a[0] > 0}
        if len(val) < 2: return _result(f"Fewer than two days have recorded trips {label}, so I cannot rank days.", "extreme:day", "typed_day", frame, ok=False)
        best = (max if frame.op == "max" else min)(val, key=val.get); unit = {"wh_per_km": "Wh/km", "energy_used": "kWh", "distance_driven": "km"}[c.id]; A.set_tz(b.tz)
        stamp = A.stamp(best - int(b.tz)).split(",")[0]
        word = ("highest" if frame.op == "max" else "lowest") + " " + c.label
        return _result(f"The day with the {word} {label} was {stamp}: {val[best]:.1f} {unit} across {len(val)} days with trips.", f"extreme:{c.id}:day", "typed_day", frame, day=best, value=val[best], days=len(val))

    def _signal_by_day(self, c, frame, lit, tier):
        """The calendar day on which a recorded signal reached its highest / lowest value (the daily peak / trough of the one-minute tier)."""
        b = self.brain; qc = b.qc(); table, col, scale = tier; lo, hi, label = resolve_time(lit.times[0] if lit.times else TimeSpec("relative", 7, "day"), qc.ctx)
        fn = "MAX" if frame.op == "max" else "MIN"
        try: rows = b.c.execute(f"SELECT CAST((ts+?)/86400 AS INTEGER) d, {fn}({col}) FROM {table} WHERE ts>=? AND ts<? AND {col} IS NOT NULL GROUP BY d", (int(b.tz), int(lo), min(int(hi), b.now + 1))).fetchall()
        except sqlite3.OperationalError: rows = []
        if len(rows) < 2: return _result(f"Fewer than two days of {c.label} are recorded {label}, so I cannot rank days.", "extreme:day", "typed_day", frame, ok=False)
        d, v = (max if frame.op == "max" else min)(rows, key=lambda r: r[1]); unit = "" if c.unit in ("none", "category") else " " + c.unit; A.set_tz(b.tz)
        return _result(f"The day with the {'highest' if frame.op == 'max' else 'lowest'} {c.label} {label} was {A.stamp(int(d) * 86400 - int(b.tz)).replace(', 00:00', '').replace(' 00:00', '')}: {v / scale:.1f}{unit} (the day's {'peak' if frame.op == 'max' else 'lowest value'}, of {len(rows)} days).", f"extreme:{c.id}:day", "typed_day", frame, day=int(d), value=v / scale, days=len(rows))

    def _at(self, text, c, frame):
        b = self.brain; A.set_tz(b.tz); ts = L.instant(text, b.now, b.tz)
        if ts is None and getattr(self, "_focus", None): ts = int(self._focus)                  # "how fast was i going then": the instant the previous answer was about
        if ts is None: return _result("Which time do you mean? Give me a clock time or how long ago (for example '3 pm yesterday' or '2 hours ago').", "clarify:instant", "clarify", frame, intent="CLARIFY", ok=False)
        label = c.label
        if c.id == "location":
            row = nearest(b.c, "gps_1s", "ts,lat,lon", ts)
            if not row or abs(row[0] - ts) > 90: return _result(f"No GPS fix is recorded close enough to {A.stamp(ts)}; I will not substitute the current location.", "at:location", "instant", frame, ok=False)
            return _result(f"At {A.stamp(row[0])} the car was at GPS {row[1] / 1e6:.5f} N, {row[2] / 1e6:.5f} E.", "at:location", "instant", frame, sample_ts=row[0], target_ts=ts)
        if c.id == "tyre_pressure":
            row = nearest(b.c, "ext_1m", "ts,tp_fl,tp_fr,tp_rl,tp_rr", ts)
            if not row or abs(row[0] - ts) > 90 or any(x is None for x in row[1:]): return _result(f"No tyre-pressure sample is recorded close enough to {A.stamp(ts)}.", "at:tyre_pressure", "instant", frame, ok=False)
            return _result(f"Tyre pressures at {A.stamp(row[0])}: front left {row[1] / 10:.0f}, front right {row[2] / 10:.0f}, rear left {row[3] / 10:.0f}, rear right {row[4] / 10:.0f} kPa.", "at:tyre_pressure", "instant", frame, sample_ts=row[0])
        store = R.STORE.get(c.id) or {}; tier = store.get("1s") or store.get("1m"); row = None; tol = 5; coarse = False
        for key, t_tol in (("1s", 5), ("1m", 90)):                                              # the 1 Hz tier covers the last days only; an older instant falls back to the per-minute tier and says so
            cand = store.get(key)
            if not cand: continue
            table, col, scale = cand
            try: got = nearest(b.c, table, f"ts,{col}", int(ts), col)
            except sqlite3.OperationalError: continue
            tier, tol = cand, t_tol
            if got is not None and abs(got[0] - ts) <= t_tol: row = got; coarse = key == "1m"; break
            if row is None: row = got
        if not tier and c.id == "motor_rpm":
            row = nearest(b.c, "sig_1s", "ts,speed", ts, "speed")
            if not row or abs(row[0] - ts) > 5: return _result(f"No speed sample is recorded close enough to {A.stamp(ts)} to estimate motor RPM.", "at:motor_rpm", "instant", frame, ok=False)
            from ..qa_core import rpm_of
            return _result(f"Estimated motor speed at {A.stamp(row[0])} was {rpm_of(row[1] / 10.0):,.0f} rpm, derived from the recorded {row[1] / 10.0:.1f} km/h with the documented nominal gear ratio and wheel radius; no RPM sensor is recorded.", "at:motor_rpm", "instant", frame, estimated=True, sample_ts=row[0])
        if not tier: return _result(f"{label.capitalize()} is not recorded in this vehicle's history, so I cannot give its value at {A.stamp(ts)}.", f"at:{c.id}", "instant", frame, ok=False)
        if not row or abs(row[0] - ts) > tol: return _result(f"There is no valid {label} sample close enough to {A.stamp(ts)}, so I will not substitute a different time.", f"at:{c.id}", "instant", frame, ok=False)
        scale = tier[2]; val = row[1] / scale; unit = "" if c.unit in ("none", "category") else f" {c.unit}"
        return _result(f"{label.capitalize()} at {A.stamp(row[0])} was {val:.1f}{unit}." + (" (That moment is older than the 1-second history, so this is the one-minute average around it.)" if coarse else ""), f"at:{c.id}", "instant", frame, value=val, sample_ts=row[0], target_ts=ts, resolution="1-minute" if coarse else "1-second")
