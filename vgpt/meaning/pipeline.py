"""MeaningBrain: the question-answering path built on the meaning frame.

    question -> [safety guard] -> learned frame (Qwen2.5-0.5B + meaning LoRA, grammar-constrained) -> typed resolver -> evidence-backed answer

It is a `Brain` (same UI contract, dialogue state, units, typed query engine, evidence composer); only `ask` differs.  This is a HYBRID and docs/BOUNDARY.md says exactly which stage is learned and which is code: the model reads
what a question means; code parses typed values (dates, numbers, units), type-checks the frame, executes it on the data, does the arithmetic, checks the wording against the numbers, and guards safety and permissions.  Between
the two sit the stages listed in docs/BOUNDARY.md as "code that influences meaning" (the pre-parser asked for on 2026-10-24, the layers of vgpt/meaning/layers.py, the executors' powertrain mapping); each can be switched off
(VGPT_PREPARSER=0, VGPT_LAYERS_OFF=...) so that what the model alone does can be measured.  The deterministic safety guard (a decision about continuing to drive with a hazard) is kept in front of the model on purpose: safety
must not depend on a learned model's confidence (PDF: safety-critical functions stay isolated from the AI)."""
import re
import time

from ..brain import Brain
import os

from . import calibration as CB
from . import layers as LY
from . import ontology as ON
from .arbitrate import arbitrate, own_vocabulary
from .discourse import Dialogue
from ..investigate.engine import Investigator
from .resolve import NONEXEC, Resolver, describe


INVESTIGATE_TRUST = 0.95                                                                # a long request with a single analysis cue is a question, not an analysis, when the model reads it with at least this confidence (interim, docs/MODEL_ONLY.md)
GROUND_MIN_CONF = 0.50                                                                  # below this a concept that no word of the question names is not trusted (interim, layer grounding_floor)
UNSAFE_MIN_CONF = 0.70                                                                  # below this a refusal is not trusted (genuine requests to tamper are read with p >= 0.9)

class MeaningBrain(Brain):
    def __init__(self, db, model, persona="driver", now=None, tau=0.0, **kw):
        super().__init__(db, slm=None, persona=persona, now=now, slm_ir=None, policy="clarify", selector=None, qc_mode="all", **kw)
        self.slm = model; self.frame_model = model; self.tau = float(tau); self.resolver = Resolver(self); self.dialogue = Dialogue(); self.investigator = Investigator(self)                  # no cache of any kind: every question is read by the model and answered from the data as it is now

    CHOICE = {1: {("1",), ("one",), ("first",), ("the", "first"), ("first", "one"), ("the", "first", "one"), ("a",), ("yes",), ("yeah",), ("yep",), ("correct",), ("the", "former"), ("option", "1"), ("number", "1"), ("choice", "1"), ("option", "a")},
              2: {("2",), ("two",), ("second",), ("the", "second"), ("second", "one"), ("the", "second", "one"), ("b",), ("no",), ("nope",), ("the", "latter"), ("option", "2"), ("number", "2"), ("choice", "2"), ("option", "b")}}

    def _choice(self, text):
        """1 or 2 when the person answers a "did you mean 1 or 2" with just that choice; None for anything else (then it is a new question)."""
        from . import literals as _L
        t = tuple(_L.tokens(text))
        for k, words in self.CHOICE.items():
            if t in words: return k
        return None

    def _option(self, frame, slot):
        """How one of the two readings is shown: the word they differ in, in the vehicle's own terms."""
        onto = self.resolver.onto; c = onto.concepts.get(frame.concept); label = c.label if c else frame.concept
        if slot == "act":
            g = onto.acts[frame.act]["gloss"].partition(":")[0].partition(" - or")[0].strip(); return f"{label}, as {g[0].lower() + g[1:]}"
        if slot == "concept": return label
        extra = getattr(frame, slot, "none") if slot else "none"
        return f"{label} ({extra})" if extra != "none" else label

    def _other_reading(self, frame, info):
        """The frame with the one hesitating word replaced by the runner-up word (the model's own second choice), or None."""
        alt = info.get("alt") if isinstance(info, dict) else None
        if not alt: return None
        from .frame import Frame
        words = frame.to_text().split(); i = alt.get("index")
        if i is None or i >= len(words) or words[i] != alt.get("chosen"): return None
        words[i] = alt["other"]
        try:
            f2 = Frame.from_text(" ".join(words))
            return f2 if not self.resolver.onto_validate(f2) else None
        except Exception: return None

    def _result(self, *a, **k):
        r = super()._result(*a, **k)
        note = getattr(self, "_canon_note", None)
        if note and isinstance(r, dict): r["understood"] = ((r.get("understood") or "") + " (" + note + ")").strip(); r["canon"] = note
        return r

    def ask(self, text, use_slm=True):
        self.turn = getattr(self, "turn", 0) + 1; self._decline = None
        clean = re.sub(r"<[^>]{0,300}>", " ", str(text)); clean = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", clean); clean = " ".join(clean.split())[:600]
        self._text = clean; t0 = time.perf_counter(); none_p = {"metric": None, "event": None, "w": "a"}
        from .. import safety_policy
        si = safety_policy.classify(clean)
        if si is not None:
            r = safety_policy.respond(si, self); self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
            return self._result(t0, r["answer"], "SAFETY", none_p, "policy", 1.0, r["facts"], None)
        self._canon_note = None                                                                                      # the pre-parser (vgpt/meaning/canon.py, askme.py): shorthand and spelling, awkward wordings, and the questions that need no model
        try:
            if not LY.preparser_on(): raise StopIteration                                                    # switched off for a measurement of the model alone
            from . import canon as _CN, askme as _AM
            pend = getattr(self, "_canon_pending", None); self._canon_pending = None
            if pend is not None:                                                                                    # the answer to "did you mean 1 or 2"
                pick = _CN.choose(clean, pend[2])
                if pick is not None: clean = re.sub(r"\b" + re.escape(pend[1]) + r"\b", pick, pend[0], flags=re.I); self._text = clean
            c2, notes = _CN.normalise(clean)
            amb = _CN.ambiguous(c2)
            if amb is not None:
                w, opts = amb; self._canon_pending = (c2, w, opts)
                msg = f'"{w}" can mean two things here. Which do you mean?' + chr(10) + chr(10).join(f" {i + 1}. {m} - {e}" for i, (m, e) in enumerate(opts)) + chr(10) + "Say 1 or 2, or ask it with the full words."
                return self._result(t0, msg, "CLARIFY", none_p, "meaning", 1.0, {"ok": False, "route": "clarify:shorthand", "word": w}, "read as: a shorthand with two meanings")
            if notes: self._canon_note = f'you wrote "{clean}", I read it as "{c2}"'; clean = c2; self._text = clean
            h = _AM.handle(self, clean)
            if h is not None:
                self.qcs.choices = []; self.qcs.pending = None; self.qcs.options = []
                return self._result(t0, h["answer"], h["intent"], none_p, "meaning", 1.0, h["facts"], "read as: " + h["read"])
        except StopIteration: pass
        except Exception as e:
            import logging; logging.getLogger("vehiclegpt").warning("pre-parser failed: %s: %s", type(e).__name__, e)
        def investigate():                                                                                           # a multi-part request for an analysis: decomposed and answered by the investigator, not squeezed into one frame
            try: res = self.investigator.run(clean, getattr(self.dialogue.last, "focus_ts", None))
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").exception("investigation failed for %r: %s", clean, e); res = None
            if res is not None:
                from .frame import Frame
                ev = res.pop("focus_event", None); onto = self.resolver.onto; concept = next((c.id for c in onto.concepts.values() if c.kind == "event" and c.registry_key == ev), "none") if ev else "none"
                fr = Frame("context", concept); self.dialogue.record(self, clean, fr, [], res); self._last_data = (clean, fr)
                return self._result(t0, res["answer"], "INVESTIGATE", none_p, "meaning", 0.9, res["facts"], "read as: investigation (" + res["facts"]["template"] + ")", {"raw": "investigation", "ms": 0.0}, time.perf_counter(), time.perf_counter())
            return None
        pending = getattr(self, "_pending", None); self._pending = None
        if pending is not None:
            pick = self._choice(clean)
            if pick is not None:
                question, f1, f2 = pending; chosen = f1 if pick == 1 else f2
                try: r = self.resolver.answer(question, chosen)
                except Exception as e:
                    import logging; logging.getLogger("vehiclegpt").exception("resolver failed for %r: %s", question, e); r = None
                if r is not None:
                    if r.get("intent") not in ("CLARIFY", "NONE"):
                        self._last_data = (question, chosen); self.dialogue.record(self, question, chosen, [], r)
                    facts = dict(r.get("facts") or {}); facts.setdefault("frame", chosen.as_dict()); facts["chosen"] = pick
                    return self._result(t0, r["answer"], r.get("intent", "ANSWER"), none_p, "meaning", 1.0, facts, "read as: " + describe(chosen, self.resolver.onto), {"raw": chosen.to_text(), "ms": 0.0}, time.perf_counter(), time.perf_counter())
        aw = getattr(self, "_await_instant", None); self._await_instant = None
        if aw is not None and len(clean.split()) <= 8:                                                                 # the answer to "which time do you mean?"
            try:
                from . import literals as _LI
                ts_ = _LI.instant(clean, self.now, self.tz)
                if ts_ is not None:
                    r = self.resolver.answer(aw[0], aw[1], None, int(ts_))
                    if r is not None and r.get("intent") not in ("CLARIFY", "NONE"):
                        self._last_data = (aw[0], aw[1]); self.dialogue.record(self, aw[0], aw[1], [], r, focus_ts=int(ts_)); facts = dict(r.get("facts") or {}); facts.setdefault("frame", aw[1].as_dict()); facts["completed_from"] = aw[0]
                        return self._result(t0, r["answer"], r.get("intent", "ANSWER"), none_p, "meaning", 1.0, facts, "read as: " + describe(aw[1], self.resolver.onto) + f' (your answer "{clean}" completed "{aw[0]}")', {"raw": aw[1].to_text(), "ms": 0.0}, time.perf_counter(), time.perf_counter())
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("completing the instant failed: %s: %s", type(e).__name__, e)
        weak = False
        if LY.on("investigate") and self.investigator.wants(clean):
            if self.investigator.strength(clean) >= 2:                                                                  # two or more analysis cues: the request is an analysis
                r = investigate()
                if r is not None: return r
            else: weak = True                                                                                           # one cue only (measured: 5 of 70 ordinary questions in a long wrapper had exactly one): the model decides below
        from . import literals as _L
        if _L.asks_what_data_is_held(clean):                                                                           # a question about the store itself: counted from its tables, not read as a signal
            from . import records as _RC
            r = _RC.inventory(self); return self._result(t0, r["answer"], r["intent"], none_p, "meaning", 1.0, r["facts"], "read as: what the database holds")
        v2 = ON.active_version() >= 2; last = self.dialogue.last; prev = last.frame.to_text(2) if (v2 and last is not None) else None          # frame version 2: the model also reads the frame of the last answered question
        frame, info = self.frame_model.parse(clean, prev=prev) if v2 else self.frame_model.parse(clean)                  # no frame cache: a repeated question is read again (owner's rule: no question cache; every latency number is a real model run)
        t1 = time.perf_counter()
        if weak and (frame is None or float(info.get("pmin") or 0.0) < INVESTIGATE_TRUST or frame.act in NONEXEC):
            r = investigate()
            if r is not None: return r
        if frame is None:
            return self._result(t0, "I could not read that as a question about this vehicle. Could you rephrase it?", "CLARIFY", none_p, "meaning", 0.0, {"ok": False, "route": "clarify:unparsed", "raw": info.get("raw")}, None, info, t1, t1)
        if LY.on("arbitrate"): frame, arb = arbitrate(clean, frame, self.resolver.onto, self.resolver.glossary)           # the rare-word rules of the first models: off since full_v11 (measured: 59 correct frames changed, 2 fixed)
        else:
            arb = None
            if LY.on("hint_terms"): frame, arb = arbitrate(clean, frame, self.resolver.onto, self.resolver.glossary, rules="B")          # a word the model is known to misread and the glossary says what it means
            # the vehicle's own vocabulary: what the model could not know (a signal this vehicle declares)
            try:
                qc = self.qc() if LY.on("own_vocab") else False; onto = self.resolver.onto
                if qc is not False:
                    cached = getattr(self, "_decl", None)
                    if cached is None or cached[0] is not self.c:
                        from ..vehicle import lake as LK
                        names = LK.declared_names(self.c)                                                                 # the signals only this vehicle's own profile declares: names the model could not have learned
                        cached = self._decl = (self.c, [cid for cid in names if cid in onto.concepts and qc.profile.concept_available(cid)])
                    avail = cached[1]
                    if frame.concept == "unlisted" and os.environ.get("VGPT_OWNVOCAB_ALL") == "1":                         # the first version of this layer (retired): any signal of the vehicle, matched by a generic word such as "engine" or "wheels"
                        avail = [cid for cid in onto.concepts if qc.profile.concept_available(cid)]
                    if avail: frame, arb = own_vocabulary(clean, frame, onto, self.resolver.glossary, avail)
            except Exception: arb = None
        try:
            from . import evmap as _EV
            g_ = _EV.guard(clean, frame, self.resolver.onto)
            if g_ is not frame: arb = f"the words say consumption, not efficiency: {frame.concept} -> {g_.concept}"; frame = g_
        except Exception: pass
        conf = float(info.get("pmin") or 0.0)
        known = info.get("pmin") is not None; conf_cal = CB.calibrated(conf) if known else None                    # the share of development questions read right at this score (vgpt/meaning/calibration.py)
        if LY.on("unsafe_floor") and frame.act == "unsafe" and conf < UNSAFE_MIN_CONF:                            # interim rule (docs/TRAINING_QUEUE.md): keyboard noise read as a request to tamper is asked again, never refused
            from .frame import Frame
            frame = Frame("ambiguous", "none"); arb = f"unsafe read with p < {UNSAFE_MIN_CONF}: asked again instead of refused"
        if LY.on("grounding_floor") and conf < GROUND_MIN_CONF and (prev is None or getattr(frame, "ref", "none") == "none") and frame.act in ("why", "define", "howto") and frame.concept not in ("none", "unlisted"):
            sup = getattr(self, "_support", None)
            if sup is None:
                from .cues import Support
                sup = self._support = Support(self.resolver.onto, self.resolver.glossary)
            if not sup.supported(clean, frame.concept):
                from .frame import Frame
                arb = f"{frame.concept} read with p < {GROUND_MIN_CONF} and no word of it in the question: not about this vehicle"; frame = Frame(frame.act, "unlisted")
        lo = CB.refuse_below()
        if known and lo and conf_cal < lo and not (prev is not None and getattr(frame, "ref", "none") != "none"):                  # a reading the model was itself very unsure of is not answered as if it were certain
            msg = f"I am not sure I understood that, so I will not guess. I read it as: {describe(frame, self.resolver.onto)}. If that is not what you meant, please say it another way."
            return self._result(t0, msg, "CLARIFY", none_p, "meaning", conf_cal, {"ok": False, "route": "clarify:low_confidence", "frame": frame.as_dict(), "pmin": conf, "confidence": round(conf_cal, 3), "refuse_below": lo}, None, info, t1, t1)
        if self.tau and conf < self.tau and frame.act not in NONEXEC:
            other = self._other_reading(frame, info)
            if other is not None:                                                                                      # the model hesitated between two words of one slot: ask which, "1" or "2" answers
                self._pending = (clean, frame, other)
                slot = (info.get("alt") or {}).get("slot"); a, b = self._option(frame, slot), self._option(other, slot)
                msg = f"I want to be sure I understood. Did you mean:\n 1. {a}\n 2. {b}\nSay 1 or 2, or ask it another way."
                return self._result(t0, msg, "CLARIFY", none_p, "meaning", conf, {"ok": False, "route": "clarify:choose", "frame": frame.as_dict(), "other": other.as_dict(), "pmin": conf}, None, info, t1, t1)
            msg = f"I am not sure I understood. I read it as: {describe(frame, self.resolver.onto)}. If that is not what you meant, please say it another way."
            return self._result(t0, msg, "CLARIFY", none_p, "meaning", conf, {"ok": False, "route": "clarify:low_confidence", "frame": frame.as_dict(), "pmin": conf}, None, info, t1, t1)
        from . import literals as L
        um = L.display_units(clean)                                         # a display unit is a VALUE the person typed (psi, miles, Fahrenheit), parsed like a number; it changes how an answer is shown, never what is asked
        if um: self.units.update(um)
        last = getattr(self, "_last_data", None)
        if um and frame.act in ("control", "unsupported", "ambiguous", "offtopic", "social") and last is not None:      # "i want it in psi": remember the unit and show the last answer again in it
            msg = "Okay - " + " ".join(f"{k}={v}" for k, v in self.units.items()) + ".\n" + self.resolver.answer(*last)["answer"]
            return self._result(t0, msg, "UNITS", none_p, "meaning", 1.0, {"ok": True, "units": dict(self.units), "frame": frame.as_dict()}, None, info, t1, t1)
        times = focus = position = None; comp = None
        if frame.act not in ("unsafe", "control", "unsupported", "offtopic") and LY.on("discourse"):                # a greeting-shaped frame may still be "how did you get that": only a provenance completion is taken from it
            try:
                comp = self.dialogue.complete(self, clean, frame, L.extract(clean, self.now, self.tz))
                if frame.act == "social" and (comp is None or comp.kind != "provenance"): comp = None
            except Exception as e:
                import logging; logging.getLogger("vehiclegpt").warning("dialogue completion failed: %s: %s", type(e).__name__, e); comp = None
        if comp is not None and comp.kind == "provenance":
            r = self.dialogue.explain(self); return self._result(t0, r["answer"], "EXPLAIN", none_p, "meaning", 1.0, r["facts"], None, info, t1, t1)
        if comp is not None and comp.kind == "ordinal":
            r = self.dialogue.rank_events(self, comp)
            if r is not None:
                self.dialogue.record(self, clean, comp.frame, comp.times, r); return self._result(t0, r["answer"], r["intent"], none_p, "meaning", 0.9, dict(r["facts"], dialogue={"kind": comp.kind, "note": comp.note}), "read as: " + describe(comp.frame, self.resolver.onto) + " (follow-up)", info, t1, t1)
        if comp is not None and comp.frame is not None: frame, times, focus, position = comp.frame, comp.times, comp.focus_ts, comp.position
        try: r = self.resolver.answer(clean, frame, times=times, focus_ts=focus, position=position) if (times or focus or position) else self.resolver.answer(clean, frame)
        except Exception as e:                                                          # an executor that fails says so; it never takes the assistant down
            import logging; logging.getLogger("vehiclegpt").exception("resolver failed for %r: %s", clean, e)
            r = {"answer": "I understood the question, but I could not compute the answer from this vehicle's data (" + type(e).__name__ + "), so I will not guess.", "intent": "CLARIFY", "facts": {"ok": False, "route": "error:resolver", "error": type(e).__name__}}
        t2 = time.perf_counter()
        if frame.act not in NONEXEC and (frame.concept not in ("none", "unlisted") or frame.act in ("summary", "anomaly")) and r.get("intent") not in ("CLARIFY", "NONE"):
            self._last_data = (clean, frame)
            lit0 = L.extract(clean, self.now, self.tz); self.dialogue.record(self, clean, frame, times if times else lit0.times, r, focus_ts=focus)
        if str((r.get("facts") or {}).get("route") or "") == "clarify:instant": self._await_instant = (clean, frame)                      # "Which time do you mean?": the next short answer with a time in it completes THIS question
        facts = dict(r.get("facts") or {}); facts.setdefault("frame", frame.as_dict()); facts["pmin"] = round(conf, 3); facts["confidence"] = round(conf_cal, 3) if known else None
        if comp is not None: facts["dialogue"] = {"kind": comp.kind, "note": comp.note}
        if arb: facts["arbitrated"] = arb
        params = {"metric": frame.concept if frame.concept != "none" else None, "event": None, "w": "a"}
        return self._result(t0, r["answer"], r.get("intent", "MEANING"), params, "meaning", round(conf_cal, 3) if known else 0.5, facts, ("read as: " + describe(frame, self.resolver.onto)) if frame.act not in NONEXEC else None, info, t1, t2)
