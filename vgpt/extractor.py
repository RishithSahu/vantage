"""Optional EVIDENCE stage: a frozen pretrained schema-conditioned extractor (GLiNER2; no training, loaded offline from the local Hugging Face cache) reads the message once and returns what it finds as evidence for the language model: ranked operation labels (descriptive semantic labels from the plan-language definition), the kind of conversation turn, and SPANS with character offsets for the target quantity, the event, the time period, the restriction, numbers with units and trip / session selections.
It is evidence, never a gate: the language model still reads the original message and the capability cards, sees the extractor's absence statements ("restriction: none found") and may disagree; no operation is ever excluded. Spans are exact substrings of the message, which also gives them exact provenance.
The extractor must be loadable by its documented loader: GLiNER2 (legacy span model) loads with `GLiNER2.from_pretrained`; the boundary-model GLiNER2.5 needs `AutoExtractor` of a newer package (not used unless installed and approved)."""
import os
import sys
import time

from . import toolspec as T

ENTITY_LABELS = {"measured quantity": "the vehicle measurement or signal being asked about, such as speed, battery temperature, energy use or tyre pressure",
                 "event or warning": "a warning, fault or occurrence such as hard braking, a blocked camera or a charging session",
                 "time period": "a span of time such as yesterday, last week, the past 3 days or this month",
                 "driving situation": "a driving or charging situation the question is limited to, such as while charging, on the highway, in the city, at night or when raining",
                 "threshold condition": "a limit on a measurement such as above 50 kW, below 30 psi or over 95 degrees fahrenheit",
                 "number with unit": "a number together with its unit, such as 60 km/h or 20 degrees",
                 "trip or session selection": "a phrase that picks particular trips or charging sessions, such as the longest trip or the last charge"}
ENTITY_LABELS_V2 = {"quantity": "the measurement being asked about", "event": "a warning or occurrence", "period": "a span of time", "situation": "a driving or charging situation such as on the highway or while charging",
                    "threshold": "a limit such as above 50 kW or below 30 psi", "number": "a number with its unit", "selection": "which trips or sessions, such as the longest or the latest"}      # short role names with short descriptions: the long descriptions of the first interface pulled numbers and thresholds into the 'measured quantity' label (bench/audit_extraction.py)
TURN_LABELS = {"new": "asks a new question", "follow_up": "continues the previous question and changes one detail", "correction": "corrects what the previous answer was about", "clarification_reply": "answers a question the assistant asked"}


class Extractor:
    def __init__(self, model_id="fastino/gliner2-base-v1", offline=True):
        if offline: os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        for st in (sys.stdout, sys.stderr):                                                               # the library prints emoji banners: never let a console code page crash the load
            try: st.reconfigure(encoding="utf-8", errors="replace")
            except Exception: pass
        t = time.perf_counter(); import gliner2
        loader = getattr(gliner2, "AutoExtractor", None) or gliner2.GLiNER2                              # a newer package dispatches on the checkpoint's own `architecture` (boundary / span); the older one loads the legacy span model only
        self.model = loader.from_pretrained(model_id); self.model_id = model_id; self.load_s = time.perf_counter() - t
        self.ops = {T.FUNC_NL[fn]: fn for fn in T.FUNC_NL}; self.turns = {v: k for k, v in TURN_LABELS.items()}
        self._schema = (self.model.create_schema().classification("operation", list(self.ops), multi_label=True, cls_threshold=0.0).classification("turn", list(TURN_LABELS.values()), multi_label=True, cls_threshold=0.0).entities(ENTITY_LABELS))

    def extract(self, text):
        """-> {"ops": [(function, confidence)] best first, "turn": kind, "spans": {label: [{"text", "start", "end", "confidence"}]}, "ms": float}"""
        t0 = time.perf_counter(); r = self.model.extract(text, self._schema, include_confidence=True, include_spans=True)
        ops = sorted(((self.ops[x["label"]], float(x["confidence"])) for x in r["operation"]), key=lambda x: -x[1]); turns = sorted(r["turn"], key=lambda x: -x["confidence"])
        spans = {k: [{"text": e["text"], "start": e.get("start"), "end": e.get("end"), "confidence": round(float(e.get("confidence", 0)), 3)} for e in v] for k, v in r.get("entities", {}).items()}
        return {"ops": ops, "turn": self.turns[turns[0]["label"]] if turns else "new", "spans": spans, "ms": (time.perf_counter() - t0) * 1000}


def evidence_lines(ex, with_ops=False, top=3):
    """The extractor's findings as the evidence lines of the prompt. Absence is stated explicitly ("none found"), because a missing restriction or period is exactly what a small language model otherwise invents."""
    sp = ex["spans"]; get = lambda label: ", ".join("“" + e["text"] + "”" for e in sp.get(label, [])) or "none found"
    lines = ["Found in the message by a reading aid (it can miss things; the message decides):",
             "- time period: " + get("time period"), "- restriction: " + (", ".join(filter(None, [get("driving situation") if sp.get("driving situation") else "", get("threshold condition") if sp.get("threshold condition") else ""])) or "none found"), "- number with unit: " + get("number with unit"), "- trips or sessions selected: " + get("trip or session selection"),
             "- quantity mentioned: " + get("measured quantity"), "- event mentioned: " + get("event or warning")]
    if with_ops: lines.append("- likely operations: " + ", ".join(f"{fn} ({c:.2f})" for fn, c in ex["ops"][:top]))
    return lines
