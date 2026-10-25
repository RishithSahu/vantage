"""The investigator: a long, multi-part request -> parsed structure -> the matching analysis -> an evidence-backed report in the assistant's normal answer format.

The frame model reads short questions; a request such as "reconstruct the preceding ten minutes and following five minutes, identify the earliest abnormal signal and distinguish confirmed facts from
possible explanations" is not a six-word frame, and collapsing it into one (as before) produced one answer for many different asks.  Here the request is decomposed (sentences, windows, counts, named
signals and events, anchor, wanted analysis), an analysis from vgpt/investigate/templates.py is run over the recorded history, and the report keeps recorded facts apart from associations and hypotheses.
A request no analysis fits returns None and the ordinary question path answers it."""
import logging
import time

from . import parse as P
from . import templates as T

from . import t_change as TC, t_events as TE, t_extra as TX, t_trips as TT

TEMPLATES = {
    "incident": lambda req, D, focus: T.incident(req, D, focus), "recovery": lambda req, D, focus: T.incident(req, D, focus, variant="recovery"), "spike_or_trend": TX.spike_or_trend, "best_conditions": TX.best_conditions, "occurrences": TX.occurrences, "unanswered": TX.unanswered, "event_summary": TX.event_summary, "status_vs_signal": TX.status_vs_signal,
    "cooccur": TE.cooccur, "group_warnings": TE.group_warnings, "leadtime": TE.leadtime, "adas": TE.adas, "discrepancy": TE.discrepancy, "fault_history": TE.fault_history, "dtc_cycle": TE.dtc_cycle, "health": TE.health,
    "frequency_trend": TE.frequency_trend, "similar": TE.similar, "conditions": TT.conditions, "thermal_stop": TT.thermal_stop, "cold_start": TT.cold_start, "range_drop": TT.range_drop,
    "change_explain": TC.change_explain, "efficiency_onset": TC.efficiency_onset, "trend_vs_outliers": TC.trend_vs_outliers, "energy_attribution": TC.energy_attribution, "largest_change": TC.largest_change,
    "joint_change": TC.joint_change, "correlate_changes": TC.correlate_changes, "thermal_rise": TC.thermal_rise, "trip_outliers": TC.trip_outliers, "compare_periods": TC.compare_periods,
}


class Investigator:
    def __init__(self, brain): self.b = brain; self.log = logging.getLogger("vehiclegpt")

    def wants(self, text): return P.is_investigation(text)

    def strength(self, text):
        """How many analysis cues the best-matching template found in the request (0 when none)."""
        try: return max((P.parse(text, self.b.now, self.b.tz).scores or {0: 0}).values())
        except Exception: return 0

    def run(self, text, focus=None):
        """-> result dict (answer, intent, facts) or None when no analysis fits."""
        from .primitives import Data
        t0 = time.perf_counter(); D = Data(self.b); req = P.parse(text, D.now, D.tz)
        if req.template is None or req.template not in TEMPLATES: return None
        rep = TEMPLATES[req.template](req, D, focus)
        if rep is None: return None
        facts = {"ok": rep.ok, "route": f"investigate:{req.template}", "executor": "investigator", "template": req.template, "investigation": {"template": req.template, "steps": rep.steps, "scores": {k: v for k, v in req.scores.items() if v},
                 "events": req.events, "concepts": req.concepts, "windows": req.windows, "counts": req.counts, "anchor_kind": req.anchor_kind, "premise": rep.premise}, "evidence": rep.evidence or None, "ms": round((time.perf_counter() - t0) * 1000)}
        facts.update(rep.facts)
        if rep.focus_ts: facts["event_ts"] = int(rep.focus_ts)
        return {"answer": rep.text(), "intent": "INVESTIGATE", "facts": facts, "focus_event": rep.focus_event}
