"""Deterministic safety policy for decision-under-hazard questions ("should I keep driving if a red brake warning appears?").

A question about whether to continue operating the vehicle while a hazard indication is present is a SAFETY request.  It must never be answered by a telemetry statistic (a warning COUNT does not answer
"should I keep driving") and it must never be silently dropped.  This module is the single deterministic policy:

    classify(text)  -> SafetyIntent | None    structural: a decision / permission frame about continued operation  AND  a hazard referent.
                                              Retrospective or analytic frames ("how many", "when did", "list", "did I", "show") are NOT decisions.
    respond(intent, brain) -> dict            a conservative recommendation + what the recorded data can and cannot say; it never names a failed component, never
                                              claims the car is safe, and quotes at most the most recent RECORDED related event (data, not a diagnosis).

The learned interpreter may also name the intent (op=topic p.topic=safety_assessment); the same `respond` renders it, and the confidence gate does not apply to a safety topic
(a conservative answer to a safety question is the fail-safe direction).  Wording follows the project's existing approved policy text (qa_i: "stop safely and call for help",
"I can't judge whether it is safe - that needs a workshop")."""
import re
from dataclasses import dataclass

_DECISION = re.compile(
    r"\b(?:should|can|could|may|must|do)\s+(?:i|we)\b.{0,30}\b(?:keep|continue|carry\s+on|go\s+on|stop|pull\s+over|drive|driving|ignore|wait|worry|use|charge|park|turn\s+back|drive\s+on)\b"
    r"|\bis\s+it\s+(?:safe|ok|okay|alright|all\s+right|fine|dangerous|risky|serious|urgent|a\s+problem|bad)\b"
    r"|\b(?:safe|ok|okay|fine)\s+to\s+(?:drive|keep|continue|carry|charge|use|ignore)\b"
    r"|\bwhat\s+(?:should|do|must)\s+i\s+do\b|\bwhat\s+now\b|\bdo\s+i\s+(?:need|have)\s+to\s+(?:stop|pull|call)\b|\bshould\s+i\s+(?:be\s+)?(?:worried|concerned|stop)\b", re.I)
_RETROSPECTIVE = re.compile(r"\b(?:how\s+many|how\s+often|how\s+much|when\s+did|when\s+was|what\s+was|did\s+i|have\s+i|has\s+the|list|show\s+me|count|number\s+of|average|total|per\s+day|last\s+(?:week|month)|yesterday)\b", re.I)
_HAZARD = re.compile(r"\b(?:warning|warn|alert|alarm|light|lamp|indicator|fault|error|malfunction|failure|failing|fail|overheat\w*|over-?temp\w*|too\s+hot|smoke|smoking|burning|smell|leak\w*|flat|puncture|"
                     r"low\s+(?:tyre|tire|pressure|battery|brake|coolant|fluid)|red|amber|orange|flashing|beeping|grinding|squealing|spongy|soft\s+pedal)\b", re.I)

CLASSES = (   # (name, pattern, related event types for a transparent 'last recorded' line, class-specific advice)
    ("brake", re.compile(r"\bbrak\w*\b", re.I), ("DTC_SET",), "A red brake warning should be treated as urgent. If it is on now, slow down gradually, stop in a safe place as soon as you can and do not keep driving until the brakes have been checked by a workshop."),
    ("battery_thermal", re.compile(r"\b(?:battery|pack|hv|traction)\b.{0,30}\b(?:overheat\w*|over-?temp\w*|hot|temperature|thermal)\b|\b(?:overheat\w*|over-?temp\w*)\b.{0,30}\b(?:battery|pack)\b", re.I), ("WARN_BATT_OVERTEMP",),
     "A battery over-temperature warning can be serious. Stop driving and charging when it is safe to do so, move away from the vehicle if you notice smoke or a burning smell, and have the battery assessed by a qualified workshop before using the car again."),
    ("tyre", re.compile(r"\b(?:tyre|tire|tpms|puncture|flat)\w*\b", re.I), ("TPMS_LOW",), "A low tyre-pressure warning can lead to a blow-out. Slow down, stop somewhere safe and check or inflate the tyre; do not continue at speed on a tyre that is flat or losing pressure."),
    ("aux_12v", re.compile(r"\b(?:12\s*v|12\s*volt|auxiliary|aux|small\s+battery)\b", re.I), ("LOW_12V",), "A low 12 V battery warning means the car's electronics may fail without notice. Avoid starting a long trip and have the 12 V battery checked soon; if other warnings appear, stop safely."),
    ("motor_thermal", re.compile(r"\b(?:motor|inverter|drive\s*unit|coolant)\b", re.I), ("DTC_SET",), "A motor, inverter or coolant over-temperature warning should be taken seriously. Reduce load, stop in a safe place and let the car cool; do not keep driving if the warning stays on."),
)
GENERIC = ("generic", None, ("ANY_WARNING",), "If a red warning light is on, or you notice smoke, a burning smell, unusual noises or a loss of braking or steering, stop safely and call for help. For other warnings, reduce speed and have the car checked soon.")
NOT_A_DIAGNOSIS = ("I only see what the car records, so I cannot tell from the data which part is affected or whether it is safe to continue - that needs a workshop. "
                   "Telemetry cannot guarantee that a journey is safe or that a historical warning is still active.")


@dataclass
class SafetyIntent:
    hazard: str
    advice: str
    related: tuple
    matched: str


def classify(text):
    t = " ".join(str(text).split())
    if not t or not _DECISION.search(t) or not _HAZARD.search(t): return None
    # A time reference describes the evidence the user is asking about; it must
    # not erase an otherwise explicit safety decision ("should I drive after
    # yesterday's red brake warning?").  Choose the hazard with the strongest
    # local warning evidence, so an unrelated normal reading cannot override it.
    candidates = []
    for name, rx, ev, adv in CLASSES:
        for match in rx.finditer(t):
            nearby = t[max(0, match.start() - 40):match.end() + 40]
            score = 2 if _HAZARD.search(nearby) else 0
            if re.search(r"\b(?:normal|fine|okay|ok|no)\b", nearby, re.I): score -= 4
            candidates.append((score, match.start(), name, ev, adv, match.group(0)))
    if candidates:
        _, _, name, ev, adv, matched = max(candidates, key=lambda x: (x[0], x[1]))
        return SafetyIntent(name, adv, ev, matched)
    n, _, ev, adv = GENERIC
    return SafetyIntent(n, adv, ev, _HAZARD.search(t).group(0))


def respond(intent, brain=None):
    """-> {"answer", "intent": "SAFETY", "facts"}: deterministic; never a count, never a diagnosis."""
    recorded = ""
    try:
        from . import signal_registry as R
        if brain is not None:
            types = []
            for e in intent.related: types += list(R.EVENT_GROUPS.get(e, (e,))) if e in getattr(R, "EVENT_GROUPS", {}) else [e]
            if types:
                marks = ",".join("?" * len(types))
                rows = brain.c.execute(f"SELECT ts,type,detail FROM events WHERE ts<=? AND type IN ({marks}) ORDER BY ts DESC", (brain.now,) + tuple(types)).fetchall()
                # DTC_SET is a generic transport event.  It is only relevant to
                # a component-specific warning when its recorded detail names
                # that component; otherwise it must not be presented as proof.
                detail_terms = {"brake": r"brak", "motor_thermal": r"motor|inverter|coolant"}
                term = detail_terms.get(intent.hazard)
                row = next((r for r in rows if r[1] != "DTC_SET" or not term or re.search(term, r[2] or "", re.I)), None)
                if row:
                    from . import analytics as A
                    recorded = f" For reference, the most recent related recorded event is {row[1]} at {A.stamp(row[0])}" + (f": {row[2]}." if row[2] else ".")
                else: recorded = " No related warning event is recorded in this vehicle's history."
    except Exception: recorded = ""
    ans = intent.advice + " " + NOT_A_DIAGNOSIS + recorded
    return {"answer": ans, "brief": None, "intent": "SAFETY", "facts": {"ok": True, "route": "safety_policy", "hazard": intent.hazard, "policy": "deterministic", "matched": intent.matched}}
