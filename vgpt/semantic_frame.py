"""Canonical coarse semantic-frame schema shared by training, runtime and tests.

Model tokens are a compact wire format.  They are never exposed as ontology
identifiers: every output is decoded to this schema before use.
"""
from dataclasses import dataclass

TASK = {"data": "telemetry", "define": "definition", "estimate": "estimate",
        "capability": "capability", "advice": "advice", "other": "other"}
OPERATION = {x: x for x in ("why", "before", "events", "compare", "largest", "trend", "now",
                                "usage", "average", "maximum", "minimum", "duration", "other")}
CONCEPT = {
    "energy": "wh_per_km", "battery": "battery_temperature", "motor": "motor_temperature",
    "coolant": "coolant_flow", "speed": "speed", "charge": "battery_soc",
    "chargepower": "charging_power", "torque": "motor_torque", "outside": "ambient_temperature",
    "climate": "hvac_power", "range": "range_estimate", "tyre": "tyre_pressure",
    "weather": "domain:weather", "service": "domain:service", "health": "domain:health",
    "driving": "domain:driving", "place": "domain:places", "vehicle": "domain:vehicle", "none": None,
}
EVENT = {"adas": "ADAS_DISENGAGE", "overheat": "WARN_BATT_OVERTEMP", "fault": "DTC_SET",
         "charging": "CHARGE_START", "camera": "CAMERA_BLOCKED", "none": None}

_INV_TASK = {v: k for k, v in TASK.items()}; _INV_OP = {v: k for k, v in OPERATION.items()}
_INV_CONCEPT = {v: k for k, v in CONCEPT.items()}; _INV_EVENT = {v: k for k, v in EVENT.items()}

LEGACY_INTENT = {"why": "WHY_RANGE", "before": "BEFORE_EVENT", "events": "EVENTS", "compare": "COMPARE",
                 "largest": "LARGEST", "trend": "TREND", "now": "STATUS", "usage": "USAGE",
                 "average": "SUMMARY", "maximum": "SUMMARY", "minimum": "SUMMARY", "duration": "ESTIMATE", "other": "NONE"}
LEGACY_CONCEPT = {"wh_per_km": "energy_use", "battery_temperature": "batt_temp", "ambient_temperature": "amb",
                  "battery_soc": "soc", "hvac_power": "hvac", "range_estimate": "range",
                  "tyre_pressure": "dom:TYRES", "domain:weather": "dom:WEATHER", "domain:service": "dom:SERVICE",
                  "domain:health": "dom:HEALTH", "domain:driving": "dom:DRIVING", "domain:places": "dom:PLACES",
                  "domain:vehicle": "dom:VEHICLE"}

@dataclass(frozen=True)
class SemanticFrame:
    task: str
    operation: str
    concept: str | None
    event: str | None

    def __post_init__(self):
        if self.task not in _INV_TASK or self.operation not in _INV_OP or self.concept not in _INV_CONCEPT or self.event not in _INV_EVENT:
            raise ValueError(f"non-canonical semantic frame: {self}")

    @classmethod
    def from_tokens(cls, task, operation, concept, event):
        try: return cls(TASK[task], OPERATION[operation], CONCEPT[concept], EVENT[event])
        except KeyError as e: raise ValueError(f"unknown semantic token: {e.args[0]}") from e

    def to_tokens(self):
        return " ".join((_INV_TASK[self.task], _INV_OP[self.operation], _INV_CONCEPT[self.concept], _INV_EVENT[self.event]))

    def as_dict(self):
        return {"task": self.task, "operator": self.operation, "metric": self.concept, "event": self.event}

    @property
    def legacy_intent(self): return LEGACY_INTENT[self.operation]

    @property
    def legacy_concept(self): return LEGACY_CONCEPT.get(self.concept, self.concept)
