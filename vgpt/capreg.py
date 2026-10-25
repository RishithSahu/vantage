"""Capability registry: ONE description of everything this vehicle can be asked about - signals, events, named conditions, trip classes, ranking metrics and the domain tools - built from the vehicle schema and the tool definitions.

The language layer (retrieval + frozen model, see plan.py) works from these descriptions; it is not given phrase lists written to pass any question set. A new vehicle needs new metadata (a schema overlay, see schema_overlay.py), not new parser rules. An opaque identifier is only as useful as the description registered with it.

Each Capability carries: an id the plan may name, a kind, a title, unit and category, an availability verdict from the data profile (known vs recorded), the operations it supports, related capabilities (a Boolean/ratio signal and the event stream derived from it), and two renderings: `text` (retrieval document) and `digest` (the compact line shown to the model)."""
import hashlib
from dataclasses import dataclass, field

from . import signal_registry as R

# Which event stream belongs to which signal (schema metadata: "a warning event associated with a temperature signal").
EVENT_SIGNALS = {"CAMERA_BLOCKED": "camera_blockage", "CAMERA_DEGRADED": "camera_sharpness", "WARN_BATT_OVERTEMP": "battery_temperature", "TPMS_LOW": "tyre_pressure", "TYRE_INFLATED": "tyre_pressure", "LOW_12V": "aux_battery_voltage",
                 "ADAS_DISENGAGE": "adas_state", "HARSH_BRAKE": "braking_intensity", "HARSH_ACCEL": "acceleration", "COOLANT_FLOW_DROP": "coolant_flow", "CHARGE_START": "charging_state", "SERVICE_DUE": "odometer"}

# Schema-level MEANING text for the concepts a reader could confuse (shown on the capability card when the registry has no description of its own). This is metadata about the vehicle's signals and events, not language rules.
MEANING = {
    "wh_per_km": "electrical energy used per kilometre driven (driving efficiency; a higher value means LESS efficient)", "energy_used": "total electrical energy drawn from the battery over the period (accumulates; a total, not a rate)",
    "distance_driven": "distance travelled over the period (accumulates)", "range_estimate": "estimated remaining driving range on the current charge", "battery_efficiency": "ratio of energy delivered at the wheels to energy drawn from the battery",
    "regen_energy": "energy recovered by regenerative braking over the period (accumulates)", "regen_power": "instantaneous power fed back to the battery while braking", "battery_soc": "state of charge of the traction battery, 0-100 % of its capacity (how much charge is left)",
    "battery_soh": "state of health of the traction battery, % of its original capacity (ageing, not charge level)", "battery_temperature": "temperature of the traction battery pack", "cell_temperature_max": "temperature of the hottest single cell in the pack",
    "cell_temperature_min": "temperature of the coolest single cell in the pack", "charging_power": "power flowing into the battery while charging", "battery_power": "electrical power at the battery terminals (positive = discharging, negative = charging or regeneration)",
    "motor_temperature": "temperature of the drive motor", "motor_power": "mechanical power at the motor shaft", "coolant_flow": "flow rate of the battery / powertrain coolant pump", "hvac_power": "electrical power used by heating and air conditioning",
    "aux_battery_voltage": "voltage of the 12 V auxiliary battery (not the traction battery)", "speed": "vehicle speed (from the vehicle network)", "gps_speed": "vehicle speed measured by GPS", "ambient_temperature": "air temperature outside the vehicle",
    "cabin_temperature": "air temperature inside the cabin", "tyre_pressure": "tyre pressure of all four tyres (the lowest one is reported)", "tyre_temperature": "tyre temperature", "drive_mode": "what the vehicle is doing: parked, city driving, highway driving or charging",
    "camera_blockage": "fraction of the front camera view that is blocked, 0 (clear) to 1 (fully blocked)", "camera_sharpness": "sharpness of the camera image (low = blurred or dirty lens)", "lane_confidence": "confidence of lane-marking detection, 0 to 1",
    "odometer": "total distance on the vehicle's clock", "brake_pad_life": "remaining brake pad life, %", "braking_intensity": "strength of braking (deceleration)", "acceleration": "longitudinal acceleration of the vehicle",
    "rain": "whether rain is detected by the rain sensor", "light": "ambient light level outside", "thermal_rise_rate": "how fast the battery temperature is rising, degrees per minute", "adas_state": "whether the driver-assistance (lane keeping) system is active",
}
EVENT_MEANING = {
    "ADAS_DISENGAGE": "the driver-assistance (lane keeping) system switched off or dropped out", "WARN_BATT_OVERTEMP": "battery over-temperature warning raised (the pack got too hot)", "DTC_SET": "a diagnostic trouble (fault) code was set",
    "HARSH_BRAKE": "a hard-braking event (strong deceleration)", "HARSH_ACCEL": "a hard-acceleration event", "HARSH_CORNER": "a hard-cornering event", "TPMS_LOW": "tyre-pressure warning: a tyre fell below its warning threshold",
    "TYRE_INFLATED": "a tyre was re-inflated after a pressure warning", "LOW_12V": "low 12 V auxiliary battery warning", "SERVICE_DUE": "a service reminder was raised", "CAMERA_BLOCKED": "the camera was judged blocked (an event derived from the camera blockage signal)",
    "CAMERA_DEGRADED": "the camera image was judged degraded", "CHARGE_START": "a charging session started", "CAN_ANOMALY": "an anomaly was detected on the vehicle network (CAN bus)",
    "ANY_WARNING": "any warning event: battery over-temperature, fault code, tyre pressure, 12 V battery, camera blockage or network anomaly", "COOLANT_FLOW_DROP": "coolant flow fell well below normal while the pump should run (detected from the signal)",
}

# Operations a plan can name. `needs` says which capability kinds the operation reads.
OPERATIONS = {
    "current": "latest recorded value of a signal",
    "aggregate": "a statistic (mean, median, std, sum, p95) of a signal over a period",
    "extreme": "highest or lowest value of a signal over a period, with when it happened",
    "count": "number of occurrences of an event type (or of episodes of a condition) in a period",
    "list": "list the individual events of one type in a period (all, first or last)",
    "exists": "did an event / condition happen at all in a period (yes, no, or not enough data to say no)",
    "share": "share of time a condition held in a period",
    "compare": "the same statistic for two periods or two condition groups",
    "change": "when the largest or most sustained change of a signal happened",
    "context": "what the signals did in a window before / after an event",
    "explain": "association-based explanation of a change in a signal, or of one event",
    "trend": "direction and size of a drift in a signal",
    "rank": "rank trips or charging sessions by a metric",
    "correlate": "relationship between two signals",
    "capability": "is a signal or event recorded on this vehicle",
    "tool": "a domain tool (service, tyres, weather, driving, health, places)",
    "clarify": "ask one specific question when the request has an unresolved distinction",
    "refuse": "not a question about recorded data, or a request to operate the vehicle",
    "chat": "greeting, thanks, help or what-can-you-do",
}

TOOLS = {   # id -> (title, what it answers, focus values)
    "service": ("service and wear", "odometer, distance and time to next service, brake pad life, 12 V battery, software version, vehicle model", ("odometer", "service", "model", "brakes")),
    "tyres": ("tyres", "per-wheel tyre pressure against spec, TPMS warnings, slow-leak trend", ()),
    "weather": ("weather sensed by the car", "rain and light now or over a period, wipers, cloud estimate", ("now", "period")),
    "driving": ("driving style", "hard braking/acceleration/cornering per 100 km, regeneration share, lane-assist share, eco speed, drive time", ("style", "eco", "adas", "duration")),
    "health": ("health summary", "one verdict across battery, cooling, tyres, 12 V, brakes and camera; use for 'is X ok / any problems'", ("all", "battery", "tyres", "brakes", "camera", "electrical", "cooling")),
    "places": ("places and position", "where the car is, heading, distance and time to saved places, trips between places", ("where", "distance", "eta", "heading", "history")),
}


@dataclass
class Capability:
    id: str
    kind: str                       # signal | event | condition | tripclass | metric | tool
    title: str
    unit: str = ""
    category: str = ""
    dtype: str = ""
    available: bool = True
    note: str = ""                  # why unavailable / how it is obtained
    related: list = field(default_factory=list)
    text: str = ""                  # retrieval document
    digest: str = ""                # compact line (identifier-level; NOT what the model is shown)
    card: str = ""                  # the capability card shown to the model: identifier, kind, unit, MEANING, relationships, availability
    synonyms: tuple = ()


def _words(s): return set(w for w in s.lower().replace("_", " ").replace("-", " ").split() if w)


class CapabilityRegistry:
    def __init__(self, profile, conn=None):
        self.items = []; self.by_id = {}; self.profile = profile
        sig_units = {}
        for name, c in R.CONCEPTS.items():
            res = profile.resolve(name) if profile is not None else None
            avail = bool(res and res.available)
            note = "" if avail else ("not recorded on this vehicle" + (": would need " + " + ".join(res.missing) if res and res.missing and res.missing != [name] else ""))
            if res is not None and res.status in ("derived", "estimated"): note = f"{res.status} from " + ", ".join(res.requires or ()) if res.requires else res.status
            kind = "signal"
            rel = [e for e, s in EVENT_SIGNALS.items() if s == name]
            cap = Capability(name, kind, c.label, c.unit, c.category, c.dtype, avail, note, rel, synonyms=c.synonyms)
            parts = [name.replace("_", " "), c.label, c.unit if c.unit not in ("none", "category", "count", "ratio") else "", c.category, c.description, " ".join(c.synonyms[:8])]
            if c.levels: parts.append("levels " + " ".join(c.levels))
            cap.text = " . ".join(p for p in parts if p)
            lab = "" if _words(c.label) <= _words(name) else f"({c.label})"
            cap.digest = f"{name}[{c.unit}]{lab}" if c.unit not in ("none", "category") else f"{name}{lab}"
            self._add(cap); sig_units[name] = c.unit
        for et, (label, syns) in R.EVENTS.items():
            if et == "ANY_WARNING": avail = True
            elif et in R.DETECTED_EVENTS: avail = profile.concept_available("coolant_flow") if profile is not None else True
            else: avail = profile.event_table_exists() if profile is not None else True
            rel = [EVENT_SIGNALS[et]] if et in EVENT_SIGNALS else []
            cap = Capability(et, "event", label, "event", "event", "event", avail, "" if avail else "no event log on this vehicle", rel, synonyms=tuple(syns))
            cap.text = " . ".join([et.replace("_", " ").lower(), label, "event", " ".join(list(syns)[:8])]); cap.digest = f"{et}({label})"
            self._add(cap)
        for pn, (desc, rule) in R.PREDICATES.items():
            cap = Capability(pn, "condition", desc, "", "condition", "predicate", True, "", [], synonyms=tuple(R.PRED_SYN.get(pn, ())))
            cap.text = " . ".join([pn.replace("_", " "), desc, "condition", " ".join(R.PRED_SYN.get(pn, ())[:6])]); cap.digest = pn
            self._add(cap)
        for tn, (desc, col, op, val) in R.TRIP_CLASSES.items():
            cap = Capability("trip:" + tn, "tripclass", desc, "", "trip class", "class", True); cap.text = f"trip class {tn} . {desc}"; cap.digest = tn
            self._add(cap)
        for ent, mets in R.ENTITY_METRICS.items():
            for mn, (col, unit, worse) in mets.items():
                cap = Capability(f"{ent}.{mn}", "metric", f"{mn.replace('_', ' ')} of a {ent.replace('_', ' ')}", unit, "ranking", "metric", True); cap.text = f"rank {ent} by {mn.replace('_', ' ')} {unit}"; cap.digest = mn
                self._add(cap)
        for tn, (title, what, focus) in TOOLS.items():
            cap = Capability("tool:" + tn, "tool", title, "", "tool", "tool", True, "", [])
            cap.text = f"{title} . {what}"; cap.digest = f"{tn}({what.split(',')[0]})"
            self._add(cap)
        for c in self.items: c.card = self._card(c)
        self.available_signals = [c for c in self.items if c.kind == "signal" and c.available]
        self.missing_signals = [c for c in self.items if c.kind == "signal" and not c.available]

    def _card(self, c):
        """One line the model reads: what the capability MEANS (not just its identifier), its type and unit, how it relates to others, and whether this vehicle records it."""
        if c.kind == "signal":
            con = R.CONCEPTS.get(c.id); desc = (con.description if con is not None and con.description else MEANING.get(c.id, "")); meaning = c.title + ("; " + desc if desc and desc.lower() != c.title.lower() else "") + (f" (levels: {', '.join(con.levels)})" if con is not None and con.levels else "")
            rel = [f"event stream {e} ({self.by_id[e].title})" for e in c.related if e in self.by_id]
            unit = c.unit if c.unit not in ("none", "category", "") else (c.dtype if c.dtype else "")
            return f"{c.id} | signal | {unit} | {meaning}" + (" | related: " + "; ".join(rel) if rel else "") + ("" if c.available else " | NOT RECORDED on this vehicle (known concept)" + (f": {c.note}" if c.note else ""))
        if c.kind == "event":
            rel = [f"derived from signal {r}" for r in c.related if r in self.by_id]
            return f"{c.id} | event | | {EVENT_MEANING.get(c.id, c.title)}" + (" | " + "; ".join(rel) if rel else "") + ("" if c.available else " | NOT RECORDED")
        if c.kind == "tool":
            foc = TOOLS.get(c.id.split(":", 1)[1], ("", "", ()))[2]
            return f"{c.id.split(':', 1)[1]} | tool | | {c.title}: " + c.text.split(" . ", 1)[-1] + (" | focus values: " + ", ".join(foc) if foc else "")
        return f"{c.id} | {c.kind} | {c.unit} | {c.title}"

    def card_text(self, ids):
        return "\n".join(self.by_id[i].card for i in ids if i in self.by_id)

    def _add(self, cap): self.items.append(cap); self.by_id[cap.id] = cap

    # ---- renderings -------------------------------------------------------------------------------------
    def digest_text(self):
        """The compact registry shown to the model (available capabilities only; unavailable ones are surfaced by retrieval when the question is about them)."""
        by_cat = {}
        for c in self.items:
            if c.kind == "signal" and c.available: by_cat.setdefault(c.category, []).append(c.digest)
        lines = ["SIGNALS:"] + [f"{cat}: " + ", ".join(v) for cat, v in by_cat.items()]
        lines.append("EVENTS: " + ", ".join(c.digest for c in self.items if c.kind == "event" and c.available))
        lines.append("CONDITIONS: " + ", ".join(c.digest for c in self.items if c.kind == "condition"))
        lines.append("TRIP CLASSES: " + ", ".join(c.digest for c in self.items if c.kind == "tripclass") + "; RANK BY: " + ", ".join(sorted({c.digest for c in self.items if c.kind == "metric"})))
        lines.append("TOOLS: " + ", ".join(c.digest for c in self.items if c.kind == "tool"))
        return "\n".join(lines)

    def signature(self):
        h = hashlib.sha256()
        for c in self.items: h.update(f"{c.id}|{c.kind}|{c.available}|{c.digest}|{c.text}|{c.card}".encode())
        return h.hexdigest()[:16]

    def ids(self, kind): return [c.id for c in self.items if c.kind == kind and (c.available or kind != "signal")]
    def signal_ids(self): return [c.id for c in self.items if c.kind == "signal"]             # known concepts; an unavailable one is answered with the explained limitation, never silently substituted
    def event_ids(self): return [c.id for c in self.items if c.kind == "event"]
    def condition_ids(self): return [c.id for c in self.items if c.kind == "condition"]
