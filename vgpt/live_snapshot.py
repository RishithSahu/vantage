"""One authoritative view of the vehicle at a requested instant.

The head-unit and the question executor must not independently decide which
row means "now".  This module owns that decision and returns ``None`` when no
sample is close enough; a missing live value is never converted to zero.
"""
from . import qa_core as Q
import math


CONCEPT_TO_STATE = {
    "speed": "speed",
    "battery_soc": "soc",
    "battery_power": "power",
    "battery_temperature": "batt",
    "coolant_flow": "flow",
    "ambient_temperature": "amb",
    "hvac_power": "hvac",
    "motor_temperature": "motor",
    "drive_mode": "mode",
}


LIVE_BUS = None                    # the application attaches the live bus (vgpt/livebus.py) while a source is delivering: the present is then read from what is arriving, not from what has been stored


def snapshot(conn, ts):
    """The vehicle at `ts`.  For the present (the live moment, or up to 20 s before it) the readings that are arriving on the bus replace the stored ones; the dataset answers for everything older."""
    base = _stored(conn, ts); bus = LIVE_BUS
    if bus is None or not bus.covers(ts): return base
    live = bus.values(CONCEPT_TO_STATE, ts)
    if not live: return base
    values = dict(base["values"]) if base["available"] else {c: None for c in CONCEPT_TO_STATE}
    for c, r in live.items(): values[c] = r["value"]
    newest = max(r["ts"] for r in live.values())
    return dict(base, available=True, values=values, sample_ts=int(newest), age_s=max(0, int(ts) - int(newest)), tier="live", live=True, live_concepts=sorted(live))


def _stored(conn, ts):
    """Return the canonical live snapshot at ``ts``, read from the stored dataset.

    ``available`` is false when neither a 1 Hz sample within five seconds nor
    a minute sample within sixty seconds exists.  Consumers may show an
    unknown state, but must not silently substitute zero or an old value.
    """
    requested = int(ts)
    state = Q.state_at(conn, requested)
    if state.get("gap") or "sample_ts" not in state:
        return {
            "available": False,
            "requested_ts": requested,
            "sample_ts": None,
            "age_s": None,
            "tier": None,
            "values": {},
            "prev": state.get("prev"),
            "next": state.get("next"),
        }
    if state.get("tier") != "1 Hz":                                                    # idempotence: a snapshot is identified by its sample_ts, so asking for that instant must give the same values.
        again = Q.state_at(conn, int(state["sample_ts"]))                              # A minute row whose timestamp is also a 1 Hz row is that 1 Hz sample, not the minute mean.
        if again.get("tier") == "1 Hz" and int(again.get("sample_ts", -1)) == int(state["sample_ts"]): state = again
    sample_ts = int(state["sample_ts"])
    def valid(value):
        return None if isinstance(value, float) and not math.isfinite(value) else value

    values = {concept: valid(state.get(key)) for concept, key in CONCEPT_TO_STATE.items()}
    return {
        "available": True,
        "requested_ts": requested,
        "sample_ts": sample_ts,
        "age_s": max(0, requested - sample_ts),
        "tier": state.get("tier"),
        "values": values,
        "gps": state.get("gps"),
        "ext": state.get("ext"),
    }


def reading(conn, ts, concept):
    """Return one canonical live reading, or ``None`` if it is unavailable."""
    if concept not in CONCEPT_TO_STATE:
        return None
    snap = snapshot(conn, ts)
    value = snap["values"].get(concept) if snap["available"] else None
    return None if value is None else dict(snap, concept=concept, value=value)
