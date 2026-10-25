"""Closed set of semantic TOPICS: whole-question intents that the typed QueryPlan schema does not decompose (health overview, fault priority, advice, energy / distance
estimates, relations between concepts, ...).  The learned parser emits `op=topic p.topic=<id>` (+ typed arguments tgt / tgt2 / t / p.v1 / p.v2 / p.n); this module turns the plan into
the executor's CANONICAL request and the deterministic route handler computes the evidence-backed answer.  The canonical phrases are ours (the executor's input, never the user's
wording); bench/topic_canon_check.py asserts that each one reaches its own route.

Honest note: the route handlers behind a topic still contain keyword logic for their canonical phrase and arguments.  Migrating them to typed arguments is separate work.
"""
from . import signal_registry as R

# id -> (template, argument names).  Arguments: sig, sig2 (concept labels), v1, v2 (numbers), n (days)
TOPICS = {
    "current_soc": ("what is my battery percentage right now", ()),
    "current_power": ("how much power is the vehicle drawing at the moment", ()),
    "current_inventory": ("show me all the current readings", ()),
    "motion_state": ("is the vehicle moving at the moment", ()),
    "latest_valid": ("when was the most recent valid speed sample recorded", ()),
    "acceleration_bands": ("compare acceleration between the speed bands of {v1} to {v2} and {v2} to {v3} km/h", ("v1", "v2", "v3")),
    "consumption_per_100km": ("what is my energy use per 100 km", ()),
    "distance_to_energy": ("how much energy does a {v1} km trip need at my efficiency", ("v1",)),
    "energy_to_distance": ("how far does {v1} kWh take me", ("v1",)),
    "battery_equiv": ("how many {v1} mAh batteries at {v2} V equal the remaining usable energy of the car", ("v1", "v2")),
    "charge_interval": ("how much energy is needed to charge from {v1}% to {v2}%", ("v1", "v2")),
    "semantic_relation": ("what is the difference between {sig} and {sig2}", ("sig", "sig2")),
    "city_highway_energy": ("compare city and highway energy use over the same period", ()),
    "temporal_duration": ("how long was the vehicle driving versus charging", ()),
    "daily_series": ("show me the trend in {sig} day by day for the last {n} days", ("sig", "n")),
    "component_unavailable": ("is the {sig} okay", ("sig",)),
    "drive_modes": ("how many drive modes does the vehicle have", ()),
    "vehicle_health": ("is the vehicle healthy overall", ()),
    "diagnostic_priority": ("what needs attention first", ()),
    "confirmed_active_faults": ("what are the three most serious confirmed active faults", ()),
    "fault_recency_severity": ("which fault is most recent and which is most severe", ()),
    "warning_comparison": ("is the coolant flow warning more important than the tyre pressure warning", ()),
    "safety_assessment": ("is it safe to drive with this warning", ()),
    "cooling_evidence": ("what evidence is there of a battery cooling problem", ()),
    "temporal_events": ("did the over-temperature warnings happen while the vehicle was parked or moving", ()),
    "tyre_status": ("is the tyre pressure problem still there", ()),
    "briefing": ("brief me", ()),
    "battery_now": ("how is the battery doing at present", ()),
    "health_evidence": ("why is the vehicle considered unhealthy", ()),
    "brake_condition": ("what is the condition of the brakes", ()),
    "capability": ("what is the maximum speed of the vehicle", ()),
    "soh_change": ("how much has the battery degraded since new", ()),
    "efficiency_advice": ("how can I improve my efficiency while driving", ()),
    "efficiency_trend": ("is the vehicle becoming less efficient", ()),
    "unsupported_precision": ("tell me the exact remaining lifespan of the battery", ()),
    "missing_inputs": ("how much CO2 have I saved against a petrol car", ()),
    "parts_advice": ("which replacement tyre should I buy", ()),
    "last_charge": ("when did I last plug the car in", ()),
    "charging_now": ("how fast is it charging right now", ()),
}


def canonical_request(topic, tgt=None, tgt2=None, v1=None, v2=None, n=None):
    """-> canonical request text for the topic, or None when the topic is unknown / an argument it needs is missing."""
    ent = TOPICS.get(topic)
    if ent is None: return None
    tpl, need = ent
    val = {"sig": R.label_of(tgt) if tgt in R.CONCEPTS else None, "sig2": R.label_of(tgt2) if tgt2 in R.CONCEPTS else None, "v1": _num(v1), "v2": _num(v2), "n": _num(n)}
    if topic == "acceleration_bands":
        val["v3"] = _num(float(v2) + 10) if v2 is not None else None
    for k in need:
        if val.get(k) in (None, ""): return None
    return tpl.format(**{k: val.get(k) for k in list(need) + ["v3"] if k in val})


def _num(v):
    try: f = float(v)
    except (TypeError, ValueError): return None
    return f"{f:g}"
