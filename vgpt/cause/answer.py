"""The answer to "why ...?": the cause engine's diagnosis, in the words of a person, with the evidence and the limits.

`maybe(res, c, frame, lit, text)` is called by the resolver for a question that asks for a cause.  It returns the assistant's usual result (answer text, brief, facts) or None when the question is not one this module
answers (then the older explanation routes run as before)."""
import logging
import sqlite3

from . import engine as E
from . import events as EV
from .data import History

# which causes can explain a symptom the driver names (None = any)
FOCUS = {
    "range_estimate": {"added_drag", "rolling_resistance_increase", "cold_weather", "aggressive_driving", "climate_overuse", "higher_speeds", "battery_aging", "tyre_slow_leak"}, "reported_range": None,
    "wh_per_km": {"added_drag", "rolling_resistance_increase", "cold_weather", "aggressive_driving", "climate_overuse", "higher_speeds", "tyre_slow_leak"},
    "energy_used": {"added_drag", "rolling_resistance_increase", "cold_weather", "aggressive_driving", "climate_overuse", "higher_speeds", "tyre_slow_leak"},
    "coolant_flow": {"cooling_pump_degradation"}, "battery_temperature": {"cooling_pump_degradation", "cold_weather"}, "motor_temperature": {"cooling_pump_degradation"},
    "tyre_pressure": {"tyre_slow_leak"}, "tyre_pressure_fl": {"tyre_slow_leak"}, "tyre_pressure_fr": {"tyre_slow_leak"}, "tyre_pressure_rl": {"tyre_slow_leak"}, "tyre_pressure_rr": {"tyre_slow_leak"},
    "aux_battery_voltage": {"weak_12v_battery"}, "camera_blockage": {"camera_obstruction"}, "camera_sharpness": {"camera_obstruction"}, "battery_soh": {"battery_aging"},
    "hvac_power": {"climate_overuse", "cold_weather"}, "battery_soc": {"added_drag", "rolling_resistance_increase", "cold_weather", "aggressive_driving", "climate_overuse", "higher_speeds", "battery_aging"},
}
SYMPTOM_WORD = {"range_estimate": "driving range", "wh_per_km": "energy use per kilometre", "energy_used": "energy use", "coolant_flow": "coolant flow", "battery_temperature": "battery temperature", "motor_temperature": "motor temperature",
                "tyre_pressure": "tyre pressure", "aux_battery_voltage": "12 V battery voltage", "camera_blockage": "camera blockage", "battery_soh": "battery health", "hvac_power": "climate power", "battery_soc": "battery charge"}


CONSUMPTION_SYMPTOMS = {"range_estimate", "wh_per_km", "energy_used"}


def _consumption_moved(F):
    """Did energy per kilometre (hence range) really change, by the raw numbers or by what the recorded conditions do not explain?"""
    wh, raw, m = F.get("wh"), F.get("wh_raw"), F.get("wh_model")
    return bool((wh is not None and wh.ok) or (raw is not None and raw.ok) or (m and m.get("ok")) or ((m or {}).get("window") and abs(m["window"]["gap"]) >= 0.04 * m["base"] and abs(m["window"]["explained"]) >= 0.5 * abs(m["window"]["gap"])))


def _words(h): return h["title"]


def _pct(x): return f"{round(100 * x)} %"


def compose_change(res, symptom_label, horizon_days, label="the last days"):
    """-> text for a change diagnosis (res = engine.diagnose output, already filtered for the symptom)."""
    hy = res["hypotheses"]; top = hy[0] if hy else None; lines = []
    if res["nothing_to_explain"] or top is None or top["confidence"] < 0.35:
        lines.append(f"Nothing changed enough in {label} to need a cause" + (f" for the {symptom_label}" if symptom_label else "") + ": every quantity I can check stayed within its usual day-to-day variation, and no known cause fits better than no cause.")
        if res.get("older_changes"): lines.append("Earlier, before that period, these did change: " + ", ".join(sorted(set(res["older_changes"]))[:4]) + ".")
    else:
        since = f" since about {top['onset_date']}" if top.get("onset_date") else ""
        qual = "Most likely cause" if top["confidence"] >= 0.7 else "Most plausible cause"
        lines.append(f"{qual}: {top['title']}{since} (confidence {_pct(top['confidence'])}).")
        lines.append("Why: " + top["mechanism"])
        if top["notes"]: lines.append("Evidence:\n" + "\n".join(" - " + n for n in top["notes"][:4]))
        others = [h for h in hy[1:4] if h["confidence"] >= 0.08]
        if others: lines.append("Also considered: " + "; ".join(f"{_words(h)} ({_pct(h['confidence'])})" for h in others) + ".")
        lines.append("To confirm: " + top["check"])
        if top["confidence"] < 0.7: lines.append("The data do not single this out strongly; treat it as the leading explanation, not as established.")
    if res.get("other"): lines.append("Also noticed (not about this question): " + "; ".join(f"{h['title']} (confidence {_pct(h['confidence'])})" for h in res["other"][:2]) + ".")
    if res.get("missing"): lines.append("Could not be checked because the vehicle does not record it: " + ", ".join(res["missing"][:4]) + ".")
    return "\n".join(lines)


def compose_event(r):
    ev = r["event"]; lines = []; top = r["causes"][0] if r["causes"] else None
    if top is None:
        lines.append(f"The {ev['type'].replace('_', ' ').lower()} at that time ({ev['detail'][:80]}) has no recorded cause: nothing logged or measured around it points to one, so I will not guess.")
        return "\n".join(lines)
    lines.append(f"Most likely cause: {top['title']} (confidence {_pct(top['confidence'])}).")
    lines.append("Why: " + top["mechanism"])
    if len(r["chain"]) > 1:
        lines.append("What happened, in order: " + " -> ".join(f"{c['type'].replace('_', ' ').lower()} ({c['t']:+d} s)" for c in r["chain"]) + ".")
    if top["evidence"]: lines.append("Evidence:\n" + "\n".join(" - " + e for e in top["evidence"][:4]))
    others = [c for c in r["causes"][1:3] if c["confidence"] >= 0.4]
    if others: lines.append("Also possible: " + "; ".join(f"{c['title']} ({_pct(c['confidence'])})" for c in others) + ".")
    lines.append("To confirm: " + top["check"])
    return "\n".join(lines)


def maybe(res, c, frame, lit, text):
    """The cause of a change or of an event.  Called for act 'why' (one day / one trip questions stay with the digest routes)."""
    if frame.act != "why" or frame.ent in ("trip", "day"): return None
    from ..meaning.resolve import _result
    b = res.brain
    try: conn = b.c
    except AttributeError: return None
    try:
        if c.kind == "event":
            etype = getattr(c, "registry_key", None)
            if not etype: return None
            now = int(b.now); lo, hi = 0, now + 1
            if lit.times:                                                                                               # a period named in the question: the latest such event inside it
                try:
                    from ..plan_exec import resolve_time
                    lo, hi, _ = resolve_time(lit.times[0], b.qc().ctx); hi = min(int(hi), now + 1)
                except Exception: pass
            row = conn.execute("SELECT ts FROM events WHERE type = ? AND ts >= ? AND ts < ? ORDER BY ts DESC LIMIT 1", (etype, int(lo or 0), hi)).fetchone()
            if not row: return None                                                                                      # none in that period: the ordinary routes say so
            r = EV.explain_event(conn, row[0], etype)
            if r is None: return None
            body = compose_event(r); stamp = res.brain.__class__ and ""
            text_out = f"About the {c.label if hasattr(c, 'label') else etype.replace('_', ' ').lower()} on {_date(b, row[0])}:\n" + body
            return _result(text_out, "cause:event", "cause", frame, cause={"kind": "event", "top": (r["causes"][0]["id"] if r["causes"] else None), "chain": r["chain"], "causes": r["causes"]})
        H = History(conn, 84, now=int(b.now)); F = E.build(H); since = H.D - 14; label = "the last 14 days"
        if lit.times:
            try:
                from ..plan_exec import resolve_time
                lo, hi, label = resolve_time(lit.times[0], b.qc().ctx); since = max(0, H.day_of(lo)) if lo else since
            except Exception: pass
        d = E.diagnose(H, F=F, since_day=since)
        focus = FOCUS.get(c.id, None) if c.id in FOCUS else None
        d = dict(d)
        if c.id in CONSUMPTION_SYMPTOMS and not _consumption_moved(F):                                                # a cause for a number that did not move would be a story, not a finding
            d["hypotheses"] = [h for h in d["hypotheses"] if h["id"] not in focus]; d["other"] = [h for h in d["hypotheses"] if h["confidence"] >= 0.5]; d["nothing_to_explain"] = True; focus = None
        if focus is not None:
            rel = [h for h in d["hypotheses"] if h["id"] in focus]; oth = [h for h in d["hypotheses"] if h["id"] not in focus and h["confidence"] >= 0.5]
            d["hypotheses"] = rel; d["other"] = oth
            if not rel or rel[0]["confidence"] < 0.35: d["nothing_to_explain"] = True
        text_out = compose_change(d, SYMPTOM_WORD.get(c.id, ""), H.D - since, label)
        top = d["hypotheses"][0] if d["hypotheses"] else None
        return _result(text_out, "cause:change", "cause", frame, cause={"kind": "change", "top": top["id"] if top and not d["nothing_to_explain"] else None, "confidence": top["confidence"] if top else None, "p_no_cause": d["p_no_known_cause"],
                                                                         "onset": top["onset_date"] if top else None, "hypotheses": [{k: h[k] for k in ("id", "confidence", "evidence", "onset_date")} for h in d["hypotheses"][:4]], "days": H.D})
    except Exception as e:
        logging.getLogger("vehiclegpt").warning("cause engine failed: %s: %s", type(e).__name__, e); return None


def _date(b, ts):
    try:
        from .. import analytics as A
        return A.stamp(int(ts))
    except Exception: return str(int(ts))
