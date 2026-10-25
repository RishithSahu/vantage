"""The analyses: one function per kind of investigation.  Each takes the parsed request and the data handle, runs typed computations over the recorded history and returns a Report.

Principles shared by all of them
  * what is RECORDED (an event, a measured value) is stated as fact with its timestamp and source; what is only ASSOCIATED (order, lead time, co-occurrence) is labelled "association, not proof";
    a POSSIBLE explanation is offered only as a hypothesis with the check that would confirm it;
  * resolution is stated (1 Hz frozen context, 1-minute averages) and never over-claimed;
  * what is missing (no data, a gap, a channel the vehicle does not record, a cause nobody measured) is listed;
  * an answer never contains a number that was not computed from the history."""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from .. import signal_registry as R
from . import parse as P
from .primitives import Departure, Event, effect, robust_z
from .report import Report

LEX = P.LEX
FAM = LEX["default_signals"]; EVFAM = LEX["event_family"]
EVLABEL = lambda t: R.EVENTS[t][0] if t in R.EVENTS else t


def hms(D, ts): return D.stamp(ts)[11:] if len(D.stamp(ts)) > 11 else D.stamp(ts)
def dur(s):
    s = abs(float(s)); return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min {s % 60:.0f} s" if s < 3600 else f"{s / 3600:.1f} h" if s < 172800 else f"{s / 86400:.1f} days"
def fmtv(v, unit, nd=1): return f"{v:,.{nd}f} {unit}".strip()


# ------------------------------------------------------------------------------------------------ anchors
CONTEXT_ONLY = {"speed", "accelerator_position", "brake_pedal", "steering_angle", "odometer", "ambient_temperature", "light", "battery_soc"}      # operating context: described, but not offered as "the abnormal signal" unless the request asks


def anchor_types(req):
    """The event the request is about: the first event it names (the later ones are the sequence it describes), else any warning."""
    return [req.events[0]] if req.events else ["ANY_WARNING"]


def pick_anchor(req, D, types, focus=None):
    """-> (Event | None, note | None).  The anchor event a request means: the latest / first / most severe / rarest of the named types, the one the conversation is about, or the one at a stated time."""
    evs = D.events(types, D.t0, D.end + 1)
    if not evs and "ANY_WARNING" not in types: evs = D.events(["ANY_WARNING"], D.t0, D.end + 1)
    if not evs: return None, None
    kind = req.anchor_kind; note = None
    if req.clock_ts:
        near = min(evs, key=lambda e: abs(e.ts - req.clock_ts))
        if abs(near.ts - req.clock_ts) <= 1800: return near, None
        note = (f"No {EVLABEL(near.type)} is recorded at {D.stamp(req.clock_ts)[11:]} on {D.stamp(req.clock_ts)[:10]}; the closest one is at {D.stamp(near.ts)}, {dur(near.ts - req.clock_ts)} {'earlier' if near.ts < req.clock_ts else 'later'}. I analysed that one instead.")
        return near, note
    if kind == "focus" and focus: return min(evs, key=lambda e: abs(e.ts - focus)), None
    if kind == "critical":
        crit = [e for e in evs if e.sev == "critical"]; return (crit or evs)[-1], None
    if kind == "rare":
        recent = [e for e in evs if e.ts >= D.end - 30 * 86400] or evs; cnt = Counter(e.type for e in recent); rare = min(cnt, key=lambda t: (cnt[t], -max(e.ts for e in recent if e.type == t))); return [e for e in recent if e.type == rare][-1], None
    if kind == "first": return evs[0], None
    return evs[-1], None


def signals_for(req, D, anchor_type, extra=()):
    asked = [c for c in req.concepts if D.available(c) and c in R.STORE and c not in ("wh_per_km", "range_estimate")]
    fam = FAM.get(EVFAM.get(anchor_type, "generic"), FAM["generic"]); out = []
    for c in asked + list(extra) + fam:
        if c not in out and D.available(c) and c in R.STORE: out.append(c)
    return out[:10], asked


# ------------------------------------------------------------------------------------------------ incident reconstruction (also: recovery, spike or trend)
def incident(req, D, focus=None, variant="incident"):
    before = req.windows.get("before", 600.0); after = req.windows.get("after", 300.0); types = anchor_types(req)
    anchor, note = pick_anchor(req, D, types, focus)
    rep = Report("", "incident")
    if anchor is None:
        rep.title = "I found no recorded event to anchor the reconstruction on."; rep.ok = False
        rep.add("What I looked for", ", ".join(EVLABEL(t) for t in D.expand(types)) or "any warning"); return rep
    rep.premise = note; ta = anchor.ts; lo, hi = ta - before, ta + after
    rep.title = f"{'Recovery check' if variant == 'recovery' else 'Incident reconstruction'}: {EVLABEL(anchor.type)} on {D.stamp(ta)} - {anchor.detail}"
    rep.focus_ts = ta; rep.focus_event = anchor.type; rep.facts.update(event_ts=ta, event_type=anchor.type, window=[int(lo), int(hi)])
    sigs, asked = signals_for(req, D, anchor.type); ser = {}
    for c in sigs:
        s = D.series(c, lo, hi, anchor)
        if s is not None and len(s.t) >= 3: ser[c] = s
    rep.step(f"anchor: {anchor.type} at {ta} ({req.anchor_kind or 'latest'}); window {dur(before)} before to {dur(after)} after"); rep.step(f"signals read: {', '.join(ser)}")
    # what the signals did
    lines = []; res_used = Counter()
    for c, s in ser.items():
        pre = s.v[s.t < ta]; post = s.v[s.t >= ta]; at = s.v[np.argmin(np.abs(s.t - ta))]; unit = D.unit(c); res_used[s.label()] += 1
        a = fmtv(float(pre[0]), unit) if len(pre) else "n/a"; z = fmtv(float(post[-1]), unit) if len(post) else "no data after"
        peak = f", peak {fmtv(float(np.max(np.r_[pre, post])), unit)}" if c in ("battery_temperature", "motor_temperature", "battery_current", "speed") else (f", lowest {fmtv(float(np.min(np.r_[pre, post])), unit)}" if c in ("coolant_flow", "aux_battery_voltage", "battery_voltage", "lane_confidence", "lead_vehicle_distance") else "")
        lines.append(f"{D.label(c)}: {a} -> {fmtv(float(at), unit)} at the event -> {z}{peak}" + ("  [asked for]" if c in asked else ""))
    rep.add(f"What the signals did ({dur(before)} before to {dur(after)} after; the event itself is the anchor)", *lines)
    # earliest abnormal
    deps = []
    for c, s in ser.items():
        if c in CONTEXT_ONLY and c not in asked: continue
        d = D.departure(s, ta)
        if d: deps.append(d)
    deps.sort(key=lambda d: d.ts)
    if deps:
        rep.add("Earliest abnormal signals (association in time, not proof of cause)", *[f"{hms(D, d.ts)}: {D.label(d.concept)} {'rose' if d.direction == 'up' else 'fell'} {fmtv(abs(d.magnitude), d.unit)} from its own baseline {fmtv(d.baseline, d.unit)} - {dur(ta - d.ts)} before the event ({R_label(d.res)})" for d in deps[:6]])
    else: rep.add("Earliest abnormal signals", "no signal left its own baseline in the window at the available resolution")
    rep.facts["departures"] = [(d.concept, d.ts, d.direction) for d in deps]
    # other events in the window
    others = [e for e in D.all_events(lo, hi) if not (e.ts == anchor.ts and e.type == anchor.type)]
    if others: rep.add("Other recorded events in the window (by time)", *[f"{hms(D, e.ts)} {EVLABEL(e.type)}: {e.detail}" for e in others[:12]])
    # the sequence the request describes (anchor type followed by the others it names)
    seq = [t for t in req.events if t not in (anchor.type, "ANY_WARNING") and t not in D.expand([anchor.type])][:3]
    if seq:
        found = []; cur = ta
        for t in seq:
            nxt = next((e for e in D.events([t], cur, cur + 1800)), None)
            found.append(f"{EVLABEL(t)}: " + (f"recorded {dur(nxt.ts - cur)} later at {hms(D, nxt.ts)}" if nxt else "not recorded within 30 minutes")); cur = nxt.ts if nxt else cur
        rep.add("The sequence you describe, checked against the record", *found)
    # recovery
    if variant == "recovery" and ser:
        rows = []
        for c, s in ser.items():
            pre = s.v[(s.t < ta - 60)]; post = s.v[s.t >= ta + 120]
            if len(pre) < 4 or len(post) < 3: continue
            e = effect(post, pre)
            if e: rows.append((abs(e["d"]), c, e))
        rows.sort(reverse=True)
        rep.add("Compared with the baseline before the event", *[f"{D.label(c)}: {fmtv(e['b'], D.unit(c))} before, {fmtv(e['a'], D.unit(c))} after the event - " + ("still abnormal" if abs(e["d"]) > 2.5 else "back near baseline") for _, c, e in rows[:8]])
    if anchor.type == "UNEXPECTED_SHUTDOWN":
        last = []
        for c, s in ser.items():
            pre = s.t < ta
            if pre.any(): last.append(f"{D.label(c)}: {fmtv(float(s.v[pre][-1]), D.unit(c))} at {hms(D, float(s.t[pre][-1]))}")
        nxt = [t for s in ser.values() for t in s.t[s.t >= ta][:1]]; resumed = min(nxt) if nxt else None
        rep.add("Last valid readings before the shutdown", *last)
        rep.add("Data after the shutdown", f"no samples for {dur(resumed - ta)} (until {hms(D, resumed)})" if resumed and resumed - ta > 3 else "samples resume immediately" if resumed else "no samples after the event in the window")
    # explanation
    hyp = hypotheses(D, deps, others, anchor)
    rep.add("Possible explanations (hypotheses, not confirmed)", *hyp)
    miss = missing(D, ser, sigs, asked, lo, hi, anchor, req)
    rep.add("Not available / cannot be established from the record", *miss)
    rep.add("Confirmed from the record", f"{EVLABEL(anchor.type)} at {D.stamp(ta)}: {anchor.detail} (severity {anchor.sev})", *[f"{EVLABEL(e.type)} at {D.stamp(e.ts)}: {e.detail}" for e in others[:3]])
    rep.evidence = {"period": f"{D.stamp(lo)} to {D.stamp(hi)}", "period_ts": [int(lo), int(hi)], "samples": int(sum(len(s.t) for s in ser.values())), "sources": sorted(res_used), "tier": ", ".join(sorted(res_used)) or "none"}
    return rep


def R_label(res): return "1 Hz" if res <= 1.5 else "1-minute averages" if res >= 30 else f"{res:g} s steps"


def hypotheses(D, deps, others, anchor):
    out = []; seen = set()
    for d in deps[:4]:
        for h in LEX["hypotheses"]:
            if h["signal"] == d.concept and h["direction"] == d.direction and h["say"] not in seen:
                seen.add(h["say"]); out.append(f"{h['say']} (because {D.label(d.concept)} {'rose' if d.direction == 'up' else 'fell'} first); to confirm: {h['check']}")
    for e in others[:6]:
        if e.type == "VOLTAGE_DIP" and "12 V supply instability" not in " ".join(out): out.append("12 V supply instability (a voltage dip is recorded in the window); to confirm: 12 V battery rest voltage, DC/DC converter DTCs")
        if e.type == "ECU_RESET" and "controller restart" not in " ".join(out): out.append(f"a controller restart ({e.detail}) disturbing dependent functions; to confirm: that unit's reset cause and boot log")
        if e.type in ("CAN_ERROR", "SENSOR_CONFLICT") and "data path" not in " ".join(out): out.append("a data-path problem (bus errors or conflicting sources) rather than a physical change; to confirm: the bus error counters and the second source's raw value")
    return out or ["no hypothesis is supported by the recorded signals; this does not mean there is no cause, only that the vehicle did not record one"]


def missing(D, ser, sigs, asked, lo, hi, anchor, req):
    out = []
    for c in req.concepts:
        if c in R.CONCEPTS and R.CONCEPTS[c].kind != "derived" and not D.available(c) and c not in ("wh_per_km", "range_estimate"): out.append(f"{D.label(c)} is not recorded by this vehicle")
    for c in sigs:
        if c not in ser: out.append(f"{D.label(c)}: no samples in the window")
    coarse = [D.label(c) for c, s in ser.items() if s.res >= 30]
    if coarse: out.append("only 1-minute averages exist for " + ", ".join(coarse) + " (that moment is older than the 1 Hz history), so brief spikes could be hidden")
    short = [D.label(c) for c, s in ser.items() if s.res < 30 and s.span() < 0.7 * (hi - lo)]
    if short: out.append("the frozen 1 Hz context of this event covers only part of the window for " + ", ".join(short[:4]) + (" ..." if len(short) > 4 else "") + "; the rest is not available at 1 Hz")
    gaps = [(c, s) for c, s in ser.items() if s.res <= 1.5 and len(s.t) > 10 and np.max(np.diff(s.t)) > 5]
    for c, s in gaps[:3]: out.append(f"{D.label(c)} has a gap of {dur(float(np.max(np.diff(s.t))))} inside the window")
    if D.c.execute("SELECT COUNT(*) FROM logs WHERE ts>=? AND ts<?", (int(lo), int(hi))).fetchone()[0] == 0: out.append("no warning-level log line from any service in the window")
    out.append("the cause itself was not measured by any sensor; the explanations above are inferred from timing")
    return out


def spike_or_trend(req, D, focus=None):
    """Was the latest abnormal reading a brief spike or part of a sustained trend?"""
    return incident(req, D, focus)
