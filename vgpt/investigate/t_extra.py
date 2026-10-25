"""Further analyses: best operating conditions, the last N occurrences of a fault, the most important open question, an event summary, spike versus trend, status changes without signal change, and the
ADAS variants (several disengagements in one trip, the most unusual incidents)."""
import math
from collections import Counter, defaultdict

import numpy as np

from .. import signal_registry as R
from .primitives import effect
from .report import Report
from .t_change import _fmt
from .t_events import SIGNAL_OF_CODE, BORING, cluster, dtc_code, period, specific, subsystem_of
from .t_trips import FEATS, feat_label, feat_nd, feat_unit, trips_feat
from .templates import CONTEXT_ONLY, EVLABEL, FAM, LEX, dur, fmtv, hms, incident, pick_anchor, signals_for

NEXT = LEX.get("next_measurement", {})


# ================================================================================================ best operating conditions
def best_conditions(req, D, focus=None):
    lo, hi, label = period(req, D, 90); rep = Report("", "best_conditions"); trips = trips_feat(D, lo, hi, 1.0)
    if len(trips) < 20: rep.title = f"Only {len(trips)} trips {label}: too few to say where efficiency is best."; rep.ok = False; return rep
    allmed = float(np.median([t["wh_km"] for t in trips])); dims = {
        "road type": lambda t: "city (under 20 % motorway)" if t["hwy_share"] < 0.2 else "motorway (over 60 %)" if t["hwy_share"] > 0.6 else "mixed",
        "average speed": lambda t: "under 30 km/h" if t["avg_speed"] < 30 else "30-50 km/h" if t["avg_speed"] < 50 else "50-80 km/h" if t["avg_speed"] < 80 else "80-110 km/h" if t["avg_speed"] < 110 else "over 110 km/h",
        "trip length": lambda t: "under 5 km" if t["km"] < 5 else "5-15 km" if t["km"] < 15 else "15-40 km" if t["km"] < 40 else "over 40 km",
        "outside temperature": lambda t: "under 10 C" if t["avg_amb"] < 10 else "10-20 C" if t["avg_amb"] < 20 else "20-28 C" if t["avg_amb"] < 28 else "over 28 C",
        "climate power": lambda t: "under 150 W" if t["avg_hvac_w"] < 150 else "150-600 W" if t["avg_hvac_w"] < 600 else "over 600 W"}
    rep.step(f"{len(trips)} trips {label} binned along {len(dims)} dimensions; overall median {allmed:.0f} Wh/km")
    best_overall = None; blocks = []
    for name, fn in dims.items():
        bins = defaultdict(list)
        for t in trips: bins[fn(t)].append(t["wh_km"])
        rows = [(float(np.median(v)), k, len(v), float(np.mean(np.array(v) < allmed))) for k, v in bins.items() if len(v) >= 6]
        if not rows: continue
        rows.sort(); blocks.append((name, rows))
        if best_overall is None or rows[0][0] < best_overall[0]: best_overall = (rows[0][0], name, rows[0][1])
    rep.title = f"Best efficiency {label}: " + (f"{best_overall[2]} ({best_overall[0]:.0f} Wh/km against {allmed:.0f} overall)" if best_overall else "no condition stands out")
    for name, rows in blocks: rep.add(name, *[f"{k}: {m:.0f} Wh/km median over {n} trips; {c:.0%} of them beat the overall median" + ("  <- best" if i == 0 else "") for i, (m, k, n, c) in enumerate(rows)])
    combo = defaultdict(list)
    for t in trips: combo[(dims["road type"](t), dims["outside temperature"](t))].append(t["wh_km"])
    rows = sorted(((float(np.median(v)), k, len(v)) for k, v in combo.items() if len(v) >= 6))
    if rows: rep.add("Best combination of road type and outside temperature", *[f"{k[0]} at {k[1]}: {m:.0f} Wh/km over {n} trips" for m, k, n in rows[:3]])
    rep.add("Reading this", "a condition is 'consistently' best when most of its trips beat the overall median, not just its average; bins with fewer than 6 trips are not shown", "this is this vehicle's own history, not a general rule")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(trips), "sources": ["trips", "sig_1m"], "tier": "trips + 1-minute"}; return rep


# ================================================================================================ the last N occurrences of a fault
def occurrences(req, D, focus=None):
    n = next((k for k, noun in req.counts if noun in ("occurrence", "instance", "incident")), 5); rep = Report("", "occurrences"); types = specific(req)
    anchor, note = pick_anchor(req, D, types or ["DTC_SET"], focus)
    if anchor is None: rep.title = "No fault is recorded to compare."; rep.ok = False; return rep
    same = [e for e in D.events([anchor.type], D.t0, D.end + 1) if (dtc_code(e) == dtc_code(anchor) if dtc_code(anchor) else True)][-n:]
    name = dtc_code(anchor) or EVLABEL(anchor.type); rep.premise = note
    if len(same) < 2: rep.title = f"{name} occurred only {len(same)} time{'s' if len(same) != 1 else ''}; there is nothing to compare."; return rep
    rows = []; sigs = [c for c in ("speed", "motor_temperature", "battery_temperature", "battery_current", "ambient_temperature", "coolant_flow", "aux_battery_voltage") if D.available(c)]
    for e in same:
        vals = {}
        for c in sigs:
            s = D.series(c, e.ts - 300, e.ts, e); vals[c] = float(np.median(s.v[-30:])) if s is not None and len(s.v) else float("nan")
        trip = next((t for t in D.trips(e.ts - 4 * 3600, e.ts + 1, 0.2) if t["start"] <= e.ts <= t["end"]), None); vals["elapsed"] = (e.ts - trip["start"]) / 60.0 if trip else float("nan"); rows.append((e, vals))
    rep.title = f"The last {len(same)} occurrences of {name}: {D.stamp(same[0].ts)} to {D.stamp(same[-1].ts)}"; rep.step(f"{len(same)} occurrences; conditions are the median of the 30 s before each (1 Hz context where kept, else 1-minute)")
    rep.add("Conditions before each occurrence", *[f"{D.stamp(e.ts)}: " + ", ".join(f"{D.label(c)} {fmtv(v[c], D.unit(c), 0)}" for c in sigs if not math.isnan(v[c])) + (f", {v['elapsed']:.0f} min into the trip" if not math.isnan(v["elapsed"]) else "") for e, v in rows])
    cons = []; chg = []
    for c in sigs:
        x = np.array([v[c] for _, v in rows]); x = x[~np.isnan(x)]
        if len(x) < 3: continue
        spread = (x.max() - x.min()); lo_, hi_ = D.rng(c); rel = spread / max(hi_ - lo_, 1e-9)
        (cons if rel < 0.08 else chg).append(f"{D.label(c)}: {fmtv(float(np.median(x)), D.unit(c), 0)} typical, range {fmtv(float(x.min()), D.unit(c), 0)} to {fmtv(float(x.max()), D.unit(c), 0)}")
    rep.add("Consistent across all of them (narrow range)", *cons or ["none of the compared signals was tightly the same"]); rep.add("What changed between them", *chg or ["nothing varied beyond a few percent of the signal's range"])
    sev = max(same, key=lambda e: ({"critical": 3, "warn": 2, "info": 1}.get(e.sev, 1), e.ts)); mag = {e.ts: max(abs(v["motor_temperature"]) if not math.isnan(v["motor_temperature"]) else 0, 0) for e, v in rows}
    hot = max(rows, key=lambda r: r[1]["motor_temperature"] if not math.isnan(r[1]["motor_temperature"]) else -1)
    rep.add("Which one stands out", f"the recorded severity is the same for all ({sev.sev})" if len({e.sev for e in same}) == 1 else f"most severe by recorded severity: {D.stamp(sev.ts)}", f"hottest motor before it: {D.stamp(hot[0].ts)} at {fmtv(hot[1]['motor_temperature'], 'C', 0)}")
    rep.evidence = {"period": f"{D.stamp(same[0].ts)} to {D.stamp(same[-1].ts)}", "period_ts": [same[0].ts, same[-1].ts], "samples": len(same), "sources": ["events", "signals"], "tier": "event context / 1-minute"}; return rep


# ================================================================================================ the most important open question
def unanswered(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "unanswered"); evs = [e for e in D.events(["ANY_WARNING", "POWER_DERATE"], lo, hi) if e.type not in BORING]
    if not evs: rep.title = f"No fault or warning {label}: there is no open diagnostic question to rank."; rep.ok = False; return rep
    key = lambda e: dtc_code(e) or e.type; cnt = Counter(key(e) for e in evs); clears = {dtc_code(e) for e in D.events(["DTC_CLEAR"], lo, hi)}
    score = {k: cnt[k] * (3 if D.dtc().get(k, {}).get("severity") == "critical" else 1) + (3 if k not in clears and k in D.dtc() else 0) for k in cnt}; top = max(score, key=score.get); es = [e for e in evs if key(e) == top]
    anchor = es[-1]; info = D.dtc().get(top, {}); name = f"{top} ({info['text']})" if info else EVLABEL(anchor.type)
    rep.title = f"Most important open question {label}: why does {name} keep occurring? ({len(es)} occurrences, " + ("never cleared" if top not in clears else "cleared and returned") + ")"
    rep.step(f"ranked {len(cnt)} fault kinds by occurrences x severity + outstanding; chose {top}")
    # observed behaviour: reuse the condition analysis
    from .t_trips import conditions as cond
    class R_: pass
    r = R_(); r.__dict__.update(req.__dict__); r.events = [anchor.type]; r.times = req.times; r.windows = req.windows; r.counts = []; sub = cond(r, D, focus)
    rep.add("Observed behaviour", *[ln for h, items in sub.lines[:2] for ln in items][:5] if sub.lines else ["too few occurrences to characterise"])
    ex = [x for h, items in sub.lines for x in items]
    rep.add("Evidence already available", f"{len(es)} dated occurrences, 1 Hz frozen context around each, the vehicle's DTC text and severity", "operating conditions at each occurrence compared with ordinary driving (above)", f"{len(D.log_lines(lo, hi, limit=50))} service log lines in the period")
    sig = SIGNAL_OF_CODE.get(top); hyp = []
    for h in LEX["hypotheses"]:
        if h["signal"] in ("motor_temperature", "aux_battery_voltage", "coolant_flow", "battery_temperature") and (sig is None or h["signal"] == sig or h["signal"] == "motor_temperature"): hyp.append(f"{h['say']}; to confirm: {h['check']}")
    rep.add("Competing explanations (hypotheses)", *hyp[:3] or ["no explanation is supported by a recorded signal"])
    nm = NEXT.get(top) or NEXT.get(anchor.type) or "the output side measurements of the unit that raised it (not recorded by this vehicle)"
    rep.add("The single measurement that would help most", nm)
    rep.add("Not claimed", "no root cause is claimed: the record shows when and under which conditions it happens, not why")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(es), "sources": ["events", "sig_1m", "logs"], "tier": "events + context"}; return rep


# ================================================================================================ summary of one kind of event
def event_summary(req, D, focus=None):
    lo, hi, label = period(req, D, 28); rep = Report("", "event_summary"); types = specific(req) or ["ANY_WARNING"]; evs = D.events(types, lo, hi)
    if not evs: rep.title = f"No {', '.join(EVLABEL(t) for t in D.expand(types)[:3])} was recorded {label}."; rep.ok = False; return rep
    rep.title = f"{len(evs)} {EVLABEL(evs[0].type)}{'s' if len(evs) != 1 else ''} {label}; first {D.stamp(evs[0].ts)}, last {D.stamp(evs[-1].ts)}"
    days = Counter(D.day(e.ts) for e in evs); hours = Counter((e.ts + D.tz) // 3600 % 24 for e in evs); groups = cluster(evs, 1800)
    rep.add("Repeated patterns", f"{len(groups)} separate episodes (events closer than 30 minutes belong to one); the busiest day was {days.most_common(1)[0][0]} with {days.most_common(1)[0][1]}", f"most occur at {', '.join(f'{h:02d}:00' for h, _ in hours.most_common(2))}")
    mags = []; chan = {"WARN_BATT_OVERTEMP": "battery_temperature", "WARN_OVERTEMP": "motor_temperature", "TPMS_LOW": "tyre_pressure_fl"}
    for e in evs:
        c = chan.get(e.type); s = D.series(c, e.ts - 300, e.ts + 300, e) if c and D.available(c) else None
        if s is not None and len(s.v):
            lowside = e.type == "TPMS_LOW"; mags.append((float(-np.min(s.v)) if lowside else float(np.max(s.v)), e))                                  # magnitude = the peak (or trough) of the channel behind it
        else:
            m = [float(x) for x in "".join(ch if (ch.isdigit() or ch == '.') else ' ' for ch in e.detail).split() if x.replace('.', '', 1).isdigit()]; mags.append((m[0] if m else 0.0, e))
    sev = sorted(mags, key=lambda r: (-{"critical": 3, "warn": 2, "info": 1}.get(r[1].sev, 1), -r[0]))
    rep.add("Most and least severe (by recorded severity, then by the peak of the measured channel)", f"most severe: {D.stamp(sev[0][1].ts)} - {sev[0][1].detail} (severity {sev[0][1].sev}, peak {sev[0][0]:.1f})", f"least severe: {D.stamp(sev[-1][1].ts)} - {sev[-1][1].detail} (severity {sev[-1][1].sev}, peak {sev[-1][0]:.1f})")
    conds = []
    for pick, name in ((sev[0][1], "most severe"), (sev[-1][1], "least severe")):
        vals = []
        for c in ("battery_temperature", "coolant_flow", "speed", "ambient_temperature"):
            if D.available(c):
                s = D.series(c, pick.ts - 300, pick.ts, pick); vals.append(f"{D.label(c)} {fmtv(float(np.median(s.v[-30:])), D.unit(c), 0)}" if s is not None and len(s.v) else "")
        conds.append(f"{name}: " + ", ".join(v for v in vals if v))
    rep.add("Conditions before them", *conds)
    rep.add("Measurements that would help distinguish the leading explanations", NEXT.get(evs[0].type, "cell-level temperatures and the cooling pump's own current and speed (not recorded by this vehicle)"))
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evs), "sources": ["events", "signals"], "tier": "events + context"}; return rep


# ================================================================================================ spike or sustained trend
def spike_or_trend(req, D, focus=None):
    rep = Report("", "spike_or_trend"); c = next((x for x in req.concepts if x in ("battery_temperature", "motor_temperature", "coolant_flow", "aux_battery_voltage") and D.available(x)), "battery_temperature")
    evs = D.events(["WARN_BATT_OVERTEMP", "WARN_OVERTEMP"], D.t0, D.end + 1)
    if evs: ev = evs[-1]; ta = ev.ts; c = "battery_temperature" if ev.type == "WARN_BATT_OVERTEMP" else "motor_temperature"
    else:
        ent = (R.STORE.get(c) or {}).get("1m"); r = D.c.execute(f"SELECT ts FROM {ent[0]} ORDER BY {ent[1]} DESC LIMIT 1").fetchone(); ta = int(r[0]); ev = None
    s = D.series(c, ta - 1800, ta + 900, ev)
    if s is None or len(s.t) < 8: rep.title = "Not enough data around the abnormal reading."; rep.ok = False; return rep
    i = int(np.argmax(s.v)) if True else 0; peak, tpk = float(s.v[i]), float(s.t[i]); base = float(np.median(s.v[:max(4, len(s.v) // 4)])); rise = peak - base
    half = base + 0.5 * rise; above = s.v >= half; dur_half = float(above.sum() * s.res); pre = s.t < tpk; pre_slope = float(np.polyfit(s.t[pre][-min(len(s.t[pre]), int(600 / s.res) + 3):], s.v[pre][-min(len(s.v[pre]), int(600 / s.res) + 3):], 1)[0] * 60) if pre.sum() >= 6 else float("nan")
    sustained = dur_half >= 300 or (not math.isnan(pre_slope) and pre_slope > 0.05 and dur_half >= 120)
    rep.title = f"{D.label(c).capitalize()} peaking at {fmtv(peak, D.unit(c))} on {D.stamp(tpk)}: " + ("part of a sustained thermal trend" if sustained else "a brief spike")
    rep.add("Shape of the excursion", f"baseline {fmtv(base, D.unit(c))}, peak {fmtv(peak, D.unit(c))} (+{rise:.1f}); above half-height for {dur(dur_half)}", f"slope over the 10 minutes before the peak: {pre_slope:+.2f} per minute" if not math.isnan(pre_slope) else "")
    others = [x for x in ("coolant_flow", "battery_current", "motor_temperature", "speed", "ambient_temperature", "battery_temperature") if x != c and D.available(x)][:5]; lines = []
    for x in others:
        t = D.series(x, ta - 1800, ta + 900, ev)
        if t is not None and len(t.v) > 5: lines.append(f"{D.label(x)}: {fmtv(float(np.median(t.v[:max(3, len(t.v) // 4)])), D.unit(x), 0)} at the start of the window, {fmtv(float(t.v[np.argmin(np.abs(t.t - tpk))]), D.unit(x), 0)} at the peak, {fmtv(float(t.v[-1]), D.unit(x), 0)} at the end")
    rep.add("Related signals in the same window", *lines)
    s1 = D.series(c, ta - 600, ta + 300, ev) if ev else None
    rep.add("Resolution check", f"read at {s.label()}" + ("; a brief spike shorter than a minute could be hidden in one-minute averages" if s.res >= 30 else "; at this resolution a spike of a few seconds would show"))
    rep.add("Verdict rule", "a spike stays above half of its own height for under 5 minutes and had no rising slope before it; a trend stays above for 5 minutes or more, or rises steadily before the peak")
    rep.evidence = {"period": f"{D.stamp(ta - 1800)} to {D.stamp(ta + 900)}", "period_ts": [ta - 1800, ta + 900], "samples": len(s.t), "sources": [s.source], "tier": s.label()}; rep.focus_ts = ta; return rep


# ================================================================================================ a status that changed while its measurements did not
def status_vs_signal(req, D, focus=None):
    lo, hi, label = period(req, D, 105); rep = Report("", "status_vs_signal"); rows = []
    for e in D.events(["DTC_SET", "ADAS_STATE_CHANGE", "ECU_RESET"], lo, hi)[-60:]:
        sig = SIGNAL_OF_CODE.get(dtc_code(e)) if e.type == "DTC_SET" else ("lane_confidence" if e.type == "ADAS_STATE_CHANGE" else None)
        if not sig or not D.available(sig): continue
        s = D.series(sig, e.ts - 180, e.ts + 180, e)
        if s is None or len(s.v) < 6: continue
        lo_, hi_ = D.rng(sig); span = float(np.max(s.v) - np.min(s.v)) / max(hi_ - lo_, 1e-9); rows.append((span, e, sig, s))
    quiet = [r for r in rows if r[0] < 0.02]; rep.title = f"{len(quiet)} of {len(rows)} reported status changes {label} happened while the related measurement stayed within 2 % of its range"
    rep.add("Candidates for a reporting inconsistency", *[f"{D.stamp(e.ts)} {EVLABEL(e.type)}: {e.detail[:60]} - {D.label(sig)} moved only {fmtv(float(np.max(s.v) - np.min(s.v)), D.unit(sig))} within +-3 min" for span, e, sig, s in quiet[:6]] or ["none: every compared status change coincided with a visible change of its measurement"])
    rep.add("Do the logs support a real change?", *[f"{D.stamp(e.ts)}: " + (", ".join(f"{s_}: {m_[:50]}" for t_, s_, m_ in D.log_lines(e.ts - 120, e.ts + 120, limit=2)) or "no log line near it") for span, e, sig, s in quiet[:4]])
    rep.add("Limits", "only status changes with a related measured channel can be checked (diagnostic codes with a tied signal, ADAS state against lane confidence)")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(rows), "sources": ["events", "signals", "logs"], "tier": "event context / 1-minute"}; return rep


# ================================================================================================ ADAS variants
def adas_trip(req, D, rep, dis, lo, hi, need):
    trips = D.trips(lo, hi, 1.0); cnt = []
    for t in trips:
        k = [e for e in dis if t["start"] <= e.ts <= t["end"]]
        cnt.append((len(k), t, k))
    cnt.sort(key=lambda r: (r[0], r[1]["start"])); cand = [r for r in cnt if r[0] >= need]
    if not cand:
        best = cnt[-1] if cnt else None
        rep.title = f"No trip {'in the period ' if True else ''}contains {need} ADAS disengagements" + (f"; the closest is the trip of {D.stamp(best[1]['start'])} with {best[0]}" if best and best[0] else ""); rep.ok = False
        if best and best[0] >= 2: rep = None
        return rep
    k, t, es = cand[-1]; rep.title = f"The trip of {D.stamp(t['start'])} ({t['km']:.0f} km) had {k} ADAS disengagements: {', '.join(hms(D, e.ts) for e in es)}"
    sigs = ("speed", "lane_confidence", "lead_vehicle_distance", "camera_blockage", "steering_angle", "brake_pressure", "adas_state"); share = defaultdict(list); lines = []
    for e in es:
        parts = []
        for c in sigs:
            if not D.available(c): continue
            s = D.series(c, e.ts - 60, e.ts + 1, e)
            if s is None or len(s.v) < 5: continue
            d = D.departure(s, e.ts, k=2.5)
            if d: share[(c, d.direction)].append(e.ts)
            parts.append(f"{D.label(c)} {fmtv(float(s.v[0]), D.unit(c), 1)} -> {fmtv(float(s.v[-1]), D.unit(c), 1)}")
        lines.append(f"{hms(D, e.ts)} ({e.detail}): " + "; ".join(parts[:5]))
    rep.add("The 60 seconds before each disengagement", *lines)
    shared = [(c, d) for (c, d), v in share.items() if len(v) == len(es)]; most = [(c, d, len(v)) for (c, d), v in share.items() if len(v) >= max(2, len(es) - 1) and len(v) < len(es)]
    rep.add("Conditions shared by all of them", *[f"{D.label(c)} {'rose' if d == 'up' else 'fell'} before every one" for c, d in shared] or ["no signal changed in the same direction before all of them"], *[f"{D.label(c)} {'rose' if d == 'up' else 'fell'} before {n} of {len(es)}" for c, d, n in most[:3]])
    rep.add("Reasons reported", ", ".join(f"{r} x{n}" for r, n in Counter(e.detail for e in es).most_common()))
    rep.evidence = {"period": f"{D.stamp(t['start'])} to {D.stamp(t['end'])}", "period_ts": [t["start"], t["end"]], "samples": len(es), "sources": ["events", "signals"], "tier": "event context / 1-minute"}; return rep


def adas_rank(req, D, rep, dis, lo, hi, n):
    scored = []
    for e in dis:
        near = [o for o in D.all_events(e.ts - 60, e.ts + 61) if o.type not in ("ADAS_DISENGAGE", "ADAS_STATE_CHANGE", "HARSH_BRAKE", "HARSH_ACCEL") and o.ts != e.ts]
        sp = D.series("speed", e.ts - 5, e.ts + 1, e); v = float(sp.v[-1]) if sp is not None and len(sp.v) else 0.0
        score = 3.0 * len(near) + (2.0 if e.detail in ("system limit", "sensor conflict", "radar fault") else 0.0) + (1.0 if v < 60 else 0.0) + (1.0 if len([x for x in dis if abs(x.ts - e.ts) < 300]) > 1 else 0.0)
        scored.append((score, e, near, v))
    scored.sort(key=lambda r: -r[0]); top = scored[:n]; rep.title = f"The {len(top)} most unusual of {len(dis)} ADAS disengagements {('in ' + str(hi - lo) + ' s') if False else ''}ranked by severity evidence"
    rep.add("Ranking rule", "3 points per other recorded event within 60 s (ECU reset, bus error, diagnostic code, sensor conflict), +2 for a system-side reason (system limit, sensor conflict, radar fault), +1 below 60 km/h, +1 when it was part of a cluster within 5 minutes")
    for rank, (score, e, near, v) in enumerate(top, 1):
        lane = D.series("lane_confidence", e.ts - 120, e.ts + 1, e); rep.add(f"{rank}. {D.stamp(e.ts)} - {e.detail} (score {score:.0f})", f"speed {fmtv(v, 'km/h', 0)}" + (f", lane confidence {float(lane.v[0]):.2f} -> {float(lane.v[-1]):.2f}" if lane is not None and len(lane.v) > 3 else ""), *[f"{EVLABEL(o.type)} {dur(o.ts - e.ts)} {'after' if o.ts >= e.ts else 'before'}: {o.detail}" for o in near[:3]] or ["no other event within 60 s"])
    rec = Counter(e.detail for _, e, _, _ in top); rep.add("Recurring pattern", ", ".join(f"{r} x{k}" for r, k in rec.most_common(3)))
    rep.add("Data needed to validate each suspected cause", NEXT.get("ADAS_DISENGAGE", "camera frames and radar object lists at the time (not stored: only camera metadata is kept)"))
    rep.evidence = {"period": f"{len(dis)} disengagements", "period_ts": [lo, hi], "samples": len(dis), "sources": ["events", "signals"], "tier": "events + context"}; return rep
