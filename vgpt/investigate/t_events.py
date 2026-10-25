"""Event-centred analyses: co-occurrence, grouping, lead times, ADAS, source discrepancies, fault history, health ranking, frequency trends, similar incidents."""
import math
from collections import Counter, defaultdict

import numpy as np

from .. import signal_registry as R
from . import parse as P
from .primitives import effect
from .report import Report
from .templates import CONTEXT_ONLY, EVLABEL, LEX, dur, fmtv, hms, incident, R_label, signals_for

SUBSYSTEM = {"WARN_BATT_OVERTEMP": "battery thermal", "WARN_OVERTEMP": "powertrain", "POWER_DERATE": "powertrain", "VOLTAGE_DIP": "12 V power", "LOW_12V": "12 V power", "UNEXPECTED_SHUTDOWN": "12 V power",
             "CAN_ERROR": "network and ECUs", "SOURCE_SILENT": "network and ECUs", "ECU_RESET": "network and ECUs", "SENSOR_CONFLICT": "sensing", "ADAS_DISENGAGE": "ADAS and camera", "ADAS_STATE_CHANGE": "ADAS and camera", "AEB_ACTIVATION": "ADAS and camera",
             "CAMERA_BLOCKED": "ADAS and camera", "CAMERA_DEGRADED": "ADAS and camera", "TPMS_LOW": "tyres", "SENSOR_INVALID": "sensing", "CAN_ANOMALY": "network and ECUs"}
CODE_SUBSYSTEM = {"P1A": "battery thermal", "P0A": "powertrain", "P05": "12 V power", "U0": "network and ECUs", "B1": "network and ECUs", "C07": "tyres", "C1A": "ADAS and camera", "C00": "brakes", "P02": "powertrain", "P00": "powertrain"}
SEV_W = {"critical": 3.0, "warn": 1.5, "info": 0.5}
BORING = ("HARSH_BRAKE", "HARSH_ACCEL", "HARSH_CORNER", "CHARGE_START", "SW_UPDATE", "SERVICE_DUE", "CAMERA_RESTORED", "TYRE_INFLATED", "DTC_CLEAR", "ADAS_STATE_CHANGE")
SIGNAL_OF_CODE = {"P1A00": "coolant_flow", "P0562": "aux_battery_voltage", "C0750": "tyre_pressure_fl", "C0751": "tyre_pressure_fr", "C0752": "tyre_pressure_rl", "C0753": "tyre_pressure_rr", "P0217": "motor_temperature"}    # only codes whose meaning IS a measured channel


def period(req, D, default_days):
    from ..query_ir import TimeSpec
    from ..query_planner import resolve_time
    t = req.times[0] if req.times else TimeSpec("relative", default_days, "day"); lo, hi, label = resolve_time(t, D.qc.ctx)
    return int(max(lo, D.t0)), int(min(hi, D.end + 1)), label


def dtc_code(ev): return ev.detail.split(" ")[0] if ev.type in ("DTC_SET", "DTC_CLEAR") else None


def subsystem_of(D, ev):
    code = dtc_code(ev)
    if code:
        info = D.dtc().get(code)
        if info and info.get("subsystem"): return {"thermal": "battery thermal", "power": "12 V power", "network": "network and ECUs", "adas": "ADAS and camera", "electronics": "network and ECUs"}.get(info["subsystem"], info["subsystem"])
        for k, v in CODE_SUBSYSTEM.items():
            if code.startswith(k): return v
    return SUBSYSTEM.get(ev.type, "other")


def cluster(events, gap):
    out = []; cur = []
    for e in sorted(events, key=lambda e: e.ts):
        if cur and e.ts - cur[-1].ts > gap: out.append(cur); cur = []
        cur.append(e)
    if cur: out.append(cur)
    return out


def specific(req, skip=()): return [e for e in req.events if e != "ANY_WARNING" and e not in skip]


def poisson_p(k1, e1, k2, e2):
    """One-sided p-value that count k2 (exposure e2) exceeds the rate of k1 (exposure e1): conditional binomial test."""
    from scipy.stats import binomtest
    n = k1 + k2
    return 1.0 if n == 0 else float(binomtest(k2, n, e2 / (e1 + e2), alternative="greater").pvalue)


# ================================================================================================ co-occurrence, grouping, lead times
def cooccur(req, D, focus=None):
    lo, hi, label = period(req, D, 30); within = req.windows.get("within", 120.0); rep = Report("", "cooccur"); spec = specific(req)
    A = spec[0] if spec else "ANY_WARNING"; Btypes = spec[1:] or [t for t in sorted({e.type for e in D.all_events(lo, hi)}) if t not in D.expand([A]) and t not in BORING]
    evA = D.events([A], lo, hi); rep.step(f"events of type {A} in {label}: {len(evA)}; partners {', '.join(Btypes)} within +-{within:g} s")
    if (len(spec) >= 3 and "within" in req.windows) or (not evA and len(spec) >= 2):                                               # several kinds of event within a window of one another: clusters
        allev = D.events(spec, lo, hi); need = min(3, len(spec)); cl = [c for c in cluster(allev, within) if len({e.type for e in c}) >= need]
        rep.title = f"{len(cl)} period{'s' if len(cl) != 1 else ''} {label} in which {', '.join(EVLABEL(t) for t in spec[:4])} occurred within {dur(within)} of one another"
        for c in cl[:6]: rep.add(f"{D.stamp(c[0].ts)} to {D.stamp(c[-1].ts)}", *[f"{hms(D, e.ts)} {EVLABEL(e.type)}: {e.detail}" for e in c[:8]])
        if not cl: rep.add("Result", f"{len(allev)} such events were recorded, but never {need} different kinds within {dur(within)} of each other")
        rep.add("What this shows", "co-occurrence in time is an association; it does not by itself show that one event caused another")
        rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(allev), "sources": ["events"], "tier": "events"}; return rep
    if not evA:
        rep.title = f"No {EVLABEL(A)} was recorded {label}, so there is nothing to correlate."; rep.ok = False; return rep
    pairs = []; with_any = 0
    for a in evA:
        near = [b for b in D.events(Btypes, a.ts - within, a.ts + within + 1) if not (b.ts == a.ts and b.type == a.type)]
        if near: with_any += 1
        for b in near: pairs.append((a, b))
    span = max(1.0, hi - lo); rows = []
    for t in Btypes:
        nb = len(D.events([t], lo, hi))
        if not nb: continue
        p1 = 1 - math.exp(-nb / span * 2 * within); exp_ = len(evA) * p1; obs = len({id(a) for a, b in pairs if b.type == t}); rows.append((t, nb, obs, exp_))
    rep.title = f"{with_any} of {len(evA)} {EVLABEL(A)}{'s' if len(evA) != 1 else ''} {label} had another recorded event within {dur(within)}"
    rep.add(f"Co-occurrences (within +-{dur(within)})", *[f"{EVLABEL(t)}: {obs} of {len(evA)} had one nearby; about {exp_:.1f} would be expected by chance from its overall rate ({nb} such events {label}) -> " + ("clearly more than chance" if obs >= 3 and obs > 2.5 * max(exp_, 0.3) else "not clearly above chance") for t, nb, obs, exp_ in sorted(rows, key=lambda r: -(r[2] - r[3]))])
    rep.add("Examples (by time)", *[f"{D.stamp(a.ts)} {EVLABEL(a.type)} ({a.detail}) and {EVLABEL(b.type)} {dur(b.ts - a.ts)} {'later' if b.ts >= a.ts else 'earlier'} ({b.detail})" for a, b in pairs[:8]])
    first = Counter()
    for a, b in pairs: first[(a.type if a.ts <= b.ts else b.type)] += 1
    if first: rep.add("Which reported first", *[f"{EVLABEL(t)} came first in {n} of the {len(pairs)} pairs" for t, n in first.most_common(3)])
    rep.add("What this shows", "an association in time; it does not prove that one event caused the other", "the ranking compares observed pairs with the number expected by chance from each event's overall rate")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evA), "sources": ["events"], "tier": "events"}; rep.facts.update(pairs=len(pairs), a_events=len(evA), with_partner=with_any)
    return rep


def group_warnings(req, D, focus=None):
    lo, hi, label = period(req, D, 1); within = req.windows.get("within", 120.0); rep = Report("", "group_warnings"); evs = D.events(["ANY_WARNING"], lo, hi)
    if not evs:
        last = D.events(["ANY_WARNING"], D.t0, D.end + 1)
        rep.title = f"No warning was recorded {label}."; rep.add("For orientation", f"the last warning on record is {EVLABEL(last[-1].type)} at {D.stamp(last[-1].ts)}: {last[-1].detail}" if last else "no warning exists anywhere in the history"); rep.ok = False
        rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": 0, "sources": ["events"], "tier": "events"}; return rep
    groups = cluster(evs, within); rep.title = f"{len(evs)} warning{'s' if len(evs) != 1 else ''} {label}, in {len(groups)} group{'s' if len(groups) != 1 else ''} (events within {dur(within)} of each other share a group)"
    for g in groups[:8]:
        ta = g[0].ts; best = None
        for c in ("coolant_flow", "battery_temperature", "motor_temperature", "aux_battery_voltage", "battery_current", "lane_confidence"):
            if D.available(c):
                s = D.series(c, ta - 600, ta, g[0]); d = D.departure(s, ta) if s is not None else None
                if d and (best is None or d.ts < best.ts): best = d
        rep.add(f"Group at {D.stamp(ta)}", *[f"{hms(D, e.ts)} {EVLABEL(e.type)}: {e.detail}" for e in g[:6]], (f"first abnormal measurement: {D.label(best.concept)} {'rose' if best.direction == 'up' else 'fell'} at {hms(D, best.ts)}, {dur(ta - best.ts)} before the first warning of the group" if best else "no measurement left its baseline before the group's first warning"))
    rep.add("Order", "groups are in time order; inside each group the first abnormal measurement is given separately from the warning times, as asked")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evs), "sources": ["events", "signals"], "tier": "events + context"}; return rep


def leadtime(req, D, focus=None):
    lo, hi, label = period(req, D, 14); look = 600.0; rep = Report("", "leadtime"); types = specific(req) or ["ANY_WARNING"]
    evs = D.events(types, lo, hi)[-40:]
    if not evs: rep.title = f"No matching warning was recorded {label}."; rep.ok = False; return rep
    sigs, asked = signals_for(req, D, evs[0].type); rows = []; first_sig = Counter(); leads = []
    for e in evs:
        best = None
        for c in sigs:
            if c in CONTEXT_ONLY and c not in asked: continue
            s = D.series(c, e.ts - look, e.ts, e); d = D.departure(s, e.ts) if s is not None else None
            if d and (best is None or d.ts < best.ts): best = d
        rows.append((e, best))
        if best: first_sig[best.concept] += 1; leads.append(e.ts - best.ts)
    n = len(rows); with_pre = len(leads); rep.step(f"{n} events of {', '.join(types)}; looked back {dur(look)} on {len(sigs)} signals")
    rep.title = f"{with_pre} of {n} warnings {label} were preceded by a signal leaving its baseline within {dur(look)}" + (f"; the median lead time was {dur(float(np.median(leads)))}" if leads else "")
    if first_sig:
        top, k = first_sig.most_common(1)[0]
        rep.add("Which signal moved first", *[f"{D.label(c)}: first in {k2} of {with_pre} cases" for c, k2 in first_sig.most_common(4)], "the pattern " + ("repeats" if k / with_pre >= 0.6 and with_pre >= 3 else "does not clearly repeat") + f" ({D.label(top)} first in {k / with_pre:.0%} of the cases that had a precursor)")
    rep.add("Cases (most recent last)", *[f"{D.stamp(e.ts)} {EVLABEL(e.type)}: " + (f"{D.label(d.concept)} {'rose' if d.direction == 'up' else 'fell'} {dur(e.ts - d.ts)} before ({R_label(d.res)})" if d else "no signal left its baseline before it") for e, d in rows[-10:]])
    rep.add("How far to trust this", "a lead time is an association in time; a signal that moves before a warning may be a symptom, not a cause", "the baseline is each signal's own first 40 % of the window; resolution is 1 Hz inside frozen event context, 1-minute averages otherwise")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": n, "sources": ["events", "signals"], "tier": "event context / 1-minute"}; return rep


# ================================================================================================ ADAS
def adas(req, D, focus=None):
    lo, hi, label = period(req, D, 30); rep = Report("", "adas"); n_ask = next((n for n, noun in req.counts if noun in ("incident", "event", "disengagement", "instance")), None); text = req.text.lower()
    dis = D.events(["ADAS_DISENGAGE"], lo, hi); ch = D.events(["ADAS_STATE_CHANGE"], lo, hi); aeb = D.events(["AEB_ACTIVATION"], lo, hi)
    from . import t_extra
    if "unusual" in text and n_ask and dis: return t_extra.adas_rank(req, D, rep, dis, lo, hi, n_ask)
    if ("during one" in text or "one highway trip" in text) and n_ask:
        r = t_extra.adas_trip(req, D, rep, dis, lo, hi, n_ask)
        if r is not None: return r; rep = Report("", "adas")
    if not dis and (n_ask or "every" in text):
        last = D.events(["ADAS_DISENGAGE"], D.t0, D.end + 1); rep.title = f"No ADAS disengagement was recorded {label}."
        rep.add("For orientation", f"the latest on record: {D.stamp(last[-1].ts)} ({last[-1].detail})" if last else "no disengagement exists anywhere in the history"); rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": 0, "sources": ["events"], "tier": "events"}; return rep
    if not D.available("adas_state") and not dis:
        rep.title = "This vehicle records no ADAS data, so there is nothing to analyse."; rep.ok = False; return rep
    expo = D.c.execute("SELECT COALESCE(SUM(e.adas),0)/100.0, COALESCE(SUM(s.speed/10.0/60.0*e.adas/100.0),0) FROM ext_1m e JOIN sig_1m s ON s.ts=e.ts WHERE e.ts>=? AND e.ts<?", (lo, hi)).fetchone() if "ext_1m" in D.have else (0, 0)
    mins_on, km_on = float(expo[0] or 0), float(expo[1] or 0); hours = mins_on / 60.0
    rep.step(f"ADAS events {label}: {len(dis)} disengagements, {len(ch)} state changes, {len(aeb)} AEB; exposure {hours:.1f} h / {km_on:.0f} km with lane keeping on")
    rep.title = f"{len(dis)} ADAS disengagement{'s' if len(dis) != 1 else ''} {label}" + (f" over {hours:.1f} h ({km_on:.0f} km) with lane keeping on: {len(dis) / max(hours, 0.01):.2f} per hour, {100 * len(dis) / max(km_on, 1):.1f} per 100 km" if hours >= 0.5 else "; too little ADAS-on time is recorded to give a rate")
    if dis:
        reasons = Counter(e.detail for e in dis); rep.add("Why it disengaged (as the system reported it)", *[f"{r}: {n} ({n / len(dis):.0%})" for r, n in reasons.most_common(5)])
        by_mode = {1: [0, 0.0], 2: [0, 0.0]}
        for e in dis:
            r = D.c.execute("SELECT mode FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (e.ts,)).fetchone()
            if r and r[0] in by_mode: by_mode[r[0]][0] += 1
        for m in (1, 2):
            r = D.c.execute("SELECT COALESCE(SUM(e.adas),0)/100.0/60.0 FROM ext_1m e JOIN sig_1m s ON s.ts=e.ts WHERE e.ts>=? AND e.ts<? AND s.mode=?", (lo, hi, m)).fetchone(); by_mode[m][1] = float(r[0] or 0)
        rep.add("City versus highway", *[f"{name}: {by_mode[m][0]} disengagements over {by_mode[m][1]:.1f} h with lane keeping on" + (f" = {by_mode[m][0] / by_mode[m][1]:.2f} per hour" if by_mode[m][1] >= 0.3 else " (too little time to give a rate)") for m, name in ((1, "city"), (2, "highway"))])
        days = Counter(D.day(e.ts) for e in dis); rep.add("When", *[f"{d}: {n} disengagements" for d, n in days.most_common(3)])
        co = Counter(); ex = []
        for e in dis:
            for o in D.all_events(e.ts - 60, e.ts + 61):
                if o.type not in ("ADAS_DISENGAGE", "ADAS_STATE_CHANGE", "HARSH_BRAKE", "HARSH_ACCEL") and not (o.ts == e.ts and o.type == e.type): co[o.type] += 1; ex.append((e, o))
        if co:
            rep.add("Recorded events within 60 s of a disengagement", *[f"{EVLABEL(t)}: {n} time{'s' if n != 1 else ''}" for t, n in co.most_common(5)], *[f"{D.stamp(e.ts)} disengagement ({e.detail}) with {EVLABEL(o.type)} {dur(o.ts - e.ts)} {'after' if o.ts >= e.ts else 'before'}: {o.detail}" for e, o in ex[:4]])
        else: rep.add("Recorded events within 60 s of a disengagement", "none - no ECU reset, communication error, sensor conflict or diagnostic code was recorded next to any of them")
        show = dis[-(n_ask or 5):]
        lines = []
        for e in show:
            s = D.series("speed", e.ts - 5, e.ts + 1, e); lane = D.series("lane_confidence", e.ts - 120, e.ts + 1, e)
            lines.append(f"{D.stamp(e.ts)}: {e.detail}" + (f", speed {fmtv(float(s.v[-1]), 'km/h', 0)}" if s is not None and len(s.v) else "") + (f", lane confidence {float(lane.v[0]):.2f} -> {float(lane.v[-1]):.2f}" if lane is not None and len(lane.v) > 3 else ""))
        rep.add(f"The {len(show)} most recent", *lines)
    flap = [c for c in cluster(ch, 300) if len(c) >= 4]
    if flap:
        block = ("Repeated switching (four or more state changes within 5 minutes)", *[f"{D.stamp(c[0].ts)}: {len(c)} changes in {dur(c[-1].ts - c[0].ts)}; time in each state: " + ", ".join(f"{('on' if x.detail.split()[-1] == 'on' else 'off')} {dur(b.ts - x.ts)}" for x, b in list(zip(c, c[1:]))[:5]) + "; disengagements and warnings around it: " + (", ".join(f"{hms(D, o.ts)} {EVLABEL(o.type)}" for o in D.all_events(c[0].ts - 60, c[-1].ts + 60) if o.type not in ("ADAS_STATE_CHANGE",))[:160] or "none") for c in flap[:3]])
        if "repeatedly" in req.text.lower() or "status changed" in req.text.lower():
            rep.lines.insert(0, (block[0], list(block[1:]))); rep.title = f"{len(flap)} period{'s' if len(flap) != 1 else ''} {label} in which ADAS switched on and off repeatedly; " + rep.title
        else: rep.add(block[0], *block[1:])
    elif "repeatedly" in req.text.lower(): rep.add("Repeated switching", "no period with four or more ADAS state changes within 5 minutes was recorded")
    if aeb: rep.add("Automatic emergency braking", *[f"{D.stamp(e.ts)}: {e.detail}" for e in aeb[-3:]])
    rep.add("Limits", "rates divide by the time lane keeping was on (from the per-minute ADAS share), so they are approximate", "a disengagement 'reason' is what the system reported, not an independently verified cause")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(dis), "sources": ["events", "ext_1m", "sig_1m"], "tier": "events + 1-minute"}; return rep


# ================================================================================================ source discrepancies
def discrepancy(req, D, focus=None):
    lo, hi, label = period(req, D, 105); rep = Report("", "discrepancy"); cf = D.events(["SENSOR_CONFLICT"], lo, hi); inv = D.events(["SENSOR_INVALID"], lo, hi); ce = D.events(["CAN_ERROR"], lo, hi)
    rep.step(f"sensor conflicts {label}: {len(cf)}; invalid readings {len(inv)}; bus errors {len(ce)}")
    if not cf and not inv:
        rep.title = f"No two sources disagreed {label}: no sensor conflict or invalid reading is recorded."; rep.ok = True
        rep.add("What was checked", "sources that report the same quantity (for example CAN and the Android vehicle layer) are compared continuously and a difference above tolerance for several seconds is recorded as a conflict", f"{len(ce)} bus-error bursts were recorded {label}")
        gaps = data_gaps(D, max(lo, D.end - 7 * 86400), hi); rep.add("Missing intervals in the last 7 days (awake, no samples)", *gaps[:5]) if gaps else None
        rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": 0, "sources": ["events"], "tier": "events"}; return rep
    by = defaultdict(list)
    for e in cf: by[e.detail.split(":")[0]].append(e)
    rep.title = f"{len(cf)} sensor conflict{'s' if len(cf) != 1 else ''} {label} between {len(by)} quantit{'ies' if len(by) != 1 else 'y'}"
    for q, es in sorted(by.items(), key=lambda kv: -len(kv[1])):
        dtcs = sum(1 for e in es if D.events(["DTC_SET"], e.ts - 600, e.ts + 600)); rep.add(f"{q}: {len(es)} conflicts", *[f"{D.stamp(e.ts)}: {e.detail.split(': ', 1)[1]}" for e in es[:3]], f"a diagnostic code was recorded within 10 minutes of {dtcs} of the {len(es)}" + (" - the vehicle's own fault memory did not flag them" if dtcs == 0 else ""))
    if inv: rep.add("Readings rejected as physically impossible", *[f"{D.stamp(e.ts)}: {e.detail}" for e in inv[:4]])
    near = [e for e in cf if D.events(["CAN_ERROR", "ECU_RESET"], e.ts - 120, e.ts + 120)]
    rep.add("Is it a data-path problem?", f"{len(near)} of the {len(cf)} conflicts coincide (within 2 minutes) with a bus error or ECU reset" if near else "no conflict coincides with a bus error or ECU reset, so the disagreement is between the sources themselves")
    gaps = data_gaps(D, max(lo, D.end - 7 * 86400), hi)
    if gaps: rep.add("Missing intervals in the last 7 days (awake, no samples)", *gaps[:4])
    rep.add("How to read this", "a conflict is a genuine discrepancy between two recorded sources; which one is right needs a third reference (for example a GPS speed or the physical sensor)")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(cf), "sources": ["events", "logs"], "tier": "events"}; return rep


def data_gaps(D, lo, hi, minimum=8):
    rows = D.c.execute("SELECT ts FROM sig_1s WHERE ts>=? AND ts<? AND mode IN (1,2,3) ORDER BY ts", (int(lo), int(hi))).fetchall(); out = []
    for (a,), (b,) in zip(rows, rows[1:]):
        if minimum <= b - a <= 600: out.append((b - a, a))
    return [f"{D.stamp(a)}: no samples for {dur(g)} while the vehicle was awake" for g, a in sorted(out, reverse=True)[:6]]


# ================================================================================================ fault history
def fault_history(req, D, focus=None):
    lo, hi, label = period(req, D, 120); rep = Report("", "fault_history"); evs = [e for e in D.events(["ANY_WARNING", "DTC_CLEAR", "ECU_RESET"], lo, hi) if e.type not in BORING or e.type == "DTC_CLEAR"]
    if not evs: rep.title = f"No fault or warning is recorded {label}."; rep.ok = False; return rep
    key = lambda e: (dtc_code(e) or e.type)
    groups = defaultdict(list)
    for e in evs: groups[key(e)].append(e)
    rep.step(f"{len(evs)} warning / fault events {label} in {len(groups)} fault kinds")
    sets = {k: [e for e in v if e.type != "DTC_CLEAR"] for k, v in groups.items()}; clears = {k: [e for e in v if e.type == "DTC_CLEAR"] for k, v in groups.items()}
    rec = {k: v for k, v in sets.items() if len(v) >= 2}
    rep.title = f"{len(groups)} kinds of fault or warning {label}; {len(rec)} recurred ({', '.join(sorted(rec)[:4])})" if rec else f"{len(groups)} kinds of fault or warning {label}; none recurred"
    by_sub = defaultdict(list)
    for k, v in sets.items():
        if v: by_sub[subsystem_of(D, v[0])].append(k)
    for sub, ks in sorted(by_sub.items(), key=lambda kv: -sum(len(sets[k]) for k in kv[1])):
        lines = []
        for k in sorted(ks, key=lambda k: sets[k][0].ts):
            v = sets[k]; first = v[0]; ts = [e.ts for e in v]; cl = clears.get(k, [])
            seg = f"{k}" + (f" ({D.dtc().get(k, {}).get('text')})" if D.dtc().get(k) else f" ({EVLABEL(first.type)})") + f": first {D.stamp(first.ts)}"
            if len(v) > 1:
                gaps = np.diff(ts); seg += f"; {len(v)} occurrences, median interval {dur(float(np.median(gaps)))}, last {D.stamp(ts[-1])}"
                longest = int(np.argmax(gaps)); seg += f"; longest quiet spell {dur(float(gaps[longest]))} after {D.stamp(ts[longest])}"
            else: seg += "; once"
            if cl:
                last_clear = cl[-1].ts; back = [e for e in v if e.ts > last_clear]
                seg += f"; cleared {len(cl)} time{'s' if len(cl) != 1 else ''} (last {D.stamp(last_clear)})" + (f" and returned {dur(back[0].ts - last_clear)} later" if back else " and has not returned")
            elif first.type == "DTC_SET": seg += "; still stored (never cleared)"
            lines.append(seg)
        rep.add(f"{sub}", *lines)
    rep.add("Reading this", "'recurred' means two or more records of the same code or event type; 'persisted' means stored and never cleared", "inactive spells are the longest gaps between occurrences")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evs), "sources": ["events"], "tier": "events"}; return rep


def dtc_cycle(req, D, focus=None):
    lo, hi, label = period(req, D, 120); rep = Report("", "dtc_cycle"); clears = D.events(["DTC_CLEAR"], lo, hi)
    if not clears:
        rep.title = f"No diagnostic code was cleared {label}: every stored code is still stored."; rep.ok = False
        sets = D.events(["DTC_SET"], lo, hi); rep.add("Latest codes still stored", *[f"{D.stamp(e.ts)} {e.detail}" for e in sets[-3:]]); return rep
    rep.title = f"{len(clears)} diagnostic code{'s' if len(clears) != 1 else ''} cleared {label}; for each, did the signal return to normal, did the code become inactive, or did the reporting just stop?"
    for c in clears[-4:]:
        code = dtc_code(c); sets = [e for e in D.events(["DTC_SET"], D.t0, c.ts) if dtc_code(e) == code]; s0 = sets[-1] if sets else None
        sig = SIGNAL_OF_CODE.get(code); lines = [f"set {D.stamp(s0.ts)}, cleared {D.stamp(c.ts)} ({dur(c.ts - s0.ts)} later)" if s0 else f"cleared {D.stamp(c.ts)}"]
        if sig and D.available(sig) and s0:
            pre = D.series(sig, s0.ts - 3600, s0.ts - 60); mid = D.series(sig, s0.ts, c.ts, None); post = D.series(sig, c.ts, c.ts + 1800)
            if pre is not None and len(pre.v) >= 3 and post is not None and len(post.v) >= 3:
                base = float(np.median(pre.v)); during = float(np.median(mid.v)) if mid is not None and len(mid.v) else float("nan"); after = float(np.median(post.v)); lo_, hi_ = D.rng(sig); tol = 0.03 * (hi_ - lo_)
                lines.append(f"{D.label(sig)}: {fmtv(base, D.unit(sig))} before the fault, {fmtv(during, D.unit(sig))} while it was set, {fmtv(after, D.unit(sig))} in the 30 minutes after the clear -> " + ("back to normal" if abs(after - base) <= tol else "still different from before the fault"))
            else: lines.append(f"{D.label(sig)}: not enough samples around the clear to compare")
        else: lines.append("no recorded signal is tied to this code, so whether the underlying condition went away cannot be checked")
        logs = D.log_lines(c.ts - 600, c.ts + 600, limit=3); lines.append("log lines around the clear: " + ("; ".join(f"{hms(D, t)} {s}: {m}" for t, s, m in logs) if logs else "none (the logs did not mention it)"))
        rep.add(f"{code}", *lines)
    rep.add("How to decide", "'signal returned to normal' needs a measured channel; 'fault became inactive' is what a clear after an ignition cycle records; 'logs stopped reporting' shows as no log lines around the clear")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(clears), "sources": ["events", "signals", "logs"], "tier": "events + 1-minute"}; return rep


# ================================================================================================ health report and ranking
def health(req, D, focus=None):
    lo, hi, label = period(req, D, 28); rep = Report("", "health"); mid = (lo + hi) // 2; evs = [e for e in D.events(["ANY_WARNING", "AEB_ACTIVATION", "POWER_DERATE", "CAMERA_DEGRADED", "ADAS_DISENGAGE"], lo, hi)]
    subs = defaultdict(list)
    for e in evs: subs[subsystem_of(D, e)].append(e)
    clears = {dtc_code(e): e.ts for e in D.events(["DTC_CLEAR"], lo, hi)}
    rows = []
    for sub, es in subs.items():
        n_old = sum(1 for e in es if e.ts < mid); n_new = len(es) - n_old; sev = max(es, key=lambda e: SEV_W.get(e.sev, 1)).sev; rec14 = sum(1 for e in es if e.ts >= hi - 14 * 86400)
        active = [e for e in es if e.type == "DTC_SET" and clears.get(dtc_code(e), 0) < e.ts]; recency = 1.0 + (2.0 if es[-1].ts >= hi - 7 * 86400 else 0.0)
        score = SEV_W.get(sev, 1) * (n_new * 2 + n_old) * recency + (4.0 if active else 0.0)
        esc = sev == "critical" or rec14 >= 3 or bool(active)
        rows.append((score, sub, es, n_old, n_new, sev, active, esc))
    rows.sort(key=lambda r: -r[0]); rep.step(f"{len(evs)} events {label} in {len(rows)} subsystems; ranked by severity x recency-weighted count + active codes")
    sig = []
    for c, label_s, fn in (("coolant_flow", "coolant flow", "drop"), ("battery_temperature", "battery temperature", "max"), ("motor_temperature", "motor temperature", "max"), ("aux_battery_voltage", "12 V battery voltage", "min")):
        ent = (R.STORE.get(c) or {}).get("1m")
        if not ent or not D.available(c): continue
        t, col, scale = ent
        r = D.c.execute(f"SELECT AVG(CASE WHEN ts<{mid} THEN {col} END)/{scale}, AVG(CASE WHEN ts>={mid} THEN {col} END)/{scale}, MAX({col})/{scale}, MIN(CASE WHEN {col}>0 THEN {col} END)/{scale} FROM {t} WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
        if r[0] is None or r[1] is None: continue
        if fn == "drop": sig.append(f"coolant flow: first half mean {r[0]:.1f}, second half {r[1]:.1f} L/min ({100 * (r[1] - r[0]) / max(r[0], 0.1):+.0f} %)")
        elif fn == "max": sig.append(f"{label_s}: peak {r[2]:.1f} {D.unit(c)} (first half mean {r[0]:.1f}, second half {r[1]:.1f})")
        else: sig.append(f"{label_s}: lowest {r[3]:.2f} V (first half mean {r[0]:.2f}, second half {r[1]:.2f})")
    rep.title = (f"Health summary {label}: {len(rows)} subsystem{'s' if len(rows) != 1 else ''} with recorded findings; highest priority: {rows[0][1]}" if rows else f"Health summary {label}: no warnings, faults or ADAS events were recorded")
    for rank, (score, sub, es, n_old, n_new, sev, active, esc) in enumerate(rows, 1):
        codes = Counter(dtc_code(e) or e.type for e in es); worst = f"{', '.join(f'{k} x{n}' for k, n in codes.most_common(3))}"
        rep.add(f"{rank}. {sub}", f"{len(es)} events ({n_old} in the first half of the period, {n_new} in the second; worst severity {sev}); latest {D.stamp(es[-1].ts)}", worst,
                ("ACTIVE: " + ", ".join(sorted({dtc_code(e) for e in active})) + " stored and not cleared") if active else "no stored code outstanding", "enough evidence to escalate: " + ("yes - " + ("a critical event" if sev == "critical" else "three or more in 14 days" if sum(1 for e in es if e.ts >= hi - 14 * 86400) >= 3 else "an active code") if esc else "not yet - fewer than three recent events and nothing critical or active"))
    if sig: rep.add("Measured trends behind them", *sig)
    rep.add("Active versus historical", "ACTIVE means a diagnostic code was set and never cleared; everything else is history", "ranking = worst severity x events weighted towards the recent half and the last week; frequency alone is not read as a safety risk")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": len(evs), "sources": ["events", "sig_1m"], "tier": "events + 1-minute"}; return rep


# ================================================================================================ frequency trend
def frequency_trend(req, D, focus=None):
    lo, hi, label = period(req, D, 30); mid = (lo + hi) // 2; rep = Report("", "frequency_trend"); k = next((n for n, noun in req.counts if noun in ("anomalie", "fault", "warning", "event")), 3)
    km = lambda a, b: float(D.c.execute("SELECT COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<?", (a, b)).fetchone()[0])
    k1, k2 = max(km(lo, mid), 1.0), max(km(mid, hi), 1.0); h1, h2 = (mid - lo) / 86400.0, (hi - mid) / 86400.0; rows = []
    for t in sorted({e.type for e in D.all_events(lo, hi)} - set(BORING)):
        a = len(D.events([t], lo, mid)); b = len(D.events([t], mid, hi))
        if a + b < 2: continue
        r1, r2 = 100 * a / k1, 100 * b / k2; rows.append((b - a, t, a, b, r1, r2, poisson_p(a, h1, b, h2)))
    rows.sort(reverse=True); up = [r for r in rows if r[0] > 0][:k]
    rep.step(f"event counts per half of {label}: first {dur(mid - lo)} vs last {dur(hi - mid)}; distance {k1:.0f} km vs {k2:.0f} km")
    rep.title = f"{len(up)} event type{'s' if len(up) != 1 else ''} became more frequent {label}" if up else f"No event type became more frequent {label}"
    for d, t, a, b, r1, r2, p in up: rep.add(EVLABEL(t), f"count: {a} in the first half, {b} in the second ({b - a:+d})", f"per 100 km driven: {r1:.2f} -> {r2:.2f} (the second half had {k2:.0f} km, the first {k1:.0f} km)", f"chance that the rise is just random variation: p = {p:.3f}" + (" (unlikely to be chance)" if p < 0.05 else " (could be chance)"))
    flat = [r for r in rows if r not in up][:4]
    if flat: rep.add("Not more frequent", *[f"{EVLABEL(t)}: {a} -> {b}" for d, t, a, b, r1, r2, p in flat])
    rep.add("Method", "equal halves of the period compared; rates are per 100 km so that a busier half does not look worse; the test asks whether the second half's share of events exceeds its share of time")
    rep.evidence = {"period": label, "period_ts": [lo, hi], "samples": sum(r[2] + r[3] for r in rows), "sources": ["events", "trips"], "tier": "events"}; return rep


# ================================================================================================ similar incidents
def similar(req, D, focus=None):
    rep = Report("", "similar"); types = specific(req) or ["ANY_WARNING"]; evs = D.events(types, D.t0, D.end + 1)
    if not evs: rep.title = "No event to compare against."; rep.ok = False; return rep
    anchor = evs[-1]; same = [e for e in D.events([anchor.type], D.t0, anchor.ts) if (dtc_code(e) == dtc_code(anchor) if dtc_code(anchor) else True)]
    rep.title = f"Events similar to the latest {EVLABEL(anchor.type)} ({D.stamp(anchor.ts)}: {anchor.detail})"
    if not same: rep.add("Result", "no earlier event of the same kind exists"); return rep
    sigs = [c for c in ("speed", "motor_temperature", "battery_temperature", "battery_current", "ambient_temperature", "coolant_flow") if D.available(c)]
    def sig(e):
        v = []
        for c in sigs:
            s = D.series(c, e.ts - 300, e.ts, e); v.append(float(np.median(s.v[-30:])) if s is not None and len(s.v) else float("nan"))
        return v
    base = np.array(sig(anchor)); rows = []; allv = np.array([sig(e) for e in same[-60:]])
    scale = np.nanstd(allv, axis=0) + 1e-6
    for e, v in zip(same[-60:], allv):
        d = float(np.nanmean(np.abs((v - base) / scale))); rows.append((d, e))
    rows.sort(key=lambda r: r[0]); rep.step(f"{len(same)} earlier events of the same kind; similarity = mean absolute difference of the last 5 minutes' median of {', '.join(sigs)}, in units of their spread")
    rep.add("Closest matches", *[f"{D.stamp(e.ts)} ({dur(anchor.ts - e.ts)} earlier): distance {d:.2f} - " + ", ".join(f"{D.label(c)} {fmtv(v, D.unit(c), 0)}" for c, v in zip(sigs, sig(e)) if not math.isnan(v))[:150] for d, e in rows[:3]])
    rep.add("The latest event", ", ".join(f"{D.label(c)} {fmtv(v, D.unit(c), 0)}" for c, v in zip(sigs, base) if not math.isnan(v)))
    seq = D.events(["POWER_DERATE", "WARN_OVERTEMP", "DTC_SET", "ECU_RESET"], anchor.ts - 300, anchor.ts + 300)
    rep.add("Sequence around the latest event", *[f"{hms(D, e.ts)} {EVLABEL(e.type)}" for e in seq[:6]])
    rep.add("What distinguishes them", "similarity is by operating conditions in the five minutes before the event; it says which earlier events were provoked by the same conditions, not that they share a cause")
    rep.evidence = {"period": f"{D.stamp(D.t0)} to {D.stamp(D.end)}", "period_ts": [D.t0, D.end], "samples": len(same), "sources": ["events", "signals"], "tier": "event context / 1-minute"}; return rep
