"""QA part E: look-ahead (temporal leakage) checks, sensor quality, protocol / CAN questions, ontology and incidents, the assistant explaining its own reasoning, poisoned text, and the
compound / whole-history investigation questions. Every answer starts with its conclusion; that first line is what is read aloud."""
import os, re, math, random, datetime as dt
from collections import Counter
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd, SIGNED, plural, pfmt
from .qa_a import UNIT, NAME, WARN_TYPES, EVENT_SCORE
from .protocols import dbc as DBC

DAY = 86400
SRC = {"events": ["events table"], "ctx": ["1 Hz signals (live tier, or the frozen event context for old events)", "events table", "logs table"], "point": ["1 Hz signals (or per-minute for older times)", "GPS fix / tyre table at that time", "events table"],
       "extreme": ["1 Hz signals and per-minute aggregates"], "trend": ["per-minute signals aggregated to daily means"], "compare_periods": ["per-minute signals", "trips table"], "why_range2": ["trips table", "trip extension table (rain, hard events)", "per-minute signals", "tyre table"],
       "why_event": ["1 Hz context of the hour before the event", "per-minute signals for the baseline", "events table"], "evidence": ["the same sources as the diagnosis it supports"], "checkup": ["events table", "per-minute signals", "trips table"]}
ASSUME = {"point": ["RPM = vehicle speed x 9:1 ratio / 0.33 m wheel radius (an estimate)", "an unrecorded second means the car was off"],
          "why_range2": ["energy per km = kWh / km per local day", "the best two-level split of the days marks the change", "a linear trip-level model attributes the change (assumes effects add up)", "conditions the car does not record (payload, wind, gradient) are not in the model"],
          "why_event": ["'normal' is this car's own first two weeks", "heat balance is quasi-steady for the first-order estimates", "coolant flow is the pump's output (no pump signal exists)"],
          "trend": ["daily means represent the day", "a straight line is the shape tested", "days are independent (weather makes them slightly dependent)"]}


class MixE:
    # ------------------------------------------------------------------ incidents
    def incidents(self):
        evs = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN (%s) AND ts<=? ORDER BY ts" % ",".join("?" * len(WARN_TYPES)), (*WARN_TYPES, self.now)).fetchall(); inc = []
        for e in evs:
            thermal = bool(re.search(r"coolant|thermal|P1A|temperature|WARN_BATT", f"{e[1]} {e[3]}", re.I))
            if inc and (e[0] - inc[-1]["end"] <= 1800 or (thermal and inc[-1]["thermal"] and e[0] - inc[-1]["end"] <= 7 * DAY and self.flow_low_between(inc[-1]["end"], e[0]))):
                inc[-1]["events"].append(e); inc[-1]["end"] = e[0]; inc[-1]["thermal"] = inc[-1]["thermal"] or thermal
                inc[-1]["why"] = "within 30 minutes of each other" if e[0] - inc[-1]["events"][-2][0] <= 1800 else "the same low-coolant-flow condition persisted between them"
            else: inc.append({"start": e[0], "end": e[0], "events": [e], "thermal": thermal, "why": "single event"})
        return inc

    def flow_low_between(self, t0, t1):
        b = Q.flow_baseline(self.c, self.now)
        if not b: return False
        days = Q.flow_days(self.c, t0 - DAY, t1 + DAY)
        return bool(days) and all(d[1] < 0.9 * b["p50"] for d in days)

    def inc_label(self, i):
        kinds = Counter(Q.KIND_NAME.get({"WARN_BATT_OVERTEMP": "overheat", "DTC_SET": "diagnostic", "TPMS_LOW": "tyre", "CAMERA_BLOCKED": "camera", "CAMERA_DEGRADED": "camera"}.get(e[1], ""), e[1]) for e in i["events"])
        return ", ".join(f"{n} x {k}" for k, n in kinds.items())

    def h_onto(self, s, raw):
        if re.search(r"engine temperature.{0,40}coolant temperature|definitely the same measurement|same measurement", s):
            return ("No - they are three different things. This is an electric vehicle: there is no engine, so 'engine temperature' means the drive motor's (MotorTemp, Thermal message). Coolant temperature is not logged at all (only coolant flow, CoolantFlow). The battery has its own pack temperature (BattTemp).\n"
                    "They are related physically (the same coolant loop carries heat from both) but they are separate signals; merging them would hide which part is hot.")
        if re.search(r"fault codes?.{0,60}(same underlying|different names)|same underlying failure", s):
            codes = self.c.execute("SELECT DISTINCT substr(detail,1,5) FROM events WHERE type='DTC_SET'").fetchall(); ov = self.evs("overheat")
            self.brief = "Not by name alone: the coolant-flow code and the over-temperature warning can be one failure (low flow leads to heat); the tyre code is separate."
            return (f"Not by name alone. Here the codes are {', '.join(c[0] for c in codes) or 'none'}" + (f" plus {len(ov)} over-temperature warning(s)" if ov else "") + ": a 'coolant flow low' code (P1A00) and a battery 'over-temperature' warning are different codes but can be one failure - low flow leads to heat - when the low flow persists until the temperature warning. A tyre-pressure code (C0750) is unrelated.\n"
                    "Test used: same time window or same persisting condition, and a physical mechanism linking them; without both I treat them as separate.")
        if re.search(r"unify|dependency graph|references to the battery", s):
            return ("'Battery' means two things here: the traction battery (BmsStatus: Soc, PackVolt, PackCurrent, BattTemp, CellTempMax/Min, CellVoltMin; the stored state-of-health and fast-charge sessions) and the 12 V battery (rest voltage, low-voltage code P0562) - they are different systems.\n"
                    "Traction battery dependency graph: charger power (ChgKw) and drive power --> PackCurrent --> heat --> BattTemp / CellTemp; CoolantFlow --| removes heat from --> BattTemp; AmbTemp --> BattTemp; BattTemp --> over-temperature warning; Soc <-- PackCurrent integrated; state of health <-- long-term ageing. The 12 V battery depends on the DC-DC converter and parked drain, not on the pack.")
        if re.search(r"acceleration", s):
            return ("Directly: the accelerometer-derived hard-acceleration / hard-braking / hard-cornering events (0.28 g and 0.35 g thresholds). By derivation: the rate of change of vehicle speed (VehDyn.Speed) and of the GPS speed (gps_1s). Indirectly: battery power (torque demand), motor temperature, estimated RPM (a function of speed) and state-of-charge drop.\n"
                    "There is no throttle or steering signal in the data.")
        if re.search(r"hierarchy|subsystems", s):
            try: db = DBC.Database.load(os.path.join(os.path.dirname(__file__), "protocols", "vehicle.dbc")); nodes = {}
            except Exception: db = None
            tree = {}
            if db:
                for m in db.messages.values(): tree.setdefault(m.sender, []).append(f"{m.name} ({', '.join(x.name for x in m.signals)})")
            names = {"PT": "powertrain", "BMS": "traction battery", "THERM": "thermal management", "HVAC": "climate", "CHG": "charging"}
            self.brief = "Seven subsystems: powertrain, traction battery, thermal management, climate, charging, body and safety, and positioning."
            return "Subsystems in the data:\n" + "\n".join(f"  - {names.get(k, k)} [{k}]: {'; '.join(v)}" for k, v in tree.items()) + "\n  - body / safety: tyre pressure (TPMS), 12 V battery, brake wear, ADAS disengagements, front camera health\n  - positioning: GNSS (NMEA) position, heading, speed\n  - platform: logs from the infotainment / thermal services, diagnostic trouble codes"
        # incidents
        inc = self.incidents()
        if not inc: return "No warning or fault events are recorded, so there are no incidents to group."
        if re.search(r"evidence proves|prove.{0,30}same incident", s):
            multi = [i for i in inc if len(i["events"]) > 1]
            if not multi: return "Every event is a separate incident, so there is no pair to prove: no two events were close in time or linked by a persisting condition."
            i = multi[0]; self.brief = "They belong together when they are within 30 minutes of each other or the same abnormal condition persisted between them."
            return f"Two events belong together when they are within 30 minutes of each other or the same abnormal condition persisted between them; here: {self.fts(i['events'][0][0])} and {self.fts(i['events'][-1][0])} - {i['why']}.\nThat is evidence of association, not proof of a single cause."
        self.brief = f"{len(inc)} distinct incident{'s' if len(inc) != 1 else ''} from {sum(len(i['events']) for i in inc)} warning and fault events."
        out = [f"{len(inc)} distinct incidents from {sum(len(i['events']) for i in inc)} warning/fault events:"]
        for k, i in enumerate(inc, 1): out.append(f"  {k}. {self.fts(i['start'])[:16]}{' to ' + self.fts(i['end'])[:16] if i['end'] != i['start'] else ''}: {self.inc_label(i)} ({i['why']})")
        out.append("Rule: events within 30 minutes of each other, or linked by a persisting low-coolant-flow condition, are one incident. Different timestamps and codes can still be one physical failure.")
        return "\n".join(out)

    # ------------------------------------------------------------------ temporal leakage
    def h_avail_at(self, s, raw):
        self.brief = "A real-time system has everything recorded up to time T, but not the stored context around an event, daily summaries or later events."
        return ("At time T a real-time system has: every signal up to T (1 Hz for the last 7 days, per-minute before), the events and trouble codes logged up to T (a code is stored when the condition is first met), the GPS fixes up to T, and the baselines learned from earlier data.\n"
                "It does not have: anything after T; the frozen +/-15 minute context stored around an event (written after it); daily summaries for the day containing T (computed at day end); the trip row of the trip in progress (written when it ends); the daily state-of-health, brake-pad and 12 V values until that day closes.")

    def h_leak_feat(self, s, raw):
        self.brief = "Leaky: anything that looks past the event (the later half of the stored context, 'what changed afterwards', whole-history fits). Leak-free: the hour of signals before it."
        return ("Features in my analysis that would be future leakage if used to predict an event: the 'after' half of the frozen event context, 'what changed afterwards' comparisons, any daily or weekly average that includes the event's own day, change points and trip regressions fitted on the whole history, z-scores whose baseline includes the event, and the 'most serious event' ranking (it knows later events).\n"
                "Leak-free in my diagnoses: the hour of signals before the event, 'normal' taken from the car's first two weeks (kept before the period judged), and the flow-onset date computed only from data up to the event.")

    def h_cutoff(self, s, raw):
        e = self.target_event(s)
        if not e: return "No fault or warning event is recorded to re-analyse."
        m = re.search(r"(\d+) ?(second|sec|s\b)", s); mm = re.search(r"(\d+|five|ten|three|two|one) ?(minute|min)", s)
        if m: cut, win, lab = int(m[1]), 3600, f"only data up to {m[1]} seconds before the event"
        elif mm: n = Q.number(mm[1]) or 5; cut, win, lab = 0, int(n * 60), f"only the previous {mm[1]} minutes"
        else: cut, win, lab = 30, 3600, "only data up to 30 seconds before the event"
        d0, d1 = self.diag_event(e), self.diag_event(e, cut=cut, window=win)
        if not d1 or not d0: return "There is not enough data in that restricted window."
        self.ctx["cutdiag"] = d1
        same = d0["alts"][0]["name"] == d1["alts"][0]["name"]; self.brief = f"Using {lab}, the leading explanation is {'unchanged' if same else 'different'}: {d1['alts'][0]['name']} (confidence {d1['confidence']})."
        return (f"Using {lab}: the leading explanation is {'unchanged' if same else 'different'} - {d1['alts'][0]['name']} ({d1['alts'][0]['strength']}), confidence {d1['confidence']} (with everything: {d0['confidence']}).\n"
                f"  - evidence still available: {d1['alts'][0]['support']}\n  - records used: signals {self.fts(d1['used'][0])} to {self.fts(d1['used'][1])}; the 'normal' baseline from {self.day(d1['baseline_span'][0])} to {self.day(d1['baseline_span'][1])}; nothing after {self.fts(d1['used'][1])}.")

    def h_conf_change(self, s, raw):
        e = self.target_event(s)
        if not e: return "No event is recorded to re-analyse."
        rows = []
        for lab, cut, win in (("all data up to the event", 0, 3600), ("up to 30 s before", 30, 3600), ("up to 5 min before", 300, 3600), ("only the previous 5 minutes", 0, 300)):
            d = self.diag_event(e, cut=cut, window=win)
            if d: rows.append((lab, d["confidence"], d["alts"][0]["name"], d["alts"][0]["strength"]))
        if not rows: return "There is not enough data to compare."
        base = rows[0][1]; chg = [r for r in rows if r[1] != base]; self.brief = "Confidence " + ("barely changes" if not chg else "drops") + " when future observations are removed: the coolant-flow evidence builds up for days before the event."
        return (("Confidence barely changes when future observations are removed: " if not chg else "Confidence drops when future observations are removed: ") + "; ".join(f"{l}: {c}" for l, c, *_ in rows) + ".\n"
                "The coolant-flow evidence is a state that builds up for days, so it is visible well before the event; a rule based on it is a state flag, not a timer.")

    def h_used_records(self, s, raw):
        d = self.ctx.get("cutdiag") or self.ctx.get("diag")
        if not d or d.get("kind") != "event": d = self.diag_event(self.target_event(s)) if self.target_event(s) else None
        if not d or d.get("kind") != "event":
            return "My previous answer was not an event diagnosis, so there is no look-ahead window to show; it used the stored aggregates for the period named in the question (daily trips and per-minute signals) and nothing beyond that period's end."
        self.brief = "No - the diagnosis used only data up to the event; one caveat is that the 'normal' baseline can include part of an abnormal period."
        return (f"No - the diagnosis used signals from {self.fts(d['used'][0])} to {self.fts(d['used'][1])} (the event is at {self.fts(d['ts'])}), the car's 'normal' from {self.day(d['baseline_span'][0])} to {self.day(d['baseline_span'][1])}, and the coolant-flow onset day computed from data up to the event.\n"
                "One thing that is not strictly leak-free: if the event itself falls inside the baseline period, the 'normal' includes part of the abnormal behaviour; and the ranking of 'most serious event' uses all events.")

    def h_predict_before(self, s, raw):
        evs = self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('WARN_BATT_OVERTEMP','DTC_SET') ORDER BY ts").fetchall()
        if not evs: return "No warning or fault has occurred in this history, so there is nothing to predict from."
        b = Q.flow_baseline(self.c, self.now); out = []; hits = 0
        for ts, ty, det in evs:
            fo = Q.flow_onset(self.c, self.now, upto=ts - 300)
            if fo and re.search(r"coolant|thermal|temp|P1A|WARN", f"{ty} {det}", re.I): hits += 1; out.append(f"  - {self.fts(ts)[:16]} {Q.safe_text(det)[:44]}: would have been flagged from {self.day(fo[0])} ({Q.dur(ts - fo[0])} earlier) - daily median coolant flow {fo[1]:.1f} vs {fo[2]:.1f} L/min")
            else: out.append(f"  - {self.fts(ts)[:16]} {Q.safe_text(det)[:44]}: nothing in the signals before it signalled it")
        flagged_days = 0
        if b: flagged_days = len([d for d in Q.flow_days(self.c, self.cov()["lo"], self.cov()["hi"] + 1) if d[1] < 0.9 * b["p50"]])
        self.brief = f"Only as a state, not a time: a 'coolant flow below normal' rule would have flagged {hits} of {len(evs)} events days in advance, but it can't say when a warning will occur."
        return (f"Partly: using only data before each event, a 'coolant flow below normal' rule would have flagged {hits} of {len(evs)} events in advance - but as a persistent state, not a time. It cannot say when the warning will occur.\n" + "\n".join(out) +
                f"\nThe rule was true on {flagged_days} days and {len(evs)} events followed, so it cannot be validated: one vehicle and very few events. A real predictor needs many failures.")

    # ------------------------------------------------------------------ sensor quality
    def sensor_scan(self):
        c = self.c; rows = c.execute("SELECT ts, speed, soc, power, batt, flow, amb, motor, mode FROM sig_1m ORDER BY ts").fetchall(); issues = Counter(); ex = {}
        for a, b in zip(rows, rows[1:]):
            gap = b[0] - a[0]
            if gap == 60:
                if abs(b[4] - a[4]) > 50: issues["pack temperature jumped more than 5 C in a minute"] += 1; ex.setdefault("pack temperature jumped more than 5 C in a minute", b[0])
                if b[2] - a[2] > 5 and b[3] > 1000 and b[8] != 3: issues["state of charge rose while the car was discharging"] += 1; ex.setdefault("state of charge rose while the car was discharging", b[0])
            if b[1] > 3000: issues["speed above 300 km/h"] += 1; ex.setdefault("speed above 300 km/h", b[0])
            if b[8] == 0 and b[1] > 20: issues["speed while the car reported standstill mode"] += 1; ex.setdefault("speed while the car reported standstill mode", b[0])
            if b[5] > 200: issues["coolant flow above 20 L/min"] += 1; ex.setdefault("coolant flow above 20 L/min", b[0])
        return rows, issues, ex

    def h_inconsistent(self, s, raw):
        rows, issues, ex = self.sensor_scan()
        out = []
        for col, nm in ((4, "pack temperature"), (7, "motor temperature"), (5, "coolant flow")):
            v = sorted(r[col] for r in rows if r[col] is not None)
            if len(v) < 50: continue
            med = v[len(v) // 2]; mad = sorted(abs(x - med) for x in v)[len(v) // 2] or 1.0; out_ = [r for r in rows if abs(r[col] - med) > 8 * 1.4826 * mad]
            if col == 5: continue
            if out_: out.append(f"{len(out_)} per-minute {nm} readings lie more than 8 robust standard deviations from the median (first at {self.fts(out_[0][0])}, {out_[0][col] / 10:.1f} C)")
        if not issues and not out: return f"I found no physically inconsistent records in {len(rows)} per-minute rows: no impossible jumps, no speed above 300 km/h, no charge rising while discharging, no speed in standstill mode."
        return "Possible inconsistencies: " + "; ".join(f"{n} x {k} (first {self.fts(ex[k])})" for k, n in issues.items()) + ("; " if issues and out else "") + "; ".join(out) + f".\nChecked {len(rows)} per-minute rows. A flagged record is a candidate for review, not proof of a faulty sensor."

    def h_sensor_faulty(self, s, raw):
        e = self.target_event(s)
        if not e: return "No over-temperature warning is recorded, so there is no suspect reading to check."
        ts = e[0]; sg = Q.event_sig(self.c, ts, 1800, 60)
        if len(sg) < 120: return "There is not enough 1 Hz data around the event to check the sensor."
        bt, pw = sg.cols["batt"], sg.cols["power"]; step = max(abs(bt[i] - bt[i - 1]) for i in range(1, len(bt)) if sg.ts[i] - sg.ts[i - 1] <= 2)
        rate = [(bt[i + 60] - bt[i]) / 60 * 60 for i in range(0, len(bt) - 60, 30)]; load = [abs(mean(pw[i:i + 60])) for i in range(0, len(bt) - 60, 30)]; r = Q.pearson(load, rate)
        smooth = step < 1.0; follows = r is not None and r > 0.3
        verdict = "The battery most likely really overheated" if smooth and follows else "The reading is suspicious" if not smooth else "The reading looks genuine but I can't confirm it"
        head = (f"{verdict}: the pack-temperature trace is smooth (largest 1-second step {step:.2f} C) and its rate of rise follows the power flowing into the pack (r = {r:+.2f})." if r is not None else f"{verdict}: the trace is smooth (largest 1-second step {step:.2f} C).")
        return head + "\nA failing sensor usually gives jumps, flat-lines or readings that ignore the load. There is only one pack-temperature sensor in the data (cell max/min exist in the CAN database but are not stored), so a second independent sensor is needed to be sure."

    def h_drift(self, s, raw):
        rows = self.c.execute("SELECT ts, batt, amb, motor FROM sig_1m ORDER BY ts").fetchall(); soak = []
        for a, b in zip(rows, rows[1:]):
            if b[0] - a[0] >= 3 * 3600: soak.append((b[0], (b[1] - b[2]) / 10.0, (b[3] - b[2]) / 10.0))
        if len(soak) < 5: return "There are too few long parked periods (over 3 hours) to use as a calibration-free reference."
        d = [x[1] for x in soak]; m = [x[2] for x in soak]; f = Q.slope_test([x[0] / DAY for x in soak], d)
        bias = mean(d); h = Q.tcrit(len(d) - 1) * sd(d) / math.sqrt(len(d))
        return (f"No clear drift: after long parked periods the pack and motor have cooled to the air temperature, and the pack reads {bias:+.2f} C relative to the outside sensor (95% CI {bias - h:+.2f} to {bias + h:+.2f}, {len(soak)} soaks), trend {f['slope']:+.3f} C/day ({pfmt(f['p'])})." if f and f['p'] > 0.05 and abs(bias) < 1.5 else
                f"Possible bias: after long parked periods the pack reads {bias:+.2f} C relative to the outside sensor (95% CI {bias - h:+.2f} to {bias + h:+.2f}, {len(soak)} soaks)" + (f", drifting {f['slope']:+.3f} C/day ({pfmt(f['p'])})" if f else "") + ".\n") + \
               f"\nMethod: with no calibration reference, use physics - a vehicle left for 6+ hours is in thermal equilibrium, so any sensor reading far from the ambient sensor is bias in one of the two. The motor reads {mean(m):+.2f} C from ambient after soak. It cannot say which of the two sensors is off."

    def h_glitch(self, s, raw):
        sg = self.sig(*self.recent(7)[:2])
        if len(sg) < 500 or sg.res != 1: return "I need the 1 Hz data for that and there is not enough of it."
        out = []
        for col, nm, un in (("batt", "pack temperature", "C"), ("flow", "coolant flow", "L/min"), ("speed", "speed", "km/h")):
            v = sg.cols[col]; j = max(range(1, len(v)), key=lambda i: abs(v[i] - v[i - 1]) if sg.ts[i] - sg.ts[i - 1] <= 2 else 0); d = v[j] - v[j - 1]
            if sg.ts[j] - sg.ts[j - 1] > 2: continue
            after = v[j:j + 30]; persists = len(after) > 5 and abs(mean(after) - v[j - 1]) > 0.5 * abs(d)
            out.append(f"  - {nm}: largest 1-second jump {d:+.1f} {un} at {self.fts(sg.ts[j])}; it {'persisted for the next 30 seconds (a real change)' if persists else 'reverted within seconds (looks like a glitch)'}")
        return "A sudden change is a real event if it persists and its neighbours agree, and a glitch if it is a single-sample spike that reverts. The biggest jumps in the last 7 days:\n" + "\n".join(out)

    def h_reconstruct(self, s, raw):
        times = re.findall(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", s)
        if len(times) < 2: return "Give me the two timestamps (for example 'between 10:00 and 10:05 yesterday') and which signal, and I will fill the gap with an interpolation and say how uncertain it is."
        d, expl, _ = Q.find_date(s, self.now, self.tz)
        d = d or self.today(); base = Q.local_midnight(d, self.tz)
        t0, t1 = [base + int(h) * 3600 + int(m) * 60 + int(sec or 0) for h, m, sec in times[:2]]
        if t1 < t0: t0, t1 = t1, t0
        sg = Q.load(self.c, t0 - 120, t1 + 121)
        inside = [i for i, t in enumerate(sg.ts) if t0 <= t <= t1]
        if inside: return f"There is no gap to reconstruct: {len(inside)} samples are recorded between {self.fts(t0)} and {self.fts(t1)}."
        before = [i for i, t in enumerate(sg.ts) if t < t0]; after = [i for i, t in enumerate(sg.ts) if t > t1]
        if not before or not after: return f"Nothing was recorded within two minutes on either side of that interval - the car was off (it records only while on or charging), so there is no telemetry to reconstruct and any values would be invented."
        i, j = before[-1], after[0]; m = Q.metric_of(s) or "speed"; m = m if m in Q.SIGS else "speed"; v0, v1 = sg.cols[m][i], sg.cols[m][j]; span = sg.ts[j] - sg.ts[i]
        steps = [abs(sg.cols[m][k] - sg.cols[m][k - 1]) for k in range(1, len(sg)) if sg.ts[k] - sg.ts[k - 1] <= 2 * sg.res]; unc = 2 * (mean(steps) if steps else 0) * math.sqrt(span / max(sg.res, 1)) / 2
        mid = (v0 + v1) / 2
        return f"Midpoint estimate of {NAME.get(m, m).lower()}: {mid:.1f} {UNIT.get(m, '')} +/- {unc:.1f} (about 95%), by straight-line interpolation between {v0:.1f} at {self.fts(sg.ts[i])} and {v1:.1f} at {self.fts(sg.ts[j])} ({Q.dur(span)} gap).\nThe interval widens with the square root of the gap using the signal's typical one-step change; for a fast-moving signal like speed the true path may leave it, because interpolation cannot know about stops or peaks."

    def h_corrupt5(self, s, raw):
        p1, p2, n1, n2 = self.two_periods(s)
        a = [t["wh_km"] for t in Q.trips_between(self.c, *p1) if t["km"] >= 3]; b = [t["wh_km"] for t in Q.trips_between(self.c, *p2) if t["km"] >= 3]
        if len(a) < 8 or len(b) < 8: return "There are too few trips in the two periods to test the effect of corrupted readings."
        base = Q.welch_ci(a, b); rnd = random.Random(11); sigs = 0; diffs = []
        for _ in range(500):
            aa = [v * rnd.uniform(0.3, 3.0) if rnd.random() < 0.05 else v for v in a]; bb = [v * rnd.uniform(0.3, 3.0) if rnd.random() < 0.05 else v for v in b]; r = Q.welch_ci(aa, bb)
            if r: diffs.append(r["diff"]); sigs += (r["p"] < 0.05) == (base["p"] < 0.05)
        diffs.sort(); self.brief = f"The conclusion {'would hold' if sigs / len(diffs) >= 0.9 else 'is fragile'} with 5% of readings corrupted: the same verdict in {100 * sigs / len(diffs):.0f}% of trials."
        return (f"The conclusion {'would hold' if sigs / len(diffs) >= 0.9 else 'is fragile'}: with 5% of the trip readings randomly corrupted (multiplied by 0.3x-3x, 500 trials) the finding '{n1} vs {n2}: {base['diff']:+.1f} Wh/km, {pfmt(base['p'])}' keeps the same significance verdict in {100 * sigs / len(diffs):.0f}% of trials; the estimated difference ranges {diffs[int(0.025 * len(diffs))]:+.1f} to {diffs[int(0.975 * len(diffs))]:+.1f} Wh/km (95% of trials).\n"
                "Robust statistics (medians, rank tests) would reduce the sensitivity; the CAN checksums and plausibility filters already stop most corrupted frames before storage.")

    def h_same_phen(self, s, raw):
        out = []; sg = self.sig(*self.recent(7)[:2])
        if len(sg) > 500 and self._has("gps_1s"):
            g = {r[0]: r[1] / 10.0 for r in self.c.execute("SELECT ts, spd FROM gps_1s WHERE ts>=? AND ts<?", (sg.ts[0], sg.ts[-1] + 1))}
            pr = [(v, g[t]) for t, v in zip(sg.ts, sg.cols["speed"]) if t in g]
            if len(pr) > 200: r = Q.pearson([a for a, _ in pr], [b for _, b in pr]); out.append(f"  - CAN vehicle speed vs GPS speed: r = {r:+.3f} over {len(pr)} seconds -> the same quantity measured twice")
        if len(sg) > 500:
            ex = [(b - a) for b, a in zip(sg.cols["batt"], sg.cols["amb"])]; out.append(f"  - pack temperature vs motor temperature: r = {Q.pearson(sg.cols['batt'][::10], sg.cols['motor'][::10]):+.2f} -> related through shared coolant, not the same quantity")
            out.append(f"  - battery power vs rate of state-of-charge change: r = {Q.pearson([abs(p) for p in sg.cols['power'][::60]], [abs(sg.cols['soc'][i + 60] - sg.cols['soc'][i]) if i + 60 < len(sg) else 0 for i in range(0, len(sg), 60)]):+.2f} -> state of charge is the integral of power")
        if out: self.brief = "CAN speed and GPS speed are the same quantity measured twice; pack and motor temperature are related but different."
        return ("Only a pair like CAN speed and GPS speed is a genuine duplicate measurement of one phenomenon:\n" + "\n".join(out)) if out else "I found no pair of signals that clearly measure the same thing; high correlation alone would also arise from a shared cause."

    def h_sampling(self, s, raw):
        sg = self.sig(*self.recent(7)[:2])
        if len(sg) < 2000 or sg.res != 1: return "The 1 Hz data needed for this is not available; the older per-minute tier cannot resolve anything shorter than about two minutes."
        bt = sg.cols["batt"]; ac = Q.autocorr(bt[::10], 6)
        e = self.target_event(s); dur = None
        if e:
            d = self.diag_event(e); dur = d["sg"].ts[-1] - next((t for t, v in zip(d["sg"].ts, d["sg"].cols["batt"]) if v > d["sg"].cols["batt"][0] + 3), d["sg"].ts[-1]) if d else None
        return (f"Yes for thermal anomalies, no for fast ones: at 1 Hz the pack-temperature rise that ended in the warning developed over {Q.dur(dur) if dur and dur > 5 else 'minutes'}, hundreds of samples, far above the 2-samples-per-cycle Nyquist limit (pack temperature changes at most {max(abs(bt[i] - bt[i - 1]) for i in range(1, len(bt))):.2f} C per second). "
                "The per-minute tier (kept 90 days) is enough for slow drifts but would miss anything shorter than ~2 minutes: hard braking (1-3 s), glitches and the exact start of a spike.")

    # ------------------------------------------------------------------ protocol-level
    def dbc(self):
        if not hasattr(MixE, "_dbc"):
            try: MixE._dbc = DBC.Database.load(os.path.join(os.path.dirname(__file__), "protocols", "vehicle.dbc"))
            except Exception: MixE._dbc = None
        return MixE._dbc

    def h_proto_decode(self, s, raw):
        db = self.dbc()
        if not db: return "The vehicle's CAN database is not available."
        m = re.search(r"(?:0x)?([0-9a-f]{2,3})\b[^0-9a-f]{0,6}((?:[0-9a-f]{2}[ ,]?){4,16})", raw.lower().replace("0x", " "))
        hexb = re.search(r"((?:\b[0-9a-f]{2}\b[ ,]*){3,64})", raw.lower().replace("0x", " "))
        idm = re.search(r"(?:0x|id )([0-9a-f]{2,3})\b", raw.lower())
        if idm and hexb:
            fid = int(idm[1], 16); msg = db.messages.get(fid); data = bytes(int(x, 16) for x in re.findall(r"\b[0-9a-f]{2}\b", hexb[1])[1:] if True) if False else None
            tail = raw.lower().split(idm[0], 1)[1]; bs = re.findall(r"\b([0-9a-f]{2})\b", tail.replace("0x", " ")); data = bytes(int(x, 16) for x in bs)
            if not msg: return f"CAN ID 0x{fid:X} is not in this vehicle's database (it defines " + ", ".join(f"0x{k:X} {v.name}" for k, v in db.messages.items()) + ")."
            if len(data) < min(msg.dlc, 1): return "I need the payload bytes after the ID."
            dec = msg.decode(data.ljust(msg.dlc, b"\0")); return f"0x{fid:X} is {msg.name} (sent by {msg.sender}): " + ", ".join(f"{k} = {v:g} {next((x.unit for x in msg.signals if x.name == k), '')}" for k, v in dec.items()) + ".\nDecoded with the DBC's start bit, length, byte order, scale and offset; a payload shorter than the message is zero-padded."
        return "Give me the CAN ID and the payload bytes, for example 'decode 0x100 E8 03 00 00 00 00 00 00' - the message is decoded with the vehicle's CAN database (vehicle.dbc); I won't guess an ID."

    def h_proto_map(self, s, raw):
        db = self.dbc()
        if not db: return "The vehicle's CAN database is not available."
        msgs = list(db.messages.values()); sig = lambda m, n: next((x for x in m.signals if x.name == n), None)
        if re.search(r"vehicle.?speed|speed signal", s):
            m = next(x for x in msgs if sig(x, "Speed")); g = sig(m, "Speed")
            return f"CAN message {m.name} (ID 0x{m.frame_id:X}, {m.dlc} bytes, sent by {m.sender}) carries the vehicle speed as signal 'Speed': start bit {g.start}, length {g.length} bits, {'little-endian (Intel)' if g.little else 'big-endian (Motorola)'}, {'signed' if g.signed else 'unsigned'}, scale {g.factor:g}, offset {g.offset:g}, unit {g.unit}."
        if re.search(r"byte order|scaling|offset", s):
            return "Each signal has its own definition in the DBC: " + "; ".join(f"{m.name}.{g.name}: bit {g.start}|{g.length}, {'LE' if g.little else 'BE'}, x{g.factor:g} {g.offset:+g}" for m in msgs for g in m.signals[:2]) + ". Tell me the signal name for one."
        if re.search(r"which ecu|transmitting", s): return "Senders per the DBC: " + "; ".join(f"{m.name} (0x{m.frame_id:X}) from {m.sender}" for m in msgs) + ". The ECU is the DBC's 'sender' node - I can't see wiring."
        if re.search(r"can.{0,5}fd.{0,30}some|some/?ip|another source|determine whether this signal", s):
            fd = [m.name for m in msgs if m.dlc > 8]
            return f"By the database: messages longer than 8 bytes are CAN-FD ({', '.join(fd) or 'none'}), the rest classic CAN. SOME/IP carries service data (not CAN frames); VHAL carries vehicle properties; NMEA carries GNSS. A raw signal name alone does not say which bus delivered it - the source adapter does."
        if re.search(r"dbc", s): return f"The vehicle's CAN database is vehicle.dbc: {len(msgs)} messages (" + ", ".join(f"{m.name} 0x{m.frame_id:X}" for m in msgs) + f") and {sum(len(m.signals) for m in msgs)} signals."
        if re.search(r"subsystem|raw can id|map this", s): self.brief = "Five CAN messages: powertrain 0x100, battery 0x200, thermal 0x210, climate 0x300 and charger 0x310."; return "ID to subsystem per the DBC: " + "; ".join(f"0x{m.frame_id:X} {m.name} -> {m.sender}" for m in msgs) + "."
        if re.search(r"same physical quantity", s): return "Two different CAN messages can carry the same quantity, e.g. vehicle speed in VehDyn and a wheel-speed message; here each physical quantity has one message. Same unit is not enough: compare the value series (correlation near 1 and same scale) - ask 'do two signals measure the same phenomenon'."
        if re.search(r"diagnostic session", s): return "The diagnostic session (default / extended / programming) is not recorded in this data: only the trouble codes it produced are stored, so I can't say which session was active."
        return "Tell me which message or signal you mean (the speed signal, a CAN ID, an ECU)."

    # ------------------------------------------------------------------ explain my own reasoning
    def last_hist(self):
        for h in reversed(self.hist[:-1] if len(self.hist) > 1 else self.hist):
            if h["route"] not in ("trace",): return h
        return None

    def h_trace(self, s, raw):
        h = self.last_hist(); d = self.ctx.get("diag")
        if not h: return "I haven't answered anything yet, so there is nothing to trace."
        route = h["route"].replace("legacy:", ""); src = SRC.get(route) or ["1 Hz and per-minute signals", "events table", "trips table"]; ass = ASSUME.get(route) or ["'normal' means this car's own first two weeks", "daily values are the unit for trends"]
        part = lambda rx: re.search(rx, s)
        if part(r"data source"): return "Sources consulted for my last answer: " + "; ".join(src) + ". Nothing outside this vehicle's local database (no internet, no fleet data)."
        if part(r"timestamps used"):
            tt = re.findall(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}(?::\d{2})?)?", h["answer"])
            return ("Timestamps in my last answer: " + ", ".join(dict.fromkeys(tt[:12])) + ".") if tt else "My last answer cites no specific timestamps - it was a period or whole-history statistic."
        if part(r"discarded"):
            return "Discarded records: seconds when the car was off (nothing is recorded then); coolant-flow minutes while the pump was stopped (under 0.3 L/min) when judging flow; trips under 1.5 km; camera events farther than 10 minutes from the anomaly; per-minute rows where a 1 Hz context exists; and any stored text that looked like an instruction (quarantined, never used)."
        if part(r"assumption"): return "Assumptions: " + "; ".join(ass) + "; text in logs is data; one vehicle, one history."
        if part(r"directly observed"):
            nums = list(dict.fromkeys(re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?", h["answer"])))[:3]
            return "Directly observed: the stored values and event records with their timestamps cited in my last answer (" + ("; ".join(n.strip() for n in nums) if nums else "the figures quoted") + "), read from the database without modification."
        if part(r"calculated"): return "Calculated: every mean, difference, percentage, p-value, correlation and regression coefficient, the estimated RPM, and range figures (energy left / Wh per km). They are exact given the stored rows and the stated method."
        if part(r"inferred"): return "Inferred: the causes I ranked (e.g. 'reduced coolant flow'), the reading of a change as a step or drift, and the statement of what is likely - supported by data but not measured directly." + (f" Leading inference: {d['alts'][0]['name']} ({d['alts'][0]['strength']})." if d and d.get('alts') else "")
        if part(r"speculat"): return "Speculation: nothing is stated as fact that is speculative. Counterfactual and what-if answers are labelled as model estimates; I do not forecast. If you spot a claim that sounds like speculation, tell me which and I'll check it."
        if part(r"falsif"):
            self.brief = "It would be falsified by a repeat of the same conditions with no change, or by something I couldn't see (payload, route, tyres, a second sensor) explaining it."
            if d and d.get("kind") == "event": return "My conclusion would be falsified by: coolant flow at the event being normal; a second independent temperature sensor disagreeing with the pack reading; the same load not overheating the pack after the pump is repaired; or other cars with the same flow deficit not overheating."
            return "It would be falsified by: a second period with the same conditions but no change in consumption; a payload / route / tyre difference found to explain the change; or the effect vanishing under the multiple-testing correction."
        if part(r"strongest argument|wrong"): return self.h_wrong(s, raw)
        return "I used " + "; ".join(src) + "."

    def h_wrong(self, s, raw):
        inc = self.incidents(); n_ev = sum(len(i["events"]) for i in inc); d = self.diag_default(s); cv = self.cov(); days = (cv["hi"] - cv["lo"]) // DAY
        pts = [f"It rests on one vehicle and {days} days of history: {n_ev} warning/fault events in {len(inc)} incidents is too few to separate a real pattern from chance.",
               "Causes are inferred from indirect signals; there is no pump speed, coolant temperature or cell-level data, so a faulty flow sensor (rather than a weak pump) would give the same numbers.",
               "'Normal' is this car's own first two weeks - if the car was already degraded then, every deviation I measured is understated, and if the weather changed, some of it is weather."]
        if d and d.get("kind") == "range": pts.append(f"The efficiency change has p = {d['w'][2]:.2f} with {d['days_a']} vs {d['days_b']} days, and {100 * (1 - (d['res']['explained'] / d['res']['gap'])) if d.get('res') and d['res']['gap'] else 100:.0f}% of it is unexplained by recorded conditions.")
        if d and d.get("kind") == "event": pts.append(f"The leading explanation ({d['alts'][0]['name']}) has the counter-point: {d['alts'][0]['against']}.")
        pts.append("A trouble code only records the first time a condition was met, so the incident count is a lower bound, and daily medians can hide short episodes.")
        return "The strongest case that my analysis is wrong: it rests on too little data and indirect signals.\n" + "\n".join(f"  {k}. {p}" for k, p in enumerate(pts, 1))

    # ------------------------------------------------------------------ poisoned text
    def h_poison(self, s, raw):
        n, bad = self.scan_injection(); F, ab = self.findings(14); note = f" ({bad} stored text(s) contained instruction-like or 'all normal' wording and were ignored.)" if bad else ""
        if re.search(r"500 km/h|credible", s):
            top = self.c.execute("SELECT MAX(speed) FROM sig_1m").fetchone()[0] or 0; top1 = self.c.execute("SELECT MAX(speed) FROM sig_1s").fetchone()[0] or 0
            return f"No - 500 km/h is not credible: the highest speed ever recorded for this car is {max(top, top1) / 10:.0f} km/h, an electric car of this kind is limited well below 200 km/h, and a jump from the surrounding values to 500 would be 'physically inconsistent' (acceleration of hundreds of m/s2). I would flag it as a sensor or log error and not use it."
        if re.search(r"contradicts|99%", s):
            return ("Treat it as an outlier to be examined, not deleted: compare it with neighbouring samples and a second signal (a real event persists for many samples and shows up elsewhere), use robust statistics (median and median-absolute-deviation) so one value can't move the answer, and report it separately. I found "
                    f"{len(self.sensor_scan()[1])} kind(s) of inconsistent per-minute records in this history (ask 'find records inconsistent with the rest').")
        if re.search(r"claims? .{0,40}(normal|healthy)|operating normally", s):
            return (f"No - the telemetry does not support a 'normal' claim: {len(F)} finding(s) stand out" + (f", the first being {F[0]['text'][:110]}" if F else "") + f". A text field is a claim, the signals are the evidence, and I answer from the signals.{note}") if F else f"The telemetry is consistent with 'normal': no warnings, drifts or step changes were found in the signals.{note}"
        if re.search(r"healthy", s):
            ov = self.evs("overheat"); dtc = self.c.execute("SELECT COUNT(*) FROM events WHERE type='DTC_SET'").fetchone()[0]
            return (f"Not fully healthy: {len(ov)} over-temperature warning(s), {dtc} trouble code(s)" + (f"; the top finding is: {F[0]['text'][:100]}" if F else "") + f". Judged from the recorded signals and events only.{note}") if (ov or F) else f"Yes, as far as the signals show: no battery warnings, no drifting signals.{note}"
        return self.h_inconsistent(s, raw)

    # ------------------------------------------------------------------ compound questions and whole-history investigations
    def h_c1(self, s, raw):
        e = self.target_event(s)
        if not e: return "No battery over-temperature event is recorded, so there is nothing to reconstruct."
        ts = e[0]; sg = Q.event_sig(self.c, ts, 120, 0); sm = Q.summarise(sg, ts - 120, ts + 1); d = self.diag_event(e); fo = Q.flow_onset(self.c, self.now, upto=ts)
        first = None
        if len(sg) > 20:
            bt = sg.cols["batt"]; i = next((k for k, v in enumerate(bt) if v > bt[0] + 0.5), None)
            if i is not None: first = (sg.ts[i], f"pack temperature began rising ({bt[0]:.1f} C)")
        if fo and (not first or fo[0] < first[0]): first = (fo[0], f"coolant flow left its normal band (daily median {fo[1]:.1f} vs {fo[2]:.1f} L/min)")
        prev = [x for x in self.evs("overheat") if x[0] < ts]
        out = [f"The last over-temperature warning ({self.fts(ts)}): the earliest abnormal sign was that {first[1] if first else 'not identifiable'}" + (f", {Q.dur(ts - first[0])} before" if first else "") + f"; most likely contributor: {d['alts'][0]['name'] if d else 'unknown'}; confirming a battery fault needs more evidence than this data holds."]
        if sm: out.append(f"  - 2 minutes before: pack {sm['batt']['first']:.1f} -> {sm['batt']['last']:.1f} C, power {sm['power']['first']:.0f} -> {sm['power']['last']:.0f} kW, coolant flow {sm['flow']['mean']:.1f} L/min, speed {sm['speed']['mean']:.0f} km/h")
        if d: out.append(f"  - contributing factors: " + "; ".join(f"{a['name']} ({a['strength']})" for a in d["alts"][:3]))
        out.append("  - previous over-temperature warnings: " + ("; ".join(f"{self.fts(x[0])[:16]} at {Q.state_at(self.c, x[0]).get('flow', float('nan')):.1f} L/min flow" for x in prev[-3:]) if prev else "none - this is the only one"))
        self.brief = (f"The earliest sign was coolant flow leaving its normal band, {Q.dur(ts - first[0])} before the {self.day(ts)} warning. Likely contributor: reduced cooling; a confirmed battery fault needs more evidence than this data holds." if first and "coolant" in first[1] else
                      f"The last warning was on {self.day(ts)}; the likely contributor is {re.sub(r' [(].*?[)]', '', d['alts'][0]['name']) if d else 'unknown'}, but a confirmed battery fault needs more evidence than this data holds.")
        out.append("  - needed before calling it a confirmed battery fault: a second independent temperature sensor, pump speed/current, coolant level and temperature, cell-level temperatures, and a workshop test; this data shows cooling and heating signals but cannot identify a failed cell or cell chemistry fault.")
        return "\n".join(out)

    def h_c2(self, s, raw):
        p1, p2, n1, n2 = self.two_periods("first and last week"); rows = self.period_rows(p1, p2)
        T1, T2 = Q.trips_between(self.c, *p1), Q.trips_between(self.c, *p2)
        pm = lambda T: (sum(t["kwh"] for t in T) * 1000 / max(sum(t["km"] for t in T), 1), mean([(t["end"] - t["start"]) / 60 for t in T]) if T else 0)
        w1, m1 = pm(T1); w2, m2 = pm(T2); ps = [(Q.welch(a, b) or (0, 0, 1))[2] for _, _, a, b, _ in rows]; sig = Q.bh(ps) if ps else []
        names = [r[0] for r, k in zip(rows, sig) if k]
        deg = self.series_set(); fl = deg.get("coolant flow")
        self.brief = f"First vs last 7 days: energy {w1:.0f} to {w2:.0f} Wh/km; " + (f"only {names[0]} changed meaningfully" if len(names) == 1 else f"{len(names)} measures changed meaningfully" if names else "nothing changed meaningfully") + (" - the cooling system is degrading." if "Coolant flow (running)" in names else ".")
        out = [f"Comparing {n1} with {n2}, normalised per km and per trip: energy {w1:.0f} -> {w2:.0f} Wh/km; average trip {m1:.0f} -> {m2:.0f} minutes; {len(names)} of {len(rows)} measures differ meaningfully after false-discovery control ({', '.join(names) or 'none'}). " + ("Degradation is occurring in the cooling system." if fl and len(fl) > 6 and Q.slope_test([t / DAY for t, _ in fl], [v for _, v in fl])["p"] < 0.01 and fl[-1][1] < 0.9 * fl[0][1] else "No clear degradation.")]
        out += self.fmt_rows([r for r in rows if r[0] in names] or rows, top=5)
        out.append("Observed: the figures above, from stored data. Inferred: that the cooling drift is wear in the pump or circuit (not measured), and that nothing else changed between the weeks. Weather differences are not normalised away except in the regression.")
        return "\n".join(out)

    def h_c3(self, s, raw):
        e = self.most_serious()
        if not e: return "No anomaly is recorded in the history."
        d = self.diag_event(e); inc = self.incidents(); fo = Q.flow_onset(self.c, self.now, upto=e[0])
        sev = EVENT_SCORE.get(e[1], 1)
        self.brief = f"The most severe anomaly is the {self.day(e[0])} {Q.KIND_NAME.get({'WARN_BATT_OVERTEMP': 'overheat', 'DTC_SET': 'diagnostic'}.get(e[1], ''), e[1])}" + (f"; its earliest precursor was coolant flow, {Q.dur(e[0] - fo[0])} earlier" if fo else "") + ". The data supports 'reduced cooling contributed' but doesn't establish causation."
        out = [f"The most severe anomaly is {self.evrow(e)}: it is the highest-severity event type recorded (critical over-temperature outranks codes and warnings), it belongs to a thermal incident and is a safety-relevant fault."
               if e[1] == "WARN_BATT_OVERTEMP" else f"The most severe anomaly is {self.evrow(e)} - the highest-ranked event by severity and, among equals, the latest."]
        if fo: out.append(f"  - earliest precursor: coolant flow left its normal band on {self.day(fo[0])}, {Q.dur(e[0] - fo[0])} earlier (daily median {fo[1]:.1f} vs {fo[2]:.1f} L/min).")
        if d:
            pre = d["pre"]; out.append(f"  - how the signals changed: pack temperature {pre['batt']['first']:.1f} -> {pre['batt']['last']:.1f} C, coolant flow {pre['flow']['first']:.1f} -> {pre['flow']['last']:.1f} L/min, power mean {pre['power']['mean']:.0f} kW in the hour before.")
        out.append("  - is the data sufficient to establish causation? No: it supports 'reduced cooling was the main contributing condition' (temporal order, dose-response, physical mechanism) but without an intervention, a second sensor or part diagnostics, it does not establish it.")
        return "\n".join(out)

    def h_c4(self, s, raw):
        D = self.days_table()
        if len(D) < 8: return "There are too few days to say."
        en = [d["wh_km"] for d in D]; cands = [("outside temperature", "amb"), ("average speed", "speed"), ("hard acceleration / braking", "harsh"), ("pack temperature", "batt"), ("state of charge", "soc"), ("climate power", "hvac"), ("motorway share", "hwy")]
        cv = self.cov(); sc = {d: v for d, v, _ in self.daily("soc", cv["lo"], cv["hi"] + 1, modes=(1, 2, 3))}
        for d in D: d["soc"] = sc.get(d["ts"], 0.0)
        rows = []
        for lab, k in cands:
            x = [d[k] for d in D]
            if sd(x) == 0: continue
            r, rho = Q.pearson(x, en), Q.spearman(x, en); rows.append((abs(rho), lab, r, rho, Q.corr_p(rho, len(D))))
        rows.sort(reverse=True); fit = Q.ols_se([[(d[k] - mean([e[k] for e in D])) / (sd([e[k] for e in D]) or 1) for _, k in cands if sd([e[k] for e in D]) > 0] for d in D], en)
        if not rows: return "None of the available variables varies enough to test."
        lead = rows[0]; self.brief = f"Range loss is mostly associated with {lead[1]} (rank correlation {lead[3]:+.2f}); association, not proof."
        return (f"Reduced range is primarily associated with {lead[1]} (Spearman rho = {lead[3]:+.2f} with daily energy use, {pfmt(lead[4])}); the others rank: " + ", ".join(f"{l} {rho:+.2f}" for _, l, r, rho, p in rows[1:]) + ".\n" +
                (f"Together the variables explain R2 = {fit['r2']:.2f} of the day-to-day variation (standardised regression, {len(D)} days). " if fit else "") +
                ("Climate power is itself driven by outside temperature (r between them " + format(Q.pearson([d["hvac"] for d in D], [d["amb"] for d in D]) or 0, "+.2f") + "), so temperature is the underlying factor. " if lead[1] == "climate power" else "") + "Association, not proof: outside temperature moves heating load, speed and pack temperature together.")

    def h_c5(self, s, raw):
        evs = self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('DTC_SET','WARN_BATT_OVERTEMP') ORDER BY ts").fetchall(); out = []; n = 0
        b = Q.flow_baseline(self.c, self.now)
        for ts, ty, det in evs:
            sg = Q.event_sig(self.c, ts, 1800, 0)
            if len(sg) < 30: out.append(f"  - {self.fts(ts)[:16]} {Q.safe_text(det)[:40]}: no signal context stored"); continue
            cand = []
            bt = sg.cols["batt"]; i = next((k for k, v in enumerate(bt) if v > bt[0] + 2), None)
            if i is not None: cand.append((sg.ts[i], "pack temperature"))
            if b: j = next((k for k, v in enumerate(sg.cols["flow"]) if 0.3 < v < 0.65 * b["p50"]), None); (j is not None) and cand.append((sg.ts[j], "coolant flow"))
            pw = [abs(v) for v in sg.cols["power"]]; pk = max(pw)
            if pk > 5: k = next((k for k, v in enumerate(pw) if v > 0.5 * pk), None); (k is not None) and cand.append((sg.ts[k], "battery power"))
            if not cand: out.append(f"  - {self.fts(ts)[:16]} {Q.safe_text(det)[:40]}: no abnormal state in the 30 minutes before"); continue
            cand.sort(); lead = ts - cand[0][0]
            if lead <= 300: n += 1
            out.append(f"  - {self.fts(ts)[:16]} {Q.safe_text(det)[:40]}: earliest indication {cand[0][1]} {Q.dur(lead)} before" + (" (within 5 minutes)" if lead <= 300 else " (longer than 5 minutes: not counted)"))
        self.brief = f"{n} of {len(evs)} warnings were preceded within five minutes by an abnormal signal state."
        return f"{n} of {len(evs)} diagnostic/over-temperature warnings were preceded within five minutes by an abnormal signal state; the earliest indication in each is below.\n" + "\n".join(out)

    def h_nuclear1(self, s, raw):
        inc = self.incidents(); F, ab = self.findings(max(14, (self.cov()["hi"] - self.cov()["lo"]) // DAY)); dg = self.h_degrade("", "").split("\n"); d = self.diag_event(self.most_serious())
        _, issues, _ = self.sensor_scan(); deg = [l.strip(" -") for l in dg[1:] if l.strip().startswith("-") and "p <" in l][:2]
        out = [f"Independent investigation of {self.span_text()}: {len(inc)} incident(s) from {sum(len(i['events']) for i in inc)} events, " + (f"one long-term degradation (cooling) and " if any("coolant" in x for x in deg) else "no clear long-term degradation and ") + f"{'sensor inconsistencies found' if issues else 'no contradictory sensor readings'}; leading hypothesis: {d['alts'][0]['name'] if d and d.get('alts') else 'none'}."]
        out.append("1. Incidents: " + ("; ".join(f"{self.fts(i['start'])[:16]} {self.inc_label(i)}" for i in inc) if inc else "none"))
        out.append("2. Long-term degradation: " + ("; ".join(deg) if deg else "none beyond normal variation"))
        if d and d.get("alts"):
            out.append("3. Ranked hypotheses for the most serious event:")
            for k, a in enumerate(d["alts"][:4], 1): out.append(f"     {k}. {a['name']} - {a['strength']}. For: {a['support']}. Against: {a['against']}.")
        out.append("4. Possible causal relationships: low coolant flow -> less heat removal -> pack temperature climbs under load (mechanism + order in time); outside temperature -> heating load -> energy use (regression).")
        out.append(f"5. Contradictory sensor readings: {', '.join(issues) if issues else 'none found in the per-minute data'}.")
        out.append("6. Observation vs inference: observed = stored values and event times; calculated = statistics; inferred = causes; I did not assume any hypothesis because it is plausible - each is scored on its own evidence.")
        out.append("7. Uncertainty and missing information: one vehicle, " + f"{(self.cov()['hi'] - self.cov()['lo']) // DAY} days, few events; no pump speed, coolant temperature/level, cell temperatures, second sensor, payload, wind or gradient; hypotheses rest on indirect signals.")
        return "\n".join(out)

    def h_nuclear3(self, s, raw):
        inc = self.incidents()
        if not inc: return "No incident is recorded, so the leak-free re-run has nothing to analyse."
        out = []; kept = 0
        for i in inc:
            e = i["events"][0]; d0 = self.diag_event(e); d1 = self.diag_event(e, cut=300); fo = Q.flow_onset(self.c, self.now, upto=e[0] - 300)
            if d0 and d1 and d0["alts"][0]["name"] == d1["alts"][0]["name"]: kept += 1
            out.append(f"  - {self.fts(e[0])[:16]} {Q.safe_text(e[3])[:36]}: " + (f"leading explanation {'unchanged' if d0 and d1 and d0['alts'][0]['name'] == d1['alts'][0]['name'] else 'changed'}: {d1['alts'][0]['name']} ({d1['alts'][0]['strength']}, confidence {d1['confidence']})" if d1 else "no signal context") + (f"; coolant flow already flagged since {self.day(fo[0])}" if fo else ""))
        self.brief = f"Using only what was known 5 minutes before, the main conclusion holds for {kept} of {len(inc)} incident{'s' if len(inc) != 1 else ''}; only the confirmation of what happened next is lost."
        return (f"With only evidence from before each incident (cut off 5 minutes before it), the leading conclusion survives for {kept} of {len(inc)} incidents; what is lost is the confirmation of what happened next and the 'after' comparisons.\n" + "\n".join(out) +
                "\nRemoved: everything recorded after each incident, including the stored +/- context's later half, the daily summaries containing it and later events.")
