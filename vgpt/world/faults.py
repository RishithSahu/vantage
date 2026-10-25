"""Fault and scenario layer of the virtual vehicle: the error conditions and event sequences the case study asks to be recreated ("driving scenarios and error condition sequences").

The life generator (vgpt/datagen_life.py) already tells stories of physics (cooling-pump failure, tyre leak, roof box, weak 12 V battery, dirty windscreen, roadworks).  This layer adds the rest of what a
service or validation engineer asks about, each one WITH ground truth (what was injected, when, why), so answers can be scored and nothing in the history is an accident:

  ecu_resets_adas      the ADAS domain controller watchdog-resets in a cluster during highway lane keeping; each reset is followed within seconds by an ADAS disengagement ("system limit")
  power_events         cold-morning 12 V brown-outs (voltage dip -> infotainment reset), then ONE unexpected shutdown while driving: last valid data, a data gap, restart, U-codes logged after
  can_burst            a loose body-CAN connector: bursts of error frames, the body-bus signals (tyres, light, rain, 12 V, odometer) drop out, a U0140 DTC
  speed_conflict       a tyre-size change makes the Android speed read 6 % above the CAN speed: SENSOR_CONFLICT on motorway runs
  ambient_glitch       the dashboard's outside temperature shows -40 C for seconds while CAN is fine: SENSOR_CONFLICT with NO diagnostic code
  dcdc_intermittent    a DC/DC converter fault that appears only when speed > 100 km/h AND the motor is hot AND it is warm outside - in about 45 % of such episodes (and never otherwise),
                       preceded by a motor-overtemperature warning and a power derate: the sequence warning -> reduced acceleration -> diagnostic code
  thermal_soak         after hot drives the battery keeps heating for ~10 minutes after the stop
  dtc_cycle            some diagnostic codes clear at the next ignition cycle; one comes back days later
  adas_flapping        lane keeping switches on and off six times in three minutes on a few motorway trips
  aeb_events           automatic emergency braking acts three times (with a hard-braking event and a collapsing lead gap)
  range_glitch         the dashboard range estimate drops 15 % for twenty minutes in the middle of a short trip while the energy use is normal (a revised prediction, not consumption)
Everything is seeded; `truth` lists every injected item."""
import numpy as np

G = 9.80665
ALL = ("ecu_resets_adas", "power_events", "can_burst", "speed_conflict", "ambient_glitch", "dcdc_intermittent", "thermal_soak", "dtc_cycle", "adas_flapping", "aeb_events", "range_glitch")


def _runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)); return [(int(x), int(y)) for x, y in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


class Faults:
    def __init__(self, gv, profile, seed=0, which=ALL):
        self.gv, self.p = gv, profile; self.rng = np.random.default_rng(seed + 31337); self.t0 = gv.t0; self.n = gv.n; self.days = gv.n // 86400
        self.events, self.logs, self.gaps, self.dropouts, self.truth = [], [], [], {}, {}; self.source_log = []; self.dips = []; self.lead_dips = []; self.range_dips = []
        self.drive = (gv.mode == 1) | (gv.mode == 2)
        self.runs = _runs(self.drive)
        for name in which:
            fn = getattr(self, "_" + name, None)
            if fn is not None: fn()
        gv.events = sorted(list(gv.events) + self.events); gv.truth["faults"] = self.truth

    # ------------------------------------------------------------------------------------------------ helpers
    def ev(self, sec, ty, sev, detail): self.events.append((self.t0 + int(sec), ty, sev, detail))
    def log(self, sec, src, msg): self.logs.append((self.t0 + int(sec), src, msg))
    def rec(self, name, **kw): self.truth.setdefault(name, []).append(kw)
    def day_of(self, sec): return int(sec) // 86400

    def label(self, concept, prio):
        b = next((x for x in sorted(self.p.bindings, key=lambda x: x.priority) if x.concept == concept and x.priority >= prio), None)
        return f"{b.bus}/{b.key}" if b else concept

    def adas_runs(self, lo_day=0, hi_day=None, minimum=120):
        hi_day = self.days if hi_day is None else hi_day; a = self.gv.adas_on
        return [(int(s), int(e)) for s, e in _runs(a) if e - s >= minimum and lo_day * 86400 <= s < hi_day * 86400]

    def ensure_adas(self, lo_day, hi_day, need):
        """Motorway runs in the window with lane keeping on - switching it on in long motorway stretches when the driver profile left too few."""
        runs = self.adas_runs(lo_day, hi_day)
        if len(runs) >= need: return runs
        for s, e in self.runs:
            if lo_day * 86400 <= s < hi_day * 86400 and (self.gv.speed[s:e] > 72).sum() > 400:
                seg = np.where(self.gv.speed[s:e] > 72)[0]; a0, a1 = s + int(seg[0]) + 30, s + int(seg[-1]) - 30
                if a1 - a0 > 300: self.gv.adas_on[a0:a1] = True
            runs = self.adas_runs(lo_day, hi_day)
            if len(runs) >= need: break
        return runs

    # ------------------------------------------------------------------------------------------------ scenarios
    def _ecu_resets_adas(self):
        lo, hi = int(self.days * 0.6), int(self.days * 0.9); runs = self.ensure_adas(lo, hi, 6); r = self.rng
        if not runs: return
        for s, e in [runs[i] for i in r.choice(len(runs), size=min(len(runs), int(r.integers(6, 10))), replace=False)]:
            t = s + int(r.integers(30, max(31, e - s - 30))); self.ev(t - 1, "ECU_RESET", "warn", "ADAS domain controller reset (watchdog timeout)")
            self.ev(t + 1, "ADAS_DISENGAGE", "info", "system limit"); self.log(t - 2, "adas_ctrl", "ERR heartbeat to the gateway lost, restarting"); self.log(t + 7, "adas_ctrl", "WARN boot complete, calibration pending")
            self.gv.adas_on[t:t + int(r.integers(40, 100))] = False
            if r.random() < 0.3: self.ev(t + 8, "DTC_SET", "warn", "B1000 ECU internal fault (status 0x09: testFailed, confirmedDTC)")
            self.rec("ecu_resets_adas", ts=self.t0 + t - 1, ecu="ADAS domain controller", cause="watchdog timeout", adas_disengage_ts=self.t0 + t + 1)

    def _power_events(self):
        r = self.rng; st = self.gv.st.get("weak_12v"); d0 = st["start"] + 3 if st else int(self.days * 0.7)
        starts = [s for s, e in self.runs if s >= d0 * 86400 and (s % 86400) > 4 * 3600 and s < (self.days - 2) * 86400 and float(self.gv.amb[s]) < 18.0]
        if len(starts) < 4: starts = [s for s, e in self.runs if s >= d0 * 86400 and s < (self.days - 2) * 86400][:8]
        picks = sorted(int(x) for x in r.choice(starts, size=min(len(starts), 6), replace=False)) if starts else []
        for s in picks:
            vmin = float(r.uniform(8.2, 10.4)); ms = int(r.integers(120, 400)); self.ev(s + 2, "VOLTAGE_DIP", "warn", f"12 V supply dipped to {vmin:.1f} V for {ms} ms"); self.dips.append((s + 2, vmin, ms))
            self.log(s + 2, "power_mgr", f"WARN 12V rail dipped to {vmin:.1f} V ({ms} ms) during start-up")
            if r.random() < 0.5: self.ev(s + 3, "ECU_RESET", "warn", "infotainment reset (brown-out)"); self.log(s + 3, "ivi", "ERR brown-out reset of the infotainment unit")
            self.rec("voltage_dip", ts=self.t0 + s + 2, min_v=round(vmin, 1), duration_ms=ms)
        cand = [(s, e) for s, e in self.runs if picks and s > picks[-1] + 2 * 86400 and e - s > 600 and s < (self.days - 1) * 86400]
        if not cand: return
        s, e = cand[0]; seg = np.where((self.gv.speed[s:e] > 18) & (self.gv.speed[s:e] < 55))[0]
        if not len(seg): return
        t = s + int(seg[len(seg) // 2]); gap = int(r.integers(9, 16)); last = {k: float(a[t - 1]) for k, a in (("speed", self.gv.speed), ("battery_temperature", self.gv.batt), ("battery_soc", self.gv.soc), ("coolant_flow", self.gv.flow))}
        self.ev(t, "UNEXPECTED_SHUTDOWN", "critical", "unexpected shutdown (12 V supply loss); last valid data 120 ms before"); self.ev(t, "VOLTAGE_DIP", "warn", "12 V supply dipped to 6.4 V for 900 ms")
        self.log(t, "power_mgr", "CRIT 12V supply lost, high-voltage contactors opened"); self.log(t + 1, "bms", "ERR contactor open while current flowing")
        self.gaps.append((t, t + gap)); self.ev(t + gap, "ECU_RESET", "warn", "gateway reset (power-on)"); self.log(t + gap, "can_gw", "WARN restart after power loss")
        self.ev(t + gap + 2, "DTC_SET", "warn", "P0562 System voltage low (status 0x09: testFailed, confirmedDTC)"); self.ev(t + gap + 2, "DTC_SET", "warn", "U0155 Lost communication with instrument panel cluster (status 0x09: testFailed, confirmedDTC)")
        self.log(t + gap + 3, "diag", "WARN stored U0155 after restart"); self.dips.append((t, 6.4, 900))
        self.rec("unexpected_shutdown", ts=self.t0 + t, gap_s=gap, last_valid=last, resumed_ts=self.t0 + t + gap, dtcs=["P0562", "U0155"], cause="12 V supply loss", missing="no signal data for %d s; the cause is inferred from the voltage dip, not measured" % gap)

    def _can_burst(self):
        r = self.rng; lo, hi = int(self.days * 0.75), self.days - 2; bursts = []
        cand = [(s, e) for s, e in self.runs if lo * 86400 <= s < hi * 86400 and e - s > 300]
        if not cand: return
        for k, (s, e) in enumerate([cand[i] for i in r.choice(len(cand), size=min(len(cand), 12), replace=False)]):
            t = s + int(r.integers(60, max(61, e - s - 60))); dur = int(r.integers(15, 45)); n_err = int(r.integers(40, 400)); bursts.append((t, t + dur))
            self.ev(t, "CAN_ERROR", "warn", f"CAN error frames on the body bus: {n_err} in {dur} s (error-passive)"); sname = self.p.sources["body"].name if "body" in self.p.sources else "body bus"
            self.ev(t + 5, "SOURCE_SILENT", "warn", f"{sname}: no frames for {dur} s while the vehicle is awake"); self.source_log += [(self.t0 + t + 1, "body", "silent", f"{sname}: no frames for {dur} s while the vehicle is awake"), (self.t0 + t + dur, "body", "restored", f"{sname}: traffic is back after {dur} s")]; self.log(t, "can_gw", f"WARN body bus error counter 128 ({n_err} error frames)")
            self.dropouts.setdefault("body", []).append((t, t + dur)); self.rec("can_burst", ts=self.t0 + t, duration_s=dur, error_frames=n_err, bus="body", affected="tyre pressure, light, rain, 12 V, odometer")
            if k == 0: self.ev(t + 20, "DTC_SET", "warn", "U0140 Lost communication with body control module (status 0x09: testFailed, confirmedDTC)"); self.rec("can_burst_dtc", ts=self.t0 + t + 20, code="U0140")

    def _speed_conflict(self):
        r = self.rng; lo = int(self.days * 0.45); hi = lo + 6; last = -10 ** 9
        for s, e in self.runs:
            if not lo * 86400 <= s < hi * 86400: continue
            seg = np.where(self.gv.speed[s:e] > 70)[0]
            if len(seg) < 60 or s + int(seg[0]) - last < 120: continue
            t = s + int(seg[len(seg) // 3]); v = float(self.gv.speed[t]); last = t
            self.ev(t, "SENSOR_CONFLICT", "warn", f"vehicle speed: {self.label('speed', 1)} reads {v:.0f} km/h but {self.label('speed', 2)} reads {v * 1.06:.0f} km/h (differ by {v * 0.06:.2g} for more than 5 s)")
            self.rec("speed_conflict", ts=self.t0 + t, can_kmh=round(v, 1), vhal_kmh=round(v * 1.06, 1), cause="tyre size change: the Android speed reads 6 % above the CAN speed")

    def _ambient_glitch(self):
        r = self.rng; lo = int(self.days * 0.55); picks = sorted(int(x) for x in r.choice(np.arange(lo * 86400, (lo + 20) * 86400), size=60, replace=False))
        cnt = 0; used = set()
        for t in picks:
            run = next(((s, e) for s, e in self.runs if s <= t < e), None)
            if run is None or t // 86400 in used or cnt >= 12: continue
            used.add(t // 86400); cnt += 1; amb = float(self.gv.amb[t])
            self.ev(t, "SENSOR_CONFLICT", "warn", f"outside temperature: {self.label('ambient_temperature', 1)} reads {amb:.0f} °C but {self.label('ambient_temperature', 2)} reads -40 °C (differ by {amb + 40:.3g} for more than 5 s)")
            self.rec("ambient_glitch", ts=self.t0 + t, can_c=round(amb, 1), dashboard_c=-40.0, diagnostic_code="none recorded")

    def _dcdc_intermittent(self):
        gv, r = self.gv, self.rng; lo = int(self.days * 0.5)
        hot = (gv.speed > 100) & (gv.motor > 48) & (gv.amb > 24); eps = [(int(s), int(e)) for s, e in _runs(hot) if e - s >= 60 and s >= lo * 86400]
        if len(eps) < 4:
            hot = (gv.speed > 95) & (gv.motor > 40); eps = [(int(s), int(e)) for s, e in _runs(hot) if e - s >= 60 and s >= lo * 86400]
        for k, (s, e) in enumerate(eps):
            fault = (k % 2 == 0) if len(eps) < 8 else bool(r.random() < 0.45)
            self.rec("dcdc_condition", start=self.t0 + s, end=self.t0 + e, speed_max=round(float(gv.speed[s:e].max()), 1), motor_c=round(float(gv.motor[s:e].max()), 1), amb_c=round(float(gv.amb[s]), 1), fault=fault)
            if not fault: continue
            t = s + 50; self.ev(t, "WARN_OVERTEMP", "warn", f"motor temperature {float(gv.motor[t]) + 8:.0f} °C above its derate limit"); self.log(t, "pt_ctrl", "WARN motor temperature above derate threshold")
            self.ev(t + 40, "POWER_DERATE", "warn", "power limited to 70 % (motor temperature)"); self.log(t + 40, "pt_ctrl", "WARN torque limited to 70 %")
            self.ev(t + 95, "DTC_SET", "warn", "P0A94 DC/DC converter performance (status 0x09: testFailed, confirmedDTC)"); self.log(t + 95, "pt_ctrl", "ERR DC/DC converter output below target")

    def _thermal_soak(self):
        gv = self.gv; tt = np.arange(1, 900, dtype=np.float64); shape = (1 - np.exp(-tt / 90.0)) * np.exp(-tt / 420.0); shape /= shape.max()
        for s, e in self.runs:
            if e + 900 >= self.n or float(gv.batt[e - 1]) < 31.0: continue
            amp = float(np.clip((gv.batt[e - 1] - 28.0) * 0.45, 0.8, 4.5)); gv.batt[e + 1:e + 900] = gv.batt[e + 1:e + 900] + (amp * shape).astype(np.float32)
            self.rec("thermal_soak", stop=self.t0 + e, rise_c=round(amp, 1), minutes_to_peak=round(float(np.argmax(shape)) / 60.0, 1))

    def _dtc_cycle(self):
        dtcs = [ev for ev in self.gv.events if ev[1] == "DTC_SET" and ev[0] < self.t0 + (self.days - 6) * 86400][:3]; r = self.rng
        for k, (ts, ty, sv, de) in enumerate(dtcs):
            nxt = next((s for s, e in self.runs if s + self.t0 > ts + 8 * 3600), None)
            if nxt is None: continue
            code = de.split(" ")[0]; self.ev(nxt + 5, "DTC_CLEAR", "info", f"{code} cleared (status 0x00)"); self.rec("dtc_cleared", code=code, set_ts=ts, cleared_ts=self.t0 + nxt + 5, signal_normal=bool(k != 1))
            if k == 0:
                again = next((s for s, e in self.runs if s > nxt + 3 * 86400), None)
                if again is not None: self.ev(again + 60, "DTC_SET", "warn", de); self.rec("dtc_returned", code=code, ts=self.t0 + again + 60)

    def _adas_flapping(self):
        r = self.rng; lo, hi = int(self.days * 0.5), int(self.days * 0.95); runs = self.ensure_adas(lo, hi, 3)
        for s, e in [runs[i] for i in r.choice(len(runs), size=min(len(runs), 3), replace=False)] if runs else []:
            if e - s < 220: continue
            t0_ = s + int(r.integers(10, e - s - 190)); times = sorted(t0_ + int(x) for x in r.choice(np.arange(0, 170), size=6, replace=False)); state = 1
            for t in times:
                state = 1 - state; self.ev(t, "ADAS_STATE_CHANGE", "info", "driver-assistance state " + ("on" if state else "off"))
                if state == 0: self.ev(t, "ADAS_DISENGAGE", "info", "lane marking lost")
                self.gv.adas_on[t:t + 12] = bool(state)
            self.rec("adas_flapping", start=self.t0 + times[0], end=self.t0 + times[-1], toggles=6)

    def _aeb_events(self):
        r = self.rng; cand = [(s, e) for s, e in self.runs if e - s > 600 and s > 0.3 * self.n]
        for s, e in [cand[i] for i in r.choice(len(cand), size=min(len(cand), 3), replace=False)] if cand else []:
            seg = np.where(self.gv.speed[s:e] > 35)[0]
            if not len(seg): continue
            t = s + int(seg[len(seg) // 2]); v = float(self.gv.speed[t]); self.ev(t, "AEB_ACTIVATION", "warn", f"automatic emergency braking at {v:.0f} km/h (0.61 g)"); self.ev(t, "HARSH_BRAKE", "info", f"0.61 g at {v:.0f} km/h")
            self.lead_dips.append((t, 2.5)); self.log(t, "adas_ctrl", f"WARN AEB triggered, closing speed {v / 3.6 * 0.6:.1f} m/s"); self.rec("aeb", ts=self.t0 + t, speed_kmh=round(v, 1), decel_g=0.61)

    def _range_glitch(self):
        gv = self.gv; r = self.rng; short = [(s, e) for s, e in self.runs if 900 < e - s < 2400 and s > 0.3 * self.n]
        if not short: return
        s, e = short[int(r.integers(len(short)))]; a = s + (e - s) // 3; b = a + 1200; m0, m1 = a // 60, b // 60
        self.range_dips.append((m0, m1, 0.85)); self.rec("range_glitch", start=self.t0 + a, end=self.t0 + b, drop_pct=15, cause="revised prediction on the dashboard; battery energy use was normal")

    # ------------------------------------------------------------------------------------------------ shaping a day of canonical channels
    def shape_day(self, d, a, out):
        b = a + 86400; sig_of = {}
        for bd in self.p.bindings: sig_of.setdefault(bd.source, set()).add(bd.concept)
        for src, ivs in self.dropouts.items():
            for (s, e) in ivs:
                if a <= s < b:
                    for c in sig_of.get(src, ()):
                        if c in out: out[c][s - a:min(e - a, 86400)] = np.nan
        for (s, e) in self.gaps:
            if a <= s < b:
                for k, arr in out.items():
                    if isinstance(arr, np.ndarray) and arr.dtype.kind == "f": arr[s - a:min(e - a, 86400)] = np.nan
                out["_awake"][s - a:min(e - a, 86400)] = False
        for (t, vmin, ms) in self.dips:
            if a <= t < b:
                i = t - a; out["aux_battery_voltage"][i:i + 2] = vmin
        for (t, closing) in self.lead_dips:
            if a <= t < b:
                i = t - a; out["lead_vehicle_distance"][max(0, i - 6):i + 1] = np.linspace(25.0, 3.0, min(7, i + 1) if i < 6 else 7); out["lead_vehicle_distance"][i + 1:i + 8] = np.linspace(4.0, 18.0, 7)
        for (m0, m1, f) in self.range_dips:
            lo, hi = max(m0 * 60 - a, 0), min(m1 * 60 - a, 86400)
            if lo < hi: k = hi - lo; w = np.sin(np.linspace(0, np.pi, k)) ** 0.5; out["reported_range"][lo:hi] = out["reported_range"][lo:hi] * (1 - (1 - f) * w)
