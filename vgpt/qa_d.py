"""QA part D: causal and counterfactual questions, variables the car does not record, fleet questions, long-horizon change and degradation, and real statistics (CIs, tests, effect sizes,
rank correlations, multiple-testing corrections, bootstrap). Every answer starts with its conclusion (that line is what is read aloud); the rest is the evidence."""
import re, math, random, datetime as dt
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd, SIGNED, plural, pfmt
from .qa_a import UNIT, NAME, WARN_TYPES

DAY = 86400


def thr_of(detail, default=None):
    m = re.search(r"above (\d+(?:\.\d+)?) ?C", detail or ""); return float(m[1]) if m else default


class MixD:
    # ------------------------------------------------------------------ small helpers
    def target_event(self, s=""):
        """The event a question is about: the last battery over-temperature warning, else the most serious event."""
        r = self.evs("overheat", newest_first=True, limit=1)
        return r[0] if r else self.most_serious()

    def days_table(self):
        """Per local day: efficiency, ambient, speed, motorway share, hard events, climate power, pack temperature (mean), coolant flow."""
        cv = self.cov(); D = Q.daily_series(self.c, cv["lo"], cv["hi"] + 1)
        bt = {d: v for d, v, _ in self.daily("batt", cv["lo"], cv["hi"] + 1, modes=(1, 2, 3))}; fl = {d: v for d, v, _ in self.daily("flow", cv["lo"], cv["hi"] + 1, modes=(1, 2, 3), minv=3.0)}
        for d in D: d["batt"] = bt.get(d["ts"]); d["flow"] = fl.get(d["ts"])
        return [d for d in D if d["batt"] is not None]

    def whkm_at(self, v):
        r = AX.eco_speed(self.c, self.now, "28")
        if not r.get("ok") or len(r["bins"]) < 2: return None, r
        pts = sorted(((b["lo"] + b["hi"]) / 2, b["wh_km"]) for b in r["bins"])
        if v <= pts[0][0]: return pts[0][1], r
        if v >= pts[-1][0]: return pts[-1][1], r
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= v <= x1: return y0 + (y1 - y0) * (v - x0) / (x1 - x0), r
        return None, r

    def cap_kwh(self): return A.cap_kwh(self.c)

    # ------------------------------------------------------------------ causal questions
    def h_effect_ctrl(self, s, raw):
        m = re.search(r"(?:above|over|faster than|exceeding) (\d+) ?(?:km|kph)", s); thr = float(m[1]) if m else 100.0
        sg, lo, hi, wtxt = self.sigwin(s)
        if sg.res != 1 or len(sg) < 1500: return "I need the 1 Hz data (kept 7 days) for that and there is not enough of it."
        sp, pw, bt, am, md = sg.cols["speed"], sg.cols["power"], sg.cols["batt"], sg.cols["amb"], sg.cols["mode"]; X, y, y0, X0 = [], [], [], []; W = 300
        for i in range(0, len(sp) - W, W):
            if sg.ts[i + W] - sg.ts[i] != W or min(md[i:i + W]) == 0 or 3 in md[i:i + W]: continue
            fr = sum(1 for v in sp[i:i + W] if v > thr) / W; p = mean([abs(v) for v in pw[i:i + W]]); a = mean(am[i:i + W]); ex = bt[i] - am[i]
            X.append([fr, p, a, ex]); X0.append([fr, a, ex]); y.append(bt[i + W] - bt[i])
        if len(X) < 30 or sum(1 for r in X if r[0] > 0.5) < 3: return f"The car spent too little time above {thr:.0f} km/h in {wtxt} ({sum(1 for r in X if r[0] > 0.5)} five-minute windows) to estimate an effect."
        f = Q.ols_se(X, y); f0 = Q.ols_se(X0, y)
        if not f: return "The regression could not be fitted (the variables are too collinear)."
        b, se, df = f["beta"][1], f["se"][1], f["df"]; h = Q.tcrit(df) * se
        out = [f"Holding load, outside temperature and starting temperature fixed, five minutes spent entirely above {thr:.0f} km/h changed the pack temperature by {b:+.2f} C (95% CI {b - h:+.2f} to {b + h:+.2f}, {pfmt(f['p'][1])}, {len(X)} windows) - an adjusted association, not proof of cause.",
               f"Without adjusting for load the figure is {f0['beta'][1]:+.2f} C: most of the heating of fast driving travels through power draw, which the adjustment removes (power is a mediator, so the adjusted number is the 'direct' effect only)." if f0 else "",
               f"Model: 5-minute temperature change ~ share of time above {thr:.0f} km/h + mean |battery power| + outside temperature + pack-minus-outside at the start (R2 {f['r2']:.2f}). Limits: no randomised speed changes, so confounding by route/traffic can't be excluded; one vehicle; {wtxt}."]
        self.brief = (f"Driving above {thr:.0f} km/h raises pack temperature by about {b:+.2f} C per five minutes once load and weather are held fixed." if f["p"][1] < 0.05 else
                      f"Once load and weather are held fixed, driving above {thr:.0f} km/h has no measurable direct effect on pack temperature ({b:+.2f} C per five minutes); its heating comes through power draw.")
        return "\n".join(x for x in out if x)

    def h_confounder(self, s, raw):
        D = self.days_table()
        if len(D) < 8: return "There are too few days of data to look for confounders."
        sp = [d["speed"] for d in D]; en = [d["wh_km"] for d in D]; base = Q.pearson(sp, en); rows = []
        for key, lab in (("amb", "outside temperature"), ("rain", "rain on the road"), ("harsh", "hard driving")):                          # things that act on both speed and consumption (not results of them)
            x = [d[key] for d in D]
            if sd(x) == 0: continue
            r1, r2 = Q.pearson(x, sp), Q.pearson(x, en); pc = Q.partial_corr(sp, en, [x])
            if r1 is not None and r2 is not None and pc is not None: rows.append((abs(pc - base), lab, r1, r2, pc))
        rows.sort(reverse=True)
        if not rows: return "No candidate confounder varies enough in the data."
        _, lab, r1, r2, pc = rows[0]; self.brief = f"{lab[0].upper() + lab[1:]} is the main confounder: it influences both speed and energy use."
        out = [f"{lab[0].upper() + lab[1:]} is the strongest confounder: it correlates with vehicle speed (r = {r1:+.2f}) and with energy use (r = {r2:+.2f}); the raw speed-vs-energy correlation of {base:+.2f} becomes {pc:+.2f} once it is held fixed."]
        out += [f"  - {l}: with speed {a:+.2f}, with energy {b:+.2f}, speed-vs-energy after controlling for it {c:+.2f}" for _, l, a, b, c in rows[1:4]]
        out.append(f"Daily values over {len(D)} days. A confounder influences both variables; outside temperature does (cold -> heating load raises consumption, and weather changes how people drive). Ranked by how much holding the variable fixed changes the speed-vs-consumption correlation.")
        return "\n".join(out)

    def h_root_cause(self, s, raw):
        e = self.target_event(s)
        if not e: return "No fault or warning event is recorded, so there is no fault to trace back to a root cause."
        ts = e[0]; fo = Q.flow_onset(self.c, self.now, upto=ts); sg = Q.event_sig(self.c, ts, 3600, 0); items = []
        if fo: items.append((fo[0], f"coolant flow left its normal band (daily median {fo[1]:.1f} vs {fo[2]:.1f} L/min)"))
        if len(sg) > 60:
            bt = sg.cols["batt"]; b0 = mean(bt[:60]); i = next((k for k, v in enumerate(bt) if v > b0 + 3), None)
            if i is not None: items.append((sg.ts[i], f"pack temperature began its climb ({b0:.1f} C -> +3 C)"))
            pw = [abs(v) for v in sg.cols["power"]]; pk = max(pw); j = next((k for k, v in enumerate(pw) if v > 0.5 * pk and pk > 3), None)
            if j is not None: items.append((sg.ts[j], f"battery power rose above {0.5 * pk:.0f} kW"))
            fb = Q.flow_baseline(self.c, self.now)
            if fb:
                fl = sg.cols["flow"]; k = next((i for i in range(10, len(fl)) if 0.3 < fl[i] < 0.75 * fb["p50"] and all(0.3 < x < 0.8 * fb["p50"] for x in fl[i:i + 10])), None)
                if k is not None and not any(it[1].startswith("coolant flow left") for it in items): items.append((sg.ts[k], f"coolant flow fell below 75% of normal ({fl[k]:.1f} vs {fb['p50']:.1f} L/min)"))
        items.append((ts, f"the recorded event: {Q.safe_text(e[3])[:60]}")); items.sort()
        trig = [it for it in items if it[1].startswith("battery power")]; states = [it for it in items if it[1].startswith("coolant flow")]; temps = [it for it in items if it[1].startswith("pack temperature")]
        clean = lambda t: re.sub(r" [(].*?[)]", "", t)
        if states and temps and states[0][0] > temps[0][0] and states[0][0] > ts - 6 * 3600:               # the flow drop came after the heating began: timing cannot make it the root cause
            lag = Q.dur(states[0][0] - temps[0][0])
            self.brief = f"Timing alone can't settle it: the load rose first, the pack began heating, and the coolant flow fell {lag} later - the flow drop may be a cause or a consequence."
            head = f"The load (charging or driving power) rose first; the pack temperature then began to climb and the coolant flow fell {lag} after that, so the data cannot tell whether the flow drop caused the heating or followed it. Order is not proof."
            out = [head] + [f"  - {self.fts(t)}: {w}" for t, w in items]
            out.append("What would prove it: a controlled test (restore the coolant flow / pump and see whether the same load still overheats it), a second independent sensor, or part-level diagnostics. None of these is in the data.")
            return "\n".join(out)
        if states:
            root = states[0]
            self.brief = (f"The likely root cause is that {clean(root[1])}, {Q.dur(ts - root[0])} before the event; " + (f"the load (charging or driving power) was the trigger and the pack temperature climb the symptom." if trig else "the later signals look like symptoms."))
            head = f"The likely root cause is that {clean(root[1])}, {Q.dur(ts - root[0])} before the event" + ("; the rise in load was only the trigger and the temperature climb the symptom" if trig else "; the later signals look like symptoms") + ". The data can show order, not proof."
        else:
            root = items[0]; self.brief = f"The first change was {clean(root[1])}, {Q.dur(ts - root[0])} before the event; that is a trigger rather than a proven root cause."
            head = f"The first change was {clean(root[1])}, {Q.dur(ts - root[0])} before the event - order is not proof."
        out = [head]
        out += [f"  - {self.fts(t)}: {w}" for t, w in items]
        out.append("What would prove it: a controlled test (restore the coolant flow / pump and see whether the same load still overheats it), a second independent sensor, or part-level diagnostics. None of these is in the data.")
        return "\n".join(out)

    def h_direction(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        if sg.res != 1 or len(sg) < 2000: return "I need the 1 Hz data for that and there is not enough of it."
        sp, pw = sg.cols["speed"], [abs(v) for v in sg.cols["power"]]; ac = sg.accel(); best = None
        for lag in range(-8, 9):
            x = pw[max(0, -lag):len(pw) - max(0, lag)]; y = ac[max(0, lag):len(ac) - max(0, -lag) or None]; n = min(len(x), len(y))
            r = Q.pearson(x[:n], y[:n])
            if r is not None and (best is None or r > best[1]): best = (lag, r)
        lag, r = best
        verdict = (f"Power demand moves first: battery power leads the change in speed by about {lag} s (r = {r:+.2f})" if lag > 0 else f"Speed change moves first by about {-lag} s (r = {r:+.2f})" if lag < 0 else f"They move together (r = {r:+.2f})")
        self.brief = "I can't tell: motor current isn't logged and RPM is computed from speed, so neither can be an independent cause."
        return (f"{verdict}, but I can't answer it for motor current and RPM themselves: motor current is not logged (battery power is the closest signal) and RPM is not measured - it is computed from vehicle speed, so it cannot be an independent cause or effect.\n"
                f"The physical chain is: demanded torque -> motor current -> acceleration -> speed (and hence RPM). Lag analysis over {wtxt} at 1 Hz shows timing, and timing alone cannot prove cause.")

    def h_causal_graph(self, s, raw):
        e = self.target_event(s); D = self.days_table(); cv = self.cov()
        sg = self.sig(*self.recent(7)[:2]); edges = []
        if len(sg) > 2000 and sg.res == 1:
            bt, am, pw, fl = sg.cols["batt"], sg.cols["amb"], sg.cols["power"], sg.cols["flow"]; ex = [b - a for b, a in zip(bt, am)]
            edges.append(("outside temperature", "pack temperature", Q.pearson(am[::10], bt[::10]), "physical: the pack exchanges heat with the air"))
            ch = [(abs(p), (bt[i + 60] - bt[i]) if i + 60 < len(bt) else 0.0) for i, p in enumerate(pw) if i % 60 == 0 and i + 60 < len(bt)]
            edges.append(("battery power (load / charging)", "pack temperature rise", Q.pearson([a for a, _ in ch], [b for _, b in ch]), "physical: resistive heating grows with current squared"))
            ru = [(f, ex[i]) for i, f in enumerate(fl) if f > 0.3 and i % 10 == 0]
            edges.append(("coolant flow", "pack temperature above outside", Q.pearson([a for a, _ in ru], [b for _, b in ru]), "physical: less flow, less heat removed"))
        thr = thr_of(e[3]) if e else None
        out = [f"Causal graph for {'the ' + self.fts(e[0]) + ' ' + Q.KIND_NAME.get('overheat', 'event') if e else 'the last event'}: outside temperature and battery power (charging or driving load) feed pack temperature, coolant flow removes heat from it, and pack temperature crossing {thr:.0f} C" + " raises the warning" if thr else "Causal graph:" + " pack temperature crossing its limit raises the warning."]
        out.append("  outside temperature --> pack temperature <-- battery power (heat in)")
        out.append("  coolant flow --| removes heat from --> pack temperature --> over-temperature warning")
        out.append("  coolant-pump / circuit health --> coolant flow --> diagnostic code P1A00 (flow low)")
        for a, b, r, why in edges:
            if r is not None: out.append(f"  evidence: {a} -> {b}: r = {r:+.2f} in the last 7 days ({why})")
        out.append("The arrows come from engineering knowledge; the correlations are the car's own data supporting them. Direction cannot be proven from observation alone.")
        self.brief = f"Outside temperature and charging or driving load heat the pack, coolant flow removes the heat, and passing {thr:.0f} C raised the warning." if thr else "Outside temperature and load heat the pack, coolant flow removes the heat, and passing its limit raised the warning."
        return "\n".join(out)

    def h_intervention(self, s, raw):
        e = self.target_event(s)
        if not e: return "No fault or warning event is recorded to prevent."
        d = self.diag_event(e)
        if not d: return "There is not enough data around the last event."
        thr = thr_of(e[3], 40.0); sg = d["sg"]; peak = max(sg.cols["batt"]); amb = d["pre"]["amb"]["mean"]; fl = d["base"]["flow"]; ratio = (d["pre"]["flow"]["mean"] / fl["p50"]) if fl else None
        opts = []
        exc = peak - amb
        if ratio and ratio < 0.9: opts.append((amb + exc * max(ratio, 0.2), "restoring the coolant flow to normal (repair or replace the pump / circuit)", f"flow was {100 * ratio:.0f}% of normal"))
        opts.append((amb + exc * 0.49, "limiting charge / load power to 70%", "heat generation grows with power squared"))
        opts.append((amb + exc / 1.2, "20% better heat removal", "peak rise above ambient scales inversely with cooling"))
        opts.sort()
        best = opts[0]; self.brief = f"{best[1][0].upper() + best[1][1:]} would have helped most: peak about {best[0]:.0f} C instead of {peak:.0f} C against the {thr:.0f} C limit (a rough estimate)."
        out = [f"{best[1][0].upper() + best[1][1:]} gives the best estimate: the peak would have been about {best[0]:.0f} C instead of {peak:.0f} C against the {thr:.0f} C warning limit ({'below' if best[0] < thr else 'still above'} it), with moderate confidence - this is a first-order estimate, not a simulation."]
        out += [f"  - {t}: peak about {p:.0f} C ({w})" for p, t, w in opts]
        out.append(f"Method: peak rise above outside temperature ({exc:.1f} C) scaled by the change in cooling or heat input, assuming quasi-steady heat balance. It cannot be tested on this car because the world where it was done never happened.")
        return "\n".join(out)

    def h_attr3(self, s, raw):
        e = self.target_event(s) or self.most_serious()
        if not e: return "No anomaly is recorded to attribute."
        d = self.diag_event(e)
        if not d: return "There is not enough data around the last anomaly."
        ts = e[0]; hb = self.c.execute("SELECT COUNT(*) FROM events WHERE type IN ('HARSH_ACCEL','HARSH_BRAKE','HARSH_CORNER') AND ts BETWEEN ? AND ?", (ts - 1800, ts)).fetchone()[0]
        fl, ba = d["base"]["flow"], d["base"]["amb"]; ratio = d["pre"]["flow"]["mean"] / fl["p50"] if fl else None; z = (d["pre"]["amb"]["mean"] - ba["mean"]) / ba["sd"] if ba else None
        st = Q.state_at(self.c, ts)
        mech = "strong" if ratio is not None and ratio < 0.7 else "weak"; env = "strong" if z is not None and z > 2 else "weak"; drv = "strong" if hb >= 5 else "weak"
        lead = "the mechanical fault (cooling)" if mech == "strong" and env == "weak" else "the environment" if env == "strong" and mech == "weak" else "the driver" if drv == "strong" and mech == "weak" and env == "weak" else "a combination"
        self.brief = f"Most likely {lead}; the driver and the weather look minor." if lead.startswith("the mech") else f"The data points to {lead}, but it can't fully separate driver, weather and fault."
        out = [f"On the available data it points to {lead}; the driver and the environment are {'weak' if lead.startswith('the mech') else 'not ruled out'}, but only partly separable from one vehicle's signals.",
               f"  - mechanical: coolant flow {100 * ratio:.0f}% of normal ({mech})" if ratio is not None else "  - mechanical: no flow baseline",
               f"  - environment: outside {d['pre']['amb']['mean']:.1f} C, z = {z:+.1f} against this car's normal ({env})" if z is not None else "  - environment: no baseline",
               f"  - driver: {hb} hard acceleration/braking/cornering events in the 30 minutes before" + (f"; the car was charging ({st['charging']['kind']}, {st['charging']['kw']:.0f} kW) - a driver choice, but a normal one" if st.get("charging") else "") + f" ({drv})",
               "A fault that only shows under load (here: fast charging) is mechanical but needs the driver's action to appear - the data cannot separate 'cause' from 'trigger'."]
        return "\n".join(out)

    def h_standstill(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        if len(sg) < 1000: return "There is not enough signal data to say."
        sp, bt, md = sg.cols["speed"], sg.cols["batt"], sg.cols["mode"]; k = max(1, int(300 / sg.res)); n_all = n_still = 0; rises_still = rises_move = []; rs, rm = [], []
        for i in range(0, len(sp) - k, k):
            if sg.ts[i + k] - sg.ts[i] > 300 + 2 * sg.res: continue
            d = bt[i + k] - bt[i]
            if d > 1.0:
                n_all += 1
                if max(sp[i:i + k]) < 2: n_still += 1; rs.append(d)
                else: rm.append(d)
        chg = sum(1 for i in range(0, len(sp) - k, k) if max(sp[i:i + k]) < 2 and 3 in md[i:i + k])
        if n_all == 0: return f"In {wtxt} the pack temperature never rose by more than 1 C within five minutes."
        yes = n_still > 0; self.brief = f"{'Yes' if yes else 'No'}: {n_still} of {n_all} temperature rises happened while the car was standing still" + (", mostly while charging." if yes and chg else ".")
        return (f"{'Yes' if yes else 'No'} - {n_still} of the {n_all} five-minute rises of more than 1 C in {wtxt} happened while the car was standing still" + (f" (largest {max(rs):.1f} C), almost all of them while charging." if yes and chg else ".") +
                f"\nWhile moving: {len(rm)} such rises" + (f" (largest {max(rm):.1f} C)." if rm else ".") + " Fast charging heats the pack with the car stationary.")

    # ------------------------------------------------------------------ counterfactuals
    CF_NOTE = "A counterfactual is a model estimate, not something the car recorded."

    def h_cf_speed(self, s, raw):
        m = re.search(r"(\d+) ?(?:km|kph)", s); dv = float(m[1]) if m else 20.0
        r = self.c.execute("SELECT start, end, km, kwh FROM trips WHERE km>2 AND start<=? ORDER BY start DESC LIMIT 1", (self.now,)).fetchone()
        if not r: return "There is no trip to re-drive."
        sg = Q.load(self.c, r[0], r[1] + 1); mv = [v for v in sg.cols["speed"] if v > 20] if len(sg) else []
        if not mv: return "The last trip has no cruising segment to adjust."
        v0 = mean(mv); v1 = max(v0 - dv, 20.0); w0, rr = self.whkm_at(v0); w1, _ = self.whkm_at(v1)
        if w0 is None or w1 is None: return "The car has too little steady cruising in its history to estimate speed-dependent consumption."
        cap = self.cap_kwh() * 1000; saved = (w0 - w1) * r[2] / 1000.0; self.brief = f"About {cap / w1 - cap / w0:.0f} km more range per full charge ({saved:.1f} kWh saved on that trip) - an estimate, not an exact figure."
        return (f"Roughly {saved:.1f} kWh less energy over the {r[2]:.0f} km trip, i.e. about {cap / w1 - cap / w0:.0f} km more range per full charge - an estimate, not an exact figure.\n"
                f"Basis: this car's own steady-cruising consumption ({rr['basis']}): {w0:.0f} Wh/km near {v0:.0f} km/h vs {w1:.0f} Wh/km near {v1:.0f} km/h. Ignored: the trip would take longer, traffic, stops and the climate load rise with time. Accurate to maybe +/-30%.")

    def h_cf_temp(self, s, raw):
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?) ?°?c", s)] or [float(x) for x in re.findall(r"(\d+)", s)[:2]]
        if len(nums) < 2: return "Tell me the two temperatures to compare, for example 'at 15 C instead of 30 C'."
        t_new, t_old = nums[0], nums[1]; cv = self.cov(); fit = Q.eff_fit(self.c, cv["lo"], cv["hi"] + 1)
        if not fit: return "There are too few trips to fit how temperature changes this car's consumption."
        bh = next((b for f, b in zip(fit["feats"], fit["beta"]) if f[0] == "heating"), 0.0); bc = next((b for f, b in zip(fit["feats"], fit["beta"]) if f[0] == "cooling"), 0.0)
        self.brief = None
        e = lambda t: bh * max(0.0, 18 - t) + bc * max(0.0, t - 24); delta = e(t_new) - e(t_old); base = mean([t["wh_km"] for t in fit["T"]]); ref = base - mean([e(t["amb"]) for t in fit["T"]])
        self.brief = f"About {abs(delta):.0f} Wh/km {'more' if delta > 0 else 'less'} at {t_new:g} C than at {t_old:g} C - a model estimate."
        return (f"About {delta:+.0f} Wh/km ({100 * delta / (ref + e(t_old)):+.0f}%): roughly {ref + e(t_new):.0f} Wh/km at {t_new:g} C against {ref + e(t_old):.0f} Wh/km at {t_old:g} C - a model estimate.\n"
                f"Basis: the trip regression on {len(fit['T'])} of this car's trips (R2 {fit['r2']:.2f}): {bh:.1f} Wh/km per degree below 18 C (heating), {bc:.1f} Wh/km per degree above 24 C (cooling). Not modelled: battery temperature effects on efficiency and charging, wind, traffic. {self.CF_NOTE}")

    def h_cf_soc(self, s, raw):
        r = self.c.execute("SELECT soc_start, wh_km FROM trips WHERE km>3").fetchall()
        if len(r) < 20: return "There are too few trips to say how the starting charge affects this car."
        x, y = [a for a, _ in r], [b for _, b in r]; f = Q.slope_test(x, y); wh = mean(y); cap = self.cap_kwh()
        eff = f["slope"] * 30 if f else 0.0; little = abs(eff) < 0.1 * wh; self.brief = f"It would mostly mean more energy, not different efficiency: about {0.3 * cap * 1000 / wh:.0f} km more range."
        return (f"Starting at 90% instead of 60% would mostly mean {0.3 * cap:.1f} kWh more energy, about {0.3 * cap * 1000 / wh:.0f} km more range at your {wh:.0f} Wh/km; consumption itself would change {'very little' if little else 'somewhat'} ({eff:+.0f} Wh/km for 30 points, {pfmt(f['p'])}, confounded by when in the day trips start)." if f
                else "I cannot estimate a starting-charge effect.") + f"\nA full battery can allow more regeneration but a warmer pack and higher cell voltage change charging limits - not modelled. {self.CF_NOTE}"

    def h_cf_cooling(self, s, raw):
        m = re.search(r"(\d+) ?%", s); pct = float(m[1]) if m else 20.0
        e = self.target_event(s)
        if not e: return "No over-temperature warning is recorded, so there is no warning to avoid."
        d = self.diag_event(e)
        if not d: return "There is not enough data around the warning."
        thr = thr_of(e[3], 40.0); peak = max(d["sg"].cols["batt"]); amb = d["pre"]["amb"]["mean"]; new = amb + (peak - amb) / (1 + pct / 100); fl = d["base"]["flow"]
        ratio = d["pre"]["flow"]["mean"] / fl["p50"] if fl else None
        self.brief = f"{'Probably not avoided' if new >= thr else 'Probably avoided'}: with {pct:.0f}% better cooling the peak would have been about {new:.0f} C against the {thr:.0f} C limit."
        out = [f"{'Probably not avoided' if new >= thr else 'Probably avoided'}: with {pct:.0f}% better cooling the peak would have been about {new:.1f} C instead of {peak:.1f} C, against the {thr:.0f} C warning limit - first-order estimate."]
        if ratio and ratio < 0.8: out.append(f"The cooling was actually down to {100 * ratio:.0f}% of normal; restoring it fully would have brought the peak to about {amb + (peak - amb) * ratio:.1f} C.")
        out.append(f"Method: the peak rise above outside temperature ({peak - amb:.1f} C) divided by the cooling improvement, same heat input, quasi-steady heat balance. {self.CF_NOTE}")
        return "\n".join(out)

    def h_cf_adas(self, s, raw):
        r = self.evs("adas", newest_first=True, limit=1)
        if not r: return "No ADAS disengagement is recorded, so there is nothing to re-run."
        e = r[0]; st = Q.state_at(self.c, e[0]); reason = Q.safe_text(e[3])
        speedy = re.search(r"speed|fast", reason, re.I)
        return (f"Probably yes: the last disengagement ({self.fts(e[0])}, at {st.get('speed', float('nan')):.0f} km/h) was recorded as '{reason}', a reason that does not depend on road speed" + (", so 10 km/h less would not have changed it." if not speedy else ".") +
                " The ADAS thresholds themselves are not exposed to this assistant, so I can't test the speed-dependent case. " + self.CF_NOTE)

    def h_cf_accel(self, s, raw):
        e = self.target_event(s)
        if not e: return "No overheating event is recorded, so there is nothing to compare."
        ts = e[0]; sg = Q.event_sig(self.c, ts, 600, 0)
        if len(sg) < 60: return "There is no 1 Hz data before the event."
        sp, pw, bt = sg.cols["speed"], sg.cols["power"], sg.cols["batt"]; ac = sg.accel(); hot = [i for i, a in enumerate(ac) if a > 2.0]
        rise = bt[-1] - bt[0]
        if not hot:
            st = Q.state_at(self.c, ts - 5)
            self.brief = "There was no aggressive acceleration before the event, so the temperature would have been the same."
            return (f"There was no aggressive acceleration to remove: in the 10 minutes before the event the car never accelerated harder than 2 m/s2 ({'it was charging at ' + format(abs(st['power']), '.0f') + ' kW' if st.get('mode') == 3 else 'speed ' + format(max(sp), '.0f') + ' km/h max'}), so the counterfactual temperature is the actual one: {bt[-1]:.1f} C.\n"
                    f"The pack rose {rise:+.1f} C in those 10 minutes from other causes (see 'why did it overheat'). {self.CF_NOTE}")
        tot = sum(p * p for p in pw); acc = sum(pw[i] ** 2 for i in hot); share = acc / tot if tot > 0 else 0.0; self.brief = f"About {bt[-1] - share * rise:.0f} C instead of {bt[-1]:.0f} C - aggressive acceleration caused roughly {100 * share:.0f}% of the heating."
        return (f"About {bt[-1] - share * rise:.1f} C instead of {bt[-1]:.1f} C: aggressive acceleration ({len(hot)} s above 2 m/s2) accounts for roughly {100 * share:.0f}% of the resistive heating (power squared) in the 10 minutes before, and the pack rose {rise:+.1f} C then - an estimate.\n"
                f"Assumes heating proportional to power squared and the same cooling. {self.CF_NOTE}")

    def h_cf_generic(self, s, raw):
        n = self.CF_NOTE
        if re.search(r"flat road|gradient|hill|slope", s): return f"I can't say - road gradient is not recorded (GPS has latitude and longitude only, no altitude), so what a flat road would have changed is unknowable from this data. {n}"
        if re.search(r"braked \d+ seconds? earlier|exact sensor trajectory|5 seconds earlier", s): return f"I can't give that - an exact trajectory needs a validated vehicle-and-driver model, and any curve I drew would be invented. What the data does show is what actually happened around the event. {n}"
        if re.search(r"next \d+ minutes|continued driving|would the next", s): return f"I can't generate that - I don't forecast telemetry; any 30-minute trace would be made up. I can show what the last 30 minutes looked like, or typical values for this car. {n}"
        if re.search(r"half the acceleration|half .{0,20}acceleration|simulate the same trip", s):
            cv = self.cov(); fit = Q.eff_fit(self.c, cv["lo"], cv["hi"] + 1); b = next((bb for f, bb in zip(fit["feats"], fit["beta"]) if f[0] == "harsh"), None) if fit else None
            hb = mean([t["harsh"] for t in fit["T"]]) if fit else None
            return ((f"I can't simulate a whole trip, but for energy: halving the hard events ({hb:.0f} to {hb / 2:.0f} per 100 km) would change consumption by about {-b * hb / 2:.1f} Wh/km, which is small. Every other metric (temperatures, flow, charge) I can't re-simulate. {n}") if b is not None and hb else f"I can't simulate a trip with different driving. {n}")
        if re.search(r"degradation curve|charging history", s): return f"I can't - there is one vehicle and one charging history in the data, and no model that links charging habits to battery ageing, so any alternative curve would be invented. What I can show is the measured trend of state of health. {n}"
        if re.search(r"prove .{0,40}would not", s): return f"No one can prove that - a statement about a world that did not happen can't be tested with data. The most I can do is estimate how much a different driving style would have changed the signals (see 'what would have happened if ...'). {n}"
        return f"I can only answer that as a model estimate, and the data does not contain the world where things were different. Tell me the specific change (speed, temperature, charge level, cooling) and I'll give an estimate with its assumptions. {n}"

    # ------------------------------------------------------------------ variables the car does not record
    def h_hidden(self, s, raw):
        c = self.c
        if re.search(r"cargo|payload|passenger|occupan", s): return "That is not recorded: the car has no load-weight or seat sensors in the data, so I can't say how much cargo or how many people there were. Payload is one of the unmeasured things that could be behind an efficiency drop."
        if re.search(r"gradient|slope|elevation|altitude", s): return "That is not recorded: the GPS fixes carry latitude, longitude, heading and speed but no altitude, and there is no inclination sensor, so I can't give the road gradient at any point."
        if re.search(r"headwind|wind", s): return "That is not recorded: the car has no wind sensor and no weather feed offline, so I can't say how much headwind there was."
        if re.search(r"chemistry|pack configuration|cell count|series|parallel", s):
            return f"Not recorded: the data gives the usable capacity ({self.cap_kwh():.1f} kWh), state of health and pack voltage/current signals (in the CAN database), but not the cell chemistry or the series/parallel layout."
        if re.search(r"individual|which cell|cell caused", s): return "I can't tell: only the pack-level temperature (and the CAN database's cell max/min signals) exist, and the per-cell values are not stored, so no individual cell can be blamed."
        if re.search(r"brake pads?", s):
            e = self.target_event(s) or self.most_serious(); ts = e[0] if e else self.now
            r = c.execute("SELECT pad FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (ts,)).fetchone() if self._has("vday") else None
            return (f"About {r[0]:.0f}% of the pad was left on the day of the anomaly (an estimate from braking energy, not a measurement)." if r else "Brake-pad condition is not recorded for this vehicle.")
        if re.search(r"tyre pressure", s):
            if not self._has("ext_1m"): return "This vehicle has no tyre-pressure data recorded."
            sg = self.sig(*self.recent(7)[:2]); runs_ = []
            if len(sg) > 100:
                idx = [i for i, m in enumerate(sg.cols["mode"]) if m == 2]
                if idx: ts = sg.ts[idx[-1]]
            else: ts = self.cov()["hi"]
            st = Q.state_at(self.c, ts).get("ext")
            if not st: return "No tyre-pressure reading exists near the last motorway segment."
            return f"During the last motorway segment (around {self.fts(ts)}): front left {st['tyres'][0]:.0f}, front right {st['tyres'][1]:.0f}, rear left {st['tyres'][2]:.0f}, rear right {st['tyres'][3]:.0f} kPa (corrected to 20 C; raw {', '.join(f'{x:.0f}' for x in st['raw_tyres'])} kPa)."
        if re.search(r"air conditioner|a/?c\b|climate|heater|hvac", s):
            d = self.diag_range() if not self.win(s, default=None)[0] else None
            lo, hi, w = (d["pb"][0], d["pb"][1], f"the period after the change ({self.day(d['pb'][0])} to {self.day(d['pb'][1] - 1)})") if d else (self.win(s, default=None) if self.win(s, default=None)[0] is not None else self.recent(7))
            r = c.execute("SELECT AVG(hvac), MAX(hvac), AVG(amb)/10.0 FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2)", (lo, hi)).fetchone()
            if not r or r[0] is None: return "No climate-control data exists for that period."
            kind = "heating" if r[2] < 16 else "air conditioning" if r[2] > 24 else "mild weather (little climate load)"
            lead = "Yes, noticeably" if r[0] > 1500 else "Only lightly" if r[0] > 400 else "Barely"
            return f"{lead}: climate control drew {r[0]:.0f} W on average (up to {r[1]:.0f} W) in {w}; with {r[2]:.1f} C outside that is {kind}. The data shows power draw, not whether it was the heater or the A/C compressor."
        if re.search(r"gps location|outside temperature at", s):
            e = self.target_event(s) or self.most_serious(); ts = e[0] if e else self.cov()["hi"]; st = Q.state_at(c, ts)
            if "amb" not in st: return "There is no outside-temperature sample at that moment (the car was off)."
            g = st.get("gps")
            return f"{st['amb']:.1f} C - measured by the car's own outside-temperature sensor at {self.fts(ts)}" + (f", at {g['lat']:.4f}, {g['lon']:.4f}" if g else "") + ". An offline assistant has no weather service for the exact spot; this sensor sits on the car, so it is the temperature where the car was."
        return "That is not recorded in this vehicle's data, and I won't estimate it."

    # ------------------------------------------------------------------ fleet / cross-entity questions
    def h_fleet(self, s, raw):
        self.brief = "I can't compare with other vehicles - this assistant only holds this car's own data."
        return ("I can't - this assistant holds only this one vehicle's data (it runs offline inside the car); there are no other vehicles, no vehicle identifiers to align, and no fleet database, so fleet comparison, percentile rank and nearest-neighbour matching have nothing to work on.\n"
                "What I can do: compare this vehicle with its own earlier weeks (try 'compare the first and last week'), or say whether its numbers look unusual for itself. A fleet answer needs an off-board fleet service that receives anonymised summaries.")

    # ------------------------------------------------------------------ long-horizon
    def series_set(self):
        cv = self.cov(); lo, hi = cv["lo"], cv["hi"] + 1; out = {}
        out["energy use per km"] = [(d["ts"], d["wh_km"]) for d in Q.daily_series(self.c, lo, hi)]
        out["coolant flow"] = [(d, v) for d, v, _ in self.daily("flow", lo, hi, modes=(1, 2, 3), minv=3.0)]
        out["pack temperature"] = [(d, v) for d, v, _ in self.daily("batt", lo, hi, modes=(1, 2, 3))]
        out["motor temperature"] = [(d, v) for d, v, _ in self.daily("motor", lo, hi, modes=(1, 2))]
        out["average speed"] = [(d["ts"], d["speed"]) for d in Q.daily_series(self.c, lo, hi)]
        if self._has("vday"):
            vd = self.c.execute("SELECT ts, soh, v12 FROM vday WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall()
            out["battery state of health"] = [(t, x) for t, x, _ in vd if x]; out["12 V rest voltage"] = [(t, v) for t, _, v in vd if v]
        return out

    def h_earliest_change(self, s, raw):
        found = []
        for name, ser in self.series_set().items():
            if len(ser) < 8: continue
            best = None
            for k in range(3, len(ser) - 2):
                w = Q.welch([v for _, v in ser[:k]], [v for _, v in ser[k:]])
                if w and (best is None or abs(w[0]) > abs(best[1][0])): best = (k, w)
            if not best: continue
            k, w = best; a, b = mean([v for _, v in ser[:k]]), mean([v for _, v in ser[k:]])
            stay = all(abs(v - b) < abs(v - a) for _, v in ser[k + 1:]) if len(ser) - k > 3 else True            # the level stays on the new side
            if abs(w[0]) >= 4 and abs(b - a) > 0.04 * abs(a or 1) and w[2] < 0.01: found.append((ser[k][0], name, a, b, w[2], stay))
        flow = Q.flow_onset(self.c, self.now)
        self.brief = None
        if not found and not flow: return "No signal shows a statistically significant, lasting change in this history (daily means, Welch test at p < 0.01 and a >4% shift)."
        found.sort(); f = found[0] if found else None
        if flow and (not f or flow[0] <= f[0]): head = f"Behaviour first left its normal band on {self.day(flow[0])}: the daily median coolant flow dropped to {flow[1]:.1f} L/min (normal {flow[2]:.1f}) and never recovered - even though no diagnostic warning had been set yet."
        else: head = f"The earliest statistically significant change was in {f[1]} on {self.day(f[0])}: {f[2]:.1f} -> {f[3]:.1f} ({pfmt(f[4])}), {'and it persisted' if f[5] else 'but it did not persist'}."
        self.brief = re.sub(r"\s*[-\u2014] even though.*$", ".", head)
        out = [head] + [f"  - {n}: change point {self.day(t)}, {a:.1f} -> {b:.1f} ({pfmt(p)}){'' if st else ', not permanent'}" for t, n, a, b, p, st in found[:5]]
        out.append("Method: for each daily series the split with the largest Welch t-statistic (>= 4, p < 0.01, shift > 4%); the earliest date wins. A change point is where the level changed, not necessarily when the cause started.")
        return "\n".join(out)

    def h_soh_rate(self, s, raw):
        ser = self.series_set().get("battery state of health")
        if not ser or len(ser) < 8: return "This vehicle records no battery state-of-health history long enough to estimate a degradation rate (it needs at least 8 days)."
        x = [t / DAY for t, _ in ser]; y = [v for _, v in ser]; n = len(x); x0 = x[0]; xs = [v - x0 for v in x]
        lin = Q.ols_line(xs, y); st = Q.slope_test(xs, y); span = xs[-1]
        rss_l = sum((v - (lin["intercept"] + lin["slope"] * a)) ** 2 for a, v in zip(xs, y))
        ly = [math.log(max(v, 1e-6)) for v in y]; ex = Q.ols_line(xs, ly); rss_e = sum((v - math.exp(ex["intercept"] + ex["slope"] * a)) ** 2 for a, v in zip(xs, y)) if ex else float("inf")
        mean_y = mean(y); rss_c = sum((v - mean_y) ** 2 for v in y)
        aics = {"linear": Q.aic(rss_l, n, 3), "exponential": Q.aic(rss_e, n, 3), "flat (no change)": Q.aic(rss_c, n, 2)}; best = min(aics, key=aics.get)
        ci = Q.bootstrap_slope(xs, y); per_year = lin["slope"] * 365
        head = (f"State of health is falling by about {-lin['slope'] * 30:.2f} points a month ({-per_year:.1f} per year, {pfmt(st['p'])}); best-fitting shape over {span:.0f} days: {best}." if st and st["p"] < 0.05 and lin["slope"] < 0 else
                f"No significant change in state of health over {span:.0f} days: the fitted rate is {lin['slope'] * 30:+.2f} points a month ({pfmt(st['p']) if st else ''}); best-fitting shape: {best}.")
        self.brief = (f"Battery health is falling by about {-lin['slope'] * 30:.2f} points a month." if st and st["p"] < 0.05 and lin["slope"] < 0 else f"No measurable battery ageing yet ({lin['slope'] * 30:+.2f} points a month) - the history is too short to say more.")
        out = [head]
        if ci: out.append(f"  - 95% bootstrap interval for the rate: {-ci[1] * 30:.2f} to {-ci[0] * 30:.2f} points/month (2000 bootstrap resamples of the {n} days).")
        out.append("  - model comparison (AIC, lower is better): " + ", ".join(f"{k} {v:.1f}" for k, v in aics.items()))
        out.append(f"  - no remaining-life date is projected from {span:.0f} days: battery ageing is non-linear and depends on calendar age, temperature and charging history that are not fully recorded here.")
        out.append("  - normal vs abnormal ageing: calendar and cycle ageing of a few percent per year is typical for EV packs (general knowledge, not from this data); a rate far above that, or a sudden step, would be abnormal." if per_year < -0.2 else "  - this rate is within what normal ageing looks like (typical packs lose a few percent per year; general knowledge, not from this data).")
        return "\n".join(out)

    def h_periodic(self, s, raw):
        rows = []
        for name, ser in self.series_set().items():
            y = [v for _, v in ser]
            if len(y) < 14: continue
            ln = Q.ols_line(list(range(len(y))), y); y = [v - (ln["intercept"] + ln["slope"] * i) for i, v in enumerate(y)] if ln else y                  # remove the trend first: a trend correlates with itself at every lag
            for lag in range(2, min(10, len(y) // 2)):
                r = Q.autocorr(y, lag)
                if r is not None and r > 2 / math.sqrt(len(y)) and r > 0.4: rows.append((r, name, lag))
        cv = self.cov(); ch = [d for d in self.c.execute("SELECT start FROM charges").fetchall()] if self._has("charges") else []
        rows.sort(reverse=True)
        if not rows: return "No recurring pattern stands out: the day-to-day autocorrelation of every daily series is within what chance would produce."
        r, name, lag = rows[0]; self.brief = f"The clearest repeating pattern is in {name}: it recurs every {lag} days."
        return f"The clearest repeating pattern is in {name}: it resembles itself every {lag} days (autocorrelation {r:+.2f}, above the +/-{2 / math.sqrt(len(self.series_set()[name])):.2f} chance band)." + "\n" + "\n".join(f"  - {n}: every {l} days, r = {rr:+.2f}" for rr, n, l in rows[1:4]) + "\nWith under a month of history a cycle of a week or longer can be seen only once or twice, so treat it as a hint."

    def h_worsening(self, s, raw):
        fo = Q.flow_onset(self.c, self.now)
        if not fo: return "No persistent abnormal behaviour (such as a coolant-flow drop) is recorded, so there is no anomaly to track for worsening."
        days = [d for d in Q.flow_days(self.c, fo[0], self.cov()["hi"] + 1)]
        if len(days) < 5: return f"The anomaly began on {self.day(fo[0])}, but only {len(days)} days follow it - too few to judge a trend."
        f = Q.slope_test([d[0] / DAY for d in days], [d[1] for d in days]); b = Q.bootstrap_slope([d[0] / DAY for d in days], [d[1] for d in days])
        word = "is still getting worse" if f["slope"] < -0.05 and f["p"] < 0.05 else "has levelled off (no further decline)" if abs(f["slope"]) < 0.05 or f["p"] >= 0.05 else "is recovering"
        self.brief = f"The coolant-flow problem (since {self.day(fo[0])}) {word}."
        return (f"The coolant-flow anomaly (since {self.day(fo[0])}) {word}: the daily median is {days[0][1]:.1f} -> {days[-1][1]:.1f} L/min over {len(days)} days, slope {f['slope']:+.2f} L/min per day ({pfmt(f['p'])}" + (f", 95% bootstrap {b[0]:+.2f} to {b[1]:+.2f}" if b else "") + ").\n"
                "Other anomalies in the history are single events (one diagnostic code each), which cannot 'worsen' on their own.")

    # ------------------------------------------------------------------ statistics
    def two_periods(self, s):
        """The two periods compared in a statistics question: the first vs the most recent week, unless the question names others."""
        cv = self.cov(); hi = cv["hi"] + 1
        if re.search(r"this week", s): lo, h2 = A.window(self.now, "tw", self.c); return (lo - 7 * DAY, lo), (lo, h2), "last week", "this week"
        n = min(7, (hi - cv["lo"]) // DAY // 2)
        return (cv["lo"], cv["lo"] + n * DAY), (hi - n * DAY, hi), f"the first {n} days", f"the last {n} days"

    def h_stat_ci(self, s, raw):
        p1, p2, n1, n2 = self.two_periods(s)
        a = [d["wh_km"] for d in Q.daily_series(self.c, *p1)]; b = [d["wh_km"] for d in Q.daily_series(self.c, *p2)]
        r = Q.welch_ci(a, b)
        if not r: return "There are too few trip days in the two periods to run a test."
        mw = Q.mann_whitney_p(a, b); sig = r["p"] < 0.05; self.brief = f"Energy use changed by {r['diff']:+.1f} Wh/km from {n1} to {n2}; that is {'statistically significant' if sig else 'not statistically significant'}."
        return (f"The change in average energy consumption from {n1} to {n2} is {r['diff']:+.1f} Wh/km (95% CI {r['lo']:+.1f} to {r['hi']:+.1f}); it is {'statistically significant' if sig else 'not statistically significant'} ({pfmt(r['p'])}), effect size d = {r['d']:+.2f}.\n"
                f"Test: Welch's two-sample t-test on daily Wh/km (t = {r['t']:.2f}, df = {r['df']:.1f}, n = {r['n'][0]} and {r['n'][1]} days). Assumptions: days are independent (weakly true - weather persists), roughly normal daily means, unequal variances allowed. Robustness: Mann-Whitney {pfmt(mw) if mw is not None else 'n/a'}. "
                f"Effect size: Cohen's d = {r['d']:+.2f} ({'small' if abs(r['d']) < 0.5 else 'medium' if abs(r['d']) < 0.8 else 'large'}); means {mean(a):.1f} vs {mean(b):.1f} Wh/km.")

    def h_corr2(self, s, raw):
        D = self.days_table()
        if len(D) < 8: return "There are too few days to correlate."
        bt, en, sp, am = [d["batt"] for d in D], [d["wh_km"] for d in D], [d["speed"] for d in D], [d["amb"] for d in D]; n = len(D)
        pr, sr = Q.pearson(bt, en), Q.spearman(bt, en); pc = Q.partial_corr(bt, en, [sp]); pc2 = Q.partial_corr(bt, en, [sp, am])
        keep = "does not remain" if abs(pc) < 0.3 and abs(pr) >= 0.3 else "remains"; self.brief = f"Pack temperature and energy use correlate at {pr:+.2f} (Spearman {sr:+.2f}); the link {keep} once speed is held fixed."
        return (f"Pearson r = {pr:+.2f} ({pfmt(Q.corr_p(pr, n))}) and Spearman rho = {sr:+.2f} ({pfmt(Q.corr_p(sr, n))}) between daily pack temperature and daily energy use over {n} days; controlling for speed the correlation {keep} (partial r = {pc:+.2f}); controlling for speed and outside temperature it is {pc2:+.2f}.\n"
                "Spearman uses ranks, so it is robust to outliers and to non-linear but monotonic links; partial correlation removes the part explained by the controls. Days are the unit, so n is small and p-values are indicative.")

    def h_hwy_test(self, s, raw):
        cv = self.cov(); T = Q.trips_between(self.c, cv["lo"], cv["hi"] + 1); T = [t for t in T if t["km"] >= 3]
        h = [t["wh_km"] for t in T if t["hwy"] > 0.6]; c = [t["wh_km"] for t in T if t["hwy"] < 0.2]
        if len(h) < 4 or len(c) < 4:                                                                                    # few motorway trips: compare driving modes within each day instead (paired)
            rows = self.c.execute(f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, mode, SUM(power)/60.0, SUM(speed)/600.0, COUNT(*) FROM sig_1m WHERE mode IN (1,2) AND speed>0 GROUP BY d, mode").fetchall(); by = {}
            for d, m, wh, km, n in rows:
                if n >= 15 and km > 3 and wh > 0: by.setdefault(d, {})[m] = wh / km
            pairs = [(v[1], v[2]) for v in by.values() if 1 in v and 2 in v]
            if len(pairs) < 4: return f"There is too little motorway driving to test it: {len(h)} mostly-motorway trips and fewer than 4 days with both city and motorway driving."
            dif = [b - a for a, b in pairs]; m_ = mean(dif); hh = Q.tcrit(len(dif) - 1) * sd(dif) / math.sqrt(len(dif)); t_ = m_ / (sd(dif) / math.sqrt(len(dif))) if sd(dif) else 0.0; p_ = Q.betainc((len(dif) - 1) / 2, 0.5, (len(dif) - 1) / ((len(dif) - 1) + t_ * t_))
            return (f"{'Yes' if m_ > 0 and p_ < 0.05 else 'No clear difference'}: on the {len(pairs)} days with both, motorway minutes used {mean([b for _, b in pairs]):.0f} Wh/km against {mean([a for a, _ in pairs]):.0f} in the city, a paired difference of {m_:+.0f} Wh/km (95% CI {m_ - hh:+.0f} to {m_ + hh:+.0f}, {pfmt(p_)}).\n"
                    f"Test: paired t-test across days (each day is its own control for weather); motorway = above 72 km/h minutes. Assumptions: differences roughly normal, days independent. Too few mostly-motorway trips ({len(h)}) for the trip-level test.")
        r = Q.welch_ci(c, h); mw = Q.mann_whitney_p(c, h); self.brief = f"{'Yes' if r['diff'] > 0 and r['p'] < 0.05 else 'No clear difference'}: motorway trips used {mean(h):.0f} Wh/km against {mean(c):.0f} in the city."
        return (f"{'Yes' if r['diff'] > 0 and r['p'] < 0.05 else 'No clear difference'}: mostly-motorway trips used {mean(h):.0f} Wh/km against {mean(c):.0f} for mostly-city trips, a difference of {r['diff']:+.0f} Wh/km (95% CI {r['lo']:+.0f} to {r['hi']:+.0f}, {pfmt(r['p'])}, effect size d = {r['d']:+.2f}).\n"
                f"Test: one-sided question 'motorway > city' answered with Welch's t-test on per-trip Wh/km (n = {len(h)} motorway, {len(c)} city; motorway = over 60% of the distance, city = under 20%); Mann-Whitney {pfmt(mw) if mw is not None else 'n/a'} as a check. Assumptions: trips independent, unequal variances. Weather and trip length differ between groups, so see the regression for the adjusted effect.")

    def h_trend_unc(self, s, raw):
        ser = self.series_set().get("pack temperature")
        if not ser or len(ser) < 8: return "There are too few days to estimate a temperature trend."
        x = [t / DAY for t, _ in ser]; y = [v for _, v in ser]; f = Q.slope_test(x, y); lin = Q.ols_line(x, y); b = Q.bootstrap_slope(x, y)
        se = abs(lin["slope"]) / math.sqrt(max((f["slope"] / 1 and 1), 1)) if False else None
        n = len(x); r = Q.ols_se([[v - x[0]] for v in x], y); h = Q.tcrit(r["df"]) * r["se"][1]; self.brief = f"Pack temperature is trending {lin['slope']:+.2f} C a day, " + ("but the uncertainty includes zero, so there is no reliable trend." if lin["slope"] - h < 0 < lin["slope"] + h else "and that trend is reliably different from zero.")
        return (f"The pack-temperature trend is {lin['slope']:+.3f} C per day (95% CI {lin['slope'] - h:+.3f} to {lin['slope'] + h:+.3f}; {pfmt(r['p'][1])}, R2 {r['r2']:.2f}, {n} days) - " + ("the interval includes zero, so there is no reliable trend." if lin["slope"] - h < 0 < lin["slope"] + h else "the interval excludes zero.") +
                (f"\nBootstrap check: {b[0]:+.3f} to {b[1]:+.3f} C/day." if b else "") + "\nDaily means follow the weather, so a trend here mostly measures the weather over the period.")

    def h_top_sig(self, s, raw):
        rows = []
        for name, ser in self.series_set().items():
            if len(ser) < 8: continue
            x = [t / DAY for t, _ in ser]; y = [v for _, v in ser]; f = Q.slope_test(x, y)
            k = max(3, len(y) // 5); a, b = mean(y[:k]), mean(y[-k:])
            if f and abs(b - a) >= 0.03 * abs(a or 1): rows.append((f["p"], name, f["slope"], a, b, f["r2"]))                                          # statistically AND practically significant (>= 3% change)
        ev = self.c.execute("SELECT type, COUNT(*) FROM events WHERE type IN (%s) GROUP BY type" % ",".join("?" * len(WARN_TYPES)), WARN_TYPES).fetchall()
        rows.sort()
        if not rows: return "There is not enough history to rank anomalies by significance."
        p, name, sl, a, b, r2 = rows[0]; self.brief = f"The most significant anomaly is the drift in {name}: {a:.1f} to {b:.1f}."
        return (f"The most statistically significant anomaly is the drift in {name} ({a:.1f} -> {b:.1f}, slope {sl:+.3f} per day, {pfmt(p)}, R2 {r2:.2f}).\n" + "\n".join(f"  - {n}: {pfmt(pp)}" for pp, n, *_ in rows[1:5]) +
                ("\nSingle events (not tested statistically): " + ", ".join(f"{c} x {t}" for t, c in ev) if ev else "") + "\nRanked by the p-value of each daily series' slope; p-values are not corrected here (ask for the multiple-testing correction).")

    def h_multitest(self, s, raw):
        p1, p2, n1, n2 = self.two_periods(s)
        rows = []
        for lab, un, a, b, w in self.period_rows(p1, p2):
            r = Q.welch(a, b)
            if r: rows.append((lab, r[2]))
        if len(rows) < 3: return "There are too few comparable signals to correct for multiple testing."
        ps = [p for _, p in rows]; raw_n = sum(1 for p in ps if p < 0.05); h = Q.holm(ps); q = Q.bh(ps); self.brief = f"Of {len(rows)} signals compared, {raw_n} {'looks' if raw_n == 1 else 'look'} significant before correction and {sum(h)} {'survives' if sum(h) == 1 else 'survive'} the strict correction."
        return (f"Of {len(rows)} signals compared between {n1} and {n2}, {raw_n} are significant at p < 0.05 before correction, {sum(h)} after Holm's correction (family-wise error 5%) and {sum(q)} after Benjamini-Hochberg (false-discovery rate 5%).\n" +
                "\n".join(f"  - {lab}: raw {pfmt(p)} -> {'survives' if hh else 'does not survive'} Holm" + (", survives BH" if qq else "") for (lab, p), hh, qq in sorted(zip(rows, h, q), key=lambda z: z[0][1])[:6]) + "\nWith many signals tested, some pass by chance; Holm is the strict correction, BH the more permissive.")

    def h_boot(self, s, raw):
        ser = self.series_set(); name = "battery state of health" if "battery state of health" in ser and len(ser["battery state of health"]) >= 8 else "coolant flow"
        sr = ser.get(name)
        if not sr or len(sr) < 8: return "There is no series with enough days to bootstrap."
        x = [(t - sr[0][0]) / DAY for t, _ in sr]; y = [v for _, v in sr]; lin = Q.ols_line(x, y); b = Q.bootstrap_slope(x, y, n=2000)
        if not b: return "The bootstrap could not run."
        unit = "points" if "health" in name else "L/min"; self.brief = f"The {name} changes by {lin['slope'] * 30:+.2f} {unit} a month (95% range {b[0] * 30:+.2f} to {b[1] * 30:+.2f})."
        return (f"The degradation rate of {name} is {lin['slope'] * 30:+.3f} {unit} per month, with a 95% bootstrap confidence interval of {b[0] * 30:+.3f} to {b[1] * 30:+.3f} ({b[2]} resamples of the {len(x)} daily values, seed fixed for reproducibility).\n"
                "Method: resample the (day, value) pairs with replacement, refit the line each time, take the 2.5th and 97.5th percentiles of the slopes. It assumes the days are independent, which weather persistence violates, so the interval is somewhat too narrow.")

    # ------------------------------------------------------------------ requests for impossible precision
    def h_precision(self, s, raw):
        n = "I can't predict that to that precision - an exact figure would be invented. "; self.brief = "I can't give that exactly - any exact figure would be invented, but I can give a range from this car's history."
        if re.search(r"battery temperature|pack temperature", s):
            m = re.search(r"(\d+) ?minutes?(?: and (\d+) ?seconds?)?", s); k = int(m[1]) if m else 17
            rows = self.c.execute("SELECT ts, batt FROM sig_1m WHERE mode IN (1,2,3) ORDER BY ts").fetchall(); idx = {t: v for t, v in rows}
            ch = sorted((idx[t + 60 * k] - v) / 10.0 for t, v in rows if t + 60 * k in idx)
            now_v = self.c.execute("SELECT batt FROM sig_1m ORDER BY ts DESC LIMIT 1").fetchone()
            if len(ch) < 30 or not now_v: return n + "There is too little history to say even how much it usually changes."
            self.brief = f"Not exactly: the pack is at {now_v[0] / 10:.1f} C now and usually moves between {ch[int(0.05 * len(ch))]:+.1f} and {ch[int(0.95 * len(ch))]:+.1f} C over {k} minutes."
            return (n + f"What I can say: the last recorded pack temperature is {now_v[0] / 10:.1f} C, and in this car's history it has changed over {k}-minute spans by between {ch[int(0.05 * len(ch))]:+.1f} and {ch[int(0.95 * len(ch))]:+.1f} C (90% of cases, median {ch[len(ch) // 2]:+.1f}), "
                    "so a range of values is defensible, an exact value is not.")
        if re.search(r"range", s) and re.search(r"500|exact", s):
            r = A.est_range(self.c, self.now); T = [t["wh_km"] for t in Q.trips_between(self.c, 0, self.now) if t["km"] >= 3]
            if not r.get("ok") or len(T) < 10: return n
            T.sort(); lo, hi = T[int(0.1 * len(T))], T[int(0.9 * len(T))]
            self.brief = f"Not exactly: roughly {self.cap_kwh() * 1000 / hi:.0f} to {self.cap_kwh() * 1000 / lo:.0f} km per charge, depending on how you drive."
            return n + f"Range depends on how you drive, the weather and charging stops en route. Right now the estimate is about {r['range_km']} km per charge; with your trips' 10th to 90th percentile consumption ({lo:.0f} to {hi:.0f} Wh/km) that is {self.cap_kwh() * 1000 / hi:.0f} to {self.cap_kwh() * 1000 / lo:.0f} km. After 500 km you would have needed to recharge anyway."
        if re.search(r"tomorrow|\d{1,2}:\d{2}", s) and re.search(r"speed", s):
            clk = Q.find_clock(s)
            if clk is not None:
                sp = []
                for back in range(0, 30):
                    st = Q.state_at(self.c, Q.local_midnight(self.today() - dt.timedelta(days=back), self.tz) + clk)
                    if "speed" in st: sp.append(st["speed"])
                if len(sp) >= 5: sp.sort(); self.brief = f"Nobody can know that; at that time of day the car has been doing anywhere from {sp[0]:.0f} to {sp[-1]:.0f} km/h."; return n + f"Nobody can know tomorrow's speed. For reference, at that time of day the car was running on {len(sp)} of the last 30 days, at between {sp[0]:.0f} and {sp[-1]:.0f} km/h (median {sp[len(sp) // 2]:.0f})."
            return n + "Tomorrow's speed depends on where you go and the traffic."
        if re.search(r"can message|next ten|ten sensor|next \d+ (sensor )?values", s): return n + "The next CAN frame or the next ten sensor values depend on the driver and the road in the next seconds; I only know what has been recorded. I can show the latest values."
        if re.search(r"component|part", s): return n + "I can't name the next component to fail - I see signals, not parts - but I can show which signals are drifting (ask for gradual degradation)."
        return n + "The next fault's time, to the second, depends on events that have not happened; I can describe the trend so far (ask 'has the anomaly got worse?') but not a timestamp."
