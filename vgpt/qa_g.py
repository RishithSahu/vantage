"""QA part G: the everyday questions a driver actually asks - battery state, how efficient was I, when did I last charge, what does charging cost, how long to charge, am I driving too fast,
how do I get more range, was the pack ever hot, is the 12 V battery / coolant pump OK, did anything unusual happen, best / worst day, "explain that warning simply".
Every answer is a short plain sentence first (that is what is spoken), numbers come from the stored data only, and what the car does not record (prices, speed limits, traffic) is said so."""
import re
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean

DAY = 86400


def hm(minutes):
    minutes = int(round(minutes))
    return f"{minutes} min" if minutes < 90 else f"{minutes // 60} h {minutes % 60:02d} min"


class MixG:
    def _soh(self):
        try: r = self.c.execute("SELECT soh FROM vday WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone()
        except Exception: r = None
        return r[0] if r else None

    def h_battery_now(self, s, raw):
        er = A.est_range(self.c, self.now); soh = self._soh(); sg = Q.load(self.c, self.now - 3600, self.now + 1, prefer="auto") if True else None
        t = self.c.execute("SELECT batt FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone()
        mx = self.c.execute("SELECT MAX(batt_max) FROM sig_1m WHERE ts>=? AND ts<=?", (self.now - 14 * DAY, self.now)).fetchone()[0]
        ot = self.c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE type='WARN_BATT_OVERTEMP' AND ts>=?", (self.now - 14 * DAY,)).fetchone()
        lines = []
        if er.get("ok"): lines.append(f"Charge: {er['soc']:.0f}%, about {er['range_km']} km of range at your recent {er['wh_km']:.0f} Wh/km.")
        if soh: lines.append(f"The battery health is {soh:.0f}% of its original capacity.")
        if t: lines.append(f"The pack temperature is {t[0] / 10:.0f} C now" + (f", peak {mx / 10:.0f} C in the last 14 days." if mx else "."))
        if ot and ot[0]: lines.append(f"Warnings: {ot[0]} over-temperature warning{'s' if ot[0] != 1 else ''} in the last 14 days (latest {self.fts(ot[1])}).")
        else: lines.append("Warnings: none in the last 14 days.")
        bad = bool(ot and ot[0])
        self.brief = (f"Battery {er['soc']:.0f}%, about {er['range_km']} km left" if er.get("ok") else "Battery state unknown") + (f", health {soh:.0f}%" if soh else "") + ("; it has overheated recently - ask me why." if bad else "; no problems recorded.")
        self.ctx.update(topic="battery")
        return "Your battery:\n" + "\n".join("  - " + l for l in lines)

    def h_eff_day(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.now - DAY, self.now + 1, "in the last 24 hours"))
        if lo is None: lo, hi, wtxt = self.now - DAY, self.now + 1, "in the last 24 hours"
        r = self.c.execute("SELECT COUNT(*), COALESCE(SUM(km),0), COALESCE(SUM(kwh),0), COALESCE(AVG(avg_amb),0), COALESCE(AVG(hwy_share),0) FROM trips WHERE start>=? AND start<?", (lo, hi)).fetchone()
        if not r[0] or r[1] < 0.5: return f"You did not drive {wtxt}, so there is no efficiency to report."
        wh = 1000 * r[2] / r[1]; all_ = self.c.execute("SELECT 1000*SUM(kwh)/SUM(km) FROM trips WHERE km>0.5").fetchone()[0]; d = 100 * (wh - all_) / all_
        why = []
        if r[4] > 0.5: why.append(f"{r[4] * 100:.0f}% of it on the motorway")
        if r[3] < 8: why.append(f"cold weather ({r[3]:.0f} C)")
        elif r[3] > 28: why.append(f"hot weather ({r[3]:.0f} C)")
        self.brief = f"{wtxt.capitalize() if not wtxt.startswith(('in', 'on')) else 'On that day'} you used {wh:.0f} Wh/km over {r[1]:.0f} km - " + ("about your usual." if abs(d) < 4 else f"{abs(d):.0f}% {'worse' if d > 0 else 'better'} than your usual {all_:.0f}.")
        return (f"{r[0]} trip{'s' if r[0] != 1 else ''}, {r[1]:.1f} km, {r[2]:.1f} kWh: {wh:.0f} Wh/km against your all-time {all_:.0f} Wh/km ({d:+.0f}%)." + (" Likely reasons: " + "; ".join(why) + "." if why and abs(d) >= 4 else ""))

    def sessions(self, lo=None, hi=None):
        """Charging sessions from the charges table, or - for a store without one - found in the signals (mode 3 = charging): [(start, end, kind, kw, kwh, soc0, soc1, grid_kwh)] oldest first."""
        lo = lo if lo is not None else 0; hi = hi if hi is not None else self.now + 1
        if self._has("charges"):
            return [tuple(r) for r in self.c.execute("SELECT start, end, kind, kw, kwh, soc0, soc1, grid_kwh FROM charges WHERE start>=? AND start<? ORDER BY start", (lo, hi))]
        rows = self.c.execute("SELECT ts, power, soc FROM sig_1m WHERE mode=3 AND power<-1000 AND ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall(); out = []; cur = None
        for ts, pw, soc in rows:
            if cur and ts - cur[1] <= 300: cur[1] = ts; cur[3].append(abs(pw) / 1000.0); cur[4] = soc / 10.0
            else:
                if cur: out.append(cur)
                cur = [ts, ts, None, [abs(pw) / 1000.0], soc / 10.0, soc / 10.0]
        if cur: out.append(cur)
        res = []
        for st, en, _, kws, s1, s0 in out:
            if len(kws) < 3: continue
            kw = max(kws); kwh = sum(kws) / 60.0; res.append((st, en, "DC" if kw >= 25 else "AC", kw, kwh, s0, s1, None))
        return res

    def h_last_charge(self, s, raw):
        ss = self.sessions(0, self.now + 1)
        if not ss: return "No charging session is recorded for this car."
        r = ss[-1]; ago = self.now - r[1]; n7 = [x for x in ss if x[0] >= self.now - 7 * DAY]
        kind = "fast DC charge" if r[2] == "DC" else "home / AC charge"
        self.brief = f"Your last charge was a {kind} on {self.day(r[0])}, {Q.dur(ago)} ago: {r[5]:.0f}% to {r[6]:.0f}% (+{r[4]:.1f} kWh)."
        return "\n".join([self.brief, f"It lasted {hm((r[1] - r[0]) / 60)} at up to {r[3]:.0f} kW. In the last 7 days you charged {len(n7)} time{'s' if len(n7) != 1 else ''}, {sum(x[4] for x in n7):.0f} kWh in total."])

    def h_charge_cost(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.now - 7 * DAY, self.now + 1, "in the last 7 days"))
        if lo is None: lo, hi, wtxt = self.now - 7 * DAY, self.now + 1, "in the last 7 days"
        ss = self.sessions(lo, hi); km = self.c.execute("SELECT COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<?", (lo, hi)).fetchone()[0]
        if not ss: return f"No charging sessions are recorded {wtxt}."
        kwh = sum(x[4] for x in ss); grid = sum((x[7] or x[4] / 0.92) for x in ss)
        m = re.search(r"(\d+(?:\.\d+)?) ?(?:p|c|cents?|pence|rs|rupees?|eur|euros?|dollars?|usd|gbp|[$£€₹])? ?(?:per|a|/) ?(?:kwh|unit)", raw.lower()) or re.search(r"[$£€₹] ?(\d+(?:\.\d+)?)", raw)
        if m:
            price = float(m[1]); cost = grid * price; pk = f" - about {100 * cost / km:.1f} per 100 km" if km > 5 else ""
            self.brief = f"About {cost:.1f} for {grid:.0f} kWh from the wall {wtxt} at {price:g} per kWh{pk}."
            return "\n".join([self.brief, f"{len(ss)} session{'s' if len(ss) != 1 else ''}, {kwh:.0f} kWh into the battery, {grid:.0f} kWh drawn from the grid (charging losses included)."])
        self.brief = f"I don't know your electricity price, but you drew about {grid:.0f} kWh from the wall {wtxt}. Tell me your price, e.g. 'at 0.30 per kWh'."
        return "\n".join([self.brief, f"{len(ss)} session{'s' if len(ss) != 1 else ''}, {kwh:.0f} kWh into the battery. Prices are not recorded by the car, so I won't guess one."])

    def h_charge_time(self, s, raw):
        m = re.search(r"(\d{2,3}) ?%", s); target = min(100.0, float(m[1])) if m else 100.0
        er = A.est_range(self.c, self.now); soc = er["soc"] if er.get("ok") else None
        if soc is None: return "I don't know the current charge level."
        if soc >= target: return f"The battery is already at {soc:.0f}%, above {target:.0f}%."
        cap = self.cap_kwh(); need = (target - soc) / 100 * cap; ss = self.sessions(0, self.now + 1)
        ac = [x[3] for x in ss if x[2] == "AC"]; dc = [x[3] for x in ss if x[2] == "DC"]; parts = []
        if ac: parts.append(f"{hm(need / (mean(ac) * 0.92) * 60)} on your usual {mean(ac):.0f} kW home charger")
        if dc: parts.append(f"about {hm(need / (mean(dc) * 0.75) * 60)} on a {mean(dc):.0f} kW fast charger (slower near the top)")
        if not parts: return "No charging history is recorded, so I cannot estimate the time."
        self.brief = f"From {soc:.0f}% to {target:.0f}% needs {need:.0f} kWh: " + "; or ".join(parts) + "."
        return "\n".join([self.brief, "An estimate from your own sessions: real charge time depends on temperature, the charger and the battery state."])

    def h_battery_equiv(self, s, raw):
        """Convert a target battery's charge capacity to energy before comparing it with the EV.

        mAh is charge, not energy, so voltage is deliberately required instead of
        silently treating unlike battery types as equivalent.
        """
        cm = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(milliamp(?:ere)?[- ]?hours?|mah)\b", s)
        vm = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:v|volts?)\b", s)
        if not cm: return "Tell me the target battery capacity in mAh and its nominal voltage so I can compare energy, not just charge."
        mah = float(cm[1].replace(",", ""))
        if not vm:
            return (f"A {mah:g} mAh rating alone is not enough: I also need the target battery's nominal voltage. "
                    f"For example, say '{mah:g} mAh at 3.7 V'. Different voltages store different amounts of energy, so I won't guess one.")
        volts = float(vm[1]); target_wh = mah * volts / 1000.0
        if not (0 < target_wh < 1_000_000): return "Those battery capacity or voltage values do not form a plausible target battery."
        er = A.est_range(self.c, self.now); soc = er.get("soc") if er.get("ok") else None
        if soc is None: return "The target battery energy can be calculated, but the car's current state of charge is unavailable."
        usable_kwh = A.cap_kwh(self.c); available_wh = usable_kwh * 1000.0 * soc / 100.0
        ideal = available_wh / target_wh; efficiency = 0.85; practical = int(ideal * efficiency)
        self.brief = (f"A {mah:g} mAh, {volts:g} V battery stores about {target_wh:.1f} Wh. "
                      f"At the car's recorded {soc:.0f}% charge, that is roughly {practical:,} complete charges after allowing for conversion losses.")
        return "\n".join([self.brief,
                          f"Ideal energy-only maximum: about {ideal:,.0f}; practical estimate: {practical:,} using an explicit {efficiency:.0%} car-to-device charging efficiency.",
                          f"Basis: {usable_kwh:.1f} kWh usable vehicle capacity at {soc:.0f}% state of charge. This estimate does not reserve energy for driving or battery-protection limits."])

    def h_speeding(self, s, raw):
        sg = Q.load(self.c, self.now - 7 * DAY, self.now + 1, prefer="1m")
        mv = [v for v, m in zip(sg.cols["speed"], sg.cols["mode"]) if m in (1, 2) and v > 5] if len(sg) else []
        if len(mv) < 30: return "There is not enough recent driving to say."
        n = len(mv); a110 = 100 * sum(1 for v in mv if v > 100) / n; a130 = 100 * sum(1 for v in mv if v > 130) / n; top = max(mv)
        w_hi, _ = self.whkm_at(120.0); w_lo, _ = self.whkm_at(90.0); extra = f" Driving at 120 instead of 90 km/h costs about {w_hi - w_lo:.0f} Wh/km more." if w_hi and w_lo else ""
        self.brief = f"I don't have speed limits, so I can't say if you broke one. {a110:.0f}% of your moving time was above 100 km/h (top speed {top:.0f} km/h)." + extra
        return self.brief + "\nSpeed limits come from map data, which this assistant does not have; the figures above are from the car's own speed record."

    def h_improve_range(self, s, raw):
        tips = []
        w_hi, _ = self.whkm_at(120.0); w_lo, _ = self.whkm_at(100.0); cap = self.cap_kwh() * 1000
        if w_hi and w_lo and w_hi > w_lo * 1.05: tips.append((cap / w_lo - cap / w_hi, f"Cruise at 100 instead of 120 km/h: about {cap / w_lo - cap / w_hi:.0f} km more per full charge."))
        d = self.c.execute("SELECT COALESCE(SUM(x.harsh_brake),0), COALESCE(SUM(x.harsh_accel),0), COALESCE(SUM(t.km),0) FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=?", (self.now - 28 * DAY,)).fetchone() if self._has("trip_ext") else None
        if d and d[2] > 50:
            hb = 100 * d[0] / d[2]
            if hb > 6: tips.append((20, f"Brake earlier and more gently: you brake hard {hb:.0f} times per 100 km (typical 2-6); smooth driving also gives the motor more regeneration."))
        hv = self.c.execute("SELECT AVG(avg_hvac_w) FROM trips WHERE start>=?", (self.now - 28 * DAY,)).fetchone()[0]
        if hv and hv > 700: tips.append((15, f"Climate control averages {hv:.0f} W; pre-heating or cooling while plugged in saves battery energy."))
        try:
            ty = AX.tyres(self.c, self.now, "14")
            low = [w["name"] for w in ty["wheels"] if w["status"] != "ok"] if ty.get("ok") else []
            if low: tips.append((10, f"Inflate the {', '.join(low)} tyre{'s' if len(low) > 1 else ''}: low pressure raises rolling resistance."))
        except Exception: pass
        if not tips: tips.append((1, "Your driving is already efficient by this car's own history - the remaining levers are speed on the motorway and pre-conditioning while plugged in."))
        tips.sort(key=lambda t: -t[0])
        self.brief = tips[0][1] + (" Ask for 'more tips' for the rest." if len(tips) > 1 else "")
        return "What would help most, from your own data:\n" + "\n".join(f"  - {t[1]}" for t in tips[:4]) + "\nThese are estimates from this car's history, not guarantees."

    def h_hot_battery(self, s, raw):
        lo = self.cov()["lo"]; mx = self.c.execute("SELECT batt_max, ts FROM sig_1m WHERE ts>=? ORDER BY batt_max DESC LIMIT 1", (lo,)).fetchone()
        if not mx: return "There is no battery temperature in the history."
        over = self.c.execute("SELECT COUNT(*) FROM sig_1m WHERE batt_max>=400").fetchone()[0]; ev = self.c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE type='WARN_BATT_OVERTEMP'").fetchone()
        st = Q.state_at(self.c, mx[1]); mode = {0: "parked", 1: "city driving", 2: "motorway driving", 3: "charging"}.get(st.get("mode"), "")
        self.brief = f"Yes - the pack peaked at {mx[0] / 10:.0f} C on {self.day(mx[1])}" + (f" while {mode}" if mode else "") + (f", and it triggered {ev[0]} over-temperature warning{'s' if ev[0] != 1 else ''}." if ev[0] else "; no warning was raised.") if mx[0] >= 400 else f"No - the pack never went above {mx[0] / 10:.0f} C (the warning level is 40-52 C)."
        return self.brief + f"\nMinutes at or above 40 C: {over}. Peak: {mx[0] / 10:.1f} C at {self.fts(mx[1])}."

    def h_v12(self, s, raw):
        try: sv = AX.service(self.c, self.now)
        except Exception: sv = {}
        v = sv.get("v12") if sv.get("ok") else None
        if v is None: return "The 12 V battery voltage is not recorded for this car."
        ev = self.c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE type='LOW_12V' AND ts>=?", (self.now - 30 * DAY,)).fetchone()
        verdict = "healthy" if v >= 12.3 and not ev[0] else "getting weak" if v >= 12.0 else "low"
        self.brief = f"The 12 V battery reads {v:.1f} V at rest - {verdict}" + (f", with {ev[0]} low-voltage warning{'s' if ev[0] != 1 else ''} in the last 30 days." if ev[0] else ".")
        return self.brief + "\nAbout 12.6 V at rest is healthy; below 12.0 V it may fail to start the car's electronics."

    def h_unusual(self, s, raw):
        days = 1 if re.search(r"\btoday|tonight|this morning", s) else 2 if re.search(r"yesterday", s) else 7
        F, ab = self.findings(days)
        if not F: self.brief = f"Nothing unusual {'today' if days == 1 else 'yesterday' if days == 2 else 'in the last 7 days'}: no warnings, faults or odd readings."; return self.brief
        self.brief = f"Yes - {len(F)} thing{'s' if len(F) != 1 else ''}: " + "; ".join(re.sub(r"\s*[(].*", "", re.sub(r"^.{0,30}?(battery over-temperature warning|diagnostic trouble code|low 12 V battery warning)", r"\1", f["text"]))[:80] for f in F[:2]) + ("; ask for details." if len(F) > 2 else ".")
        return "What stood out:\n" + "\n".join("  - " + f["text"] for f in F[:6])

    def h_best_day(self, s, raw):
        cv = self.cov(); rows = [r for r in A.daily_efficiency(self.c, cv["lo"], cv["hi"] + 1)]
        if len(rows) < 3: return "There are too few driving days to compare."
        best = min(rows, key=lambda r: r["wh_km"]); worst = max(rows, key=lambda r: r["wh_km"]); want_worst = bool(re.search(r"least|worst|less efficient|most wasteful|lowest", s))
        f = lambda r: f"{self.day(r['ts'])}: {r['wh_km']:.0f} Wh/km at {r['amb']:.0f} C outside"
        self.brief = (f"The least efficient day was {f(worst)}." if want_worst else f"The most efficient day was {f(best)}.")
        return "\n".join([f"Most efficient: {f(best)}.", f"Least efficient: {f(worst)}.", f"Average of all {len(rows)} driving days: {mean([r['wh_km'] for r in rows]):.0f} Wh/km."])

    def h_explain_simple(self, s, raw):
        e = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN ('WARN_BATT_OVERTEMP','DTC_SET','TPMS_LOW','LOW_12V','SERVICE_DUE') AND ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone()
        if not e: return "There is no warning on record to explain."
        plain = {"WARN_BATT_OVERTEMP": "the battery pack got hotter than its safe limit", "DTC_SET": f"the car stored a fault code ({Q.safe_text(e[3])[:60]})", "TPMS_LOW": "a tyre's pressure fell below the safe level", "LOW_12V": "the small 12 V battery that runs the electronics was low", "SERVICE_DUE": "a service is due"}[e[1]]
        cause = ""
        thermal = e[1] == "WARN_BATT_OVERTEMP" or (e[1] == "DTC_SET" and re.search(r"coolant|pack|batt", e[3] or "", re.I))
        try:
            d = self.diag_event(e) if thermal else None
            if d: cause = re.sub(r" [(].*?[)]", "", d["alts"][0]["name"])
        except Exception: pass
        if e[1] == "LOW_12V": cause = "an ageing 12 V battery or something draining it while parked"
        self.brief = f"On {self.day(e[0])}, {plain}." + (f" The most likely reason: {cause}." if cause else "") + " It is worth having it checked."
        return self.brief + "\nSay 'why did it happen' for the evidence."

    def h_pump_ok(self, s, raw):
        cv = self.cov(); fb = Q.flow_baseline(self.c, self.now); fl = Q.flow_days(self.c, cv["hi"] - 5 * DAY, cv["hi"] + 1, minn=10) if fb else None
        if not fb or not fl: return "There is not enough coolant-flow history to judge."
        now_med = fl[-1][1]
        on = Q.flow_onset(self.c, self.now)
        if now_med is None: return "There is not enough recent coolant-flow data."
        drop = 100 * (1 - now_med / fb["p50"])
        if drop < 12 and not on: self.brief = f"Yes - coolant flow is normal ({now_med:.1f} L/min against {fb['p50']:.1f} typical)."; return self.brief
        self.brief = f"No - coolant flow is {drop:.0f}% below normal ({now_med:.1f} against {fb['p50']:.1f} L/min)" + (f" since {self.day(on[0])}" if on else "") + "; the pump or cooling circuit needs checking."
        return self.brief + "\nThe car has no pump-speed signal, so I can see the effect (low flow), not which part is at fault."

    def h_eff_trend(self, s, raw):
        a = self._tr(self.now - 7 * DAY, self.now + 1); b = self._tr(self.now - 28 * DAY, self.now - 7 * DAY)
        if not a or not b: return "There is not enough driving in both periods to compare."
        d = 100 * (a - b) / b
        self.brief = ("About the same" if abs(d) < 4 else f"Yes - {abs(d):.0f}% more" if d > 0 else f"No - {abs(d):.0f}% less") + f": {a:.0f} Wh/km in the last 7 days against {b:.0f} over the three weeks before."
        why = self.c.execute("SELECT AVG(avg_amb) FROM trips WHERE start>=?", (self.now - 7 * DAY,)).fetchone()[0]; why0 = self.c.execute("SELECT AVG(avg_amb) FROM trips WHERE start>=? AND start<?", (self.now - 28 * DAY, self.now - 7 * DAY)).fetchone()[0]
        return self.brief + (f"\nOutside temperature averaged {why:.0f} C against {why0:.0f} C before." if why is not None and why0 is not None and abs(why - why0) >= 3 else "")

    def _tr(self, lo, hi):
        r = self.c.execute("SELECT SUM(kwh), SUM(km) FROM trips WHERE start>=? AND start<? AND km>0.5", (lo, hi)).fetchone()
        return 1000 * r[0] / r[1] if r and r[1] and r[1] > 10 else None
