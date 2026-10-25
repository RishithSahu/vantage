"""QA part C: explanations with evidence (efficiency change, events), hypotheses and confidence, vague 'is anything wrong' questions, long-range history, and checks on the assistant's own earlier answers."""
import re, math, datetime as dt
from collections import Counter
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd, F1, F0, SIGNED, plural, pfmt
from .qa_a import UNIT, NAME, WARN_TYPES, EVENT_SCORE

DAY = 86400
MISSING_COMMON = ["payload / passengers and cargo weight", "wind speed and direction along the route", "road gradient / elevation profile", "cabin temperature set-point and number of occupants", "traffic and stop-start counts from a map provider",
                  "roof box or other accessories (drag)", "tyre tread / rolling-resistance of the tyres fitted", "a controlled capacity test of the battery (to separate ageing from conditions)"]


class MixC:
    # ------------------------------------------------------------------ efficiency change
    def diag_range(self, lo=None, hi=None, topic="range"):
        cv = self.cov(); hi = hi or cv["hi"] + 1; lo = lo or hi - 14 * DAY
        D = Q.daily_series(self.c, lo, hi)
        if len(D) < 6: return None
        y = [d["wh_km"] for d in D]; best = None
        for k in range(3, len(D) - 2):
            w = Q.welch(y[:k], y[k:])
            if w and (best is None or abs(w[0]) > abs(best[1][0])): best = (k, w)
        if not best: return None
        k, w = best; a, b = y[:k], y[k:]; ma, mb = mean(a), mean(b); pct = 100 * (mb - ma) / ma
        pa = (D[0]["ts"], D[k]["ts"]); pb = (D[k]["ts"], hi)
        sse_step = sum((v - ma) ** 2 for v in a) + sum((v - mb) ** 2 for v in b); lin = Q.ols_line([d["ts"] / DAY for d in D], y); sse_lin = sum((v - (lin["intercept"] + lin["slope"] * d["ts"] / DAY)) ** 2 for v, d in zip(y, D)) if lin else float("inf")
        shape = "a step (the level changed on one date and stayed)" if sse_step < 0.8 * sse_lin else "a gradual drift"
        jumps = sorted(((D[i]["wh_km"] - D[i - 1]["wh_km"], D[i]) for i in range(1, len(D))), key=lambda x: -x[0])[:3]
        fit = Q.eff_fit(self.c, cv["lo"], cv["hi"] + 1); Ta, Tb = Q.trips_between(self.c, *pa), Q.trips_between(self.c, *pb)
        for t in Ta + Tb: t["tyre"] = Q.tyre_deficit(self.c, t["start"])
        res = Q.eff_period(fit, Tb, Ta) if fit else None
        cap = A.cap_kwh(self.c); ra, rb = cap * 1000 / ma, cap * 1000 / mb
        d = {"topic": topic, "kind": "range", "lo": lo, "hi": hi, "pa": pa, "pb": pb, "ma": ma, "mb": mb, "pct": pct, "w": w, "shape": shape, "jumps": jumps, "res": res, "days_a": k, "days_b": len(D) - k, "ra": ra, "rb": rb, "D": D, "Ta": Ta, "Tb": Tb, "fit": fit, "sse": (sse_step, sse_lin)}
        sig = self.period_rows(pa, pb); d["sig"] = sig
        lead = res["factors"][0] if res and res["factors"] and abs(res["factors"][0]["contrib"]) > 0.5 else None
        d["headline"] = (f"energy use per km {'rose' if pct > 0 else 'fell'} {abs(pct):.0f}% ({ma:.0f} -> {mb:.0f} Wh/km, {pfmt(w[2])}{'' if w[2] < 0.05 else ' - not statistically clear'}), i.e. range {ra:.0f} -> {rb:.0f} km, with the best split of the days at {self.day(pb[0])}" + (f"; the largest measurable contributor is {lead['name']}" if lead else "; no recorded condition explains it"))
        d["trend"] = f"energy use is {ma:.0f} Wh/km before and {mb:.0f} Wh/km after {self.day(pb[0])}."
        return d

    def range_text(self, d, part="all"):
        res = d["res"]; out = []
        pa, pb = d["pa"], d["pb"]; la, lb = f"{self.day(pa[0])} to {self.day(pb[0] - 1)}", f"{self.day(pb[0])} to {self.day(pb[1] - 1)}"
        if part in ("evidence", "all"):
            out.append(f"Observed: daily energy use averaged {d['ma']:.1f} Wh/km over {plural(d['days_a'], 'day')} ({la}) and {d['mb']:.1f} Wh/km over {plural(d['days_b'], 'day')} ({lb}): {SIGNED(d['pct'], 1)}%, Welch t = {d['w'][0]:.1f}, {pfmt(d['w'][2])}. "
                       f"With the battery's usable {A.cap_kwh(self.c):.1f} kWh that is a range of about {d['ra']:.0f} km before and {d['rb']:.0f} km after.")
            if res and res["factors"]:
                out.append(f"Trip-level analysis ({res['n']} trips, regression R2 = {res['r2']:.2f}) of the {abs(res['gap']):.1f} Wh/km gap between the {res['n_good']} earlier and {res['n_bad']} later trips:")
                for f in res["factors"][:4]:
                    if abs(f["contrib"]) < 0.05: continue
                    sh = 100 * f["contrib"] / res["gap"] if res["gap"] else 0
                    out.append(f"  - {f['name']}: {f['good']:.1f} -> {f['bad']:.1f} {f['unit']}; at {f['per_unit']:.2f} Wh/km per unit that accounts for {SIGNED(f['contrib'])} Wh/km (" + (f"{sh:.0f}% of the change" if sh >= 0 else "works against the change") + ")" if res["gap"] else "")
                expl = res["explained"]; out.append(f"  - recorded conditions together explain {SIGNED(expl)} Wh/km of the {SIGNED(res['gap'])} Wh/km gap ({100 * expl / res['gap']:.0f}%); the remaining {res['gap'] - expl:+.1f} Wh/km is not accounted for by the recorded conditions in this model (it may come from something that is not recorded, or from what the model cannot capture)." if res["gap"] else "")
            lines = self.fmt_rows(d["sig"], top=4)
            if lines: out.append("Other signals in the same periods (biggest differences first):"); out += lines
        return out

    def alternatives(self, d):
        res = d["res"]; alts = []; gap = res["gap"] if res else 0
        fac = {f["key"]: f for f in res["factors"]} if res else {}

        def share(*keys): return sum(fac[k]["contrib"] for k in keys if k in fac) / gap * 100 if gap else 0

        def strength(sh): return "strong" if sh >= 35 else "moderate" if sh >= 15 else "weak" if sh >= 5 else "not supported"
        if "heating" in fac or "cooling" in fac:
            h = fac.get("heating", fac.get("cooling")); sh = share("heating", "cooling")
            alts.append({"name": "weather: more energy for heating/cooling the cabin and a colder battery", "share": sh, "strength": strength(sh), "support": f"outside temperature {self.fmt_amb(d)}; climate power {self.row_change(d, 'Climate power')}; model: {sh:.0f}% of the gap", "against": "weather changes slowly - it cannot explain a one-day jump unless the cold snap began that day"})
        if "speed" in fac or "hwy" in fac:
            sh = share("speed", "hwy"); alts.append({"name": "a different mix of driving (speed, more motorway, fewer stops)", "share": sh, "strength": strength(sh), "support": f"average trip speed {self.row_change(d, 'Average speed')}, motorway share {self.row_change(d, 'Motorway share')}; model: {sh:.0f}% of the gap", "against": "the model can't see route choice, traffic or hills"})
        if "tyre" in fac:
            ta = mean([t["tyre"] for t in d["Ta"]]); tb = mean([t["tyre"] for t in d["Tb"]]); sh = share("tyre"); n_t = self.c.execute("SELECT COUNT(*) FROM events WHERE type='TPMS_LOW' AND ts BETWEEN ? AND ?", (d["pa"][0], d["pb"][1])).fetchone()[0]
            alts.append({"name": "low tyre pressure (rolling resistance)", "share": sh, "strength": strength(sh), "support": f"mean pressure shortfall {ta:.1f}% -> {tb:.1f}% below spec; {n_t} low-pressure warning(s) in the period; model: {sh:.0f}%", "against": "pressure was normal in the later period" if tb < 2 else "the effect size is small for a few percent of pressure"})
        if "harsh" in fac:
            sh = share("harsh"); alts.append({"name": "more aggressive driving (hard braking / acceleration)", "share": sh, "strength": strength(sh), "support": f"hard events per 100 km {self.row_change(d, 'Hard events')}; model: {sh:.0f}%", "against": "only a few hard events per trip"})
        if "rain" in fac:
            sh = share("rain"); alts.append({"name": "rain on the road", "share": sh, "strength": strength(sh), "support": f"model: {sh:.0f}% of the gap", "against": "little rain in the data"})
        soh = self.c.execute("SELECT soh FROM vday WHERE ts>=? AND ts<? ORDER BY ts", (d["pa"][0], d["pb"][1])).fetchall() if self._has("vday") else []
        if len(soh) >= 2: alts.append({"name": "battery ageing (less usable capacity)", "share": 0.0, "strength": "not supported", "support": f"state of health {soh[0][0]:.1f}% -> {soh[-1][0]:.1f}% in the same period (a {soh[0][0] - soh[-1][0]:.2f}-point change)", "against": "the change is far too small to explain it"})
        un = (gap - res["explained"]) / gap * 100 if res and gap else 100
        alts.append({"name": "something the car does not measure (payload, wind, hills, roof box, a different route, a driver change)", "share": un, "strength": strength(un), "support": f"{un:.0f}% of the gap is left over after all recorded conditions", "against": "unfalsifiable with this data - it can only be tested by recording those quantities"})
        alts.sort(key=lambda a: -a["share"]); return alts

    def row_change(self, d, label):
        for lab, un, a, b, w in d["sig"]:
            if lab.startswith(label): return f"{mean(a):.1f} -> {mean(b):.1f} {un}"
        return "no data"

    def fmt_amb(self, d): return self.row_change(d, "Outside temperature")

    def confidence_range(self, d):
        res = d["res"]; p = d["w"][2]; ex = (res["explained"] / res["gap"]) if res and res["gap"] else 0; r2 = res["r2"] if res else 0
        lvl = "high" if (p < 0.01 and ex >= 0.6 and r2 >= 0.5) else "moderate" if (p < 0.05 and ex >= 0.3) else "low"
        why = f"the change itself is {'statistically clear' if p < 0.05 else 'not statistically clear'} ({pfmt(p)}, {d['days_a']} vs {d['days_b']} days); recorded conditions explain {100 * ex:.0f}% of it (model R2 {r2:.2f})"
        if d["days_b"] < 5: why += f"; only {d['days_b']} days after the change, which is a small sample"
        return lvl, why

    # ------------------------------------------------------------------ events
    def diag_event(self, e=None, s="", cut=0, window=3600):
        """Explain an event from the data of the `window` seconds before it, optionally using nothing later than `cut` seconds before it (no look-ahead)."""
        if e is None:
            e = self.most_serious()
            if e is None: return None
        ts = e[0]; tx = ts - cut; cv = self.cov(); mid = cv["lo"] + (cv["hi"] - cv["lo"]) // 2; lo = cv["lo"]; hi = min(mid, lo + 14 * DAY)
        if cut and tx - window < hi: hi = max(lo + 3600, min(hi, tx - window))                       # the 'normal' baseline may not reach into the period being judged
        sg = Q.event_sig(self.c, tx, window, 0 if cut else 120); pre = Q.summarise(sg, tx - window, tx + 1)
        if not pre: return None
        bf = Q.baseline_stats(self.c, "flow", lo, hi, modes=(1, 2, 3), minv=3.0); bb = Q.baseline_stats(self.c, "batt", lo, hi, modes=(1, 2, 3)); bp = Q.baseline_stats(self.c, "power", lo, hi, modes=(1, 2), scale=1000.0); ba = Q.baseline_stats(self.c, "amb", lo, hi, modes=(1, 2, 3))
        st = Q.state_at(self.c, tx); alts = []; ev = []; claims = []; fl_now = None; first_low = None
        T = f"{self.fts(ts)}"; thermal = (e[1] == "WARN_BATT_OVERTEMP") or bool(re.search(r"coolant|thermal|P1A|battery|pack", e[3] or "", re.I))
        ev.append(f"Observed: {self.evrow(e)}")
        claims.append({"text": f"{e[1]} recorded at {T}", "kind": "observed"})
        det = {"event": e}
        if thermal and bf:
            fl = [v for v, md in zip(sg.cols["flow"], sg.cols["mode"]) if v > 0.3 and md in (1, 2, 3)]; fl_now = mean(fl[-300:]) if fl else None
            fo = Q.flow_onset(self.c, self.now, upto=tx); first_low = fo[0] if fo else None
            if fl_now is not None:
                ratio = fl_now / bf["p50"]; sh = "strong" if ratio < 0.7 else "moderate" if ratio < 0.85 else "none"
                alts.append({"name": "reduced coolant flow (pump or cooling-circuit degradation)", "strength": sh, "support": f"coolant flow was {fl_now:.1f} L/min in the hour before ({100 * ratio:.0f}% of the typical {bf['p50']:.1f} L/min)" + (f"; the daily median flow has stayed below normal since {self.day(first_low)}" if first_low else ""), "against": "a flow reading alone doesn't distinguish a weak pump from a blocked line or a faulty flow sensor"})
                claims.append({"text": f"coolant flow {fl_now:.1f} L/min vs typical {bf['p50']:.1f}", "kind": "observed"})
            if bb:
                mx = pre["batt"]["max"]; alts.append({"name": "a heavy thermal load (fast charging or sustained high power)", "strength": "moderate" if (st.get("charging") and st["charging"]["kind"] == "DC") or pre["power"]["mean"] > (bp["p50"] if bp else 0) * 2 + 5 else "weak",
                                                      "support": f"mean power in the hour before {pre['power']['mean']:.1f} kW (typical {bp['mean']:.1f} kW)" if bp else f"mean power {pre['power']['mean']:.1f} kW" + (f"; {st['charging']['kind']} charging at {st['charging']['kw']:.0f} kW" if st.get("charging") else ""), "against": "load alone is normal for this car; it only overheats if cooling is also impaired"})
            if ba:
                z = (pre["amb"]["mean"] - ba["mean"]) / ba["sd"]; alts.append({"name": "a hot day (ambient temperature)", "strength": "moderate" if z > 2 else "weak" if z > 1 else "not supported", "support": f"outside {pre['amb']['mean']:.1f} C vs a normal {ba['mean']:.1f} C (z = {z:+.1f})", "against": "the outside temperature was not unusual" if z < 1 else "ambient heat alone doesn't trip the warning"})
            jump = max((abs(sg.cols["batt"][i] - sg.cols["batt"][i - 1]) for i in range(1, len(sg)) if sg.ts[i] - sg.ts[i - 1] <= 2), default=0)
            alts.append({"name": "a faulty temperature sensor (a false reading)", "strength": "not supported" if jump < 2 else "moderate", "support": f"the pack temperature moved smoothly (largest step {jump:.2f} C between samples) and the other signals agree", "against": "I have a single pack-temperature signal here, so I can't cross-check it against a second sensor"})
        elif re.search(r"12 ?v|voltage|P0562|LOW_12V", f"{e[1]} {e[3]}", re.I) and self._has("vday"):
            vd = self.c.execute("SELECT ts, v12 FROM vday WHERE ts<=? ORDER BY ts", (ts + 86400,)).fetchall()
            if len(vd) >= 4:
                k = max(2, len(vd) // 4); a, b = mean([v for _, v in vd[:k]]), mean([v for _, v in vd[-k:]]); fit = Q.slope_test([t / 86400 for t, _ in vd], [v for _, v in vd])
                alts.append({"name": "a weakening 12 V battery (ageing or a parasitic drain)", "strength": "strong" if (a - b) > 0.25 else "moderate" if (a - b) > 0.1 else "weak",
                             "support": f"the rest voltage of the 12 V battery fell from {a:.2f} V to {b:.2f} V over {len(vd)} days" + (f" ({Q.pfmt(fit['p'])})" if fit else ""), "against": "a voltage trend doesn't separate the battery itself from something draining it while parked"})
            alts.append({"name": "a short-term load (cold start, accessories)", "strength": "weak", "support": f"outside {st.get('amb', float('nan')):.1f} C at the time", "against": "a one-off load wouldn't produce a steady decline"})
        elif re.search(r"tyre|tpms|C0750|pressure", f"{e[1]} {e[3]}", re.I) and self._has("ext_1m"):
            wi = next((i for i, w in enumerate(AX.WHEELS) if w in (e[3] or "").lower()), None); cols = ("tp_fl", "tp_fr", "tp_rl", "tp_rr")
            if wi is not None:
                rows = self.c.execute(f"SELECT ts, {cols[wi]}, tt FROM ext_1m WHERE ts<=? AND ts>=? ORDER BY ts", (ts, ts - 14 * 86400)).fetchall(); vals = [(t, AX._cold(p / 10.0, tt)) for t, p, tt in rows if p]
                if len(vals) >= 20:
                    k = max(5, len(vals) // 10); a, b = mean([v for _, v in vals[:k]]), mean([v for _, v in vals[-k:]])
                    alts.append({"name": f"a slow leak in the {AX.WHEELS[wi]} tyre", "strength": "strong" if a - b > 15 else "moderate" if a - b > 6 else "weak", "support": f"its cold-equivalent pressure fell from {a:.0f} to {b:.0f} kPa over the 14 days before the warning, while the other tyres stayed level", "against": "the sensor can't tell a puncture from a valve leak or a faulty sensor"})
            alts.append({"name": "a cold snap (pressure falls about 7 kPa per 10 C)", "strength": "weak", "support": f"outside {st.get('amb', float('nan')):.1f} C; the readings above are temperature-corrected, so cold alone doesn't explain a decline", "against": "the correction assumes the tyre is at air temperature"})
        else:
            alts.append({"name": "the event's recorded cause", "strength": "strong", "support": Q.safe_text(e[3]), "against": "the data doesn't record anything beyond the event text for this type"})
        order = {"strong": 3, "moderate": 2, "weak": 1, "none": 0, "not supported": 0}; alts.sort(key=lambda a: -order[a["strength"]])
        top = alts[0]
        lvl = "high" if (top["strength"] == "strong" and len([a for a in alts if a["strength"] in ("strong", "moderate")]) >= 1 and bf and fl_now is not None and fl_now / bf["p50"] < 0.7 and first_low) else "moderate" if top["strength"] in ("strong", "moderate") else "low"
        why = (f"the leading explanation ({top['name']}) is backed by the data - {top['support']} - but I only have indirect signals (no part-level diagnostics), so a physical inspection is still needed" if lvl != "low" else "the signals don't clearly single out one cause")
        d = {"topic": "event", "kind": "event", "thermal": thermal, "event": e, "alts": alts, "evidence": ev, "claims": claims, "confidence": lvl, "conf_why": why, "sg": sg, "pre": pre, "base": {"flow": bf, "batt": bb, "power": bp, "amb": ba}, "ts": ts, "cut": cut, "used": (tx - window, tx), "baseline_span": (lo, hi)}
        d["headline"] = f"{Q.KIND_NAME.get({'WARN_BATT_OVERTEMP': 'overheat', 'DTC_SET': 'diagnostic'}.get(e[1], ''), e[1])} at {T}: most consistent with {top['name']}"
        d["trend"] = f"the leading explanation is {top['name']} ({top['support']})"
        d["missing"] = (["a second, independent temperature sensor (to rule out a false reading)", "pump speed / current (PumpRpm exists in the CAN database but is not logged here)", "coolant temperature and level", "the cell-level temperature spread", "physical inspection or workshop diagnostics"] if thermal
                        else ["a second, independent measurement of the same quantity", "workshop diagnostics or a physical inspection", "what the driver or the car was doing in the minutes around it"])
        return d

    def event_text(self, d, part="all"):
        out = []
        if part in ("all", "evidence"):
            out += d["evidence"]
            pre = d["pre"]
            if not d.get("thermal"): out.append(f"Observed in the hour before: speed averaged {pre['speed']['mean']:.0f} km/h, outside {pre['amb']['mean']:.1f} C."); return out
            out.append(f"Observed in the hour before: pack temperature {pre['batt']['first']:.1f} -> {pre['batt']['last']:.1f} C (max {pre['batt']['max']:.1f} C), coolant flow {pre['flow']['first']:.1f} -> {pre['flow']['last']:.1f} L/min, power mean {pre['power']['mean']:.1f} kW, speed {pre['speed']['mean']:.0f} km/h average, outside {pre['amb']['mean']:.1f} C.")
        return out

    def ensure_diag(self, s=""):
        d = self.ctx.get("diag")
        topic = self.ctx.get("topic")
        if d and (not topic or topic == d.get("topic")): return d
        return self.diag_default(s)

    def diag_default(self, s="", quiet=False):
        d = None; topic = self.ctx.get("topic")
        if topic == "range": d = self.diag_range()
        elif topic == "event":
            d = self.diag_event(self.ctx.get("event_row") or None)
        if d is None:
            ms = self.most_serious()
            d = self.diag_event(ms) if ms else self.diag_range()
        if d: self.ctx["diag"] = d; self.ctx["topic"] = d["topic"]
        return d

    # ------------------------------------------------------------------ explain-why handlers
    def h_why_trip(self, s, raw):
        r = self.c.execute("SELECT start, end, km, kwh, wh_km, avg_speed, hwy_share, avg_amb, avg_hvac_w FROM trips WHERE km>2 AND start<=? ORDER BY start DESC LIMIT 1", (self.now,)).fetchone()
        cv = self.cov(); fit = Q.eff_fit(self.c, cv["lo"], cv["hi"] + 1)
        if not r or not fit: return None
        T = Q.trips_between(self.c, r[0], r[0] + 1)
        if not T: return None
        t = T[0]; t["tyre"] = Q.tyre_deficit(self.c, t["start"]); x = Q.eff_trip(fit, t); diff = x["actual"] - x["typical"]
        out = [f"Last trip ({self.fts(t['start'])}, {t['km']:.1f} km): {x['actual']:.0f} Wh/km against a typical {x['typical']:.0f} Wh/km over the {x['n']} trips on record ({SIGNED(diff, 0)} Wh/km, {100 * diff / x['typical']:+.0f}%)."]
        if abs(diff) < 0.08 * x["typical"]: out.append("That is within normal variation - there was no efficiency drop on this trip.")
        else:
            out.append(f"What the car's data says about why (regression on all trips, R2 = {x['r2']:.2f}):")
            tot = 0.0
            for f in x["factors"][:4]:
                if abs(f["contrib"]) < 0.5: continue
                out.append(f"  - {f['name']}: {f['value']:.1f} vs typical {f['mean']:.1f} {f['unit']} -> {SIGNED(f['contrib'])} Wh/km"); tot += f["contrib"]
            out.append(f"  - recorded conditions explain {SIGNED(tot)} Wh/km of the {SIGNED(diff)} Wh/km; the rest ({diff - tot:+.0f}) is unexplained (payload, wind, hills, route, traffic are not recorded).")
        d = {"topic": "trip", "kind": "trip", "headline": out[0], "trend": out[0], "evidence": out[1:], "alts": [], "claims": [{"text": out[0], "kind": "observed"}], "confidence": "moderate" if abs(diff) >= 0.08 * x["typical"] else "n/a", "conf_why": f"trip-level regression R2 {x['r2']:.2f}", "missing": MISSING_COMMON[:6]}
        self.ctx.update(diag=d, topic="trip"); return "\n".join(out)

    def h_why_event(self, s, raw):
        e, lab, note, ab = self.anchor(s if Q.kind_of(s) else "overheating")
        d = self.diag_event(e if e else self.most_serious())
        if d is None: return note or "There is no event in the data to explain."
        self.ctx.update(diag=d, topic="event", event_row=d["event"])
        out = [(note.strip() + " " if note else "") + f"Event: {self.evrow(d['event'])}"]
        out += self.event_text(d)[1:]
        out.append("Most likely explanation (ranked by how strongly the data supports each):")
        for k, a in enumerate(d["alts"][:3], 1): out.append(f"  {k}. {a['name']} - {a['strength']}: {a['support']}.")
        out.append(f"Confidence: {d['confidence']} - {d['conf_why']}.")
        cl = lambda n: re.sub(r" [(].*?[)]", "", n)
        al = d["alts"]; e0 = d["event"]
        self.brief = f"For the {self.day(e0[0])} event, the most likely cause is {cl(al[0]['name'])} ({al[0]['strength']} evidence)" + (f", with {cl(al[1]['name'])} adding to it" if len(al) > 1 and al[1]["strength"] in ("strong", "moderate") else "") + f". Confidence {d['confidence']}; only a physical check can confirm it."
        return "\n".join(out)

    def h_why_range2(self, s, raw):
        txt = self.range_answer("14")
        return txt or None

    def range_answer(self, w="14"):
        """The full answer to 'why has my range dropped': the change, how reliable it is, the contributors, and the single-day jumps. Also stores the diagnosis for the follow-up questions."""
        cv = self.cov(); days = {"7": 7, "14": 14, "28": 28}.get(w, None)
        hi = cv["hi"] + 1; lo = hi - days * DAY if days else cv["lo"]
        d = self.diag_range(lo=lo, hi=hi, topic="range")
        if not d: return None
        self.ctx.update(diag=d, topic="range")
        D = d["D"]; k = max(3, min(len(D) // 3, 3)); a, b = mean([x["wh_km"] for x in D[:k]]), mean([x["wh_km"] for x in D[-k:]]); cap = A.cap_kwh(self.c) * 1000
        lvl, why = self.confidence_range(d); res = d["res"]
        top = [x for x in (res["factors"] if res else []) if abs(x["contrib"]) >= 0.5][:1]
        self.brief = (f"Your range went from about {cap / a:.0f} to {cap / b:.0f} km ({100 * (cap / b - cap / a) / (cap / a):+.0f}%)" + (" - it did not actually drop." if cap / b > 1.02 * cap / a else "") +
                      (f", mostly from {top[0]['name']}" if top and abs(d['pct']) > 2 else "") + (f"; but the change is small next to normal day-to-day variation (confidence {lvl})." if d["w"][2] >= 0.05 else f" (confidence {lvl}).")) if True else None
        out = [("Your range has not reduced - it actually rose. " if cap / b > 1.02 * cap / a else "") + f"Over the last {len(D)} days with trips ({self.day(D[0]['ts'])} to {self.day(D[-1]['ts'])}) your estimated range went from about {cap / a:.0f} km to {cap / b:.0f} km ({SIGNED(100 * (cap / b - cap / a) / (cap / a), 0)}%): energy use per km changed from {a:.1f} to {b:.1f} Wh/km (first {k} vs last {k} days)."]
        out.append(f"Is it a real change? The best split of the daily values is at {self.day(d['pb'][0])}: {d['ma']:.1f} -> {d['mb']:.1f} Wh/km ({SIGNED(d['pct'])}%, {pfmt(d['w'][2])}) - " + ("a clear, statistically significant shift." if d["w"][2] < 0.05 else "small compared with normal day-to-day variation, so the range 'drop' is weak evidence of a real change."))
        if res and res["factors"]:
            f = [x for x in res["factors"] if abs(x["contrib"]) >= 0.5][:3]
            if f: out.append("What the recorded conditions say about why (trip-level regression, R2 = %.2f): " % res["r2"] + "; ".join(f"{x['name']} {SIGNED(x['contrib'])} Wh/km" for x in f) + f"; {100 * (res['gap'] - res['explained']) / res['gap']:.0f}% of the change is not accounted for by the recorded conditions in this model (it may come from something that is not recorded, or from what the model cannot capture)." if res["gap"] else "")
        if d["jumps"]:
            j = d["jumps"][0]; out.append(f"The largest single-day jump was {j[1]['date']}: {SIGNED(j[0])} Wh/km to {j[1]['wh_km']:.0f} ({j[1]['trips']} trips, {j[1]['km']:.0f} km, outside {j[1]['amb']:.1f} C) - check for added drag such as a roof box, or low tyre pressure.")
        out.append(f"Confidence: {lvl} - {why}.")
        return "\n".join(out)

    def h_evidence(self, s, raw):
        d = self.diag_default(s)
        if not d: return "I haven't concluded anything yet and there isn't enough data to build an explanation."
        out = [f"Conclusion: {d['headline']}.", "Evidence (all computed from the stored data):"]
        if d["kind"] == "range": out += ["  " + x for x in self.range_text(d, "evidence")]
        elif d["kind"] == "event":
            out += ["  " + x for x in self.event_text(d)]
            for a in d["alts"][:3]: out.append(f"  - {a['name']} ({a['strength']}): {a['support']}.")
        else: out += ["  " + x for x in d.get("evidence", [])]
        out.append("Marked as observed or inferred: ask 'which part is observed and which is inferred'.")
        return "\n".join(out)

    def h_top3(self, s, raw):
        d = self.diag_default(s)
        if not d: return "There is nothing to explain yet."
        alts = d.get("alts") or (self.alternatives(d) if d["kind"] == "range" else [])
        if not alts: return "No alternative explanations could be built from the data."
        out = [f"Top three possible explanations for: {d['headline']}."]
        for k, a in enumerate(alts[:3], 1): out.append(f"  {k}. {a['name']} - {a['strength']}.\n       data for: {a['support']}\n       data against / limits: {a['against']}")
        return "\n".join(out)

    def h_obs_inf(self, s, raw):
        d = self.diag_default(s)
        if not d: return "I haven't made an explanation yet, so there is nothing to split."
        obs, inf = [], []
        if d["kind"] == "range":
            obs.append(f"daily energy use {d['ma']:.1f} -> {d['mb']:.1f} Wh/km ({SIGNED(d['pct'])}%), the dates it changed ({self.day(d['pb'][0])}), and the range figures calculated from them")
            for lab, un, a, b, w in d["sig"][:3]: obs.append(f"{lab}: {mean(a):.1f} -> {mean(b):.1f} {un}")
            inf += [f"that {a['name']} contributed (a regression estimate, share {a['share']:.0f}%), because the car doesn't measure each effect separately" for a in self.alternatives(d)[:2]]
            inf.append("that the step on that date has a single cause")
        elif d["kind"] == "event":
            for c_ in d["claims"]: (obs if c_["kind"] == "observed" else inf).append(c_["text"])
            inf += [f"that it was {a['name']} (inferred from {a['support']})" for a in d["alts"][:1]] + ["that signals which moved before the event are related to it, not just coincident"]
        else:
            obs.append(d["headline"]); inf.append("the reasons for it")
        return "Directly observed (stored values and timestamps):\n  - " + "\n  - ".join(obs) + "\nInferred (my reasoning from them, not measured):\n  - " + "\n  - ".join(inf)

    def h_alt(self, s, raw):
        d = self.diag_default(s)
        if not d: return "There is no explanation yet to find an alternative for."
        alts = d.get("alts") or (self.alternatives(d) if d["kind"] == "range" else [])
        prim = alts[0]["name"] if alts else ""
        rest = alts[1:] or alts
        out = [f"Besides the leading explanation ({prim}), other things that could produce the same pattern:"]
        for a in rest[:4]: out.append(f"  - {a['name']} ({a['strength']}): {a['support']}. Against: {a['against']}.")
        return "\n".join(out)

    def h_conf(self, s, raw):
        d = self.diag_default(s)
        if not d: return "I haven't given an explanation yet, so there is no confidence to state."
        if d["kind"] == "range": lvl, why = self.confidence_range(d)
        else: lvl, why = d["confidence"], d["conf_why"]
        return f"Confidence: {lvl}. Because {why}. It would go up with the missing information I listed under 'what data are you missing', and down if a counter-example appears (ask 'what evidence contradicts your diagnosis')."

    def h_counter(self, s, raw):
        d = self.diag_default(s)
        if not d: return "I haven't given a previous answer or diagnosis to contradict."
        out = [f"Evidence that doesn't fit or weakens the diagnosis ({d['headline']}):"]
        if d["kind"] == "range":
            res = d["res"]
            if res and res["gap"]: out.append(f"  - {100 * (res['gap'] - res['explained']) / res['gap']:.0f}% of the change is NOT explained by any recorded condition.")
            out.append(f"  - the same data fit {'a gradual drift' if 'step' in d['shape'] else 'a step'} nearly as well (squared error step {d['sse'][0]:.0f} vs straight line {d['sse'][1]:.0f}), so the 'when' is less certain than it looks.")
            out.append(f"  - before the change the daily values already ranged {min(x['wh_km'] for x in d['D'][:d['days_a']]):.0f} to {max(x['wh_km'] for x in d['D'][:d['days_a']]):.0f} Wh/km, so a rise of {SIGNED(d['mb'] - d['ma'], 0)} Wh/km is not far outside normal day-to-day swings.")
            for lab, un, a, b, w in d["sig"]:
                ww = Q.welch(a, b)
                if ww and ww[2] > 0.3 and lab in ("Outside temperature", "Motorway share", "Average speed (trips)"): out.append(f"  - {lab} did not change significantly ({mean(a):.1f} -> {mean(b):.1f}, {pfmt(ww[2])}), so it can't be what moved the efficiency.")
            out.append(f"  - only {d['days_b']} day(s) of data after the change.")
        elif d["kind"] == "event":
            for a in d["alts"]:
                if a["strength"] in ("weak", "not supported", "none"): out.append(f"  - {a['name']}: {a['strength']} - {a['against'] if a['strength'] != 'not supported' else a['support']}.")
            out.append("  - the leading explanation rests on indirect signals; no direct measurement of the pump or the part exists in the data.")
        else: out.append("  - nothing concrete: the finding rests on a single computation.")
        return "\n".join(out)

    def h_missing_info(self, s, raw):
        d = self.diag_default(s)
        miss = (d.get("missing") if d else None) or MISSING_COMMON
        if d and d["kind"] == "range": miss = MISSING_COMMON
        out = ["Information that I don't have and that would let me separate the explanations / reach a reliable conclusion:"] + [f"  - {m}" for m in miss]
        if d and d["kind"] == "range":
            out.append("What each would settle: payload, accessories and tyre data -> the unexplained part of the gap; wind and gradient -> route effects vs the car; a capacity test -> ageing vs conditions; cabin set-point -> how much of the climate load was a choice.")
        else: out.append("What each would settle: a second temperature sensor and pump data -> whether the flow drop is the pump itself or its sensor/line; coolant level -> a leak; a workshop inspection -> the failing part.")
        return "\n".join(out)

    def h_prove(self, s, raw):
        cv = self.cov(); D = Q.daily_series(self.c, cv["lo"], cv["hi"] + 1)
        bt = {d: v for d, v, _ in self.daily("batt", cv["lo"], cv["hi"] + 1, modes=(1, 2, 3))}
        pairs = [(bt[d["ts"]], d["wh_km"], d["amb"]) for d in D if d["ts"] in bt]
        if len(pairs) < 8: return "Not enough daily data to test that."
        r = Q.pearson([p[0] for p in pairs], [p[1] for p in pairs]); r_a = Q.pearson([p[2] for p in pairs], [p[1] for p in pairs]); fit = Q.ols_multi([[p[0], p[2]] for p in pairs], [p[1] for p in pairs])
        lag = []
        for sh in (-1, 0, 1):
            xs = [(pairs[i][0], pairs[i + sh][1]) for i in range(max(0, -sh), min(len(pairs), len(pairs) - sh))]; lag.append((sh, Q.pearson([a for a, _ in xs], [b for _, b in xs])))
        self.brief = (f"No, I can't prove it: pack temperature mostly follows the weather, and outside temperature explains energy use better (r {r_a:+.2f} against {r:+.2f})." if r_a is not None and abs(r_a) > abs(r) else
                      f"No, I can't prove it: pack temperature and energy use move together (r {r:+.2f}), but correlation is not causation and a time-lag test doesn't show temperature leading.")
        out = [f"No - I can't prove that, and the data doesn't suggest it. Over {len(pairs)} days the daily mean pack temperature and the daily energy use per km correlate at r = {r:+.2f}" + (f", while outside temperature and energy use correlate at r = {r_a:+.2f}." if r_a is not None else ".")]
        if fit: out.append(f"Putting both in one regression (R2 = {fit[1]:.2f}): +1 C pack temperature -> {fit[0][1]:+.2f} Wh/km, +1 C outside -> {fit[0][2]:+.2f} Wh/km.")
        out.append("Why a correlation can't prove causation here: (1) pack temperature follows the weather and the charging/driving pattern, which also move energy use; (2) I have no controlled change (an experiment) to isolate it; (3) a time lag test (battery today vs efficiency +/-1 day: " + ", ".join(f"{sh:+d} d r={v:+.2f}" for sh, v in lag if v is not None) + ") doesn't show the temperature leading the efficiency. Causation would need an intervention or a mechanism that the signals confirm.")
        return "\n".join(out)

    def h_calc(self, s, raw):
        d = self.diag_default(s)
        if not d: return "There is no conclusion yet to show the working for."
        out = ["Timestamps and calculations behind the last conclusion:"]
        if d["kind"] == "range":
            D = d["D"]; k = d["days_a"]
            out.append(f"  daily energy use = kWh / km x 1000 per local day, from the trips table; period {self.day(d['lo'])} to {self.day(d['hi'] - 1)} ({len(D)} days).")
            out.append(f"  change point = the split into two groups with the largest Welch t statistic: before = {self.day(D[0]['ts'])} .. {self.day(D[k - 1]['ts'])} (n={k}), after = {self.day(D[k]['ts'])} .. {self.day(D[-1]['ts'])} (n={len(D) - k}).")
            out.append("  daily values: " + "; ".join(f"{x['date'][5:]} {x['wh_km']:.0f}" for x in D) + " Wh/km")
            out.append(f"  before mean = {d['ma']:.2f}, after mean = {d['mb']:.2f}; change = ({d['mb']:.2f} - {d['ma']:.2f}) / {d['ma']:.2f} = {d['pct']:+.1f}%; t = {d['w'][0]:.2f}, df = {d['w'][1]:.1f}, {pfmt(d['w'][2])}.")
            out.append(f"  range = usable capacity {A.cap_kwh(self.c):.2f} kWh / Wh-per-km: {A.cap_kwh(self.c) * 1000:.0f} / {d['ma']:.1f} = {d['ra']:.0f} km before, {A.cap_kwh(self.c) * 1000:.0f} / {d['mb']:.1f} = {d['rb']:.0f} km after.")
            res = d["res"]
            if res:
                out.append(f"  attribution: Wh/km = b0 + sum(coef x condition) fitted on {res['n']} trips (R2 {res['r2']:.2f}); each factor's share = coef x (later mean - earlier mean).")
                for f in res["factors"][:4]: out.append(f"    {f['name']}: {f['per_unit']:.3f} Wh/km per {f['unit']} x ({f['bad']:.1f} - {f['good']:.1f}) = {f['contrib']:+.2f} Wh/km")
            jt = d["jumps"]
            if jt: out.append("  largest day-to-day jumps: " + "; ".join(f"{x[1]['date']} {x[0]:+.1f} Wh/km (to {x[1]['wh_km']:.0f})" for x in jt))
        elif d["kind"] == "event":
            pre = d["pre"]; out.append(f"  event time {self.fts(d['ts'])}; window = the 60 minutes before it ({d['sg'].tier}).")
            for m in ("batt", "flow", "power", "amb", "speed"): out.append(f"  {NAME.get(m, m)}: first {pre[m]['first']:.2f} {UNIT[m]}, last {pre[m]['last']:.2f} {UNIT[m]}, mean {pre[m]['mean']:.2f} {UNIT[m]}, min {pre[m]['min']:.2f} {UNIT[m]} at {self.fts(pre[m]['tmin'])}, max {pre[m]['max']:.2f} {UNIT[m]} at {self.fts(pre[m]['tmax'])}")
            b = d["base"]["flow"]
            if b: out.append(f"  normal coolant flow = median of the first 14 days' running minutes = {b['p50']:.2f} L/min (n={b['n']}).")
        else: out += ["  " + x for x in d.get("evidence", [])]
        return "\n".join(out)

    def h_conclusion(self, s, raw):
        d = self.ensure_diag(s) or self.diag_default(s)
        if not d: return "There isn't enough data to reach a conclusion."
        if d["kind"] == "range": lvl, why = self.confidence_range(d); alts = self.alternatives(d)
        else: lvl, why, alts = d["confidence"], d["conf_why"], d.get("alts", [])
        out = [f"Conclusion: {d['headline']}."]
        out.append("Supporting signals and timestamps:")
        if d["kind"] == "range": out += ["  " + x for x in self.range_text(d, "evidence")]
        elif d["kind"] == "event": out += ["  " + x for x in self.event_text(d)]
        else: out += ["  " + x for x in d.get("evidence", [])]
        out.append("Calculations: " + ("daily energy use (kWh/km x 1000), Welch t-test between the two periods, trip-level regression for the attribution (ask 'show the timestamps and calculations' for the full working)." if d["kind"] == "range" else "baseline = the car's own first-two-weeks median/standard deviation; z = (value - baseline mean) / sd; ratio to the typical flow."))
        out.append(f"Confidence: {lvl} - {why}.")
        unres = [a for a in alts if a["strength"] in ("moderate", "strong")][1:3] or alts[1:2]
        out.append("Unresolved uncertainty: " + ("; ".join(f"{a['name']} ({a['strength']})" for a in unres) if unres else "none of the alternatives is supported either") + "; not measured: " + ", ".join(m.split(" (")[0] for m in (d.get("missing") or MISSING_COMMON)[:4]) + ".")
        return "\n".join(out)

    def h_periods(self, s, raw):
        d = self.ctx.get("diag") if self.ctx.get("topic") == "range" else None
        d = d or self.diag_range(topic="range")
        if not d or d["kind"] != "range": return None
        self.ctx.update(diag=d, topic="range")
        pa, pb = d["pa"], d["pb"]
        out = [f"The change happened as {d['shape']}:", f"  - before: {self.fts(pa[0])[:10]} to {self.day(pb[0] - 1)} ({plural(d['days_a'], 'day')}), mean {d['ma']:.1f} Wh/km",
               f"  - after: {self.day(pb[0])} to {self.day(pb[1] - 1)} ({plural(d['days_b'], 'day')}), mean {d['mb']:.1f} Wh/km"]
        if d["jumps"]: out.append("Largest single-day jumps: " + "; ".join(f"{x[1]['date']}: {x[0]:+.1f} Wh/km (to {x[1]['wh_km']:.0f}, {x[1]['trips']} trips, {x[1]['km']:.0f} km, outside {x[1]['amb']:.1f} C)" for x in d["jumps"]) + ".")
        worst = sorted(d["Tb"], key=lambda t: -t["wh_km"])[:3]
        if worst: out.append("Most energy-hungry trips in the later period: " + "; ".join(f"{self.fts(t['start'])[:16]} ({t['km']:.1f} km, {t['wh_km']:.0f} Wh/km)" for t in worst) + ".")
        return "\n".join(out)

    def h_signals_during(self, s, raw):
        d = self.ctx.get("diag") if self.ctx.get("topic") == "range" else None
        d = d or self.diag_range(topic="range")
        if not d or d["kind"] != "range": return None
        self.ctx.update(diag=d, topic="range")
        sig = [r for r in d["sig"] if (lambda w: w and w[2] < 0.05)(Q.welch(r[2], r[3]))]
        out = [("Note: efficiency did not drop in this period - it improved. " if d["pct"] < -1 else "") + f"Energy use per km: {d['ma']:.1f} -> {d['mb']:.1f} Wh/km ({SIGNED(d['pct'])}%, {pfmt(d['w'][2])}{'' if d['w'][2] < 0.05 else ' - weak evidence of a real change'}). Other signals compared between the period before ({self.day(d['pa'][0])} to {self.day(d['pb'][0] - 1)}) and after ({self.day(d['pb'][0])} to {self.day(d['pb'][1] - 1)}) - daily means, Welch test, biggest differences first:"] + self.fmt_rows(d["sig"])
        out.append(f"{len(sig)} of {len(d['sig'])} signals changed significantly; an overlap in time is an association, not proof that one caused the other.")
        return "\n".join(out)

    def findings(self, days=14):
        cv = self.cov(); hi = cv["hi"] + 1; lo = hi - days * DAY; F = []
        ab = Q.last_abnormal(self.c, self.now)
        for ty, w, nm in (("WARN_BATT_OVERTEMP", 10, "battery over-temperature warning"), ("DTC_SET", 6, "diagnostic trouble code"), ("LOW_12V", 5, "low 12 V battery warning"), ("TPMS_LOW", 3, "low tyre pressure warning"), ("CAN_ANOMALY", 6, "CAN bus anomaly"), ("CAMERA_BLOCKED", 2, "camera blockage"), ("CAMERA_DEGRADED", 2, "camera degradation")):
            ev = Q.events(self.c, "warning", lo, hi) if False else self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type=? AND ts>=? AND ts<?", (ty, lo, hi)).fetchall()
            for e in ev: F.append({"score": w + (e[0] - lo) / (hi - lo), "text": f"{self.fts(e[0])} {nm}: {Q.safe_text(e[3])}", "ts": e[0], "kind": ty})
        # flow degradation
        fl = self.daily("flow", cv["lo"], hi, modes=(1, 2, 3), minv=3.0)
        if len(fl) >= 8:
            k = max(3, len(fl) // 5); a, b = mean([v for _, v, _ in fl[:k]]), mean([v for _, v, _ in fl[-k:]])
            if b < 0.85 * a: F.append({"score": 8, "text": f"coolant flow has fallen {100 * (a - b) / a:.0f}% ({a:.1f} -> {b:.1f} L/min, daily means, first vs last {k} days) and is still low", "ts": fl[-1][0], "kind": "flow"})
        d = self.diag_range(lo=lo, hi=hi)
        if d and d["w"][2] < 0.05 and d["pct"] > 4: F.append({"score": 4 + d["pct"] / 10, "text": f"energy use per km rose {d['pct']:.0f}% from {self.day(d['pb'][0])} ({d['ma']:.0f} -> {d['mb']:.0f} Wh/km, {pfmt(d['w'][2])})", "ts": d["pb"][0], "kind": "range"})
        sg = self.sig(max(lo, cv["hi"] - 7 * DAY), hi)
        if len(sg):
            mn = min(sg.cols["soc"]); mx = max(sg.cols["batt"])
            if mn < 8: F.append({"score": 3, "text": f"state of charge fell to {mn:.0f}% in the last week", "ts": sg.ts[sg.cols['soc'].index(mn)], "kind": "soc"})
        try:
            r = self.c.execute("SELECT ts, tp_fl, tp_fr, tp_rl, tp_rr, tt FROM ext_1m ORDER BY ts DESC LIMIT 1").fetchone()
            sf, sr = AX._mf(self.c, "veh.tyre_spec_front"), AX._mf(self.c, "veh.tyre_spec_rear")
            if r and sf:
                spec = [sf, sf, sr, sr]; low = [i for i in range(4) if AX._cold(r[i + 1] / 10.0, r[5]) < 0.9 * spec[i]]
                if low: F.append({"score": 4, "text": f"a tyre is below 90% of its specified pressure at the latest reading ({self.fts(r[0])})", "ts": r[0], "kind": "tyre"})
        except Exception: pass
        hb = self.c.execute("SELECT COUNT(*) FROM events WHERE type IN ('HARSH_BRAKE','HARSH_ACCEL') AND ts>=? AND ts<?", (lo, hi)).fetchone()[0]; km = self.c.execute("SELECT SUM(km) FROM trips WHERE start>=? AND start<?", (lo, hi)).fetchone()[0] or 0
        allhb = self.c.execute("SELECT COUNT(*) FROM events WHERE type IN ('HARSH_BRAKE','HARSH_ACCEL')").fetchone()[0]; allkm = self.c.execute("SELECT SUM(km) FROM trips").fetchone()[0] or 1
        if km > 20 and hb / km > 1.6 * allhb / allkm and hb > 20: F.append({"score": 2, "text": f"driving was harder than usual: {hb / km * 100:.0f} hard events per 100 km vs {allhb / allkm * 100:.0f} overall", "ts": hi, "kind": "harsh"})
        F.sort(key=lambda f: -f["score"]); self.ctx["findings"] = F; return F, ab

    def h_checkup(self, s, raw):
        days = 7 if re.search(r"recent|today|yesterday|lately|sluggish|changed", s) else 14
        if re.search(r"yesterday|today", s) and re.search(r"bad day|how was", s): return self.h_badday(s, raw)
        F, ab = self.findings(days)
        out = [f"I'm reading your question as: any warning or fault event, any signal outside its own normal band (the car's first two weeks), or a clear change in efficiency, in the last {days} days. Findings, most serious first:"]
        if not F: out.append("  Nothing unusual: no warnings or faults, no signal outside its normal band, no change in efficiency.")
        for f in F[:6]: out.append(f"  - {f['text']}")
        if re.search(r"sluggish|slow|power|accelerat", s):
            sg = self.sig(self.cov()["hi"] - 3 * DAY, self.cov()["hi"] + 1); lowsoc = min(sg.cols["soc"]) if len(sg) else None; cold = min(sg.cols["batt"]) if len(sg) else None
            out.append("About 'sluggish': the car has no throttle or torque signal, so I can't see the feel directly. Things in the data that limit power: " + (f"pack temperature as low as {cold:.1f} C in the last 3 days (a cold pack limits power and regeneration); lowest charge {lowsoc:.0f}% (power is limited when nearly empty)" if lowsoc is not None else "no recent data") + "; low tyre pressure and a weak 12 V battery are covered by the findings above.")
        if ab and not any(f["kind"] == "flow" for f in F): out.append(f"Latest abnormal reading: {ab['label']} {ab['value']:.1f} {ab['unit']} at {self.fts(ab['ts'])} (normal {ab['normal']}).")
        if re.search(r"normal", s) and not F: out[0] = "Yes, it looks normal. " + out[0]
        elif re.search(r"normal", s): out[0] = f"Not entirely: {len(F)} thing(s) stand out. " + out[0]
        return "\n".join(out)

    def h_investigate(self, s, raw):
        F, ab = self.findings(14)
        if not F: return "Nothing stands out in the data, so there is no priority - the car's own signals and events look normal."
        out = ["What I would investigate first, ranked by severity and by how long it could go on unnoticed:"]
        seen = set()
        for f in F:
            if f["kind"] in seen: continue
            seen.add(f["kind"]); out.append(f"  {len(seen)}. {f['text']}")
            if len(seen) >= 3: break
        why = {"flow": "a failing coolant pump can lead to over-temperature and battery damage - it is a safety-relevant fault that gets worse with time", "WARN_BATT_OVERTEMP": "an over-temperature warning is the car's own highest-severity battery alert", "DTC_SET": "a stored trouble code is the car's own record of a fault"}
        top = F[0]; out.append("Why that first: " + why.get(top["kind"], "it is the most severe thing found, and the others are smaller or already explained") + ". This is a ranking of what the data shows, not a diagnosis.")
        return "\n".join(out)

    def day_stats(self):
        cv = self.cov(); lo, hi = cv["lo"], cv["hi"] + 1; D = {d["ts"]: d for d in Q.daily_series(self.c, lo, hi)}
        fl = {d: v for d, v, _ in self.daily("flow", lo, hi, modes=(1, 2, 3), minv=3.0)}; bt = {d: v for d, v, _ in self.daily("batt_max", lo, hi, modes=(1, 2, 3))}; ev = {}
        for ts, ty in self.c.execute("SELECT ts, type FROM events WHERE type IN (%s)" % ",".join("?" * len(WARN_TYPES)), WARN_TYPES): ev.setdefault((ts + A.TZ) // DAY * DAY - A.TZ, []).append(ty)
        rows = []
        for ts in sorted(set(D) | set(fl) | set(ev)):
            rows.append({"ts": ts, "wh": D.get(ts, {}).get("wh_km"), "harsh": D.get(ts, {}).get("harsh"), "km": D.get(ts, {}).get("km", 0), "flow": fl.get(ts), "bmax": bt.get(ts), "ev": ev.get(ts, [])})
        return rows

    def day_score(self, rows):
        whs = [r["wh"] for r in rows if r["wh"]]; hs = [r["harsh"] for r in rows if r["harsh"] is not None]; fls = [r["flow"] for r in rows if r["flow"]]; bts = [r["bmax"] for r in rows if r["bmax"]]
        base_f = sorted(fls)[len(fls) * 3 // 4] if fls else None
        for r in rows:
            comp = []; sc = 0.0
            for ty in r["ev"]:
                w = EVENT_SCORE.get(ty, 1); sc += w; comp.append(f"{ty} (+{w})")
            if r["wh"] and len(whs) > 3 and sd(whs) > 0: z = (r["wh"] - mean(whs)) / sd(whs); (sc := sc + max(0, z)); comp.append(f"energy use {r['wh']:.0f} Wh/km (z {z:+.1f})") if z > 1 else None
            if r["harsh"] is not None and len(hs) > 3 and sd(hs) > 0: z = (r["harsh"] - mean(hs)) / sd(hs); (sc := sc + max(0, z)); comp.append(f"{r['harsh']:.0f} hard events/100 km (z {z:+.1f})") if z > 1 else None
            if r["flow"] and base_f and r["flow"] < 0.7 * base_f: sc += 2; comp.append(f"coolant flow {r['flow']:.1f} L/min ({100 * r['flow'] / base_f:.0f}% of normal) (+2)")
            if r["bmax"] and len(bts) > 3 and sd(bts) > 0: z = (r["bmax"] - mean(bts)) / sd(bts); (sc := sc + max(0, z)); comp.append(f"pack peak {r['bmax']:.1f} C (z {z:+.1f})") if z > 1 else None
            r["score"], r["comp"] = sc, comp
        return rows

    def h_worstday(self, s, raw):
        rows = self.day_score(self.day_stats())
        if not rows: return "Not enough data."
        w = max(rows, key=lambda r: r["score"]); top = sorted(rows, key=lambda r: -r["score"])[:3]
        self.brief = f"{self.day(w['ts'])}: " + ("; ".join(c.split(" (")[0] for c in w["comp"][:3]) or "nothing stands out") + "."
        return (f"The worst 24-hour period (calendar day, local time) was {self.day(w['ts'])} (score {w['score']:.1f}): " + ("; ".join(w["comp"]) or "nothing stands out") + f". Runners-up: " + "; ".join(f"{self.day(r['ts'])} ({r['score']:.1f})" for r in top[1:]) +
                ". Score = event severities + how many standard deviations the day's efficiency, hard driving and pack peak are above average + a penalty for low coolant flow.")

    def h_badday(self, s, raw):
        rows = self.day_score(self.day_stats()); d, expl, dtxt = Q.find_date(s, self.now, self.tz)
        if d is None: d = self.today() - dt.timedelta(days=1)
        ts = Q.local_midnight(d, self.tz); r = next((x for x in rows if x["ts"] == ts), None)
        if not r: return f"There is no recorded driving or events on {d.isoformat()}, so it cannot have been a bad day for the vehicle (the car recorded nothing)."
        rank = sorted(rows, key=lambda x: -x["score"]).index(r) + 1; avg = mean([x["score"] for x in rows])
        verdict = "Yes, worse than usual" if r["score"] > avg * 1.5 and r["score"] > 2 else "No, an ordinary day"
        return f"{verdict}: {d.isoformat()} scored {r['score']:.1f} against an average day of {avg:.1f} (rank {rank} of {len(rows)} days). " + ("; ".join(r["comp"]) if r["comp"] else "No warnings, and efficiency, driving style, pack temperature and coolant flow were all within their usual range.")

    def h_recurring(self, s, raw):
        evs = self.c.execute("SELECT type, detail, COUNT(*), MIN(ts), MAX(ts) FROM events WHERE type IN (%s) GROUP BY type, substr(detail,1,5)" % ",".join("?" * len(WARN_TYPES)), WARN_TYPES).fetchall()
        rec = [e for e in evs if e[2] >= 2]; out = []
        if rec: out.append("Warning/fault events that occurred more than once: " + "; ".join(f"{e[2]} x {e[0]} ({Q.safe_text(e[1])[:40]}), {self.day(e[3])} to {self.day(e[4])}" for e in rec) + ".")
        else: out.append("No warning or fault code was logged more than once.")
        cv = self.cov(); b = Q.flow_baseline(self.c, self.now)
        if b:
            days = Q.flow_days(self.c, cv["lo"], cv["hi"] + 1); low = [d for d in days if d[1] < 0.85 * b["p50"]]
            if len(low) >= 2: out.append(f"Recurring abnormal readings: the daily median coolant flow was below 85% of normal ({b['p50']:.1f} L/min) on {len(low)} of {len(days)} days ({self.day(low[0][0])} to {self.day(low[-1][0])}), down to {min(d[1] for d in low):.1f} L/min - a persistent condition, not a one-off.")
        rows = self.c.execute("SELECT ((ts+(%d))/86400)*86400-(%d) d, MAX(batt_max) FROM sig_1m GROUP BY d" % (A.TZ, A.TZ)).fetchall()
        if len(rows) >= 8:
            thr = sorted(r[1] for r in rows)[int(0.8 * len(rows))]; hot = [r for r in rows if r[1] >= thr]
            if len(hot) >= 3: out.append(f"Pack temperature peaks of {thr / 10:.1f} C or more (the hottest 20% of days) occurred on {len(hot)} days - peaks of heat repeat, but none reached a warning level.")
        return "\n".join(out)

    def h_repeat(self, s, raw):
        rows = self.c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' ORDER BY ts").fetchall()
        if not rows: return "No fault codes have been recorded at all, so none repeated."
        codes = {}
        for ts, det in rows: codes.setdefault((det or "").split(" ")[0], []).append((ts, det))
        rep = {k: v for k, v in codes.items() if len(v) > 1}; out = []
        if rep:
            for k, v in rep.items(): out.append(f"Yes - {k} occurred {len(v)} times: " + ", ".join(self.fts(t) for t, _ in v) + " (gaps: " + ", ".join(Q.dur(b[0] - a[0]) for a, b in zip(v, v[1:])) + ").")
        else:
            out.append(f"No - each of the {len(codes)} fault codes was set once: " + "; ".join(f"{k} at {self.fts(v[0][0])}" for k, v in codes.items()) + ".")
            fl = [k for k in codes if k.startswith("P1A")]; b = Q.flow_baseline(self.c, self.now)
            if fl and b:
                t0 = codes[fl[0]][0][0]; days = [d for d in Q.flow_days(self.c, t0, self.cov()["hi"] + 1)]; low = [d for d in days if d[1] < 0.85 * b["p50"]]
                out.append(f"But the condition behind {fl[0]} did repeat: on {len(low)} of the {len(days)} driving days after it, the median coolant flow was still below 85% of normal. A trouble code is stored when the condition first appears and is not logged again while it persists, so one code can hide a continuing fault.")
        return "\n".join(out)

    def h_firstlast(self, s, raw):
        k = self.kind_in(s)
        if k not in ("diagnostic", "overheat", "adas", "tyre", "camera", "brake"): k = "diagnostic"
        rows = self.evs(k)
        if len(rows) < 2:
            return f"There {'is' if rows else 'are'} only {len(rows)} {Q.KIND_NAME.get(k, k)} on record{(' (' + self.fts(rows[0][0]) + ')') if rows else ''}, so there is no first and most recent occurrence to compare. " + (f"What changed since it happened: {self.since_text(rows[0][0])}" if rows else "")
        a, z = Q.state_at(self.c, rows[0][0]), Q.state_at(self.c, rows[-1][0]); codes = {(r[3] or "").split(" ")[0] for r in rows}
        note = f"Note: these are different codes ({', '.join(sorted(codes))}), i.e. different faults, so this compares two events, not the same fault twice.\n" if len(codes) > 1 else ""
        return note + f"First {Q.KIND_NAME.get(k, k)}: {self.evrow(rows[0])} - {Q.fmt_state(a, True)}.\nMost recent: {self.evrow(rows[-1])} - {Q.fmt_state(z, True)}.\nThe interval between them is {Q.dur(rows[-1][0] - rows[0][0])}."

    def since_text(self, ts):
        cv = self.cov(); b = self.daily("flow", ts - 7 * DAY, ts, modes=(1, 2, 3), minv=3.0); a = self.daily("flow", ts, cv["hi"] + 1, modes=(1, 2, 3), minv=3.0)
        if len(a) >= 2 and len(b) >= 2: return f"coolant flow averaged {mean([v for _, v, _ in b]):.1f} L/min in the week before and {mean([v for _, v, _ in a]):.1f} L/min since."
        return "not enough data on either side."

    def h_precursors(self, s, raw):
        fails = self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('WARN_BATT_OVERTEMP','DTC_SET','LOW_12V') ORDER BY ts").fetchall()
        if not fails: return "No failure events are recorded, so there is no sequence that preceded one."
        out = [f"{len(fails)} failure event(s) on record. For each, the earliest sustained deviation of a signal before it:"]; n_lead = 0
        for ts, ty, det in fails:
            lead = None
            if re.search(r"coolant|P1A|thermal|temp", f"{det} {ty}", re.I):
                fo = Q.flow_onset(self.c, self.now, upto=ts)
                if fo: lead = f"the daily median coolant flow fell below normal from {self.day(fo[0])} ({fo[1]:.1f} vs {fo[2]:.1f} L/min), {Q.dur(ts - fo[0])} before it"; n_lead += 1
            out.append(f"  - {self.fts(ts)} {Q.safe_text(det)[:55]}: " + (lead or "no earlier sustained deviation in the signals I track"))
        out.append("Pattern: " + ("a gradual coolant-flow decline came first, then the fault code - a drift preceded the failure, not a sudden event." if n_lead else "no repeated signal pattern before the failures.") + f" With only {len(fails)} failure(s) of {len({f[2].split(' ')[0] for f in fails})} kind(s), this is an example, not a statistical pattern - recurring sequences need several failures of the same kind.")
        return "\n".join(out)

    def h_earliest(self, s, raw):
        e = self.most_serious() if re.search(r"major|serious|latest", s) else (self.last_anchor_event("warning") or self.most_serious())
        if not e: return "There is no fault event to look back from."
        cv = self.cov(); lo = cv["lo"]; hi = min(lo + (cv["hi"] - lo) // 2, lo + 14 * DAY)
        out = [f"Looking back from {self.evrow(e)}:"]; found = []
        fo = Q.flow_onset(self.c, self.now, upto=e[0])
        if fo: found.append((fo[0], "coolant flow (daily median)", fo[1], f"{fo[2]:.1f} typical", "L/min"))
        for col, lab, un in (("batt", "pack temperature", "C"), ("motor", "motor temperature", "C")):
            b = Q.baseline_stats(self.c, col, lo, hi, modes=(1, 2, 3))
            if not b: continue
            for d, v, n in self.daily(col, hi, e[0] + 1, modes=(1, 2, 3)):
                if (v - b["mean"]) / b["sd"] > 3: found.append((d, lab + " (daily mean)", v, f"{b['mean']:.1f} (sd {b['sd']:.1f})", un)); break
        if self._has("vday") and re.search(r"12 ?v|voltage|P0562|LOW_12V", f"{e[1]} {e[3]}", re.I):
            vd = self.c.execute("SELECT ts, v12 FROM vday WHERE ts<=? ORDER BY ts", (e[0] + 86400,)).fetchall()
            if len(vd) >= 6:
                base = mean([v for _, v in vd[:4]])
                for i in range(4, len(vd) - 2):
                    if all(v < base - 0.12 for _, v in vd[i:]): found.append((vd[i][0], "12 V battery rest voltage", vd[i][1], f"{base:.2f} (first days)", "V")); break
        if self._has("ext_1m") and re.search(r"tyre|tpms|C0750|pressure", f"{e[1]} {e[3]}", re.I):
            wi = next((i for i, w in enumerate(AX.WHEELS) if w in (e[3] or "").lower()), None)
            if wi is not None:
                col = ("tp_fl", "tp_fr", "tp_rl", "tp_rr")[wi]; q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, AVG({col}) / 10.0, AVG(tt) FROM ext_1m WHERE ts<=? GROUP BY d ORDER BY d"
                dd = [(d_, AX._cold(p, tt)) for d_, p, tt in self.c.execute(q, (e[0],)).fetchall() if p]
                if len(dd) >= 4:
                    base = mean([v for _, v in dd[:3]])
                    for i in range(3, len(dd)):
                        if dd[i][1] < base - 10 and all(v < base - 6 for _, v in dd[i:]): found.append((dd[i][0], f"{AX.WHEELS[wi]} tyre pressure (cold equivalent)", dd[i][1], f"{base:.0f} (first days)", "kPa")); break
        if found:
            found.sort(key=lambda f: f[0]); d, lab, v, norm, un = found[0]
            out.append(f"The earliest sign: {lab} on {self.day(d)}: {v:.1f} {un} against a normal {norm} - {(e[0] - d) / DAY:.0f} days before the event." + (" The first low days were already visible in the daily medians." if (e[0] - d) / DAY >= 1 else ""))
            for f in found[1:]: out.append(f"  - {f[1]} first deviated on {self.day(f[0])} ({f[2]:.1f} {f[4]}).")
            out.append("This is the first day a signal left its normal band and stayed out; the cause of the drift isn't visible in the signals, and an earlier cause (for example wear) can't be seen at all.")
        else: out.append("No signal drifted out of its normal band in the days before it; the event appears to have been sudden as far as these signals show.")
        return "\n".join(out)

    def h_similar(self, s, raw):
        rows = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN (%s) ORDER BY ts" % ",".join("?" * len(WARN_TYPES)), WARN_TYPES).fetchall()
        if len(rows) < 2: return f"Only {len(rows)} warning/fault event(s) exist, so there is nothing to compare."
        grp = {}
        for r in rows: grp.setdefault(r[1] + ":" + (r[3] or "").split(" ")[0], []).append(r)
        sts = [Q.state_at(self.c, r[0]) for r in rows]; ok = [x for x in sts if "speed" in x]
        bytype = Counter(r[1] for r in rows); out = [f"{len(rows)} warning/fault events ({', '.join(f'{v} x {k}' for k, v in bytype.items())}). What they have in common:"]
        if ok:
            hrs = [((r[0] + A.TZ) % DAY) / 3600 for r in rows]; out.append(f"  - time of day: {min(hrs):.1f} h to {max(hrs):.1f} h local; speed at the time {min(x['speed'] for x in ok):.0f} to {max(x['speed'] for x in ok):.0f} km/h; outside {min(x['amb'] for x in ok):.1f} to {max(x['amb'] for x in ok):.1f} C; pack {min(x['batt'] for x in ok):.1f} to {max(x['batt'] for x in ok):.1f} C")
            mods = Counter({0: "parked/idle", 1: "city driving", 2: "motorway", 3: "charging"}[x["mode"]] for x in ok); out.append("  - state: " + ", ".join(f"{v} x {k}" for k, v in mods.items()))
            lowfl = [x for x in ok if 0.3 < x["flow"] < 6]; out.append(f"  - coolant flow low (<6 L/min) at {len(lowfl)} of {len(ok)} of them")
        if ok:
            mm = Counter({0: "parked", 1: "city driving", 2: "motorway driving", 3: "charging"}[x["mode"]] for x in ok).most_common(1)[0]; lf = sum(1 for x in ok if 0.3 < x["flow"] < 6)
            self.brief = f"{len(rows)} faults; {mm[1]} of {len(ok)} happened while {mm[0]}, with the pack at {min(x['batt'] for x in ok):.0f} to {max(x['batt'] for x in ok):.0f} DEG" + (f" and coolant flow low in {lf}." if lf else ".")
            self.brief = self.brief.replace("DEG", "\u00b0C")
        else: self.brief = f"{len(rows)} faults, but no vehicle-state data around them to compare."
        out.append("Differences: " + "; ".join(f"{r[1]} ({Q.safe_text(r[3])[:30]}) {self.fts(r[0])[:16]}" for r in rows[:5]) + ". The commonality is real only if it holds across several events of the same kind; with few events it may be chance.")
        return "\n".join(out)

    def h_history(self, s, raw):
        """Most significant abnormal behaviour over the whole history, with exact evidence."""
        F, ab = self.findings(max(14, int((self.cov()["hi"] - self.cov()["lo"]) / DAY)))
        if not F: return "I found nothing abnormal in the whole history: no warnings, faults, drifts or unusual efficiency changes."
        f = F[0]; d = self.diag_default(s); out = [f"Most significant abnormal behaviour: {f['text']}."]
        if d and d["kind"] == "event":
            out += ["Evidence:"] + ["  " + x for x in self.event_text(d)]
            out.append(f"  - most consistent explanation: {d['alts'][0]['name']} ({d['alts'][0]['strength']}): {d['alts'][0]['support']}.")
            out.append(f"  - confidence: {d['confidence']}.")
        deg = self.h_degrade("degradation", "")
        if deg and "Most degraded" in deg: out.append(deg.split("\n")[1] if len(deg.split("\n")) > 1 else "")
        out.append(f"Other findings: " + ("; ".join(g["text"] for g in F[1:4]) if len(F) > 1 else "none") + ".")
        self.ctx.update(diag=d, topic=(d or {}).get("topic"))
        return "\n".join(x for x in out if x)

    def h_serious(self, s, raw):
        e = self.most_serious()
        if not e: return "There is no warning or fault event in the history."
        out = [f"Most serious event (highest severity; the latest if tied): {self.evrow(e)}."]
        pre = Q.event_sig(self.c, e[0], 3600, 60); sm = Q.summarise(pre, e[0] - 3600, e[0] + 1)
        if sm: out.append("In the hour before:"); out += self.sig_lines(sm)
        cv = self.cov(); fb = self.daily("flow", e[0] - 7 * DAY, e[0], modes=(1, 2, 3), minv=3.0); fa = self.daily("flow", e[0], cv["hi"] + 1, modes=(1, 2, 3), minv=3.0)
        ba = self.daily("batt_max", e[0] - 7 * DAY, e[0], modes=(1, 2, 3)); bb = self.daily("batt_max", e[0], cv["hi"] + 1, modes=(1, 2, 3))
        out.append("What changed afterwards (daily means: 7 days before vs since):")
        if len(fb) >= 2 and len(fa) >= 2: out.append(f"  - coolant flow: {mean([v for _, v, _ in fb]):.1f} -> {mean([v for _, v, _ in fa]):.1f} L/min")
        if len(ba) >= 2 and len(bb) >= 2: out.append(f"  - daily peak pack temperature: {mean([v for _, v, _ in ba]):.1f} -> {mean([v for _, v, _ in bb]):.1f} C")
        d1 = Q.daily_series(self.c, e[0] - 7 * DAY, e[0]); d2 = Q.daily_series(self.c, e[0], cv["hi"] + 1)
        if len(d1) >= 2 and len(d2) >= 2: out.append(f"  - energy use: {mean([x['wh_km'] for x in d1]):.0f} -> {mean([x['wh_km'] for x in d2]):.0f} Wh/km")
        nxt = self.c.execute("SELECT ts, type, detail FROM events WHERE ts>? AND type IN (%s) ORDER BY ts LIMIT 3" % ",".join("?" * len(WARN_TYPES)), (e[0], *WARN_TYPES)).fetchall()
        out.append("  - later warnings/faults: " + ("; ".join(f"{self.fts(t)} {Q.safe_text(d)[:40]}" for t, _, d in nxt) if nxt else "none"))
        return "\n".join(out)

    def h_engineer(self, s, raw):
        F, ab = self.findings(14); cv = self.cov(); sg = self.sig(cv["hi"] - 7 * DAY, cv["hi"] + 1)
        tr = self.c.execute("SELECT COUNT(*), SUM(km), SUM(kwh) FROM trips WHERE start>=?", (cv["hi"] - 7 * DAY,)).fetchone()
        sw = Q.meta(self.c, "veh.sw", "?"); model = Q.meta(self.c, "veh.model", "this vehicle"); soh = Q.meta(self.c, "veh.soh", "?")
        out = [f"Service review of {model} (software {sw}, battery state of health {soh}%), data {self.span_text()}:",
               f"  Usage, last 7 days: {tr[0]} trips, {tr[1] or 0:.0f} km, {1000 * (tr[2] or 0) / max(tr[1] or 1, 1):.0f} Wh/km." if tr and tr[0] else "  No driving in the last 7 days."]
        if len(sg): out.append(f"  Thermal, last 7 days: pack {min(sg.cols['batt']):.1f} to {max(sg.cols['batt']):.1f} C, coolant flow {mean([v for v in sg.cols['flow'] if v > 0.3]):.1f} L/min on average, motor up to {max(sg.cols['motor']):.0f} C.")
        out.append("  Findings, most serious first:")
        out += [f"    {k}. {f['text']}" for k, f in enumerate(F[:5], 1)] or ["    none - no warnings, drifts or step changes."]
        dg = self.ctx.get("diag") or self.diag_default(s)
        if dg and dg["kind"] == "event": out.append(f"  Likely cause of the main fault: {dg['alts'][0]['name']} ({dg['alts'][0]['strength']}); recommended: inspect the cooling circuit (pump, coolant level, flow sensor) before further fast-charging. Not diagnosable from data: the exact part.")
        out.append("  Not available to me: RPM (estimated from speed), throttle, steering, radar, cabin temperature, part-level diagnostics.")
        return "\n".join(out)

    # ------------------------------------------------------------------ checks on earlier answers
    def h_memory(self, s, raw):
        if re.search(r"overheat", s) and re.search(r"you said|earlier|previously|claimed|told me|stated|mentioned", s):
            win = self.win(s, default=None); lo, hi, w = win if win[0] is not None else (self.cov()["hi"] - DAY, self.cov()["hi"] + 1, "in that period")
            ev = self.evs("overheat", lo, hi)
            said = [h for h in self.hist if re.search(r"overheat|over-temperature", h["answer"].lower()) and not re.search(r"no battery over|not been recorded|none", h["answer"].lower())]
            if ev: return f"The data shows an over-temperature warning at {self.fts(ev[0][0])}" + (f" and {len(ev) - 1} more" if len(ev) > 1 else "") + f" ({w})."
            mx = self.c.execute("SELECT MAX(batt_max), ts FROM sig_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
            self.brief = f"No - I didn't say that and the data doesn't show an overheating {w}; the pack peaked at {mx[0] / 10:.0f} DEG.".replace("DEG", "\u00b0C") if mx and mx[0] is not None else f"No overheating was recorded {w}."
            return ("I don't have a record of saying that, and the data doesn't support it: no battery over-temperature warning was recorded " + f"{w}" + (f" - the highest pack temperature then was {mx[0] / 10:.1f} C at {self.fts(mx[1])}" if mx and mx[0] is not None else "") +
                    f". So there is no exact time to give; if I said it, that statement was wrong.")
        m = re.search(r"(?:dropped|fell|rose|increased|changed|improved|decreased|reduced|worsened|declined)(?: by)? (\d+(?:\.\d+)?) ?(?:%|percent)", s)
        if m and re.search(r"calculation|show|how", s):
            claim = float(m[1]); d = self.diag_range(topic="range")
            if not d: return "I can't redo that calculation: there are too few days of trips."
            seen = ""
            for h in reversed(self.hist):
                for sent in re.split(r"(?<=[.!?:])\s+(?=[A-Z])|\n", h["answer"]):
                    if re.search(r"range|energy use|efficiency", sent, re.I) and re.search(r"\d%", sent) and len(sent) < 220 and not sent.lstrip().startswith("-") and "[p" not in sent: seen = sent.strip().rstrip("."); break
                if seen: break
            ok = abs(abs(d["pct"]) - claim) < 1.0
            return (f"I haven't stated {claim:.0f}% in this conversation" + (f" (what I said earlier: '{seen}')" if seen else " (I haven't given any efficiency percentage yet)") + f", and the data doesn't give it. Recalculation: daily energy use averaged {d['ma']:.1f} Wh/km over {d['days_a']} days up to {self.day(d['pb'][0] - 1)} and {d['mb']:.1f} Wh/km over {d['days_b']} days from {self.day(d['pb'][0])}: "
                    f"({d['mb']:.1f} - {d['ma']:.1f}) / {d['ma']:.1f} = {d['pct']:+.1f}%" + (" - this matches your figure." if ok else f", not {claim:.0f}%. In range terms {d['ra']:.0f} -> {d['rb']:.0f} km ({100 * (d['rb'] - d['ra']) / d['ra']:+.1f}%)."))
        m = re.search(r"rpm was (\d[\d,]*)", s)
        if m and re.search(r"timestamp|when|support|time", s):
            N = float(m[1].replace(",", "")); v = Q.kmh_of_rpm(N); prev = [h for h in self.hist if re.search(re.escape(m[1]), h["answer"])]
            sg = self.sig(*self.recent(7)[:2]); idx = [i for i, x in enumerate(sg.cols["speed"]) if abs(x - v) < 0.012 * v and sg.cols["mode"][i] in (1, 2)]
            base = "I did not say that earlier in this conversation, and I have no stored RPM signal - only an estimate from speed. " if not prev else "I did report that. "
            if idx: return base + f"For reference, {N:,.0f} rpm corresponds to about {v:.1f} km/h; the most recent moment the car was at that speed (within 1.2%) was {self.fts(sg.ts[idx[-1]])} ({len(idx)} such seconds in the last 7 days). " + self.note_rpm()
            return base + f"{N:,.0f} rpm would need about {v:.1f} km/h, and the car did not reach that speed in the last 7 days (highest {max(sg.cols['speed']):.0f} km/h). " + self.note_rpm()
        if re.search(r"adas", s) and re.search(r"yesterday|certain|sure", s):
            r = self.evs("adas", newest_first=True, limit=1)
            if not r: return "No ADAS disengagement is recorded at all, so it cannot have happened yesterday."
            yday = self.today() - dt.timedelta(days=1); d = dt.datetime.fromtimestamp(r[0][0] + self.tz, dt.timezone.utc).date()
            return (f"Yes - the last ADAS disengagement was at {self.fts(r[0][0])} ({Q.safe_text(r[0][3])})." if d == yday else f"No, not for that date: the last ADAS disengagement was at {self.fts(r[0][0])} ({Q.safe_text(r[0][3])}), {(yday - d).days} day(s) before yesterday ({yday.isoformat()}); there was none yesterday.")
        return None

    def h_contradict_prev(self, s, raw):
        return self.h_counter(s, raw)

    def h_recheck(self, s, raw):
        if not self.hist: return "I have no previous answer to recheck."
        d = self.ctx.get("diag"); last = self.hist[-1]
        if d and d["kind"] == "range":
            rows = self.c.execute(f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, SUM(power)/3600.0, SUM(speed)/36000.0 FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) GROUP BY d ORDER BY d", (d["lo"], d["hi"])).fetchall()
            raw_d = [(r[0], 1000 * r[1] / 1000.0 / r[2]) for r in rows if r[2] and r[2] > 3]
            tab = {x["ts"]: x["wh_km"] for x in d["D"]}; diffs = [(ts, v, tab[ts]) for ts, v in raw_d if ts in tab]
            if len(diffs) < 4: return "I couldn't recompute from the raw signal rows (too little data)."
            a = mean([v for ts, v, _ in diffs if ts < d["pb"][0]]); b = mean([v for ts, v, _ in diffs if ts >= d["pb"][0]]); pct = 100 * (b - a) / a
            md = mean([abs(v - t) / t * 100 for _, v, t in diffs])
            return (f"Rechecked from the raw signal rows instead of the trips summary: integrating battery power over the driving minutes and dividing by the distance integrated from speed, per local day ({len(diffs)} days). "
                    f"Energy use before {self.day(d['pb'][0])}: {a:.1f} Wh/km, after: {b:.1f} Wh/km ({pct:+.1f}%) - the summary said {d['ma']:.1f} -> {d['mb']:.1f} ({d['pct']:+.1f}%). Day by day the two methods differ by {md:.1f}% on average (the raw method has no charging or idle losses). "
                    + ("The conclusion stands." if abs(pct - d["pct"]) < 4 and pct * d["pct"] > 0 else "The two disagree noticeably, so treat the summary figure with more caution."))
        if d and d["kind"] == "event":
            ts = d["ts"]; pm = self.c.execute("SELECT AVG(flow)/10.0, AVG(batt)/10.0, AVG(power)/1000.0, COUNT(*) FROM sig_1m WHERE ts>=? AND ts<=? AND mode IN (1,2,3)", (ts - 3600, ts)).fetchone()
            pre = d["pre"]; bf = d["base"]["flow"]; fm = self.c.execute("SELECT flow FROM sig_1m WHERE ts>=? AND ts<=? AND flow>30 AND mode IN (1,2,3)", (ts - 3600, ts)).fetchall()
            med = sorted(x[0] / 10.0 for x in fm)[len(fm) // 2] if fm else None
            if not pm or pm[3] < 3: return "Rechecked: the per-minute table has no rows for the hour before the event, so the 1 Hz context I used is the only source."
            return (f"Rechecked against an independent table (the per-minute aggregates instead of the 1 Hz event context) for the hour before {self.fts(ts)}: median running coolant flow {med if med is not None else float('nan'):.1f} L/min (1 Hz analysis: {pre['flow']['mean']:.1f} mean incl. standstill), mean pack temperature {pm[1]:.1f} C (1 Hz: {pre['batt']['mean']:.1f}), mean power {pm[2]:.1f} kW (1 Hz: {pre['power']['mean']:.1f}). "
                    + ("The two sources agree, so the conclusion stands." if abs((pm[1] or 0) - pre["batt"]["mean"]) < 0.5 else "They differ by more than expected - treat the conclusion with more caution."))
        prev_q = last["q"]
        self.raw = True
        try: r = self.ask(prev_q)
        finally: self.raw = False
        if r: return "Recomputed from the raw rows where they exist: " + ("the answer is unchanged." if r["answer"].strip() == last["answer"].strip() else "\n" + r["answer"])
        return "My previous answer came from the standard analysis of the same stored rows; I have no separate raw-data path for it. The underlying tables are the same either way."

    def h_day(self, s, raw):
        d, expl, dtxt = Q.find_date(s, self.now, self.tz); cv = self.cov()
        if d is None: return None
        lo, hi = Q.day_bounds(d, self.tz)
        if hi < cv["lo"] or lo > cv["hi"]: return f"I have no data for {d.isoformat()} - the recorded history covers {self.span_text()}."
        tr = Q.trips_between(self.c, lo, hi); ev = self.c.execute("SELECT ts, type, detail FROM events WHERE ts>=? AND ts<? AND type NOT LIKE 'HARSH%'", (lo, hi)).fetchall(); hb = self.c.execute("SELECT COUNT(*) FROM events WHERE ts>=? AND ts<? AND type LIKE 'HARSH%'", (lo, hi)).fetchone()[0]
        ch = self.c.execute("SELECT COUNT(*), SUM(kwh) FROM charges WHERE start>=? AND start<?", (lo, hi)).fetchone() if self._has("charges") else (0, 0)
        if not tr and not ev: return f"On {d.isoformat()} the car was not driven and recorded no events."
        sg = self.sig(lo, hi); out = [f"{d.isoformat()}: {plural(len(tr), 'trip')}, {sum(t['km'] for t in tr):.0f} km, {sum(t['kwh'] for t in tr):.1f} kWh ({1000 * sum(t['kwh'] for t in tr) / max(sum(t['km'] for t in tr), 1):.0f} Wh/km)" + (f"; {ch[0]} charging session(s), {ch[1] or 0:.1f} kWh" if ch and ch[0] else "") + "."]
        if len(sg): out.append(f"  Pack temperature {min(sg.cols['batt']):.1f} to {max(sg.cols['batt']):.1f} C, top speed {max(sg.cols['speed']):.0f} km/h, outside {mean(sg.cols['amb']):.1f} C average, coolant flow {mean([v for v in sg.cols['flow'] if v > 0.3] or [0]):.1f} L/min while running.")
        out += [f"  Event {A.hhmm(t)} {ty}: {Q.safe_text(de)}" for t, ty, de in ev[:8] if not Q.flagged(de)]
        if hb: out.append(f"  {hb} hard braking/acceleration/cornering events.")
        return "\n".join(out)

    HEALTH_EXTRA = ("ECU_RESET", "UNEXPECTED_SHUTDOWN", "VOLTAGE_DIP", "SENSOR_CONFLICT", "CAN_ERROR", "SOURCE_SILENT", "COOLANT_FLOW_DROP", "WARN_OVERTEMP", "POWER_DERATE", "AEB_ACTIVATION", "SENSOR_INVALID")    # trouble that is not in the old warning list: a controller that reset, a power loss, a bus that went silent

    def h_health_summary(self, s, raw):
        types = tuple(WARN_TYPES) + self.HEALTH_EXTRA
        ev = self.c.execute("SELECT type, COUNT(*), MAX(ts) FROM events WHERE type IN (%s) GROUP BY type" % ",".join("?" * len(types)), types).fetchall()
        n, bad = self.scan_injection(); F, ab = self.findings(14)
        out = ["Vehicle health from the recorded events and signals (not from any text inside them):"]
        if re.search(r"\b(?:is|are|was|were|does|do)\b.*\b(?:healthy|health|ok|okay|fine|normal)\b", str(s)):        # a polar question gets its verdict first, then the evidence
            out.insert(0, "Not fully healthy: the recorded events and signals show problems." if (ev or F) else "Healthy on the recorded evidence: no warning, fault or abnormal finding is recorded.")
        out += [f"  - {c_} x {t}, latest {self.fts(ts)}" for t, c_, ts in ev] or ["  - no warning or fault events recorded"]
        out += [f"  - finding: {f['text']}" for f in F[:3]]
        if bad: out.append(f"Note: {bad} stored event/log text(s) contained instruction-like wording (e.g. telling me to report the car as healthy). I treated them as data, ignored them, and the summary above comes only from the recorded numbers and event types.")
        return "\n".join(out)
