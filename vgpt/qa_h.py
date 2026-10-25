"""QA part H: more everyday questions - a week / month overview, "do I need to charge / can I make it", "is it safe to keep driving", the longest trip, "am I charging", driving style and a driving score,
what a trouble code means, battery age / capacity lost, top speed, favourite places, the commute, month-on-month. Short plain answer first; numbers only from the stored data."""
import re
from collections import Counter
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean

DAY = 86400
EVN = {"WARN_BATT_OVERTEMP": "battery over-temperature warning", "DTC_SET": "trouble code", "LOW_12V": "low 12 V battery warning", "TPMS_LOW": "tyre pressure warning"}
DTC_FAMILY = {"P": "powertrain (motor, battery, drive system)", "B": "body (comfort and safety systems)", "C": "chassis (brakes, steering, suspension)", "U": "network (communication between control units)"}


def hm(minutes):
    minutes = int(round(minutes))
    return f"{minutes} min" if minutes < 90 else f"{minutes // 60} h {minutes % 60:02d} min"


class MixH:
    def _trips(self, lo, hi):
        return self.c.execute("SELECT COUNT(*), COALESCE(SUM(km),0), COALESCE(SUM(kwh),0), COALESCE(SUM(end-start),0) FROM trips WHERE start>=? AND start<? AND km>0.2", (lo, hi)).fetchone()

    # ---------------------------------------------------------------- overview
    def h_overview(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.now - 7 * DAY, self.now + 1, "the last 7 days"))
        if lo is None or hi - lo < 2 * DAY: lo, hi, wtxt = self.now - 7 * DAY, self.now + 1, "the last 7 days"
        span = hi - lo; t = self._trips(lo, hi); p = self._trips(lo - span, lo)
        if not t[0]: return f"No trips were recorded in {wtxt}."
        wh = 1000 * t[2] / t[1]; ph = 1000 * p[2] / p[1] if p[1] > 1 else None
        ch = self.sessions(lo, hi); ev = self.c.execute("SELECT type, COUNT(*) FROM events WHERE ts>=? AND ts<? AND type IN ('WARN_BATT_OVERTEMP','DTC_SET','LOW_12V','TPMS_LOW') GROUP BY type", (lo, hi)).fetchall()
        nw = sum(n for _, n in ev); h = AX.health(self.c, self.now, "14"); bad = h.get("problems", 0) if h.get("ok") else 0
        cmp_ = f", {abs(100 * (wh - ph) / ph):.0f}% {'worse' if wh > ph else 'better'} than the period before" if ph and abs(100 * (wh - ph) / ph) >= 3 else ""
        self.brief = f"In {wtxt} you drove {t[1]:.0f} km in {t[0]} trips at {wh:.0f} Wh/km{cmp_}; " + (f"charged {len(ch)} time{'s' if len(ch) != 1 else ''}; " if ch else "") + (f"{nw} warning{'s' if nw != 1 else ''} to look at." if nw else "no warnings.")
        lines = [f"Driving: {t[1]:.0f} km, {t[0]} trips, {hm(t[3] / 60)} behind the wheel, {wh:.0f} Wh/km" + (f" (before: {ph:.0f})" if ph else "") + ".", f"Charging: {len(ch)} session{'s' if len(ch) != 1 else ''}, {sum(x[4] for x in ch):.0f} kWh added." if ch else "Charging: none recorded."]
        lines.append(("Warnings: " + ", ".join(f"{n} x {EVN.get(ty, ty.replace('_', ' ').lower())}" for ty, n in ev) + ".") if ev else "Warnings: none.")
        if bad: lines.append(f"Health: {bad} area{'s' if bad != 1 else ''} need attention - ask 'is my car healthy?'.")
        return "Overview of " + wtxt + ":\n" + "\n".join("  - " + l for l in lines)

    # ---------------------------------------------------------------- will I make it / do I need to charge
    def h_reach(self, s, raw):
        er = A.est_range(self.c, self.now)
        if not er.get("ok"): return "I can't estimate the range yet."
        rng = er["range_km"]; usable = rng * 0.9; m = re.search(r"(\d{2,4}) ?(?:km|kilomet|miles?|mi\b)", s); need = None; what = ""
        if m: need = float(m[1]) * (1.609 if re.search(r"miles?|\bmi\b", m[0]) else 1.0); what = f"{m[1]} {'miles' if re.search('mile|mi', m[0]) else 'km'}"
        elif re.search(r"\bhome\b", s) and self._has("places"):
            pl = self.c.execute("SELECT lat, lon FROM places WHERE kind='home' OR lower(name)='home' LIMIT 1").fetchone(); pos = AX.latest_fix(self.c, self.now)
            if pl and pos:
                need = AX.haversine_km((pos["lat"], pos["lon"]), (pl[0], pl[1])) * 1.3; what = f"home (about {need:.0f} km by road)"
                if need < 0.6: self.brief = f"You're already at home, with about {rng} km of range."; return self.brief
        if need is not None:
            ok = usable >= need; margin = rng - need
            self.brief = (f"Yes - about {rng} km of range against {what}: roughly {margin:.0f} km to spare." if ok else f"Not safely - about {rng} km of range, {what} needs more than the {usable:.0f} km I'd trust; charge first.") + " (An estimate at your recent consumption.)"
            return self.brief
        day = self.c.execute("SELECT SUM(km)/COUNT(DISTINCT start/86400) FROM trips WHERE start>=?", (self.now - 14 * DAY,)).fetchone()[0] or 40.0; days = rng / max(day, 1)
        low = er["soc"] < 25
        if re.search(r"do i need to charge|should i charge|need a charge|charge (now|soon|tonight)", s):
            self.brief = (f"Yes, soon - the battery is at {er['soc']:.0f}% (about {rng} km)." if low else f"Not yet - {er['soc']:.0f}% is about {rng} km, enough for {days:.0f} days at your usual {day:.0f} km a day.")
        else: self.brief = f"Estimated range is about {rng} km ({er['soc']:.0f}%) at your recent {er['wh_km']:.0f} Wh/km - roughly {days:.0f} days of your usual driving."
        return self.brief

    # ---------------------------------------------------------------- trips
    def h_longest_trip(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(0, self.now + 1, "in the whole history"))
        if lo is None: lo, hi, wtxt = 0, self.now + 1, "in the whole history"
        by = "end-start" if re.search(r"longest (trip )?(time|duration)|took the longest|most time", s) else "km"
        r = self.c.execute(f"SELECT start, end, km, kwh, avg_speed FROM trips WHERE start>=? AND start<? ORDER BY {by} DESC LIMIT 1", (lo, hi)).fetchone()
        if not r: return f"No trips {wtxt}."
        self.brief = f"Your longest trip {wtxt} was {r[2]:.0f} km on {self.day(r[0])}, taking {hm((r[1] - r[0]) / 60)} at an average {r[4]:.0f} km/h."
        return self.brief + f"\nIt used {r[3]:.1f} kWh ({1000 * r[3] / max(r[2], 0.1):.0f} Wh/km)."

    def h_last_trip(self, s, raw):
        r = self.c.execute("SELECT start, end, km, kwh, avg_speed, wh_km FROM trips WHERE end<=? AND km>0.2 ORDER BY end DESC LIMIT 1", (self.now,)).fetchone()
        if not r: return "No trips are recorded."
        fr = self.c.execute("SELECT p1.name, p2.name FROM trip_ext x LEFT JOIN places p1 ON p1.id=x.from_id LEFT JOIN places p2 ON p2.id=x.to_id WHERE x.start=?", (r[0],)).fetchone() if self._has("trip_ext") else None
        route = f" from {fr[0]} to {fr[1]}" if fr and fr[0] and fr[1] else ""
        self.brief = f"Your last trip was {r[2]:.1f} km{route} on {self.day(r[0])}, {hm((r[1] - r[0]) / 60)}, using {r[3]:.1f} kWh ({r[5]:.0f} Wh/km)."
        return self.brief

    def h_charging_now(self, s, raw):
        st = Q.state_at(self.c, self.now)
        if st.get("gap"): return "The car is switched off and not charging; current charging power is 0 kW. There is no live battery sample at this moment."
        er = A.est_range(self.c, self.now)
        if st.get("mode") == 3 and st.get("power", 0) < -0.5:
            self.brief = f"Yes - charging at {abs(st['power']):.0f} kW, battery {st['soc']:.0f}%."
        else:
            ss = self.sessions(0, self.now + 1); last = f" Last charge ended {Q.dur(self.now - ss[-1][1])} ago." if ss else ""
            self.brief = f"No, it isn't charging or plugged in right now; battery {er['soc']:.0f}%." + last if er.get("ok") else "No, it isn't charging."
        return self.brief

    # ---------------------------------------------------------------- driving style
    def _style(self, lo, hi):
        if not self._has("trip_ext"): return None
        r = self.c.execute("SELECT COALESCE(SUM(t.km),0), COALESCE(SUM(x.harsh_brake),0), COALESCE(SUM(x.harsh_accel),0), COALESCE(SUM(x.harsh_corner),0), COALESCE(SUM(x.regen_kwh),0), COALESCE(SUM(t.kwh),0) FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<?", (lo, hi)).fetchone()
        if r[0] < 20: return None
        return dict(km=r[0], brake=100 * r[1] / r[0], accel=100 * r[2] / r[0], corner=100 * r[3] / r[0], regen=100 * r[4] / (r[4] + r[5]) if r[4] + r[5] > 0 else 0)

    def _score(self, st, fast_share):
        sc = 100 - 2.5 * max(0, st["brake"] - 3) - 2.5 * max(0, st["accel"] - 2) - 3 * max(0, st["corner"] - 1) - 0.3 * max(0, fast_share - 10)
        return max(0, min(100, sc))

    def h_style(self, s, raw):
        st = self._style(self.now - 14 * DAY, self.now + 1)
        if not st: return "There is not enough recent driving, or no hard-driving record, to judge your style."
        sg = Q.load(self.c, self.now - 14 * DAY, self.now + 1, prefer="1m"); mv = [v for v, m in zip(sg.cols["speed"], sg.cols["mode"]) if m in (1, 2) and v > 5] if len(sg) else []
        fast = 100 * sum(1 for v in mv if v > 110) / len(mv) if mv else 0.0; sc = self._score(st, fast)
        grade = "smooth" if sc >= 85 else "fairly smooth" if sc >= 70 else "a bit hard" if sc >= 50 else "aggressive"
        wantc = "braking" if re.search(r"brak", s) else "acceleration" if re.search(r"accelerat", s) else None
        if wantc == "braking": self.brief = f"{'Yes, a bit' if st['brake'] > 6 else 'No'} - you brake hard {st['brake']:.0f} times per 100 km (typical is 2-6)."
        elif wantc == "acceleration": self.brief = f"{'Yes, a bit' if st['accel'] > 4 else 'No'} - you accelerate hard {st['accel']:.0f} times per 100 km (typical is 1-3)."
        else: self.brief = f"Driving score {sc:.0f} out of 100 - {grade}."
        return "\n".join([self.brief, f"Last 14 days, {st['km']:.0f} km: hard braking {st['brake']:.1f}, hard acceleration {st['accel']:.1f}, hard cornering {st['corner']:.1f} (all per 100 km); {fast:.0f}% of moving time above 110 km/h; regeneration recovers {st['regen']:.0f}% of the energy used.",
                          "The score is my own simple measure (starts at 100, loses points for hard events and sustained high speed), not an industry standard."])

    def h_improve_score(self, s, raw):
        st = self._style(self.now - 14 * DAY, self.now + 1)
        if not st: return "There is not enough recent driving, or no hard-driving record, to give evidence-based score improvements."
        sg = Q.load(self.c, self.now - 14 * DAY, self.now + 1, prefer="1m")
        mv = [v for v, m in zip(sg.cols["speed"], sg.cols["mode"]) if m in (1, 2) and v > 5] if len(sg) else []
        fast = 100 * sum(1 for v in mv if v > 110) / len(mv) if mv else 0.0
        score = self._score(st, fast); actions = []
        ranked = sorted(((max(0, st["brake"] - 3) * 2.5, f"Brake earlier and more progressively; hard braking is {st['brake']:.1f} per 100 km."),
                         (max(0, st["accel"] - 2) * 2.5, f"Accelerate more smoothly; hard acceleration is {st['accel']:.1f} per 100 km."),
                         (max(0, st["corner"] - 1) * 3.0, f"Enter bends more gently; hard cornering is {st['corner']:.1f} per 100 km."),
                         (max(0, fast - 10) * 0.3, f"Reduce sustained high-speed driving; {fast:.0f}% of moving time was above 110 km/h.")), reverse=True)
        actions = [txt for penalty, txt in ranked if penalty > 0]
        if not actions: actions = ["Keep the same smooth inputs; none of the measured score components is currently losing meaningful points."]
        self.brief = f"Your project-defined driving score is {score:.0f}/100. The largest measured opportunity is: {actions[0]}"
        return "\n".join([self.brief] + [f"  - {x}" for x in actions[:4]] + ["This is a transparent VehicleGPT project score, not an industry-standard or safety rating."])

    def h_drive_modes(self, s, raw):
        # drive_mode in telemetry describes observed operating state. It is not
        # proof of the selectable modes fitted to the vehicle.
        try: meta = dict(self.c.execute("SELECT k, v FROM meta WHERE k LIKE 'veh.%'").fetchall())
        except Exception: meta = {}
        modes = [x.strip() for x in meta.get("veh.selectable_drive_modes", "").split(",") if x.strip()]
        if modes:
            return f"This vehicle profile lists {len(modes)} selectable driving mode{'s' if len(modes) != 1 else ''}: {', '.join(modes)}."
        return ("The vehicle profile does not record its selectable driving-mode configuration, so I cannot give a verified count. "
                "The telemetry does label four operating states - parked, city driving, highway driving and charging - but those are observed states, not evidence of four driver-selectable modes.")

    def h_city_highway(self, s, raw):
        d = A.compare_modes(self.c, self.now, "a")
        base = ("City driving usually has lower speeds and more stopping, while highway driving is steadier and faster. "
                "In an EV, repeated slowing can recover some energy through regeneration, while higher highway speed usually increases aerodynamic demand.")
        if not d.get("ok"): return base + " This vehicle does not have enough comparable city and highway history for a reliable numeric comparison."
        direction = "more" if d["highway"] > d["city"] else "less"
        self.brief = (f"{base} In this vehicle's recorded history, energy consumption was {d['city']:.0f} Wh/km in city driving and "
                      f"{d['highway']:.0f} Wh/km on highways ({abs(d['diff_pct']):.0f}% {direction} energy per km on the highway).")
        return self.brief + f"\nThe comparison uses {d['minutes']} per-minute samples with matching filters. It is descriptive, not proof that road type caused the difference."

    def h_style_trend(self, s, raw):
        a = self._style(self.now - 14 * DAY, self.now - 7 * DAY); b = self._style(self.now - 7 * DAY, self.now + 1)
        if not a or not b: return "There is not enough driving in both weeks to compare."
        sa, sb = self._score(a, 0), self._score(b, 0); d = sb - sa
        self.brief = ("About the same as the week before" if abs(d) < 3 else f"Yes - slightly {'worse' if d < 0 else 'better'} than the week before") + f": score {sb:.0f} against {sa:.0f} (hard braking {b['brake']:.0f} against {a['brake']:.0f} per 100 km)."
        return self.brief

    # ---------------------------------------------------------------- codes, battery age, top speed
    def h_dtc_explain(self, s, raw):
        m = re.search(r"\b([pbcu][0-9a-f]{4})\b", s)
        rows = self.c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' ORDER BY ts DESC").fetchall()
        if m: code = m[1].upper(); hit = [r for r in rows if code in (r[1] or "").upper()]
        else: hit = rows[:1]; code = None
        if not hit and m: return f"{m[1].upper()} is a {DTC_FAMILY.get(m[1][0].upper(), 'diagnostic')} code. It does not appear in this car's history."
        if not hit: return "No trouble code is stored in this car's history."
        det = Q.safe_text(hit[0][1]); code = code or (re.search(r"\b[PBCU][0-9A-F]{4}\b", det) or [None])[0]; fam = DTC_FAMILY.get(code[0], "diagnostic") if code else "diagnostic"
        txt = re.sub(r"\s*[(].*?[)]", "", det); txt = re.sub(r"^[PBCU][0-9A-F]{4}\s*", "", txt)
        illustrative = bool(re.search(r"illustrative|simulat", det, re.I))
        self.brief = f"{code or 'The code'} is a {fam.split(' (')[0]} fault: {txt.lower()}. It was set {len(hit)} time{'s' if len(hit) != 1 else ''}, last on {self.day(hit[0][0])}."
        if illustrative: self.brief += " This is an illustrative record from the simulated demo history, not a confirmed production-vehicle fault."
        try:
            d = self.diag_event((hit[0][0], "DTC_SET", "warning", hit[0][1])) if re.search(r"coolant|pack|batt", det, re.I) else None
            if d: self.brief += f" Likely reason: {re.sub(r' [(].*?[)]', '', d['alts'][0]['name'])}."
        except Exception: pass
        return self.brief

    def h_battery_age(self, s, raw):
        r = self.c.execute("SELECT odo, soh FROM vday ORDER BY ts").fetchall() if self._has("vday") else []
        if not r: return "Battery age and capacity are not recorded for this car."
        odo, soh = r[-1]; lost = 100 - soh
        self.brief = f"The battery holds {soh:.0f}% of its original capacity ({lost:.0f}% lost) at {odo:,.0f} km. I don't know its calendar age - the car doesn't record that."
        self.brief += " This short history cannot support a reliable remaining-life forecast because calendar age, temperature, charging pattern and non-linear ageing are not all known."
        return self.brief

    def h_top_speed(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.now - 7 * DAY, self.now + 1, "in the last 7 days"))
        if lo is None: lo, hi, wtxt = self.now - 7 * DAY, self.now + 1, "in the last 7 days"
        r = self.c.execute("SELECT MAX(speed), ts FROM sig_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
        if not r or r[0] is None: return "No speed data in that period."
        self.brief = f"Your top speed {wtxt} was {r[0] / 10:.0f} km/h on {self.day(r[1])}."
        return self.brief + "\nThat is the highest recorded, not the car's rated maximum."

    # ---------------------------------------------------------------- places
    def h_places_most(self, s, raw):
        if not self._has("trip_ext"): return "No saved places are recorded for this car."
        if re.search(r"charg", s):
            r = self.c.execute("SELECT p.name, COUNT(*) FROM charges c JOIN places p ON p.id=c.place GROUP BY p.name ORDER BY 2 DESC LIMIT 3").fetchall() if self._has("charges") else []
            if not r: return "Charging locations are not recorded."
            self.brief = "You charge mostly at " + ", ".join(f"{n} ({k}x)" for n, k in r) + "."; return self.brief
        r = self.c.execute("SELECT p.name, COUNT(*) FROM trip_ext x JOIN places p ON p.id=x.to_id GROUP BY p.name ORDER BY 2 DESC LIMIT 4").fetchall()
        if not r: return "No saved places are recorded for this car."
        self.brief = "You go most often to " + ", ".join(f"{n} ({k} trips)" for n, k in r) + "."; return self.brief

    def h_commute(self, s, raw):
        if not self._has("trip_ext"): return "No saved places are recorded, so I can't tell which trip is your commute."
        r = self.c.execute("SELECT x.from_id, x.to_id, COUNT(*) n FROM trip_ext x GROUP BY 1,2 ORDER BY n DESC LIMIT 1").fetchone()
        if not r: return "Not enough trips to find a regular route."
        rows = self.c.execute("SELECT t.end-t.start, t.km, t.start FROM trips t JOIN trip_ext x ON x.start=t.start WHERE x.from_id=? AND x.to_id=?", (r[0], r[1])).fetchall()
        nm = dict(self.c.execute("SELECT id, name FROM places").fetchall()); d = sorted(x[0] for x in rows)
        self.brief = f"Your most regular trip is {nm.get(r[0], '?')} to {nm.get(r[1], '?')}: {r[2]} times, usually {hm(d[len(d) // 2] / 60)} for {mean([x[1] for x in rows]):.0f} km."
        return self.brief

    def h_leave_time(self, s, raw):
        if not self._has("trip_ext"): return "No saved places are recorded."
        h = self.c.execute("SELECT id FROM places WHERE kind='home' OR lower(name)='home' LIMIT 1").fetchone()
        if not h: return "I don't know where home is."
        r = self.c.execute("SELECT t.start FROM trips t JOIN trip_ext x ON x.start=t.start WHERE x.from_id=? AND t.start<=? ORDER BY t.start DESC LIMIT 1", (h[0], self.now)).fetchone()
        if not r: return "No trip from home is recorded."
        self.brief = f"You last left home on {self.day(r[0])} at {A.hhmm(r[0])}."; return self.brief

    # ---------------------------------------------------------------- months
    def h_compare_months(self, s, raw):
        cv = self.cov(); hi = cv["hi"] + 1; span = hi - cv["lo"]
        if span >= 58 * DAY: p2 = (hi - 30 * DAY, hi); p1 = (hi - 60 * DAY, hi - 30 * DAY); note = "the last 30 days with the 30 before"
        else: half = (span // DAY // 2) * DAY; p2 = (hi - half, hi); p1 = (hi - 2 * half, hi - half); note = f"the latest {half // DAY} days with the {half // DAY} before (the history is only {span // DAY} days long)"
        rows = self.period_rows(p1, p2)
        if not rows: return "There is not enough data in both periods."
        self.ctx.update(compare=(p1, p2)); lines = self.fmt_rows(rows, top=6)
        sg = [r for r in rows if (lambda w: w and w[2] < 0.05)(Q.welch(r[2], r[3]))]
        self.brief = f"I compared {note}: " + (f"{len(sg)} of {len(rows)} measures changed clearly." if sg else "nothing changed beyond normal day-to-day variation.")
        return "\n".join([f"Comparing {note}:"] + lines)

    def h_trip_count(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.now - 7 * DAY, self.now + 1, "in the last 7 days"))
        if lo is None: lo, hi, wtxt = self.now - 7 * DAY, self.now + 1, "in the last 7 days"
        t = self._trips(lo, hi)
        if not t[0]: self.brief = f"No trips {wtxt}."; return self.brief
        self.brief = f"{t[0]} trip{'s' if t[0] != 1 else ''} {wtxt}: {t[1]:.0f} km in total, {hm(t[3] / 60)} driving."
        return self.brief

    def h_regen(self, s, raw):
        if not self._has("trip_ext"): return "Regeneration (energy recovered when slowing down) is not recorded for this vehicle."
        lo, hi, wtxt = self.win(s, default=(self.now - 7 * DAY, self.now + 1, "in the last 7 days"))
        if lo is None: lo, hi, wtxt = self.now - 7 * DAY, self.now + 1, "in the last 7 days"
        r = self.c.execute("SELECT COALESCE(SUM(x.regen_kwh),0), COALESCE(SUM(t.kwh),0), COALESCE(SUM(t.km),0) FROM trips t JOIN trip_ext x ON x.start=t.start WHERE t.start>=? AND t.start<?", (lo, hi)).fetchone()
        if r[2] < 1: return f"No driving {wtxt}, so no regeneration to report."
        share = 100 * r[0] / (r[0] + r[1]) if r[0] + r[1] > 0 else 0
        self.brief = f"Regenerative braking recovered {r[0]:.1f} kWh {wtxt} - about {share:.0f}% of the energy you used, worth roughly {1000 * r[0] / r[2] * 1.0:.0f} Wh per km driven."
        return self.brief

    def h_motor_now(self, s, raw):
        r = self.c.execute("SELECT motor FROM sig_1m WHERE ts<=? ORDER BY ts DESC LIMIT 1", (self.now,)).fetchone(); mx = self.c.execute("SELECT MAX(motor), ts FROM sig_1m WHERE ts>=? AND ts<=?", (self.now - 14 * DAY, self.now)).fetchone()
        if not r or r[0] is None: return "The motor temperature is not recorded for this vehicle."
        now_c, peak = r[0] / 10.0, (mx[0] or 0) / 10.0
        verdict = "well within its normal range" if peak < 110 else "hot at times"
        prefix = "This EV has no combustion engine; its traction motor" if re.search(r"\bengine\b", raw.lower()) else "The motor"
        self.brief = f"{prefix} is at {now_c:.0f} C now; its hottest in the last 14 days was {peak:.0f} C - {verdict} (normal operation stays below about 110 C)."
        return self.brief

    def h_compare_days(self, s, raw):
        names = {"today": 0, "yesterday": 1, "the day before": 2}
        found = [(m.start(), m[0]) for m in re.finditer(r"today|yesterday|the day before", s)]
        a, b = (names[found[0][1]], names[found[1][1]]) if len(found) >= 2 else (0, 1)
        last = (self.now // DAY) * DAY
        def day(k):
            lo = last - k * DAY - A.TZ % DAY; r = self.c.execute("SELECT COUNT(*), COALESCE(SUM(km),0), COALESCE(SUM(kwh),0) FROM trips WHERE start>=? AND start<? AND km>0.2", (lo, lo + DAY)).fetchone()
            return (r[0], r[1], 1000 * r[2] / r[1] if r[1] > 0.5 else None)
        x, y = day(a), day(b); lab = {0: "Today", 1: "Yesterday", 2: "The day before"}
        if not x[0] and not y[0]: return "There are no trips on either of those days."
        f = lambda d, k: f"{lab[k]}: " + (f"{d[1]:.0f} km in {d[0]} trips at {d[2]:.0f} Wh/km" if d[0] else "no trips")
        cmp_ = ""
        if x[2] and y[2] and abs(x[2] - y[2]) / y[2] >= 0.03: cmp_ = f" - {abs(100 * (x[2] - y[2]) / y[2]):.0f}% {'worse' if x[2] > y[2] else 'better'} efficiency"
        fy = f(y, b); self.brief = f(x, a) + "; " + fy[0].lower() + fy[1:] + cmp_ + "."
        return self.brief
