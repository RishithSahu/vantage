"""QA part B: periods and durations, statistics, comparisons, trends, degradation, correlations and driving conditions."""
import re, math, datetime as dt
from collections import Counter
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd, F1, F0, SIGNED, plural, pfmt
from .qa_a import UNIT, NAME, fmt_val, WARN_TYPES

DAY = 86400


def runs(sg, pred, gap=None):
    """Contiguous runs of samples satisfying pred(i): [(start_ts, end_ts, i0, i1)]; a gap in the recording longer than `gap` seconds ends a run."""
    gap = gap or (2.5 * sg.res); out = []; i0 = None; n = len(sg.ts)
    for i in range(n):
        ok = pred(i)
        if ok and i0 is not None and sg.ts[i] - sg.ts[i - 1] > gap: out.append((sg.ts[i0], sg.ts[i - 1] + sg.res, i0, i - 1)); i0 = None
        if ok and i0 is None: i0 = i
        elif not ok and i0 is not None: out.append((sg.ts[i0], sg.ts[i - 1] + sg.res, i0, i - 1)); i0 = None
    if i0 is not None: out.append((sg.ts[i0], sg.ts[n - 1] + sg.res, i0, n - 1))
    return out


class MixB:
    def daily(self, col, lo, hi, modes=(1, 2), minv=None, scale=10.0, agg="AVG", minn=20):
        q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, {agg}({col}), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? AND mode IN ({','.join(str(m) for m in modes)})" + (f" AND {col}>{int(minv * scale)}" if minv is not None else "") + " GROUP BY d ORDER BY d"
        return [(d, v / scale, n) for d, v, n in self.c.execute(q, (lo, hi)).fetchall() if v is not None and n >= minn]

    def sigwin(self, s, default_days=7):
        lo, hi, wtxt = self.win(s, default=(None, None, None))
        if lo is None: lo, hi, wtxt = self.recent(default_days)
        hi = min(hi, self.cov()["hi"] + 1)
        sg = self.sig(lo, hi)
        return sg, lo, hi, wtxt

    def tier_note(self, sg): return "" if sg.res == 1 else " (older than 7 days: per-minute averages, so short peaks are smoothed)"

    # ------------------------------------------------------------------ longest periods, idling
    def h_longest(self, s, raw):
        m = re.search(r"(?:below|under|less than|slower than|<) ?(\d+) ?(?:km|kph)", s); thr = float(m[1]) if m else 10.0
        idle = bool(re.search(r"idl", s)); sg, lo, hi, wtxt = self.sigwin(s)
        if len(sg) < 5: return f"There is no signal data for {wtxt}."
        sp, md = sg.cols["speed"], sg.cols["mode"]
        pred = (lambda i: md[i] in (0, 1, 2) and sp[i] < 1.0) if idle else (lambda i: sp[i] < thr)
        R = runs(sg, pred)
        if not R: return f"The vehicle was never {'idle' if idle else f'below {thr:.0f} km/h'} in {wtxt}."
        top = max(R, key=lambda r: r[1] - r[0]); kind = lambda r: ("charging" if mean([md[j] for j in range(r[2], r[3] + 1)]) > 2.5 else ("ignition on, standing still" if mean([md[j] for j in range(r[2], r[3] + 1)]) < 0.5 else "driving/in traffic"))
        out = []
        if idle:
            out.append(f"The longest idle period in {wtxt} (car on, not moving, not charging) lasted {Q.dur(top[1] - top[0])}: {self.fts(top[0])} to {self.fts(top[1])}.")
        else:
            out.append(f"The longest continuous period below {thr:.0f} km/h in {wtxt} lasted {Q.dur(top[1] - top[0])}: {self.fts(top[0])} to {self.fts(top[1])} ({kind(top)}).")
            nc = [r for r in R if kind(r) != "charging"]
            if kind(top) == "charging" and nc:
                t2 = max(nc, key=lambda r: r[1] - r[0]); out.append(f"Excluding charging, the longest was {Q.dur(t2[1] - t2[0])}: {self.fts(t2[0])} to {self.fts(t2[1])} ({kind(t2)}).")
        la, lo_, pl = self.where(top[0])
        if la: out.append(f"Where: {la:.4f}, {lo_:.4f}" + (f" ({pl})" if pl else "") + ".")
        out.append(f"({len(R)} such periods in total; {sg.tier} data, recording is continuous only while the car is on or charging.)")
        self.ctx.update(anchor_ts=top[0], idle_run=top)
        return "\n".join(out)

    def h_idle_before(self, s, raw):
        r = self.h_longest("longest idle period", "")
        top = self.ctx.get("idle_run")
        if not top: return r
        ts = top[0]; sg = Q.event_sig(self.c, ts, 180, 5); sm = Q.summarise(sg, ts - 180, ts + 1)
        out = [r.split("\n")[0], f"Immediately before it (the 3 minutes to {A.hhmm(ts)}, {sg.tier}):"]
        if sm: out += self.sig_lines(sm)
        sp = [v for t, v in zip(sg.ts, sg.cols["speed"]) if t <= ts]
        if sp and max(sp) > 5: out.append(f"The car was driving (up to {max(sp):.0f} km/h) and slowed to a stop - it had just arrived or stopped.")
        ev = self.c.execute("SELECT ts, type, detail FROM events WHERE ts BETWEEN ? AND ? AND type NOT LIKE 'HARSH%'", (ts - 300, ts + 5)).fetchall()
        if ev: out.append("Events: " + "; ".join(f"{A.hhmm(t)} {ty} {Q.safe_text(d)[:50]}" for t, ty, d in ev))
        return "\n".join(out)

    def h_duration(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        if len(sg) < 5: return f"There is no signal data for {wtxt}."
        sp, md = sg.cols["speed"], sg.cols["mode"]; res = sg.res
        mr = re.search(r"(\d[\d,]*) ?rpm", s) or re.search(r"rpm (?:was |were |is |stayed |went )?(?:exceed\w*|above|over|greater than|higher than|more than|>) ?(\d[\d,]*)", s); mk = re.search(r"(?:above|over|exceed\w*|faster than|more than|greater than|>) ?(\d+) ?(?:km|kph)", s)
        if mr: thr_kmh = Q.kmh_of_rpm(float(mr[1].replace(",", ""))); label = f"above {mr[1]} rpm (about {thr_kmh:.0f} km/h)"; extra = " " + self.note_rpm()
        elif mk: thr_kmh = float(mk[1]); label = f"above {thr_kmh:.0f} km/h"; extra = ""
        else: thr_kmh = None
        moving = sum(res for i in range(len(sp)) if md[i] in (1, 2) and sp[i] > 1.0)
        if re.search(r"idl", s):
            idle = sum(res for i in range(len(sp)) if md[i] in (0, 1, 2) and sp[i] < 1.0); chg = sum(res for i in range(len(sp)) if md[i] == 3)
            self.brief = f"The car idled for {idle / 60:.0f} minutes in {wtxt} - {100 * idle / max(idle + moving, 1):.0f}% of the time it was on and not charging."
            return f"In {wtxt} the vehicle spent {idle / 60:.0f} minutes idling (car on, standing still, not charging), against {moving / 60:.0f} minutes moving and {chg / 60:.0f} minutes charging - idling was {100 * idle / max(idle + moving, 1):.0f}% of the time the car was on and not charging. ({sg.tier} data.)"
        if re.search(r"percent|share|proportion|%", s) and thr_kmh is not None and not mr:
            ab = sum(res for i in range(len(sp)) if sp[i] > thr_kmh and md[i] in (1, 2))
            return f"In {wtxt}, {100 * ab / max(moving, 1):.1f}% of the driving time (the {moving / 60:.0f} minutes the car was moving) was spent above {thr_kmh:.0f} km/h - {ab / 60:.0f} minutes. ({sg.tier} data.)"
        if re.search(r"median|average|mean", s) and re.search(r"accelerat", s):
            ac = sg.accel(); sel = [Q.rpm_of(sp[i]) for i in range(len(sp)) if ac[i] > 0.3 and sp[i] > 1]
            if not sel: return "There is no 1 Hz acceleration data to analyse."
            sel.sort(); return f"During periods of acceleration (more than 0.3 m/s2, {len(sel)} s in {wtxt}) the median estimated motor speed was {sel[len(sel) // 2]:,.0f} rpm (average {mean(sel):,.0f}). " + self.note_rpm()
        if thr_kmh is None: return None
        R = runs(sg, lambda i: sp[i] > thr_kmh); tot = sum(r[1] - r[0] for r in R)
        if re.search(r"calculate|average speed|mean speed", s) and re.search(r"average speed|mean speed", s):
            v = [sp[i] for i in range(len(sp)) if sp[i] > thr_kmh]
            if not v: return f"The vehicle was never {label} in {wtxt}."
            return f"During the periods when the vehicle was {label} in {wtxt} ({plural(len(R), 'period')}, {tot / 60:.0f} minutes) its average speed was {mean(v):.1f} km/h. " + ("This is nearly circular - RPM is computed from speed, so 'RPM above N' simply means 'speed above a threshold'. " + self.note_rpm() if mr else "")
        if not R: return f"The vehicle was never {label} in {wtxt}; the highest was {max(sp):.0f} km/h ({Q.rpm_of(max(sp)):,.0f} rpm est.)." + (extra if mr else "")
        top = max(R, key=lambda r: r[1] - r[0])
        return (f"The vehicle was {label} for {Q.dur(tot)} in total over {wtxt} ({plural(len(R), 'separate period')}); longest single stretch {Q.dur(top[1] - top[0])} starting {self.fts(top[0])}. That is {100 * tot / max(moving, 1):.0f}% of the time it was moving. ({sg.tier} data.)" + (extra if mr else ""))

    def h_rpm_after(self, s, raw):
        mr = re.search(r"(\d[\d,]*) ?rpm", s) or re.search(r"rpm (?:was |were |is |stayed |went )?(?:exceed\w*|above|over|greater than|higher than|more than|>) ?(\d[\d,]*)", s)
        if not mr: return None
        N = float(mr[1].replace(",", "")); thr = Q.kmh_of_rpm(N); sg, lo, hi, wtxt = self.sigwin(s)
        sp = sg.cols["speed"]; cross = [i for i in range(1, len(sp)) if sp[i - 1] <= thr < sp[i] and sg.ts[i] - sg.ts[i - 1] <= 2.5 * sg.res]
        if not cross: return f"The estimated motor speed never exceeded {N:,.0f} rpm (about {thr:.0f} km/h) in {wtxt}; the highest was {Q.rpm_of(max(sp)):,.0f} rpm at {max(sp):.0f} km/h. " + self.note_rpm()
        i = cross[-1]; ts = sg.ts[i]; sg2 = Q.event_sig(self.c, ts, 5, 60); sm = Q.summarise(sg2, ts, ts + 61)
        out = [f"It last crossed {N:,.0f} rpm (about {thr:.0f} km/h) at {self.fts(ts)}; it did so {plural(len(cross), 'time')} in {wtxt}. In the 60 seconds after:"]
        out += self.sig_lines(sm)
        out.append(self.note_rpm()); return "\n".join(out)

    # ------------------------------------------------------------------ trips
    def h_trip(self, s, raw):
        r = self.c.execute("SELECT start, end, km, kwh, wh_km, avg_speed, hwy_share, avg_amb FROM trips WHERE end<=? ORDER BY end DESC LIMIT 1", (self.now,)).fetchone()
        if not r: return "No trip is recorded."
        st, en, km, kwh, whk, av, hw, amb = r; sg = Q.load(self.c, st, en + 1); sp = [v for v in sg.cols["speed"] if v > 1] if len(sg) else []
        la, lo_, pl = self.where(st); la2, lo2, pl2 = self.where(en - 5)
        out = [f"Last trip: {self.fts(st)} to {A.hhmm(en)} ({Q.dur(en - st)}), {km:.1f} km, {kwh:.2f} kWh ({whk:.0f} Wh/km)" + (f", from {pl} to {pl2}" if pl and pl2 else "") + "."]
        out.append(f"Average speed {av:.1f} km/h over the whole trip (including stops)" + (f"; {mean(sp):.1f} km/h while moving, top speed {max(sp):.0f} km/h" if sp else "") + f"; {100 * hw:.0f}% of it on the motorway; outside {amb:.1f} C.")
        self.ctx.update(trip=r); return "\n".join(out)

    # ------------------------------------------------------------------ simple statistics
    def h_speedwin(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s, 1)
        if len(sg) < 5: return f"There is no signal data for {wtxt}."
        sp, md = sg.cols["speed"], sg.cols["mode"]; mv = [sp[i] for i in range(len(sp)) if md[i] in (1, 2) and sp[i] > 1]
        if not mv: return f"The car did not move in {wtxt}."
        tmax = max((sp[i], sg.ts[i]) for i in range(len(sp)) if md[i] in (1, 2)); res = sg.res
        above80 = sum(res for v in mv if v > 80); nmv = len(mv) * res
        return (f"{wtxt.capitalize()}: the car was moving for {Q.dur(nmv)}, averaging {mean(mv):.1f} km/h while moving (median {sorted(mv)[len(mv) // 2]:.1f}); top speed {tmax[0]:.0f} km/h at {A.hhmm(tmax[1])}; {100 * above80 / nmv:.0f}% of the moving time above 80 km/h. ({sg.tier} data.)")

    def h_avgtemp_mode(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        mode = 2 if re.search(r"motorway", s) else 1
        sel = [sg.cols["batt"][i] for i in range(len(sg)) if sg.cols["mode"][i] == mode]; oth = [sg.cols["batt"][i] for i in range(len(sg)) if sg.cols["mode"][i] == (1 if mode == 2 else 2)]
        if not sel: return f"There is no {'motorway' if mode == 2 else 'city'} driving in {wtxt}."
        w = Q.welch(sel[::max(1, len(sel) // 2000)], oth[::max(1, len(oth) // 2000)]) if oth else None
        return (f"Average battery temperature during {'motorway' if mode == 2 else 'city'} driving in {wtxt} (samples where the car was above 72 km/h, 1 Hz): {mean(sel):.1f} C (peak {max(sel):.1f}, {len(sel)} samples)" +
                (f"; for comparison {'city' if mode == 2 else 'motorway'} driving averaged {mean(oth):.1f} C." if oth else ".") + f" Outside temperature averaged {mean([sg.cols['amb'][i] for i in range(len(sg)) if sg.cols['mode'][i] == mode]):.1f} C. ({sg.tier} data.)")

    def h_temprise(self, s, raw):
        win = int(Q.find_duration(s, 300)); sg, lo, hi, wtxt = self.sigwin(s)
        if len(sg) < 10: return f"There is no signal data for {wtxt}."
        step = max(1, int(win / sg.res)); b = sg.cols["batt"]; best = (-1e9, 0)
        for i in range(len(b) - step):
            if sg.ts[i + step] - sg.ts[i] > win + 2 * sg.res: continue
            d = b[i + step] - b[i]
            if d > best[0]: best = (d, i)
        d, i = best
        self.brief = f"{d:.1f} DEG, from {b[i]:.1f} to {b[i + step]:.1f} DEG within {Q.dur(win)}, on {self.fts(sg.ts[i])[:-3]}.".replace("DEG", "\u00b0C")
        if d < -1e8: return "There is not enough continuous data to measure a rise."
        t0 = sg.ts[i]; t1 = sg.ts[i + step]; ch = [self.c.execute("SELECT kind, kw FROM charges WHERE start<=? AND end>=?", (t1, t0)).fetchone()] if self._has("charges") else [None]
        s1 = Q.state_at(self.c, t0)
        return (f"The largest pack-temperature rise over any {Q.dur(win)} interval in {wtxt} was {d:.1f} C ({b[i]:.1f} -> {b[i + step]:.1f} C), from {self.fts(t0)} to {self.fts(t1)}" +
                (f" - during {ch[0][0]} charging at up to {ch[0][1]:.0f} kW" if ch[0] else f" while the car was {'driving at %.0f km/h' % s1.get('speed', 0) if s1.get('speed', 0) > 3 else 'standing still'}") + f". ({sg.tier} data.)")

    def h_tripdiff(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(0, self.now + 1, "the whole history"))
        T = [t for t in Q.trips_between(self.c, lo, hi) if t["km"] >= 5]
        if len(T) < 4: return "Not enough trips (over 5 km) to compare."
        best = min(T, key=lambda t: t["wh_km"]); worst = max(T, key=lambda t: t["wh_km"])
        d = worst["wh_km"] - best["wh_km"]; per100 = d * 100 / 1000
        row = lambda t: f"{self.fts(t['start'])}: {t['km']:.1f} km, {t['kwh']:.2f} kWh = {t['wh_km']:.0f} Wh/km (avg speed {t['speed']:.0f} km/h, {100 * t['hwy']:.0f}% motorway, outside {t['amb']:.1f} C, climate {t['hvac']:.0f} W)"
        return (f"Of {len(T)} trips over 5 km in {wtxt}:\n  - most efficient: {row(best)}\n  - least efficient: {row(worst)}\nDifference: {d:.0f} Wh/km ({worst['wh_km']:.0f} - {best['wh_km']:.0f}), i.e. {per100:.1f} kWh more per 100 km, "
                f"{100 * d / best['wh_km']:.0f}% more energy for the same distance. (Calculation: kWh / km x 1000 for each trip, trips table.)")

    def h_tripwarn(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(0, self.now + 1, "the whole history"))
        T = Q.trips_between(self.c, lo, hi)
        if not T: return "No trips recorded."
        marks = ",".join("?" * len(WARN_TYPES)); n_w = 0; byt = Counter()
        for t in T:
            r = self.c.execute(f"SELECT type FROM events WHERE type IN ({marks}) AND ts>=? AND ts<=?", (*WARN_TYPES, t["start"], t["end"])).fetchall()
            if r: n_w += 1; byt.update(x[0] for x in r)
        alln = self.c.execute(f"SELECT COUNT(*) FROM events WHERE type IN ({marks}) AND ts>=? AND ts<?", (*WARN_TYPES, lo, hi)).fetchone()[0]
        return (f"{n_w} of {len(T)} trips ({100 * n_w / len(T):.1f}%) contained at least one warning or fault event during the drive in {wtxt}" + (f" ({', '.join(f'{v} x {k}' for k, v in byt.most_common())})" if byt else "") +
                f". In total {alln} warning/fault events were recorded in that period; the rest happened while parked or charging, or between trips. (Warning = trouble code, over-temperature, low tyre pressure, low 12 V, camera fault, CAN anomaly.)")

    # ------------------------------------------------------------------ period comparison
    def period_rows(self, p1, p2):
        rows = []
        D1, D2 = Q.daily_series(self.c, *p1), Q.daily_series(self.c, *p2)
        ser = [("Energy use", "Wh/km", "wh_km", True), ("Average speed (trips)", "km/h", "speed", None), ("Distance per day", "km", "km", None), ("Trips per day", "", "trips", None), ("Motorway share", "% of km", "hwy", None),
               ("Outside temperature", "C", "amb", None), ("Climate power", "W", "hvac", True), ("Hard events", "per 100 km", "harsh", True)]
        for lab, un, key, worse in ser:
            a = [d[key] * (100 if key == "hwy" else 1) for d in D1]; b = [d[key] * (100 if key == "hwy" else 1) for d in D2]
            if len(a) >= 2 and len(b) >= 2: rows.append((lab, un, a, b, worse))
        for lab, col, un, kw, worse in (("Pack temperature (mean)", "batt", "C", {}, True), ("Coolant flow (running)", "flow", "L/min", {"minv": 3.0}, False), ("Motor temperature", "motor", "C", {}, True)):
            a = [v for _, v, _ in self.daily(col, *p1, modes=(1, 2, 3), **kw)]; b = [v for _, v, _ in self.daily(col, *p2, modes=(1, 2, 3), **kw)]
            if len(a) >= 2 and len(b) >= 2: rows.append((lab, un, a, b, worse))
        a = [v for _, v, _ in self.daily("batt_max", *p1, modes=(1, 2, 3))]; b = [v for _, v, _ in self.daily("batt_max", *p2, modes=(1, 2, 3))]
        if len(a) >= 2 and len(b) >= 2: rows.append(("Pack temperature (daily peak)", "C", a, b, True))
        return rows

    def fmt_rows(self, rows, sort=True, top=None):
        out = []
        for lab, un, a, b, worse in rows:
            w = Q.welch(a, b); out.append((abs(w[0]) if w else 0, lab, un, a, b, w))
        if sort: out.sort(key=lambda r: -r[0])
        lines = []
        for _, lab, un, a, b, w in out[:top]:
            ma, mb = mean(a), mean(b); pct = f" ({SIGNED(100 * (mb - ma) / ma, 0)}%)" if abs(ma) > 1e-9 else ""
            sig = "" if w is None else (f"  [{pfmt(w[2])}{', significant' if w[2] < 0.05 else ', not significant'}]")
            lines.append(f"  - {lab}: {ma:.1f} -> {mb:.1f}{' ' + un if un else ''}{pct}{sig}")
        return lines

    def h_compare_periods(self, s, raw):
        cv = self.cov()
        if re.search(r"first (recorded )?week.*(final|last|most recent|recent|latest)|(final|last|most recent|recent|latest) week.*first", s):
            p1 = (cv["lo"], cv["lo"] + 7 * DAY); p2 = (cv["hi"] + 1 - 7 * DAY, cv["hi"] + 1); n1, n2 = "first recorded week", "most recent week"
        else:
            lo, hi = A.window(self.now, "tw", self.c); lw = (lo - 7 * DAY, lo)
            if re.search(r"this week", s): p2 = (lo, self.now); p1 = lw; n1, n2 = "last week", "this week"
            else: p1 = (cv["hi"] + 1 - 14 * DAY, cv["hi"] + 1 - 7 * DAY); p2 = (cv["hi"] + 1 - 7 * DAY, cv["hi"] + 1); n1, n2 = "the week before", "the latest week"
        rows = self.period_rows(p1, p2)
        if not rows: return "Not enough days of data in one of the two periods to compare."
        only_speed = re.search(r"average speed", s) and not re.search(r"statistic|everything|all", s)
        if only_speed:
            rows = [r for r in rows if r[0].startswith("Average speed")]
        n_d = (len(rows[0][2]), len(rows[0][3]))
        head = f"{n1.capitalize()} ({self.day(p1[0])} to {self.day(p1[1] - 1)}) vs {n2} ({self.day(p2[0])} to {self.day(p2[1] - 1)}); Welch t-test on daily values ({n_d[0]} vs {n_d[1]} days):"
        out = [head] + self.fmt_rows(rows)
        sg = [r for r in rows if (lambda w: w and w[2] < 0.05)(Q.welch(r[2], r[3]))]
        out.append(f"{len(sg)} of {len(rows)} measures differ significantly (p < 0.05); with a handful of days per period, small differences may just be day-to-day variation.")
        if only_speed:
            sg1, sg2 = self.sig(*p1), self.sig(*p2)
            if len(sg1) > 5 and len(sg2) > 5 and sg1.res == sg2.res:
                m1 = mean([v for v, md in zip(sg1.cols["speed"], sg1.cols["mode"]) if md in (1, 2) and v > 1]); m2 = mean([v for v, md in zip(sg2.cols["speed"], sg2.cols["mode"]) if md in (1, 2) and v > 1])
                out.append(f"Average speed while moving (1 Hz): {m1:.1f} -> {m2:.1f} km/h.")
        ds = sorted(((abs(w[0]) if w else 0, r, w) for r in rows for w in [Q.welch(r[2], r[3])]), key=lambda x: -x[0]); _, r0, w0 = ds[0]; ma, mb = mean(r0[2]), mean(r0[3]); pc = f" ({SIGNED(100 * (mb - ma) / ma, 0)}%)" if abs(ma) > 1e-9 else ""
        if only_speed or len(rows) == 1: self.brief = f"{r0[0]} went from {ma:.0f} to {mb:.0f} {r0[1]}{pc} - " + ("a clear difference." if w0 and w0[2] < 0.05 else "no clear difference, within normal day-to-day variation.")
        else: self.brief = (f"{len(sg)} of {len(rows)} measures differ clearly; the biggest is {r0[0].lower()}: {ma:.0f} to {mb:.0f} {r0[1]}{pc}." if sg else f"No clear difference: none of {len(rows)} measures moved beyond normal day-to-day variation.")
        self.ctx.update(compare=(p1, p2)); return "\n".join(out)

    def h_changes(self, s, raw):
        cv = self.cov(); hi = cv["hi"] + 1; p2 = (hi - 7 * DAY, hi); p1 = (hi - 14 * DAY, hi - 7 * DAY)
        rows = self.period_rows(p1, p2)
        if not rows: return "Not enough data in the last two weeks."
        scored = []
        for r in rows:
            w = Q.welch(r[2], r[3])
            if w: scored.append((abs(w[0]), r, w))
        scored.sort(key=lambda x: -x[0]); top = scored[:3]
        out = [f"The three biggest changes in the last week ({self.day(p2[0])} to {self.day(p2[1] - 1)}) compared with the week before, ranked by how large the change is relative to normal day-to-day variation:"]
        for k, (z, (lab, un, a, b, worse), w) in enumerate(top, 1):
            ma, mb = mean(a), mean(b); out.append(f"  {k}. {lab}: {ma:.1f} -> {mb:.1f}{' ' + un if un else ''} ({SIGNED(mb - ma)}{' ' + un if un else ''}, {pfmt(w[2])})")
        ev = self.c.execute("SELECT type, COUNT(*) FROM events WHERE ts>=? AND ts<? AND type IN (%s) GROUP BY type" % ",".join("?" * len(WARN_TYPES)), (p2[0], p2[1], *WARN_TYPES)).fetchall()
        if ev: out.append("Events this week: " + ", ".join(f"{n} x {t}" for t, n in ev) + ".")
        self.ctx.update(compare=(p1, p2)); return "\n".join(out)

    # ------------------------------------------------------------------ trends over time
    def series_for(self, m, lo, hi):
        if m == "eff":
            return [(d["ts"], d["wh_km"]) for d in Q.daily_series(self.c, lo, hi)], "Wh/km", "energy use"
        if m == "rpm":
            q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, AVG(speed), COUNT(*) FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2) AND speed>10 GROUP BY d ORDER BY d"
            return [(d, Q.rpm_of(v / 10.0)) for d, v, n in self.c.execute(q, (lo, hi)).fetchall() if n >= 20], "rpm", "estimated motor speed while moving"
        col = {"batt": "batt", "motor": "motor", "flow": "flow", "soc": "soc", "amb": "amb", "speed": "speed", "power": "power", "hvac": "hvac"}[m]
        agg = "batt_max" if False else col; sc = Q.SIGS[col][1]
        mn = 3.0 if col == "flow" else None
        return [(d, v) for d, v, n in self.daily(col, lo, hi, modes=(1, 2, 3) if col not in ("speed",) else (1, 2), minv=mn, scale=sc)], UNIT[m], NAME[m].lower()

    def h_trend(self, s, raw):
        m = Q.metric_of(s)
        if re.search(r"efficien|consumption|range|energy|economy|mileage|fuel", s) and not re.search(r"battery temp|motor temp|coolant", s): m = "eff"
        if m is None or m in ("cabin", "throttle", "steering", "radar", "engine_load", "gear", "tyre"): return None
        lo, hi, wtxt = self.win(s, default=(self.cov()["lo"], self.cov()["hi"] + 1, "the whole recorded history"))
        vol = bool(re.search(r"volatil|variab|stab|fluctuat|erratic|spread|consisten", s))
        ser, un, nm = self.series_for(m, lo, hi)
        if len(ser) < 6: return f"Not enough days of {nm} data in {wtxt} ({len(ser)}) to judge a trend."
        if vol:
            q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, {Q.SIGS['batt' if m == 'batt' else m][0]} FROM sig_1m WHERE ts>=? AND ts<? AND mode IN (1,2,3)" if m in Q.SIGS else None
            if q is None: return None
            by = {}
            for d, v in self.c.execute(q, (lo, hi)).fetchall():
                if v is not None: by.setdefault(d, []).append(v / Q.SIGS[m][1])
            sds = sorted((d, sd(v)) for d, v in by.items() if len(v) >= 30); h = len(sds) // 2
            if h < 3: return "Not enough days to compare variability."
            a, b = [x[1] for x in sds[:h]], [x[1] for x in sds[h:]]; w = Q.welch(a, b); ratio = mean(b) / mean(a) if mean(a) > 0 else float("nan"); fit = Q.slope_test([x[0] / DAY for x in sds], [x[1] for x in sds])
            verdict = "Yes" if ratio > 1.25 and w and w[2] < 0.05 else "No"
            return (f"{verdict} - the day-by-day spread (standard deviation of the per-minute readings, car on) of {nm} was {mean(a):.2f} {un} in the first half of {wtxt} ({self.day(sds[0][0])} to {self.day(sds[h - 1][0])}) and "
                    f"{mean(b):.2f} {un} in the second half ({self.day(sds[h][0])} to {self.day(sds[-1][0])}): ratio {ratio:.2f}, Welch {pfmt(w[2]) if w else 'n/a'}" + (f"; trend of the daily spread {SIGNED(fit['slope'], 3)} {un} per day ({pfmt(fit['p'])})." if fit else "."))
        x = [d / DAY for d, _ in ser]; y = [v for _, v in ser]; fit = Q.slope_test(x, y); k = max(3, len(ser) // 4); a, b = mean(y[:k]), mean(y[-k:]); pct = 100 * (b - a) / a if a else 0
        sig = fit and fit["p"] < 0.01 and abs(pct) >= 3
        word = {"eff": ("worse (more Wh/km)", "better (less Wh/km)"), }.get(m, ("upward", "downward"))
        if not sig: verdict = f"No clear trend: {nm} was {a:.1f} -> {b:.1f} {un} ({SIGNED(pct)}%) between the first and last {k} days, slope {SIGNED(fit['slope'], 3) if fit else 'n/a'} {un}/day ({pfmt(fit['p']) if fit else ''}, R2 {fit['r2']:.2f})." if fit else "No clear trend."
        else:
            up = fit["slope"] > 0; verdict = f"Yes, {'upward' if up else 'downward'}: {nm} went {a:.1f} -> {b:.1f} {un} ({SIGNED(pct)}%) between the first and last {k} days of {wtxt}; slope {SIGNED(fit['slope'], 3)} {un}/day, {pfmt(fit['p'])}, R2 {fit['r2']:.2f}, {len(ser)} days."
            if m == "eff": verdict = ("Yes - energy use per km is getting worse: " if up else "No - energy use per km is getting better: ") + verdict.split(": ", 1)[1]
        if m == "motor" and re.search(r"engine", raw.lower()): verdict = "This is an electric vehicle - there is no engine; the drive motor temperature is the equivalent. " + verdict
        self.brief = (f"Yes, {'up' if fit['slope'] > 0 else 'down'}: {nm} went from {a:.0f} to {b:.0f} {un} ({SIGNED(pct, 0)}%)." if sig else f"No clear trend: {nm} stayed around {a:.0f} to {b:.0f} {un}.") if fit else "No clear trend."
        if sig and m == "eff": self.brief = ("Getting worse: " if fit["slope"] > 0 else "Getting better: ") + f"energy use went from {a:.0f} to {b:.0f} {un} ({SIGNED(pct, 0)}%)."
        if m == "rpm": verdict += " " + self.note_rpm()
        if m == "eff" and re.search(r"fuel|mileage", raw.lower()): verdict += " (This is an electric vehicle, so energy per km replaces fuel economy.)"
        self.ctx.update(trend=(m, a, b), topic="range" if m == "eff" else self.ctx.get("topic")); return verdict

    def h_degrade(self, s, raw):
        cv = self.cov(); lo, hi = cv["lo"], cv["hi"] + 1; items = []

        def add(name, ser, worse_up, un, evtype=None, floor=0.0):
            if len(ser) < 8: return
            x = [d / DAY for d, _ in ser]; y = [v for _, v in ser]; k = max(3, len(y) // 5); a, b = mean(y[:k]), mean(y[-k:]); fit = Q.slope_test(x, y)
            if not fit or a == 0: return
            pct = 100 * (b - a) / abs(a); bad = (pct > 0) == worse_up
            items.append({"name": name, "a": a, "b": b, "pct": pct, "p": fit["p"], "bad": bad, "un": un, "r2": fit["r2"], "days": len(y), "ev": evtype})
        add("coolant flow (pump running)", [(d, v) for d, v, _ in self.daily("flow", lo, hi, modes=(1, 2, 3), minv=3.0)], False, "L/min", "DTC_SET")
        add("pack temperature (daily mean)", [(d, v) for d, v, _ in self.daily("batt", lo, hi, modes=(1, 2, 3))], True, "C", "WARN_BATT_OVERTEMP")
        add("pack temperature (daily peak)", [(d, v) for d, v, _ in self.daily("batt_max", lo, hi, modes=(1, 2, 3))], True, "C", "WARN_BATT_OVERTEMP")
        add("motor temperature", [(d, v) for d, v, _ in self.daily("motor", lo, hi, modes=(1, 2))], True, "C")
        add("energy use per km", [(d["ts"], d["wh_km"]) for d in Q.daily_series(self.c, lo, hi)], True, "Wh/km")
        if self._has("vday"):
            vd = self.c.execute("SELECT ts, soh, pad, v12 FROM vday WHERE ts>=? AND ts<? ORDER BY ts", (lo, hi)).fetchall()
            add("battery state of health", [(t, x) for t, x, _, _ in vd if x], False, "%"); add("brake-pad thickness", [(t, p) for t, _, p, _ in vd if p], False, "%"); add("12 V battery at rest", [(t, v) for t, _, _, v in vd if v], False, "V", "LOW_12V")
        try:
            q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, AVG(tp_fl+tp_fr+tp_rl+tp_rr)/40.0 FROM ext_1m WHERE ts>=? AND ts<? AND tt IS NOT NULL GROUP BY d ORDER BY d"
            add("average tyre pressure (as measured)", [(d, v) for d, v in self.c.execute(q, (lo, hi)).fetchall() if v], False, "kPa", "TPMS_LOW")
        except Exception: pass
        if not items: return "There isn't enough history (at least 8 days) to look for gradual degradation."
        hits = [i for i in items if i["bad"] and i["p"] < 0.01 and abs(i["pct"]) >= 3]; hits.sort(key=lambda i: -abs(i["pct"]))
        self.brief = (f"Yes - one slow drift: {hits[0]['name']} went from {hits[0]['a']:.1f} to {hits[0]['b']:.1f} {hits[0]['un']} ({hits[0]['pct']:+.0f}%) without a single fault event flagging it." if len(hits) == 1 else
                      f"Yes - {len(hits)} slow drifts, the biggest in {hits[0]['name']} ({hits[0]['pct']:+.0f}%)." if hits else "Nothing is drifting in a worrying direction beyond normal variation.")
        out = [f"Looking for slow drifts across the whole history ({self.day(lo)} to {self.day(hi - 1)}), comparing the first and last fifth of the days and testing the slope of each daily series:"]
        if hits:
            for i in hits:
                flagged = ""
                if i["ev"]:
                    n = self.c.execute("SELECT COUNT(*) FROM events WHERE type=?", (i["ev"],)).fetchone()[0]; flagged = f" A related event was recorded ({n} x {i['ev']})." if n else " No single fault event flagged it."
                out.append(f"  - {i['name']}: {i['a']:.1f} -> {i['b']:.1f} {i['un']} ({SIGNED(i['pct'], 0)}%, {pfmt(i['p'])}, R2 {i['r2']:.2f}, {i['days']} days).{flagged}")
            out.append(f"Most degraded: {hits[0]['name']} ({SIGNED(hits[0]['pct'], 0)}%).")
        else: out.append("  Nothing is drifting in a worrying direction beyond normal variation.")
        calm = [i["name"] for i in items if i not in hits]
        if calm: out.append("Stable or improving: " + ", ".join(calm) + ".")
        self.ctx.update(degrade=hits); return "\n".join(out)

    # ------------------------------------------------------------------ relationships between signals
    def minute_rows(self, lo, hi, modes=(1, 2, 3)):
        return self.c.execute(f"SELECT ts, speed, power, batt, flow, amb, motor, soc, mode, hvac FROM sig_1m WHERE ts>=? AND ts<? AND mode IN ({','.join(str(m) for m in modes)})", (lo, hi)).fetchall()

    def h_corr(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.cov()["lo"], self.cov()["hi"] + 1, "the whole recorded history"))
        R = self.minute_rows(lo, hi, (1, 2))
        if len(R) < 100: return "Not enough driving data to measure a relationship."
        sp = [r[1] / 10 for r in R]; pw = [r[2] / 1000 for r in R]; bt = [r[3] / 10 for r in R]; am = [r[5] / 10 for r in R]; fl = [r[4] / 10 for r in R]
        r_sb = Q.pearson(sp, bt); r_pb = Q.pearson(pw, bt); r_ab = Q.pearson(am, bt); r_sa = Q.pearson(sp, am)
        fit = Q.ols_multi([[a, p] for a, p in zip(am, pw)], bt)
        # within-trip heating: change of pack temperature over the next 10 minutes vs mean speed
        sg = Q.load(self.c, max(lo, self.cov()["hz_lo"] or lo), hi); d_sp = []; d_dt = []
        if len(sg) > 1200:
            step = 600
            for i in range(0, len(sg) - step, 120):
                if sg.ts[i + step] - sg.ts[i] != step: continue
                if min(sg.cols["mode"][i:i + step]) == 0 or 3 in sg.cols["mode"][i:i + step]: continue
                d_sp.append(mean(sg.cols["speed"][i:i + step])); d_dt.append(sg.cols["batt"][i + step] - sg.cols["batt"][i])
        r_h = Q.pearson(d_sp, d_dt) if len(d_sp) > 20 else None
        out = [f"Relationship between vehicle speed and battery temperature over {wtxt} ({len(R)} driving minutes):",
               f"  - raw correlation speed vs pack temperature: r = {r_sb:+.2f}" + (f"; but outside temperature vs pack temperature r = {r_ab:+.2f} and speed vs outside temperature r = {r_sa:+.2f}" if r_ab is not None else "")]
        if r_pb is not None: out.append(f"  - battery power vs pack temperature: r = {r_pb:+.2f}")
        if fit: out.append(f"  - pack temperature ~ outside temperature + battery power: R2 = {fit[1]:.2f}; +1 C outside -> {fit[0][1]:+.2f} C pack, +10 kW -> {10 * fit[0][2]:+.2f} C pack.")
        if r_h is not None: out.append(f"  - within drives (1 Hz, 10-minute steps): mean speed vs temperature rise r = {r_h:+.2f} over {len(d_sp)} steps (speed raises battery heat through power draw).")
        weak = abs(r_sb or 0) < 0.3
        out.append("Reading: " + (("there is only a weak direct link between speed and pack temperature; ambient temperature and charging dominate the pack temperature. " if weak else "speed and pack temperature move together, but ") + "This shows association, not cause - a hot day also means faster/longer drives, and charging heats the pack with the car stationary."))
        return "\n".join(out)

    def episodes(self, col, lo, hi, kind="rise", win=300, top=4):
        """Largest changes of a signal over `win` seconds (non-overlapping), 1 Hz where available else per-minute."""
        sg = self.sig(lo, hi); b = sg.cols[col]; step = max(1, int(win / sg.res)); cand = []
        for i in range(0, len(b) - step):
            if sg.ts[i + step] - sg.ts[i] > win + 2 * sg.res: continue
            d = b[i + step] - b[i]; cand.append((d if kind == "rise" else -d, i))
        cand.sort(reverse=True); out = []
        for d, i in cand:
            if all(abs(sg.ts[i] - sg.ts[j]) > 2 * win for _, j in out): out.append((d, i))
            if len(out) >= top: break
        return sg, step, out

    def h_cooccur(self, s, raw):
        lo, hi, wtxt = self.win(s, default=None)
        if lo is None: lo, hi, wtxt = self.recent(7)
        sg, step, eps = self.episodes("batt", lo, hi, "rise", 300)
        if not eps: return f"There is no pack-temperature data for {wtxt}."
        base = sd(sg.cols["batt"]) or 1.0; mu_rise = mean([sg.cols["batt"][i + step] - sg.cols["batt"][i] for i in range(0, len(sg) - step, max(1, step))])
        out = [f"The largest 5-minute pack-temperature rises in {wtxt} (typical 5-minute change {mu_rise:+.2f} C), and what else changed in the same 5 minutes ({sg.tier}):"]
        for d, i in eps[:3]:
            j = i + step; seg = {m: (sg.cols[m][i], sg.cols[m][j]) for m in ("speed", "power", "flow", "soc", "amb", "motor")}; mode = {0: "stationary, car on", 1: "city driving", 2: "motorway driving", 3: "charging"}.get(int(sg.cols["mode"][j]), "")
            out.append(f"  - {self.fts(sg.ts[i])}: pack +{d:.1f} C ({sg.cols['batt'][i]:.1f} -> {sg.cols['batt'][j]:.1f}); {mode}; speed {seg['speed'][0]:.0f} -> {seg['speed'][1]:.0f} km/h; power {mean(sg.cols['power'][i:j]):.1f} kW avg; coolant flow {seg['flow'][0]:.1f} -> {seg['flow'][1]:.1f} L/min; motor {seg['motor'][0]:.0f} -> {seg['motor'][1]:.0f} C; outside {seg['amb'][1]:.1f} C; SoC {seg['soc'][0]:.0f} -> {seg['soc'][1]:.0f}%")
        out.append("These are signals that moved at the same time; heating usually follows load (power draw, DC fast charging) and is amplified when coolant flow is low - the timing alone doesn't prove which caused which.")
        return "\n".join(out)

    def h_rpm_lowspeed(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        ac = sg.accel(); pw = sg.cols["power"]; sp = sg.cols["speed"]
        sel = runs(sg, lambda i: sp[i] < 20 and (pw[i] > 12 or ac[i] > 1.5))
        out = ["RPM is not a separate sensor on this EV - I compute it from vehicle speed (fixed ratio), so 'high RPM but low speed' cannot occur in this data by construction (e.g. a slipping clutch or wheel-spin would show in a real engine's RPM, not here). " + self.note_rpm()]
        if sel:
            top = sorted(sel, key=lambda r: -max(pw[r[2]:r[3] + 1]))[:3]
            out.append(f"The nearest real equivalent is high motor power demanded at low speed (above 12 kW below 20 km/h): {plural(len(sel), 'episode')} in {wtxt}. Strongest:")
            for r in top: out.append(f"  - {self.fts(r[0])} for {Q.dur(r[1] - r[0])}: up to {max(pw[r[2]:r[3] + 1]):.1f} kW at {mean(sp[r[2]:r[3] + 1]):.0f} km/h average")
        else: out.append(f"I also found no episode of high motor power (above 12 kW) at low speed (below 20 km/h) in {wtxt}.")
        return "\n".join(out)

    def h_multi(self, s, raw):
        sg, lo, hi, wtxt = self.sigwin(s)
        if sg.res != 1: return "That needs 1 Hz data, which is kept for 7 days - ask about the last week."
        sp, pw = sg.cols["speed"], sg.cols["power"]; ev = []; i = 0
        while i < len(sp) - 10:
            if sg.ts[i + 10] - sg.ts[i] != 10: i += 1; continue
            dv = sp[i + 10] - sp[i]; dp = pw[i + 10] - pw[i]
            if abs(dv) > 25 and abs(dp) > 12: ev.append((abs(dv) + abs(dp), i, dv, dp)); i += 30
            else: i += 1
        out = ["Throttle position isn't recorded and RPM is derived from speed, so I use the signals that exist: acceleration (from speed), estimated RPM (from speed) and battery power (the throttle's effect). " + ("Events where speed changed by more than 25 km/h AND power by more than 12 kW within 10 s in " + wtxt + ": ") + (f"{len(ev)} found. Largest:" if ev else "none found.")]
        for _, i, dv, dp in sorted(ev, reverse=True)[:5]:
            out.append(f"  - {self.fts(sg.ts[i])}: speed {sp[i]:.0f} -> {sp[i + 10]:.0f} km/h ({dv / 3.6 / 10:+.1f} m/s2), est. RPM {Q.rpm_of(sp[i]):,.0f} -> {Q.rpm_of(sp[i + 10]):,.0f}, power {pw[i]:.1f} -> {pw[i + 10]:.1f} kW")
        return "\n".join(out)

    def h_pairs(self, s, raw):
        cv = self.cov(); lo, hi = cv["lo"], cv["hi"] + 1; cps = []
        for lab, col, kw, sc, mode in (("coolant flow", "flow", {"minv": 3.0}, 10.0, (1, 2, 3)), ("pack temperature", "batt", {}, 10.0, (1, 2, 3)), ("motor temperature", "motor", {}, 10.0, (1, 2)), ("outside temperature", "amb", {}, 10.0, (1, 2, 3)),
                                       ("climate power", "hvac", {}, 1.0, (1, 2)), ("average speed", "speed", {}, 10.0, (1, 2))):
            ser = [(d, v) for d, v, _ in self.daily(col, lo, hi, modes=mode, scale=sc, **kw)]
            cp = self.changepoint(ser)
            if cp: cps.append((lab,) + cp)
        D = Q.daily_series(self.c, lo, hi); cp = self.changepoint([(d["ts"], d["wh_km"]) for d in D])
        if cp: cps.append(("energy use per km",) + cp)
        try:
            q = f"SELECT ((ts+({A.TZ}))/86400)*86400-({A.TZ}) d, AVG(tp_fl+tp_fr+tp_rl+tp_rr)/40.0 FROM ext_1m WHERE ts>=? AND ts<? GROUP BY d ORDER BY d"; cp = self.changepoint(self.c.execute(q, (lo, hi)).fetchall())
            if cp: cps.append(("tyre pressure",) + cp)
        except Exception: pass
        evs = self.c.execute("SELECT MIN(ts), type, substr(detail,1,5) FROM events WHERE type IN (%s) GROUP BY type, substr(detail,1,5)" % ",".join("?" * len(WARN_TYPES)), WARN_TYPES).fetchall()
        for ts, ty, dd in evs: cps.append((f"first {ty} ({Q.safe_text(dd).strip()}) event", ts, 0, 0, 0))
        if len(cps) < 2: return "No two signals show a clear step change in the daily data."
        cps.sort(key=lambda r: r[1]); pairs = []
        for i in range(len(cps)):
            for j in range(i + 1, len(cps)):
                gap = abs(cps[i][1] - cps[j][1]) / DAY
                if gap <= 2.0 and not (cps[i][2] == 0 and cps[j][2] == 0): pairs.append((gap, cps[i], cps[j]))
        out = ["Step or ramp changes found in the daily averages, plus the first warning events (what, date, size, strength):"] + [(f"  - {c[0]}: {self.fts(c[1])}" if c[2] == 0 and c[3] == 0 else f"  - {c[0]}: {self.day(c[1])}, {c[2]:.1f} -> {c[3]:.1f} ({SIGNED(100 * (c[3] - c[2]) / c[2] if c[2] else 0, 0)}%), t = {c[4]:.1f}") for c in cps]
        if pairs:
            pairs.sort(key=lambda p: p[0]); g, a, b = pairs[0]; out.append(f"Closest pair: {a[0]} and {b[0]} both changed within {g:.1f} day(s) of each other ({self.day(a[1])} and {self.day(b[1])}). They may look unrelated, but they could share a cause - or be coincidence; "
                                                                      "a shared cause is plausible if the physics connects them (e.g. a slower coolant pump -> hotter pack -> more cooling load).")
        else: out.append("No two of them changed within 2 days of each other, so I found no near-simultaneous pair in this data.")
        return "\n".join(out)

    def changepoint(self, ser, min_side=3):
        n = len(ser)
        if n < 2 * min_side + 1: return None
        y = [v for _, v in ser]; best = None
        for k in range(min_side, n - min_side + 1):
            a, b = y[:k], y[k:]; w = Q.welch(a, b)
            if not w: continue
            if best is None or abs(w[0]) > abs(best[3]): best = (ser[k][0], mean(a), mean(b), w[0], k)
        if not best or abs(best[3]) < 3: return None
        d_rel = abs(best[2] - best[1]) / (abs(best[1]) or 1)
        if d_rel < 0.05: return None
        return best[0], best[1], best[2], best[3]

    # ------------------------------------------------------------------ conditions
    def h_conditions(self, s, raw):
        lo, hi, wtxt = self.win(s, default=(self.cov()["lo"], self.cov()["hi"] + 1, "the whole recorded history"))
        T = Q.trips_between(self.c, lo, hi)
        for t in T: t["tyre"] = Q.tyre_deficit(self.c, t["start"])
        T = [t for t in T if t["km"] >= 3]
        if len(T) < 12: return "Not enough trips to compare driving conditions."
        groups = [("cold weather (below 8 C)", lambda t: t["amb"] < 8), ("mild weather (8 to 24 C)", lambda t: 8 <= t["amb"] <= 24), ("hot weather (above 24 C)", lambda t: t["amb"] > 24), ("mostly motorway (over 60% of the distance)", lambda t: t["hwy"] > 0.6),
                  ("mostly city (under 20% motorway)", lambda t: t["hwy"] < 0.2), ("rain on the road (over 20% of the trip)", lambda t: t["rain"] > 0.2), ("hard driving (over 15 hard events per 100 km)", lambda t: t["harsh"] > 15),
                  ("low tyre pressure (over 5% below spec)", lambda t: t["tyre"] > 5), ("short trips (under 8 km)", lambda t: t["km"] < 8), ("long trips (over 30 km)", lambda t: t["km"] > 30)]
        rows = []
        for lab, f in groups:
            sel = [t["wh_km"] for t in T if f(t)]; oth = [t["wh_km"] for t in T if not f(t)]
            if len(sel) >= 5 and len(oth) >= 5: w = Q.welch(sel, oth); rows.append((mean(sel), lab, len(sel), mean(oth), w[2] if w else 1.0))
        rows.sort(reverse=True)
        out = [f"Average energy use per km by driving condition over {wtxt} ({len(T)} trips; overall {mean([t['wh_km'] for t in T]):.0f} Wh/km), highest first:"]
        for v, lab, n, o, p in rows: out.append(f"  - {lab}: {v:.0f} Wh/km over {n} trips (the other trips: {o:.0f}; {pfmt(p)})")
        fit = Q.eff_fit(self.c, lo, hi)
        if fit:
            eff = sorted(((b * s_, f[1], b, f[3]) for f, b, s_ in zip(fit["feats"], fit["beta"], fit["sg"])), reverse=True)
            out.append(f"Conditions overlap (cold days are also heating days), so I also fitted all of them together (regression on the {len(T)} trips, R2 = {fit['r2']:.2f}). Effect of one typical step in each condition, holding the others fixed:")
            for e_, nm, b, un in eff[:5]:
                if e_ > 0.05: out.append(f"  - {nm}: {b:.2f} Wh/km per {un.split(' (')[0]} -> {SIGNED(e_)} Wh/km per standard step")
            if eff and eff[0][0] > 0.05: out.append(f"Largest effect on energy per km: {eff[0][1]}.")
        return "\n".join(out)

    def classify_ts(self, ts):
        sg = Q.event_sig(self.c, ts, 15, 15); i = Q.nearest(sg, ts, 3 * sg.res)
        if i is None: return "unknown (no 1 Hz data)"
        if sg.cols["mode"][i] == 3: return "charging"
        sp = sg.cols["speed"]; lo = max(0, i - 10); hi = min(len(sp) - 1, i + 10)
        if sp[i] < 2 and max(sp[lo:hi + 1]) < 3: return "idling / stationary"
        a = (sp[min(hi, i + 5)] - sp[max(lo, i - 5)]) / 3.6 / max(1, sg.ts[min(hi, i + 5)] - sg.ts[max(lo, i - 5)])
        return "accelerating" if a > 0.3 else ("decelerating" if a < -0.3 else "cruising")

    def h_overheat_cond(self, s, raw):
        ev = self.evs("overheat"); note = ""
        if ev: pts = [(e[0], "warning") for e in ev]
        else:
            sg, step, eps = self.episodes("batt", self.cov()["lo"], self.cov()["hi"] + 1, "rise", 300, 3)
            hot = sorted(((sg.cols["batt"][i], sg.ts[i]) for i in range(len(sg))), reverse=True)[:1]
            pts = [(hot[0][1], "hottest minute")] if hot else []
            note = "No over-temperature warning has ever been recorded for this vehicle, so I classified the hottest pack-temperature moment instead (not a warning). "
        if not pts: return "No data."
        kinds = [self.classify_ts(ts) for ts, _ in pts[:8]]
        self.brief = (f"{len(pts)} over-temperature event{'s' if len(pts) != 1 else ''}: " + (kinds[0] if len(set(kinds)) == 1 else "; ".join(f"{v} x {k}" for k, v in Counter(kinds).items())) + ".") if not note else "No over-temperature warning exists; the hottest moment was while " + kinds[0] + "."
        out = [note + "Driving state at each:"]
        for ts, lab in pts[:8]:
            stt = Q.fmt_state(Q.state_at(self.c, ts), brief=True); stt = stt.split(", ", 1)[1] if ", " in stt else stt
            out.append(f"  - {self.fts(ts)} ({lab}): {self.classify_ts(ts)}; {stt}")
        return "\n".join(out)

    def h_abn_cond(self, s, raw):
        cv = self.cov(); R = self.minute_rows(cv["lo"], cv["hi"] + 1)
        if len(R) < 200: return "Not enough data."
        bt = sorted(r[3] for r in R); thr = bt[int(0.98 * len(bt))]; ab = [r for r in R if r[3] >= thr]
        def share(rows, f): return sum(1 for r in rows if f(r)) / max(len(rows), 1)
        conds = [("while charging", lambda r: r[8] == 3), ("city driving", lambda r: r[8] == 1), ("motorway driving", lambda r: r[8] == 2), ("hot weather (outside above 24 C)", lambda r: r[5] / 10 > 24), ("heavy load (over 15 kW)", lambda r: r[2] / 1000 > 15),
                 ("low coolant flow (under 6 L/min while running)", lambda r: 3 < r[4] / 10 < 6)]
        out = [f"'Abnormal' here = the hottest 2% of all recorded minutes (pack temperature at or above {thr / 10:.1f} C; {len(ab)} of {len(R)} minutes). How those minutes differ from all minutes:"]
        for lab, f in conds:
            a, b = share(ab, f), share(R, f); lift = a / b if b else float("inf"); out.append(f"  - {lab}: {100 * a:.0f}% of abnormal minutes vs {100 * b:.0f}% of all minutes (x{lift:.1f})")
        top = max(conds, key=lambda c: (share(ab, c[1]) / share(R, c[1])) if share(R, c[1]) else 0)
        out.append(f"Strongest association: {top[0]}. Association only - for example hot weather and charging are also the times the pack is under most load.")
        return "\n".join(out)
