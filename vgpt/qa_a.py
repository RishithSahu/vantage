"""QA part A: anchors (which event / reading a question points at), point-in-time lookups, extremes, event lists and the signals around an event."""
import re, math, datetime as dt
from collections import Counter
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd, F1, F0, SIGNED, plural

UNIT = {"speed": "km/h", "soc": "%", "power": "kW", "batt": "C", "flow": "L/min", "amb": "C", "hvac": "W", "motor": "C", "rpm": "rpm"}
NAME = {"speed": "Vehicle speed", "soc": "State of charge", "power": "Battery power", "batt": "Battery temperature", "flow": "Coolant flow", "amb": "Outside temperature", "hvac": "Climate-control power", "motor": "Motor temperature",
        "rpm": "Estimated motor speed (RPM)", "tyre": "Tyre pressure"}
EVENT_SCORE = {"WARN_BATT_OVERTEMP": 10, "DTC_SET": 6, "LOW_12V": 5, "TPMS_LOW": 3, "CAN_ANOMALY": 6, "CAMERA_BLOCKED": 2, "CAMERA_DEGRADED": 2}
WARN_TYPES = ("WARN_BATT_OVERTEMP", "DTC_SET", "TPMS_LOW", "LOW_12V", "CAMERA_BLOCKED", "CAMERA_DEGRADED", "CAN_ANOMALY")


def fmt_val(m, v):
    return f"{v:,.0f} rpm" if m == "rpm" else (f"{v:.1f} {UNIT[m]}" if m in ("power", "batt", "flow", "amb", "motor", "speed", "soc") else f"{v:.0f} {UNIT[m]}")


class MixA:
    # ------------------------------------------------------------------ anchors
    def evs(self, kind, lo=0, hi=None, newest_first=False, limit=None):
        return Q.events(self.c, kind, lo, hi or (self.now + 1), self.now, limit=limit, newest_first=newest_first)

    def last_anchor_event(self, s):
        k = self.kind_in(s)
        if k in ("major", "serious"): return self.most_serious()
        if k == "warning" or k is None:
            r = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN (%s) AND ts<=? ORDER BY ts DESC LIMIT 1" % ",".join("?" * len(WARN_TYPES)), (*WARN_TYPES, self.now)).fetchone()
            return r
        r = self.evs(k, newest_first=True, limit=1)
        return r[0] if r else None

    def kind_in(self, s):
        if re.search(r"most serious|major fault|biggest fault|worst (event|fault)|latest major|critical", s): return "major"
        return Q.kind_of(s)

    def most_serious(self):
        rows = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN (%s) AND ts<=?" % ",".join("?" * len(WARN_TYPES)), (*WARN_TYPES, self.now)).fetchall()
        if not rows: return None

        def score(r):
            sc = EVENT_SCORE.get(r[1], 1)
            if r[1] == "DTC_SET" and re.search(r"P1A|coolant|thermal|battery|pack", r[3] or "", re.I): sc += 3
            return (sc, r[0])
        return max(rows, key=score)

    def anchor(self, s):
        """-> (event_row or None, label, note, abnormal_dict or None). Falls back to the closest related event when the asked-for kind never happened, and says so."""
        k = self.kind_in(s)
        if re.search(r"abnormal|anomal|unusual", s) and k not in ("overheat", "adas", "diagnostic", "brake", "major") and not re.search(r"last (warning|fault)|that warning", s):
            ab = Q.last_abnormal(self.c, self.now)
            if ab: return None, f"the last abnormal reading ({ab['label']} {ab['value']:.1f} {ab['unit']}, normal {ab['normal']})", "", ab
            e = Q.last_anomaly(self.c, self.now)
            if e: return None, e["detail"], "", {"ts": e["ts"], "label": e["type"], "value": 0, "unit": "", "signal": e["type"], "normal": ""}
        if k in ("major", "serious"):
            e = self.most_serious(); lab = "the most serious recorded event"
        elif k in ("warning", "fault", None):
            e = self.last_anchor_event("warning"); lab = "the last warning/fault event"
        else:
            r = self.evs(k, newest_first=True, limit=1); e = r[0] if r else None; lab = f"the last {Q.KIND_NAME.get(k, k)}"
        note = ""
        if e is None and k == "overheat":
            f = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type='DTC_SET' AND (detail LIKE '%coolant%' OR detail LIKE '%thermal%' OR detail LIKE '%P1A%') ORDER BY ts DESC LIMIT 1").fetchone()
            if f: e, lab = f, "the thermal fault code (the closest related event)"; note = "No battery over-temperature warning has been recorded for this vehicle, so I used the nearest related event instead: "
            else:
                mx = self.c.execute("SELECT MAX(batt_max), ts FROM sig_1m").fetchone()
                note = f"No battery over-temperature warning has been recorded. The highest pack temperature in the history was {mx[0] / 10:.1f} C at {self.fts(mx[1])}." if mx and mx[0] is not None else "No battery over-temperature warning has been recorded."
        elif e is None:
            note = f"No {Q.KIND_NAME.get(k, 'matching event')} has been recorded in {self.span_text()}."
        return e, lab, note, None

    # ------------------------------------------------------------------ point in time
    def when_ts(self, s):
        """Resolve 'at 10:32:15' (+ optional date). -> (ts or None, text, status). status: ok / nodata / gap."""
        clk = Q.find_clock(s); d, expl, dtxt = Q.find_date(s, self.now, self.tz); cv = self.cov(); today = self.today()
        if clk is None: return None, "", "noclock"
        if d is not None:
            ts = Q.local_midnight(d, self.tz) + clk
            if ts < cv["lo"] - 60 or ts > cv["hi"] + 60: return ts, d.isoformat(), "nodata"
            return ts, d.isoformat(), ("ok" if "speed" in Q.state_at(self.c, ts) else "gap")
        last = None
        for back in range(0, 80):
            date = today - dt.timedelta(days=back); ts = Q.local_midnight(date, self.tz) + clk
            if ts > self.now or ts > cv["hi"] + 60: continue
            if ts < cv["lo"] - 60: break
            last = last or (ts, date.isoformat())
            if "speed" in Q.state_at(self.c, ts): return ts, date.isoformat() + " (no date given - the most recent day the car was running at that time)", "ok"
        return (last[0], last[1], "gap") if last else (None, "", "nodata")

    def h_point(self, s, raw):
        m = Q.metric_of(s) if not re.search(r"state|status|condition", s) else None
        if m in ("cabin", "throttle", "steering", "radar", "engine_load", "gear", "rear_camera", "fuel"): return self.h_missing(s, raw)
        ts, dtext, st = self.when_ts(s); cv = self.cov()
        clock = Q.find_clock(s)
        if st == "noclock": return None
        hh = f"{clock // 3600:02d}:{clock % 3600 // 60:02d}:{clock % 60:02d}"
        if st == "nodata":
            return (f"I have no data for {dtext} - the recorded history covers {self.span_text()}. So I can't tell you the {NAME.get(m, 'vehicle state').lower()} at {hh} on that date, and I won't guess one.")
        what = NAME.get(m, "Vehicle state")
        if st == "gap":
            sp = Q.state_at(self.c, ts)
            near = []
            if sp.get("prev"): near.append(f"the last sample before was at {self.fts(sp['prev'])}")
            if sp.get("next"): near.append(f"the next at {self.fts(sp['next'])}")
            return f"No signal was recorded at {self.fts(ts)}: the car was switched off or asleep then (the car records only while it is on or charging). {'; '.join(near).capitalize() if near else ''}."
        stt = Q.state_at(self.c, ts)
        src = ("1 Hz sample" if stt.get("tier") == "1 Hz" else "per-minute average (the 1 Hz data is kept 7 days)")
        head = f"{what} at {self.fts(ts)}"
        if m is None or m == "gps":
            ans = f"{head}: {Q.fmt_state(stt)}. ({src}.)"
        elif m == "tyre":
            e = stt.get("ext")
            if not e: ans = "This vehicle has no tyre-pressure data recorded, so I can't give a value for that time."
            else:
                ans = (f"Tyre pressure at {self.fts(ts)} (last reading {A.hhmm(e['ts'])}): front left {e['tyres'][0]:.0f}, front right {e['tyres'][1]:.0f}, rear left {e['tyres'][2]:.0f}, rear right {e['tyres'][3]:.0f} kPa at 20 C "
                       f"(raw {', '.join(f'{x:.0f}' for x in e['raw_tyres'])} kPa at the tyres' own temperature).")
        else:
            v = stt[m]; ans = f"{head}: {fmt_val(m, v)}. ({src}.)"
            if "accel" in stt and m == "speed": ans += f" Acceleration at that second {stt['accel']:+.2f} m/s2."
            if m == "rpm": ans += f" At {stt['speed']:.1f} km/h. " + self.note_rpm()
        if dtext.endswith("at that time)"): ans += f" (Date: {dtext}.)"
        self.ctx.update(point=ts); return ans

    # ------------------------------------------------------------------ extremes
    def extreme(self, m, kind, lo, hi):
        col = "speed" if m == "rpm" else m; cv = self.cov(); hz = cv["hz_lo"]; best = None; mx = kind == "max"; sc = Q.SIGS[col][1]
        if hz is not None and hi > hz:
            extra = " AND flow>30" if (col == "flow" and not mx) else ""
            r = self.c.execute(f"SELECT ts, {col} FROM sig_1s WHERE ts>=? AND ts<?{extra} ORDER BY {col} {'DESC' if mx else 'ASC'}, ts DESC LIMIT 1", (max(lo, hz), hi)).fetchone()
            if r: best = (r[1] / sc, r[0], "1 Hz")
        hi2 = min(hi, hz) if hz is not None else hi
        if lo < hi2:
            c1 = {"batt": "batt_max", "flow": "flow_min"}.get(col, col) if ((col == "batt" and mx) or (col == "flow" and not mx)) else col
            extra = " AND flow>30" if (col == "flow" and not mx) else ""
            r = self.c.execute(f"SELECT ts, {c1} FROM sig_1m WHERE ts>=? AND ts<?{extra} ORDER BY {c1} {'DESC' if mx else 'ASC'}, ts DESC LIMIT 1", (lo, hi2)).fetchone()
            if r and r[1] is not None:
                cand = (r[1] / sc, r[0], "per-minute")
                if best is None or (cand[0] > best[0] if mx else cand[0] < best[0]): best = cand
        return best

    def h_extreme(self, s, raw):
        m = Q.metric_of(s.split(" at the ")[0]) or Q.metric_of(s)
        am = None
        mm = re.search(r"at the (highest|lowest|maximum|minimum|max|min|peak|fastest|slowest)[ -]?(speed|power|temperature|temp|rpm|flow)?", s)
        kind = "min" if re.search(r"lowest|minimum|\bmin\b|least|slowest|smallest", s) else "max"
        if mm:
            kind = "min" if mm[1] in ("lowest", "minimum", "min", "slowest") else "max"; am = Q.metric_of(mm[2] or "speed") or "speed"
            if am == "batt" and mm[2] == "temp": am = "batt"
        if m in (None, "tyre", "cabin", "throttle", "steering", "radar", "engine_load", "gear"): return None
        lo, hi, wtxt = self.win(s, default=(self.cov()["lo"], self.cov()["hi"] + 1, "the entire recorded history"))
        key = am or m
        r = self.extreme(key, kind, lo, hi)
        if not r: return f"I have no {NAME[key].lower()} data for {wtxt}. The recorded history covers {self.span_text()}."
        v, ts, tier = r; stt = Q.state_at(self.c, ts)
        if key == "rpm": v = Q.rpm_of(v)
        word = "highest" if kind == "max" else "lowest"
        if am and am != m:
            tv = stt.get(m)
            if tv is None: return f"The {word} {NAME[am].lower()} in {wtxt} was {fmt_val(am, v)} at {self.fts(ts)} ({tier} data), but there is no sample of {NAME[m].lower()} at that second."
            ans = f"At the {word}-{am} point {wtxt if wtxt.startswith('yesterday') or wtxt.startswith('today') else 'in ' + wtxt} ({self.fts(ts)}, {fmt_val(am, v)}, {tier} data) the {NAME[m].lower()} was {fmt_val(m, tv)}."
        else:
            ans = f"The {word} {NAME[m].lower()} in {wtxt} was {fmt_val(m, v)} at {self.fts(ts)} ({tier} data)."
        if m == "rpm" or am == "rpm" or key == "rpm": ans += " " + self.note_rpm()
        if "speed" in stt and m not in ("speed", "rpm") and am is None: ans += f" Other signals then: speed {stt['speed']:.0f} km/h, pack {stt['batt']:.1f} C, SoC {stt['soc']:.0f}%."
        if m == "speed" and "peak speed is not the top speed" and False: pass
        if tier == "per-minute" and m in ("speed", "rpm", "soc", "power"): ans += f" (Older than 7 days, so only per-minute averages exist - a shorter {'peak' if kind == 'max' else 'dip'} may have gone {'higher' if kind == 'max' else 'lower'}.)"
        self.ctx.update(point=ts); return ans

    # ------------------------------------------------------------------ events
    def h_events(self, s, raw):
        k = self.kind_in(s)
        if k in (None, "major"): k = "warning" if k is None else "fault"
        if k == "tyre" and not re.search(r"diagnostic|dtc|trouble|\bgps\b|position|speed|rpm|state", s): return None          # tyre warnings are the tyre module's job
        n = re.search(r"\blast (\d+)\b|\b(\d+) (most recent|latest|last)\b", s); n = int(n[1] or n[2]) if n else None
        count = bool(re.search(r"how many|count|number of|how often|how frequently", s)); first = bool(re.search(r"\bfirst\b|earliest", s)); last1 = bool(re.search(r"\blast\b|latest|most recent|when was|when did|when is", s)) and not n
        lo, hi, wtxt = self.win(s, default=(0, self.now + 1, f"in the whole history ({self.span_text()})"))
        rows = self.evs(k, lo, hi, newest_first=not first)
        mv = re.search(r"(faster|higher|above|over|exceeding|more) than (\d+) ?km", s) or re.search(r"(above|over|exceeding) (\d+) ?km", s); stat = re.search(r"stationary|standing still|not moving|at a standstill|was it moving|vehicle moving", s)
        need_state = bool(re.search(r"\brpm\b|speed|temperature|\bstate\b|steering|acceleration|accelerating|throttle|stationary|moving|at the event|at that|show .*at the", s))
        need_pos = bool(re.search(r"\bgps\b|position|location|route|where", s))
        label = Q.KIND_NAME.get(k, k)
        if not rows:
            if k == "overheat":
                mx = self.c.execute("SELECT MAX(batt_max), ts FROM sig_1m WHERE ts>=? AND ts<?", (lo, hi)).fetchone()
                extra = f" The highest pack temperature in that period was {mx[0] / 10:.1f} C at {self.fts(mx[1])}." if mx and mx[0] is not None else ""
                prev = self.evs(k, 0, lo, newest_first=True, limit=1) if lo > 0 else []
                hot = ""
                if mx and mx[0] is not None and re.search(r"\bat the event|rpm|speed", s):
                    st = Q.state_at(self.c, mx[1]); hot = " For reference, at that hottest moment: " + Q.fmt_state(st, True) + "."
                return f"No battery over-temperature warning was recorded {wtxt}." + extra + (f" The last one before that was at {self.fts(prev[0][0])}." if prev else (" None is recorded anywhere in this vehicle's history." if lo > self.cov()["lo"] + 1 else "")) + hot
            prev = self.evs(k, 0, lo, newest_first=True, limit=1) if lo > 0 else []
            whole = lo <= self.cov()["lo"] + 1
            return f"No {label} was recorded {wtxt}." + (f" The most recent one before that was {self.evrow(prev[0])}." if prev else ("" if whole else f" None is recorded anywhere in {self.span_text()}.")) + (" The car has no ADAS use recorded at all in this period." if k == "adas" and whole else "")
        extra = ""
        rows_all = rows
        if mv: thr = float(mv[2]); rows2 = [r for r in rows if Q.state_at(self.c, r[0]).get("speed", -1) > thr]; extra = f" Of {len(rows)}, {len(rows2)} happened while the car was faster than {thr:.0f} km/h."; rows = rows2
        if not rows:
            sp = [Q.state_at(self.c, r[0]).get("speed") for r in rows_all]
            return f"None of the {len(rows_all)} {label}(s) happened while the vehicle was faster than {thr:.0f} km/h - their speeds were " + ", ".join(f"{x:.0f} km/h" if x is not None else "n/a (car off)" for x in sp) + "."
        self.ctx.update(event=rows[0], events=rows)
        self.out_facts = {"count": len(rows), "type": (Q.KINDS[k][0] if k in Q.KINDS else k), "items": [{"ts": r[0], "time": self.fts(r[0]), "detail": Q.safe_text(r[3])} for r in rows[:50]]}
        if count:
            reasons = Counter(Q.safe_text(r[3]) for r in rows if not Q.flagged(r[3])).most_common(4); days = len({A.day_of(r[0]) for r in rows})
            return (f"{plural(len(rows), label)} {wtxt}, on {plural(days, 'day')}; first {self.fts(min(r[0] for r in rows))}, last {self.fts(max(r[0] for r in rows))}." + (" By reason: " + "; ".join(f"{v} x {k_}" for k_, v in reasons) + "." if reasons and len(reasons) > 1 or (reasons and reasons[0][0]) else "") + extra)
        lim = 1 if (last1 or first) else (n or (12 if not mv else 30)); show = rows[:lim]
        head = (f"The {'first' if first else 'last'} {label}" if lim == 1 else f"{'The last ' + str(lim) if n and len(rows) >= lim else 'The only' if n and len(rows) == 1 else 'All'} {label}{'s' if len(rows) != 1 else ''} {wtxt}") + (f" ({len(rows)} found)" if lim > 1 and len(rows) > lim else "") + ":"
        out = [head] + ["  - " + self.evrow(r, with_state=need_state) for r in show]
        if need_pos:
            for r in show[:8]:
                la, lo_, pl = self.where(r[0])
                if la: out.append(f"    position at {A.hhmm(r[0])}: {la:.4f}, {lo_:.4f}" + (f" ({pl})" if pl else ""))
        if lim > 1 and len(rows) > lim: out.append(f"  ... and {len(rows) - lim} more.")
        if extra: out.append(extra.strip())
        if stat and len(show) >= 1:
            sp = Q.state_at(self.c, show[0][0]).get("speed")
            out.append("Yes - it was stationary (0 km/h) when it appeared." if sp is not None and sp < 1 else (f"Almost - it was creeping at {sp:.0f} km/h (pulling away or stopping) when it appeared." if sp is not None and sp < 6 else f"No - it was moving at {sp:.0f} km/h when it appeared." if sp is not None else "There is no speed sample at that moment (the car was off)."))
        if k == "brake": out.append("(Hard braking = deceleration above 0.28 g from the accelerometer; the friction brakes have no event log of their own - pads are at about " + (f"{self.c.execute('SELECT pad FROM vday ORDER BY ts DESC LIMIT 1').fetchone()[0]:.0f}%" if self._has("vday") else "unknown") + ".)")
        if re.search(r"steering|throttle|pedal", s): out.append("Steering angle and throttle position are not recorded by this car, so they are not shown.")
        if k == "diagnostic" and len(show) < (n or 1) and n and len(show) > 1: out.append(f"Only {len(rows)} diagnostic trouble code event(s) exist in the whole history, so that is all there is to show.")
        return "\n".join(out)

    def _has(self, tbl):
        try: self.c.execute(f"SELECT 1 FROM {tbl} LIMIT 1").fetchone(); return True
        except Exception: return False

    def h_cluster(self, s, raw):
        evs = self.c.execute("SELECT ts, type, sev, detail FROM events WHERE type IN (%s) AND ts<=? ORDER BY ts" % ",".join("?" * len(WARN_TYPES)), (*WARN_TYPES, self.now)).fetchall()
        out = []; pts = []; inc = []
        for e in evs:
            if not inc or e[0] - inc[-1][0] > 90: inc.append(e)
        for e in evs:
            la, lo, pl = self.where(e[0])
            if la: pts.append((e, la, lo, pl))
        if re.search(r"correlate|gps|position", s) and not re.search(r"concentrat|same (route )?location|cluster", s):
            if not pts: return "None of the diagnostic events has a GPS position on record."
            self.brief = f"{len(pts)} of {len(evs)} warning events have a GPS position" + (f"; they are spread over several kilometres, so the place is not the cause." if len(pts) >= 2 and max(AX.haversine_km((a[1], a[2]), (b[1], b[2])) for a in pts for b in pts) > 1.0 else ".")
            out = [f"Diagnostic / warning events with the car's position ({len(pts)} of {len(evs)} have a fix):"]
            for e, la, lo, pl in pts[:12]: out.append(f"  - {self.evrow(e)} - at {la:.4f}, {lo:.4f}" + (f" ({pl})" if pl else ""))
            if len(pts) >= 2:
                d = max(AX.haversine_km((a[1], a[2]), (b[1], b[2])) for a in pts for b in pts)
                out.append(f"The events are spread over up to {d:.1f} km. " + ("They are not tied to one place; the faults are in the vehicle's systems, not the road." if d > 1.0 else "They happened close together."))
            return "\n".join(out)
        pts = [p for p in pts if p[0] in inc]
        if len(pts) < 2: return f"Only {len(inc)} separate warning/fault incident(s) exist ({len(pts)} with a position), so no location pattern can be established - concentration needs at least several incidents."
        cells = Counter((round(la / 0.005), round(lo / 0.007)) for e, la, lo, pl in pts); (cell, k), = cells.most_common(1)
        if k >= 2: near = [p for p in pts if (round(p[1] / 0.005), round(p[2] / 0.007)) == cell]; return f"{k} of {len(pts)} located warning incidents occurred within about 500 m of {near[0][1]:.4f}, {near[0][2]:.4f}" + (f" ({near[0][3]})" if near[0][3] else "") + "."
        return f"No. The {len(pts)} located warning incidents (events within 90 s of each other counted once) are at {len(cells)} different places (no two within about 500 m), so they are not concentrated on one stretch of road."

    # ------------------------------------------------------------------ the signals around an event
    def sig_lines(self, sm, before_note=""):
        out = []
        for m, lab in (("speed", "speed"), ("power", "battery power"), ("batt", "pack temperature"), ("flow", "coolant flow"), ("motor", "motor temperature"), ("soc", "state of charge"), ("amb", "outside temperature")):
            x = sm.get(m)
            if not x: continue
            out.append(f"  - {lab}: {x['first']:.1f} -> {x['last']:.1f} {UNIT[m]} (min {x['min']:.1f}, max {x['max']:.1f} {UNIT[m]})")
        return out

    def h_ctx(self, s, raw):
        dur = Q.find_duration(s)
        imm = bool(re.search(r"immediately|just before|just after", s)); after = bool(re.search(r"\bafter\b|following|afterwards|afterward", s)) and not re.search(r"before and after|before or after|before after", s)
        between = bool(re.search(r"between .*(before|prior).*(and|&).*after|before and after|before vs after", s))
        e, lab, note, ab = self.anchor(s)
        if e is None and ab is None: return note or "There is no such event in the recorded data to look around."
        ts = e[0] if e else ab["ts"]; desc = (f"{self.evrow(e)}" if e else lab)
        b = int(dur if dur else (30 if imm else 600)); a = int(dur if (dur and (after or between)) else (30 if imm else 300))
        if re.search(r"accelerat|decelerat", s): b = int(dur or 30)
        if re.search(r"signals changed|suspicious|weird|unusual", s) and not dur: b = 3600
        sg = Q.event_sig(self.c, ts, max(b, 5), max(a, 5)) if (b or a) else None
        out = []
        if note: out.append(note.strip())
        out.append(f"Anchor: {desc}")
        if sg is None or len(sg) < 3: return "\n".join(out + ["There are no signal samples around that moment."])
        res = f"{sg.tier} data" + (f", one sample per {sg.res} s" if sg.res != 1 else "")
        self.ctx.update(event=e, anchor_ts=ts)
        if re.search(r"accelerat|decelerat", s):
            pre = Q.summarise(sg, ts - b, ts + 1)
            if not pre: return "\n".join(out + ["No samples in the seconds before that moment."])
            sp = pre["speed"]; ac = Q.slope([t for t in sg.ts if ts - b <= t <= ts], [v for t, v in zip(sg.ts, sg.cols["speed"]) if ts - b <= t <= ts]) / 3.6
            verdict = "accelerating" if ac > 0.15 else ("decelerating" if ac < -0.15 else "cruising at a steady speed" if sp["mean"] > 5 else "standing still")
            out.append(f"In the {b} s before it the vehicle was {verdict}: speed {sp['first']:.1f} -> {sp['last']:.1f} km/h (trend {ac:+.2f} m/s2). ({res}.)")
            return "\n".join(out)
        if between:
            pre, post = Q.summarise(sg, ts - b, ts), Q.summarise(sg, ts, ts + a + 1)
            if not post or not pre: return "\n".join(out + ["There is not enough data on both sides of that moment to compare."])
            out.append(f"Averages over the {Q.dur(b)} before vs the {Q.dur(a)} after ({res}):")
            for m, lab in (("speed", "speed"), ("power", "battery power"), ("batt", "pack temperature"), ("flow", "coolant flow"), ("motor", "motor temperature"), ("soc", "state of charge"), ("amb", "outside temperature"), ("rpm", "est. motor speed")):
                out.append(f"  - {lab}: {pre[m]['mean']:.1f} -> {post[m]['mean']:.1f} {UNIT[m]} ({SIGNED(post[m]['mean'] - pre[m]['mean'])})")
            return "\n".join(out)
        lo_, hi_ = (ts, ts + a + 1) if after else (ts - b, ts + 1)
        sm = Q.summarise(sg, lo_, hi_)
        if not sm: return "\n".join(out + [f"There are no samples in the {'after' if after else 'before'} window."])
        span = a if after else b
        out.append(f"{'After' if after else 'Before'} it ({Q.dur(span)}, {sm['speed']['n']} samples, {res}):")
        out += self.sig_lines(sm)
        try:
            ph = []
            b_ = sm["batt"]
            if abs(b_["last"] - b_["first"]) >= 1: ph.append(f"the pack {'warmed' if b_['last'] > b_['first'] else 'cooled'} from {b_['first']:.0f} to {b_['last']:.0f} DEG")
            f_ = sm["flow"]
            if f_["mean"] > 0.3 and abs(f_["last"] - f_["first"]) >= 1.5: ph.append(f"coolant flow went from {f_['first']:.1f} to {f_['last']:.1f} L/min")
            elif f_["mean"] > 0.3: ph.append(f"coolant flow stayed near {f_['mean']:.1f} L/min")
            w_ = sm["power"]
            if abs(w_["mean"]) >= 3: ph.append(f"{'charging at' if w_['mean'] < 0 else 'drawing'} about {abs(w_['mean']):.0f} kW")
            sp_ = sm["speed"]
            ph.append("the car was standing still" if sp_["max"] < 1 else f"speed averaged {sp_['mean']:.0f} km/h")
            self.brief = f"In the {Q.dur(span)} {'after' if after else 'before'} it: " + "; ".join(ph[:4]) + "."
            self.brief = self.brief.replace("DEG", "\u00b0C")
        except Exception: pass
        evs = self.c.execute("SELECT ts, type, detail FROM events WHERE ts>=? AND ts<=? AND type NOT IN ('HARSH_ACCEL','HARSH_BRAKE','HARSH_CORNER') ORDER BY ts", (lo_, hi_)).fetchall()
        if evs: out.append("  - events in that window: " + "; ".join(f"{A.hhmm(t)} {Q.KIND_NAME.get({'WARN_BATT_OVERTEMP': 'overheat', 'DTC_SET': 'diagnostic', 'ADAS_DISENGAGE': 'adas', 'CHARGE_START': 'charging', 'TPMS_LOW': 'tyre', 'CAMERA_BLOCKED': 'camera', 'CAMERA_DEGRADED': 'camera'}.get(ty, ''), ty)} {Q.safe_text(d)[:60]}" for t, ty, d in evs[:6]))
        hb = self.c.execute("SELECT COUNT(*) FROM events WHERE ts>=? AND ts<=? AND type IN ('HARSH_BRAKE','HARSH_ACCEL','HARSH_CORNER')", (lo_, hi_)).fetchone()[0]
        if hb: out.append(f"  - hard braking / acceleration / cornering events in the window: {hb}")
        try: lg = self.c.execute("SELECT ts, src, msg FROM logs WHERE ts>=? AND ts<=? ORDER BY ts LIMIT 5", (lo_, hi_)).fetchall()
        except Exception: lg = []
        if lg: out.append("  - log lines: " + "; ".join(f"{A.hhmm(t)} [{sr}] {Q.safe_text(m_)[:70]}" for t, sr, m_ in lg if not Q.flagged(m_)))
        if "rpm" in s: out.append(self.note_rpm())
        if e is not None and not after and e[1] in ("WARN_BATT_OVERTEMP", "DTC_SET") and re.search(r"thermal|coolant|P1A|temperature", e[3] or "", re.I) and re.search(r"what happened|walk me|sequence|leading|precede", s):
            try:
                d = self.diag_event(e)
                if d: out.append(f"What the data suggests: {d['alts'][0]['name']} ({d['alts'][0]['strength']}) - {d['alts'][0]['support']}.")
            except Exception: pass
        return "\n".join(out)

    def h_proximity(self, s, raw):
        dur = Q.find_duration(s, 30); e, lab, note, ab = self.anchor("last abnormal reading" if re.search(r"abnormal|anomal", s) else s)
        ts = ab["ts"] if ab else (e[0] if e else None)
        if ts is None: return note or "There is no abnormal reading on record to compare against."
        near = self.c.execute("SELECT ts, type, detail FROM events WHERE type='DTC_SET' AND ts BETWEEN ? AND ?", (ts - dur, ts + dur)).fetchall()
        what = lab if ab else self.evrow(e)
        if near: return f"Yes. {what} at {self.fts(ts)}; a diagnostic event {Q.safe_text(near[0][2])} was recorded at {self.fts(near[0][0])}, {abs(near[0][0] - ts):.0f} s away."
        prev = self.c.execute("SELECT ts, detail FROM events WHERE type='DTC_SET' AND ts<=? ORDER BY ts DESC LIMIT 1", (ts,)).fetchone()
        extra = ""
        if prev: extra = f" The nearest earlier diagnostic event was {Q.safe_text(prev[1])} at {self.fts(prev[0])} ({Q.dur(ts - prev[0])} before). A trouble code is set once, when a condition is first met, and stays stored - so an ongoing abnormal reading doesn't create a new code each time."
        return f"No. {what[0].upper() + what[1:]} was at {self.fts(ts)}, and no diagnostic event was recorded within {dur:.0f} s of it.{extra}"
