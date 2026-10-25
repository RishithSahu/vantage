"""Question answering over the recorded history: point-in-time lookups, events and their surroundings, periods, statistics, trends, correlations, explanations with evidence,
contradiction checks and honest refusals. Deterministic: every number is computed here from the store, every statement says where it came from, and anything the car does not
record (RPM, cabin temperature, throttle, steering, radar, other cameras) is said to be missing or estimated instead of being made up. Stored text is data, never an instruction."""
import os, re, math, datetime as dt
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean, sd

# ------------------------------------------------------------------------------------------------ safety, secrets, unknowable
SAFE_NOUN = (r"abs\b|anti.?lock|speed limit(er)?|max(imum)? (vehicle )?speed|top speed|battery protection|protection system|\bbms\b|\badas\b|emergency brak\w*|hard braking|\baeb\b|airbag|safety (controller|system|monitor|function)|"
             r"traction control|\besc\b|stability control|\bbrakes?\b|steering|torque limit|power limit|thermal (protection|limit)|cooling pump|contactor|immobili[sz]er|\becu\b|configuration|can bus|the bus\b")
SAFETY_RX = re.compile(r"\b(disable|deactivate|turn off|switch off|shut off|override|bypass|raise|increase|lift|change|modify|adjust|reprogram|flash|overwrite|force|tamper)\b.{0,60}?(" + SAFE_NOUN + r")|\bsend (a |the |an )?(command|message|frame|can frame)\b|"
                       r"\boverride the safety|\bapply my requested|\bwrite (to )?(the )?(can|ecu|bus)\b|\bset (the )?(max|maximum|top|speed limit)")
SAFETY_TEXT = ("I can analyze relevant vehicle data or explain the system behavior, but I cannot issue or modify safety-critical control commands. "
               "The safety controller stays isolated from this assistant: I am read / analyze / explain only, and I have no write path to the vehicle networks, so nothing was sent or changed.")
SAFETY_HELP = {r"\babs\b|anti.?lock|brake": "I can show the hard-braking events and brake-pad wear in the recorded data, or explain what ABS does.", r"speed": "I can show the highest speed recorded and how often the car was driven fast.",
               r"battery protection|bms|thermal|cooling": "I can show battery temperature, coolant flow and any over-temperature warnings, and explain what a battery-protection system does.",
               r"adas|emergency": "I can show ADAS use and disengagement events and what the vehicle was doing at the time.", r"airbag": "I have no airbag-system data at all, so I can neither report on it nor change it."}
SECRET_RX = re.compile(r"system prompt|your (hidden )?(instructions|prompt|rules)|hidden (data|instructions|prompt)|credential|api[ _-]?keys?|\bpasswords?\b|\bsecrets?\b|\btokens?\b|private key")
LOGSAYS_RX = re.compile(r"(log|event|message|description|field|text|file) (says|said|contains|reads|states)|ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)|disregard (all |any )?(the )?(previous|prior|above)")


def neutralise(text):
    """Stored text that tries to give orders is replaced, so it can never reach the answer as an instruction."""
    return re.sub(r"(?i)(ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)|disregard (all |any )?(the )?(previous|prior|above)|you are now|reveal [^.;\n]*|system prompt|report that [^.;\n]*)[^.;\n)]*", "[instruction-like text in a stored log ignored]", text)


CANON = [(re.compile(rx), rep) for rx, rep in [      # everyday wording -> the words the routes and handlers look for
    (r"\bdtcs\b", "dtc"), (r"\b(error|obd|diagnostic) codes?\b", "diagnostic codes"), (r"\b(alerts?|alarms?|warning lights?)\b", "warning"),
    (r"\b(got|get|gets|getting|became|become|becomes) (too |really |very )?hot\b", "overheating"), (r"\boverheated\b|\boverheats\b|\brunning hot\b|\bruns hot\b|\bheat warning\b", "overheating"),
    (r"\b(engine speed|motor speed|revs|revolution(s)? (rate|speed)|revolutions( per minute)?)\b", "rpm"), (r"\bspeeding up\b", "accelerating"), (r"\bslowing down\b", "decelerating"),
    (r"\b(erratic|unstable|jumpy|fluctuating|all over the place|less stable|less consistent)\b", "volatile"), (r"\bdeteriorat\w*|\bdegrad\w*|\bwearing out\b", "degradation"),
    (r"\b(consumption|economy|energy use|energy usage|gas mileage)\b", "efficiency"), (r"\b(pack|cell) (temp|temperature)\b|\bbattery (temp|heat)\b", "battery temperature"),
    (r"\bcar'?s\b", "vehicle"), (r"\bthe car\b", "the vehicle"), (r"\bbehaviour\b", "behavior"), (r"\bstandstill\b|\bstopped\b", "idle"),
    (r"\bgotten\b", "got"), (r"\blane (keeping|keep|assist|centering|centring|support)( system| assist)?\b(?= (cut out|cuts out|cut-out|gave up|given up|gives up|dropped out|drop-?out|switched|turned|disengag|last|latest))", "adas"), (r"\b(cut out|cuts out|cut-out|given up|gave up|gives up|drop-?out|dropped out|switched itself off|turned itself off|turns itself off)\b", "disengagement"),
    (r"\b(emergency|panic|sudden|hard|harsh|heavy|really hard) (stop|stops|braking|brake|brakes|braked)\b", "hard braking"), (r"\bbraked (really )?hard\b", "hard braking"), (r"\bemptiest\b|\bleast charge\b", "lowest state of charge"),
    (r"\bchain of events\b", "sequence of events"), (r"\bper kilomet(re|er)\b", "per km"), (r"\blane\b(?! assist)(?= (assist|keeping))", "lane")
]]


NUMW = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def prep(raw):
    s = re.sub(r"\s+", " ", raw.lower().replace("’", "'").replace("“", '"').replace("”", '"')).strip().rstrip("?.! ")
    s = re.sub(r"(\d) ?: ?(\d)", r"\1:\2", s)
    s = re.sub(r"\btires?\b", "tyre", s)
    s = re.sub(r"\b(highway|freeway)\b", "motorway", s)
    for rx, rep in CANON:
        s = rx.sub(rep, s)
    s = re.sub(r"\b(" + "|".join(NUMW) + r") (percent|%)", lambda m: f"{NUMW[m[1]]}%", s)
    return s.replace("engine temp", "motor temp")


from .qa_core import F1, F0, SIGNED, plural


STRONG_PHRASE = {"ctx": re.compile(r"^(?!.*\b(longest|most serious|worst|biggest|strongest|largest|highest|idle|first|root cause)\b).*?\b(what|how|which)\b.{0,25}\b(happen\w*|chang\w*|signals?|went)\b.{0,25}\b(before|after)\b")}
WEAK_ROUTES = {"events", "point", "extreme", "trend", "ctx", "longest", "duration", "missing", "memory", "cooccur", "speedwin", "unknowable", "repeat", "prove", "obs_inf", "calc", "degrade", "predict_before", "cf_speed", "proto_map", "evidence", "alt", "conf"}
ROUTE_INTENT = {"advice": "HEALTH", "signal_why": "STATUS", "drive_safely": "DRIVING", "style": "DRIVING", "regen": "DRIVING", "style_trend": "DRIVING", "trip_count": "DRIVING", "battery_now": "HEALTH", "pump_ok": "HEALTH", "v12": "HEALTH", "advice": "HEALTH", "last_charge": "EVENTS", "charge_cost": "EVENTS", "charging_now": "STATUS", "reach": "STATUS", "top_speed": "SUMMARY", "events": "EVENTS", "ctx": "BEFORE_EVENT", "trend": "TREND", "extreme": "SUMMARY", "point": "STATUS", "conditions": "LARGEST", "compare_periods": "COMPARE", "changes": "LARGEST", "degrade": "TREND", "why_range2": "WHY_RANGE",
                "speedwin": "SUMMARY", "checkup": "HEALTH", "health_summary": "HEALTH", "trip": "USAGE", "day": "USAGE", "safety": "NONE", "inject": "NONE", "unknowable": "NONE", "missing": "NONE", "missing_soft": "NONE", "precision": "NONE", "fleet": "NONE"}


class QABase:
    def __init__(self, brain):
        self.b = brain; self.hist = []; self.ctx = {}; self.raw = False; self.errors = []; self.brief = None
        self._cache = {}

    # ------------------------------------------------------------------ plumbing
    @property
    def c(self): return self.b.c

    @property
    def now(self): return self.b.now

    @property
    def tz(self): return self.b.tz

    def cov(self):
        k = ("cov", self.now)
        if k not in self._cache: self._cache = {k: Q.coverage(self.c, self.now)}
        return self._cache[k]

    def fts(self, ts): return Q.ts_s(ts)
    def day(self, ts): return A.day_of(ts)

    def today(self): return dt.datetime.fromtimestamp(self.now + self.tz, dt.timezone.utc).date()

    def span_text(self):
        cv = self.cov(); return f"{self.day(cv['lo'])} to {self.day(cv['hi'])}"

    def win(self, s, default=None):
        """-> (lo, hi, text) from the words of the question, else the default (a (lo, hi, text) triple)."""
        cv = self.cov(); d, expl, dtxt = Q.find_date(s, self.now, self.tz)
        if d:
            lo, hi = Q.day_bounds(d, self.tz); return lo, hi, (dtxt if expl else f"{dtxt} ({d.isoformat()})")
        if re.search(r"first (recorded )?week|first seven days|beginning of (the )?(data|history)", s): return cv["lo"], cv["lo"] + 7 * 86400, "the first recorded week"
        if re.search(r"(final|last recorded|most recent|latest|current) week|recent week", s) and not re.search(r"last week", s): return cv["hi"] - 7 * 86400, cv["hi"] + 1, "the most recent week"
        if re.search(r"entire|whole|all (the )?(history|data|dataset|time|recorded)|ever\b|overall|across|every|so far|lifetime|since the beginning|full (history|dataset)", s): return cv["lo"], cv["hi"] + 1, "the entire recorded history"
        m = re.search(r"(?:last|past|previous) (\d+|two|three|four|five|six|seven|ten) (day|week|month)s?", s)
        if m:
            n = Q.number(m[1]) or 7; days = int(n * (1 if m[2] == "day" else 7 if m[2] == "week" else 30)); return self.now - days * 86400, self.now, f"the last {days} days"
        from . import nlu
        code = nlu.parse_when(s, None)
        if code:
            lo, hi = A.window(self.now, code, self.c); return lo, hi, A.WHEN_TEXT.get(code, code)
        return default if default else (None, None, None)

    def recent(self, days=7):
        cv = self.cov(); hi = cv["hi"] + 1; lo = hi - days * 86400
        if days <= 7 and cv["hz_lo"] is not None and lo < cv["hz_lo"]: lo = cv["hz_lo"]                     # stay inside the 1 Hz tier
        return lo, hi, f"the last {days} days of recorded data"

    def sig(self, lo, hi, prefer="auto"):
        key = (lo, hi, prefer, self.raw, self.cov()["hi"])
        sc = self.__dict__.setdefault("_sigc", {})
        if key in sc: return sc[key]
        sg = Q.load(self.c, max(lo, 0), hi, prefer="1s" if (self.raw and prefer == "auto" and lo >= (self.cov()["hz_lo"] or 1 << 60)) else prefer)
        if len(sg) > 20000: self._sigc = {key: sg}                                   # keep only the latest large window (a week of 1 Hz is tens of MB)
        return sg

    def place_name(self, lat, lon, near_km=0.45):
        try: rows = self.c.execute("SELECT name, lat, lon FROM places").fetchall()
        except Exception: rows = []
        best = min(((AX.haversine_km((lat, lon), (r[1], r[2])), r[0]) for r in rows), default=None)
        if best and best[0] <= near_km: return best[1]
        if best: return f"{best[0]:.1f} km from {best[1]}"
        return None

    def where(self, ts):
        st = Q.state_at(self.c, ts)
        g = st.get("gps")
        if not g: return None, None, None
        return g["lat"], g["lon"], self.place_name(g["lat"], g["lon"])

    def evrow(self, e, with_state=False, fields="brief"):
        ts, ty, sev, det = e; lab = Q.KIND_NAME.get({"WARN_BATT_OVERTEMP": "overheat", "DTC_SET": "diagnostic", "ADAS_DISENGAGE": "adas", "HARSH_BRAKE": "brake", "HARSH_ACCEL": "accel", "HARSH_CORNER": "corner", "TPMS_LOW": "tyre",
                                                                      "CAMERA_BLOCKED": "camera", "CAMERA_DEGRADED": "camera", "CHARGE_START": "charging", "CAN_ANOMALY": "can"}.get(ty, ""), ty)
        d = Q.safe_text(det)
        if Q.flagged(det): d = "[stored text looked like an instruction - treated as data and ignored]"
        s = f"{self.fts(ts)}  {lab}" + (f": {d}" if d else "")
        if with_state:
            st = Q.state_at(self.c, ts)
            if "speed" in st: s += f"\n      speed {st['speed']:.0f} km/h, est. motor speed {st['rpm']:.0f} rpm, pack {st['batt']:.1f} C, coolant flow {st['flow']:.1f} L/min, power {st['power']:.1f} kW"
            else: s += "\n      (no signal sample at that second - car was off)"
        return s

    def note_rpm(self): return "RPM note: this car has no RPM sensor (it is an EV); I estimate motor speed from vehicle speed x a fixed 9:1 reduction ratio and 0.33 m wheel radius, so treat rpm as a derived estimate."

    def note(self, q, answer, intent, p, facts):
        """Remember an answer that came from the older intent pipeline, so follow-ups ('what evidence supports that?', 'you said ...') can refer to it."""
        self.hist.append({"q": q, "route": "legacy:" + str(intent), "answer": answer})
        if intent == "WHY_RANGE": self.ctx.update(topic="range")
        elif intent in ("TREND", "LARGEST", "USAGE", "COMPARE") and p.get("metric") in (None, "energy_use", "range"): self.ctx.update(topic="range", diag=None)
        elif intent in ("BEFORE_EVENT", "EVENTS", "HEALTH"): self.ctx.update(topic="event", diag=None, event_row=None)

    def say(self, text, topic=None, **kw):
        self.ctx.update(kw)
        if topic: self.ctx["topic"] = topic
        return text

    # ------------------------------------------------------------------ entry point
    def ask(self, raw, t=None, only=None):
        """Run the domain handlers, optionally restricted to named capabilities.

        `only` is used for capabilities that the analytical Query IR cannot
        represent (for example a forward charge-time estimate).  This keeps
        capability arbitration separate from signal/statistic parsing.
        """
        from .qa_final import ROUTES
        from . import qa_clf as _clf
        from . import qa_fuzzy as _fz
        g = self.guard(raw)
        if g: self.hist.append({"q": raw, "route": "guard", "answer": g}); return {"answer": g, "intent": "NONE", "route": "guard", "facts": {"ok": False, "why": "bad date or time", "route": "guard"}}
        s = prep(raw)
        from .qa_fuzzy import OOD
        if OOD.search(s) and not re.search(r"\b(car|vehicle|battery|motor|tyre|tire|brake|trip|drive|driving|charge|range)\b", s): return None          # a recipe, a movie, a capital city: not about the car
        alts = [s] if t is None else [prep(t)] + ([s] if prep(t) != s else [])                 # typo-repaired wording first, then exactly what was typed
        routes = ROUTES if only is None else [(n, rx) for n, rx in ROUTES if n in only]
        top = [(n, rx) for n, rx in routes if n in STRONG_PHRASE and any(STRONG_PHRASE[n].search(x) and rx.search(x) for x in alts)]     # an explicit "what happened before/after ..." goes straight to the context route
        for name0, rx in top + routes:                                                          # priority order of the routes first; either wording (repaired / as typed) may trigger one
            for texts in alts:
                if rx.search(texts):
                    name = name0
                    if name in _clf.EVERYDAY or name in _clf.GENERIC:                  # the classifier may prefer another everyday route when it is clearly sure (more sure when the keyword route is an older, broader one)
                        ct = self.clf_top(raw, t)
                        if ct and ct[0][0][0] in _clf.EVERYDAY and ct[0][0][0] != name and _fz.allowed(ct[0][0][0], texts) and not _fz.blocked(ct[0][0][0], texts):
                            need = _clf.T["override"] if name in _clf.EVERYDAY else _clf.T["any"]
                            lead = ct[0][0][1] - _clf.get().score_of(_clf.get().last_scores, name)                 # how much more the model likes its own pick than the keyword route's
                            if ct[0][0][1] >= need[0] and lead >= need[1]: name = ct[0][0][0]
                    if name in WEAK_ROUTES and not (name in STRONG_PHRASE and STRONG_PHRASE[name].search(texts)):                                         # a broad keyword route only wins when the fuzzy router does not clearly prefer another route
                        fz = self.fuzzy_route(texts)
                        if fz and fz[0] != name and fz[1] >= 0.55: name = fz[0]
                    self.out_facts = None
                    try: r = getattr(self, "h_" + name)(texts, raw)
                    except Exception as e:                                           # a database without the extension tables, or thin data: let the older pipeline answer
                        self.errors.append((name, repr(e))); r = None
                    if r:
                        ans, intent = (r, ROUTE_INTENT.get(name, "ANALYSIS")) if isinstance(r, str) else r
                        self.hist.append({"q": raw, "route": name, "answer": ans}); self.ctx["last_route"] = name; self.ctx["last_q"] = raw
                        bf, self.brief = self.brief, None
                        return {"answer": ans, "intent": intent, "route": name, "brief": bf, "facts": dict(self.out_facts or {}, ok=True, route=name)}
        return None

    def guard(self, raw):
        """A date that does not exist, a time of day that does not exist, or a date outside the recorded history: say so instead of answering about some other day."""
        s = prep(raw)
        if SAFETY_RX.search(s) or SECRET_RX.search(s): return None
        d, expl, dtxt = Q.find_date(s, self.now, self.tz)
        if expl and d is None and dtxt.startswith("INVALID"): return f"'{dtxt[8:]}' is not a real calendar date, so I can't look anything up for it."
        m = re.search(r"(?<![\d:])(\d{1,3}):(\d{2})(?::(\d{2}))?(?!\d)", s)
        if m and not re.search(r"\b(am|pm)\b", s[m.end():m.end() + 4]) and (int(m[1]) > 23 or int(m[2]) > 59 or (m[3] and int(m[3]) > 59)): return f"'{m[0]}' is not a valid time of day (hours 0-23, minutes and seconds 0-59)."
        if expl and d is not None:
            cv = self.cov(); lo, hi = Q.day_bounds(d, self.tz)
            if cv["lo"] is not None and (hi < cv["lo"] or lo > cv["hi"] + 86400) and not re.search(r"\b(buy|bought|purchase|manufactur\w*|build|built|service|serviced|warranty)\b", s):
                return f"I have no data for {d.isoformat()} - the recorded history covers {self.span_text()}, so I can't tell you anything about that date and won't guess."
        return None

    def fuzzy_route(self, s):
        from .qa_fuzzy import Fuzzy
        if not hasattr(QABase, "_fuzzy"): QABase._fuzzy = Fuzzy()
        r = QABase._fuzzy.route(s, have_context=bool(self.hist), min_score=0.45)
        return (r[0], r[1]) if r else None

    def clf_top(self, raw, t=None):
        """(best (route, score) list, margin) from the linear classifier, or None when it is not installed. Cached per question."""
        if os.environ.get("VGPT_NO_QA_CLF"): return None
        from . import qa_clf
        c = qa_clf.get()
        if c is None: return None
        key = (raw, t)
        if getattr(self, "_clf_key", None) != key: self._clf_key = key; self._clf_val = c.predict(raw)
        top, margin = self._clf_val
        return (top, margin) if top else None

    def ask_clf(self, raw, t=None, mode="answer"):
        """Answer an everyday question that no keyword rule matched, when the classifier is sure enough (mode 'answer' = strict, 'last' = last resort before giving up)."""
        from . import qa_clf
        ct = self.clf_top(raw, t)
        if not ct: return None
        (name, sc), margin = ct[0][0], ct[1]; lo_sc, lo_mg = qa_clf.T[mode]
        if name not in qa_clf.EVERYDAY or sc < lo_sc or margin < lo_mg: return None
        s = prep(t if t else raw)
        from .qa_fuzzy import allowed, blocked
        if not allowed(name, s) or blocked(name, s): return None
        self.out_facts = None
        try: res = getattr(self, "h_" + name)(s, raw)
        except Exception as e:
            self.errors.append((name, repr(e))); return None
        if not res: return None
        ans, intent = (res, ROUTE_INTENT.get(name, "ANALYSIS")) if isinstance(res, str) else res
        self.hist.append({"q": raw, "route": name, "answer": ans}); self.ctx["last_route"] = name; self.ctx["last_q"] = raw
        bf, self.brief = self.brief, None
        return {"answer": ans, "intent": intent, "route": name, "brief": bf, "example": None, "facts": dict(self.out_facts or {}, ok=True, route=name, clf=round(sc, 2))}

    def clf_says_off_topic(self, raw, t=None):
        from . import qa_clf
        ct = self.clf_top(raw, t)
        if not ct: return False
        (name, sc), margin = ct[0][0], ct[1]; lo_sc, lo_mg = qa_clf.T["gate"]
        return name == "gate" and sc >= lo_sc and margin >= lo_mg

    def ask_fuzzy(self, raw, min_score=0.45, t=None, soft=False):
        """Route by similarity to known phrasings when no keyword route matched. soft=True is the last resort before 'I did not understand': a lower bar, and the caller says which question it was taken for."""
        from .qa_fuzzy import Fuzzy
        if not hasattr(QABase, "_fuzzy"): QABase._fuzzy = Fuzzy()
        s = prep(t if t else raw); r = QABase._fuzzy.soft_route(s, have_context=bool(self.hist), min_score=min_score) if soft else QABase._fuzzy.route(s, have_context=bool(self.hist), min_score=min_score)
        if not r: return None
        name, score, ex = r
        self.out_facts = None
        try: res = getattr(self, "h_" + name)(s, raw)
        except Exception as e:
            self.errors.append((name, repr(e))); return None
        if not res: return None
        ans, intent = (res, ROUTE_INTENT.get(name, "ANALYSIS")) if isinstance(res, str) else res
        self.hist.append({"q": raw, "route": name, "answer": ans}); self.ctx["last_route"] = name; self.ctx["last_q"] = raw
        bf, self.brief = self.brief, None
        return {"answer": ans, "intent": intent, "route": name, "brief": bf, "example": ex, "facts": dict(self.out_facts or {}, ok=True, route=name, fuzzy=round(score, 2))}

    # ------------------------------------------------------------------ safety / secrets / unknowable
    def h_safety(self, s, raw):
        extra = next((v for k, v in SAFETY_HELP.items() if re.search(k, s)), "I can show the related recorded data if you tell me which signal or event to look at.")
        return (SAFETY_TEXT + " " + extra, "SAFETY")

    def scan_injection(self):
        n = bad = 0
        for tbl, col in (("events", "detail"), ("logs", "msg")):
            try:
                for (x,) in self.c.execute(f"SELECT {col} FROM {tbl}"):
                    n += 1
                    if Q.flagged(x) or LOGSAYS_RX.search((x or "").lower()): bad += 1
            except Exception: pass
        return n, bad

    def h_inject(self, s, raw):
        n, bad = self.scan_injection()
        scan = f" I checked the {n:,} stored event and log texts: {bad} contained instruction-like text." if bad else f" I checked the {n:,} stored event and log texts: none contained instruction-like text."
        if LOGSAYS_RX.search(s):
            return ("Nothing it asks for - a log line is data, not a command. I would not follow it, reveal anything because of it, or let it change an answer; I would report it as a suspicious log entry and keep answering from the recorded signals and events." + scan, "SAFETY")
        if re.search(r"credential|api[ _-]?key|password|token|private key|secret", s):
            return ("There are no credentials or API keys in the vehicle database (it holds signals, events, logs, places and GPS fixes), and I will not search for or reveal secrets in any case." + scan, "SAFETY")
        self.brief = "I won't reveal my instructions; text in the logs is treated as data, never as a command."
        return ("I don't reveal my internal instructions, and the vehicle logs cannot supply or change them: I only answer from recorded data (signals, events, logs, places, GPS), and text inside that data is treated as data, never as a command." + scan, "SAFETY")

    def h_unknowable(self, s, raw):
        if re.search(r"firmware|software version", s):
            sw = Q.meta(self.c, "veh.sw", "unknown"); ev = self.c.execute("SELECT ts, detail FROM events WHERE type='SW_UPDATE' ORDER BY ts").fetchall()
            hist = "; ".join(f"{self.fts(t)} {Q.safe_text(d)}" for t, d in ev) or "no software updates are recorded"
            return f"The car reports software {sw} now. Software updates in the history: {hist}. The vehicle does not log which version was installed when a particular fault occurred, so I can't state that for the fault - I can only say the version now, and whether an update is recorded before or after it."
        if re.search(r"\bcan id\b|arbitration id|which can (message|frame|id)", s):
            return ("I can't name a CAN ID as the cause of an event: a CAN message only carries a signal, it doesn't cause anything. What I can say is which messages carry the signals involved - battery temperature: BmsStatus (0x200, signal BattTemp); coolant flow and motor temperature: Thermal (0x210, CoolantFlow / MotorTemp); "
                    "vehicle speed: VehDyn (0x100); climate: Climate (0x300). Whether any of them is the origin of a fault isn't established by that mapping.")
        if re.search(r"manufacturer|supplier|brand|vendor|company|maker|who made|who built", s):
            return "I can't say which manufacturer's sensor was involved: the data has signal names and values (e.g. BattTemp, CoolantFlow), not part numbers or suppliers, so naming one would be invention. I can show which signal looked abnormal and when."
        if re.search(r"mechanical part|which part|what part|exact part|component is failing|failing component|broken (component|part)|(point out|identify|name).{0,25}(component|part)", s):
            d = self.diag_default(quiet=True); hint = ""
            if d and d.get("headline"): hint = f" The most I can say from the data: {d['headline']} - that points at the cooling circuit as a system (pump, coolant level, hoses or flow sensor), not at one named part; confirming it takes a physical inspection."
            return "I can't tell you the exact mechanical part that is failing - I only see electrical signals and fault codes, not the parts themselves." + hint
        if re.search(r"predict|forecast|will happen|going to happen|next (fault|failure|warning|event)|what will .{0,40}\b(be|do|happen|read)\b.{0,30}in (exactly |about )?(an?|\d+) (hour|minute|second|day)", s):
            d = self.diag_default(quiet=True); extra = ""
            if d and d.get("trend"): extra = f" What I can do is describe the trend so far: {d['trend']}"
            return "I can't predict an exact time or outcome - I only analyse what has been recorded and I have no forecasting model, so any specific time would be invented." + extra
        return None

    # ------------------------------------------------------------------ missing data
    def h_missing(self, s, raw):
        m = Q.metric_of(s); st = Q.state_at(self.c, self.cov()["hi"])
        if re.search(r"can.?fd|all (the )?(can )?signals|list (all )?(the )?signals|which signals|every (can )?signal", s): return self.signal_list()
        if re.search(r"rear.?(left|right)|(left|right).{0,6}(rear|side)|side camera|left camera|right camera|other camera|camera captured|which camera|what did the .*camera (see|show|record)|camera (see|show|record)", s):
            self.brief = "I only keep metadata for the one front camera, never what it saw; there is no other camera in the data."
            return ("This vehicle has one camera, the front camera, and I only keep metadata for it (frames per minute, blockage, sharpness, lane-marking confidence) plus a 64x48 grey thumbnail at camera-fault events - never what it saw. "
                    "There is no rear-left or side camera in the data, and I can't describe what any camera saw at a given time." + (" " + self.camera_for_event(s) if re.search(r"event|anomal|warning|fault", s) else ""))
        if m == "radar" or "radar" in s:
            return "There is no radar data in this vehicle's recorded signals, so I can't say what the vehicle was doing when a radar detected an object. The only perception source is the front camera's quality metadata (blockage, sharpness, lane confidence)."
        if m == "cabin" or re.search(r"\bcabin\b", s):
            return f"Cabin temperature is not recorded by this car. The closest signals are the outside temperature ({st.get('amb', float('nan')):.1f} C at the latest sample) and the climate-control power ({st.get('hvac', 0):.0f} W) - which show the heating / cooling effort, not the temperature inside."
        if m == "throttle": return "Accelerator-pedal (throttle) position is not among the recorded signals. The nearest things I have are traction power (kW) and acceleration calculated from the speed signal."
        if m == "steering": return "Steering angle is not recorded by this vehicle's data set (the signals are speed, gear, battery, thermal, climate, charger and GPS). I can give speed and acceleration for the same moment, and the GPS heading."
        if m == "engine_load": return "This is an electric vehicle - there is no engine, so no engine load. The equivalent I do have is traction power (kW) from the battery current x voltage."
        return None

    def camera_for_event(self, s):
        e = self.last_anchor_event(s)
        if not e: return ""
        cams = self.c.execute("SELECT ts, type, detail FROM events WHERE type IN ('CAMERA_BLOCKED','CAMERA_DEGRADED') AND ABS(ts-?)<=600 ORDER BY ABS(ts-?) LIMIT 1", (e[0], e[0])).fetchone()
        if cams: return f"For the event at {self.fts(e[0])}, the nearest camera record is the front camera: {Q.safe_text(cams[2])} at {self.fts(cams[0])}."
        return f"For the event at {self.fts(e[0])} there is no camera fault recorded within 10 minutes; the front camera was the only camera that could have been involved."

    def signal_list(self):
        try: txt = open(__import__("os").path.join(__import__("os").path.dirname(__file__), "protocols", "vehicle.dbc"), encoding="utf-8", errors="ignore").read()
        except Exception: txt = ""
        msgs = []; cur = None
        for line in txt.splitlines():
            m = re.match(r"BO_ (\d+) (\w+): (\d+) (\w+)", line)
            if m: cur = [f"{m[2]} (0x{int(m[1]):X}, {m[3]} bytes, from {m[4]})", []]; msgs.append(cur); continue
            m = re.match(r"\s+SG_ (\w+) .*\"([^\"]*)\"", line)
            if m and cur: cur[1].append(f"{m[1]}" + (f" [{m[2]}]" if m[2] else ""))
        n = sum(len(x[1]) for x in msgs)
        big = sum(1 for x in msgs if int(re.search(r"(\d+) bytes", x[0])[1]) > 8)
        out = [f"The vehicle's CAN database defines {len(msgs)} messages with {n} signals ({big} message(s) longer than 8 bytes, i.e. CAN-FD frames):"]
        out += [f"  - {m[0]}: {', '.join(m[1])}" for m in msgs]
        out.append("Other sources recorded: UDS diagnostic trouble codes, OBD-II PIDs, vehicle properties (Android VHAL), SOME/IP services, GNSS (NMEA position, heading, speed), system logs and front-camera metadata.")
        out.append("Not recorded: RPM (estimated from speed), throttle, steering angle, cabin temperature, radar, other cameras.")
        return "\n".join(out)


