"""VANTAGE semantic acceptance ledger (cases A1..G5 of the campaign brief), run against the real brain/HTTP app.
Each case: the question exactly as written, structural expectations (what the answer must establish / must not substitute) and an automatic verdict; a human read-through of the answers remains the authority (results/vantage_ledger.md records both). A case passes only when every `ok` pattern matches, no `bad` pattern matches and, unless `clar=True`, the answer is not a clarification menu.
    python bench/vantage_acceptance.py [--http URL | --db data/gen_demo.db] [--now 1787109669] [--only A,B] [--out results/vantage_ledger.json]"""
import argparse, json, os, re, sys, time, urllib.request
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)

CLAR = re.compile(r"i want to be sure|do you mean|which (?:temp|signal)|which .* do you mean|reply with a number|could not turn that", re.I)
C = lambda *a, **k: (a, k)
# id, question, ok-patterns (all), bad-patterns (none), options
CASES = [
 ("A1", "What is the battery percentage right now?", [r"\d+(?:\.\d+)?\s?%"], [r"average|seven|7 days"], {}),
 ("A2", "How much power is the car using at this moment?", [r"-?\d+(?:\.\d+)? ?kw"], [r"7 days|seven|average"], {}),
 ("A3", "What is the coolant flow right now?", [r"\d+(?:\.\d+)? ?l/min"], [], {}),
 ("A4", "What is the temperature of the battery pack?", [r"\d+(?:\.\d+)? ?°c", r"battery|pack"], [r"motor temperature is|outside temperature is"], {}),
 ("A5", "What is the motor RPM right now?", [r"\d+ ?rpm"], [r"km/h"], {}),
 ("A6", "What is the outside temperature?", [r"\d+(?:\.\d+)? ?°c", r"outside|ambient"], [r"battery temperature is|motor temperature is"], {}),
 ("A7", "Give me every value you can currently see from the car.", [r"speed", r"battery", r"coolant"], [r"predict|invented"], {}),
 ("A8", "Is the car moving right now?", [r"moving|stationary|parked|stopped|standing|driving|not moving"], [r"outside temperature"], {}),
 ("A9", "What was the last valid speed reading, and when was it recorded?", [r"km/h", r"\d{1,2}:\d{2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|ago"], [], {}),
 ("A10", "Is the battery power reading current or stale?", [r"stale|current|fresh|seconds?|minutes?|ago|age"], [r"battery itself has a problem|over-temperature"], {}),
 ("A11", "What was my average coolant flow yesterday?", [r"l/min", r"yesterday"], [r"right now"], {}),
 ("A12", "What was my average battery temperature yesterday?", [r"°c", r"yesterday"], [r"right now"], {}),
 ("A13", "What was my average motor RPM yesterday?", [r"rpm", r"yesterday"], [r"km/h|right now"], {}),
 ("A14", "What was my average outside temperature yesterday?", [r"°c", r"yesterday"], [r"right now"], {}),
 ("B1", "How much time did I take to accelerate from 30 to 50 km/h", [r"\d+(?:\.\d+)? ?s\b|seconds|no (?:valid|qualifying|run)|not enough"], [], {}),
 ("B2", "Compare my acceleration from 20–30 and 30–40 km/h", [r"20.{0,3}30", r"30.{0,3}40"], [r"distance|average vehicle speed"], {}),
 ("B3", "Do I accelerate harder at low speeds or high speeds?", [r"low", r"high"], [r"average vehicle speed over"], {}),
 ("B4", "What is my average acceleration magnitude over the last 24 hours?", [r"m/s", r"24 hours|24 h|day"], [r"signed"], {}),
 ("B5", "How much energy did I consume on my last trip?", [r"\d+(?:\.\d+)? ?kwh", r"trip"], [r"wh/km\b(?!.*kwh)"], {}),
 ("B6", "What was my energy consumption per 100 km?", [r"kwh ?/ ?100 ?km|kwh per 100"], [], {}),
 ("B7", "How much energy did regenerative braking recover last week?", [r"regen|recover", r"kwh|wh"], [r"you used"], {}),
 ("B8", "At my recent efficiency, how much energy would a 250 km trip require?", [r"\d+(?:\.\d+)? ?kwh", r"250"], [], {}),
 ("B9", "How far could I travel using 10 kWh?", [r"\d+(?:\.\d+)? ?km", r"10 ?kwh"], [r"battery 8\d"], {}),
 ("B10", "How many 7000 mAh batteries at 3.7 V equal the car’s remaining usable energy?", [r"25\.9 ?wh"], [], {}),
 ("B11", "How much energy would I need to charge the car from 35% to 80%?", [r"\d+(?:\.\d+)? ?kwh", r"35", r"80"], [r"already at"], {}),
 ("B12", "If the battery is at 81%, does that mean its health is 81%?", [r"no|not", r"charge", r"health"], [], {}),
 ("C1", "What was my average speed yesterday?", [r"km/h", r"yesterday"], [], {}),
 ("C2", "What was my average speed over the last nine days?", [r"9 days|nine days", r"km/h"], [r"7 days|14 days"], {}),
 ("C3", "Compare my energy consumption over the last seven days with the previous seven days", [r"previous|prior|before"], [r"same for the last 7 days and the last 7 days"], {}),
 ("C4", "Has my coolant flow changed over the past 14 days", [r"l/min", r"14 days|two weeks"], [], {}),
 ("C5", "When was my battery temperature highest in the last month?", [r"°c", r"\d{1,2}:\d{2}"], [], {}),
 ("C6", "What was my maximum speed while driving on the highway last week", [r"km/h", r"highway|motorway"], [], {}),
 ("C7", "Compare city and highway energy consumption over the same 14-day period", [r"city", r"highway|motorway", r"wh/km|kwh"], [], {}),
 ("C8", "How much time did I spend driving versus charging yesterday!", [r"\b(?:h|hours?|min|minutes?)\b", r"driving", r"charging"], [r"(?<!not )(?:average )?state of charge was|soc was|%\s+for (?:driving|charging)"], {}),
 ("C9", "What happened immediately before the coolant-flow fault", [r"coolant"], [], {}),
 ("C10", "Show me the trend in battery temperature, day by day, for the last seven days", [r"°c"], [], {"dates": 3}),
 ("C11", "What was my average battery temperature yesterday?", [r"°c", r"yesterday"], [], {}),
 ("D1", "What is the difference between battery temperature and motor temperature?", [r"battery", r"motor"], [r"^battery temperature is [\d.]+ °c right now\.?\s*$"], {}),
 ("D2", "Does high state of charge mean the battery is healthy?", [r"charge", r"health"], [], {}),
 ("D3", "What is the difference between charging power and battery energy?", [r"kw\b", r"kwh"], [], {}),
 ("D4", "Does normal tyre pressure prove that my tyres are safe?", [r"tread|damage|puncture|not (?:enough|prove)|cannot|can't|doesn't|does not"], [r"^tyre pressure is"], {}),
 ("D5", "Is my brake fluid okay?", [r"brake.fluid", r"no |not |isn't|unavailable|cannot|can't"], [r"^brake.pad"], {}),
 ("D6", "What is the difference between the car's rated top speed and its highest recorded speed?", [r"rated", r"record"], [], {}),
 ("D7", "Does low coolant flow prove the battery is overheating right now?", [r"not|doesn't|does not|cannot|no\b", r"coolant", r"temperature"], [], {}),
 ("D8", "How many driving modes does this vehicle support?", [r"mode"], [r"%"], {}),
 ("D9", "Does a diagnostic code from last week mean the fault is still active?", [r"active|status|current"], [], {}),
 ("E1", "Is my car healthy?", [r"battery|coolant|warning"], [], {}),
 ("E2", "What happened before the last warning?", [r"before|prior|minutes"], [], {}),
 ("E3", "What should I investigate first?", [r"coolant|battery|cooling|warning"], [], {}),
 ("E4", "Show the last diagnostic events", [r"p1a00|diagnostic|dtc|warning|event"], [], {}),
 ("E5", "What are the three most serious confirmed active faults", [r"confirmed|active|cannot|no confirmed|illustrative|simulated"], [], {}),
 ("E6", "Which fault is most recent, and which is most severe", [r"recent", r"severe|severity"], [], {}),
 ("E7", "Is the coolantflow warning more important than the tyre press warning", [r"coolant", r"tyre"], [], {}),
 ("E8", "What should I have the workshop check first", [r"coolant|cooling|battery|workshop|inspect"], [], {}),
 ("E9", "Can I safely drive 200 km with this warning", [r"warning", r"cannot|can't|not (?:guarantee|safe)|recommend|inspect|workshop"], [], {"clar": True}),
 ("E10", "What evidence suggests that battery cooling has a problem", [r"coolant", r"l/min|%|temperature"], [r"state of charge"], {}),
 ("E11", "Have the over-temperature warnings happened while the vehicle was parked or moving", [r"parked|moving|speed|driving"], [r"gps|bengaluru"], {}),
 ("E12", "Is my tyre pressure problem still present", [r"tyre", r"psi|bar|kpa|no |not |warning|normal|present"], [], {}),
 ("E13", "What should I fix immediately", [r"coolant|cooling|battery|warning|nothing"], [r"doesn't look like a question"], {}),
 ("E14", "Brief me", [r"battery|range"], [], {}),
 ("E15", "How's my battery?", [r"charge|%", r"health"], [], {}),
 ("E16", "Why do you think my car is unhealthy", [r"coolant|warning|temperature|battery"], [], {}),
 ("E17", "what needs attention most", [r"coolant|cooling|battery|warning"], [r"average battery health"], {}),
 ("E18", "no, i meant battery health, not charge", [r"health", r"%"], [r"state of charge"], {"ctx": ["How's my battery?"]}),
 ("F1", "wats my battary temp rn", [r"°c", r"battery"], [], {}),
 ("F2", "hows the breaks looking", [r"brake", r"pad|fluid"], [], {}),
 ("F3", "how fast is it charging rn", [r"charg"], [r"km/h"], {}),
 ("F4", "when did i last plug it in", [r"charg", r"\d{1,2}:\d{2}|ago|aug|sep"], [], {}),
 ("F5", "whats the cars max speed", [r"rated|specification|km/h|not (?:available|recorded)|unavailable"], [], {}),
 ("F6", "how much has my battery degraded since i got it", [r"health|degrad", r"%"], [r"a month"], {}),
 ("F7", "how is the car doing overall", [r"battery|coolant|warning|health"], [], {}),
 ("F8", "is my car getting less efficient lately", [r"wh/km|efficien|consumption", r"week|days"], [], {}),
 ("F9", "how can i drive more efficiently", [r"smooth|speed|accelerat|regen|brak|climate|tip|advice|efficien"], [], {}),
 ("F10", "what about highway driving?", [r"highway|motorway", r"ask|which|metric|consumption|speed"], [r"battery health"], {"clar": True}),
 ("F11", "how much has it changed over 30 days", [r"30 days|month", r"health|%"], [r"7 days"], {"ctx": ["How's my battery health?"]}),
 ("F12", "What evidence supports that change", [r"health|%", r"30 days|month"], [r"7 days"], {"ctx": ["How's my battery health?", "how much has it changed over 30 days"]}),
 ("F13", "What was the brake-fluid quality reading at 14:30 yesterday?", [r"brake.fluid", r"no |not |unavailable|isn't|cannot"], [r"share of time"], {}),
 ("F14", "What was the battery temperature exactly 30 days ago", [r"30 days", r"°c|not|no |unavailable"], [r"average .* last 7 days"], {}),
 ("F15", "What is my car's officially rated maximum speed", [r"rated|specification|not|unavailable|km/h"], [r"last 7 days"], {}),
 ("F16", "Tell me the exact remaining battery lifespan in months", [r"cannot|can't|not|no\b|too short|insufficient"], [r"-0\.05"], {}),
 ("F17", "How much CO₂ have I saved compared with a petrol car", [r"co.?2|emission|baseline|petrol"], [r"closest electric equivalent"], {}),
 ("F18", "Which replacement tyre has the highest speed rating", [r"tyre|tire", r"cannot|can't|not|no\b|verify|catalog|specification"], [r"highest vehicle speed"], {}),
 ("F19", "How many fictional game characters would it take to destroy my car", [r"fiction|hypothetical|cannot|can't|telemetry"], [], {"clar": True}),
 ("F20", "You said the battery was at 81% earlier. What is it now", [r"\d+(?:\.\d+)? ?%", r"now|current|right now"], [r"average"], {}),
 ("F21", "The last recorded fault was marked simulated. Is that a confirmed fault", [r"simulat|illustrative", r"not|no\b|isn't|cannot"], [], {}),
]
SEQ = {
 "G1": ["What is the motor temperature?", "What about the battery", "Is that normal", "Compare it with the motor temperature"],
 "G2": ["What is the motor temperature", "How can I drive more efficiently", "What was my energy consumption last week"],
 "G3": ["What is my battery percentage", "No, I mean battery health", "How much has it changed over 30 days", "What evidence supports that change"],
 "G4": ["Compare city and highway driving", "Compare energy consumption specifically", "Now compare average speed instead", "Do both comparisons use the same dates?"],
 "G5": ["What happened before the last warning?", "Show the last diagnostic events", "What should I investigate first?", "Is it still active?"],
}
SEQ_CHECK = {  # (sequence, step index) -> (ok, bad)
 ("G1", 0): ([r"motor", r"°c"], []), ("G1", 1): ([r"battery", r"°c|%"], [r"state of charge"]), ("G1", 2): ([r"normal|within|typical|usual|range|expected"], []), ("G1", 3): ([r"battery", r"motor"], [r"^motor temperature is [\d.]+ °c right now\.?\s*$"]),
 ("G2", 0): ([r"°c"], []), ("G2", 1): ([r"efficien|smooth|accelerat|regen|speed|tip"], [r"motor temperature is"]), ("G2", 2): ([r"wh/km|kwh", r"7 days|week"], [r"motor temperature"]),
 ("G3", 0): ([r"%"], []), ("G3", 1): ([r"health", r"%"], [r"state of charge"]), ("G3", 2): ([r"30 days|month", r"health|%"], [r"7 days"]), ("G3", 3): ([r"health|%", r"30 days|month"], [r"7 days"]),
 ("G4", 0): ([r"city", r"highway|motorway"], []), ("G4", 1): ([r"wh/km|consumption", r"city"], []), ("G4", 2): ([r"km/h|speed", r"city"], [r"wh/km"]), ("G4", 3): ([r"same|dates|window|period|days", r"yes|no|both|same"], []),
 ("G5", 0): ([r"before|prior|minutes"], []), ("G5", 1): ([r"diagnostic|p1a00|event|warning"], []), ("G5", 2): ([r"coolant|cooling|battery|warning"], []), ("G5", 3): ([r"active|current|status|still|not|unknown|historical"], []),
}


def judge(ans, ok, bad, opt):
    a = ans or ""; flags = []
    if CLAR.search(a) and not opt.get("clar"): flags.append("clarification")
    for p in ok:
        if not re.search(p, a, re.I | re.M): flags.append("missing /" + p + "/")
    for p in bad:
        if re.search(p, a, re.I | re.M): flags.append("forbidden /" + p + "/")
    if opt.get("dates") and len(re.findall(r"\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", a, re.I)) < opt["dates"]: flags.append("not day-by-day")
    return flags


class Local:
    def __init__(self, db, now):
        from vgpt.product import make_legacy_brain
        from vgpt.brain import LazySlm
        self.slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words")      # the one permitted model
        self.mk = lambda: make_legacy_brain(db, slm=self.slm, now=now)
        self.b = self.mk()
    def reset(self): self.b = self.mk()
    def ask(self, q):
        t = time.perf_counter(); r = self.b.ask(q); return {"answer": r.get("answer") or "", "route": (r.get("facts") or {}).get("route") if isinstance(r.get("facts"), dict) else None, "intent": r.get("intent"), "ms": round((time.perf_counter() - t) * 1000), "path": r.get("path")}


class Http:
    def __init__(self, url): self.url = url.rstrip("/")
    def reset(self):
        try: urllib.request.urlopen(urllib.request.Request(self.url + "/api/reset_chat", b"{}", {"Content-Type": "application/json"}), timeout=10).read()
        except Exception: pass
    def ask(self, q):
        t = time.perf_counter(); r = json.loads(urllib.request.urlopen(urllib.request.Request(self.url + "/api/ask", json.dumps({"q": q}).encode(), {"Content-Type": "application/json"}), timeout=180).read())
        return {"answer": r.get("answer") or "", "route": r.get("route"), "intent": r.get("intent"), "ms": round((time.perf_counter() - t) * 1000), "path": r.get("path")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--http"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db")); ap.add_argument("--now", type=int, default=1787109669); ap.add_argument("--only", default=""); ap.add_argument("--out", default=os.path.join(HERE, "results", "vantage_ledger.json")); ap.add_argument("--show", action="store_true"); a = ap.parse_args()
    S = Http(a.http) if a.http else Local(a.db, a.now); only = [x for x in a.only.split(",") if x]; res = []
    for cid, q, ok, bad, opt in CASES:
        if only and not any(cid.startswith(o) for o in only): continue
        S.reset()
        for pre in opt.get("ctx", []): S.ask(pre)
        r = S.ask(q); fl = judge(r["answer"], ok, bad, opt); res.append({"id": cid, "q": q, "status": "PASS" if not fl else "FAIL", "flags": fl, **r})
    for sid, qs in SEQ.items():
        if only and not any(sid.startswith(o) for o in only): continue
        S.reset()
        for i, q in enumerate(qs):
            r = S.ask(q); ok, bad = SEQ_CHECK[(sid, i)]; fl = judge(r["answer"], ok, bad, {}); res.append({"id": f"{sid}.{i + 1}", "q": q, "status": "PASS" if not fl else "FAIL", "flags": fl, **r})
    n = len(res); p = sum(r["status"] == "PASS" for r in res); print(f"{p}/{n} pass")
    for r in res:
        if r["status"] != "PASS" or a.show: print(f"[{r['status']}] {r['id']}: {r['q']}\n      -> {r['answer'][:230]!r}  route={r['route']} {r['ms']}ms  {r['flags']}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
