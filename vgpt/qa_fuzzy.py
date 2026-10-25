"""Fuzzy router for the analysis routes: when no keyword route matches (and the older intent rules are not confident), compare the question with the example phrasings in data/qa_protos.json
(tf-idf cosine over lightly stemmed words) and use the best route if it is clearly ahead. Deterministic, ~400 examples, no model. Follow-up style routes also need a previous answer to refer to."""
import json, math, os, re
from collections import Counter

STOP = set("the a an of to is are was were be been do does did you your my me i it its this that and or for with at in on how what which when where who can could would should please tell show give find any there have has had will us about hey hi so uh um umm well just actually really quick want wanna gonna let lets also then ok okay".split())
FOLLOW = {"evidence", "alt", "conf", "missing_info", "counter", "calc", "periods", "signals_during", "top3", "obs_inf", "prove", "conclusion", "trace", "cutoff", "conf_change", "used_records", "wrong", "nuclear3"}
VEHICLE = re.compile(r"vehicle|car\b|battery|motor|rpm|speed|energy|efficien|range|trip|drive|driv|charge|charging|coolant|pump|tyre|brake|brak|adas|fault|dtc|diagnostic|warning|overheat|temperature|signal|sensor|camera|gps|lane|event|anomal|abnormal|reading|alarm|cell|incident|hypothes|analysis|diagnos|sensor|cabin|radar|throttle|steering|can\b|idle|accelerat|decelerat|firmware|mileage|consumption|trend|week|history|log")


CLUSTERS = {   # everyday synonyms -> one concept token, so different wording of the same idea matches
    "c_fault": "fault dtc diagnostic trouble error code failure defect malfunction faulty broken", "c_warn": "warning alarm alert warn notification flag", "c_heat": "hot hotter overheat overheating heat warm warmer warming thermal",
    "c_before": "before preceded preceding precede ahead prior earlier leading lead", "c_after": "after following afterwards afterward next subsequent straight", "c_speed": "speed fast fastest velocity quick quickest",
    "c_rpm": "rpm revolution rev revs motor-speed", "c_compare": "compare versus against differ difference between vs contrast alike", "c_change": "change changed changing shift shifted vary varied moved",
    "c_trend": "trend trending creep creeping rising rise increasing increase climbing drift drifted upward worsen worse worsening deteriorate deteriorating gradual degradation declined declining", "c_why": "why explain explanation cause reason account caused because how-come",
    "c_evid": "evidence support supports supporting proof prove proven backs basis based", "c_trust": "confident confidence trust sure certain reliable certainty", "c_alt": "alternative another other else different",
    "c_miss": "missing lack need unknown unavailable", "c_contra": "contradict contradicts contradicting disprove conflict opposite", "c_recheck": "recheck redo again verify raw recompute double",
    "c_normal": "normal ordinary usual expected behaving behaves", "c_odd": "strange weird odd unusual abnormal suspicious fishy anomaly anomalous irregular unexpected", "c_crit": "serious major severe critical worst biggest troublesome",
    "c_low": "lowest minimum least low smallest", "c_high": "highest maximum peak top max largest greatest", "c_count": "count many often number frequently times", "c_avg": "average mean typical median",
    "c_pct": "percentage percent share proportion fraction", "c_idle": "idle idling standstill stationary stopped sitting standing crawling", "c_accel": "accelerate accelerating speeding", "c_decel": "decelerate decelerating slowing braking",
    "c_trip": "trip drive journey ride", "c_first": "first earliest opening initial beginning", "c_last": "last latest recent newest", "c_hist": "entire whole overall full history",
    "c_calc": "calculation calculate sums maths math computed working", "c_period": "period periods interval stretch", "c_safe": "disable deactivate override bypass raise modify alter switch",
    "c_go": "go going travel travelling traveling head leave depart set-off outing", "c_okay": "safe safely okay ok fine alright advisable advised advise recommend recommended wise sensible risky dangerous unsafe ready fit roadworthy", "c_far": "long far distance highway motorway journey airport",
    "c_neg": "negative minus below", "c_energy": "energy power electricity juice", "c_charge": "charge charged charging plug plugged recharge", "c_cost": "cost price paid pay bill expensive spend spent money",
    "c_remain": "left remaining remain", "c_tip": "tip tips advice recommendations ways", "c_save": "save saving improve increase stretch extend better lower reduce cut", "c_hot2": "garam warm hot", "c_okay2": "good great proper properly",
    "c_cabin": "cabin interior inside", "c_signal": "signal reading readings sensor value values", "c_eff": "efficiency mileage consumption economy",
}
CONCEPT = {}
for _c, _ws in CLUSTERS.items():
    for _w in _ws.split(): CONCEPT[_w] = _c


NEGATE = {"unusual": r"lately|recently|\bever\b|whole history|overall", "speeding": r"cause|effect|heat|temperature|because|due to|once|account|control|regress", "trip_count": r"\bpark|\bparking", "charging_now": r"phone|laptop|tablet|remote|watch|earbuds|\btv\b", "style": r"\b(last|when|ago|emergency|latest|date|time|times)\b", "charge_time": r"wear|degrad|\bage\b|ageing|aging|replac|last before|health", "reach": r"how long|direction|heading|which way|bearing|\beta\b|arrive|replac|wear|degrad|mean\b|show it|\bin (miles|km)",
          "dtc_explain": r"do you mean|you mean", "battery_now": r"warning|warnings|alert|alarm"}
REQUIRE = {"trip_count": r"trip|drive|drives|driven|drove|journey|ride", "charging_now": r"charg|plug|connect", "speeding": r"fast|speed|limit|quick|km/h|over \d|above \d|faster|slow", "top_speed": r"speed|fast|fastest|quick", "hot_battery": r"battery|pack|hot|warm|overheat|temperature|temp|vehicle",
           "regen": r"regen|recover|recuper|braking|brake|coast", "leave_time": r"leave|left|depart|set off|go out|went out|head|step|house|home|work|out", "best_day": r"day|week|efficien|economical|consum|energy|thrift|green", "reach": r"charg|range|battery|km|kilomet|mile|reach|get there|get home|make it|far|distance|left|enough|plug|juice|power|home|work|town|city|office",
           "v12": r"12|twelve|auxiliary|low.?voltage|small battery|accessory", "dtc_explain": r"code|fault|error|dtc|trouble|\b[pbcu][0-9a-f]{4}\b", "advice": r"\b(go|going|drive|driving|trip|travel\w*|journey|out|use|using|take|road|leave|leaving|head|commute|beach|airport|warning|condition|faults?|highway|motorway|continue|keep)\b", "signal_why": r"\bwhy\b|\bwhat does\b|\bmean", "battery_now": r"battery|pack|charge|juice",
           "drive_safely": r"safe|safely|safer|careful|accident|risk", "charge_cost": r"cost|pay|paid|price|bill|spend|spent|money|expens|cheap|electricity", "day": r"happen|occur|went on|go on|going on|anything|summar|tell me about|how was (the )?(day|vehicle|car)"}      # routes that need their own key word, so a date alone does not pull a question in


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf): return w[: -len(suf)]
    return w


VEHICLE_SOFT = re.compile(r"car\b|vehicle|battery|pack\b|motor|drive|driv|trip|journey|charg|range|tyre|tire|brake|coolant|cooling|pump|warning|fault|code|speed|energy|efficien|consum|\bkm\b|mile|\bgo\b|going|safe|accident|engine|dashboard|\blight|power|temperature|12 ?v|service|camera|lane|sensor|signal|week|yesterday|today|commute|route|place|visit")
OOD = re.compile(r"\b(phone|laptop|tv|movie|recipe|cook|capital of|flight|bitcoin|stocks?|weight|joke|poem|song|homework|restaurant|minister|president|football|cricket|tokyo|paris|delhi|pasta|dinosaurs?|everest|bake|baking|cake|pizza|biryani|noodles|sushi|burger|recipes?|comedian|bollywood|dinner|lunch|breakfast|novel|lyrics|singer|actor|actress|celebrity|election|horoscope|jokes|riddle|crossword|chess|poetry|essay|homework|math homework|calories|diet|workout|gym routine|girlfriend|boyfriend|marriage|salary|shopping|amazon|netflix|youtube|instagram|whatsapp)\b|time is it|translate|meaning of life")


def trigrams(s):
    out = []
    for w in re.findall(r"[a-z]+", s.lower()):
        if w in STOP or len(w) < 3: continue
        w = f" {w} "; out += [w[i:i + 3] for i in range(len(w) - 2)]
    return out


def tokens(s):
    out = []
    for w in re.findall(r"[a-z]+|\d+", s.lower()):
        if w in STOP: continue
        w = "#" if w.isdigit() else w
        out.append(CONCEPT.get(w) or CONCEPT.get(stem(w)) or stem(w))
    return out


def blocked(r, s):
    """True when the question contains a word that rules the route out."""
    return r in NEGATE and bool(re.search(NEGATE[r], s.lower()))


def allowed(r, s):
    """False when the question lacks a word the route needs."""
    sl = s.lower()
    return not (r in REQUIRE and not re.search(REQUIRE[r], sl))          # (a NEGATE hit rejects the whole match further down - those questions belong to the older rules)



class Fuzzy:
    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(__file__), "data", "qa_protos.json")
        try:
            with open(path, encoding="utf-8") as f: self.protos = json.load(f)
        except Exception: self.protos = {}
        self.docs = [(r, q, Counter(tokens(q))) for r, qs in self.protos.items() for q in qs]
        df = Counter(); [df.update(set(c)) for _, _, c in self.docs]; n = max(len(self.docs), 1)
        self.idf = {w: math.log(1 + n / (1 + d)) for w, d in df.items()}
        self.vecs = [(r, q, self._vec(c)) for r, q, c in self.docs]
        tdocs = [Counter(trigrams(q)) for _, q, _ in self.docs]; tdf = Counter(); [tdf.update(set(c)) for c in tdocs]
        self.tidf = {w: math.log(1 + n / (1 + d)) for w, d in tdf.items()}; self.tvecs = [self._tvec(c) for c in tdocs]

    def _tvec(self, c):
        v = {w: (1 + math.log(k)) * self.tidf.get(w, 1.0) for w, k in c.items()}; nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}

    def soft_route(self, s, have_context=False, min_score=0.30):
        """Last resort: blend of word similarity (with the synonym clusters) and character-trigram similarity, which tolerates different word forms (degraded / degradation) and typos. -> (route, score, example) or None."""
        c = Counter(tokens(s)); t = Counter(trigrams(s))
        if not c or not self.vecs or not VEHICLE_SOFT.search(s.lower()) or OOD.search(s.lower()): return None
        qv = self._vec(c); tv = self._tvec(t); per = {}
        for (r, q, v), tvec in zip(self.vecs, self.tvecs):
            if r == "_legacy" or r in FOLLOW: continue
            w = sum(x * v.get(k, 0.0) for k, x in qv.items()); tr = sum(x * tvec.get(k, 0.0) for k, x in tv.items()); per.setdefault(r, []).append((0.7 * w + 0.3 * tr, q))
        best = {}
        for r, lst in per.items():                                                              # a route with several close examples beats a route with one accidental match
            lst.sort(reverse=True); top = lst[:3]; best[r] = (0.6 * top[0][0] + 0.4 * sum(x[0] for x in top) / len(top), top[0][1])
        rank = sorted(best.items(), key=lambda kv: -kv[1][0])
        rank = [x for x in rank if allowed(x[0], s)]
        if not rank: return None
        r, (sc, ex) = rank[0]; gap = sc - (rank[1][1][0] if len(rank) > 1 else 0.0)
        if sc < min_score or gap < 0.015: return None
        if len(set(tokens(s)) & set(tokens(ex)) - {"good", "state", "ok", "c_okay"}) < 2 and sc < 0.6: return None          # one shared everyday word is not enough
        if r in REQUIRE and not re.search(REQUIRE[r], s.lower()): return None
        if r in NEGATE and re.search(NEGATE[r], s.lower()): return None
        return r, sc, ex

    def _vec(self, c):
        v = {w: (1 + math.log(k)) * self.idf.get(w, 0.0) for w, k in c.items()}; nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}

    def route(self, s, have_context=False, min_score=0.45, soft=False):
        """-> (route, score, example) or None."""
        c = Counter(tokens(s))
        if not c or not self.vecs or OOD.search(s.lower()): return None
        qv = self._vec(c); best = {}
        for r, q, v in self.vecs:
            sc = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if sc > best.get(r, (0, ""))[0]: best[r] = (sc, q)
        rank = sorted(best.items(), key=lambda kv: -kv[1][0])
        rank = [x for x in rank if x[0] == "_legacy" or allowed(x[0], s)]
        if not rank: return None
        r, (sc, ex) = rank[0]
        if r == "_legacy": return None
        gap = sc - (rank[1][1][0] if len(rank) > 1 else 0.0)
        if sc < min_score or (gap < 0.04 and sc < 0.75): return None
        if r in REQUIRE and not re.search(REQUIRE[r], s.lower()): return None
        if r in NEGATE and re.search(NEGATE[r], s.lower()): return None
        if r in FOLLOW and (not have_context or sc < 0.5): return None
        if r not in FOLLOW and sc < 0.55 and not soft and not VEHICLE.search(s.lower()): return None
        return r, sc, ex

    def topk(self, s, k=3, skip=()):
        """The k best routes (and their nearest example) for a question, even below the confidence threshold - used to suggest what the person may have meant."""
        c = Counter(tokens(s))
        if not c or not self.vecs: return []
        qv = self._vec(c); best = {}
        for r, q, v in self.vecs:
            if r == "_legacy" or r in FOLLOW or r in skip: continue
            sc = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if sc > best.get(r, (0, ""))[0]: best[r] = (sc, q)
        return [(r, sc, q) for r, (sc, q) in sorted(best.items(), key=lambda kv: -kv[1][0])[:k] if sc > 0.12]


CANON = {"advice": "Is it safe to keep driving?", "battery_now": "How's my battery?", "last_charge": "When did I last charge?", "reach": "Can I make it 150 km?", "improve_range": "How can I improve my range?", "charge_cost": "What did charging cost me?", "charge_time": "How long to charge to 80%?",
         "speeding": "Am I driving too fast?", "hot_battery": "Was the battery ever hot?", "pump_ok": "Is the coolant pump OK?", "v12": "How is the 12 V battery?", "motor_now": "How hot is the motor?", "unusual": "Did anything unusual happen today?", "overview": "Give me a summary of the week",
         "style": "What is my driving score?", "dtc_explain": "What does that code mean?", "battery_age": "How old is the battery?", "charging_now": "Is the car charging?", "top_speed": "How fast did I go?", "places_most": "Which places do I visit most?", "commute": "How long is my commute?",
         "longest_trip": "What was my longest trip?", "last_trip": "How was my last trip?", "eff_day": "How efficient was I yesterday?", "best_day": "Which day was most efficient?", "eff_trend": "Am I using more energy than usual?", "trip_count": "How many trips did I make?",
         "regen": "How much regen did I get?", "compare_months": "Compare this month with last month", "briefing": "Brief me", "signal_why": "Why is my battery power negative?", "drive_safely": "How do I drive safely?", "explain_simple": "Explain the last warning simply",
         "help": "What can I ask you?", "why_range2": "Why has my range dropped lately?", "events": "Any warnings this week?", "ctx": "What happened before the last warning?", "why_event": "Why did the last warning happen?", "checkup": "Is my car healthy?", "degrade": "Identify any gradual degradation",
         "trend": "Has my efficiency changed over two weeks?", "compare_periods": "Compare this week with last week", "soh_rate": "How fast is the battery ageing?"}
DEFAULT_SUGGEST = ["Brief me", "Is my car healthy?", "How's my battery?"]


def related(text, n=3):
    """Up to n questions the person may have meant (nearest known routes), as clickable suggestions; falls back to a sensible default set."""
    try:
        f = Fuzzy._inst if getattr(Fuzzy, "_inst", None) else None
        if f is None: Fuzzy._inst = f = Fuzzy()
        out = []
        for r, sc, ex in f.topk(text, k=8):
            q = CANON.get(r)
            if q and q not in out: out.append(q)
            if len(out) >= n: break
    except Exception:
        out = []
    for d in DEFAULT_SUGGEST:
        if len(out) >= n: break
        if d not in out: out.append(d)
    return out[:n]
