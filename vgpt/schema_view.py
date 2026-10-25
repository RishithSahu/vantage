"""A SchemaView is what the planner is shown about a vehicle: readable descriptions of the signals, events, conditions and operations it may name, under the names THIS catalog version uses.

It is built from the registry (and could be built from any VSS-style catalog). Three things make it more than a vocabulary list:
  * `alias`  - the display name of an item can differ from the registry id (battery_temperature -> pack_thermal_c). The planner reads and writes display names; `to_registry` maps a model-written query back before
               it is parsed. This is how schema transfer is tested: the same questions, a catalog the model has never seen, no retraining.
  * `drop`   - items can be withheld from the view (a sensor the vehicle does not have, a signal family held out of the examples) to test that capability questions and refusals degrade honestly.
  * `render` - the text that goes into the prompt, one short line per item; `grammar` - the GBNF that restricts the model's output to names that exist in THIS view.
The catalog can therefore change between vehicles and software versions without touching the planner."""
import hashlib
import re

from . import signal_registry as R
from .query_ir import OPS, AGGS, TIME_UNITS, CALENDAR, COND_OPS, SELECTORS, ENTITIES, ORDERS

OP_HELP = {
    "current_value": "the latest value of a signal (now, right now, at the moment)",
    "aggregate": "a statistic of a signal over a period (agg = mean min max median std first last delta count p95 sum percent_time mode)",
    "acceleration_band": "average acceleration over valid upward crossings of a lower and upper speed bound",
    "extreme": "the highest or lowest value of a signal over a period (agg=max or agg=min)",
    "trend": "is a signal rising or falling over a period",
    "compare": "compare a signal between two periods (t, t2) or two conditions (grp=a,b)",
    "count": "how many times an event happened (ev=...) or how many trips / charging sessions (sel=...)",
    "list_events": "list events of one type (evs=all, last, first, most_severe)",
    "event_context": "what the signals did around an event (bef / aft = seconds before / after)",
    "rank": "rank trips or charging sessions by a metric (sel=entity:metric:order:k:class)",
    "correlation": "relationship between two signals (tgt and tgt2)",
    "percent_time": "share of the time a condition was true (c=...)",
    "explain_event": "why did an event happen",
    "capability": "does the vehicle record a signal",
    "explain_change": "why did a signal or efficiency measure change over a period compared with the period before it",
    "detect_change": "when did a signal change the most (the largest shift or drop) within a period",
    "exists": "did a signal ever cross a level in a period (p.thr=<gt|lt|ge|le>:<number>)",
    "threshold_test": "whether a level was reached or held, with a threshold (p.thr)",
    "duration": "the total time a level or condition was true in a period",
    "share": "the share (percent) of a period, or of driving, in which a condition held",
    "episodes": "the separate stretches (episodes) during which a condition held",
    "continuous": "the longest continuous stretch during which a condition held",
    "days_with": "on which days a condition held (p.list=1 lists them, p.mind = minimum seconds)",
    "percentile": "a percentile of a signal over a period (p.pct=<n,n>)",
    "percentile_vs": "a percentile of a signal compared with the previous period (t2=prev:...)",
    "window_change": "how a statistic changed between two windows",
    "halves": "compare the first half of a period with the second half",
    "by_period": "a statistic per day or weekday and which one is highest or lowest (p.by, p.k, p.ord)",
    "pair_gap": "the gap between two paired values or events",
    "range": "the spread (maximum minus minimum) of a signal over a period",
    "parked_days": "the days on which the vehicle was parked or not used",
    "longest_run": "the longest run of consecutive days or events that met a condition",
    "trip_events": "events counted or listed per trip",
    "trip_extreme": "the trip with the highest or lowest value of a metric",
    "session_attr": "an attribute of a charging session or trip (p.attr = duration start energy soc temperature power distance speed)",
    "normality": "whether a value is typical compared with the vehicle's own history",
    "clock_slot": "a statistic restricted to a time-of-day slot (p.slot, or p.from / p.to)",
    "epistemic": "how certain, provable or causal a conclusion is (p.kind = certainty conflict proof causation exact tail criteria regen typical)",
    "topic": "a named topic answered by a fixed analysis (p.topic=<name>)",
}
FORMAT_HELP = ("op=<operation> [tgt=<signal>] [tgt2=<signal>] [agg=<statistic>] t=<time> [t2=<time>] [c=<conditions>] [sel=<entity:metric:order:k:class>] [grp=<a>,<b>] [ev=<event>] [evs=<selector>] [bef=<seconds>] [aft=<seconds>] [unit=<unit>]")
TIME_HELP = "t = now | all | last:<n>:<minute|hour|day|week|month> | cal:<" + "|".join(CALENDAR) + "> | date:YYYY-MM-DD"
COND_HELP = "c = comma-separated conditions: a condition name, or sig:<signal>:<gt|lt|ge|le|eq>:<number>[:<unit>]"


def _alts(names): return " | ".join('"' + n + '"' for n in names) if names else '"none"'


class SchemaView:
    def __init__(self, alias=None, drop=(), signals=None, name="registry"):
        """alias: {registry id: display name} for signals, events and conditions; drop: registry ids to withhold; signals: restrict to these registry ids."""
        self.alias = dict(alias or {}); self.drop = set(drop); self.name = name
        keep = [n for n in R.CONCEPTS if n not in self.drop and (signals is None or n in signals)]
        self.signals = keep
        self.events = [e for e in R.EVENTS if e not in self.drop]
        self.preds = [p for p in R.PREDICATES if p not in self.drop]
        self.back = {self.show(n): n for n in self.signals + self.events + self.preds}; self.shuffle = None

    def show(self, rid): return self.alias.get(rid, rid)

    def version(self): return hashlib.sha256(self.render().encode()).hexdigest()[:12]

    # ---- text for the prompt ---------------------------------------------------------------------
    def signal_line(self, rid, aliases=3):
        c = R.CONCEPTS[rid]; syn = [s for s in c.synonyms if s.lower() != c.label.lower()][:aliases]
        u = "" if c.unit in ("none", "category") else f" [{c.unit}]"
        return f"{self.show(rid)}: {c.label}{u}" + (f"; also {', '.join(syn)}" if syn else "")

    def event_line(self, rid):
        label, syn = R.EVENTS[rid]; return f"{self.show(rid)}: {label}" + (f"; also {', '.join(syn[:2])}" if syn else "")

    def pred_line(self, rid): return f"{self.show(rid)}: {R.PREDICATES[rid][0]}"

    def render(self, signals=None, events=None, preds=None):
        import random
        order = lambda xs: random.Random(self.shuffle).sample(list(xs), len(xs)) if self.shuffle is not None else list(xs)          # a catalog has no canonical order: training shows it in random orders
        s = [self.signal_line(n) for n in order(signals if signals is not None else self.signals)]
        e = [self.event_line(n) for n in order(events if events is not None else self.events)]
        p = [self.pred_line(n) for n in order(preds if preds is not None else self.preds)]
        cls = [f"{k}: {v[0]}" for k, v in R.TRIP_CLASSES.items()]
        met = ", ".join(sorted({m for e_ in R.ENTITY_METRICS.values() for m in e_}))
        return ("SIGNALS (name: meaning [unit]; also called)\n" + "\n".join(s) + "\nEVENTS\n" + "\n".join(e) + "\nCONDITIONS (use in c= or grp=)\n" + "\n".join(p) +
                "\nTRIP CLASSES (last field of sel=)\n" + "\n".join(cls) + "\nENTITY METRICS (sel=)\n" + met + "\n")

    def ops_text(self): return "\n".join(f"{o}: {OP_HELP[o]}" for o in OPS)

    # ---- grammar --------------------------------------------------------------------------------------
    def grammar(self, signals=None, events=None, preds=None):
        sig = [self.show(n) for n in (signals if signals is not None else self.signals)]
        ev = [self.show(n) for n in (events if events is not None else self.events)]
        pr = [self.show(n) for n in (preds if preds is not None else self.preds)]
        metrics = sorted({m for e in R.ENTITY_METRICS.values() for m in e}); classes = list(R.TRIP_CLASSES)
        units = sorted({u.replace("/", "_per_") for us in R.UNIT_FAMILIES.values() for u in us} | {"%"})
        return f'''root ::= "op=none" | "op=" op target? target2? agg? " t=" time time2? conds? sel? sel2? groups? event? before? after? unit?
op ::= {_alts(OPS)}
target ::= " tgt=" signal
target2 ::= " tgt2=" signal
agg ::= " agg=" aggname
aggname ::= {_alts(AGGS)}
signal ::= {_alts(sig)}
time ::= "now" | "all" | "last:" int ":" tunit | "cal:" calname | "date:" digit digit digit digit "-" digit digit "-" digit digit
time2 ::= " t2=" time
tunit ::= {_alts(TIME_UNITS)}
calname ::= {_alts(CALENDAR)}
conds ::= " c=" cond ("," cond)*
cond ::= pred | "sig:" signal ":" cmp ":" number (":" unitname)?
pred ::= {_alts(pr)}
cmp ::= {_alts(COND_OPS)}
sel ::= " sel=" selbody
sel2 ::= " sel2=" (selbody | "-")
selbody ::= entity ":" (metric | "-") ":" order ":" (int | "-") ":" (klass | "-")
entity ::= {_alts(ENTITIES)}
metric ::= {_alts(metrics)}
order ::= {_alts(ORDERS)}
klass ::= {_alts(classes)}
groups ::= " grp=" pred "," pred
event ::= " ev=" evname (" evs=" selector)?
evname ::= {_alts(ev)}
selector ::= {_alts(s for s in SELECTORS if s not in ("nth", "page"))}
before ::= " bef=" int
after ::= " aft=" int
unit ::= " unit=" unitname
unitname ::= {_alts(units)}
int ::= digit digit? digit? digit? digit?
number ::= "-"? digit+ ("." digit+)?
digit ::= [0-9]
'''

    # ---- back to the registry --------------------------------------------------------------------------
    def to_registry(self, text):
        """Replace display names in a model-written query by registry ids. Unknown names are left as they are (the parser / validator will reject them)."""
        if not self.alias: return text
        def m(n): return self.back.get(n, n)
        out = []
        for tok in text.split():
            if "=" not in tok: out.append(tok); continue
            k, v = tok.split("=", 1)
            if k in ("tgt", "tgt2", "ev"): v = m(v)
            elif k == "grp": v = ",".join(m(x) for x in v.split(","))
            elif k == "c":
                parts = []
                for c in v.split(","):
                    if c.startswith("sig:"): p = c.split(":"); p[1] = m(p[1]); c = ":".join(p)
                    else: c = m(c)
                    parts.append(c)
                v = ",".join(parts)
            out.append(f"{k}={v}")
        return " ".join(out)

    def to_display(self, text):
        """The inverse of to_registry (used to show gold queries to the model as examples under a renamed catalog)."""
        if not self.alias: return text
        def m(n): return self.alias.get(n, n)
        out = []
        for tok in text.split():
            if "=" not in tok: out.append(tok); continue
            k, v = tok.split("=", 1)
            if k in ("tgt", "tgt2", "ev"): v = m(v)
            elif k == "grp": v = ",".join(m(x) for x in v.split(","))
            elif k == "c":
                parts = []
                for c in v.split(","):
                    if c.startswith("sig:"): p = c.split(":"); p[1] = m(p[1]); c = ":".join(p)
                    else: c = m(c)
                    parts.append(c)
                v = ",".join(parts)
            out.append(f"{k}={v}")
        return " ".join(out)


def renamed_view(seed=0, drop=(), keep_fraction=1.0):
    """A perturbed catalog for schema-transfer tests: every signal / event / condition gets a different identifier (built from its own words), optionally some are withheld. Deterministic in `seed`."""
    import random
    rnd = random.Random(seed); alias = {}; used = set()
    def fresh(base):
        n = base; i = 2
        while n in used: n = f"{base}_{i}"; i += 1
        used.add(n); return n
    for n, c in R.CONCEPTS.items():
        words = [w for w in re.findall(r"[a-z]+", (rnd.choice([c.label] + list(c.synonyms[:2]))).lower()) if w not in ("the", "of")][:3] or [n]
        alias[n] = fresh("_".join(words) + rnd.choice(["", "_sig", "_val", "_ch"]))
    for e, (label, syn) in R.EVENTS.items(): alias[e] = fresh("ev_" + "_".join(re.findall(r"[a-z]+", label.lower())[:3]))
    for p in R.PREDICATES: alias[p] = fresh("cond_" + p)
    return SchemaView(alias=alias, drop=drop, name=f"renamed{seed}")
