"""The output grammar and prompts of the query-compiling language model.

The grammar is GENERATED from the registry, so the model can only write names that exist (signals, events, predicates, entity metrics, units, calendar words) and numbers; it cannot write SQL, a path, a vehicle
command or free text - the grammar has no rule for them. Adding a signal to the registry extends the grammar with no change here. Sampling is greedy and constrained, so every completion parses with Query.from_text
(whether it makes sense for THIS vehicle is the validator's and the capability engine's job, not the model's).

Prompts: `SYSTEM_FEWSHOT` for the base model (a handful of worked examples, cached as a shared prefix), `SYSTEM_TUNED` for the LoRA-adapted model (the adapter has learned the format; the prompt is one line)."""
from . import signal_registry as R
from . import topics as _TOPICS
from .query_ir import OPS, AGGS, TIME_UNITS, CALENDAR, COND_OPS, SELECTORS, ENTITIES, ORDERS, PLAN_PARAMS


def _alts(names): return " | ".join('"' + n + '"' for n in names)


PARAM_ENUMS = {
    "attr": ["duration", "start", "energy", "soc", "temperature", "power"],
    "by": ["day", "weekday"], "den": ["all", "driving", "parked", "city", "highway", "charging"], "dir": ["higher", "lower"], "list": ["0", "1"], "ord": ["asc", "desc"], "part": ["first", "second"],
    "vs": ["prev"], "gap": ["0", "1"], "iqr": ["0", "1"], "skiplatest": ["0", "1"], "k": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], "mode": ["count", "none", "any", "hours", "all_day", "some"],
    "slot": ["morning", "afternoon", "evening", "night"],
    "topic": sorted(_TOPICS.TOPICS),
    "why": ["unsupported", "unrecorded", "control", "offtopic", "definition", "ambiguous", "greeting", "unsafe"],
    "kind": ["certainty", "conflict", "proof", "causation", "exact", "tail", "criteria", "regen", "typical",
             "can_bus", "soc", "soh", "regen", "tpms", "adas", "lane_assist", "dtc", "wh_per_km", "kwh", "dc_fast", "ac_charging", "odometer", "aux_12v", "heat_pump", "obd", "uds",
             "someip", "vhal", "language_model", "range", "coolant", "ota", "pressure_units", "speed_units"],
}
PARAM_TYPED = {"thr": "cmpnum", "tripthr": "cmpnum", "mind": "int", "pct": "pcts", "hours": "int", "from": "hhmm", "to": "hhmm", "at": "hhmm", "n": "int", "span": "int", "v1": "number", "v2": "number"}


def _alts_params(pnames):
    out = []
    for n in pnames:
        if n in PARAM_ENUMS: out.append('"' + n + '=" (' + " | ".join('"' + v + '"' for v in PARAM_ENUMS[n]) + ")")
        elif n in PARAM_TYPED: out.append('"' + n + '=" ' + PARAM_TYPED[n])
        else: out.append('"' + n + '=" pval')
    return " | ".join(out)


def gbnf():
    targets = [n for n in R.CONCEPTS]
    metrics = sorted({m for e in R.ENTITY_METRICS.values() for m in e})
    events = list(R.EVENTS)
    preds = list(R.PREDICATES)
    classes = list(R.TRIP_CLASSES)
    units = sorted({u.replace("/", "_per_") for us in R.UNIT_FAMILIES.values() for u in us} | {"%"})
    pnames = sorted(PLAN_PARAMS)
    g = f'''root ::= "op=none" (" t=" time)? params? | "op=" op target? target2? agg? " t=" time time2? conds? conds2? den? sel? sel2? groups? event? before? after? params? unit?
params ::= (" p." param)+
param ::= {_alts_params(pnames)}
cmpnum ::= ("gt" | "lt" | "ge" | "le" | "eq") ":" number
pcts ::= int ("," int)*
hhmm ::= digit digit ":" digit digit
pval ::= [A-Za-z0-9_:.,/%-]+
den ::= ""
op ::= {_alts(OPS)}
target ::= " tgt=" signal
target2 ::= " tgt2=" signal
agg ::= " agg=" aggname
aggname ::= {_alts(AGGS)}
signal ::= {_alts(targets)}
time ::= "now" | "all" | "last:" int ":" tunit | "prev:" int ":" tunit | "cal:" calname | "date:" digit digit digit digit "-" digit digit "-" digit digit
time2 ::= " t2=" time
tunit ::= {_alts(TIME_UNITS)}
calname ::= {_alts(CALENDAR)}
conds ::= " c=" cond ("," cond)*
conds2 ::= " c2=" (cond ("," cond)* | "-")
cond ::= ("not:")? pred | "sig:" signal ":" cmp ":" number
pred ::= {_alts(preds)}
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
evname ::= {_alts(events)}
selector ::= {_alts(s for s in SELECTORS if s not in ("nth", "page"))} | "nth" int | "page" int
before ::= " bef=" int
after ::= " aft=" int
unit ::= " unit=" unitname
unitname ::= {_alts(units)}
int ::= digit digit? digit? digit? digit?
number ::= "-"? digit+ ("." digit+)?
digit ::= [0-9]
'''
    return g


SYSTEM_TUNED = "Translate the vehicle question into a query. Output only the query."

SYSTEM_FEWSHOT = """You translate a question about a vehicle's recorded data into one query line. Output only the query, nothing else.
Format: op=<operation> tgt=<signal> agg=<statistic> t=<time> [c=<conditions>] [sel=<entity:rank_by:order:k:class>] [grp=<a>,<b>] [ev=<event>] [unit=<unit>]
Operations: current_value aggregate extreme trend compare count list_events event_context rank correlation percent_time explain_event capability.
Time: now | all | last:<n>:<minute|hour|day|week|month> | cal:<yesterday|today|last_week|this_week|last_month|this_month|...> | date:YYYY-MM-DD.
Conditions: a name (highway city charging parked hard_braking raining night cold ...) or sig:<signal>:<gt|lt|ge|le|eq>:<number>.
Q: What is the battery temperature right now?
op=current_value tgt=battery_temperature t=now
Q: Highest battery temperature in the last 14 days
op=extreme tgt=battery_temperature agg=max t=last:14:day
Q: Average speed during hard braking last week
op=aggregate tgt=speed agg=mean t=cal:last_week c=hard_braking
Q: What share of the time was speed above 100 km/h yesterday?
op=percent_time t=cal:yesterday c=sig:speed:gt:100
Q: Average battery temperature during my three least efficient highway trips last month
op=aggregate tgt=battery_temperature agg=mean t=cal:last_month sel=trip:wh_per_km:desc:3:highway
Q: Is consumption higher on the highway than in the city?
op=compare tgt=wh_per_km agg=mean t=last:7:day grp=highway,city
Q: How many hard braking events in the last 14 days?
op=count t=last:14:day ev=HARSH_BRAKE
Q: What was the motor torque yesterday?
op=aggregate tgt=motor_torque agg=mean t=cal:yesterday
Q: Show me my 5 longest trips
op=rank t=all sel=trip:distance_driven:desc:5:-
Q: Is the battery getting hotter over the last two weeks?
op=trend tgt=battery_temperature t=last:2:week
Q: Does the car record tyre temperature?
op=capability tgt=tyre_temperature t=all
Q: How much did the charge level drop on my last trip?
op=aggregate tgt=battery_soc agg=delta t=all sel=trip:recency:desc:1:-
Parameters p.<name>=<value>: thr=<gt|lt|ge|le>:<number> (the level compared with), mind=<seconds>, pct=<n,n>, den=<state>, by=<day|weekday>, ord=<asc|desc>, k=<n>, part=<first|second>, list=<0|1>.
Extra operations: exists duration share episodes continuous days_with percentile percentile_vs window_change halves by_period pair_gap range parked_days longest_run trip_events epistemic none.
A state may be negated with not: (c=not:charging,parked). t2=prev:7:day is the 7 days before the last 7 days.
Q: Has my speed ever been over 120 km/h this week?
op=exists tgt=speed t=cal:this_week p.thr=gt:120 unit=km_per_h
Q: For how long was the pack above 45 degrees last week, not counting charging?
op=duration tgt=battery_temperature t=last:7:day c=not:charging p.thr=gt:45 unit=C
Q: On which days was coolant flow under 3 litres a minute for at least ten minutes?
op=days_with tgt=coolant_flow t=last:14:day p.list=1 p.mind=600 p.thr=lt:3 unit=L_per_min
Q: Give me the 75th percentile motor temperature while driving over the last month
op=percentile tgt=motor_temperature t=last:30:day c=driving p.pct=75
Q: Which day had the lowest average speed in the last two weeks?
op=by_period tgt=speed agg=mean t=last:14:day p.by=day p.k=1 p.ord=asc
Q: How many hard brakes in the city when not charging in the last 30 days?
op=count t=last:30:day c=city,not:charging ev=HARSH_BRAKE
Q: Will it rain in Paris tomorrow?
op=none t=all p.why=unsupported"""


def messages(question, mode="fewshot", context=None):
    """context: the previous turn's plan text (or None).  It is shown to the model as `Previous: <plan>` so that a follow-up can be resolved into one complete plan."""
    user = ("Previous: " + context + chr(10) if context else "") + "Q: " + question
    return [{"role": "system", "content": SYSTEM_FEWSHOT if mode == "fewshot" else SYSTEM_TUNED}, {"role": "user", "content": user}]


def training_prompt(question, context=None):
    """The exact text the LoRA is trained on and served with (chat template applied by the tokenizer / llama.cpp)."""
    return messages(question, "tuned", context)
