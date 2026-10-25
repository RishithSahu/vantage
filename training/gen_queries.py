"""Formal-template generation of (question, Query IR) pairs.

The IR is sampled FIRST, from the registry, so it is valid and its meaning is exact; the question is then rendered from it with many surface templates and phrase choices. The label is therefore correct by
construction (no human or model labelling noise) and every example can be executed against a database to check that the IR is answerable (build_dataset.py).

Compositional splits - the point of the exercise: the model must handle COMBINATIONS it never saw, because the space of questions is combinatorial and training data is not.
  train / val / test_iid   random split of everything not held out
  test_comp                held-out combinations of familiar atoms (e.g. coolant flow x highway, a city-trip ranking x any signal, two named conditions never seen as a pair, 95th percentile x last month)
  test_template            question templates (sentence shapes) never used in training
  test_phrasing            signal names / time wording never used in training (reserved synonyms)
  test_adv                 typos, casing, fillers, STT-style noise, off-topic questions, injection attempts (must become op=none or a safe query, never SQL / commands)
Every example records its tags, so a failure can be traced to the atom that broke."""
import datetime as dt
import random
import re
from dataclasses import dataclass, field

from vgpt import signal_registry as R
from vgpt.query_ir import Condition, EntitySelect, EventRef, Query, Scope, TimeSpec
from vgpt.query_validator import validate
from . import phrases as P

HELD_TEMPLATES = {3, 4}                                   # template ids reserved for test_template
COMP_TARGETS = {"coolant_flow", "hvac_power", "motor_temperature"}
COMP_PREDS = {"highway", "charging"}
COMP_GROUPS = {frozenset({"night", "cold"}), frozenset({"raining", "highway"}), frozenset({"hot_weather", "city"})}
COMP_RANK = {("trip", "city"), ("trip", "rainy")}
COMP_AGG_TIME = {("p95", "last_month"), ("std", "last_week"), ("median", "this_month")}
EXCLUSIVE = set(R.MODE_PREDICATES)
SYMMETRIC_GROUPS = [("city", "highway"), ("charging", "driving"), ("night", "cold"), ("raining", "highway"), ("hot_weather", "city"), ("cold", "hot_weather"), ("parked", "driving"), ("fast_charging", "charging")]
NUMERIC_ASK = [c for c, v in R.CONCEPTS.items() if v.dtype == "numeric" and c not in R.EXTENSIVE and c not in ("wh_per_km",) and c in P.EXTRA_TARGET_PHRASES]


@dataclass
class Example:
    text: str
    ir: str
    op: str
    tags: list = field(default_factory=list)
    template: int = 0
    reserved: bool = False
    noisy: bool = False
    split: str = ""

    def to_dict(self): return {"text": self.text, "ir": self.ir, "op": self.op, "tags": self.tags, "template": self.template, "split": self.split, "noisy": self.noisy}


class Gen:
    def __init__(self, seed=0, ref_year=2026):
        self.r = random.Random(seed); self.year = ref_year; self.reserved_used = False

    # ---------------------------------------------------------------- pieces
    def pick(self, xs): return self.r.choice(xs)

    def target_phrase(self, concept, allow_reserved):
        """-> phrase. With allow_reserved the reserved (held-out) phrase may be chosen, which marks the example."""
        if allow_reserved:
            res = P.phrases_for(concept, True)
            if res: self.reserved_used = True; return self.pick(res)
        return self.pick(P.phrases_for(concept, False))

    def time_spec(self, kinds=("cal", "rel", "all", "date"), weights=(45, 42, 6, 7), forbid_cal=(), default_ok=False):
        if default_ok and self.r.random() < 0.08: return TimeSpec("relative", 7, "day"), "", "time:default"                     # no period in the question: the engine's default (last 7 days) applies
        k = self.r.choices(kinds, weights)[0]
        if k == "cal":
            name = self.pick([c for c in P.CALENDAR_PHRASES if c not in forbid_cal]); ph = self.pick(P.CALENDAR_PHRASES[name])
            if name == "last_week": return TimeSpec("relative", 7, "day"), ph, f"time:cal:{name}"                         # a lone "last week" is the last 7 days, "last month" the last 30 (the tag keeps the phrase family)
            if name == "last_month": return TimeSpec("relative", 30, "day"), ph, f"time:cal:{name}"
            return TimeSpec("calendar", name=name), ph, f"time:cal:{name}"
        if k == "rel":
            unit = self.r.choices(list(P.REL_N), (1, 3, 8, 3, 2))[0]; n = self.pick(P.REL_N[unit]); u = unit + ("s" if n != 1 else "")
            if n == 1 and unit in ("hour",): return TimeSpec("relative", 1, unit), self.pick(["in the last hour", "over the past hour"]), "time:rel"
            return TimeSpec("relative", n, unit), self.pick(P.REL_TEMPLATES).format(n=self.pick([n, P.NUMBER_WORDS.get(n, n)]) if n <= 10 else n, u=u), "time:rel"
        if k == "all": return TimeSpec("all"), self.pick(P.ALL_PHRASES), "time:all"
        d = dt.date(self.year, self.r.randint(1, 9), self.r.randint(1, 28))
        return TimeSpec("date", date=d.isoformat()), self.pick([f"on {d.day} {P.MONTHS[d.month - 1]} {d.year}", f"on {d.isoformat()}", f"on {P.MONTHS[d.month - 1]} {d.day} {d.year}"]), "time:date"

    def predicate(self, exclude=()):
        name = self.pick([p for p in P.PRED_PHRASES if p not in exclude]); return name, self.pick(P.PRED_PHRASES[name])

    def threshold(self, exclude_target=None):
        sig = self.pick([s for s in P.THRESHOLDS if s != exclude_target]); unit, vals = self.pick(P.THRESHOLDS[sig]); v = self.pick(vals); op = self.r.choices(["gt", "lt", "ge", "le"], (55, 30, 8, 7))[0]
        spoken = self.pick(P.SPOKEN_UNIT[unit]); sp = self.target_phrase(sig, False)
        reg_unit = R.CONCEPTS[sig].unit
        cond = Condition(signal=sig, op=op, value=float(v), unit=None if unit == reg_unit else unit)
        ph = f"when the {sp} {self.pick(P.SIG_CONDITION_VERBS[op])} {v} {spoken}" if self.r.random() < 0.7 else f"with the {sp} {self.pick(P.CMP_PHRASES[op])} {v} {spoken}"
        self.last_sig = (sp, f"{self.pick(P.CMP_PHRASES[op])} {v} {spoken}")
        return cond, ph, f"sig:{sig}"

    def conditions(self, n, target=None):
        """n conditions without contradictions -> (list[Condition], [phrases], [tags])."""
        conds, phs, tags = [], [], []; mode_used = False
        for _ in range(n):
            if self.r.random() < 0.65:
                excl = set(EXCLUSIVE) if mode_used else set()
                excl |= {c.predicate for c in conds if c.predicate}
                if any(c.predicate in ("hard_braking", "hard_acceleration") for c in conds): excl |= {"parked", "charging", "idling", "fast_charging"}
                name, ph = self.predicate(excl)
                if name in ("hard_braking", "hard_acceleration"): excl |= {"parked", "charging"}
                if name in EXCLUSIVE and mode_used: continue
                if name in EXCLUSIVE: mode_used = True
                if name in ("hard_braking", "hard_acceleration") and any(c.predicate in ("parked", "charging", "idling", "fast_charging") for c in conds): continue
                if name in ("parked", "charging", "idling", "fast_charging") and any(c.predicate in ("hard_braking", "hard_acceleration") for c in conds): continue
                conds.append(Condition(predicate=name)); phs.append(ph); tags.append(f"pred:{name}")
            else:
                c, ph, tg = self.threshold(target)
                if any(x.signal == c.signal for x in conds): continue
                conds.append(c); phs.append(ph); tags.append(tg)
        return conds, phs, tags

    def selection(self, rank_needed=False):
        ent = "trip" if self.r.random() < 0.88 else "charge_session"
        keys = [k for k in P.RANK_ADJ if k[0] in R.ENTITY_METRICS[ent]]
        metric, order = self.pick(keys); adj = self.pick(P.RANK_ADJ[(metric, order)])
        klass = self.pick(list(P.KLASS_WORDS)) if ent == "trip" and metric != "recency" and self.r.random() < 0.55 else None
        k = None
        if metric == "recency": k = 1 if self.r.random() < 0.7 else self.pick([2, 3, 5])
        else: k = self.r.choice([1, 1, 2, 3, 3, 5])
        kw = self.pick(P.NUMBER_WORDS and [str(k), P.NUMBER_WORDS.get(k, str(k))]) if k > 1 else ""
        noun_pl = self.pick(P.ENTITY_WORDS[ent]); noun_s = self.pick(P.ENTITY_SINGULAR[ent]); kl = self.pick(P.KLASS_WORDS[klass]) + " " if klass else ""
        if k == 1: ph = f"my {adj} {kl}{noun_s}" if self.r.random() < 0.6 else f"the {adj} {kl}{noun_s}"
        else: ph = f"my {kw} {adj} {kl}{noun_pl}"
        sel = EntitySelect(ent, klass, metric, order, k)
        return sel, ph, [f"sel:{ent}:{klass or '-'}", f"rank:{metric}"], ent, klass

    def agg_noun(self, agg, tphrase):
        a = self.pick(P.AGG_PHRASES[agg])
        if "{T}" in a: return a.replace("{T}", tphrase)
        return f"{a} {tphrase}"

    def tid(self): return self.r.choices([0, 1, 2, 3, 4], (30, 30, 24, 8, 8))[0]

    # ---------------------------------------------------------------- families (each returns Example or None)
    def f_aggregate(self):
        tgt = self.pick(NUMERIC_ASK); allow = self.r.random() < 0.10
        agg = self.r.choices(["mean", "median", "std", "p95", "delta"], (60, 8, 9, 7, 16))[0]
        if tgt in ("battery_soc", "battery_temperature", "battery_soh") and agg == "delta" and self.r.random() < 0.5: agg = "delta"
        if R.CONCEPTS[tgt].dtype != "numeric": return None
        n_cond = self.r.choices([0, 1, 2], (50, 38, 12))[0]; sel = None
        conds, cph, ctags = self.conditions(n_cond, tgt) if n_cond else ([], [], [])
        sph = ""; stags = []
        if self.r.random() < 0.14 and not conds:
            sel, sph, stags, _, _ = self.selection(); conds_sel = True
        tm, tph, ttag = self.time_spec(default_ok=True)
        if sel is not None and sel.rank_by == "recency" and self.r.random() < 0.7: tm, tph, ttag = TimeSpec("all"), "", "time:all"
        tphrase = self.target_phrase(tgt, allow); noun = self.agg_noun(agg, tphrase); tid = self.tid()
        c_txt = " ".join(cph); s_txt = f"during {sph}" if sph else ""
        pats = [f"what was the {noun} {c_txt} {s_txt} {tph}", f"{noun} {c_txt} {s_txt} {tph}", f"{tph} what was the {noun} {c_txt} {s_txt}", f"how much was the {noun} {c_txt} {s_txt} {tph}", f"tell me the {noun} {c_txt} {s_txt} {tph}"]
        q = Query(op="aggregate", target=tgt, agg=agg, scope=Scope(tm, sel, conds)); tags = [f"op:aggregate", f"tgt:{tgt}", f"agg:{agg}", ttag] + ctags + stags
        return self.finish(q, pats[tid], tid, tags)

    def f_extreme(self):
        tgt = self.pick(NUMERIC_ASK); agg = self.pick(["max", "min"]); allow = self.r.random() < 0.10
        if R.CONCEPTS[tgt].dtype != "numeric": return None
        n_cond = self.r.choices([0, 1], (70, 30))[0]; conds, cph, ctags = self.conditions(n_cond, tgt) if n_cond else ([], [], [])
        tm, tph, ttag = self.time_spec(default_ok=True); tphrase = self.target_phrase(tgt, allow); noun = self.agg_noun(agg, tphrase); tid = self.tid(); c_txt = " ".join(cph)
        pats = [f"what was the {noun} {c_txt} {tph}", f"{noun} {c_txt} {tph}", f"{tph} what was the {noun} {c_txt}", f"how high did it get: the {noun} {c_txt} {tph}" if agg == "max" else f"how low did it get: the {noun} {c_txt} {tph}", f"tell me the {noun} {c_txt} {tph}"]
        q = Query(op="extreme", target=tgt, agg=agg, scope=Scope(tm, None, conds)); return self.finish(q, pats[tid], tid, ["op:extreme", f"tgt:{tgt}", f"agg:{agg}", ttag] + ctags)

    def f_extensive(self):
        tgt = self.pick(["energy_used", "distance_driven", "regen_energy"]); tm, tph, ttag = self.time_spec(default_ok=True); tid = self.tid(); allow = self.r.random() < 0.08
        n_cond = self.r.choices([0, 1], (65, 35))[0]; conds, cph, ctags = self.conditions(n_cond, None) if n_cond else ([], [], [])
        if tgt == "regen_energy": conds = [c for c in conds if c.predicate not in ("parked", "charging", "idling", "regenerating")]; cph = [self.phrase_of(c) for c in conds]
        tp = self.target_phrase(tgt, allow); c_txt = " ".join(cph)
        base = {"energy_used": ["how much energy did i use", "how much electricity did the car use", "what was the energy used"], "distance_driven": ["how far did i drive", "how many kilometres did i drive", "what distance did i cover"],
                "regen_energy": ["how much energy did regen recover", "how much energy did i get back from regenerative braking", "what was the regenerated energy"]}[tgt]
        pats = [f"{self.pick(base)} {c_txt} {tph}", f"total {tp} {c_txt} {tph}", f"{tph} {self.pick(base)} {c_txt}", f"what is the combined {tp} {c_txt} {tph}", f"tell me the total {tp} {c_txt} {tph}"]
        q = Query(op="aggregate", target=tgt, agg="sum", scope=Scope(tm, None, conds)); return self.finish(q, pats[tid], tid, ["op:aggregate", f"tgt:{tgt}", "agg:sum", ttag] + ctags)

    def phrase_of(self, c):
        return self.pick(P.PRED_PHRASES[c.predicate]) if c.predicate else ""

    def f_efficiency(self):
        tm, tph, ttag = self.time_spec(default_ok=True); conds, cph, ctags = self.conditions(self.r.choices([0, 1], (60, 40))[0], None); tid = self.tid(); tp = self.target_phrase("wh_per_km", self.r.random() < 0.08)
        pats = [f"what was my {tp} {' '.join(cph)} {tph}", f"{tp} {' '.join(cph)} {tph}", f"{tph} what was my {tp} {' '.join(cph)}", f"how was my {tp} {' '.join(cph)} {tph}", f"tell me my {tp} {' '.join(cph)} {tph}"]
        q = Query(op="aggregate", target="wh_per_km", agg="mean", scope=Scope(tm, None, conds)); return self.finish(q, pats[tid], tid, ["op:aggregate", "tgt:wh_per_km", "agg:mean", ttag] + ctags)

    def f_current(self):
        tgt = self.pick(NUMERIC_ASK + ["range_estimate", "battery_soh"]); tid = self.tid(); tp = self.target_phrase(tgt, self.r.random() < 0.10)
        pats = [f"what is the {tp} right now", f"current {tp}", f"how is the {tp} at the moment", f"what's the {tp} now", f"tell me the {tp} currently"]
        q = Query(op="current_value", target=tgt, scope=Scope(TimeSpec("current"))); return self.finish(q, pats[tid], tid, ["op:current_value", f"tgt:{tgt}", "time:now"])

    def f_percent(self):
        tm, tph, ttag = self.time_spec(); tid = self.tid(); bodies = []; tags = []; conds = []
        if self.r.random() < 0.6:
            c, ph, tg = self.threshold(); sp, tail = self.last_sig; conds.append(c); bodies.append(f"was the {sp} {tail}"); tags.append(tg)
            if self.r.random() < 0.2:
                extra, ep, et = self.conditions(1, c.signal)
                if extra and extra[0].predicate: conds += extra; bodies.append("and it was " + ep[0].replace("when ", "").replace("while ", "")); tags += et
        else:
            cs, phs, tags = self.conditions(1); conds = cs
            if not cs: return None
            bodies = [f"was it {phs[0]}"]
        body = " ".join(bodies)
        pats = [f"what share of the time {body} {tph}", f"what percent of the time {body} {tph}", f"{tph} how much of the time {body}", f"how often {body} {tph}", f"what fraction of the time {body} {tph}"]
        q = Query(op="percent_time", scope=Scope(tm, None, conds)); return self.finish(q, pats[tid], tid, ["op:percent_time", ttag] + tags)

    @staticmethod
    def strip_when(ph): return re.sub(r"^(when|with|while) ", "", ph) if ph.startswith(("when ", "with ")) else ph

    def f_count(self):
        ev = self.pick([e for e in P.EVENT_PLURAL if e not in ("COOLANT_FLOW_DROP",)]); tm, tph, ttag = self.time_spec(); tid = self.tid(); plural = self.pick(P.EVENT_PLURAL[ev])
        n_cond = 1 if self.r.random() < 0.18 and ev not in ("HARSH_BRAKE", "HARSH_ACCEL") else 0; conds, cph, ctags = self.conditions(n_cond) if n_cond else ([], [], [])
        pats = [f"how many {plural} {' '.join(cph)} {tph}", f"number of {plural} {' '.join(cph)} {tph}", f"{tph} how many {plural} {' '.join(cph)}", f"did i have any {plural} {' '.join(cph)} {tph}", f"count the {plural} {' '.join(cph)} {tph}"]
        q = Query(op="count", scope=Scope(tm, None, conds), event=EventRef(ev, "last")); return self.finish(q, pats[tid], tid, ["op:count", f"ev:{ev}", ttag] + ctags)

    def f_list_events(self):
        ev = self.pick(list(P.EVENT_PLURAL)); tm, tph, ttag = self.time_spec(); tid = self.tid(); plural = self.pick(P.EVENT_PLURAL[ev]); sing = self.pick(P.EVENT_PHRASES[ev])
        first = self.r.random() < 0.15
        pats = [f"list the {plural} {tph}", f"show me the {plural} {tph}", f"{tph} which {plural} were there", f"when was the {'first' if first else 'last'} {sing}", f"what {plural} happened {tph}"]
        if tid == 3: q = Query(op="list_events", scope=Scope(TimeSpec("all")), event=EventRef(ev, "first" if first else "last")); return self.finish(q, pats[3], 3, ["op:list_events", f"ev:{ev}", "time:all"])
        q = Query(op="list_events", scope=Scope(tm), event=EventRef(ev, "last")); return self.finish(q, pats[tid], tid, ["op:list_events", f"ev:{ev}", ttag])

    def f_context(self):
        ev = self.pick(["WARN_BATT_OVERTEMP", "DTC_SET", "TPMS_LOW", "ADAS_DISENGAGE", "COOLANT_FLOW_DROP", "ANY_WARNING", "HARSH_BRAKE", "CAMERA_BLOCKED"]); sing = self.pick(P.EVENT_PHRASES[ev]); tid = self.tid()
        mins = self.pick([5, 10, 15, 30]); e = EventRef(ev, "last", None, mins * 60 if tid in (2, 4) else 600, 0)                # only the templates that SAY the number carry it
        pats = [f"what happened before the {sing}", f"what led up to the {sing}", f"what happened in the {mins} minutes before the {sing}", f"what was going on just before the last {sing}", f"show me the {mins} minutes leading up to the {sing}"]
        q = Query(op="event_context", scope=Scope(TimeSpec("all")), event=e); return self.finish(q, pats[tid], tid, ["op:event_context", f"ev:{ev}"])

    def f_explain(self):
        ev = self.pick(P.EXPLAINABLE); sing = self.pick(P.EVENT_PHRASES[ev]); tid = self.tid()
        pats = [f"why did the {sing} happen", f"what caused the {sing}", f"why was there a {sing}", f"what is behind the last {sing}", f"explain the {sing}"]
        q = Query(op="explain_event", scope=Scope(TimeSpec("all")), event=EventRef(ev, "last")); return self.finish(q, pats[tid], tid, ["op:explain_event", f"ev:{ev}"])

    def f_compare(self):
        tid = self.tid(); tgt = self.pick(["wh_per_km", "energy_used"] + NUMERIC_ASK[:25]); tp = self.target_phrase(tgt, self.r.random() < 0.08)
        if self.r.random() < 0.7:
            a, b = self.pick(SYMMETRIC_GROUPS);
            if self.r.random() < 0.5: a, b = b, a
            tm, tph, ttag = self.time_spec(); na, nb = self.pick(P.PRED_PHRASES[a]), self.pick(P.PRED_PHRASES[b]); agg = "mean" if tgt not in R.EXTENSIVE else None
            pats = [f"compare {tp} {na} and {nb} {tph}", f"is the {tp} higher {na} than {nb} {tph}", f"how does the {tp} {na} differ from {nb} {tph}", f"{tp} {na} versus {nb} {tph}", f"which is higher, {tp} {na} or {nb} {tph}"]
            if tgt == "energy_used" and a in ("charging", "parked", "fast_charging") or b in ("charging", "parked", "fast_charging") and tgt in ("energy_used", "wh_per_km"): return None
            q = Query(op="compare", target=tgt, agg=agg or "sum", groups=[a, b], scope=Scope(tm)); return self.finish(q, pats[tid], tid, ["op:compare", f"tgt:{tgt}", f"grp:{a}+{b}", ttag])
        (t1, p1, g1), (t2, p2, g2) = self.two_times();
        pats = [f"compare the {tp} {p1} with {p2}", f"is the {tp} higher {p1} than {p2}", f"how does the {tp} {p1} differ from {p2}", f"{tp} {p1} versus {p2}", f"{tp} {p1} compared to {p2}"]
        q = Query(op="compare", target=tgt, agg="mean" if tgt not in R.EXTENSIVE else "sum", scope=Scope(t1), scope_b=Scope(t2)); return self.finish(q, pats[tid], tid, ["op:compare", f"tgt:{tgt}", "cmp:periods"])

    def two_times(self):
        pairs = [("this_week", "last_week"), ("this_month", "last_month"), ("today", "yesterday"), ("yesterday", "day_before_yesterday")]
        a, b = self.pick(pairs)
        if self.r.random() < 0.5: a, b = b, a
        return (TimeSpec("calendar", name=a), self.pick(P.CALENDAR_PHRASES[a]), f"time:cal:{a}"), (TimeSpec("calendar", name=b), self.pick(P.CALENDAR_PHRASES[b]), f"time:cal:{b}")

    def f_trend(self):
        tgt = self.pick(NUMERIC_ASK + ["wh_per_km"]); tm, tph, ttag = self.time_spec(kinds=("cal", "rel"), weights=(35, 65), forbid_cal=("today", "this_morning", "last_night", "yesterday", "day_before_yesterday", "last_weekend")); tid = self.tid()
        if tm.type == "relative" and tm.unit in ("minute", "hour"): tm, tph, ttag = TimeSpec("relative", 14, "day"), "over the past 14 days", "time:rel"
        tp = self.target_phrase(tgt, self.r.random() < 0.08); w = self.pick(P.TREND_WORDS)
        pats = [f"is the {tp} {w} {tph}", f"how has the {tp} changed {tph}", f"what is the trend of the {tp} {tph}", f"has the {tp} been {w} {tph}", f"{tp} trend {tph}"]
        q = Query(op="trend", target=tgt, scope=Scope(tm)); return self.finish(q, pats[tid], tid, ["op:trend", f"tgt:{tgt}", ttag])

    def f_correlation(self):
        a, b = self.r.sample(NUMERIC_ASK, 2); tm, tph, ttag = self.time_spec(kinds=("cal", "rel"), weights=(30, 70), forbid_cal=("today", "this_morning", "last_night", "day_before_yesterday", "last_weekend")); tid = self.tid()
        if tm.type == "relative" and tm.unit in ("minute", "hour"): tm, tph, ttag = TimeSpec("relative", 7, "day"), "in the last 7 days", "time:rel"
        pa, pb = self.target_phrase(a, False), self.target_phrase(b, False)
        pats = [f"is the {pa} related to the {pb} {tph}", f"what is the correlation between {pa} and {pb} {tph}", f"does the {pa} depend on the {pb} {tph}", f"how strongly is {pa} linked to {pb} {tph}", f"relationship between {pa} and {pb} {tph}"]
        q = Query(op="correlation", target=a, target2=b, scope=Scope(tm)); return self.finish(q, pats[tid], tid, ["op:correlation", f"tgt:{a}", f"tgt2:{b}", ttag])

    def f_rank(self):
        sel, ph, tags, ent, klass = self.selection(); tm, tph, ttag = self.time_spec(kinds=("cal", "rel", "all"), weights=(30, 30, 40)); tid = self.tid()
        if sel.rank_by == "recency": tm, tph, ttag = TimeSpec("all"), "", "time:all"
        if sel.k == 1: sel2 = EntitySelect(sel.entity, sel.klass, sel.rank_by, sel.order, 1)
        adj_ph = ph
        pats = ([f"show me {ph} {tph}", f"which was {ph} {tph}", f"what was {ph} {tph}", f"tell me about {ph} {tph}", f"give me {ph} {tph}"] if sel.k == 1 else
                [f"show me {ph} {tph}", f"which were {ph} {tph}", f"list {ph} {tph}", f"what were {ph} {tph}", f"give me {ph} {tph}"])
        if sel.rank_by == "recency": return None
        q = Query(op="rank", scope=Scope(tm, sel)); return self.finish(q, pats[tid], tid, ["op:rank"] + tags + [ttag])

    def f_capability(self):
        tgt = self.pick(list(P.EXTRA_TARGET_PHRASES)); tp = self.target_phrase(tgt, self.r.random() < 0.08); tid = self.tid()
        pats = [f"does the car record {tp}", f"do you have {tp} data", f"can you measure {tp}", f"is {tp} available", f"does it log {tp}"]
        q = Query(op="capability", target=tgt, scope=Scope(TimeSpec("all"))); return self.finish(q, pats[tid], tid, ["op:capability", f"tgt:{tgt}"])

    def f_none(self):
        if self.r.random() < 0.55: tpl = self.pick(P.OFF_TEMPLATES); fill = P.OFF_FILL; tag = "offtopic"
        else: tpl = self.pick(P.ATTACK_TEMPLATES); fill = P.ATTACK_FILL; tag = "attack"
        text = re.sub(r"\{(\w+)\}", lambda m: self.pick(fill[m.group(1)]), tpl)
        return Example(text, "op=none", "none", ["op:none", tag], 0)

    # ---------------------------------------------------------------- assembly
    def finish(self, q, text, tid, tags):
        text = " ".join(text.split()); text = text.replace(" ?", "?")
        if validate(q).codes(): return None
        ex = Example(text.strip(), q.to_text(), q.op, list(tags), tid, self.reserved_used); self.reserved_used = False; return ex

    FAMILIES = [("aggregate", 20), ("extreme", 8), ("extensive", 8), ("efficiency", 4), ("current", 8), ("percent", 8), ("count", 6), ("list_events", 4), ("context", 4), ("explain", 3), ("compare", 8), ("trend", 6),
                ("correlation", 4), ("rank", 5), ("capability", 4), ("none", 12)]

    def one(self):
        fams = [f for f, _ in self.FAMILIES]; w = [x for _, x in self.FAMILIES]
        for _ in range(20):
            ex = getattr(self, "f_" + self.r.choices(fams, w)[0])()
            if ex is not None: return ex
        return None

    # ---------------------------------------------------------------- noise
    def noisy(self, ex, level=1):
        t = ex.text
        if ex.op != "none" or self.r.random() < 0.2: t = self.pick(P.FILLERS_PRE) + t + self.pick(P.FILLERS_POST)
        for _ in range(level):
            kind = self.r.choice(["typo", "typo", "case", "drop_space", "stt", "punct"])
            words = t.split()
            if kind == "typo" and words:
                i = self.r.randrange(len(words)); w = words[i]
                if len(w) > 4: j = self.r.randrange(1, len(w) - 1); w = w[:j] + w[j + 1] + w[j] + w[j + 2:] if self.r.random() < 0.5 else w[:j] + w[j + 1:]
                words[i] = w; t = " ".join(words)
            elif kind == "case": t = t.upper() if self.r.random() < 0.3 else t.capitalize() if self.r.random() < 0.5 else t.title()
            elif kind == "drop_space" and len(words) > 3: i = self.r.randrange(len(words) - 1); words[i] = words[i] + words[i + 1]; del words[i + 1]; t = " ".join(words)
            elif kind == "stt": t = re.sub(r"[?.,']", "", t.lower()); t = t.replace(" km/h", " kilometres per hour") if self.r.random() < 0.3 else t
            elif kind == "punct": t = t + self.pick(["?", "??", "!", " ...", "?!"])
        return Example(t.strip(), ex.ir, ex.op, ex.tags + ["noise"], ex.template, ex.reserved, True)


# ------------------------------------------------------------------------------------------------ splitting
def comp_held_out(ex):
    tags = set(ex.tags)
    tg = {t[4:] for t in tags if t.startswith("tgt:")}; pr = {t[5:] for t in tags if t.startswith("pred:")}
    if tg & COMP_TARGETS and pr & COMP_PREDS: return True
    for t in tags:
        if t.startswith("grp:") and frozenset(t[4:].split("+")) in COMP_GROUPS: return True
        if t.startswith("sel:"):
            _, ent, kl = t.split(":");
            if (ent, kl) in COMP_RANK: return True
    agg = {t[4:] for t in tags if t.startswith("agg:")}; cal = {t[len("time:cal:"):] for t in tags if t.startswith("time:cal:")}
    return any((a, c) in COMP_AGG_TIME for a in agg for c in cal)


def assign_split(ex, r):
    if ex.noisy: return "test_adv"
    if ex.op == "none": return "train" if r.random() < 0.9 else "test_adv"
    if ex.reserved: return "test_phrasing"
    if ex.template in HELD_TEMPLATES: return "test_template"
    if comp_held_out(ex): return "test_comp"
    x = r.random(); return "train" if x < 0.90 else "val" if x < 0.95 else "test_iid"


def generate(n=24000, seed=0, noise_rate=0.12, adv_n=1500):
    """-> list[Example] with .split assigned. Deduplicated by question text (a text maps to one IR)."""
    g = Gen(seed); r = random.Random(seed + 1); seen = {}; out = []
    # templates 3 and 4 and reserved phrasing must be generated too: they only flow into test splits
    while len(out) < n:
        ex = g.one()
        if ex is None or ex.text in seen: continue
        seen[ex.text] = ex.ir
        if r.random() < noise_rate and ex.op != "none": ex = Gen.noisy(g, ex, 1); ex.noisy = False; ex.tags = [t for t in ex.tags if t != "noise"] + ["light_noise"]
        ex.split = assign_split(ex, r); out.append(ex)
    adv = []
    while len(adv) < adv_n:                                                                   # adversarial split: fresh examples with heavier noise
        ex = g.one()
        if ex is None: continue
        ex = Gen.noisy(g, ex, r.choice([1, 2, 3]));
        if ex.text in seen: continue
        seen[ex.text] = ex.ir; ex.split = "test_adv"; adv.append(ex)
    return out + adv
