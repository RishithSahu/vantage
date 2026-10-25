"""Evidence: the structured, auditable basis of an answer. An Evidence object is built from an executed Result and travels with the answer to the UI (concise answer first, evidence underneath),
to the verifier (every number in the text must be traceable to an item here) and to the logs.

    claim        the one-sentence conclusion the evidence supports
    items        the numbers: metric, value, unit, label   (the ONLY numbers an answer may state)
    period       the time window, in words and as timestamps
    samples / sources / tier / quality / flags    how much data, from where, how trustworthy
    kind         measured | derived | estimated   (an estimate is never presented as a measurement)
    graph        for causal-style questions: linked evidence nodes with careful relation words (preceded, associated with, likely contributor) - never a bare "caused"."""
from dataclasses import dataclass, field
from typing import Optional

from . import signal_registry as R

RELATIONS = ("preceded", "followed", "associated_with", "likely_contributor")


@dataclass
class EvidenceItem:
    metric: str
    value: object
    unit: str = ""
    label: str = ""
    role: str = "value"                 # value | comparison | derived | threshold | count

    def to_dict(self): return {"metric": self.metric, "value": self.value, "unit": self.unit, "label": self.label, "role": self.role}


@dataclass
class Node:
    id: str
    kind: str                           # event | signal_change | state | warning
    label: str
    ts: Optional[int] = None
    value: Optional[float] = None
    unit: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    strength: str = "moderate"          # strong | moderate | weak
    note: str = ""


@dataclass
class EvidenceGraph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def add(self, node): self.nodes.append(node); return node.id
    def link(self, src, dst, relation, strength="moderate", note=""):
        assert relation in RELATIONS, relation
        self.edges.append(Edge(src, dst, relation, strength, note))

    def to_dict(self): return {"nodes": [n.__dict__ for n in self.nodes], "edges": [e.__dict__ for e in self.edges]}

    def text_lines(self):
        by = {n.id: n for n in self.nodes}
        words = {"preceded": "preceded", "followed": "followed", "associated_with": "is associated with", "likely_contributor": "is a likely contributor to"}
        return [f"{by[e.src].label} {words[e.relation]} {by[e.dst].label}" + (f" ({e.strength})" if e.strength else "") + (f" - {e.note}" if e.note else "") for e in self.edges if e.src in by and e.dst in by]


@dataclass
class Evidence:
    claim: str = ""
    items: list = field(default_factory=list)
    period: str = ""
    period_ts: tuple = (0, 0)
    samples: int = 0
    sources: list = field(default_factory=list)
    tier: str = ""
    quality: str = "high"
    flags: list = field(default_factory=list)
    kind: str = "measured"
    assumptions: list = field(default_factory=list)
    method: str = ""
    graph: Optional[EvidenceGraph] = None
    caveats: list = field(default_factory=list)
    status: str = "ok"

    def numbers(self):
        out = []
        for it in self.items:
            if isinstance(it.value, bool): continue
            if isinstance(it.value, (int, float)): out.append(float(it.value))
        return out

    def add(self, metric, value, unit="", label="", role="value"):
        self.items.append(EvidenceItem(metric, value, unit, label or R.label_of(metric), role)); return self

    def to_dict(self):
        return {"claim": self.claim, "items": [i.to_dict() for i in self.items], "period": self.period, "period_ts": list(self.period_ts), "samples": self.samples, "sources": self.sources, "tier": self.tier, "quality": self.quality,
                "flags": self.flags, "kind": self.kind, "assumptions": self.assumptions, "method": self.method, "caveats": self.caveats, "status": self.status, "graph": self.graph.to_dict() if self.graph else None}


def _flags_unique(flags):
    seen, out = set(), []
    for f in flags:
        k = (f.get("code"), f.get("signal"), f.get("detail"))
        if k not in seen: seen.add(k); out.append(f)
    return out


def from_result(res, q=None, profile=None):
    """Result (+ Query) -> Evidence. Items are exactly the numbers the composer is allowed to quote."""
    ev = Evidence(period=res.period_label, period_ts=tuple(res.period), samples=res.samples, sources=list(res.sources), tier=res.tier, quality=res.quality, flags=_flags_unique(res.flags), kind=res.quality_kind or "measured",
                  assumptions=list(res.assumptions), method=res.formula, status=res.status)
    if res.estimated or (res.quality_kind == "estimated"): ev.kind = "estimated"
    t = res.target
    if res.status != "ok": return ev
    op = res.op
    if op in ("current_value", "aggregate", "extreme", "percent_time") and isinstance(res.value, (int, float)): ev.add(t or op, res.value, res.unit, R.label_of(t) if t else "")
    if op == "percent_time" and isinstance(res.value, (int, float)) and res.samples:
        mins = int(round(res.value / 100.0 * res.samples)); ev.add("time_spent_min", mins, "min", "time spent", role="derived"); ev.add("time_spent_h", round(mins / 60.0, 1), "h", "time spent", role="derived")
    elif op == "current_value" and isinstance(res.value, str): ev.items.append(EvidenceItem(t, res.value, "", R.label_of(t)))
    if op == "current_value":
        for k, v in res.extra.get("wheels", {}).items():
            if v is not None: ev.add(f"tyre_pressure_{k}", v, "kPa")
        if res.extra.get("gps") is not None: ev.add("gps_speed", res.extra["gps"], "km/h")
        if res.extra.get("soc") is not None: ev.add("battery_soc", res.extra["soc"], "%")
        if res.extra.get("wh_per_km") is not None: ev.add("wh_per_km", res.extra["wh_per_km"], "Wh/km", role="derived")
    if op == "extreme":
        for k, v in res.extra.get("state", {}).items(): ev.add(k, v, R.CONCEPTS[k].unit, role="value")
    if op == "trend":
        x = res.extra; ev.add(t, res.value, res.unit, "slope", "derived"); ev.add(t + "_first", x["first"], R.CONCEPTS[t].unit, "start of period"); ev.add(t + "_last", x["last"], R.CONCEPTS[t].unit, "end of period")
        ev.add(t + "_change", x["change"], R.CONCEPTS[t].unit, "change", "derived")
        if x.get("pct") is not None: ev.add(t + "_pct", x["pct"], "%", "relative change", "derived")
        if x.get("p") is not None: ev.add("p_value", x["p"], "", "p-value", "derived")
        ev.add("days", x["days"], "count", "days of data", "count")
    if op == "compare":
        for i in res.items: ev.add(f"{t}_{i['group']}", i["value"], i["unit"], str(i["group"]), "comparison")
        ev.add(t + "_difference", res.value, res.unit, "difference", "derived")
        if res.extra.get("ratio_pct") is not None: ev.add(t + "_ratio_pct", res.extra["ratio_pct"], "%", "relative difference", "derived")
        if res.extra.get("p") is not None: ev.add("p_value", res.extra["p"], "", "p-value", "derived")
    if op in ("count", "list_events"): ev.add(t or "events", res.value, "count", "number of events", "count")
    if op == "rank":
        for n, it in enumerate(res.items, 1):
            ev.add(f"rank_{n}", it["value"], it.get("unit", ""), f"#{n}", "value")
            if it.get("unit") == "s": ev.add(f"rank_{n}_min", it["value"] / 60.0, "min", f"#{n} in minutes", "derived")                       # durations are worded in minutes
            ev.add(f"rank_{n}_km", it.get("km"), "km") if it.get("km") is not None else None
            if it.get("start") is not None and it.get("end") is not None: ev.add(f"rank_{n}_duration", (it["end"] - it["start"]) / 60.0, "min", f"#{n} duration", "derived")      # an answer may say how long the trip lasted
        ev.add("selected", len(res.items), "count", "selected", "count")
    if op == "correlation":
        ev.add("pearson_r", res.value, "r", "Pearson r", "derived")
        if res.extra.get("spearman") is not None: ev.add("spearman_r", res.extra["spearman"], "r", "Spearman r", "derived")
    if op == "event_context":
        for s in res.items:
            for k in ("first", "last", "min", "max", "mean", "change"): ev.add(f"{s['signal']}_{k}", s[k], s["unit"], f"{s['label']} {k}", "value" if k != "change" else "derived")
    if op == "detect_change":
        x = res.extra; u = x["unit"]; sh = x.get("shift"); jp = x.get("jump")
        if sh: ev.add(t + "_before", sh["before"], u, "level before the change"); ev.add(t + "_after", sh["after"], u, "level after the change"); ev.add(t + "_shift", sh["change"], u, "change", "derived"); ev.add("p_adj", sh["p_adj"], "", "adjusted p-value", "derived"); ev.add("effect_size", sh["effect_size"], "", "effect size", "derived")
        if sh and sh.get("pct") is not None: ev.add(t + "_shift_pct", sh["pct"], "%", "relative change", "derived")
        if jp: ev.add(t + "_jump_from", jp["from"], u, "day before the largest move"); ev.add(t + "_jump_to", jp["to"], u, "day of the largest move"); ev.add(t + "_jump", jp["change"], u, "largest day-to-day move", "derived")
        ev.add("days", x["days"], "count", "days of data", "count")
    if op == "explain_change":
        x = res.extra; o = x["obs"]; u = x["unit"]; ev.add(t + "_before", o["before"], u, "earlier period"); ev.add(t + "_after", o["after"], u, "recent period"); ev.add(t + "_change", o["change"], u, "change", "derived")
        if o.get("pct") is not None: ev.add(t + "_pct", o["pct"], "%", "relative change", "derived")
        if o.get("p") is not None: ev.add("p_value", o["p"], "", "p-value", "derived"); ev.add("effect_size", o["effect_size"], "", "effect size", "derived")
        if o.get("start_p_adj") is not None: ev.add("start_p_adj", o["start_p_adj"], "", "adjusted p-value of the start", "derived")
        for k, v in (("range_before", o.get("range_before")), ("range_after", o.get("range_after"))):
            if v is not None: ev.add(k, v, "km", "estimated range", "derived")
        ev.add("days_before", o["days"][0], "count", "days before", "count"); ev.add("days_after", o["days"][1], "count", "days after", "count")
        cb = x.get("contrib")
        if cb:
            ev.add("trip_before", cb["trip_before"], "Wh/km", "distance-weighted trip mean before"); ev.add("trip_after", cb["trip_after"], "Wh/km", "distance-weighted trip mean after"); ev.add("gap", cb["gap"], "Wh/km", "change in consumption", "derived"); ev.add("explained", cb["explained"], "Wh/km", "explained by recorded conditions", "derived"); ev.add("residual", cb["residual"], "Wh/km", "not explained", "derived"); ev.add("r2", cb["r2"], "", "model fit", "derived")
            ev.add("trips_before", cb["n_trips"][0], "count", "trips before", "count"); ev.add("trips_after", cb["n_trips"][1], "count", "trips after", "count"); ev.add("trips_total", cb["n_trips"][0] + cb["n_trips"][1], "count", "trips used", "count")
            if abs(cb["gap"]) > 1e-9: ev.add("explained_pct", 100 * cb["explained"] / cb["gap"], "%", "share explained", "derived"); ev.add("residual_pct", abs(100 - 100 * cb["explained"] / cb["gap"]), "%", "share not explained", "derived"); ev.add("residual_pct_of_gap", abs(100 * cb["residual"] / cb["gap"]), "%", "residual share", "derived")
            for i, f in enumerate(cb["factors"], 1):
                ev.add(f"factor_{i}", f["contribution"], "Wh/km", f["name"], "derived"); ev.add(f"factor_{i}_share", f["share_pct"], "%", f["name"] + " share", "derived"); ev.add(f"factor_{i}_before", f["before"], f["unit"], f["name"] + " before"); ev.add(f"factor_{i}_after", f["after"], f["unit"], f["name"] + " after")
        for a in x.get("assoc", []):
            d = a["driver"]; ev.add(d + "_before", a["before"], a["unit"], a["label"] + " before"); ev.add(d + "_after", a["after"], a["unit"], a["label"] + " after"); ev.add(d + "_change", a["change"], a["unit"], a["label"] + " change", "derived"); ev.add(d + "_p", a["p"], "", a["label"] + " p-value", "derived")
        if o.get("normalised"): n = o["normalised"]; ev.add("normalised_before", n["before"], "kPa", "pressure at 20 C before"); ev.add("normalised_after", n["after"], "kPa", "pressure at 20 C after"); ev.add("normalised_change", n["change"], "kPa", "pressure at 20 C change", "derived")
    if op in ("aggregate",) and res.extra.get("trip_count") is not None: ev.add("trips", res.extra["trip_count"], "count", "trips", "count")
    if res.items and op == "aggregate":
        for n, it in enumerate(res.items[:10], 1):
            if it.get("wh_per_km") is not None: ev.add(f"trip_{n}_wh_per_km", it["wh_per_km"], "Wh/km", f"trip {n}");
            if it.get("km") is not None: ev.add(f"trip_{n}_km", it["km"], "km")
        ev.add("selected", len(res.items), "count", "trips selected", "count")
    return ev


# ------------------------------------------------------------------------------------------------ evidence graphs
def graph_from_context(res, event_label):
    """Event context -> linked evidence: what changed in the signals before the event, in order of onset. Relations are temporal / associative only ('preceded', 'associated with')."""
    g = EvidenceGraph(); ev_id = g.add(Node("event", "event", event_label, res.when))
    items = [s for s in res.items if abs(s["change"]) > 1e-9]
    series = res.extra.get("series", {}); t = series.get("t", [])
    moves = []
    for s in items:
        v = series.get(s["signal"]); rng = (s["max"] - s["min"]) or 1.0
        if not v or not t: continue
        onset = next((tt for tt, x in zip(t, v) if x == x and abs(x - v[0]) > 0.25 * rng), None)
        if abs(s["change"]) > 0.15 * max(abs(s["first"]), abs(s["last"]), 1e-9) or abs(s["change"]) > 0.4 * rng: moves.append((onset if onset is not None else 0, s))
    moves.sort(key=lambda m: m[0]); prev = None
    for onset, s in moves[:5]:
        word = "rose" if s["change"] > 0 else "fell"
        nid = g.add(Node(s["signal"], "signal_change", f"{s['label']} {word} ({s['first']:.1f} to {s['last']:.1f} {s['unit']})", int(res.when + onset) if res.when is not None else None, s["change"], s["unit"]))
        g.link(nid, ev_id, "preceded", "moderate"); prev = nid
    return g


def graph_from_diagnosis(diag, event_label, ts):
    """Legacy diagnosis (ranked explanations) -> graph with 'likely_contributor' edges and the strength the engine measured."""
    g = EvidenceGraph(); eid = g.add(Node("event", "event", event_label, ts))
    for i, a in enumerate((diag.get("alts") or [])[:3]):
        nid = g.add(Node(f"cause{i}", "state", a["name"].split(" (")[0])); strength = {"strong": "strong", "moderate": "moderate", "weak": "weak"}.get(a.get("strength"), "weak")
        g.link(nid, eid, "likely_contributor" if strength != "weak" else "associated_with", strength, a.get("support", ""))
    return g
