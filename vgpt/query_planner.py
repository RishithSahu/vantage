"""Trusted query planner: turns a VALIDATED Query into an explicit, inspectable Plan and compiles the only SQL this system ever runs.

    The model (or the rule parser) never writes SQL. This module is the only place SQL text exists, and it is built from
      (1) templates fixed in this file,
      (2) table / column names taken from the registry whitelist (signal_registry.TABLE_COLUMNS, ENTITY_METRICS, TRIP_CLASSES),
      (3) values passed as bound parameters (`?`) - never concatenated.
    Every statement is SELECT-only; the executor additionally opens the database read-only with PRAGMA query_only=ON and re-checks each statement with `assert_read_only`.

A Plan lists the steps in order (resolve time -> select entities -> load frame -> derive -> filter -> reduce) with the chosen data tier and the compiled SQL, so a log line (or a test) can show exactly what will
be computed and from which tables."""
import datetime as dt
import re
from dataclasses import dataclass, field

from . import signal_registry as R
from .query_ir import Query, TimeSpec

DAY = 86400
SELECT_ONLY = re.compile(r"^\s*SELECT\b", re.I)
FORBIDDEN = re.compile(r";|--|/\*|\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|REINDEX|LOAD_EXTENSION|READFILE|WRITEFILE|EDIT|FTS3_TOKENIZER|SQLITE_\w+|ZEROBLOB|RANDOMBLOB)\b", re.I)


class PlanError(ValueError):
    pass


def assert_read_only(sql):
    if not SELECT_ONLY.match(sql) or FORBIDDEN.search(sql): raise PlanError(f"refusing to run non-read-only SQL: {sql[:60]}")


@dataclass
class Ctx:
    """What the planner needs to know about 'now' and the data: end of the history, the local time zone, where the 1 Hz tier starts."""
    now: int
    tz: int = 0
    lo: int = 0                       # first second of any data
    hi: int = 0                       # last second of any data
    hz_lo: int = None                 # first second of the 1 Hz tier (None = no 1 Hz data)
    hz_hi: int = None
    arch_lo: int = None               # first second the archive of expired seconds holds (vgpt/archive.py); None = no archive
    arch_hi: int = None


@dataclass
class Plan:
    query: Query
    tier: str = "1m"
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def explain(self): return [f"{i + 1}. {s['kind']}" + (f"  [{s['sql']}]" if s.get("sql") else "") for i, s in enumerate(self.steps)]


# ------------------------------------------------------------------------------------------------ time
def local_midnight(ts, tz): return ((ts + tz) // DAY) * DAY - tz


def _month_start(ts, tz, back=0):
    d = dt.datetime.fromtimestamp(ts + tz, dt.timezone.utc); y, m = d.year, d.month - back
    while m <= 0: m += 12; y -= 1
    return int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp()) - tz


def resolve_time(t, ctx):
    """TimeSpec -> (lo, hi, label) in epoch seconds [lo, hi). Never reads the clock: 'now' is the end of the recorded history, so the same question gives the same answer on the same data."""
    now = ctx.now; tz = ctx.tz; today = local_midnight(now - 1 if (now + tz) % DAY == 0 else now, tz)     # a history that ends exactly at midnight ends the day before: "today" is that day, not the empty day that has not begun
    if t.type == "range":                                                                                # exact interval, set by the executor itself (never accepted from text)
        lo, hi = int(t.start), int(t.end)
        if (lo + tz) % DAY == 0 and (hi + tz) % DAY == 0 and hi > lo:                                      # whole local days: say the dates
            f = lambda x: dt.datetime.fromtimestamp(x + tz, dt.timezone.utc).strftime("%d %b").lstrip("0")
            return lo, hi, (f"on {f(lo)}" if hi - lo == DAY else f"from {f(lo)} to {f(hi - DAY)}")
        return lo, hi, "the selected interval"
    if t.type == "current": return now - 300, now + 1, "right now"
    if t.type == "between":
        if t.unit == "day":
            lo = today - t.value * DAY; hi = (today - (t.value2 - 1) * DAY) if t.value2 else now + 1
            if t.value >= 14:                                                                                   # far back, dates read better than a count of days
                f = lambda x: dt.datetime.fromtimestamp(x + tz, dt.timezone.utc).strftime("%d %b %Y").lstrip("0")
                return lo, hi, (f"from {f(lo)} to {f(hi - 1)}" if t.value2 else f"since {f(lo)}")
            return lo, hi, (f"between {t.value} and {t.value2} days ago" if t.value2 else f"since {t.value} day{'s' if t.value != 1 else ''} ago")
        u = 3600 if t.unit == "hour" else 60; lo, hi = now - t.value * u, ((now - t.value2 * u) if t.value2 else now + 1)
        f = lambda x: dt.datetime.fromtimestamp(x + tz, dt.timezone.utc).strftime("%d %b %H:%M").lstrip("0")
        return lo, hi, (f"from {f(lo)} to {f(hi)}" if t.value >= 180 else f"between {t.value} and {t.value2} {t.unit}s ago")
    if t.type == "all": return ctx.lo, ctx.hi + 1, "the whole recorded history"
    if t.type == "relative" and t.unit == "month" and t.value >= 2:                                    # "the last 12 months" is twelve calendar months back from now, not 360 days
        d = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc); mi = d.year * 12 + d.month - 1 - t.value; y, m0 = divmod(mi, 12)
        lo = int(d.replace(year=y, month=m0 + 1, day=min(d.day, 28)).timestamp()) - tz
        return lo, now + 1, f"the last {t.value} months"
    if t.type == "relative":
        unit = {"minute": 60, "hour": 3600, "day": DAY, "week": 7 * DAY, "month": 30 * DAY}[t.unit]
        return now - t.value * unit, now + 1, f"the last {t.value} {t.unit}{'s' if t.value != 1 else ''}"
    if t.type == "previous":
        unit = {"minute": 60, "hour": 3600, "day": DAY, "week": 7 * DAY, "month": 30 * DAY}[t.unit]
        return now - 2 * t.value * unit, now - t.value * unit, f"the {t.value} {t.unit}{'s' if t.value != 1 else ''} before that"
    if t.type == "date":
        d = dt.date.fromisoformat(t.date); lo = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()) - tz
        return lo, lo + DAY, f"on {t.date}"
    n = t.name
    if n[0] in "qhsf" and n[1:2].isdigit():                                                           # a quarter, a half, a run of months, "since" a month
        def mstart(y, m): return int(dt.datetime(y + (m - 1) // 12, (m - 1) % 12 + 1, 1, tzinfo=dt.timezone.utc).timestamp()) - tz
        P = n[1:].split("_")
        if n[0] == "q": y, qn = int(P[0]), int(P[1]); lo, hi = mstart(y, 3 * qn - 2), mstart(y, 3 * qn + 1); label = f"in the {('first', 'second', 'third', 'fourth')[qn - 1]} quarter of {y}"
        elif n[0] == "h": y, hn = int(P[0]), int(P[1]); lo, hi = mstart(y, 6 * hn - 5), mstart(y, 6 * hn + 1); label = f"in the {('first', 'second')[hn - 1]} half of {y}"
        elif n[0] == "f": y, m0 = int(P[0]), int(P[1]); lo, hi = mstart(y, m0), now + 1; label = "since " + dt.datetime(y, m0, 1).strftime("%B %Y")
        else:
            y1, m1, y2, m2 = int(P[0]), int(P[1]), int(P[2]), int(P[3]); lo, hi = mstart(y1, m1), mstart(y2, m2 + 1)
            label = (dt.datetime(y1, m1, 1).strftime("from %B %Y") + dt.datetime(y2, m2, 1).strftime(" to %B %Y")) if (y1, m1) != (y2, m2) else dt.datetime(y1, m1, 1).strftime("in %B %Y")
        return lo, min(hi, now + 1), label
    if n[0] in "ym" and n[1:2].isdigit() or n in ("last_year", "this_year"):                              # a calendar year or month in local time
        ny = dt.datetime.fromtimestamp(now + tz, dt.timezone.utc).year
        if n in ("last_year", "this_year"): y = ny - (1 if n == "last_year" else 0); m0 = None
        elif n[0] == "y": y = int(n[1:]); m0 = None
        else: y, m0 = int(n[1:5]), int(n[6:8])
        a = dt.datetime(y, m0 or 1, 1, tzinfo=dt.timezone.utc); b = dt.datetime(y + (1 if (m0 or 12) == 12 else 0), (m0 % 12 + 1) if m0 else 1, 1, tzinfo=dt.timezone.utc)
        lo, hi = int(a.timestamp()) - tz, int(b.timestamp()) - tz
        return lo, min(hi, now + 1), (a.strftime("in %B %Y") if m0 else f"in {y}")
    wd = ((today + tz) // DAY + 3) % 7                                   # Monday = 0
    if n == "today": return today, now + 1, "today"
    if n == "yesterday": return today - DAY, today, "yesterday"
    if n == "day_before_yesterday": return today - 2 * DAY, today - DAY, "the day before yesterday"
    if n == "this_week": return today - wd * DAY, now + 1, "this week"
    if n == "last_week": return today - (wd + 7) * DAY, today - wd * DAY, "last week"
    if n == "this_month": return _month_start(now, tz), now + 1, "this month"
    if n == "last_month": return _month_start(now, tz, 1), _month_start(now, tz), "last month"
    if n == "last_weekend":
        sat = today - ((wd - 5) % 7) * DAY if wd != 5 else today
        if wd in (5, 6): sat = today - (wd - 5 + 7) * DAY                  # the weekend before the one in progress
        return sat, sat + 2 * DAY, "last weekend"
    if n == "this_morning": return today, min(today + 12 * 3600, now + 1), "this morning"
    if n == "last_night": return today - 6 * 3600, today + 6 * 3600, "last night"
    raise PlanError(f"cannot resolve time '{n}'")


# ------------------------------------------------------------------------------------------------ SQL compilation (the ONLY place SQL text is made)
def _col(table, col):
    if col not in R.TABLE_COLUMNS.get(table, ()): raise PlanError(f"column {table}.{col} is not in the whitelist")
    return col


def compile_select(table, columns, lo, hi, tier_key="ts"):
    """SELECT ts, <whitelisted columns> FROM <whitelisted table> WHERE ts in [lo, hi) ORDER BY ts."""
    if table not in R.TABLE_COLUMNS: raise PlanError(f"table {table} is not in the whitelist")
    cols = ", ".join(_col(table, c) for c in columns)
    sql = f"SELECT {tier_key}, {cols} FROM {table} WHERE {tier_key}>=? AND {tier_key}<? ORDER BY {tier_key}"
    assert_read_only(sql); return sql, (int(lo), int(hi))


ENTITY_COLUMNS = {"trip": ("t.start", "t.end", "t.km", "t.kwh", "t.wh_km", "t.avg_speed", "t.hwy_share", "t.avg_amb", "t.avg_hvac_w", "t.soc_start", "t.soc_end"), "charge_session": ("t.start", "t.end", "t.kind", "t.kw", "t.kwh", "t.soc0", "t.soc1")}
_CLASS_COL = {"hwy_share": "t.hwy_share", "km": "t.km", "rain_share": "x.rain_share"}


def compile_entities(entity, klass, rank_by, order, k, lo, hi, with_ext=False):
    """Trips (or charge sessions) starting in [lo, hi), optionally of a class, ranked by an entity metric, limited to k."""
    if entity not in R.ENTITY_TABLE: raise PlanError(f"unknown entity {entity}")
    if klass is not None and (entity != "trip" or klass not in R.TRIP_CLASSES): raise PlanError(f"unknown class {klass}")
    if rank_by is not None and rank_by not in R.ENTITY_METRICS[entity]: raise PlanError(f"cannot rank {entity}s by {rank_by}")
    table, startcol, endcol = R.ENTITY_TABLE[entity]
    cols = ", ".join(ENTITY_COLUMNS[entity])
    join = " LEFT JOIN trip_ext x ON x.start=t.start" if (klass == "rainy" and entity == "trip") else ""
    where = ["t.start>=?", "t.start<?"]; args = [int(lo), int(hi)]
    if klass:
        _, col, op, val = R.TRIP_CLASSES[klass]
        if op not in (">=", "<"): raise PlanError("bad class operator")
        where.append(f"{_CLASS_COL[col]} {op} ?"); args.append(float(val))
    if rank_by:
        mcol = R.ENTITY_METRICS[entity][rank_by][0]
        if not re.match(r"^(\(end-start\)|[a-z_]+)$", mcol): raise PlanError("bad metric column")
        mcol = "(t.end-t.start)" if mcol == "(end-start)" else f"t.{mcol}"
        # NULL is unknown, not a smallest possible value.  Filtering at the
        # selection boundary prevents a top-k query from calling an
        # unmeasured trip the most efficient one.
        where.append(f"{mcol} IS NOT NULL")
    sql = f"SELECT {cols} FROM {table} t{join} WHERE {' AND '.join(where)}"
    if rank_by:
        sql += f" ORDER BY {mcol} {'DESC' if order == 'desc' else 'ASC'}, t.start"
    else: sql += " ORDER BY t.start"
    if k: sql += " LIMIT ?"; args.append(int(k))
    assert_read_only(sql); return sql, tuple(args)


def compile_events(types, lo, hi, order="desc", limit=None):
    known = []
    for t in types:
        known += list(R.EVENT_GROUPS.get(t, (t,)))
    for t in known:
        if t not in R.EVENTS: raise PlanError(f"unknown event {t}")
    sql = f"SELECT ts, type, sev, detail FROM events WHERE type IN ({','.join('?' * len(known))}) AND ts>=? AND ts<? ORDER BY ts {'DESC' if order == 'desc' else 'ASC'}"
    args = list(known) + [int(lo), int(hi)]
    if limit: sql += " LIMIT ?"; args.append(int(limit))
    assert_read_only(sql); return sql, tuple(args)


# ------------------------------------------------------------------------------------------------ planning
def measured_needs(concept, profile, seen=None):
    """The MEASURED concepts that have to be loaded to obtain `concept` (itself if measured, its dependencies if derived)."""
    seen = seen if seen is not None else set()
    if concept in seen: return set()
    seen.add(concept); res = profile.resolve(concept)
    if res.status == "measured": return {concept}
    out = set()
    for sub in res.sub.values(): out |= measured_needs(sub.concept, profile, seen)
    if res.status in ("derived", "estimated") and not res.sub:
        for r in res.requires: out |= measured_needs(r, profile, seen)
    return out


EXACT_OPS = {"extreme", "percent_time", "duration", "threshold_test", "episodes", "continuous", "longest_run", "percentile", "pair_gap", "acceleration_band"}      # answers that are only exact over the seconds themselves
ARCH_MAX_SPAN = 45 * 86400                                                                                                                                       # how much of the archive one question may load


def choose_tier(concepts_needed, lo, hi, ctx, profile, prefer_fine=False, exact=False):
    """1 Hz data lives in the database for the last ~7 days and is heavy: use it for short windows (or when asked for sub-minute context), the per-minute tier for everything else - except that an answer which is only
    exact over the seconds (a highest value, a duration above a limit) reads the archive of expired seconds when it covers the window (it is loaded for the question and shadows the live table)."""
    span = hi - lo
    if exact and ctx.arch_lo is not None and lo >= ctx.arch_lo - 1 and span <= ARCH_MAX_SPAN and (ctx.hz_lo is None or lo < ctx.hz_lo - 1):
        if all(("1s" in profile.tiers_of(c)) for c in concepts_needed): return "1s"
    if ctx.hz_lo is not None and lo >= ctx.hz_lo - 1 and (prefer_fine or span <= 6 * 3600):
        if all(("1s" in profile.tiers_of(c)) or ("1m" in profile.tiers_of(c)) for c in concepts_needed): return "1s"
    return "1m"


def plan(q, profile, ctx):
    """Build the Plan for a validated Query. The structure mirrors the execution order; the executor interprets the step kinds."""
    p = Plan(q)
    tgt = [c for c in (q.target, q.target2) if c]
    cond_concepts = []
    from .capability import concepts_of
    for c in concepts_of(q):
        if c not in tgt: cond_concepts.append(c)
    needs = set()
    for c in tgt + cond_concepts: needs |= measured_needs(c, profile)
    lo, hi, label = resolve_time(q.scope.time, ctx)
    fine = q.op in ("current_value", "event_context", "acceleration_band")
    p.tier = choose_tier(needs, lo, hi, ctx, profile, prefer_fine=fine, exact=q.op in EXACT_OPS)
    p.steps.append({"kind": "resolve_time", "scope": "A", "lo": lo, "hi": hi, "label": label})
    if q.scope_b:
        lo2, hi2, label2 = resolve_time(q.scope_b.time, ctx); p.steps.append({"kind": "resolve_time", "scope": "B", "lo": lo2, "hi": hi2, "label": label2})
    for name, sc in (("A", q.scope), ("B", q.scope_b)):
        if sc is None or sc.select is None: continue
        e = sc.select; l, h, _ = resolve_time(sc.time, ctx); sql, args = compile_entities(e.entity, e.klass, e.rank_by, e.order, e.k, l, h)
        p.steps.append({"kind": "select_entities", "scope": name, "entity": e.entity, "sql": re.sub(r"\s+", " ", sql), "args": args})
    if q.event:
        types = list(R.EVENT_GROUPS.get(q.event.type, (q.event.type,)))
        if q.event.type not in R.DETECTED_EVENTS:
            sql, args = compile_events([q.event.type], ctx.lo, ctx.hi + 1, "desc" if q.event.selector in ("last", "nth", "all") else "asc"); p.steps.append({"kind": "find_event", "sql": sql, "args": args, "selector": q.event.selector})
        else: p.steps.append({"kind": "detect_event", "event": q.event.type})
    if needs:
        loads = []
        for c in sorted(needs):
            tiers = profile.tiers_of(c); use = p.tier if p.tier in tiers else ("1m" if "1m" in tiers else next(iter(tiers), None))
            if use: loads.append((c, tiers[use][0], tiers[use][1], use))
        tables = {}
        for c, tb, col, t in loads: tables.setdefault((tb, t), []).append(col)
        for (tb, t), cols in tables.items():
            if tb == "vday": sql, args = compile_select(tb, cols, lo, hi)
            else: sql, args = compile_select(tb, [c for c in cols if c != "tp_*"] or ["ts"], lo, hi) if cols != ["tp_*"] else compile_select(tb, ["tp_fl", "tp_fr", "tp_rl", "tp_rr"], lo, hi)
            p.steps.append({"kind": "load_frame", "table": tb, "tier": t, "columns": cols, "sql": sql, "args": args})
    if q.scope.conditions or q.groups: p.steps.append({"kind": "filter", "conditions": [c.to_text() for c in q.scope.conditions], "groups": list(q.groups)})
    for c in tgt:
        r = profile.resolve(c)
        if r.status in ("derived", "estimated") and c in R.DERIVED_REQUIRES or c in R.DERIVED_ALTERNATIVES and r.status != "measured": p.steps.append({"kind": "derive", "metric": c, "via": list(r.requires), "quality": r.quality})
    p.steps.append({"kind": "reduce", "op": q.op, "agg": q.agg, "target": q.target})
    return p
