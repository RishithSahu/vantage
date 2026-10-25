"""Historical questions over the 3-year world (data/world_3yr.db): the assistant's answer against the number the database itself gives.

    python bench/history_3yr.py [--db data/world_3yr.db] [--json out.json]

Each case is (question, expected numbers computed by SQL from the same file, tolerance).  A case passes when one of the numbers in the answer text equals an expected number within the tolerance (counts: exactly).
Periods that lie before the one-year minute tier were deliberately written (tools_merge_years.py joins three one-year builds), so every question here is answerable from what the file holds; a failure is a failure of
the assistant to find or compute it, never of the data."""
import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def num_list(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text.replace(" ", " ")) if x.strip(",")]


class Data:
    def __init__(self, db):
        self.c = sqlite3.connect(f"file:{db}?mode=ro", uri=True); self.tz = int(float(self.c.execute("SELECT v FROM meta WHERE k='env.tz_h'").fetchone()[0]) * 3600); self.end = int(self.c.execute("SELECT v FROM meta WHERE k='end'").fetchone()[0])

    def span(self, y, m=None):
        a = dt.datetime(y, m or 1, 1, tzinfo=dt.timezone.utc); b = dt.datetime(y + (1 if (m or 12) == 12 else 0), (m % 12 + 1) if m else 1, 1, tzinfo=dt.timezone.utc)
        return int(a.timestamp()) - self.tz, int(b.timestamp()) - self.tz

    def q(self, sql, *a): return self.c.execute(sql, a).fetchone()[0]
    def trips(self, lo, hi): return self.q("SELECT COUNT(*) FROM trips WHERE start>=? AND start<?", lo, hi)
    def km(self, lo, hi): return self.q("SELECT COALESCE(SUM(km),0) FROM trips WHERE start>=? AND start<?", lo, hi)
    def longest(self, lo, hi): return self.q("SELECT MAX(km) FROM trips WHERE start>=? AND start<?", lo, hi)
    def ev(self, ty, lo, hi): return self.q("SELECT COUNT(*) FROM events WHERE type=? AND ts>=? AND ts<?", ty, lo, hi)
    def charges(self, lo, hi): return self.q("SELECT COUNT(*) FROM charges WHERE start>=? AND start<?", lo, hi)
    def kwh(self, lo, hi): return self.q("SELECT COALESCE(SUM(kwh),0) FROM charges WHERE start>=? AND start<?", lo, hi)
    def amb(self, lo, hi): return self.q("SELECT AVG(amb)/10.0 FROM sig_1m WHERE ts>=? AND ts<? AND amb IS NOT NULL", lo, hi)
    def months_ago(self, n):
        d = dt.datetime.fromtimestamp(self.end + self.tz, dt.timezone.utc); mi = d.year * 12 + d.month - 1 - n; y, m = divmod(mi, 12)
        return int(d.replace(year=y, month=m + 1, day=min(d.day, 28)).timestamp()) - self.tz

    def first(self, table, col, where="1=1", args=(), desc=False):
        return self.q(f"SELECT {col} FROM {table} WHERE {where} ORDER BY {col} {'DESC' if desc else 'ASC'} LIMIT 1", *args)

    def day(self, ts): return (lambda d: f"{d.day} {d.strftime('%b')}")(dt.datetime.fromtimestamp(ts + self.tz, dt.timezone.utc))
    def maxspeed(self, lo, hi): return self.q("SELECT MAX(speed)/10.0 FROM sig_1m WHERE ts>=? AND ts<?", lo, hi)
    def minsoc(self, lo, hi): return self.q("SELECT MIN(soc)/10.0 FROM sig_1m WHERE ts>=? AND ts<? AND soc IS NOT NULL", lo, hi)
    def whkm(self, lo, hi): return self.q("SELECT 1000.0*SUM(kwh)/SUM(km) FROM trips WHERE start>=? AND start<? AND km>0", lo, hi)


def cases(D):
    y = {n: D.span(n) for n in (2023, 2024, 2025, 2026)}; m = lambda yy, mm: D.span(yy, mm); whole = (0, 2 ** 62)
    ex = lambda *v: list(v)
    return [
        ("how many trips did I take in 2024", ex(D.trips(*y[2024])), 0),
        ("how many trips did I take in 2025", ex(D.trips(*y[2025])), 0),
        ("how many trips last year", ex(D.trips(*y[2025])), 0),
        ("how many trips in March 2025", ex(D.trips(*m(2025, 3))), 0),
        ("how many trips in total", ex(D.trips(*whole)), 0),
        ("how far did I drive in 2024", ex(D.km(*y[2024])), 1.0),
        ("how far did I drive in 2025", ex(D.km(*y[2025])), 1.0),
        ("total distance driven in July 2025", ex(D.km(*m(2025, 7))), 1.0),
        ("what was my longest trip in 2024", ex(D.longest(*y[2024])), 0.6),
        ("what was my longest trip ever", ex(D.longest(*whole)), 0.6),
        ("how many charging sessions in 2025", ex(D.charges(*y[2025])), 0),
        ("how many times did I charge in 2024", ex(D.charges(*y[2024])), 0),
        ("how much energy did I charge in 2025", ex(D.kwh(*y[2025])), 1.5),
        ("how many hard braking events in 2025", ex(D.ev("HARSH_BRAKE", *y[2025])), 0),
        ("how many harsh accelerations in 2024", ex(D.ev("HARSH_ACCEL", *y[2024])), 0),
        ("how many times did the battery overheat in 2025", ex(D.ev("WARN_BATT_OVERTEMP", *y[2025])), 0),
        ("how many ADAS disengagements in 2024", ex(D.ev("ADAS_DISENGAGE", *y[2024])), 0),
        ("how many diagnostic trouble codes in 2025", ex(D.ev("DTC_SET", *y[2025])), 0),
        ("how many ECU resets in total", ex(D.ev("ECU_RESET", *whole)), 0),
        ("how many unexpected shutdowns have there been", ex(D.ev("UNEXPECTED_SHUTDOWN", *whole)), 0),
        ("average outside temperature in July 2025", ex(D.amb(*m(2025, 7))), 0.6),
        ("average outside temperature in January 2025", ex(D.amb(*m(2025, 1))), 0.6),
        ("average outside temperature in 2024", ex(D.amb(*y[2024])), 0.6),
        ("what was my average energy use per km in 2025", ex(D.whkm(*y[2025])), 2.0),
        ("what was my average energy use per km in 2024", ex(D.whkm(*y[2024])), 2.0),
        ("compare my energy use per km in 2024 and 2025", ex(D.whkm(*y[2024]), D.whkm(*y[2025])), 2.0),
        ("how many trips in the last 12 months", ex(D.trips(D.months_ago(12), D.end + 1)), 0),
        ("how many trips in the last 2 years", ex(D.trips(D.end - 730 * 86400, D.end + 1)), 0),
        ("how far did I drive in the last 6 months", ex(D.km(D.months_ago(6), D.end + 1)), 1.0),
        ("how many hard braking events in total", ex(D.ev("HARSH_BRAKE", *whole)), 0),
        ("what was my highest speed in 2025", ex(D.maxspeed(*y[2025])), 1.0),
        ("what was the lowest battery charge in 2024", ex(D.minsoc(*y[2024])), 1.0),
        ("when was the first charging session", ["text", D.day(D.first("charges", "start"))], 0),
        ("when was the last unexpected shutdown", ["text", D.day(D.first("events", "ts", "type='UNEXPECTED_SHUTDOWN'", (), True))], 0),
        ("when was the first unexpected shutdown", ["text", D.day(D.first("events", "ts", "type='UNEXPECTED_SHUTDOWN'"))], 0),
        ("when did the battery last overheat", ["text", D.day(D.first("events", "ts", "type='WARN_BATT_OVERTEMP'", (), True))], 0),
        ("when was the first diagnostic trouble code", ["text", D.day(D.first("events", "ts", "type='DTC_SET'"))], 0),
    ]



def cases_hard(D):
    """Time conditions of every kind (a run of months, a quarter, a half, a season, 'since', long relative windows, two periods) mixed with a count, a sum, an extreme, a mean, an event or a comparison."""
    def rng(y1, m1, y2, m2): return D.span(y1, m1)[0], D.span(y2, m2)[1]
    q = lambda y, n: rng(y, 3 * n - 2, y, 3 * n); h = lambda y, n: rng(y, 6 * n - 5, y, 6 * n); since = lambda y, m: (D.span(y, m)[0], D.end + 1)
    whole = (0, 2 ** 62); ex = lambda *v: list(v)
    return [
        ("how many trips between january and june 2025", ex(D.trips(*rng(2025, 1, 2025, 6))), 0),
        ("how far did I drive in the second quarter of 2025", ex(D.km(*q(2025, 2))), 1.0),
        ("how many charging sessions in summer 2025", ex(D.charges(*rng(2025, 6, 2025, 8))), 0),
        ("how many hard braking events in the first half of 2024", ex(D.ev("HARSH_BRAKE", *h(2024, 1))), 0),
        ("how many trips since march 2026", ex(D.trips(*since(2026, 3))), 0),
        ("how far did I drive since january 2026", ex(D.km(*since(2026, 1))), 1.0),
        ("what was my highest speed in the first quarter of 2025", ex(D.maxspeed(*q(2025, 1))), 1.0),
        ("average outside temperature in winter 2025", ex(D.amb(*rng(2024, 12, 2025, 2))), 0.6),
        ("what was my average energy use per km between 2024 and 2025", ex(D.whkm(D.span(2024)[0], D.span(2025)[1])), 2.0),
        ("what was my longest trip in summer 2024", ex(D.longest(*rng(2024, 6, 2024, 8))), 0.6),
        ("how many fast charging sessions in 2025", ex(D.q("SELECT COUNT(*) FROM charges WHERE kind='DC' AND start>=? AND start<?", *D.span(2025))), 0),
        ("compare the number of trips in 2024 and 2025", ex(D.trips(*D.span(2024)), D.trips(*D.span(2025))), 0),
        ("compare distance driven in the first half of 2025 and the second half of 2025", ex(D.km(*h(2025, 1)), D.km(*h(2025, 2))), 1.0),
        ("how many unexpected shutdowns in 2024", ex(D.ev("UNEXPECTED_SHUTDOWN", *D.span(2024))), 0),
        ("how many diagnostic trouble codes in march 2025", ex(D.ev("DTC_SET", *D.span(2025, 3))), 0),
        ("how many trips in the last 3 months", ex(D.trips(D.months_ago(3), D.end + 1)), 0),
        ("how far did I drive in the last 18 months", ex(D.km(D.months_ago(18), D.end + 1)), 1.0),
        ("total energy used in the last 6 months", ex(D.q("SELECT COALESCE(SUM(kwh),0) FROM trips WHERE start>=? AND start<?", D.months_ago(6), D.end + 1)), 1.5),
        ("when was the last charging session in 2024", ["text", D.day(D.q("SELECT MAX(start) FROM charges WHERE start>=? AND start<?", *D.span(2024)))], 0),
        ("when was the first diagnostic trouble code in 2025", ["text", D.day(D.q("SELECT MIN(ts) FROM events WHERE type='DTC_SET' AND ts>=? AND ts<?", *D.span(2025)))], 0),
        ("how many adas disengagements in the second half of 2025", ex(D.ev("ADAS_DISENGAGE", *h(2025, 2))), 0),
        ("how many hard accelerations between march and may 2026", ex(D.ev("HARSH_ACCEL", *rng(2026, 3, 2026, 5))), 0),
    ]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=os.path.join(ROOT, "data", "world_3yr.db")); ap.add_argument("--json"); ap.add_argument("--only", default="")
    a = ap.parse_args(); D = Data(a.db)
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import make_brain
    b = make_brain(a.db, now=D.end); out = []; ok = 0; cs = [c for c in (cases(D) + cases_hard(D)) if a.only in c[0]]
    for q, expect, tol in cs:
        t = time.time(); r = b.ask(q); ans = r["answer"]; got = num_list(ans)
        good = (any(e in ans for e in expect[1:]) if expect and expect[0] == 'text' else None)
        if good is None: good = any(abs(g - e) <= tol + (0 if tol else 1e-9) or (tol == 0 and int(round(g)) == int(round(e)) and abs(g - e) < 0.5) for g in got for e in expect if isinstance(e, (int, float)))
        ok += good; out.append({"q": q, "expect": expect, "ok": good, "answer": ans[:600], "route": (r.get("facts") or {}).get("route"), "read": r.get("understood"), "s": round(time.time() - t, 2)})
        print(("PASS " if good else "FAIL ") + q, "| expect", [e if isinstance(e, str) else None if e is None else round(e, 2) for e in expect], "| got", ans.replace("\n", " ")[:170])
    print(f"{ok}/{len(cs)} passed")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0 if ok == len(cs) else 1


if __name__ == "__main__":
    sys.exit(main())
