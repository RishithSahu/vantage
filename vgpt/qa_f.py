"""QA part F: the pre-drive briefing ("Brief me", "Good morning", "Is it OK to drive?") and the evidence chart behind any event answer.
The briefing is deterministic: it reads the same health / range / trip / tyre tables as the other answers, ends in one go / take-care / service verdict and gives a
short plain suggestion. It never tells the driver to override anything - it only reports and advises."""
import datetime as dt
from . import analytics as A, analytics_x as AX, qa_core as Q

SUGGEST = {"cooling": "Avoid fast charging and long motorway runs until the cooling system is checked.", "battery": "Keep loads gentle and have the battery checked at the next service.",
           "tyres": "Check the tyre pressures before setting off.", "tpms": "Check the tyre pressures before setting off.", "brakes": "Book a brake check soon.", "electrical": "Have the 12 V battery tested.",
           "camera": "Clean or check the front camera before relying on driver assistance.", "service": "Book the service that is due.", "motor": "Go easy on acceleration until the motor temperature is checked.",
           "diagnostics": "Have the stored trouble code read at a workshop."}


class MixF:
    def part_of_day(self):
        try: h = dt.datetime.fromtimestamp(self.now + A.TZ, dt.timezone.utc).hour
        except Exception: return "Hello"
        return "Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 18 else "Good evening"

    def h_briefing(self, s, raw):
        h = AX.health(self.c, self.now, "14") if True else {}
        items = h.get("items", []) if h.get("ok") else []
        bad = [i for i in items if i["level"] == "problem"]; watch = [i for i in items if i["level"] == "attention"]
        er = A.est_range(self.c, self.now); lines = []
        if er.get("ok"): lines.append(f"Battery {er['soc']:.0f}%, about {er['range_km']} km of range at your recent {er['wh_km']:.0f} Wh/km.")
        t = self.c.execute("SELECT start, end, km, wh_km FROM trips WHERE end<=? ORDER BY end DESC LIMIT 1", (self.now,)).fetchone()
        if t: lines.append(f"Last trip: {t[2]:.1f} km on {self.day(t[0])}, {t[3]:.0f} Wh/km.")
        for i in bad[:3]: lines.append(f"Needs attention - {i['text']}.")
        for i in watch[:2]: lines.append(f"Keep an eye on - {i['text']}.")
        if not bad and not watch: lines.append("No warnings or unusual readings in the last 14 days.")
        verdict = "Service advised before any long drive" if len({i["area"] for i in bad}) >= 2 else "Drive with care" if bad else "OK to drive" if not watch else "OK to drive, with one thing to watch" if len(watch) == 1 else "OK to drive, with a few things to watch"
        why = [SUGGEST.get(i["area"], SUGGEST.get(i["key"], "")) for i in bad + watch]; sug = next((w for w in why if w), "")
        if sug: lines.append("Suggestion: " + sug)
        head = f"{self.part_of_day()}. {verdict}."
        self.brief = head + (f" {lines[0]}" if er.get("ok") else "")
        self.ctx.update(topic="briefing")
        self.out_facts = {"verdict": verdict, "problems": len(bad), "watch": len(watch)}
        return head + "\n" + "\n".join("  - " + l for l in lines) + "\nThis is a read-only summary of the recorded data; it does not change any setting."


def chart(c, ts, before=900, after=180, points=140):
    """Signals around an event for the evidence chart: a few aligned series (offsets in seconds from the event) and marks. Reads the frozen event context, so it also works for old events."""
    sg = Q.event_sig(c, ts, before, after)
    if sg is None or len(sg) < 10: return {"ok": False, "why": "no signal data around that moment"}
    step = max(1, len(sg) // points); idx = list(range(0, len(sg), step)); t = [sg.ts[i] - ts for i in idx]; out = []
    for key, label, unit in (("batt", "Battery temperature", "\u00b0C"), ("flow", "Coolant flow", "L/min"), ("power", "Battery power", "kW"), ("speed", "Speed", "km/h")):
        v = sg.cols.get(key)
        if not v: continue
        vals = [round(v[i], 1) for i in idx]
        if max(vals) - min(vals) < 1e-6: continue
        out.append({"key": key, "label": label, "unit": unit, "v": vals})
    marks = [{"t": 0, "label": "event"}]
    try:
        fb = Q.flow_baseline(c, ts); fl = sg.cols.get("flow")
        if fb and fl:
            k = next((i for i in range(10, len(fl)) if 0.3 < fl[i] < 0.75 * fb["p50"] and all(0.3 < x < 0.8 * fb["p50"] for x in fl[i:i + 10])), None)
            if k is not None and sg.ts[k] < ts: marks.append({"t": sg.ts[k] - ts, "label": "coolant flow falls below 75% of normal"})
    except Exception: pass
    return {"ok": True, "ts": ts, "t": t, "series": out, "marks": marks, "res": sg.res}
