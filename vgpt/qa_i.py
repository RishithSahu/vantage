"""QA part I: judgement questions ("is it advisable to go on a long trip in this condition"), "why is <signal> negative / zero / high", and safe-driving tips.
Everything is read from the stored data; the car's own sign conventions and physics are explained in a sentence, then backed with the car's numbers."""
import re
from . import analytics as A, analytics_x as AX, qa_core as Q
from .qa_core import mean

DAY = 86400
LONG = re.compile(r"long|far|road ?trip|highway|motorway|airport|journey|distance|out of town|intercity")

SIGNALS = {   # key -> (words, plain explanation, column)
    "power": (r"(battery )?(power|current|kw|kilowatt)", "power", "In this car's data, positive battery power means the battery is giving energy (driving) and negative means energy is flowing into it: charging, or regenerative braking when you slow down."),
    "flow": (r"(coolant|cooling)( pump)?( flow)?|pump", "flow", "Coolant only circulates while the pump runs, which is when the car is driving or charging. Parked and switched off, the flow reads zero. Low flow while the pack is hot is the sign of a cooling problem."),
    "motor": (r"motor( temp\w*)?", "motor", "The drive motor warms up with the power it delivers (hard acceleration, hills, motorway speeds) and cools when you ease off; its temperature follows the load with a delay."),
    "batt": (r"(battery|pack)( temp\w*)?|temperature", "batt", "The pack warms up with the current going in or out of it - fast charging and hard driving especially - and with a hot day; the cooling system is meant to hold it down."),
    "speed": (r"speed|velocity", "speed", "Speed reads zero whenever the wheels are not turning - parked, stopped at a light, or charging."),
    "soc": (r"state of charge|soc|charge level|battery level", "soc", "State of charge falls as you drive and rises while you charge or brake with regeneration."),
    "amb": (r"outside|ambient|air temp\w*", "amb", "That is the air temperature outside, measured by the car, so it follows the weather and time of day."),
    "hvac": (r"(climate|hvac|heater|heating|a/?c|air con\w*)( power)?", "hvac", "Climate control draws power for heating or cooling the cabin; it is higher in cold or hot weather."),
}
STATE = {"negative": r"negative|minus|below zero|less than zero|−", "zero": r"zero|0\b|nothing|not (moving|flowing|showing|reading)|nil|empty", "high": r"high|rising|going up|climbing|increas\w+|hot|jump\w*|spik\w+|up\b", "low": r"low|dropping|falling|going down|decreas\w+|cold|down\b"}


class MixI:
    def h_advice(self, s, raw):
        h = AX.health(self.c, self.now, "14"); items = h.get("items", []) if h.get("ok") else []
        bad = [i for i in items if i["level"] == "problem"]; watch = [i for i in items if i["level"] == "attention"]
        er = A.est_range(self.c, self.now); long_trip = bool(LONG.search(s))
        cool = [i for i in bad if i["area"] in ("cooling", "battery", "motor")]; tyre = [i for i in bad if i["area"] in ("tyres", "tpms")]; areas = sorted({i["area"] for i in bad})
        soc_note = f" The battery is at {er['soc']:.0f}% (about {er['range_km']} km)." if er.get("ok") else ""
        low = er.get("ok") and er["soc"] < 25
        if not bad:
            if low: self.brief = f"Mechanically nothing is wrong, but charge first: the battery is only at {er['soc']:.0f}% (about {er['range_km']} km)."
            else: self.brief = "Yes - nothing in the data argues against it: no active warnings" + (f"{', and ' + str(len(watch)) + ' thing to keep an eye on' if len(watch) == 1 else ''}" if watch else "") + "." + soc_note
            return self.brief + "\nI only see what the car records - if a red warning light is on, or you notice a burning smell, smoke or odd noises, stop safely and call for help."
        top = re.sub(r" [(].*", "", bad[0]["text"])
        if tyre and not cool: verdict = "Fix the tyre pressure first, then it is fine to go."
        elif long_trip and cool: verdict = "I'd avoid a long trip until the cooling problem is checked; short, gentle local drives are okay."
        elif cool: verdict = "Short, gentle drives are okay, but avoid fast charging and motorway runs until the cooling is checked."
        else: verdict = "Have it checked soon; drive gently until then."
        self.brief = verdict
        lines = [f"  - {re.sub(r' [(].*', '', i['text'])}" for i in bad[:4]]
        head = f"{verdict} The data shows {len(bad)} problem{'s' if len(bad) != 1 else ''} ({' and '.join(areas)})." + (f" Battery {er['soc']:.0f}%, about {er['range_km']} km." if er.get("ok") and long_trip else "")
        return "\n".join([head, "What I see:"] + lines + ["I can't judge whether it is safe - that needs a workshop. If a red warning light is on, or you notice a burning smell, smoke or odd noises, stop safely and call for help."])

    def h_signal_why(self, s, raw):
        key = None
        for k, (rx, col, expl) in SIGNALS.items():
            if re.search(rx, s): key = k; break
        if key is None: return None
        state = next((k for k, rx in STATE.items() if re.search(rx, s)), None)
        rx, col, expl = SIGNALS[key]; cv = self.cov(); lo = cv["hi"] - 7 * DAY
        sg = Q.load(self.c, lo, cv["hi"] + 1, prefer="1m")
        if not len(sg): return expl
        vals = sg.cols.get(col) or []; n = len(vals)
        if not n: return expl
        extra = ""
        if col == "power":
            neg = 100 * sum(1 for v in vals if v < -0.3) / n; chg = 100 * sum(1 for v, m in zip(vals, sg.cols["mode"]) if v < -0.3 and m == 3) / max(1, sum(1 for v in vals if v < -0.3))
            extra = f" In the last 7 days it was negative {neg:.0f}% of the time, {chg:.0f}% of that while charging and the rest while slowing down."
        elif col == "flow":
            z = 100 * sum(1 for v, m in zip(vals, sg.cols["mode"]) if v < 0.3 and m == 0) / max(1, sum(1 for m in sg.cols["mode"] if m == 0)); r = sum(1 for v, m in zip(vals, sg.cols["mode"]) if m in (1, 2, 3))
            run = [v for v, m in zip(vals, sg.cols["mode"]) if m in (1, 2, 3) and v > 0.3]; fb = Q.flow_baseline(self.c, self.now)
            extra = f" When the car is parked it reads zero {z:.0f}% of the time; while running it averages {mean(run):.1f} L/min" + (f" against {fb['p50']:.1f} normally." if fb else ".") if run else ""
        elif col == "speed":
            z = 100 * sum(1 for v in vals if v < 1) / n; extra = f" Over the last 7 days the speed was zero {z:.0f}% of the time (parked, stopped or charging)."
        elif col in ("batt", "motor"):
            ch = [v for v, m in zip(vals, sg.cols["mode"]) if m == 3]; dr = [v for v, m in zip(vals, sg.cols["mode"]) if m in (1, 2)]
            if ch and dr: extra = f" On average it is {mean(ch):.0f} C while charging and {mean(dr):.0f} C while driving (last 7 days; peak {max(vals):.0f} C)."
        self.brief = expl + extra
        if state is None and col == "power" and not re.search(r"negative|minus", s): self.brief = expl
        return self.brief

    def h_drive_safely(self, s, raw):
        tips = []
        st = self._style(self.now - 14 * DAY, self.now + 1) if hasattr(self, "_style") else None
        if st and st["brake"] > 5: tips.append(f"Look further ahead and brake earlier: you brake hard {st['brake']:.0f} times per 100 km (typical is 2-6).")
        if st and st["accel"] > 3: tips.append(f"Accelerate more smoothly: {st['accel']:.0f} hard accelerations per 100 km (typical 1-3).")
        sg = Q.load(self.c, self.now - 14 * DAY, self.now + 1, prefer="1m"); mv = [v for v, m in zip(sg.cols["speed"], sg.cols["mode"]) if m in (1, 2) and v > 5] if len(sg) else []
        if mv and 100 * sum(1 for v in mv if v > 110) / len(mv) > 10: tips.append(f"Keep a cushion at speed: {100 * sum(1 for v in mv if v > 110) / len(mv):.0f}% of your moving time is above 110 km/h, where stopping distances grow quickly.")
        try:
            ty = AX.tyres(self.c, self.now, "14"); low = [w["name"] for w in ty["wheels"] if w["status"] != "ok"] if ty.get("ok") else []
            if low: tips.append(f"Fix the {', '.join(low)} tyre pressure - it affects braking and grip.")
        except Exception: pass
        h = AX.health(self.c, self.now, "14"); bad = [i for i in (h.get("items", []) if h.get("ok") else []) if i["level"] == "problem"]
        if bad: tips.append(f"Get the {bad[0]['area']} problem checked ({re.sub(r' [(].*', '', bad[0]['text'])}).")
        tips += ["Use lane assist as a help, not a replacement: keep your hands on the wheel and your eyes on the road.", "Leave more room in rain or cold - the car's own rain and temperature readings are in its history."]
        self.brief = tips[0] if tips else "Drive smoothly."
        if len(tips) > 1: self.brief += " More tips below."
        return "\n".join(["Safe-driving tips from your own driving data:"] + [f"  - {t}" for t in tips[:5]] + ["General advice, not a substitute for the owner's manual or a driving instructor."])
