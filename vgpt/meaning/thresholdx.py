"""A limit that is a property of this vehicle, not a number the person typed: "when did my tyre pressure last fall below the recommended pressure".

"Recommended", "specified", "the threshold" name the vehicle's own specification (the tyre placard value in its profile), and the pressure is judged against it the way the tyre answers do: corrected to 20 degrees C, and a tyre is low when it is more than 8 %
under its specification.  The answer is read from the stored tyre pressures (not from the warning events, which are only one way the car noticed), at the last minute any wheel was under the limit."""
import re

from .. import analytics_x as AX
from ..plan_exec import resolve_time
from . import digest as DG

BELOW = re.compile(r"\b(?:below|under|beneath|fall(?:s|en)?|fell|drop(?:s|ped)?|dip(?:s|ped)?|lower than|less than)\b")
LIMIT = re.compile(r"\b(?:recommended|specified|specification|spec|manufacturer|placard|threshold|minimum|safe|required|proper|correct|normal)\b")
COLS = ("tp_fl", "tp_fr", "tp_rl", "tp_rr")
LOW = 0.92


def maybe(res, text, frame, lit, c):
    t = " " + str(text).lower() + " "
    if frame.concept not in ("tyre_pressure", "tyre_pressure_warning") and not frame.concept.startswith("tyre_pressure_"): return None
    if not (BELOW.search(t) and LIMIT.search(t)) or frame.act in ("define", "howto", "spec", "social"): return None
    b = res.brain; cx = DG.Ctx(res, c, frame, lit, text)
    sf, sr = AX._mf(b.c, "veh.tyre_spec_front"), AX._mf(b.c, "veh.tyre_spec_rear")
    if sf is None: return DG._gap(cx, "This vehicle has no recommended tyre pressure on record, so I cannot say when a tyre fell below it.", "tyre:nospec", ok=False)
    spec = [sf, sf, sr or sf, sr or sf]
    lo, hi, label = resolve_time(lit.times[0], cx.qc.ctx) if lit.times else (cx.qc.ctx.lo, cx.now + 1, "in the whole recorded history"); hi = min(int(hi), cx.now + 1)
    margin = int(max(spec) * LOW * 10 * 1.10)                                                                 # a wide first filter in the stored unit (0.1 kPa); the exact test below corrects for temperature
    cols = [COLS[only_]] if (only_ := {"fl": 0, "fr": 1, "rl": 2, "rr": 3}.get(getattr(lit, "position", None))) is not None else list(COLS)
    q = "SELECT ts, " + ", ".join(COLS) + ", tt FROM ext_1m WHERE ts>=? AND ts<? AND (" + " OR ".join(f"{k}<{margin}" for k in cols) + ") ORDER BY ts DESC"
    found = None; n_rows = 0; only = {"fl": 0, "fr": 1, "rl": 2, "rr": 3}.get(getattr(lit, "position", None))                        # "my front left tyre": that wheel only
    for row in b.c.execute(q, (int(lo), int(hi))):
        n_rows += 1
        for i in ([only] if only is not None else range(4)):
            raw = row[1 + i]
            if raw is None: continue
            p = AX._cold(raw / 10.0, row[5]); 
            if p < LOW * spec[i]: found = (row[0], AX.WHEELS[i], p, spec[i]); break
        if found or n_rows > 200000: break
    cx.step(f"the stored tyre pressures of the four wheels against the {sf:.0f} kPa front / {(sr or sf):.0f} kPa rear specification")
    if found is None: return DG._result(cx, f"No tyre was below its recommended pressure ({LOW * 100:.0f} % of the specification, corrected to 20 °C) {label}.", "tyre:below_spec", count=0)
    ts, wheel, p, sp = found; d = cx.stamp(ts)
    return DG._result(cx, f"The last time a tyre was below its recommended pressure was on {d}: the {wheel} tyre at {p:.0f} kPa, {100 * (1 - p / sp):.0f} % under its {sp:.0f} kPa specification (a tyre counts as low below {LOW * 100:.0f} % of it, corrected to 20 °C).", "tyre:below_spec", event_ts=int(ts), wheel=wheel, kpa=round(p, 1))
