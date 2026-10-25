"""The causes the engine knows: each is a physical explanation with the signature it leaves in a vehicle's data.

A signature is a list of tests on the changes found in the history (engine.build): `need` tests must hold (each in [0, 1] from how strongly the change goes the expected way); `support` tests add weight; `against` tests
subtract.  The scoring is in `score()`; the wording of the mechanism, what to check and how to act is data here, so that the answer can say it and a service engineer can challenge it.

Cause ids are physical, not brand or fault-code specific: "added_drag" covers a roof box, a bike rack, a flat-bed; "rolling_resistance_increase" covers a mis-aligned wheel, a dragging brake, worn or soft tyres."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cause:
    id: str
    title: str
    prior: float
    mechanism: str
    check: str                            # what would confirm it (a measurement, an inspection)
    severity: str = "info"                # info | attention | urgent
    tests: list = field(default_factory=list)
    group: str = ""                       # causes of the same symptom compete with one another (only one is the reason a number moved); the others can all be true at once


def g(s, lo=3.0, hi=8.0):
    """Evidence from a strength in spread units: 0 below `lo`, 1 at `hi`."""
    return float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))


def _sig(z): return float(np.clip((z - 3.0) / 3.0, 0.0, 1.0))


def h(x, lo, hi): return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def t_mag(feat, sign, r0, r1, note=None):
    """The change must be significant (the series said so) AND physically meaningful: a relative size between r0 (0) and r1 (1) in the wanted direction."""
    def f(F):
        c = F.get(feat)
        if c is None: return None
        if not c.ok or sign * c.effect <= 0: return 0.0, ""
        v = h(sign * c.rel, r0, r1)
        return v, ((note or "").format(b=c.before, a=c.after, p=abs(c.rel) * 100) if v > 0 else "")
    return f


# ---- individual tests: (name, weight, fn(F) -> (value in [0,1], note or None)); fn returns None when the data to judge it is not stored
def t_count(feat, sign, note=None):
    """An event rate that rose significantly (Poisson z): 3 -> 0, 6 -> 1."""
    def f(F):
        c = F.get(feat)
        if c is None: return None
        return (_sig(c.strength(sign)) if c.ok else 0.0), (note or "")
    return f


def t_change(feat, sign, lo=3.0, hi=8.0, note=None):
    def f(F):
        c = F.get(feat)
        if c is None: return None
        return g(c.strength(sign), lo, hi), (note or "")
    return f


def t_flat(feat, lo=2.0):
    """Evidence AGAINST a change: 1 when the quantity is steady, 0 when it moved."""
    def f(F):
        c = F.get(feat)
        if c is None: return None
        return (0.0 if c.ok and abs(c.effect) >= lo else 1.0), ""
    return f


def tyre_leak(F):
    best = None
    for w in ("fl", "fr", "rl", "rr"):
        c = F.get(f"tyre_{w}")
        if c is None: return None
        d = c.extra.get("depth", 0.0)
        if c.ok and c.effect < 0 or d >= 10.0:
            sc = g(max(-c.effect, 0.0), 3.0, 10.0) * 0.5 + g(d, 8.0, 40.0) * 0.5
            if best is None or sc > best[0]: best = (sc, w, d)
    if best is None: return 0.0, ""
    others = [F[f"tyre_{w}"] for w in ("fl", "fr", "rl", "rr") if w != best[1] and f"tyre_{w}" in F]
    moved_together = sum(1 for c in others if c.ok and c.effect < -3.0)
    name = {"fl": "front-left", "fr": "front-right", "rl": "rear-left", "rr": "rear-right"}[best[1]]
    return best[0] * (0.25 if moved_together >= 2 else 1.0), f"the {name} tyre fell {best[2]:.0f} kPa below where it stood against the other three, which stayed level"


def _sigm(x): return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def whlaw(kind):
    def f(F):
        m = F.get("wh_model")
        if m is None: return None
        if not (m["ok"] and m["up"]): return 0.0, ""
        s = _sig(m["z"]) * h(m["rel"], 0.03, 0.10)
        p_ramp = _sigm((m["sse_step"] - m["sse_ramp"]) / max(0.04 * m["sse_step"], 1e-9)); p_drag_law = _sigm((m["drag_t"] - 1.5) / 1.0)
        if kind == "drag": return s * (0.35 + 0.65 * (0.5 * (1 - p_ramp) + 0.5 * p_drag_law)), f"{m['resid']:.0f} Wh/km ({m['rel'] * 100:.0f} %) more than the recorded conditions explain; the extra grows with speed squared (t = {m['drag_t']:.1f}) and began abruptly"
        return s * (0.35 + 0.65 * (0.5 * p_ramp + 0.5 * (1 - p_drag_law))), f"{m['resid']:.0f} Wh/km ({m['rel'] * 100:.0f} %) more than the recorded conditions explain, the same per kilometre at every speed"
    return f


def cold(F):
    m = F.get("wh_model"); w = (m or {}).get("window")
    if w is None: return None
    if w["gap"] <= 0 or w["amb_change"] > -1.5: return 0.0, ""
    cold_c = w["terms"].get("cold weather", 0.0) + w["terms"].get("climate power per km", 0.0); t = cold_c / max(w["se"], 1e-9); rel = cold_c / max(m["base"], 1e-9)
    return _sig(t) * h(rel, 0.02, 0.08) * (0.5 + 0.5 * min(1.0, cold_c / max(w["gap"], 1e-9))), f"it got {abs(w['amb_change']):.0f} °C colder, which with the heating accounts for {cold_c:.0f} of the {w['gap']:.0f} Wh/km rise"


def aggressive(F):
    h = F.get("harsh")
    if h is None: return None
    return _sig(h.strength(+1)), (f"harsh events {h.after:.1f} per 100 km against {h.before:.1f} usually" if h.ok else "")


def climate(F):
    hv = F.get("hvac")
    if hv is None: return None
    return g(hv.strength(+1), 4.0, 8.0), ""


def speedier(F):
    sp, hw = F.get("spd"), F.get("hwy")
    if sp is None and hw is None: return None
    c = max((x.strength(+1) for x in (sp, hw) if x is not None), default=0.0)
    return g(c, 4.0, 8.0), ""


LIBRARY = [
    Cause("cooling_pump_degradation", "cooling pump or coolant circuit degradation", 0.12,
          "The pack's coolant flow has fallen well below what it was, so the pack and the motor shed less heat at the same load.",
          "Read the coolant pump's commanded duty against measured flow; inspect the pump, the coolant level and the lines for air or a blockage.", "attention",
          [("flow_down", 3.0, t_mag("flow", -1, 0.15, 0.50, "coolant flow fell from {b:.1f} to {a:.1f} L/min ({p:.0f} % lower)")), ("batt_up", 0.5, t_mag("batt", +1, 0.03, 0.15, "the pack ran hotter ({b:.0f} to {a:.0f} °C) after adjusting for the weather and the load")),
           ("warn_up", 0.5, t_count("ev_overtemp", +1))]),
    Cause("tyre_slow_leak", "a slow leak in one tyre", 0.12,
          "One tyre has lost pressure steadily while the other three held, which is what a puncture, a valve or a rim leak does; a temperature change would move all four together.",
          "Check that tyre for a nail or a damaged valve, then re-inflate and watch whether the pressure falls again.", "attention", [("one_wheel_down", 3.0, tyre_leak)]),
    Cause("weak_12v_battery", "a weakening 12 V battery or charging path", 0.07,
          "The 12 V battery's resting voltage has been sinking day by day; low voltage makes control units reset and sensors misread.",
          "Load-test the 12 V battery and check the DC/DC converter's output and the earth connections.", "attention",
          [("v12_down", 3.0, t_mag("v12", -1, 0.01, 0.05, "the resting voltage fell from {b:.2f} to {a:.2f} V")), ("events", 0.8, t_count("ev_12v", +1))]),
    Cause("camera_obstruction", "the camera's view is blocked (dirt, ice, a sticker)", 0.08,
          "The share of the camera image that lost structure rose for some days, and blocked-camera events were recorded: the lens or the windscreen in front of it was covered.",
          "Clean the windscreen in front of the camera and check the lens for film or ice; confirm the blockage events stop.", "info",
          [("blocked", 2.0, t_mag("cam_block", +1, 0.5, 3.0, "the blocked share of the camera view rose from {b:.2f} to {a:.2f}")), ("events", 1.5, t_count("ev_cam", +1))]),
    Cause("added_drag", "extra aerodynamic drag (a roof box, a rack, an open window)", 0.07,
          "Energy per kilometre stepped up on one day, by an amount the recorded conditions (weather, heating, speed, tyres) do not explain, and the extra grows with speed squared, the signature of drag.",
          "Look for something new on the roof or the back (box, bike rack) and compare consumption at highway speed with it removed.", "info",
          [("unexplained_up", 3.0, whlaw("drag"))]),
    Cause("rolling_resistance_increase", "increased rolling resistance (alignment, a dragging brake, tyre wear)", 0.06,
          "Energy per kilometre crept up by an amount the recorded conditions do not explain, equally at all speeds, which is what extra friction at the wheels does.",
          "Have the wheel alignment and the brakes checked for drag; check the tyre pressures and the tread.", "attention",
          [("unexplained_up", 3.0, whlaw("rolling"))]),
    Cause("aggressive_driving", "a period of harder driving", 0.10,
          "Harsh braking, acceleration or cornering became much more frequent for a stretch of days, and hard driving costs energy.",
          "Compare the driver's trips in that period; no vehicle fault is implied.", "info", [("harsh_up", 3.0, aggressive)]),
    Cause("cold_weather", "colder weather (and the heating it needs)", 0.15,
          "Energy per kilometre rose as the outside temperature fell; the cold raises rolling losses and the cabin heating, and the recorded conditions account for most of the rise.",
          "No repair needed; consumption returns as the weather warms.  Preconditioning on the charger reduces the loss.", "info", [("cold", 3.0, cold)]),
    Cause("climate_overuse", "much more use of the cabin climate system", 0.05,
          "The power the climate system draws rose without the outside temperature changing.", "Compare the cabin set-point and the recirculation setting over the period.", "info", [("hvac_up", 3.0, climate)]),
    Cause("higher_speeds", "faster driving (more highway)", 0.08,
          "The average speed or the share of highway driving rose, and consumption rose with it.", "No fault implied; the routes or habits changed.", "info", [("faster", 3.0, speedier)]),
    Cause("battery_aging", "battery capacity loss", 0.03,
          "The battery's state of health fell while consumption stayed the same, so the range shrinks without the car using more energy.", "Run a capacity test at the service centre.", "attention", [("soh_down", 3.0, t_change("soh", -1, 3.0, 8.0))]),
]


CONSUMPTION = {"added_drag", "rolling_resistance_increase", "cold_weather", "aggressive_driving", "climate_overuse", "higher_speeds"}
for _c in LIBRARY: _c.group = "consumption" if _c.id in CONSUMPTION else _c.id
