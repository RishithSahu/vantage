# Explaining causes: how Vantage answers "why?", and how well

Two kinds of question: *why did a quantity change over days* ("why has my range dropped?") and *why did this event happen* ("why did the driver assistance switch off at 14:02?"). Both end with a cause, the evidence, what else was considered, how to confirm, and a confidence that was checked against outcomes. A third answer is always allowed: **nothing changed enough to need a cause**.

Code: `vgpt/cause/` (`stats.py`, `data.py`, `engine.py`, `library.py`, `events.py`, `answer.py`); it is called from `vgpt/meaning/resolve.py` for the act "why" and replaces the older association-only routes for that act.

## Change over days

1. **Detect.** Each daily series (energy per km, coolant flow, tyre pressures, 12 V voltage, camera sharpness, climate power, ...) is scanned for a step, a ramp, a transient spike or an end-of-period trend. A candidate is accepted only if a permutation test says a change that size is unlikely in the series' own noise. Spikes are removed first; noise is de-trended against the recorded conditions only when those conditions explain at least a quarter of the variance (r² ≥ 0.25).
2. **Attribute.** A physical trip model (energy per km as a function of speed², climate power per km, cold, highway share) is fitted to the older period, then refitted before the onset to confirm it was stable. The remaining difference is split by its shape: *added drag* grows with speed squared, *rolling resistance* is flat in speed, *cold weather* follows the recent-window temperature, *climate overuse* follows HVAC power. A library of cause signatures (cooling pump degradation, tyre slow leak, weak 12 V battery, obstructed camera, roof box, alignment drift, aggressive driving, higher speeds, cold weather, battery ageing ...) says which quantities each cause moves, in what direction and shape.
3. **Judge.** Hypotheses that explain the same symptom compete with each other and with "no known cause"; independent causes can both be reported. Confidence is calibrated on the benchmark below. If the cause lists the quantity but the quantity did not move, the answer says nothing changed.

## One event

For an event (a shutdown, a driver-assistance disengage, a bus error burst, a speed-source conflict) the engine builds the causal chain from the event log and the signals around it (controller reset → driver assistance drop; supply dip → shutdown), ranks causes from the library, and quotes the log lines and signal values that support each. A time with no recorded event gets "no event recorded", not a guess.

## How it is measured

`bench/causal_bench.py`: histories are generated with a known cause planted (the generator's truth is never read by the assistant; any `gen.*` metadata is ignored). Scenarios vary city, season, driver and vehicle, add a distractor (road works) or two causes at once, and include negative controls with no cause at all. A scenario whose planted effect is too small to be seen in the generated data is **not scored** (it is listed in the log); this is a property of the data, and it is reported, not hidden.

* **Dev set** (used to build the engine): `results/causal/dev.json`.
* **Sealed test set** (generated once with different seeds; scored once, after the engine was frozen): `results/causal/test.json`, log `results/causal/test_run.txt`.

Sealed result, 2026-10-23: 74 scored scenarios, **first answer right 87.8 %, right answer within the top three 95.9 %**, confident false causes on negative controls **0 %** (8 controls; 75 % answered "nothing to explain", the rest gave a low-confidence suggestion). Stated confidence 0.85-1.00 was right 95 % of the time (40 cases); 0.50-0.85 was right 100 % (10 cases); below 0.50 it was right 71 % (24 cases).

The nine first-answer misses (of 74): two were confident and wrong — `cooling_pump_4` named *higher speeds* (99 %; the true cause was second), `dirty_windscreen_1` named *climate overuse* (98 %; true cause second) — and the other seven were low-confidence (0.02-0.25: effectively "no cause found"), mostly *alignment drift* (1/4 first answer, 3/4 in the top three) where drift and extra drag are hard to tell apart, plus `roof_box_0`, `roof_box_5`, `harsh_week_3` and `cold_season_7`. In six of the nine the true cause was ranked second. These are open problems; none was tuned on the sealed set.

Events: `bench/event_cause_bench.py` over every planted fault family of the four 105-day vehicle histories: 322 / 322 traced to the planted cause. Caveat: those histories were also what the event rules were developed on, so this is a regression check, not a held-out test.

## Not claimed

* Causes outside the library are reported as unknown, not invented.
* The physical trip model is a calibrated approximation, not a vehicle simulation; effects under a few percent of energy per km are not separable from noise (those scenarios are the "not scored" ones).
* The answers use the vehicle's recorded data only; things the vehicle does not record (battery health in the generated histories) are listed as "could not be checked".
