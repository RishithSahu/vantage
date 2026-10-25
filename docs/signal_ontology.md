# Signal ontology and the registry

`vgpt/signal_registry.py` is the single formal description of **what a vehicle could know** (the ontology) and, separately, **what one particular database really contains** (the profile). Everything else — validator, capability engine, planner, executor, composer, verifier, parsers, and the language model's output grammar — reads from it. Nothing is hard-coded twice.

## Ontology is not data

| Question | Answer comes from | Possible outcomes |
|---|---|---|
| Is "motor torque" a vehicle concept? | `CONCEPTS` (the ontology) | known / unknown |
| Does *this* vehicle record it? | `Profile` (tables and columns that actually hold values) | measured / derived / estimated / unavailable |
| Can it be computed from what *is* recorded? | `DERIVED_REQUIRES` / `DERIVED_ALTERNATIVES` | derived (exact physics) / estimated (uses a stated assumption) / unavailable |

Hence the three honest answers the assistant gives (see `capability.py`):

* **UNKNOWN** — not a signal or metric the system models ("flux capacitor temperature").
* **MISSING_DATA** — a known concept that this vehicle's database contains no values for, and nothing recorded determines it ("motor torque" on a vehicle with no torque or power-and-rpm signals). The answer lists exactly which signals would be needed and the formula that would be used.
* **INSUFFICIENT** — derivable in principle, but not from what is recorded (turn radius with speed but no yaw rate, steering angle or lateral acceleration).

An electrical quantity is **never** used in place of a mechanical one (battery power is not motor power); an estimate is always labelled and lists its assumptions (for example the nominal gear ratio and wheel radius used for the rpm estimate).

## What the registry contains

* **81 concepts** in 8 categories (powertrain, battery, chassis, ADAS, camera, environment, vehicle, trip) — 68 measured, 13 derived. Each has a canonical name, label, unit, plausible physical range (used by the sensor-quality checks), data type (numeric / binary / categorical), synonyms, and a kind.
* **Storage mapping (`STORE`)** from concept to the real schema: `(table, column, scale)` per resolution tier (`1s` = `sig_1s`, 7 days; `1m` = `sig_1m`, 90 days; daily tables). Extended signals live in the additive tables `xsig_1s` / `xsig_1m`.
* **`TABLE_COLUMNS`** — the whitelist of tables and columns the SQL compiler may name. Nothing a person or a model types ever reaches an SQL string.
* **Events** (16): table events (`HARSH_BRAKE`, `DTC_SET`, `WARN_BATT_OVERTEMP`, …), groups (`ANY_WARNING`) and *detected* events found in the signals (`COOLANT_FLOW_DROP`).
* **Named conditions / predicates** (15): `highway`, `city`, `charging`, `parked`, `driving`, `fast_charging`, `raining`, `hard_braking`, `hard_acceleration`, `regenerating`, `night`, `cold`, `hot_weather`, `high_speed`, `idling` — each defined as a deterministic rule over a mode, a signal threshold, or an event window.
* **Trip classes** (`highway`, `city`, `rainy`, `long`, `short`) and **entity metrics** (what trips / charging sessions can be ranked by: consumption, distance, energy, duration, average speed, recency …).
* **Derived metrics** (13): `energy_used`, `distance_driven`, `wh_per_km`, `regen_energy`, `range_estimate`, `acceleration`, `braking_intensity`, `horsepower`, `turn_radius`, `wheel_slip`, `thermal_rise_rate`, `battery_efficiency`, `motor_power_estimate` — with ordered alternatives, e.g. horsepower = motor power × 1.341, or torque × rpm / 7120.
* **Vocabulary** (`LEXICON`, ~340 phrases) shared by the rule parser, the training-data generator and the model grammar; ambiguity (a phrase naming more than one concept) is resolved by context or by asking, never silently.

## Resolution

`Profile.resolve(concept)` returns a `Resolution` with `status`, the storage tiers if measured, the dependencies and parameters if derived, `assumptions`, `formula`, `missing`, and the `alternatives` that were considered. Vehicle parameters (`wheelbase`, `wheel radius`, `gear ratio`, `steering ratio`, `drivetrain efficiency`, `battery capacity`) are read from the `meta` table when present (provenance *vehicle*) and fall back to project nominals only where an estimate is explicitly labelled.

## Sensor quality

`sensor_quality.py` judges whether the data a number came from is trustworthy: freshness, dropouts (short gaps *inside* activity, and a single sensor missing while others report), range violations (against the registry's physical ranges, including derived values — a stuck wheel sensor shows up as impossible wheel slip), stuck values while the vehicle is moving, isolated spikes, timestamp problems, steering **bias** while driving straight, and cross-sensor disagreement (vehicle speed vs GPS vs wheel speeds). Flags travel with the result into the evidence and into the answer ("according to the vehicle speed signal; GPS reported an invalid zero"). No check changes a value.

## Extending

Add a `C(...)` line (and, if it is stored, a `STORE` entry) and the concept is immediately understood by the validator, the capability engine, the planner, the rule parser (through its synonyms), the training-data generator and the model's output grammar. Nothing else needs to change.
