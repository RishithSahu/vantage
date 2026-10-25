# VANTAGE / VehicleGPT - system architecture

Written 2026-10-20 for Case Study 5 (*Natural Language Vehicle Information using On-Device SLM/LLM*).  It follows the seven headings the case study asks for and says, for every claim, whether it is **tested here**,
**measured here (on this x86 laptop)** or **proposed**.  The companion document `docs/GAP_REPORT.md` lists what the case study asks for that is still not satisfied.

## 0. In one page

A vehicle's raw traffic - CAN / CAN-FD frames, SOME/IP notifications, Android VHAL properties, UDS/OBD-II diagnostics, QNX / Android log lines, ADAS and camera metadata, GNSS - is decoded **by data, not by code**:
one *profile pack* per vehicle (DBC files, a SOME/IP service catalogue, a DTC table and a `profile.json` that maps every source signal to a canonical concept with unit, scale, enum and priority) is the only
integration artefact; for a car nobody has configured, `vehiclegpt.py ingest` reads whatever capture it is given (candump, ASC, TRC, BLF, CSV, JSON, OpenXC, VSS, ELM327, NMEA, logs) and *writes that pack
itself* from the standards (OBD-II, SAE J1939, COVESA VSS, Android VHAL) and from each signal's name, unit and values (`docs/DATA_SOURCES.md`).  A single-writer **runtime** turns that traffic into a self-describing SQLite database (1 Hz rows, a per-signal "lake", events, logs, trips, charging sessions, frozen event context).  A
0.5-billion-parameter language model with a LoRA adapter reads a question into a *meaning frame* (six words in the shipped adapter; eight - with a second quantity and a reference to the last answer - in the frame-2 adapter, `docs/FRAME_V2.md`) constrained by a grammar generated from the database's own vocabulary; typed deterministic code executes the
frame against the database; long investigative requests are decomposed and answered by an investigation engine that never calls the model.  Everything is local; the assistant has a read-only handle and no path to any bus.

```
 vehicle (or the virtual car)                         on-device runtime                                   assistant (read-only)
 ------------------------------                       -----------------------------------------------     ---------------------------------------------
 CAN / CAN-FD   (SocketCAN, candump)  --\
 SOME/IP        (Ethernet tap, gateway) --\             collectors (listen-only)                            question
 Android VHAL   (property stream)        ---> JSON/UDP/TCP/file/SocketCAN ---> Runtime.feed                   |  safety guard (deterministic, before the model)
 UDS / OBD-II   (ISO-TP responses)       --/                                     |                            |  dialogue completion (follow-ups)
 QNX / logcat / syslog lines            --/                                      v                            |  investigation planner  --> 32 analyses (no model)
 ADAS + camera metadata (and luma)      -/                      ProfileRouter  (this vehicle's mappings)      |  meaning frame (Qwen2.5-0.5B + LoRA, GBNF)
 GNSS (NMEA)                           -/                       redundancy, conflicts, plausibility,          |  arbitration + typed resolver + SQL
                                                                signal events, harsh-event detectors          v
                                                                         |                                  evidence-labelled answer + working
                                                                         v
                                                 Ingestor -> sig_1s / events / logs / gps / cam_1m  +  Lake -> lk_1s / lk_1m / ext_1m
                                                 ledger (trips, charges)  | event_ctx_x (frozen channels)  | source_log (silent / restored)
                                                                         |
                                                              SQLite (WAL), ONE writer, many read-only readers
```

On two devices (the way it is shown and tested; `docs/VANTAGE_LINK.md`, `docs/DEPLOYMENT.md`):

```
 CAR (collector domain)                                         VANTAGE (the Raspberry Pi: runtime + assistant + the app's screens)
 ------------------------------------                          ------------------------------------------------------------
 the vehicle's own traffic, as it streams                       TLS (certificate pinned under the shared key), one-way data
 (or a capture file: ingest -> profile pack)  ==signed stream==>  receiver -> single-writer runtime -> database in RAM
 collectors are receive-only; SocketCAN                          (+ AES-GCM snapshots on disk) -> read-only assistant -> the Vantage app
 controller in listen-only mode                                  <== closed command list: select / mult / stop (no bus access)
```

The boundary between the two domains is *measured*: `vgpt/isolation.py` reads the application's own source and `tests/test_isolation.py` fails if anything in the assistant domain can reach a network, start a process
or transmit (`docs/SAFETY_ISOLATION.md`).

## 1. High-level system architecture

| Layer | Components (code) | Notes |
|---|---|---|
| Sources | real vehicle buses, or `vgpt/live/car.py` (virtual car) -> `vgpt/world/emit.py` (profile-driven bus emitter) | the emitter walks the profile's bindings in reverse, so the *same* car drives any vehicle |
| Collectors | `vgpt/live/collect.py`: UDP, TCP, file tail (candump / ASC / JSON lines), NMEA and syslog over UDP, SocketCAN | listen-only; malformed input is counted, never fatal |
| Runtime | `vgpt/live/runtime.py`: `ProfileRouter` + `LiveIngestor` + `Lake` + ledger + housekeeping + source health + bus monitor | the only writer of the database |
| Store | `vgpt/store.py`, `vgpt/vehicle/lake.py`, `vgpt/vehicle/ledger.py` | self-describing (the database carries its own vehicle profile and signal catalogue) |
| Understanding | `vgpt/meaning/`: dialogue completion, frame model, arbitration, resolver; `vgpt/investigate/` | text patterns live in one file (`literals.py`) under a test that counts them |
| Interface | `ui/server.py` + `ui/modern/` (Ask, Car, Energy, Map, Camera, **Live**) | local only, bound to 127.0.0.1 with a cross-site guard |
| Simulation | `vgpt/world/` (105-day life generator with injected faults and ground truth), `vgpt/live/` (interactive virtual car, road, 13 faults), `ui/live.py` (procedural dash-camera) | CARLA is *not* used (see GAP_REPORT) |
| Scaling | `deploy/tiers.json`, `vgpt/tiers.py`, `vgpt/sessions.py` | one application, four resource profiles |

Hardware (proposed, not built): an automotive SoC or board with 2+ ARM A5x/A7x cores and 2 GB (reference tier), a gateway that forwards frames read-only to the runtime (a hardware data diode is the strongest
isolation), flash storage sized by the retention in section 3.  Operating system: the assistant runs on Linux (ran on a Raspberry Pi 4) and needs only Python 3.10+, numpy, SQLite and llama.cpp from the system; the runtime has no Windows-, POSIX- or Linux-only call outside a guard (a test reads every module) and no other
dependency is required (OpenCV, cryptography, psutil and SQLite FTS5 are optional, each absence is tested).  QNX, Android Automotive / Cuttlefish, AUTOSAR Adaptive and hypervisor guests are covered as data sources and displays, and QNX as a possible host, in `docs/PLATFORMS.md`; none of them was run.  The reasoning for the OS choice is there too; the bill of materials is `docs/HARDWARE_REFERENCE.md`; the safety argument is `docs/SAFETY_CASE.md`.

## 2. Vehicle data collection strategy

**The integration artefact is a profile pack** (`vgpt/vehicle/packs/<vehicle>/`): `profile.json` (sources, signal bindings, event rules, thresholds, retention), the DBC files, `services.json` (SOME/IP), `dtc.json`.
A binding says *where* a concept comes from (`can:pt/VehDyn.Speed`, `vhal:hal/PERF_VEHICLE_SPEED`, `someip:eth/AdasState.SystemState.engaged`, `camera:cam/lane_conf`) and how to convert it (unit, scale, offset, enum map,
validity range, priority).  `Profile.validate()` rejects unknown units, missing signals, duplicate priorities and retention below ten weeks, with the path of every error (`tests/test_vehicle_platform.py`).
Three packs ship: a hatchback EV on three CAN buses (one CAN-FD), a second EV with different buses, names, units and byte order, and a diesel truck on J1939 with a log-only gateway - they differ in every way a vehicle can.
`vgpt/vehicle/onboard.py` drafts a pack from a vehicle's own DBC files (measured against the hand-written packs: precision 1.0, recall 0.81-0.95 on the signals the shared registry defines - a diesel's own parts are kept as vehicle-specific concepts; what it does not recognise it lists with candidates, it never guesses).

| Source | Wire format handled | Decoder | Becomes |
|---|---|---|---|
| CAN / CAN-FD | classic and FD frames, Intel/Motorola, signed, scaled, multiplexed (J1939 tyre report) | DBC parser + `ProfileRouter` | canonical signals, error-frame events |
| SOME/IP | binary notifications and SOME/IP-SD offers, service catalogue with enums | `protocols/someip.py` | signals (ADAS state, lead vehicle, lane quality, charging) and events (ECU reset, shutdown, voltage dip, AEB, disengagement) |
| Android VHAL | `VehiclePropValue` parcelables with AOSP property ids and areas | `protocols/vhal.py` | secondary (redundant) signals and range / ignition / charge state |
| UDS / OBD-II | ISO-TP segmentation, 0x59 DTC reports, mode 01 PIDs, DIDs | `protocols/uds.py`, `isotp.py`, `obd.py` | DTC set / clear events with the vehicle's own code text |
| QNX / Android / syslog | slog2info, logcat, RFC 5424 / 3164 lines | `protocols/logs.py` | searchable log table (FTS) and events matched by profile rules |
| ADAS, camera | SOME/IP object and lane services; camera feature records; or a luma plane analysed on the box (only metadata is stored) | router + `camera.py` | lane confidence, blockage, sharpness, brightness; CAMERA_BLOCKED / DEGRADED |
| GNSS | NMEA 0183 RMC / GGA with checksums | `protocols/nmea.py` | GPS rows, places, trips |

What the router adds beyond decoding (all tested): **redundant sources** (a secondary source takes over when the primary is stale; when both are fresh and disagree for 5 s a `SENSOR_CONFLICT` event with both
values is recorded), **plausibility** (a value outside the physical range is dropped and counted, with a rate-limited `SENSOR_INVALID` event), **signal events** declared in the profile (coolant > 45 C for 10 s),
**derived driving events** (hard braking / acceleration / cornering from the acceleration signals), **source health** (a bus that falls silent while the car is awake is written to `source_log` with its recovery).

How traffic reaches the runtime: a collector next to wherever the bytes already are.  UDP/TCP JSON records from a gateway, `candump -L` or Vector ASC files (recorded drives), a growing log file, NMEA/syslog
datagrams, a SocketCAN interface (`tools_live.py ingest ...`).  The virtual car can run as a separate process and send over UDP exactly as a gateway would (`tools_live.py car ...`; tested through the loopback).

## 3. Historical data storage approach

| Tier | Content | Default retention | Written by |
|---|---|---|---|
| `sig_1s` | nine core signals at 1 Hz while the car is awake (1 per minute parked) | 7 days (tier-dependent) | `Ingestor` |
| `lk_1s` | every other signal at 1 Hz (EAV: `sid, ts, v`) | 7 days | `Lake` |
| `sig_1m`, `lk_1m` | per-minute mean / min / max; `ext_1m` tyres, rain, light, 12 V, ADAS share | **120 days (17 weeks)** | live minute buckets + `store.compact` |
| `events`, `logs` (+ FTS) | every event with detail; WARN+ log lines | events kept; logs 120 days | `Ingestor` |
| `event_ctx`, `event_ctx_x` | the channels 2 min before to 30 s after (15 min for thermal / DTC) around every important event, frozen when the post-event window exists | 365 days | `LiveIngestor` |
| `trips`, `trip_ext`, `charges` | one row per journey / charging session derived from the 1 Hz rows (same derivation for bulk and live) | full window | `ledger.update` every 2 min |
| `source_log`, `gps_1s`, `cam_1m`, `vday`, `places` | bus health, position, camera quality, daily odometer / 12 V, named places | per tier | runtime |

The database describes its own vehicle (`meta vehicle.profile`, `sig_catalog`, per-signal SQL views `lkv_<sid>_1s/_1m`): the typed query engine reads a lake signal exactly like a column, so statistics, extremes,
thresholds, trends, change detection and "why" work on any signal of any vehicle with no code change.  A bulk writer (used to build the 105-day histories) and the live path are tested to produce the same rows
(`tests/test_world.py`: an hour of raw frames equals the bulk rows within the wire resolution).  Measured: 105 days of the hatchback = 92 MB, 1.0 M lake minutes (`results/scaling/*history.json`).

## 4. SLM selection rationale

* **Model**: Qwen2.5-0.5B-Instruct, 4-bit (llama.cpp GGUF, ~400 MB) + a 35 MB LoRA adapter (rank 32; `meaning-lora-full_v11.gguf` since 2026-10-20, `full_v6` before); a newer adapter is adopted only if it passes the frozen-set gates, see
  `docs/TRAINING_QUEUE.md`).  Chosen for the case study's budget: ~2 GB and two A5x cores, answers in < 2 s.  Measured here: whole assistant process 627 MB, 0.75 s median / 0.91 s p95 per answer on x86;
  pinned to two cores 0.83 s / 1.16 s.  ARM is **not measured** (no QEMU / board run has been approved).
* **What the model does**: it only *reads* a question - it writes six words (`act concept op cond vs ent`) chosen from the ontology's vocabulary under a grammar generated from that ontology and from the database's
  own catalogue (a new vehicle's concepts become decodable without code).  It never writes numbers, SQL or the answer.  Measured exact-frame accuracy of the shipped adapter (docs/semantic_method.md, before the platform
  vocabulary of 2026-10-20 was added): 93-94 % on the novel, messy and sealed sets, 90 % on the case study's own questions, 80-95 % on the adversarial `hard` sets; the rest becomes a clarification or an honest
  "not recorded".  Some of these sets were used while the adapter was being improved; the sealed ones were not.  Results for the retrained adapter are appended to `docs/TRAINING_QUEUE.md`.
* **What it does not do**: arithmetic, retrieval, windowing, statistics, explanation and the safety decision are deterministic code.  A deterministic **safety guard** runs *before* the model: a question about
  continuing to drive with a hazard is answered by policy, never by a model's confidence.  Long multi-part investigative requests bypass the model altogether (section 5).
* **Why not larger**: memory and latency on the reference board; a bigger base needs its own adapter (training), which is a configuration, not a code change (section 9).  Earlier trials of other small models
  (LFM2.5, Granite) are in `docs/semantic_method.md` / `VANTAGE_STATUS.md`.

## 5. Search and retrieval workflow

1. **Guard**: safety policy, prompt-injection neutralisation of text fields (log text is data, never an instruction), length and control-character limits.
2. **Investigation planner** (`vgpt/investigate/`): a request that names several things to establish (windows, counts, sources, "rank", "compare", "what evidence is missing") is decomposed (sentences, windows,
   counts, concepts, events, anchor) and routed by cue scoring to one of 32 deterministic analyses: incident reconstruction, ADAS performance, cross-source discrepancies, co-occurrence versus chance, lead times,
   trip outliers and attribution, change points, health ranking, DTC cycles, intermittent-fault conditions, thermal soak, range-estimate drops, "the most important unanswered question".  Every report keeps
   *recorded facts*, *associations* and *hypotheses* apart, states the resolution of the data and what is missing, and checks premises ("a battery warning at 14:32" that is not in the record is said first).
   62 of the 85 prompts of the owner's own test session are routed to an analysis (the labels in `tests/investigation_prompts.tsv` were reviewed by hand; the other 23 are short ordinary questions for the frame
   model), and the analyses are checked against the dataset's injected ground truth (`tests/test_investigate.py`).
3. **Dialogue completion** (`vgpt/meaning/discourse.py`): "and the week before that", "for just yesterday", "which one was the hardest", "what about the rear left", "how did you get that" are completed from the
   previous turn (period shift, concept swap, ordinal, referent, wheel position, provenance).  This layer currently uses small cue lists; moving them into trained frames is queued (`docs/TRAINING_QUEUE.md`).
4. **Frame**: the model reads the question (above); `arbitrate.py` lets a longer glossary term beat a term it contains; frames that do not type-check are not answered.
5. **Resolver** (`vgpt/meaning/resolve.py`) maps the frame to typed queries (`vgpt/query_*`), executed over the right tier; the **evidence composer** adds samples, tier, data quality, gaps and an evidence class
   (measured / derived / estimated) to every answer; an unavailable value says so and never becomes zero.
   Beyond single values the resolver answers *how often* and *from the vehicle's own records* (`meaning/eventx.py`, `meaning/records.py`): whether an event is becoming more frequent (exact binomial test on the two
   halves of the window, or the week with the largest rise), two periods or two driving conditions (counts **and** rates per driving hour), the day / week with the most events, the text logs searched by words and
   source (repeated lines folded), the reporting unit of an event (subsystem of a code, unit that reset, bus that went silent), the state of the diagnostic codes (active / cleared / came back), one typed code's
   history, how long a source was silent, where a measurement comes from (sources, rate, unit, range - read from the profile stored in the database) and the charging ledger.  All of it is arithmetic over SQL; none of it
   decides what a question means.

**Where learning ends and code begins.**  The model decides what a sentence *means* (the six-word frame); the ontology and glossary say which words exist; code computes, parses values (a number, a date, a fault code)
and decides what is safe.  Follow-up completion, long-request routing, rare-word arbitration and log-search stop words are still rules - they are listed, with the test that would retire each, in
`docs/TRAINING_QUEUE.md`.  The ontology was extended on 2026-10-20 (events as count series, `diagnostic_code_cleared`, `software_update`, `source_dropout`, `service_log`, `diagnostic_status`, entities
`week` / `source` / `place`, `spec` on a signal) instead of adding phrase rules, and the model is retrained on the new vocabulary under a pre-registered gate.

## 6. Edge deployment architecture

Processes (all local): `tools_live.py ingest` (runtime + collectors, the only writer) and the assistant server (`ui/server.py`: read-only handles, one model context).  The assistant and the runtime meet only
at the SQLite file.  The virtual car is a development tool and is not part of the in-car footprint.

| Tier (`deploy/tiers.json`, `VGPT_TIER`) | Target | Model contexts | Sessions | SQLite cache | Fine / coarse history |
|---|---|---|---|---|---|
| `edge2g` (default) | 2 A5x cores, 2 GB (the case study) | 1 | 2 | 4 MB | 7 d / 120 d |
| `pi5` | Raspberry Pi 5 / Cortex-A76 board, 8 GB | 1 | 4 | 16 MB | 14 d / 180 d |
| `vm8` | validation VM, 8 vCPU, 16 GB | 3 | 12 | 64 MB (+512 MB mmap) | 30 d / 365 d |
| `hpc` | vehicle HPC, 12 cores, 32 GB | 3 | 32 | 256 MB (+2 GB mmap) | 60 d / 730 d |

**Answer latency, and the decoder that halves it (2026-10-21).**  A question costs one decode of the 0.5 B model.  llama.cpp's grammar sampler, which keeps the output inside the closed word lists, walked the whole 151,000-token
vocabulary at every step: **62 % of the time of a parse on this laptop (913 ms with it, 351 ms without) and 1.9 s on the Raspberry Pi**.  `TokenMask` (`vgpt/meaning/interface.py`) applies the same constraint as a token mask built from the
same word lists (a trie over each slot's token sequences, a set of states because one word can be a prefix of another): on 200 frozen questions the frames and the confidences are **identical** (200 of 200, `tests/test_token_mask.py`),
and the parse takes **348 ms instead of 905 ms (frame 2) and 298 ms instead of 749 ms (full_v11)** at two threads.  The grammar path stays as the reference and as the fallback if a mask ever yields a non-frame (`VGPT_DECODER=grammar`).
End to end (question to answer, two threads, this laptop) the frame-2 adapter is now well under the PDF's 2 s; **the Raspberry Pi was off and was not re-measured**, so its 4-6 s for a new wording is still the last measured number there.

Measured on this laptop (12 logical CPUs, ~6 physical; `tools_scaling.py`, results in `results/scaling/`):

* **Ingest**: one runtime process decodes and stores 32.9 k messages/s - a vehicle sends ~87 per simulated second, so one process could serve ~380 vehicles' real-time traffic; 8.5 CPU-seconds per simulated hour
  of driving per vehicle; decode 18 us/message.  N independent runtimes on N processes: 1.76x (2), 3.63x (4), 5.27x (8) - linear up to the physical core count.  Runtime process alone: 37 MB.
* **History**: 14 -> 105 days: database 60 -> 92 MB; the data side of a standard question (model excluded) p50 25 -> 47 ms, worst 80 -> 376 ms (the range explanation); investigations p50 26-39 ms.
* **Concurrent people** (real model, 2 threads per context): one model context serves ~1.5-2 answers/s however many people ask, so latency grows with the queue (1 / 2 / 4 / 8 sessions: p50 0.75 / 1.2 / 2.4 / 4.6 s).
  Three contexts: 4.2 answers/s at 8 sessions, p50 1.7 s.  Memory: ~0.6 GB for the first context, ~1.1 GB for each additional one (the LoRA is applied per context) - which is why the 2 GB board has one.
* **Not measured**: any ARM core.  Cortex-A5x is several times slower per core than this x86 laptop; the p50 on the reference board will be higher than 0.75 s and must be measured (GAP_REPORT).

## 7. User interaction design

The interface (`ui/modern`, a car-cockpit design: speed always visible, colour reserved for meaning) runs in an application window on the vehicle's display (section 8e) and also in any browser for development.  **Ask** (natural language, follow-ups, evidence chips, "show your working", persona driver / service engineer),
**Car** (instrument cluster, tyres, warning lamps that open a question), **Energy** (range, power, efficiency, pack temperature trends), **Map** (last 30 minutes), **Camera** (live view and events with thumbnails),
**Live** (new): the simulation controls (drive, charge, climate, rain, speed of time, jump to the next morning), 13 injectable faults, every source with its state and rate, the **bus monitor** (message name, id,
rate, raw bytes, decoded signals), the events and log lines the buses reported, and all decoded signals with their source.  Press a fault, watch it travel from the bus to the database, then ask the assistant
about it.  Voice input (wake word "Vantage", offline recogniser), spoken answers and unit preferences exist; no user study has been run.

## 8. Simulation and its integration with the interface

* **105-day worlds** (`python -m vgpt.world.build`): real driving traces (VIF-AEGIS) for speed and accelerations, synthetic offline weather, commuter routines, charging, 8 physics stories (cooling pump, tyre leak,
  weak 12 V, dirty windscreen, roadworks, roof box, harsh week, alignment drift) and **11 injected fault families with ground truth** (`*_truth.json`): ECU resets during lane keeping, power events and one
  unexpected shutdown, body-bus dropouts, speed-source and ambient-temperature sensor conflicts, an intermittent DC/DC fault (speed > 100 AND hot motor AND warm outside, ~45 % of such episodes), thermal soak,
  DTC clear / return cycles, ADAS flapping, three emergency-braking events, a dashboard range revision.  Tests compare what the analyses find with what was injected.
* **The virtual car** (`vgpt/live`): kinematics along a real route between the vehicle's home and work places, EV (road load, drivetrain / regen efficiency, pack voltage sag, thermal loop with a pump, AC / DC charging
  with taper) or diesel (road load, gearbox, fuel) powertrain, tyres (ideal gas, leaks), brakes, 12 V, lane keeping, camera, and 13 faults.  It emits **wire-format** traffic of the chosen vehicle's profile.
  Verified: physics sanity (energy per km, no violent driving), every concept the three profiles bind is produced, decoded and stored, each fault leaves its evidence through the real decoders (events, codes,
  signal behaviour, silent source), ledger energy matches the car's own state-of-charge change (within 15 %), and the head-unit server shows the car on the cluster, the map and the camera while the same assistant answers about it
  (`tests/test_live.py`, 34 tests).  196x real time on one core; ~87 messages per simulated second.
* **CARLA** (the case study's example) is **not** used; the virtual car is the "other vehicle driving simulation" route.  A bridge that maps CARLA's vehicle state onto the same canonical state is a small adapter
  (the emitter only needs `{concept: value}` per second) but CARLA is a ~20 GB download and was not installed (GAP_REPORT).

## 8b. What kind of vehicle is this?  Decided by the data (2026-10-20)

Nobody tells a real vehicle's head unit "you are an EV" or "you are a diesel": the vehicle sends signals.  `vgpt/vehicle/caps.py` therefore reads the **database** (its signal catalogue and the rows actually stored) and decides:

| it reports | the vehicle is |
|---|---|
| a battery state of charge, no fuel level or engine signal | battery-electric |
| a fuel level or engine speed, no battery | a combustion vehicle - and **diesel** if it reports diesel-only hardware (particulate filter soot / regeneration, AdBlue level, boost, exhaust gas temperature, rail pressure), **petrol** if it reports spark-ignition signals (lambda, short / long fuel trims, manifold pressure, mass air flow, catalyst temperature, ignition timing); when neither group appears the rev range decides and the summary says so |
| both | hybrid |
| neither (speed and position only, e.g. the recorded CS5 files) | "energy source not reported": the page shows no battery, no fuel, no gauge it cannot fill.  If the recorded file *labels* itself (`veh.powertrain = diesel_ice`) that label is used - and said to be a label - only because the signals say nothing; signals always win |

`caps.summary` says in words what the car is and why ("Diesel vehicle. Reports ... Basis: diesel-only signals: ...; no traction-battery signals. Data arrives over CAN (j1939)."); it is shown under the car on the Car page and appended to the answer of "what kind of vehicle is this".

**The page adapts to it** (`ui/modern/app.js`, `applyKind`): the energy arc is a battery or a fuel gauge, the Energy page becomes **Fuel** (litres in the tank, range from the vehicle's own fuel economy, "when did I last refuel"), readings, warning lamps, quick tiles, suggested questions and charts are built from the signals that exist (a particulate-filter and an AdBlue lamp for a diesel, an emissions lamp for a petrol car, no charging lamp without a battery), and anything the data does not hold is simply absent.

**The assistant adapts the same way, without a new model** (`vgpt/meaning/fuelx.py`): the model reads *what* is asked ("how much energy is left", "what is the consumption", "when was the last fill-up", "how is the motor") whatever powers the car; for a vehicle whose data shows fuel and no battery the resolver executes those meanings in fuel terms (fuel level and range, litres per 100 km and litres used, the refuel ledger, engine health from the signals this engine has).  Signals only a vehicle declares ("AdBlue", "long-term fuel trim", "catalyst") are found through that vehicle's own vocabulary (`arbitrate.own_vocabulary`, from the profile stored in the database).  The older rare-word arbitration rules are **off by default** since `full_v11` (`VGPT_ARBITRATE=1` restores them): over 2,176 frozen questions they changed 59 correct frames and corrected 2.

**Data for it**: `vgpt/world/ice.py` lets the world builder live the same 105-day life as a diesel truck (`rhino_diesel`, SAE J1939) or a petrol hatchback (`kestrel_petrol`, OBD-II style) - gears and engine speed, torque and load, fuel rate and level with refuelling events, coolant and oil warm-up, and the diesel / petrol specific signals - and removes every battery channel.  `python -m vgpt.world.build --profile rhino_diesel --days 105 --seed 9 --city graz --driver long_distance --out data/world_rhino.db` (and `--profile kestrel_petrol ... --out data/world_kestrel.db`).  Not covered yet: a hybrid with data (none exists), DPF regeneration as an *event* the assistant can list (it is a state signal and a log line), and the canned texts of the older `dialogue.py` that assume an EV.

**Power, in engine terms** (`fuelx.engine_power` / `power_stat`): a combustion vehicle has no battery power, and the model (trained mostly on electric questions) reads "how much power do we produce" as battery power or capacity and "engine power" as engine load.  The glossary tags the words that mean power (`quantity: power`: power, kW, hp ...; a data line, not code), and for a vehicle that reports **torque and engine speed** the answer is power = torque x rpm / 9549, in kW and hp, now or the highest / average over a period (from the minute averages while the engine ran).  An engine that reports load but no torque (an OBD-II style petrol car) is told so, with what it does report; nothing is estimated from something else.  A vehicle whose file declares an engine but whose data holds only speed is answered in one plain sentence ("a diesel, no traction battery; its data holds only speed ...").

**Recordings with a sample every few seconds or minutes** (the CS5 files): the head-unit replay (`ui/server.py` `make_plan`) detects the stored sample spacing; when it is more than 5 s the recording is played faster than real time, two samples per second (at most 120x), from the first movement, looping, and the page says "recording shown 120x faster".  Real-time replay would hold each reading for a minute, look parked, and pause the camera clip.  Whatever the recording holds is shown as recorded: this file keeps each speed for a typical 37 minutes, so the speed changes about every 18 s.  A fault code that stays active is **one** event (the importer writes an event when the set of active codes changes, plus a "cleared" event), not one per row.

## 8c. How the data reaches the assistant (2026-10-20)

The application behaves as it would inside a vehicle.  A recorded vehicle is only a **source**: it sends its data as bus traffic, in real time, and the application stores what arrives and answers from that.

```
sensors / ECUs --bus--> collectors > ProfileRouter (decode) > LiveIngestor --> the application's own database --> the assistant
 (a recording plays                                  |  (single writer)         (1 Hz rows, minute rows, events,        |
  this part)                                         +--> live bus (last ~20 s, in memory) --------------------------------+
                                                                                    "right now" questions are answered from here
```

* **Source** (`vgpt/live/replay.py`).  `WireSource` (vehicles with a profile: aster, meridian, rhino, kestrel) re-sends every recorded second as the raw traffic of that vehicle (CAN frames, SOME/IP events, VHAL properties, GNSS sentences, UDS diagnostic reports, log lines, camera metadata) through the **same** decoders a real car would use; `RecordSource` (the Case Study 5 files, which have no bus description) delivers each recorded row at the moment it was recorded.  Before the replay starts the application holds the history up to that moment and nothing after it (`cut_after`).  The clock follows the data that has arrived, never the other way round.
* **Ingest** (`vgpt/live/runtime.py`, `vgpt/store.py`): one writer.  Decoded values become 1 Hz rows; every finished minute is rolled into the minute tier at once (`store.rollup`, the same duration-weighted averages the daily compaction writes) and the open minute is kept up to date; events the vehicle reports (a diagnostic code, a charge start) are stored as sent, and events the ingest derives from the values (hard braking, charge sessions, trips) are stored as derived.  A car waking after a silence sends every slow message once (`Emitter.tick(full=True)`), so a parked car's ignition state is never missing.
* **Answering** (`vgpt/livebus.py`, `live_snapshot.py`, `query_executor.py`).  **Data from the last ~20 seconds is read live from the bus** and the evidence line says so ("Taken live from the vehicle's bus as it arrives; it is also being stored in the dataset."); **anything older, any period, any history question comes from the dataset.**  Only signals that are worth more live (speed, position, charge, temperatures, pressures, power, drive mode) are offered live; a derived estimate (range, efficiency) never is.
* **Time**: real time is the default and the only mode of a vehicle.  *Settings -> Test track* has a **Time speed** slider (1x ... 600x) for the test bench; the speed actually reached is limited by decode throughput (about 220x on one core) and the clock shows what was reached.  Choosing a dataset returns to real time.
* **How faithful is it?**  `bench/meaning_eval/pipeline_fidelity.py` asks the 1,075 frozen questions of the finished recording and of the database built from the arrived data at the same moment, with the correct reading supplied, so a difference can only come from the data: **83.1 % identical answers, 84.2 % the same content** (evidence lines aside), up from 67.6 % / 75.4 % before the minute tier and the waking-car fix.  What remains is mostly expected: a trip or charge still in progress is not yet a finished record; the recording contains seconds after the moment of asking; the ingest derives a few events (for instance ADAS state changes) the recording does not hold; events with no wire rule (power derating, over-temperature warnings) are not replayed.  Results: `results/meaning/pipeline_fidelity.json`.

## 8d. Roles, privacy and security (2026-10-20)

The case study asks for value to *drivers, service engineers, fleet operators and validation teams* and for **local-only processing and protection of vehicle and driver data**.  Roles are therefore **access control**, not a presentation choice (`vgpt/access.py`, `privacy.py`, `rolepages.py`, `securestore.py`):

| role | sees | never sees |
|---|---|---|
| Driver | everything about their own car (no passcode: the person in the seat) | - |
| Service engineer | **Service report** (every component's verdict with its evidence; "no data" is never "fine"), diagnostic codes with history and context, source health | coordinates, saved places, camera pictures |
| Fleet operator | **Fleet** page (use, trips, energy, warnings of every vehicle in the folder; no positions) | positions, camera pictures, who drove where |
| Validation team | **Validation** page (data sources and their silences, sensor conflicts, event counts, timing of answers, pipeline status) and the **audit log** | personal data |

* Passcodes are stored as salted scrypt hashes, with lock-out after repeated failures; the session is an HttpOnly, SameSite=Strict cookie; every response to a non-driver role is **scrubbed on the server** (coordinates, saved-place names, camera pictures, position and track keys), and the question gate refuses location questions for those roles - editing the page's HTML is not a way round it.
* **Audit**: a hash-chained log (`access.Audit`) records who did what (sign-in, export, erase, role page viewed, voice used) - never the content of a question or answer; `/api/audit` verifies the chain.
* **Exports** (`securestore`) are signed (HMAC) and may be encrypted with AES-256-GCM under a passphrase; a tampered pack fails verification.
* **Erase**: the driver can erase location history and camera pictures from the working copy; new fixes are then not stored (`drop_location`).
* **Live posture** (`privacy.posture`): the *Privacy* page checks, on the running machine, that the server listens on loopback only, that no outbound connection exists, and that the database opens read-only - the page shows what was measured, not what is promised.
* **Driving lock** (optional, *Settings -> Test track -> Lock typing while the car moves*): while the vehicle moves, typing, settings, role changes and privacy actions are refused **by the server** (423) and answers are shortened and read aloud; questions are tapped or spoken.  It is **off by default in this test build** so a recorded drive can be typed to; a vehicle build turns it on (`VGPT_RESTRICT=1`, or the switch; the choice is kept in the application's folder).
* Still not done: encryption of the database at rest, a hardware-backed key store, per-driver profiles inside the driver role, and any ASIL argument.

## 8e. The application window, the car display and voice (2026-10-20)

* **An application, not a web page.**  `python vehiclegpt.py app [--kiosk]` starts the assistant on a random loopback port and opens its interface in an application window of its own (the Chromium that every Raspberry Pi OS image contains, or Edge / Chrome on a PC; `vgpt/appshell.py`): no address bar, no tabs, a private profile folder, full screen with `--kiosk`.  The window is opened with a one-time launch token that becomes a cookie; the server answers nothing else (403), so another browser or program cannot talk to it (`tests/test_appshell.py`).  `deploy/raspberrypi/` holds the launcher, the autostart entry and a systemd user unit.  **Written and tested on a Windows PC only; the first start on Pi hardware is the real test.**  A native Qt window would need `PyQt6-WebEngine` (a download) and was not added.
* **For the car display**: desktop / laptop / car screens only (no phone layouts to design for); the opening view fits without scrolling at 800x480 (a 7-inch Pi display), 1024x600, 1280x720 and 1912x990; touch targets are 48-52 px in an application window or on a touch screen; no hover-only controls; no text selection, context menu or visible cursor (after 5 s) in the application window; daylight theme from the ambient-light signal (with hysteresis) or the clock; the layout checker (`ui/modern/layout_check.js`) fails a page that scrolls twice, overflows, clips text, overlaps the header or draws something marked hidden.
* **Voice input** (`vgpt/voice.py`, `ui/modern/voice.js`).  The microphone button starts listening; "**Vantage**, how is my battery?" writes the question into the text box while it is being said (and asks it at once while the car moves, or if *Ask spoken questions at once* is on); "Vantage" alone listens to the next sentence; "**Vantage stop**" or the Stop button ends listening.  The audio is turned into 16 kHz samples in the page, sent to the local server in quarter-second pieces, and recognised **on this computer** by an offline recogniser (Vosk with the 40 MB `vosk-model-small-en-us-0.15`, installed 2026-10-20 with the owner's approval; where it is missing the button says exactly what is).  Anything said that is not for the assistant is thrown away on recognition: it is not stored, logged or sent to the page; audio never touches the disk; the microphone is not listened to while the car speaks an answer.  The browser's own speech recognition is **not** used (it sends audio to a company's server), and spoken answers use only a voice installed on this computer.  The wake word and the stop command are matched on the recognised words (a fuzzy match of the first word) - an interim rule listed in `docs/TRAINING_QUEUE.md`; a trained keyword spotter would replace it.  Tested with a scripted stand-in (`tests/test_voice.py`) **and with the real recogniser**: sentences spoken by this PC's own Windows voices (Hazel, Zira) were played into the page as a simulated microphone, through the sound worklet, the server and Vosk, in real time.  Result (two voices, 8 sentences each): talk not addressed to the car produced no event at all; "Vantage, how is my battery" appeared word by word in the text box within about half a second and was complete about a second after the sentence ended; "Vantage" then "where am I" worked; "Vantage stop" ended listening on the page and the server; the rare word "AdBlue" is **not** in this small model's vocabulary and was written "a blue level" / "a global level" (the meaning model then sees different words; a vehicle-specific vocabulary for the recogniser is open), and "how fast am I going" was written "how fast and my going" once.  Cost on this x86 PC: 137 MB of memory, 0.55 s to load, 5.8 % of one core while listening (real-time factor 0.058); **not measured on ARM**.  Synthetic voices are cleaner than a cabin full of road noise: **no word-error rate with real speakers and real noise has been measured.**

## 8f. The car and Vantage on two devices (2026-10-20/21; `docs/VANTAGE_LINK.md`)

* The laptop is the **car**: it holds the recorded vehicles and sends the chosen one over a signed link (`vgpt/link.py`).  The Raspberry Pi is **Vantage**: it stores what arrives in its own database (wiping the previous vehicle's storage), decodes the bus traffic, runs the assistant (the language model is on the Pi) and hosts the website the laptop shows.
* A vehicle with a profile streams its **raw bus traffic** at its profile's rates, each message released at its own time; a recording without a profile streams rows.  **The receiver needs nothing per car**: the car sends its profile pack (profile.json + DBC/DTC files) with the history and Vantage builds its decoders from it.
* Vantage runs under a kernel-enforced **2 GB / 2-core** limit (peak 978 MB measured over five vehicles); it survives restarts of either side (resume from the last stored second).  Not done: encryption of the link; answer time for a new wording on the Pi (3.7-6.2 s against 2 s).

## 9. Safety isolation, privacy and security

* The collectors are listen-only and this is **measured, not promised**: `vgpt/isolation.py` reads the application's own source and fails if any module outside a short, documented collector domain can reach a
  network, start a process or transmit, or if anything but the receive-only SocketCAN collector touches a CAN socket; the real assistant is run under socket/process recording while it is asked to operate, attack
  and leak (`tests/test_isolation.py`); its database handle is read-only at the file, SQL and authorizer level.  The safety guard runs before and independently of the model.  This is a **software boundary, tested
  on one host** (`docs/SAFETY_ISOLATION.md`).  Vehicle-level isolation (separate partition / hypervisor, hardware diode on the gateway) is proposed, not built, and no ASIL argument is made.
* No cloud dependency: the server binds 127.0.0.1 with a DNS-rebinding and cross-site guard; no network call exists outside the local collectors.  Log text is neutralised before it can reach an answer.
* Roles are access control (section 8d).  Encryption at rest is built for Vantage (`--encrypt-at-rest`: the database lives in RAM, only AES-256-GCM snapshots reach the disk, `vgpt/atrest.py`, `tests/test_atrest.py`); the
  link car -> Vantage is TLS with a certificate pinned under the shared key (`vgpt/tlsmat.py`).  Not done: retention-based deletion of personal data on request, per-driver profiles, key storage in a TPM / secure element.

## 10. How to run it

```bash
python -m vgpt.world.build --profile aster_ev --days 105 --out data/world_aster.db     # a 105-day history with ground truth (also meridian_ev, rhino_diesel)
python ui/server.py --db data/world_aster.db                                           # head unit + live car; open http://127.0.0.1:8765/#live
python tools_live.py run --db data/live_demo.db --from-db data/world_aster.db --minutes 40 --fault coolant_pump@300   # headless drive with a fault
python tools_live.py ingest --profile aster_ev --db data/live.db --udp 9001            # the runtime as a service ...
python tools_live.py car --profile aster_ev --udp 127.0.0.1:9001 --speed 5 --trip      # ... fed by the virtual car in another process
python tools_scaling.py all                                                            # the scaling measurements
python -m pytest tests/test_live.py tests/test_investigate.py tests/test_scaling.py tests/test_vehicle_platform.py tests/test_world.py
VGPT_TIER=vm8 python ui/server.py --db data/world_aster.db                             # a different resource tier, same application
```
