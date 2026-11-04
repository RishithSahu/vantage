# 03. System architecture

This follows the seven headings the case study asks for under "Expected Output". Where something is measured it says so; where it is only proposed it says so. A one-page picture is `docs/VANTAGE_ARCHITECTURE.drawio`.

## 1. High-level system architecture

### 1.1 Two programs and one pipe

```
 CAR (separate program; in a real vehicle the buses simply exist)         VANTAGE (the assistant: Raspberry Pi, head unit, HPC or laptop)
 ----------------------------------------------------------------         -----------------------------------------------------------------
 a car model (one of 49) + a drive                                         receiver  --- sniff -> fingerprint -> decode (ProfileRouter)
   simulated route, or a recorded history                                     |
 emitter walks the car's profile backwards:                                   v
   CAN / CAN-FD, LIN, FlexRay, CANopen, J1939, OBD-II/UDS (ISO-TP),         single-writer runtime -> SQLite tiers (seconds / minutes / daily / records)
   SOME/IP, DoIP, Android VHAL, logs, NMEA GPS, camera pictures                 |                      + lossless archive + sealed at rest
 transmitter gives each wire a channel and writes the true form                 v
 writer: WRITE-ONLY end of a local pipe  ======== one-way pipe =========>   assistant (read-only handle): guards -> model -> resolver -> executors
        (Windows named pipe / Linux FIFO, no network)                          -> evidence -> composer -> verifier -> privacy -> screen
```

* **The Car never imports Vantage and Vantage never imports the Car.** Only a local pipe joins them. Vantage is never told the car's model, protocol or channel numbers: it works them out from the bytes (section 2).
* **The wire writer exists only in the car's code**, so Vantage cannot transmit to the car even by mistake.
* In a real vehicle the Car program is replaced by the vehicle's own buses and a listen-only collector (for example a SocketCAN interface in listen-only mode).

### 1.2 Layers and where they live

| Layer | What it is | Main modules |
|---|---|---|
| Sources | real buses, or the simulated car with a profile-driven emitter | `vgpt/car/`, `vgpt/world/`, `vgpt/live/car.py` |
| Link | the one-way pipe, framing, recognition of what arrives | `vgpt/pipe/` |
| Decoding | protocol decoders and the per-vehicle profile router | `vgpt/protocols/`, `vgpt/vehicle/router.py`, `vgpt/ingest/` |
| Runtime | the one writer: decode, quality checks, events, trips, charging, context freezing | `vgpt/live/runtime.py`, `vgpt/store.py`, `vgpt/vehicle/lake.py`, `ledger.py` |
| Storage | tiered SQLite, lossless archive, sealed at rest | `vgpt/store.py`, `archive.py`, `atrest.py`, `runtime_vault.py` |
| Understanding | pre-parser, the frame model, confidence, discourse | `vgpt/meaning/` |
| Execution | typed query, planner, executors, cause and investigation engines | `vgpt/query_*.py`, `vgpt/meaning/*x.py`, `vgpt/cause/`, `vgpt/investigate/` |
| Answer | evidence, composer, verifier, privacy, role lens | `vgpt/evidence.py`, `answer_*.py`, `privacy.py`, `rolelens.py` |
| Interface | application window, pages, voice | `ui/`, `vgpt/shell/`, `vgpt/voice.py` |
| Scaling | probe, plan, autoscale, fleet hub | `vgpt/scale/`, `deploy/tiers.json` |

### 1.3 Hardware and software components (reference configuration)

| Item | Reference | Notes |
|---|---|---|
| Board | Raspberry Pi 4 Model B, 4 GB, Cortex-A72, used as 2 cores / under 2 GB | the case study allows Pi 4/5; the A72 has no dot-product instructions (important for speed) |
| OS | Raspberry Pi OS (Debian 13), Python 3.13, systemd user service | the assistant belongs on Linux; QNX and Android Automotive are data sources and displays |
| Runtime | Python, numpy, SQLite, ONNX Runtime (int8) or llama.cpp | optional: OpenCV, cryptography, psutil, FTS5 (each absence is tested) |
| Model | Qwen2.5-0.5B-Instruct (Q4_K_M base) + LoRA adapter `frame2r12`, merged and stored as int8 for ONNX Runtime | about 640 MB of weights |
| Speech | Vosk small English model, offline | 40 MB model |
| Memory | about 700 MB resident on the Pi; `MemoryMax=2G` in the unit | measured |
| Storage | SQLite per car, 6 GB budget by default (history of up to 5 years of minutes) | see [05](05_data_protocols_storage.md) |
| Simulated guest | Android 17 Automotive (Cuttlefish) under KVM on the Pi | data source only; 2.8 GB with the host, so the assistant runs with the guest stopped |

For a real car the proposed hardware is an automotive SoC or board with 2 or more ARM A5x/A7x cores and 2 GB, a gateway that forwards frames read-only (a hardware data diode is the strongest isolation), and flash sized by the retention. This part is **proposed, not built** (`docs/HARDWARE_REFERENCE.md`).

## 2. Vehicle data collection strategy

**The integration artefact is a profile pack** (the "decoder file"): `profile.json` (sources, signal bindings, event rules, thresholds, retention), the DBC files, a SOME/IP service catalogue and a DTC table. A binding says where a concept comes from (for example `can:pt/VehDyn.Speed`, `vhal:hal/PERF_VEHICLE_SPEED`) and how to convert it (unit, scale, offset, enum map, valid range, priority). `Profile.validate()` rejects unknown units, missing signals, duplicate priorities and retention below ten weeks.

**How Vantage learns a car it was never told about** (`docs/CAR_AND_VANTAGE.md`):

1. **Wire** (`pipe/wire.py`): a record is a start marker `A71C`, a length, a CRC-32, a microsecond time, a channel number and bytes. Nothing in it names a vehicle. A reader that joins mid-stream or meets damage skips to the next good record and counts what it skipped.
2. **Sniff** (`pipe/sniff.py`): every block is classified from its structure (CAN or CAN-FD frame, SOME/IP, VHAL parcel, camera picture, NMEA, JSON, log line). A channel is read the way most of its blocks read.
3. **Fingerprint** (`pipe/fingerprint.py`): the CAN identifiers, SOME/IP services and VHAL properties seen are scored against every installed decoder file. The best fit is used, with reasons; which wire is which bus is decided from the same identifiers. Measured scores 0.94 to 1.0 on our cars.
4. **Learn** (`ingest/autoprofile.py`, `ingest/mapper.py`): if no decoder file fits, what the standards allow (OBD-II, SAE J1939, Android VHAL, NMEA, named values) is read from the traffic and written down as a new decoder file. The **signal mapper** decides which quantity a signal is and **refuses rather than guesses** (on 831 reviewed signals of 76 real vehicle databases it makes no wrong mapping).
5. **Unidentified**: if even that finds nothing (a proprietary network), the link says so and lists the unexplained CAN identifiers. Adding a decoder file makes the car recognised at once, without a restart.
6. **Adapt**: the inventory keeps growing; a clearly better decoder file replaces the current one (re-evaluated every 30 seconds); two cars of one model are told apart by the VIN they announce.

**What the router adds beyond decoding** (all tested): redundant sources (a secondary takes over when the primary is stale; two fresh sources that disagree for 5 s record a `SENSOR_CONFLICT`), plausibility (an impossible value is dropped and counted, with a rate-limited `SENSOR_INVALID`), signal events declared in the profile, derived driving events (hard braking, acceleration, cornering), source health (a silent bus is written down with its recovery), CAN bus-health (unknown ids and rate spikes, learned from clean traffic) and signal-level anomaly monitors.

**Which kind of vehicle is this?** Decided by the data, never by a setting (`vehicle/caps.py`): a state of charge means battery-electric; a fuel level or engine speed means combustion (diesel if it reports diesel-only hardware such as particulate-filter soot or AdBlue; petrol if it reports spark-ignition signals such as lambda or fuel trims); both mean hybrid; neither means "energy source not reported". The page and the executors adapt to the result.

The case study's list of sources and how each is handled:

| Source | Handled by | Becomes |
|---|---|---|
| CAN / CAN-FD | DBC parser (Intel and Motorola, signed, scaled, multiplexed, float), ProfileRouter | canonical signals, error events |
| SOME/IP | binary notifications and service discovery with a catalogue | ADAS state, lead vehicle, lane quality, charging; events such as ECU reset, voltage dip, AEB, disengagement |
| Android Vehicle HAL | `VehiclePropValue` parcels, property ids from the real AOSP files | secondary signals, range, ignition, charge state |
| QNX / Android / Linux logs | slog2info, logcat, syslog (RFC 5424 and 3164) parsers | searchable log table, events matched by profile rules |
| UDS / OBD-II (diagnostics) | ISO-TP reassembly, 0x59 DTC reports, mode 01 PIDs | DTC set and clear events with the vehicle's own text |
| ADAS and camera | SOME/IP object and lane services; camera metadata or the luma plane analysed on the box | lane confidence, blockage, sharpness, brightness; only metadata and a 64x48 thumbnail at fault events are stored |
| GNSS | NMEA 0183 RMC and GGA with checksum | position rows, places, trips |
| Also | LIN, FlexRay, CANopen, J1939 (including multi-packet transport and DM1), DoIP | see [05](05_data_protocols_storage.md) |

**Driving scenarios and error sequences** (the case study's first key point) are recreated from simulated and recorded data, not CARLA: a virtual car with a drivable route, traffic lights and 13 injectable faults (coolant pump, tyre leak, weak 12 V, ECU reset, shutdown and others), a world builder that lives 105 days or 4 years with planted faults and a ground-truth file, and replay of recorded captures. CARLA was not used (stated in `docs/GAP_REPORT.md`; the owner was to be asked before adding it).

## 3. Historical data storage approach

| Tier | Content | Kept |
|---|---|---|
| Seconds (`sig_1s`, `lk_1s`) | every signal once a second while the car is awake (once a minute parked) | 14 days |
| Minutes (`sig_1m`, `lk_1m`, `cam_1m`, `ext_1m`) | duration-weighted mean, min, max | 5 years (1,830 days) |
| Daily (`agg_1d`) | per-day mean, max, min of every column | forever |
| Records | events, trouble codes, logs, trips, charging sessions, service history, the frozen context around events | 5 years |
| Lossless archive (`archive/*.vga`) | the expired seconds, exactly (codec v2, 0.4 bytes per sample), hash-chained | up to the size budget |

* Everything is **self-describing**: the database carries its own vehicle profile and signal catalogue, so the typed engine reads any signal of any car with no code change.
* A **size budget** (6 GB by default: the database plus the archive) is enforced nightly, removing the oldest archive files first.
* The archive is read only for questions that are exact over the seconds themselves (a highest speed, a time above a limit), only for windows up to 45 days, and loaded for that question into temporary tables; the database is never written by the assistant.
* A cold-minute archive was built and **reverted** because loading it cost 0.9 to 10 s per question. The owner's rule: not even 1 ms more latency for storage savings.

## 4. SLM / LLM selection rationale

Detailed in [06](06_ai_model_and_training.md). In short:

* **Model**: Qwen2.5-0.5B-Instruct (about 400 MB at 4-bit) plus a LoRA adapter. It was chosen for the budget (2 GB, two cores, under 2 s) after a long comparison against LFM2.5, Granite 4, xLAM-2, Qwen 1.5B, Qwen3-0.6B and extractors (GLiNER, Decide, DeBERTa NLI).
* **What it does**: it only *reads* a question. It writes an 8-word frame from a closed vocabulary. It never writes numbers, SQL, a command or the answer.
* **Why not larger**: memory and latency. A 1.5 B model plus runtime peaked near 2.14 GB, over the limit, and was far too slow per prompt token on the reference board.
* **Why int8 ONNX Runtime**: the Pi 4's Cortex-A72 has no dot-product instructions, so llama.cpp's quantised kernels reach only about 11 prompt tokens/s there; ONNX Runtime's int8 kernel reaches about 30. The A55 (the case study's own core) has dot-product, so llama.cpp may be fine there; this was never tested.

## 5. Search and retrieval workflow

Detailed in [04](04_question_journey.md). The shape is:

1. **Guard**: deterministic safety policy, text neutralisation, length and control-character limits.
2. **Pre-parser**: spelling and shorthand repair, and questions that need no model.
3. **Investigation check**: a long multi-part request with two or more analysis cues goes to one of **32 deterministic analyses** (incident reconstruction, change points, lead times, outlier trips, ADAS performance, cross-source discrepancies, co-occurrence versus chance, range-estimate drops, and more). They never call the model and keep recorded facts, associations and hypotheses apart.
4. **Frame**: the model reads the question; confidence is calibrated; low confidence is refused.
5. **Resolver**: the frame becomes a typed, validated query. The planner builds the only SQL in the system from fixed templates and whitelisted names, runs it read-only on the cheapest tier that is exact enough.
6. **Executors and explanation**: statistics, trends (yes or no verdict first), event rates, trips, charging, fuel economy, correlations, digests; "why" goes to the cause engine.
7. **Evidence and wording**: evidence object (claim, numbers, units, samples, tier, data quality, gaps); templates write the words; the verifier checks every number.

There is **no vector database and no free-text retrieval over the data**: the data is structured, so structured queries are exact and fast. Retrieval-augmented prompting was tried in phase 4 and rejected for latency.

## 6. Edge deployment architecture

* **Processes** (all local): the receiver and runtime (the only writer) and the assistant server (read-only handles, one model context). They meet only at the SQLite file. The virtual car is a development tool and is not part of the in-car footprint.
* **Resource tiers** (`deploy/tiers.json`, `VGPT_TIER`):

| Tier | Target | Model contexts | Sessions | SQLite cache | Seconds / minutes kept |
|---|---|---|---|---|---|
| `edge2g` (default) | 2 A5x cores, 2 GB | 1 | 2 | 4 MB | 7 d / 120 d (tier defaults; the product setting keeps 14 d / 5 y) |
| `pi5` | Raspberry Pi 5, 8 GB | 1 | 4 | 16 MB | 14 d / 180 d |
| `vm8` | validation VM, 8 vCPU, 16 GB | 3 | 12 | 64 MB (+512 MB mmap) | 30 d / 365 d |
| `hpc` | vehicle HPC, 12 cores, 32 GB | 3 | 32 | 256 MB (+2 GB mmap) | 60 d / 730 d |

* **Scaling** (`vgpt/scale/`): a probe reads the real cores, memory and instruction sets; a planner chooses the configuration with a reason for every number; a calibration step measures how many cores one context still gains from; an autoscaler adds a model context only if it **verifiably** raises throughput by at least 20 %. On a 2 to 4 core board, one wide context is best (the Pi 4 is memory-bandwidth bound: a second context lowered throughput); on many cores, worker processes scale. See [08](08_performance_and_scaling.md).
* **Deployment**: a per-user Windows installer (PyInstaller + Inno Setup, about 757 MB, no administrator), a Raspberry Pi systemd user service with `MemoryMax=2G`, and a hardened Linux unit (`deploy/linux/vgpt-assistant.service.in`: no network, address family AF_UNIX only, no swap). `docs/DEPLOYMENT.md`.
* **Car operating systems** (`docs/PLATFORMS.md`): the assistant runs on Linux (Raspberry Pi OS, Yocto, AWS Graviton, QEMU). QNX, Android Automotive and AUTOSAR Adaptive are covered as data sources and displays (QNX also as a possible host). Only Linux on the Pi and Android Automotive on Cuttlefish (as a data source) have been run.

## 7. User interaction design

* A **cockpit interface** (`ui/modern`): speed always visible, colour only for meaning, bundled fonts and icons (no web calls). Pages: **Ask** (natural language, follow-ups, evidence chips, Show your working), **Car** (instrument cluster, tyres, warning lamps that open a question), **Energy** (or **Fuel** for combustion cars), **Map**, **Camera**, **Settings** (test track, units).
* An **application window**, not a website: `vehiclegpt.py app [--kiosk]`, no address bar, a launch token that becomes a cookie, 7-inch layouts (800x480) supported.
* **Voice**: wake word "Vantage", offline recognition, "Vantage stop"; spoken answers use local voices only. The page captures audio; the browser's cloud speech API is deliberately not used.
* **Four roles** with different access and different readings of the same answer (driver, service engineer, fleet operator, validation team), with scrypt passcodes and a hash-chained audit trail.
* **Honesty in the UI**: it shows "I read it as...", the calibrated confidence, the data tier and data-quality notes, "no data" instead of zero, and a thumbs up/down that becomes a test case.

## 8. How the parts guarantee the case study's constraints

| Constraint | Mechanism | Evidence |
|---|---|---|
| No cloud | audit hook refuses every non-CAN socket and name lookup; service unit denies the network; nothing downloaded at run time | answered questions in an empty network namespace on the Pi; 0 refused attempts |
| ~2 GB, ~2 A5x | one model context, int8 weights, memory guard, `MemoryMax=2G` | 697 to 750 MB resident on the Pi |
| < 2 s common queries | closed-vocabulary frame (short decode), token mask, speculative drafts, no whole-vocabulary scan, daily tier | Pi median about 1.4 s; about a quarter of questions still above 2 s |
| Weeks of history | tiers, archive, budget | 5 years of minutes in 6 GB; 4-year worlds of 389 to 875 MB |
| Safety isolated | two domains, source scan, read-only handle, receive-only link, closed-output model | Pi real-time rig: 0 missed deadlines; 22 of 22 escapes refused |
