# Case Study 5 - what is satisfied, what is not, what is still required

Written 2026-10-20 after the plug-and-play platform, the 105-day datasets, the investigation engine, the live virtual car and the scalability work.  It supersedes `docs/REQUIREMENTS_EVIDENCE.md` wherever the two differ
(that file describes the earlier rule-based revision).  The case study PDF was read in full three times while writing this.

Status words: **MET** = runs here and a test or a measurement shows it.  **MET (simulated)** = runs end to end, but on generated or recorded traffic, not a vehicle.  **PARTIAL** = part of it.  **NOT MET** = not done.
"x86" always means this laptop; **no ARM core has been measured** (a QEMU / board run has not been approved).

## A. Constraints

| Constraint | Status | Evidence | What is missing |
|---|---|---|---|
| Works without cloud | **MET** | every component is local; the server binds 127.0.0.1 with a cross-site guard; no network call outside the local collectors | - |
| ~2 GB RAM, ~2 ARM A5x cores | **PARTIAL** | x86: assistant process 627 MB (one model context), runtime process alone 37 MB; pinned to two cores the assistant answers in 0.83 s median / 1.16 s p95 | **ARM not measured.** A5x cores are several times slower than this laptop; the board's latency and memory must be measured |
| Common queries under 2 s | **MET on x86, not on ARM** | single person: 0.75 s median, 0.91 s p95 (model + data); data side alone p50 25-47 ms, worst 376 ms on 105 days (`results/scaling`) | with **one** model context, four people asking at once get 2.4 s median (over budget); the `vm8` / `hpc` tiers add contexts (4.2 answers/s at 8 people, p50 1.7 s) - not available on the 2 GB tier |
| History searchable for several weeks | **MET** | per-minute history for 120 days (17 weeks) on the default tier, 105-day datasets tested end to end, events and frozen event context kept longer; `Profile.validate` refuses a pack below ten weeks | - |
| Safety-critical functions isolated from the AI | **PARTIAL** | collectors are listen-only (no send path), the assistant opens the database read-only, the safety guard runs before and independently of the model; tested on one host | isolation at **vehicle** level (separate partition / hypervisor, hardware data diode) is proposed, not built; no ASIL argument |

## B. "Key points your solution should address"

| Point | Status | Evidence | What is missing |
|---|---|---|---|
| Recreate driving scenarios and error-condition sequences | **MET (simulated)** | 105-day worlds from real driving traces + 8 physics stories + 11 injected fault families with ground truth; a virtual car with 13 injectable faults and a runtime that stores what its buses carry (`vgpt/live`, driven by `tools_live.py`; the head-unit page does not expose it: since 2026-10-20 the page replays the recorded drives of the chosen dataset by itself) | **CARLA is not used** (not installed; a ~20 GB download needing your approval); the virtual car stands for the case study's "other vehicle driving simulation" |
| Collect from CAN / CAN-FD | **MET (simulated)** | DBC decoding (Intel / Motorola, signed, multiplexed, FD) round-trip on three vehicles; candump / ASC / UDP / TCP collectors tested | **SocketCAN collector is untested** (needs a CAN interface); no capture from a real vehicle has been run through it |
| SOME/IP services | **MET (simulated)** | binary notifications + service discovery decoded through a catalogue; events and signals stored | no vsomeip / Ethernet-tap client; a gateway must forward frames |
| Android Vehicle HAL | **MET (simulated)** | `VehiclePropValue` with AOSP ids, areas, native units; used as the redundant source (conflicts detected) | no Binder client; a gateway must forward properties |
| QNX services / logs | **MET (simulated)** | slog2info, logcat, syslog parsed; searchable; profile rules turn lines into events | no live `slog2info` / `syslogd` hookup (UDP syslog collector exists) |
| ADAS | **MET (simulated)** | lane keeping state, disengagement reasons, lead vehicle, lane quality, AEB through SOME/IP; investigation templates for performance, coincidence with faults, flapping | no real ADAS feed |
| Camera feeds | **MET (simulated)** | feature stream (blockage, sharpness, brightness, lane confidence) and a luma-plane analyser; **pixels are never stored**; the live runtime uses a procedural camera | the live runtime path with a real camera was not exercised in this revision (the older dashcam rig still works) |
| Diagnostic logs | **MET (simulated)** | UDS/OBD-II over ISO-TP, DTC set / clear with the vehicle's own text, DTC cycles investigation | - |
| Real-time and historical retention | **MET** | single-writer runtime, tiers, daily compaction, frozen event context; measured throughput 32.9 k messages/s per process | - |

## C. Expected output headings

All seven are written in `docs/ARCHITECTURE.md`: system architecture (sections 0-1), data collection strategy (2), SLM rationale (4), search and retrieval workflow (5), historical storage (3), edge deployment (6),
user interaction design (7), plus simulation (8) and isolation / privacy (9).  Status: **written**; hardware components other than a laptop are proposed.

## D. Evaluation criteria

| Criterion | Status | Honest summary |
|---|---|---|
| System architecture, incl. logical integration of simulation and interface | **MET (simulated)** | the head-unit page replays the recorded drives of the chosen dataset by itself (a drive plan made from the trips table drives the clock; `tests/test_driveplan.py`); the virtual car's bus traffic goes through the real decoders into the real database and the same assistant answers about it (`tools_live.py`, `--live-car auto`; `tests/test_live.py` covers it over HTTP) |
| AI capability - understand queries | **PARTIAL** | exact-frame accuracy 93-94 % on frozen sets, 90 % on the case study's own questions, 80-95 % on adversarial sets (before today's vocabulary; see `docs/TRAINING_QUEUE.md` for the retrained adapter).  Follow-ups ("and the week before that") work through a rule layer, not a trained one.  The owner's own transcript found real failures (lost context, inconsistent tyre-pressure answers, same answer for different phrasing); each is fixed and tested, but unseen phrasing will still fail sometimes |
| AI capability - accuracy of responses | **PARTIAL** | arithmetic and windowing are checked against plain SQL / numpy; investigations are checked against injected ground truth; no human-rated sample of free-form answers beyond the owner's sessions |
| Data integration - multiple protocols and sources | **MET (simulated)** | section B; three vehicles that differ in buses, names, units, byte order and powertrain are plug-and-play from a data pack; auto-mapper drafts a pack from DBCs with precision 1.0 and recall 0.75-0.9 (a human reviews the rest) |
| Search - historical events, trends, anomalies | **MET** | typed operators (events, counts, trends, change points, extremes, comparisons), 32 investigation analyses, and (added 2026-10-20) **event frequency as a series**: is X getting more frequent (exact binomial test on the two halves, or the week with the largest rise), this week versus last, highway versus city (counts AND rates per driving hour), the day / week with the most, on how many days; text-log search by words and source; the state of the diagnostic codes (active, cleared, returned); the reporting unit of an event; where a measurement comes from; the charging ledger.  Each is recomputed with plain SQL in `tests/test_eventx.py` and `tests/test_records.py` |
| Search - explain causes | **PARTIAL** | explanations are *associations* with recorded facts, associations and hypotheses kept apart, and the missing measurement named; causes are never claimed.  Only the 32 templated request shapes are investigated; a novel shape falls back to the single-frame path |
| Edge performance - memory, latency | **MET on x86; ARM NOT MEASURED** | section A |
| Edge performance - scalability reference board -> HPC | **PARTIAL** | tiers (`deploy/tiers.json`) change only numbers; measured on x86: ingest linear in processes up to the physical cores, query latency flat in history length, concurrency 2x with three model contexts.  Never run on a Raspberry Pi, an ARM VM or an HPC.  A different (larger) base model needs its own adapter |
| Privacy - local-only processing | **MET** | section A |
| Privacy - protection of vehicle and driver data | **PARTIAL (much improved 2026-10-20)** | local-only store, read-only assistant, injection-safe text, location stays on the device; **roles are access control** (driver / service engineer / fleet operator / validation) with salted-scrypt passcodes and lock-out, server-side scrubbing of coordinates, places and camera pictures for the three non-driver roles, a hash-chained audit log that never records content, signed and optionally AES-256-GCM-encrypted exports, erase of location and camera data, a live on-machine posture check (loopback only, no outbound connection, read-only database), an application window that only its own launcher can talk to, and voice input that is recognised on-device and keeps nothing it was not addressed with (`docs/ARCHITECTURE.md` 8d, 8e; `tests/test_roles_privacy.py`, `test_appshell.py`, `test_voice.py`).  **Still no encryption of the database at rest and no hardware-backed key store; no ASIL argument** |
| Practicality - ease of deployment in production SDVs | **PARTIAL** | integration = a data pack + a collector next to the bytes (`tools_live.py ingest`); no vehicle trial; the release builder (`tools_release.py`) was not re-run for this revision |
| Value for drivers / service engineers / validation teams | **MET (simulated)** | driver questions; a **Service report** (each component's verdict with its evidence, "no data" never shown as fine) and the investigations (shutdown reconstruction, ECU-reset coincidence, source conflicts, intermittent-fault conditions) for the service engineer; a **Validation** page (source silences, conflicts, event counts, answer timing, pipeline status, audit chain) with fault injection and recorded-drive replay - on simulated data |
| Value for fleet operators | **MET (simulated)** | a Fleet page for every vehicle database in the folder (use, trips, energy, warnings; no positions, no camera pictures, nobody's name), reachable only with the fleet passcode; role questions (use, risk).  Built on recorded and generated vehicles, not a real fleet |

## E. Known defects

* **13 tests of the older rule-based engine fail** (`tests/test_qa.py` 2, `test_qa2.py` 4, `test_round52_fixture.py` 2, `test_routing.py` 5).  They fail identically with the meaning front end switched off; one cause is
  `vgpt/semantic_ops.py` (edited 2026-10-18) answering "what was the CAN ID responsible for the battery overheating event" as a thermal question instead of refusing.  They test the fallback engine
  (`VGPT_MEANING=0`), not the default path.  Fix or retire.
* The interim cue lists (`vgpt/meaning/discourse.json`, `vgpt/investigate/lexicon.json`) and the vehicle-vocabulary layer are rules; the rare-word arbitration is retired (off by default).  `docs/MODEL_ONLY.md` measures each against the model alone; replacing them with trained frames is queued (`docs/TRAINING_QUEUE.md`).
* `README.md`, `DESIGN.md` and `docs/REQUIREMENTS_EVIDENCE.md` still describe the earlier revision in places; `docs/ARCHITECTURE.md` and this file are current.
* The 2 GB tier can hold one model context; a second person's question waits for the first.

## F. What is still required, in the order that matters for defending the claims

1. **Measure on ARM.**  A QEMU Cortex-A5x/A53 guest or a board: assistant memory, p50 / p95 latency, runtime CPU.  (`tools_scaling.py` is ready to run there.)  *Needs your approval for QEMU.*
2. **Run a real capture through the platform.**  One `candump` / ASC log plus the DBC of any production vehicle -> `vgpt/vehicle/onboard.py` -> pack -> `tools_live.py ingest` -> ask questions.  This is the only
   thing that turns "plug and play" from a measured simulation into a demonstrated integration.  *Needs a capture from someone who has one.*
3. **Protect the data**: encryption at rest (SQLCipher or file-system level).  *(Roles and authentication were added 2026-10-20: see D, privacy.)*
4. **Fleet view on real fleets**: the Fleet page reads every database in the folder; a deployment that collects from many vehicles needs an uplink the case study forbids by default - the design choice (an export the operator carries, or a local gateway) is open.
5. **CARLA bridge** (optional, the case study names it as an example): a small adapter from CARLA's vehicle state to the canonical state the emitter takes.  *Needs your approval to download CARLA.*
6. **Train the interim rules into the model** (`docs/TRAINING_QUEUE.md`): the platform vocabulary was retrained on 2026-10-20 (`full_v7`, accepted only if it passes a gate fixed beforehand - result in that file); still rule-based: follow-up completion, long-request routing, rare-word arbitration, log-search stop words.  A multi-slot investigation frame and a continuation frame come next, then every frozen set is re-run.
7. **Fix or retire the 13 legacy tests**; refresh README / DESIGN; re-run `tools_release.py` and the self-test.
8. **Vehicle-level isolation design**: partition / hypervisor boundary, a hardware diode on the gateway, and a written safety argument.
9. **User study** with drivers and service engineers (none has been run).

## G. Test status

See the end of `docs/TRAINING_QUEUE.md` for the last full-suite result.  New suites of this revision: `tests/test_investigate.py` (16), `tests/test_live.py` (34), `tests/test_scaling.py` (15), plus
`test_followups.py`, `test_vehicle_platform.py`, `test_world.py`, `test_onboard.py`.

## H. Added 2026-10-20 (the data pipeline, roles, the application window, voice)

* The application receives its data as a vehicle would (a recording is a source; ingest stores it; the last ~20 s is read live, older data from the dataset) - `docs/ARCHITECTURE.md` 8c.  Fidelity against the finished recording: 83.1 % identical answers (84.2 % same content) over 1,075 frozen questions.
* It is an application window (`vehiclegpt.py app --kiosk`, Raspberry Pi files in `deploy/raspberrypi/`), **never run on Pi hardware**.
* Voice input with an on-device recogniser (Vosk, small English model, installed 2026-10-20 with approval) is built and tested: stand-in, and the real recogniser on this PC's synthesised speech (see `docs/ARCHITECTURE.md` 8e).  **Not measured: real speakers, cabin noise, ARM.**
* Time speed-up exists only as a test-track slider; real time is the default.

## I. Added 2026-10-21 (the car <-> Vantage test on real hardware)

* The assistant now runs on the Raspberry Pi 4 (Vantage) and is fed by the laptop over a signed link: `docs/VANTAGE_LINK.md`.  Memory (kernel limit 2 GB, peak 978 MB) and CPU (2 cores) are inside the budget.
* **Answer time on the Pi is not inside the case study's 2 s for a wording the model has not seen**: 3.7-6.2 s (the common and repeated questions take 0.1-0.5 s).  Cause measured: the output grammar costs 1.9 s of it and the adapter is not merged (about 1.5 s reachable without touching the weights).  Not done.
* The link is signed but not encrypted (plain HTTP on the Pi's own network).
