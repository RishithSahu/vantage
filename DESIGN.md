# VANTAGE (package `vgpt`) - Design document (Case Study 5: Natural Language Vehicle Information using On-Device SLM/LLM)

Structured to the case study's **Expected Output** list, in its order, with the engineering justification it asks for ("AI model, OS, architecture choices, and communication protocols"). This document describes the **current** system. To keep it from going stale it does not repeat the accuracy, latency and test figures of the current revision: those are in `VANTAGE_STATUS.md` (one source revision, with hashes) and in the requirement-to-evidence table `docs/REQUIREMENTS_EVIDENCE.md`. Section 9 lists what is not met or not measured.

Status words: **Implemented** = code that runs and is covered by a test or a measurement in this repository; **Replayed** = native wire format decoded, but the traffic is recorded or generated; **Proposed** = design only; **Open** = required and not demonstrated.

Platform used so far: an x86 laptop, two pinned cores (`tools_pin.py`). **No measurement on Cortex-A5x hardware exists.** An ARM64 QEMU harness (`vm/driver.py`: `-M virt -cpu cortex-a55 -smp 2 -m 2048`, Debian 12 arm64) was built; QEMU runs are deferred at the owner's request, and anything it produces is an *emulation* timing. Figures marked *historical* below were measured in that guest for the earlier VehicleGPT 1.1 configuration (Qwen2.5-0.5B slot filler, 3-word output); they are kept because they are the only ARM-emulation data that exist, and they do not describe the current configuration.

---
## 1. High-level system architecture
Diagram: `docs/architecture.svg` (data path); the question path is in section 4.

```
Vehicle networks (read-only taps)            AI partition (QM-level, 2 cores, ~0.7 GB target)                          Users
CAN / CAN-FD ──┐                              ┌ ingestd (only writer) ─▶ SQLite tiers ◀─(read-only)─ assistant ┐
SOME/IP (+SD) ─┤  one-way gateway             │   adapters: DBC, ISO-TP/UDS, OBD-II,        interpret ▶ policy ▶ │  driver, service engineer,
Android VHAL ──┼─────────────────────────────▶│   SOME/IP, VHAL, logs, NMEA;                 plan ▶ execute ▶      │  fleet operator, validation team
QNX/Android logs┤                             └   bus-health monitors                       verify ▶ compose ────┘
UDS diagnostics┘
```
* **Hardware target**: 2 x ARM Cortex-A5x class cores, 2 GB RAM, flash storage, no GPU/NPU required. *Not measured on such hardware.*
* **Software**: Python 3.11 + SQLite + numpy; an optional llama.cpp model for the experimental configurations only; no cloud client.
* **Two processes** in the AI partition: `ingestd` (decodes buses, the *only* writer) and the assistant (opens the database **read-only**).
* **Product configuration** (`vgpt/product.py::make_brain`, used by `ui/server.py` and `app.py`): rule parser + clarification-first policy, **no language model on the answer path**, frozen selector off, typed-query pipeline for every data question. A second configuration with a model is a measured comparator, not the product (section 3).
* **Safety isolation - two different claims, kept apart.**
  *Tested (software boundary on one host):* the assistant opens the store `mode=ro` + `PRAGMA query_only`; SQL is compiled only from a whitelist of tables/columns with bound parameters and each statement is re-checked read-only (`tests/test_qc.py`); the adapters are read-only consumers with no transmit path; control requests (disable, set, unlock ...) are refused by a gate before any interpretation (`tests/test_product.py`, `tests/test_qa.py`); text stored in logs is treated as data and neutralised before it can reach an answer (injection test).
  *Proposed (vehicle level, not built or tested):* running the AI partition as a separate hypervisor guest or SoC domain, a hardware one-way gateway or data diode between the vehicle buses and the partition, and ASIL-rated partitioning of the safety functions. None of this exists in the repository.
* **Simulation integration** (the document's "logical integration of simulation and interface to user"): a seeded 28-day EV simulator emits **real wire formats** (section 2) into the same adapters a real vehicle would feed; the user interface is identical either way.

## 2. Vehicle data collection strategy
The document lists: CAN / CAN-FD, SOME/IP services, Android Vehicle HAL, QNX services/logs, ADAS, camera feeds, diagnostic logs.

| Source | Decoder (`vgpt/protocols/`) | Validated against |
|---|---|---|
| CAN / CAN-FD | `dbc.py` (Intel/Motorola, signed, scaled, float, multiplexed, value tables), `canframes.py` (Vector ASC, SocketCAN text + binary, FD DLC table up to 64 B) | `cantools`: **0 signal mismatches on 187,373 real ROAD frames (105 messages)**; `python-can`: identical on two real CAN-FD ASC files (64-byte payloads) |
| Diagnostic logs / DTCs | `isotp.py` (single/first/consecutive frames, SN wrap, loss detection), `uds.py` (0x59 ReadDTCInformation, 0x62, 0x7F, SAE J2012 DTC text), `obd.py` (SAE J1979 mode 01) | `can-isotp`: **300/300** random payloads (1-4,095 B) reassembled identically; `udsoncan`: every DTC id/status matches; **270,005 real OBD samples** (VIF-AEGIS) round-trip through raw J1979 bytes, 0 errors |
| SOME/IP services | `someip.py` (16-byte header, message types, return codes, stream splitting, SOME/IP-SD offers/subscribes + IPv4 endpoint options, service catalog `services.json`) | `someipy`: **2,000/2,000** random messages identical in both directions; SD vectors written from the AUTOSAR layout. *No public real SOME/IP capture was found.* |
| Android Vehicle HAL | `vhal.py` + `vhal_props.json` **generated from the real AOSP `VehicleProperty.aidl`** (280 properties, ids/types/units/enums computed, not typed) | ids spot-checked (`PERF_VEHICLE_SPEED = 0x11600207`); units applied from the AIDL `@unit` tags |
| QNX services / logs | `logs.py`: slog2info text, Android logcat, syslog RFC 3164/5424, severity mapping, WARN-and-above retention | **Android 2,000/2,000 and Linux 1,999/2,000 real log lines parsed** (Loghub). *slog2info layout is taken from QNX documentation, not from a real QNX capture.* |
| ADAS | SOME/IP `AdasState` events (disengagement reason + speed) and CAN | catalog + round-trip tests |
| GNSS / GPS | `nmea.py` (NMEA 0183 RMC + GGA, checksums verified, invalid fixes dropped) -> `NmeaAdapter` -> `gps_1s` (1 Hz fixes while the car is awake, 1/min parked, 7 days) + a per-minute position kept for the whole history | `pynmea2`: identical on 300 random sentences; **real GPS fixes** from a VIF-AEGIS drive decode exactly and form a track of sensible length; live ingest test (1 row/s moving, 1/min parked, garbage rejected) |
| Camera feeds | `camera.py` (numpy only, no OpenCV in the car): luma plane at 5 fps -> per-region texture + large-scale structure vs a slowly adapting baseline, judged relative to brightness; states OK / BLOCKED / DEGRADED (+ low-light info); 1.5 s debounce | **Real dashcam clip (comma2k19) with injected faults: frame accuracy 97.5%, 0 fault events on the untouched clip, time-to-detect 1.6 s (cover) / 2.8 s (mud) / 1.6 s (blur, glare); tunnel/dusk darkening is correctly NOT a fault.** ~4 ms/frame on a laptop core, 1.6 MB Python heap. Synthetic-scene unit tests in `tests/test_camera.py` |

**Generated vehicle histories** (`tools_datagen.py`, `vgpt/datagen*.py`, `vgpt/datasets.py`): the single scripted story was replaced by a seeded, profile-driven generator so the data is varied and realistic:
real **hourly weather** (Open-Meteo archive: temperature, rain, cloud, sunlight -> ambient temperature, rain-sensor/wiper state, ambient light, HVAC load, glare, rainy-day behaviour), real **driving**
(VIF-AEGIS: 31 trips with 24 Hz accelerometer, 12.5 h; EPA UDDS/HWFET/US06/SC03/FTP dynamometer schedules -> speed traces, hard braking / acceleration / cornering from the real accelerometer),
vehicle physics (mass, drag, rolling resistance vs tyre pressure and temperature, regeneration, heat-pump HVAC, pack thermal model, charging curves vs temperature), 5 driver profiles x 4 vehicles x 8 cities,
places (home, work, shops) with trips between them, and randomly chosen faults (cooling-pump degradation, slow tyre leak, weak 12 V battery, dirty windscreen, roadworks lane loss, roof box, a harsh-driving week).
In a real car the extra signals arrive as: tyre pressure <- VHAL `TIRE_PRESSURE` (kPa per wheel; real AOSP id 0x17600309); rain <- rain sensor / `WINDSHIELD_WIPERS_STATE`; ambient light <- light sensor;
odometer <- `PERF_ODOMETER`; brake wear <- `BRAKE_PAD_WEAR_PERCENTAGE`; lane keeping <- `LANE_KEEP_ASSIST_STATE`; position <- GNSS; DTCs <- UDS. **In this prototype they are bulk-loaded as history
(like the legacy simulator); the legacy signals, camera, logs and now GNSS (a continuous NMEA stream) have live adapters** (see section 9). Ground truth for every generated vehicle is written next to it (`data/gen_*_truth.json`) from the generator's arrays,
independently of the in-car analytics, and `tests/test_datagen.py` checks that the assistant's answers equal it (km, hard-braking counts, rain minutes, charged kWh, odometer, service km, places per trip, tyre warnings, health vs planted fault).

**Retention and bounded cost**: unknown ids and malformed frames are counted, never fatal; log volume is cut to WARN+; high-rate frames are reduced to 1 Hz rows.
Throughput inside the ARM guest: **502 msg/s real time at 16% of one emulated core; ~8,100 msg/s unpaced** (`results/guest_q05_*.json`).

**Recreating scenarios and fault sequences** ("CARLA or other ... simulated/recorded data"): scripted EV simulator (`vgpt/sim.py`, `livegen.py`) with planted stories - cold weather
+ more highway driving, a roof-box drag step, a cooling-pump failure during a DC fast charge -> DTC -> over-temperature warning, ADAS drop-outs on roadworks and rain days.
Speed traces are **real recorded driving cycles** (VIF-AEGIS OBD, 30 trips). CARLA is not used; the adapters would accept a CARLA bridge unchanged.

## 3. SLM/LLM selection rationale
**Current position, stated plainly: the product configuration contains no language model on its answer path, and no measurement in this repository shows a small model earning its place.** The case study asks for an on-device SLM/LLM; the deterministic configuration is a *control* that answers the system requirements very cheaply, not a completed VehicleGPT. What was tried, and what each result allows us to say (numbers: `VANTAGE_STATUS.md` sections 4-5):

| Model role | Status | What was measured | Constraint |
|---|---|---|---|
| None (rules + generator + clarification policy) | **Product configuration** | the reference: lowest latency and memory | no SLM is used |
| Frozen LFM2.5-350M (Q4_K_M, 229 MB) choosing among deterministic candidate readings (`vgpt/candidate_planner.py`) | **Experimental, disabled** (`selector=None`) | pre-registered criteria; passed on an earlier revision, **failed** against the stronger no-model baseline (fewer directly correct answers; fewer correct-or-clarified) | frozen: no tuning after the verdict |
| Frozen LFM2.5-1.2B (Q4_K_M, 731 MB) + retrieval writing the whole plan (route `slm` / `hybrid`, section 4d) | **Built and measured; recommended only as the hybrid** | dev 41 / final 73 correct alone, 80 / 142 in the hybrid (rules product 73 / 132); about 4 s per model question on 2 laptop cores, never under 2 s; whole simulated target peaks at 1,792 of 2,048 MB | no training; Granite 4.0-1B not run (not approved) |
| Qwen2.5-0.5B + LoRA, whole-query generation (the earlier VehicleGPT 1.1 model) | **Experimental comparator** | modestly more directly correct answers than the no-model policy on the available sets (not significant), at equal wrong-answer counts; model path above 2 s at p95 on the laptop | its interface does not receive conversation history, so it cannot be fairly compared on follow-ups; no new training was done |

Why a model remains worth asking about: unfamiliar phrasing is where the rules lose most (clarifications that contain no intended reading are the largest failure category). A model that is *trained* to choose among generator candidates or to fill only unresolved fields is the untested next step; it was not attempted because no new training was allowed. Loading a model while all useful work stays deterministic would be a weak answer to the case study, and this document does not claim it.

Model-selection reasoning that still holds (historical, host-side sweep, `results/q15_*.json`): a 1.5B model needed 1.9-2.4 GB resident next to a 700 MB OS, cost twice the latency, and was no more accurate than 0.5B on the unseen question set; sub-0.5B models are the only class that fits a shared 2 GB system. The model never writes answers: all numbers come from SQL and deterministic analytics, and a template composer writes the text; a verifier rejects numbers, units, signals or causal wording the evidence does not support.

Engineering that applies to any model configuration (each measured when introduced): `OPENBLAS_NUM_THREADS=1` -354 MB at import; `n_batch=32` and no python-side state cache -55 MB and removed a crash; `mmap` -435 MB commit; lazy load + idle unload; persisted prompt state (keyed on model size, prompt hash, n_ctx and llama.cpp version). *Historical (ARM emulation, VehicleGPT 1.1):* boot-to-ready 393-697 s -> 14 s with the persisted state. The candidate configurations use an 82-token prompt instead of the 3,696-token whole-schema prompt, which is why the state-build stall under QEMU does not carry over; that has not been measured on ARM.

Language robustness without a model (`vgpt/nlu.py`, `vgpt/vocab_parts.py`, `vgpt/schema_overlay.py`): typo repair by Damerau-Levenshtein against the assistant's own vocabulary (second attempt only), synonym mapping, richer time windows, and a lexicon that is rebuilt from the schema overlay when signals are added.

### 3a. Update 2026-10-18 - understanding without per-question rules (the product default since 2026-10-19; `VGPT_MEANING=0` selects the previous front end)

The language model's job is now one thing: read a question into a six-word **meaning frame** (`act concept op cond vs ent`) from a closed vocabulary generated from the vehicle ontology; glossary hints (word meanings, one JSON line each) are added to the prompt only when a rare word occurs; a typed resolver executes the frame on the existing deterministic engine. Code parses values (periods, numbers, limits) but never decides meaning (enforced by `tests/test_meaning.py::RuleBudget`). On three never-patched question sets the answer is correct 91.7 % of the time versus 51.0 % for the rule-based front end; latency 1.1 s p50 / 1.7 s p95 on two pinned x86 cores (ARM not measured). Full method, evidence, the PDF clause map and the open items: `docs/semantic_method.md`.

## 4. Search and retrieval workflow (current)
```
question ─▶ command gate (read-only: "disable / set / unlock ..." is refused, never clarified)
         ─▶ fuel gate (EV records no fuel: say so, offer the electric equivalent, "neither of these" path)
         ─▶ rule parser (nl2ir, ~1 ms) ─▶ typed Query IR + parse provenance
         ─▶ interpretation contract: every field has a provenance (question / parts of a phrase / cue / prior turn / default / vague / removed / ambiguous / ungrounded);
              SUPPORT check (is it backed by the words or the conversation?) and COVERAGE check (is every explicit constraint represented?)
         ─▶ clarification-first policy ──▶ ANSWER   (supported + covered)
                    │                    ├▶ CLARIFY  (concrete readings, always with "neither of these"; reply "2", "the first", "neither")
                    │                    └▶ REFUSE   (no executable reading, command, out of scope)
                    └ candidates: the deterministic generator (candidates.py) always proposes readings; a reading that did not come from a confident, fully supported
                      parse is CONFIRMED with the person, never answered on structural checks alone
         ─▶ validator ─▶ capability engine (is this signal recorded / derivable on THIS vehicle?) ─▶ whitelisted SQL planner ─▶ executor
         ─▶ evidence contract (resolution, completeness, coverage, class: observed / derived / estimated / associated) ─▶ composer ─▶ answer verifier ─▶ answer + evidence
```
Questions that are not typed data questions (greetings, help, units, glossary, safety refusals, **present-state status and health summaries** such as "how's my battery" or "is the camera blocked right now") are answered by the older engines (section 4c). **Those routes are not claimed to carry the typed checks.** `bench/route_audit.py` runs the case study's five questions verbatim plus representative failures through `make_brain` and records, per question, the route, the interpreted query, the evidence class, the verifier status and the displayed answer (`results/route_audit.md`); all ten PDF-question runs (five questions x two vehicle histories) leave through the typed pipeline.

Retrieval is **structured + lexical, not embedding-based**: SQL over tiers, FTS5 over logs, per-signal z-scores against each signal's own baseline in the same driving mode. No embedding model is loaded.

### 4a. Operations
`aggregate`, `extreme`, `current_value`, `compare` (two scopes), `count`, `list_events` (paged; the concatenated pages equal the SQL list), `event_context` (an explicit interval before/after an event; the event itself is the anchor and is excluded from the figures), `rank`, `trend`, `detect_change` (exploratory change points, p-values adjusted for the search), `explain_change` (distance-weighted trip decomposition so the explained quantity equals the headline; an unrecorded cause stays in the "unexplained" part), `explain_event`, `correlation`. Explanations are *associations*, worded "preceded", "associated with", "likely contributor" - the verifier rejects "caused".

### 4b. The query compiler and its contracts
**Problem it solves.** Routing a question to one of ~20 hand-written intents does not scale, and anything outside the routes is answered by the nearest known question - including questions about signals the vehicle does not record.

1. **Compile** to a typed **Query IR** (`vgpt/query_ir.py`): operation, targets, statistic, scope (time x entity selection x conditions), comparison groups, event reference, unit - registry names, numbers, ISO dates; never code or SQL. Rules parse first (`nl2ir.py`). A model may *propose* in the experimental configurations; its proposal passes the same contract.
2. **Interpret and decide** (`interpretation.py`, `candidates.py`, `policy.py`, `paraphrase.py`, `convstate.py`): described in the diagram above. Conversation state records what each turn changed and inherited (`Trace`); accepted state changes only on **answered** turns, so a clarification or a refusal never erases the user's context. A reading is rendered back to the person in plain English (`paraphrase.describe`) so they can confirm it.
3. **Validate and plan** (`query_validator.py`, `capability.py`, `query_planner.py`): vocabulary and structure, then what *this* vehicle can answer - unknown vs known-but-not-recorded vs not-derivable, with exact reasons; no silent electrical-for-mechanical substitution. A trusted SQL compiler (whitelisted tables / columns, bound parameters, `SELECT` only, read-only connection), tier selection (1 Hz for short windows, per-minute for the rest), derived metrics with declared dependencies, sensor-quality checks.
4. **Execute, assess, explain** (`query_executor.py`, `contracts.py`, `evidence.py`, `answer_composer.py`, `answer_verifier.py`): the interval is enforced exactly before aggregation; coverage is judged against days with recorded trips or charges; the evidence contract states resolution, completeness and class; the composer fills templates; the verifier refuses numbers, units, signals or causal wording the evidence does not support, and falls back to an evidence-only answer (status `unverified`) if the wording fails.
5. **Schema transfer** (`schema_overlay.py`): another vehicle's storage (different columns, units with scale and offset, missing wheels, aggregate-only pressure, registered-but-empty or intermittent signals, a brand-new signal, an opaque identifier known only by description) is described by an overlay that extends the SQL whitelist and rebuilds the lexicon. **Not covered:** a vehicle whose *buses*, not storage, differ; identifiers of signals the older engines hard-code.

**Design intent.** The model's job, when there is one, is to choose among enumerated readings; it never sees data and never writes numbers (no hallucinated values); the security boundary is structural (validator + SQL compiler + read-only connection); each stage is unit-testable; new signals need a registry entry, not a new route. **What this does not give:** language understanding that generalises - the rule parser's coverage of unfamiliar phrasing is the weakest part (`VANTAGE_STATUS.md` sections 4 and 10).

Policy trade-off: stricter confirmation of generator readings **lowers directly answered questions and may lower measured coverage**; whether it improves or worsens accuracy is *measured* on the evaluation sets (never assumed), and the full baseline was regenerated after the change.

Details: `docs/query_compiler.md`, `docs/signal_ontology.md`, `docs/evaluation.md`.

### 4d. The product: model-only understanding (rebuilt 7 Oct 2026; supersedes the earlier two-stage / hybrid attempt)
> **Superseded.**  This section describes the planner design of 7 Oct 2026.  The product today reads a meaning frame and is an honest hybrid: `docs/BOUNDARY.md` says which stage is learned and which is code.
```
question + conversation state
   -> capability retrieval        FTS5 + bge-small embeddings over capability descriptions (signals, events, tools)       ~5 ms
   -> ONE frozen-model tool call  operation + arguments together; the model sees today's date, the capability CARDS (id, kind, unit, what it MEANS, availability, relationships),
                                  the previous question and the plan it resolved to (as an earlier chat turn), any open clarification, and the message
   -> structural validation       well-formed, registered identifiers, type / unit compatibility, required arguments, valid state references, supported calculation; an exception = failure (fails closed)
   -> faithful compilation        explicit state edits (add / remove / replace), nothing the plan states is changed, only omitted fields get documented, DISCLOSED defaults; fidelity check on the finished query
   -> typed pipeline              validator -> capability engine -> SQL-whitelisted planner -> executor -> evidence contract -> composer -> answer verifier  (or a domain tool with an explicit evidence statement)
```
There is no rule parser, no rule-generated alternative, no lexical interpretation check and no hybrid on this path (a test asserts that `nl2ir.parse`, `policy.decide` and `interpretation.interpret` are never called), and the route is never changed silently: when the model cannot form a valid plan the outcome is an explicit failure or one specific question. The model writes a plan, never numbers or answers. The earlier rule-based implementation is untouched and reachable only as `product.make_legacy_brain` (the measured baseline).
* **Capability cards, not identifiers** (`capreg.card_text`): an opaque signal is only as useful as the description registered with it; unavailable-but-known concepts are shown as NOT RECORDED so the model can name them and the answer can explain. If no registered concept means what is asked, the model calls `unresolved(what=...)`: it is never told to choose "the nearest signal". Cards show only identifiers a call may contain (a ranking-metric or tool-id card whose text the grammar forbids pushed the model onto the wrong signal).
* **Plan language** (`toolspec.py`, 22 functions): operation and arguments in one call; periods are structural (relative amount + unit, calendar name, date), event windows are amount + unit, thresholds are {signal, op, value, unit}; `turn` states whether the message is new / follow_up / correction / clarification_reply; `conditions_edit` (add / remove / replace) and `drop` are explicit; `ask_user`, `unresolved`, `decline`, `say` are first-class outcomes.
* **Grammar** (`plangrammar.py`): types and registered identifiers only (every signal, event, condition, tool - including known-but-unrecorded). No gate depends on the words of the question, no shortlist of functions; a call can refer to conversation state only if a state exists.
* **Each model in its own documented tool-call form** (`dialects.py`): LFM2.5 - tools listed in the system turn, Pythonic call between its call markers; Granite 4 - OpenAI-style schemas, JSON call inside `<tool_call>`.
* **Prompt** (`prompt.py`): a compact instruction, brief operation definitions (the tool descriptions), a closed list of named conditions, and ONE short example conversation (about 20 turns, one per kind of call and state edit) as real chat turns - the call format and argument forms, not a bank of questions; a test checks they are not within 0.93 embedding similarity of any development / final / diagnostic question. The static part (instruction + declarations + examples) is evaluated once and kept as ONE saved model state keyed by runtime version, model identity, context and batch settings and the exact prompt text; per message only the suffix (cards + conversation turn + message) is evaluated.
* **Existence answers**: "Yes" on a valid positive observation; "No recorded occurrence" only when the data contract says the period's records are complete; otherwise "Cannot establish absence", saying why.
* **Outcomes** (separate, never conflated): answered | no_data | missing_data | unsupported_calculation | invalid_plan | ambiguous | unresolved_target | refused | chat | failed_plan | tool | execution_error.
Code: `vgpt/capreg.py`, `retrieval.py`, `toolspec.py`, `dialects.py`, `plangrammar.py`, `prompt.py`, `modelplan.py`, `plancompile.py`, `modelbrain.py`, `slm_backend.py`. Evaluation and measurements: `bench/eval_model.py`, `bench/retrieval_recall.py`, `bench/diag_function.py`, `bench/perf_model.py`; results in VANTAGE_STATUS.md section 13.

### 4c. Older engines (still on the product route for non-typed intents)
*Text inherited from VehicleGPT 1.1: it describes the older pipeline, whose routing rules are kept for the intents that are not typed data questions. The case study's five questions and every statistic / event / comparison question on the product route use the typed pipeline above.*

Tools: why-range (attribution to heating load, highway share, a step day), largest change, trend,
usage (energy, distance and driving time in a period), summary (weather/temperature), city-vs-highway, estimated range, events (+ charged kWh), **before-event** (z-scores + frozen context +
first log line + expert knowledge-base causes), CAN-bus incidents. *Explaining causes* is rule-based expert knowledge (`analytics.KB`), small and auditable.
**Extended domains** (`vgpt/analytics_x.py`, rule-routed; the SLM's metric vocabulary has matching words so paraphrases can still reach them): **tyres** (per-wheel pressure vs spec, TPMS warning state, slow-leak
trend by regression on each day's start-up reading, days to warning), **weather** (rain minutes / days, live rain and light class from the rain and ambient-light sensors; "cloudy" is inferred by comparing measured light
with a clear-sky model for the car's position and time), **service** (odometer, km and months to the next service, brake-pad life, 12 V battery, battery state of health, software version), **health** (one verdict
across faults, coolant-flow trend, tyres, 12 V, pads, camera; answers "is my battery good?"), **driving** (hard braking / acceleration / cornering per 100 km against real-world typical rates, regeneration share, lane-assist
share of motorway time; the assistant does not invent a driver score), **places / GPS** (the live fix: where the car is, heading and speed; distance, direction and time to any saved place from the current position, using the car's own earlier trips for the time; trips between places; a clearly-labelled habit-and-heading guess for "where am I going?"; the assistant knows only where the CAR is, never where the user is).

**Analysis engine** (`vgpt/qa_core.py`, `qa.py`, `qa_a.py`, `qa_b.py`, `qa_c.py`, `qa_final.py`, `qa_fuzzy.py`): the questions an operator or engineer would ask of the *history*, not just of the latest value. It sits in front of the intent rules and answers by computing, never by generating:
point-in-time lookups ("speed at 10:32:15", tyre pressure at 11:42 yesterday), extremes with their timestamps, event lists / counts / last-N with the vehicle state at each event, the signals before / after / around an event (frozen 1 Hz context for old events), longest periods and durations
(below a speed, idle, above an RPM), statistics (percentages, medians, rises over a sliding window, trip-vs-trip differences, week-vs-week Welch t-tests), trends with an OLS slope and p-value, volatility, gradual-degradation scans, correlations (speed vs pack temperature, with the ambient confound), change points,
and an **explanation chain**: a diagnosis object (headline, observed vs inferred claims, ranked hypotheses with evidence for and against, confidence with its reason, missing information, calculations) that the follow-ups read ("what evidence supports that?", "exact time periods", "what other signals changed", "another explanation?", "what are you missing?", "prove it", "how confident?", "what contradicts it?").
Efficiency changes are attributed with a trip-level regression (heating degrees, cooling degrees, speed, motorway share, rain, hard events, tyre shortfall) and a two-period decomposition; events are explained against the car's own baseline (z-scores, coolant flow vs its typical value, thermal load, ambient, sensor plausibility).
Routing: ordered keyword routes -> (for broad routes) arbitration with a **fuzzy router** (tf-idf cosine over ~800 labelled example phrasings with synonym clusters; the older pipeline's questions are negative examples) -> otherwise the older intent rules / SLM. Guard rails: RPM is derived and labelled; missing signals (cabin temperature, throttle, steering, radar, other cameras) are said to be missing;
dates outside the data are refused; "you said X" premises are checked against the conversation and the data; safety-critical control requests are refused (the assistant has no write path - read / analyse / explain only); stored log / event text is treated as data and instruction-like text is quarantined and reported; a handler that cannot run (e.g. a legacy database without the extension tables) falls back to the older pipeline.

## 5. Historical data storage approach
SQLite, tiered (`vgpt/store.py`): `sig_1s` 7 days @1 Hz (1/min parked); `sig_1m` 90 days (avg/min/max); `trips` per-trip summaries; `events` + zlib-frozen +-15 min context for important
events (root cause survives expiry of 1 Hz data); `logs` WARN+ with FTS5. Nightly compaction in pure SQL; WAL; 4 MB page cache; `mmap` off.
Extension tables for generated vehicles: `ext_1m` (per active minute: tyre kPa x4, rain 0-3, lux, 12 V, ADAS share, odometer, position), `vday` (daily odometer, pad %, battery health, 12 V), `places`, `charges` (kWh per session), `trip_ext` (per trip: places, hard events, regen, rain share, ADAS share); the vehicle's time zone is in `meta`. All readers tolerate their absence.
**28 simulated days = 5.4 MB** (`data/vehicle.db`; generated vehicles 2-7 MB), 90-day retention budgeted at ~20-30 MB. Writers: only `ingestd`; the assistant handle is read-only.
Real-time vs historical: live frames feed a 30-minute RAM ring (event-context capture) and 1 Hz rows; older data exists only as tiers - the document's "several weeks" is met with large margin.

## 6. Edge deployment architecture
* **Partitioning (Proposed)**: AI partition (container or hypervisor guest) on the vehicle HPC; safety-critical functions stay on the safety OS; the partition sees buses through a one-way gateway. Only the software boundary of section 1 is tested.
* **Footprint, current configurations (x86 laptop, 2 pinned cores; `results/perf_final_nomodel.json`, `results/perf_final.json`)**: see `VANTAGE_STATUS.md` section 8 - the no-model configuration peaks at about 105 MB process memory; the selector configuration at about 480 MB. **These are laptop figures. ARM memory and latency are Open.**
* **Footprint, historical (ARM emulation, VehicleGPT 1.1 with a loaded Qwen-0.5B; `results/guest_q05_secondboot.json`)**: guest OS idle 129 MB; model loaded 275 MB used; after 248 questions 312 MB; minimum available 1,663 MB; assistant 651 MB resident; swap 0. In that run 96% of the 136 original questions answered under 2 s (median 46 ms) and 91% of 694 history/analysis questions (median 171 ms, p95 3.1 s); the **model path took 17-32 s per call under emulation** (0.3-0.4 s on the laptop). Emulated timings are not Cortex-A55 timings and the configuration is not the current one.
* **Boot/update**: for configurations with a model, prompt state is built once on the target (`state_path`) and reloaded afterwards; stale files rebuild automatically.
* **Scalability** (the document's "Reference board / Raspberry Pi / VM to vehicle HPC"): same code on QEMU aarch64 (VehicleGPT 1.1 only), Raspberry Pi 5 / SBC (**Open**, not done), vehicle HPC with more RAM (could use a larger model with the same pipeline; **Proposed**). A 90-day history (1,189 trips, 16 MB) and 300 extra signals were run on the laptop (`VANTAGE_STATUS.md` section 8).
* **Supervision (Proposed)**: separate processes with crash isolation, a `systemd`-style unit; lazy model unload after 120 s idle (implemented for the llama.cpp wrapper).

## 7. User interaction design
* **Surfaces**: terminal chat, a tiny local web page (`app.py`), and a **car head-unit simulator** (`ui/`, `start_ui.bat`): 1920x720 infotainment layout with Drive / Camera / Assistant / Alerts / System screens, a persistent "offline, on-device" indicator, a camera-fault banner with sound and telltales (lane assist / camera), suggestion chips, voice button and optional spoken answers, an optional driving lockout (typing disabled above 5 km/h), and a Test bench to switch camera source (recorded clip / webcam / video file) and inject faults. The car's own ASR/TTS would be the voice front-end (the browser's speech recogniser is off by default because it may use a cloud service; this is the one place the demo can leave the "no cloud" rule, and only if you enable it).
* **Conversation layer** (`vgpt/dialogue.py`, rule-based, no model): unit preferences ("I want it in psi / miles / Fahrenheit" - remembered, re-shows the last answer, mirrored on the Drive screen), help ("what can I ask you?"), a 25-term glossary (CAN bus, SoC, TPMS, DTC ...),
  follow-ups ("what do you mean by more?" explains the previous answer in plain words, with the evidence view), and refusals (parking search, instructions to attack the car's networks).
* **Driving use**: every answer has a full text, a one-or-two-sentence `brief` (shown first, evidence behind a *Details* link) and a `speech` string with dates and units spelled out. Voice in = browser speech-to-text -> text -> assistant; voice out = the `speech` string read by speech synthesis (stand-ins for the car's voice stack; the assistant only handles text). Hard questions (causal, counterfactual, statistical, look-ahead-free, sensor-quality, protocol, poisoning, impossible precision) are answered by the analysis engine and counterfactuals are always labelled model estimates - see README.
* **Co-pilot behaviour**: a read-only pre-drive briefing (verdict, range, what needs attention, one suggestion), proactive alerts when a warning appears while driving (the assistant explains the likely cause itself), an evidence chart of the recorded signals around any event, and an optional hands-free voice loop. All deterministic and offline; the model never writes a number.
* **Front end**: `ui/app.html` is a responsive single-page app (chat + live vehicle panel + settings drawer, phone to wide screens); `ui/index.html` (served at `/headunit`) keeps the 1920x720 infotainment simulator. Both talk to the same local API (`/api/ask`, `/api/state`, `/api/evidence`, ...), so the assistant is the same process and the same memory budget.
* **Personas**: *driver* (short, plain answers) and *service engineer* (adds evidence: z-scores, timeline, data source).
* **Behaviours**: honest "I don't have that" for unsupported topics; clarification offering concrete readings and a "neither of these" choice when unsure (never a guess); states assumptions ("no period given, used the last 7 days",
  "this is an electric vehicle, so I report energy rather than fuel", "understood as ..." after typo repair).
* **Value by user** (the document's list): drivers - range/efficiency/usage/weather; service engineers - before-event causes, DTC context; fleet operators - weekly counts, trends,
  energy per period; validation teams - event windows and CAN-bus intrusion incidents.

## 8. Evaluation criteria - evidence
Numbers for the current revision are in `VANTAGE_STATUS.md` (do not quote older figures from this repository's history). The requirement-to-evidence table with a status word per row is `docs/REQUIREMENTS_EVIDENCE.md`.

| Criterion (document) | Evidence |
|---|---|
| System architecture: components + simulation/UI integration | sections 1, 2, 6; architecture diagram; wire-format simulator feeding the real adapters |
| AI capability: understand queries | rules alone answer 77 of 120 development questions correctly and 137 of 308 on the older final set (regression data); the clarification-first policy changes the risk/coverage balance rather than fixing generalisation; the frozen small model did not help. Sealed acceptance set: drafted, **unreviewed, never run** |
| AI capability: accuracy of responses | evaluator tested against deliberately wrong predictions (`tests/test_evaluator.py`); executor checked against plain SQL / numpy (`tests/test_reference.py`, `tests/test_windows.py` incl. a constructed midnight-crossing trip). The listed operations passed the specified independent checks; this is not a claim that every possible query is correct |
| Data integration: multiple protocols | section 2 (decoders validated against independent libraries and real captures where they exist; traffic is replayed) |
| Search & analytics: events, trends, anomalies, causes | operators in section 4a; paging and retention tests; counterfactual twin histories for attribution (associated, not causal); CAN anomalies below |
| Edge performance: memory, latency, scalability | x86 two-core measurements in `VANTAGE_STATUS.md` section 8 (configuration-specific); **ARM Open** |
| Privacy & security | local only, no network use by the assistant; read-only DB; text fields neutralised; bus-intrusion monitors (below). Not implemented: encryption at rest, access control for engineer mode |
| Innovation & practicality | candidate generation + interpretation contract + clarification-first policy; schema overlay; real-protocol validation against independent implementations. Practicality: no user study |

**CAN-bus intrusion monitoring on real datasets** (frame-level `canhealth.py`, signal-level `signalhealth.py`; data read in place, never copied):
HCRL Car-Hacking (16.6 M frames): DoS and fuzzing F1 1.00, RPM spoof 0.96, gear spoof 0.94, zero false alarms. OTIDS: 0 false alarms on 947,760 unseen attack-free frames, DoS F1 1.00, fuzzing 97% of windows
flagged (unlabelled), **impersonation 0% (undetectable by a rate monitor)**. Autohack2025 (2.7 M frames, 4 attack classes): P 0.87 / R 0.60 / F1 0.71. ROAD (real dynamometer vehicle, trained on 13.4 M clean messages):
correlated-signal F1 0.88-0.90, max-speedometer 0.76-0.92, max-coolant 0.61, **reverse-light flips undetected**, ~0.6-0.9% false alarms. SynCAN: flooding F1 0.75 (precision 0.999),
**continuous / plateau / playback / suppress near zero recall** - a range/step/period monitor is the wrong tool for subtle value attacks.

## 9. What is not met, not measured, or deliberately left out (read this)
**Current system (VANTAGE):**
1. **No ARM measurement.** Latency and memory are x86 laptop figures; QEMU is deferred. Whether any model path meets 2 s on a Cortex-A55 is unknown. The only ARM-emulation numbers are historical (section 6) and describe the earlier configuration.
2. **The SLM requirement is not met by evidence.** The product configuration has no model on the answer path; the frozen selector failed its pre-registered criterion and is disabled and frozen; the Qwen-LoRA is an experimental comparator whose model path exceeds 2 s at p95 on the laptop and which ignores conversation history through its interface.
3. **Language understanding generalises poorly** and the clarification-first policy trades errors for coverage; the largest failure category is clarifications that contain no intended reading. "Correct after one clarification" in the reports is an **oracle upper bound** (a user who always recognises the right reading); candidate availability and real user success are different quantities and the second was not measured.
4. **Older engines on the product route** (greetings, help, units, glossary, safety refusals, present-state status and health summaries, tyres/weather/service/driving domain engines) are not covered by the typed-pipeline checks; they have their own tests but no evidence contract or answer verifier. A few present-state phrasings may therefore be answered by them while an equivalent historical phrasing goes through the typed pipeline.
5. **Question sets are the author's**; the acceptance draft is unreviewed and has never been run; the development and older final sets are regression data.
6. **Isolation**: only the software boundary is tested (section 1). Vehicle-level isolation is a proposal.
7. **Generated data**: the vehicle, faults, weather coupling and 28/90 days are generated; no real EV battery data, no live bus, no CARLA/QNX/Android Automotive run.

**Inherited limits (VehicleGPT 1.1; still true):**
I1. (historical) **QEMU TCG timings are not A55 timings.** In the VehicleGPT 1.1 ARM-emulation runs the model path took 17-32 s per call (laptop 0.3-0.4 s); the 2 s target was shown only for the rules path. This says nothing about the current configurations, which have not been run on ARM.
I2. **Camera feeds** are implemented as a **health monitor, not as vision**: the assistant can say whether the front camera is blocked/degraded now or was earlier, and how lane-marking visibility was. Limits: evaluated on ONE real 48 s clip with *injected* faults (no real mud/ice/condensation footage); gradual dirt build-up is not detected (baseline adapts, by design); a scene that is inherently textureless (thick fog, a blank wall) can look "degraded"; `lane_conf` is an indicative texture/geometry score, not an ADAS lane detector; this is **not** a certified camera-blindness function. Privacy policy: no video is stored - per-minute metadata (`cam_1m`) plus a 64x48 grey thumbnail at fault events (`event_thumb`). Camera speed/memory was measured on the laptop only, not inside the ARM guest.
I3. **SOME/IP, VHAL and slog2info** were validated against specifications and independent libraries, not against live buses or real captures; VHAL's Binder transport is not implemented (the parcelable is the input).
I4. **The vehicle, its faults and 28 days are simulated** (real speed cycles aside); no real EV battery data was used (none available). No QNX, Android Automotive or CARLA was actually run.
I6. **Security gaps**: no encryption at rest, no authentication for engineer mode, CAN monitors are weak on subtle value and impersonation attacks (numbers above).
I7. Unused datasets: the password-protected "Survival Analysis" IDS archive, `18801994.zip`.
I8. **Generated data caveats**: weather is real, but rain/light are what the car's sensors would report (derived), and the car can only sense them while it is on. Routes are straight-line paths with a wiggle between city-anchored points,
   not real roads. Tyre, 12 V, brake-wear and battery-health behaviour are physics/engineering models, not measurements (no open dataset exists for them). The hard-event rates come from ONE real car in Austria (VIF-AEGIS).
   The extended question set (`bench/questions_ext.json`) was written by me after the rules and four misses were fixed - it is not independent; the "field" items are the tester's own words.
I9. **No live adapters yet for the other extra signals** (TPMS, rain, light, odometer, brake wear; GNSS position now has one): they exist as history in the store; wiring them through VHAL/SOME/IP adapters is straightforward (the VHAL ids exist) but not done.
I10. **Analysis-engine limits**: the statistics are honest but small-sample (a 28-day history gives 5-12 days per period; the answers say so and give p-values). Hypotheses are ranked from indirect signals (no part-level diagnostics) - the assistant cannot name a failing part, a supplier or a firmware version at the time of a fault, and says so. Causal claims are never made from correlation.
    The trip regression cannot see payload, wind, gradient or route; the unexplained share is reported, not hidden. RPM is an estimate (fixed 9:1 ratio, 0.33 m wheel). Wording robustness: the stress-list paraphrase sets are mine; on the first unseen set 84% / 83% were routed as intended (then fixed and added to the example pool), real drivers' wording will fall outside that.
    Generated vehicles with the cooling-pump story now overheat during a fast charge a week after the pump degrades (the warning threshold is 40 C); the heat-exchange model is an engineering approximation. Speed traces are rate-limited to what a road car can do (3.4 m/s2 fading with speed, 6 m/s2 braking) after real trips were stitched together.
