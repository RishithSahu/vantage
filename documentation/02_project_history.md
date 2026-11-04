# 02. Project history, from the first day

All dates are 2026. The first recorded work on Case Study 5 is **2 October**. The earlier days of the Teliport work went to other case studies (Case Study 2 was dropped for good; Case Study 1 predictive maintenance and Case Study 3 adaptive QoS were separate projects) and are not part of this record.

The project went through **six different designs for "how does the system understand a question"**. Reading the history as a straight line would be wrong; it is a sequence of experiments, most of which were abandoned for measured reasons. Each phase below says what was built, what was measured, and why it was kept or dropped.

## Overview

| Dates | Phase | Folder / name | Core idea | Outcome |
|---|---|---|---|---|
| 2 to 4 Oct | 1. Prototype | `vehiclegpt/` | Rules and keyword routes first, a small model fills slots | Working end to end; released as 1.0.0 |
| 5 Oct | 2. Query compiler | `vehiclegpt/` (v1.1.0) | The model only compiles a question into a typed Query IR; everything after is code | Clean architecture; kept as the engine for the typed query path |
| 5 to 6 Oct | 3. VIR / VehicleFM research | `vehiclefm/` | The model is only a planner over a catalog | Research only; rules baseline beat small planners |
| 6 to 8 Oct | 4. VANTAGE (model route) | `vehiclegpt2/` | One frozen model, retrieval, tool calls | Frozen models too weak; function selection was the bottleneck |
| 13 to 14 Oct | 5. Learned query-plan parser | `vehiclegpt2/` | Fine-tune a LoRA to write a whole typed plan | 76 to 83 / 104 acceptance, legacy 102 / 104; not ready |
| 18 to 19 Oct | 6. Meaning frame | `vgpt/meaning/` | The model writes a short closed-vocabulary frame; code does the rest | **The design that became the product** |
| 19 to 25 Oct | 7. Platform and product | everywhere | Plug-and-play cars, pipe, receiver, roles, storage, safety, scaling, installers | v1.3.4 |
| 25 to 26 Oct | 8. Latency and audit | `bench/`, Pi | Measure the Pi honestly, fix per-question overheads, audit accuracy | Ongoing |

## Phase 1: the prototype, 2 to 4 October (`vehiclegpt/`)

**Owner's instructions (2 Oct):** build it on the user's own laptop; simulate the case study's embedded target (about 2 GB RAM, 2 ARM A5x cores, no cloud, under 2 s, weeks of history, safety isolation); enforce the limits for real (a Windows job object and a pinned set of cores), and say plainly that a laptop core is faster than an A55 so latency must be caveated; ask before downloading any model. The case study PDF is "the word of god": follow it literally.

**Built:**
* A pipeline of rules router, a domain gate, a Qwen2.5 slot filler (words grammar), SQLite analytics and templates. A simulated 28-day electric car on real OBD speed cycles; real CAN attack recordings (HCRL) scored with the CAN health monitor at F1 0.94 to 1.00.
* A real protocol layer (`vgpt/protocols/`) validated against independent libraries and real data (DBC, ISO-TP, UDS, SOME/IP, NMEA, VHAL, logs).
* An ARM test: Debian 12 arm64 under QEMU, Cortex-A55, 2 cores, 2 GB. Model calls took 33 to 44 s there; 503 analysis questions ran 91 % under 2 s because most are answered by code, not the model.
* A camera health monitor in numpy (blocked, degraded, low light, 97.5 % frame accuracy on a real clip), a head-unit UI, voice in and out, a data generator on real weather and real driving data, a conversation layer, GPS (NMEA) and a map.
* On 3 Oct a 120-question stress list (retrieval, temporal, trends, explain-why, cross-source, missing data, false premises, safety, prompt injection) drove a deterministic analysis engine (`qa_*.py`) and a fuzzy router. Later the same day: hard-reasoning, voice and "judge wow" features (pre-drive briefing, proactive alerts, evidence chart).
* 4 Oct: a linear question classifier (held-out 94 to 98 %), a launcher, a hardened local server, release tooling and documents. **Release 1.0.0.**

**Honest reading:** the prototype understood questions mostly through keyword rules and nearest-example matching; each new batch of failures was fixed with new rules. This is the pattern the owner later rejected (see phase 6).

## Phase 2: the query compiler, 5 October (v1.1.0)

**Idea:** the language model (or rules) should only *compile* a question into a typed Query IR; validation, capability analysis, planning, execution, evidence, wording and verification are deterministic.

**Built:** `query_ir`, `query_validator`, `capability`, `query_planner` (the only place SQL exists), `query_executor`, `evidence`, `answer_composer`, `answer_verifier`, `qc.py`; a derived-metric engine and a sensor-quality engine; follow-up conversation state; a grammar-constrained model path; a simulator extension with 23 extra signals and five faults with ground truth; and a `training/` pipeline (own LoRA trainer, GGUF adapter converter, 13-metric evaluation). Qwen3-0.6B was downloaded with approval.

**Kept:** nearly all of it. This typed engine is still what executes every answer today.

## Phase 3: VIR / VehicleFM research, 5 to 6 October (`vehiclegpt/`, `vehiclefm/`)

**Idea (VIR):** the LM is only a planner over a Vehicle Semantic Catalog; deterministic store, executor and diagnostics do all facts. Constraints from the owner: no model training, every test under 2 cores and 2 GB, model swappable by a config file.

**Measured (no training):** oracle plan 99.6 %; plain retrieval-augmented 0.5 B model 34.7 % and 1.5 B 43.8 % at 15 to 28 s per question; retrieval plus hand cues (no model) 88 % on test, 80 % on held-out phrasing; planners with catalog-generated exemplars about 0.5 (0.5 B) and 0.84 (1.5 B) on validation but about 0.6 on paraphrases; the 1.5 B model plus runtime peaked near 2.14 GB, over the limit.

**Conclusion recorded 6 Oct:** the case study is judged as an architecture submission; recommended compile-at-ingest vehicle memory, a catalog with causal relations, a tiered planner and proof-carrying answers. An A55-core latency of 8 to 15 s for LM calls was estimated, so the LM could not sit on a 2 s path.

## Phase 4: VANTAGE and the model route, 6 to 8 October (`vehiclegpt2/`)

The project was renamed **VANTAGE**. The audit of 6 Oct (see `docs/history/`) found six defects in the experiment setup, such as retrieved examples that never reached the prompt (so every earlier "bank4" result was void), and a demonstration that taught a substitution the design forbids. All were fixed and guarded by tests.

A **pre-registered comparison** was written *before* running (variants, the rule for keeping one, differences under about 3 points on 120 questions to be treated as noise). Findings:

| System | Result |
|---|---|
| Frozen LFM2.5-350M, planner only | 18.3 % correct (CI 12 to 26) |
| Frozen Qwen-0.5B / 1.5B | about 31 % / 49 % |
| Rules only (no model) | dev 77 / 120; older final set 137 / 308 (44 %) |
| Rules + generator + clarification policy | dev 85, final 150, but more wrong answers |
| Qwen-LoRA cascade | more correct but also more wrong; p95 2.4 s on x86 |
| Granite 4.0 1B, with 8 generated demonstrations | 34 / 60 lenient, 31 / 60 strict, about 91 s per question |
| xLAM-2-1B native tool call | 0 to 1 of 12 |
| GLiNER2 / GLiNER2.5 / Decide extractors | top-1 52 to 78 %, 9.8 s per question, wording-sensitive |
| Qwen2.5-7B reference (research only) | plain description 40 / 60 (67 %); typed tool call 9 / 60 |

**Key finding:** *function selection* (which operation the question asks for) was the bottleneck: LFM 38 to 45 %, Qwen 57 %, Granite 67 %. Prompts of hundreds of tokens were incompatible with 2 s on a small core (Granite prefill about 10 tokens/s on the pinned E-cores).

**Also built:** an editable interpretation preview (the person can see and correct how the question was read), a user-study scaffold (never run with real participants), and the rule that "correct after one extra turn" is an oracle upper bound, not a user result. The 7 Oct report concluded the no-model product was a *control*, "not the completed VehicleGPT": the case study asks for an SLM and the evidence did not yet show one earning its place.

## Phase 5: the learned query-plan parser, 13 to 14 October

A LoRA (`qp-lora`, versions v12 to v14d) trained to write a whole typed Query plan. Held-out-template plan exact 0.88, but on the acceptance set only 39 / 104 in learned-only mode at v12, 76 to 83 / 104 at v14d, against 102 / 104 for the legacy rules. It learned template frames; natural multi-intent questions need human-written paraphrases. **Not adopted.** Lesson recorded: one meaning must have exactly one label encoding, otherwise confidence collapses.

## Phase 6: the meaning frame, 18 to 19 October (the pivot)

**Owner's complaint (18 Oct):** another coding agent kept fixing failing questions with new regexes in `semantic_ops.py` and `brain.py`; "that is no better than humans writing answers". The owner wanted the 0.5 B model to understand on its own, with humans supplying only the *meaning of words*.

**The method (`docs/semantic_method.md`):**
* An **ontology** (what a vehicle question can be about) and a **glossary** (what a word means) as data.
* The model writes a **frame** of a few words from the ontology's closed vocabulary, constrained so it cannot write anything else.
* Code may parse typed values, type-check, and calculate. It may **not** decide what a question means. A test (`RuleBudget`) counts every text-pattern call site and fails if one is added.

**Results (18 Oct, `full` adapter, trained once in 55 minutes on a 6 GB RTX 3050):** exact frame novel 93.0 %, messy 92.9 %, sealed 91.3 %, user 74.6 %; end-to-end 91.7 % correct against 51.0 % for the old product on 506 never-patched questions; p50 1.1 s, p95 1.7 s, 679 MB on two pinned x86 cores. Ablations: words alone 0 %; generated-only data far worse (78 %) than hand-written rows (90.6 %); hints helped only a little.

**Nineteen October:** the owner approved making the meaning route the default. A "night loop" of hard adversarial sets (hard_v1 to v5) took adapters v2 to v6; the owner stopped training after `full_v6`. A full audit (`docs/AUDIT_2026-10-19.md`) listed what was fixed and what was not.

## Phase 7: the platform and the product, 19 to 25 October

### 19 to 20 October: any vehicle, a long history, a live car
Owner's asks: fix context loss; plug-and-play for any vehicle; at least 10 weeks of history; collectors for CAN, SOME/IP, VHAL, QNX, ADAS, camera and diagnostics; a long CARLA-like dataset; a live car simulation; "I hope we are not just doing a rule based method".

Built: **profile packs** (the single per-vehicle integration artefact), a profile router, a signal "lake", a trips and charging ledger, a **105-day world builder** with ground truth, a virtual car with 13 injectable faults, 32 investigation analyses, resource tiers, sessions, event and records executors, and an ontology extension.

Training continued under pre-registered gates (adopt only if the target sets rise at least 10 points and no frozen set drops more than 1.5): v7 (rejected), **v8 (adopted)**, v10 (rejected on two gates), v11 (met the owner's target on all five special sets, 92.2 to 95.0 %, missed a gate by 2.2 points on one contested set). At about 10:00 on 20 Oct the owner ordered **no more training** to test the system.

The owner then rejected the Live page (sliders and buttons) as the wrong model: they wanted *a car that drives by itself following a dataset*, like sitting in a car. The interface was rebuilt, with a drive plan following the trips table. A UI audit in the browser found and fixed many bugs. Petrol and diesel were added as first-class: vehicle type is detected from the data (`vehicle/caps.py`), with an engine overlay for the world builder, `fuelx` executors, and new world files.

On the evening of 20 Oct: pooled end-to-end same-answer 91.3 % (v11, rules off), p50 745 ms, 322 questions won and 68 lost against v6.

### 20 to 21 October: pipeline, roles, app, voice, the link
* A recording is a **source**; a single writer runtime stores it; the assistant answers from the stored data (and the last ~20 seconds from a live bus). Roles (driver, service engineer, fleet operator, validation) with scrypt passcodes, scrubbing, and a hash-chained audit. An app window and kiosk mode. Offline voice input (Vosk, wake word "Vantage") was installed with approval.
* A car (laptop) to Vantage (Raspberry Pi) link: first HTTP, then replaced on 21 Oct by a **one-way local pipe** between two separate programs (see [03](03_architecture.md)). Owner's rule: the receiver must adapt to any car with no per-car setup.

### 21 October: the honest assessment, the scorecard, frame v2
Re-reading the PDF produced a brutal self-score of **5.8 / 10** (`docs/HONEST_ASSESSMENT_2026-10-21.md`): strong architecture and discipline, weak *proof* (everything measured on data we made ourselves; model-alone accuracy on diesel and petrol had been only 37 to 40 %, carried by rule layers). The owner then asked to raise everything except the model to 10 honestly; the scorecard reached **8.65**. In parallel **frame version 2** (eight slots including a second quantity and a reference to the last answer, new acts `summary`, `anomaly`, `explain`) was trained in four rounds and two weight averages, ending with `frame2d`. The owner then said: leave the model alone, no more training.
Also that day: a refuse-rather-than-guess **signal mapper**; Android Automotive (Cuttlefish) running on the Pi as a data source; the correction that "any OS" meant any *car* OS (QNX, Android Automotive, Linux, AUTOSAR), not laptops.

### 22 to 23 October: the Pi becomes the reference
* **Root cause of 3.7 to 6 s per new wording on the Pi:** the Cortex-A72 has no dot-product instruction, so llama.cpp tops out near 11 prompt tokens/s. An int8 **ONNX Runtime** build of the same model (no training) reached about 30 GOPS: median Pi answer 4.7 s down to 2.19 s, same accuracy (90.9 % exact on both backends over 220 frozen questions). The owner removed old models (30.5 GB freed).
* **The four requirements the judges weigh most** were raised and measured: protocols (nine families verified against independent libraries), causes (a cause engine with a sealed test), safety isolation (a real-time rig on the Pi) and scalability (probe, plan, calibrate, autoscale). A one-slide architecture diagram was generated from the result files.
* Latency work: a token-mask decoder, previous-turn gate, guess-ahead decoding; Pi median 1.34 s. Security: sealed database in RAM, hardening, DoIP, kernel CAN checks (53 of 53).
* New protocols (LIN, DoIP, CANopen, FlexRay), 49 cars, four-year worlds, a storage budget, an "instant load" testing option, installer 1.3.0.

### 24 October: failures analysed, training approved again, the pre-parser
* A failure analysis of 283 wrong frames (of 3,038): overlapping labels 40 %, unseen wording 21 %, operator and slot conventions 23 %, the ontology's edge 12 %, test artefacts 4 %; **58 % were made with high confidence** (`docs/FAILURE_ANALYSIS_2026-10-24.md`). The owner turned ask-back off and asked for a model "with no confusion".
* The owner approved training again (overnight, autonomous) with strict honesty rules. Rounds r10, r11, r12 followed; **r12 was adopted at 12:00**: fresh questions 67.7 % to 85.4 %, sealed set 70.5 % to 84.2 %. A further patch (r13) did not help and was not adopted ("stop patching the same adapter").
* After an installer went to a friend and the owner listed many wrong answers, a **rule-based pre-parser** (`canon.py`, `askme.py`) was added in front of the model, visible to the user ("you wrote..., I read it as..."), with one rule for when it may ask back. Installers 1.3.1 to 1.3.3 followed, driven by the owner's screenshots.

### 25 October: petrol and diesel parity, the hardening round, v1.3.4
* The owner demanded exactly the same answers for petrol and diesel as for electric cars. Fixed: per-trip fuel, hour-of-day fuel, two named periods, "fall below" not read as autumn, electric words put to a fuel car, trip estimates in fuel words.
* A review table (P0 to P2 issues) was fixed thoroughly: one start-up for every door, **no answer or frame cache**, a worker with a bounded queue and deadlines, calibrated confidence with refusal, route policy (deny by default), sealed and expiring records, a small versioned test corpus, a release manifest, and `docs/BOUNDARY.md`. The word "model-only" was dropped from the product's description. A local git repository was created and tagged `v1.3.4` (never uploaded).
* Testing only on the Pi found two real bugs that the laptop hid: a Python 3.13 thread-attribute clash, and a full-table MIN/MAX scan on every question. The Pi measurement of every question, per car type, led to fixes for repeated availability probes, daily-tier trends, and more.

## Phase 8: latency work and the accuracy audit, 25 to 26 October

* **Decode-pass investigation** (no training): 4-bit weights, node fusion, more threads, an S8S8 variant were all measured on the Pi's A72 and gave no gain; only a row-layout output head helped (a decode pass from about 166 ms to about 135 ms in isolation, about 70 ms end to end). Details in [08](08_performance_and_scaling.md).
* **Accuracy audit** (2026-10-25): 3,842 questions, 90.0 % exact; where the model lacks, and the non-standard training methods proposed because ordinary training had stalled (see [09](09_results_and_evaluation.md) and [10](10_research.md)).
* **A parallel session** (2026-10-26) worked on per-question tool-stage speed on the Pi: removed a hidden 1.8 s per question caused by a lake-view presence check, moved long-window fuel economy and engine-temperature trends onto the daily tier, vectorised a per-day loop and stopped the first reading early for questions read twice. Over-2-second questions on the Pi fell from 26.9 % to 22.1 % with no answer changed, and to 9.7 % with an opt-in prompt-hint policy that changes 3.3 % of answers. See [08](08_performance_and_scaling.md).

## What did not work, in one list

* Frozen small models used as planners, selectors or extractors (LFM, Granite, xLAM, GLiNER, DeBERTa NLI, Qwen 1.5B and 7B): too inaccurate or too slow for a 2 s, 2 GB target.
* Retrieval-augmented prompts on a small model: long prompts break the latency limit.
* A model-written whole query plan (`qp-lora`): did not generalise to natural wording.
* A "vocabulary layer" and the arbitration rules that boosted accuracy on diesel and petrol in early versions: when measured, they changed 65 frames, fixed 2 and broke 56; the arbitration layer was retired.
* 4-bit weights on the Raspberry Pi 4: 2.4 times slower than int8.
* A cold-minute archive: built, measured at 0.9 to 10 s to load, and reverted to protect latency.
* A frame cache and an answer cache: removed on owner's order; "a cache is not allowed at all".
* Predicting the user (prefetch / read-ahead): rejected by the owner.
* Patching the same adapter again (r13): no gain.
