# 14. Repository map

Where things live in `vehiclegpt2/`. The Python package is still named `vgpt` (the project started as a copy of VehicleGPT 1.1 and was renamed Vantage; the package name was kept so every earlier test keeps working). A module-by-module picture is not repeated here; this page says what each area is for.

## Top level

| Path | What it is |
|---|---|
| `vehiclegpt.py`, `app.py` | the launcher (run, doctor, smoke, ingest, receive, car, car-aaos, ...) and the terminal chat |
| `vgpt/` | the product (about 250 files): everything below |
| `ui/` | the head-unit server (`server.py`), the cockpit interface (`ui/modern/`), live car glue, research dashboard |
| `models/` | model files (not tracked): base GGUF, adapters, int8 ONNX builds, `product.json` (names the adapter and ONNX build), the speech model |
| `data/` | vehicle histories, 4-year worlds under `data/world4/<car>/`, reference data, captures (not tracked) |
| `results/` | every result file (not tracked): `results/meaning/` (accuracy runs), `causal/`, `scaling/`, `isolation/`, Pi runs |
| `training/` | the LoRA trainer, dataset builders, the natural-wording banks and evaluation helpers |
| `bench/` | about 175 scripts: evaluation, latency, Pi benches, mapper and protocol checks, probes; `bench/meaning_eval/` holds the frozen and sealed sets with hashes; `bench/probe_questions/` the question files per car type |
| `tests/` | about 135 test files, `tests/corpus/` (small versioned test cars), `tests/fixtures/` |
| `tools/`, `tools_*.py` | build tools: ONNX build (`tools/build_ort_model.py`), real-time task (`tools/rt_task.c`), world and car builders, release, manifest, test data, storage report, scaling, triage, training queue |
| `deploy/` | the Windows installer (`windows/`), Linux and Raspberry Pi units (`linux/`), the resource tiers (`tiers.json`) |
| `vm/` | the QEMU ARM guest driver from the first prototype |
| `docs/` | the project's own working documents (list below); the two draw.io diagrams |
| `documentation/` | **this folder**: the consolidated, end-to-end documentation |
| `.git` | a local repository (tag `v1.3.4`, no remote) |

## Inside `vgpt/`

| Area | Files | Purpose |
|---|---|---|
| Start-up and platform | `bootstrap`, `hardening`, `offline`, `doctor`, `paths`, `version`, `applog`, `memuse`, `governor`, `releasecheck`, `portability`, `isolation`, `product`, `appinstall` | one start-up for every door; offline guard; checks of the install and of the source |
| Protocol decoders | `protocols/` (canframes, dbc, isotp, uds, obd, j1939, canopen, lin, flexray, doip, someip, nmea, logs, vhal, vhalparcel) | stdlib-only, listen-only |
| Car side | `car/` (car, drive, models, transmit, writer, transport, dashcam, aaos) | the separate simulated vehicle; never imported by Vantage |
| Simulation | `world/` (build, emit, faults, ice, signals), `live/car`, `road`, `sim`, `datagen*`, `datasets`, `realcycles`, `livegen`, `sim` | worlds, virtual car, bus emitter, generators |
| Link | `pipe/` (wire, sniff, fingerprint, receiver, transport), `testbench`, `live/replay`, `live/collect` | one-way pipe and recognition |
| Ingest and mapping | `ingest/` (readers, standards, units, catalog, mapper, autoprofile, capture, ingest) | any capture to a vehicle history; the refuse-not-guess mapper |
| Vehicle profile | `vehicle/` (profile, router, onboard, provision, caps, lake, ledger) | profile packs, decoding, vehicle type from data, the signal lake, trips and charging |
| Runtime and storage | `live/runtime`, `store`, `dailytier`, `archive`, `storage_budget`, `atrest`, `runtime_vault`, `securestore`, `recordstore`, `tiers`, `sqlfeatures`, `signal_registry`, `schema_overlay`, `canhealth`, `signalhealth`, `sensor_quality`, `camera`, `frame` | the single writer, tiers, archive, sealing, quality monitors |
| Understanding | `meaning/` (canon, askme, layers, literals, anchor, discourse, prevgate, interface, ort_backend, cues, frame, ontology, glossary, calibration, arbitrate, evmap, pipeline) | the pre-parser, the frame model, confidence, follow-ups |
| Executors | `meaning/` (resolve, compile, records, eventx, calx, digest, pairs, tripx, thresholdx, trendv, binx, fuelx, careadv, plain) and `query_*`, `plan_exec`, `derived`, `kinematics`, `analytics*`, `temporal*`, `composed`, `semantic_ops`, `change_ops`, `capability`, `contracts`, `evidence`, `answer_*`, `outcomes`, `paraphrase` | typed queries, the only SQL, all arithmetic, wording and verification |
| Explanation | `cause/` (data, stats, engine, library, events, answer), `investigate/` (engine, parse, primitives, report, templates, t_*) | the cause engine; the 32 analyses |
| Interface and roles | `access`, `privacy`, `routepolicy`, `rolelens`, `rolepages`, `sessions`, `voice`, `dialogue`, `shell/` | roles, permissions, the window, voice |
| Scaling | `scale/` (probe, plan, calibrate, autoscale, procpool, service, fleet) | hardware to configuration; fleet hub |
| Earlier designs, still in the tree | `brain`, `qc`, `nl2ir`, `nlu`, `lexicon`, `qa*` (keyword route handlers), `qa_clf`, `qa_fuzzy`, `modelbrain`, `modelplan`, `retrieval`, `prompt`, `schema*`, `toolspec`, `plangrammar`, `plancompile`, `dialects`, `candidates`, `ir_slm`, `learned_front`, `slm_backend`, `demos`, `exemplar_bank`, `extractor`, `interpret*`, `policy`, `capreg`, `*_grounding`, `semantic_frame`, `topics`, `dialogue_state` | retired front ends and the older route handlers that the resolver still reaches for topic meanings through a small adapter; `brain.py` is the base class of the meaning brain |

## `docs/`: the project's own documents

| Document | About |
|---|---|
| `ARCHITECTURE.md` | system architecture along the case study's seven headings (partly older numbers; this folder supersedes it) |
| `BOUNDARY.md` | every stage as learned, code that calculates, or code that influences meaning |
| `semantic_method.md` | the meaning-frame method, ablations, hard-set loops, research basis |
| `FRAME_V2.md`, `FRAME_SLOTS.md` | the eight-word frame and the evidence for it |
| `TRAINING_QUEUE.md`, `training.md`, `V11_RESULTS.md`, `ROUND10_2026-10-24.md` | training rounds, gates, results |
| `FAILURE_ANALYSIS_2026-10-24.md` | the 283 wrong frames by cause, and the literature table |
| `SCORECARD.md`, `HONEST_ASSESSMENT_2026-10-21.md`, `GAP_REPORT.md`, `REQUIREMENTS_EVIDENCE.md`, `AUDIT_2026-10-19.md` | self-assessment against the case study |
| `CAUSES.md` | the cause engine and its sealed test |
| `SAFETY_ISOLATION.md`, `SAFETY_CASE.md`, `SECURITY.md`, `ARM_PROOF.md` | safety argument, security, ARM evidence |
| `PROTOCOL_EVIDENCE.md`, `DATA_SOURCES.md`, `REAL_DATA_PATHS.md`, `signal_ontology.md` | protocols and sources |
| `CAR_AND_VANTAGE.md`, `CAR_CATALOGUE.md`, `PLATFORMS.md` | the two programs, the 45+ cars, car operating systems |
| `STORAGE.md`, `SCALING.md`, `DEPLOYMENT.md`, `HARDWARE_REFERENCE.md` | storage tiers, scaling, installers, bill of materials |
| `CANON.md`, `ASK_BACK.md`, `MODEL_ONLY.md` | the pre-parser rules; ask-back; the earlier model-only measurement (superseded by BOUNDARY) |
| `history/` | archived status reports of 6 to 7 October (hybrid era) |
| `VANTAGE_ARCHITECTURE.drawio`, `VANTAGE_PIPELINE.drawio`, `architecture.svg`, `architecture_2026-10-21.svg` | the diagrams |
| Root: `README.md`, `DESIGN.md`, `CHANGELOG.md`, `HANDOFF.md`, `VANTAGE_STATUS.md`, `KNOWN_LIMITATIONS.md`, `USER_GUIDE.md`, `TROUBLESHOOTING.md`, `PRIVACY.md`, `RELEASE_CHECKLIST.md`, `THIRD_PARTY_NOTICES.md` | user and developer documents of the product |

## Key data files

| File | Purpose |
|---|---|
| `vgpt/meaning/ontology_data.json`, `glossary.jsonl`, `bindings.json`, `lexicon.json`, `discourse.json`, `concept_docs.json`, `canon_vocab.json` | the vocabulary the model and the code share |
| `vgpt/meaning/calibration.json` | the calibrated confidence map |
| `vgpt/ingest/catalog.json` | the universal signal catalogue (OBD-II, J1939, VSS, VHAL name maps) |
| `vgpt/protocols/vhal_props.json` | the Android VHAL property table (generated from AOSP) |
| `vgpt/vehicle/packs/<id>/`, `vgpt/car/models/<id>/` | decoder files (profile pack: `profile.json`, DBCs, `services.json`, `dtc.json`) |
| `models/product.json` | which adapter and ONNX build the product uses |
| `RELEASE.json` (in a release) | the manifest: source digest, models, vocabulary, test data, backend, platform, installer |
| `deploy/tiers.json` | resource tiers |
