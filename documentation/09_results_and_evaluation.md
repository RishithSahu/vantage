# 09. Results and evaluation

How we measured, what we found, and how far to trust it. Where numbers come from a result file, the file is named.

## 1. How evaluation is done

### 1.1 Rules the evaluation follows
* **Exact frame**: the unit of measure for understanding is the whole 8-word frame, compared word for word with a hand-written gold frame. "Reviewed" counts rows where a person agreed that a different frame prints an equivalent answer; "strict" does not.
* **Frozen sets**: each evaluation file has a SHA-256 in a manifest and is never edited. Training rows are filtered against them (`--eval-leak vN`).
* **Sealed sets**: written once, kept out of development, never read for failures and never trained on; scored once after the system was frozen.
* **Pre-registered gates**: for each training round, the adoption rule was written *before* the run (target sets up by a stated margin, no frozen set down by more than 1.5 points).
* **Layer ablation** (`bench/meaning_eval/layer_ablation.py`): every rule layer can be switched off to show what the model alone does; this is how three layers were found to hurt and removed.
* **Independent checks of the code**: arithmetic is recomputed with plain SQL or numpy in tests (`tests/test_digest.py` etc.); protocols are compared with independent libraries; investigations and causes are scored against planted ground truth.
* **Latency is measured on the target**: the Pi, pinned to 2 cores, real model, real databases, nothing cached.

### 1.2 Evaluation sets (bench/meaning_eval/)
Roughly 40 files: `novel_v1`, `messy_v1`, `user_v1` (the owner's own sessions), `basic_v2`, `catalog_v1`, `digest_v1`, `hist_v2`, `slots_v2`, `followup_v2/v3/fresh_v4`, `hard_v1` to `hard_v5` (adversarial), `adv_v1` to `adv_v3`, `platform_v1/v2`, `ice_v1`, `icevocab_diesel_v1`, `icevocab_petrol_v1`, `longwrap_v1`, `anomaly_v1/v2`, `trend_v1/v2`, `wording_v1`, `holdout_v10`, and sealed sets `sealed_v2` to `sealed_v7`. Case Study 5's own end-to-end questions (the `cs5` set, a different author) are scored separately. The 500-question file `VehicleGPT_500_QA.csv`, written by another person, has no gold labels, so it is audited for obviously broken answers, not scored.

## 2. The scorecard against the case study's criteria

Written 2026-10-21 and updated 2026-10-23 (`docs/SCORECARD.md`). Scores are hard: 5 = a student demo that works, 8 = a team would deploy it after a trial, 10 = nothing left to ask for. By the owner's instruction, simulated vehicles and generated data are not marked down, and edge performance is not marked down (the measured edge numbers are still printed).

| Criterion | Evening of 21 Oct (honest baseline) | After the overnight work |
|---|---|---|
| System architecture | 7 | 9.5 |
| AI capability: understands questions | 5.5 | 7.5 |
| AI capability: accuracy of answers and summaries | 4.5 | 7.5 |
| Data integration | 7 | 9.5 |
| Search and analytics: events, trends, anomalies | 6.5 | 8.5 |
| Search and analytics: explain causes | 5 | 7.5, then 8.5 after the cause engine |
| Privacy: local only | 8 | 9.5 |
| Privacy: protection of data | 5 | 9.5 |
| Ease of deployment | 6 | 9.5 |
| Value for the four roles | 6 | 8 |
| **Overall** | **5.8** | **8.65** (mean of ten rows, edge excluded) |

On 23 Oct the four requirements the judges weigh most were re-scored with evidence: protocols 7 to 9, causes 5 to 8.5, safety isolation 6 to 9, scalability 4 to 8.5.

**Why not 10**: a run on QNX, Android Automotive, a Yocto image or a hypervisor guest; a real vehicle bus; and people who use it. The two model rows stay lower on purpose.

## 3. The audit of 2026-10-25: the shipped adapter on every set

Run: `python bench/eval_parallel.py models/meaning-lora-frame2r12.gguf audit1025 --procs 3 --threads 4`; results `results/meaning/audit1025_r10/predictions.json`; table by `bench/meaning_eval/rescore.py audit1025`. The frames the **model writes**, exact word for word, with the car type told to it.

| Group | Questions | Reviewed / strict |
|---|---|---|
| All sets | 3,842 | **90.0 % strict** |
| Older frozen sets (26 sets) | 3,038 | 95.3 / **91.5** |
| Newer sets, unseen wording | 804 | 84.6 / **84.3** |

Per set (reviewed / strict, %):

| Set | n | Result | Set | n | Result |
|---|---|---|---|---|---|
| adv_v1 | 112 | 96.4 / 93.8 | hard_v5 | 100 | 99.0 / 94.0 |
| adv_v2 | 112 | 95.5 / 91.1 | hist_v2 | 59 | 94.9 / 94.9 |
| adv_v3 | 80 | 92.5 / 90.0 | holdout_v10 | 32 | 65.6 / 62.5 |
| anomaly_v1 | 25 | 88.0 / 88.0 | ice_v1 | 121 | 86.0 / 85.1 |
| anomaly_v2 | 51 | 84.3 / 84.3 | icevocab_diesel_v1 | 38 | 100 / 100 |
| basic_v2 | 139 | 97.8 / 96.4 | icevocab_petrol_v1 | 37 | 100 / 100 |
| catalog_v1 | 73 | 87.7 / 84.9 | longwrap_v1 | 70 | 94.3 / 90.0 |
| digest_v1 | 94 | 97.9 / 95.7 | messy_v1 | 70 | 100 / 100 |
| followup_fresh_v4 | 167 | 90.4 / 88.0 | novel_v1 | 287 | 98.6 / 95.8 |
| followup_v2 | 72 | 95.8 / 95.8 | platform_v1 | 90 | 94.4 / 92.2 |
| followup_v3 | 29 | 93.1 / 93.1 | platform_v2 | 80 | 96.2 / 92.5 |
| hard_v1 | 220 | 91.8 / 82.7 | sealed_v2 | 149 | 96.6 / 92.6 |
| hard_v2 | 220 | 94.5 / 90.9 | sealed_v3 | 149 | 98.0 / 96.0 |
| hard_v3 | 220 | 95.9 / 91.8 | sealed_v4 | 100 | 85.0 / 85.0 |
| hard_v4 | 220 | 90.5 / 81.8 | sealed_v5 | 133 | 88.0 / 88.0 |
| slots_v2 | 84 | 96.4 / 88.1 | sealed_v6 | 107 | 85.0 / 85.0 |
| trend_v1 | 30 | 86.7 / 86.7 | sealed_v7 | 95 | 84.2 / 84.2 |
| trend_v2 | 50 | 76.0 / 76.0 | user_v1 | 67 | 97.0 / 92.5 |
| wording_v1 | 60 | 88.3 / 88.3 | | | |

Note: the sets used while developing (older, hard_v1 to v5, platform, adv) overstate quality; sealed v4 to v7 and the newer unseen-wording sets are closer to what a new user would see (84 to 88 %).

### 3.1 Where the model is wrong

* **Slots**: of 385 strict-wrong frames, the wrong word was in `concept` for 218, `act` for 189 (overlapping), `op` for 126; `cond`, `ent`, `vs`, `c2`, `ref` are rarely the problem. 98 frames differed only in concept; 56 only in act; 56 in act and concept; 30 only in the operator.
* **Confidence**: pmin below 0.50: n=43, 25.6 % right. 0.50 to 0.80: n=280, 53.6 % right. 0.80 to 0.95: n=366, 76.0 % right. 0.95 and above: n=3,153, 95.7 % right. **223 of the 385 wrong frames (58 %) had pmin at or above 0.8.**
* **Weakest answer kinds** (gold act): why 80.3 %, advice 80.9 %, offtopic 81.8 %, unsupported 84.4 %, anomaly 86.8 %, define 87.0 %, health 87.0 %, trend 87.5 %, stat 87.9 %.
* **Weakest concepts** (n at least 8): throttle_position 33 %, engine_oil_pressure 60 %, briefing 63 %, regen_power 65 %, unlisted 69 %, overtemp_warning 73 %, horsepower 75 %, trip_record 79 %, fuel_rate 79 %.
* **Most common confusions**: throttle_position against accelerator_position (8), regen_power against regen_energy (8 both ways), none against vehicle_health (5), tyre_pressure against tyre_condition (4), battery_overheat_warning against battery_temperature (4), unlisted against gear (4), fuel_rate against wh_per_km (4), engine_oil_pressure against unlisted (3), aux_battery_condition against aux_battery_voltage (3). Acts: now against health (19 in total), now against stat, now against spec, health against summary, events against stat, control against unsafe. Operators: none against mean (20), none against exists (10), drop against none (10), mean against sum (11).
* **By car type**: EV sets 90.1 %, petrol 90.2 %, diesel 87.9 %.
* **By wording style (families with enough rows)**: slang and typo 100 % (small), terse 92 %, Hinglish 89.5 %, figurative 40 %, "wrapper" (long preambles) 72.7 %.
* **A sample of unseen-set errors** (28 read by hand): many are label coin-flips (a joke asked: social or off-topic; "the 12 volt supply": voltage or condition; "what time is it" is handled by the pre-parser before the model); the rest are genuine confusions between neighbouring concepts, operator conventions and unlisted things the model maps to the nearest real signal ("a massage seat" read as a seat count).

## 4. History of the main understanding numbers

| Date | Adapter | Measure | Value |
|---|---|---|---|
| 18 Oct | `full` | exact frame novel / messy / sealed_v2 / user_v1 | 93.0 / 92.9 / 91.3 / 74.6 |
| 18 Oct | `full` | end-to-end answers right on 506 never-patched questions | 91.7 % (old product 51.0 %) |
| 19 Oct | `full_v6` | novel / messy / user / sealed_v2 / sealed_v3 | 93.4 / 94.3 / 94.0 / 94.0 / 93.3 |
| 19 Oct | `full_v6` | hard_v1 / v2 / v3 / v4 / v5 | 79.5 / 88.6 / 89.1 / 79.5 / 95.0 |
| 20 Oct | `full_v11` | five special sets | 92.2 to 95.0 |
| 20 Oct | `full_v11`, rules off | end-to-end same answer as a perfect reader | 91.3 % (v6 + rules 79.6 %, v8 + rules 87.5 %); 322 questions won, 68 lost against v6; p50 745 ms, p95 984 ms |
| 21 Oct | `frame2d` | 15 older frozen sets pooled | 90.3 % (the same as `full_v11`) |
| 24 Oct | `frame2w` | fresh 368 questions, strict | 60.3 % |
| 24 Oct | **`frame2r12`** | older 3,038 reviewed / strict | 95.75 / 91.80 (frame2w 94.96 / 90.72) |
| 24 Oct | `frame2r12` | fresh 677 (car type told), strict | 85.38 (frame2w 67.65) |
| 24 Oct | `frame2r12` | sealed_v7 (95 unseen) | 84.2 (frame2w 70.5) |
| 25 Oct | `frame2r12` | audit on all 3,842 | 90.0 % strict |

The gate for r12 failed on one rule only: `slots_v2` fell 2.4 points (2 questions). It was adopted anyway, and this is documented.

## 5. Other results by area

### Cause explanation (`docs/CAUSES.md`, `bench/causal_bench.py`)
Histories are generated with a known cause planted (the assistant never reads the generator's truth). Sealed test set, generated once with new seeds and scored once after the engine was frozen: **74 scored scenarios, first answer right 87.8 %, the right answer within the top three 95.9 %, confident false causes on negative controls 0 %** (8 controls; 75 % answered "nothing to explain"). Stated confidence 0.85 to 1.00 was right 95 % of the time (40 cases); 0.50 to 0.85 right 100 % (10 cases); below 0.50 right 71 % (24 cases). Nine first-answer misses of 74: two confident and wrong (cooling-pump read as higher speeds at 99 %; dirty windscreen read as climate overuse at 98 %), seven low-confidence (mostly alignment drift, 1 of 4 first answer, 3 of 4 in the top three). Single events: 322 of 322 traced to the planted cause across the four 105-day histories (a regression check, not held out). A scenario whose planted effect was too small to see in the data is not scored and is listed.

### Anomalies
Recall against the world builder's injected faults: 35 of 40 by "is anything unusual", 39 of 40 with the warnings list (`bench/anomaly_truth_eval.py`). CAN attack incidents on real HCRL recordings: F1 0.94 to 1.00 with 0 false alarms on the clean run (early prototype).

### Counterfactual attribution (6 Oct)
13 twin pairs (a roof box exists in one history only): an unrecorded step-like cause mostly stays in the "unexplained" part (median 76 %); recorded factors absorb a median 30 % of it. The supported claim is qualified: an unrecorded cause stays unexplained, not causal accuracy.

### Digests and arithmetic
Every number in summaries, "what changed", "unusual", "usual range" and parked-night battery loss is recomputed with plain SQL in a test (`tests/test_digest.py`, 16 tests on an EV, a petrol car, a diesel truck and a speed-only file). Round 10 and 11 executors: numbers recomputed with SQL (`tests/test_round10_executors.py`, `test_round11_executors.py`, 18 tests).

### Protocols, mapper, storage, scaling, safety
See [05](05_data_protocols_storage.md), [07](07_safety_security_privacy.md) and [08](08_performance_and_scaling.md).

### Pipeline fidelity (20 Oct)
The 1,075 frozen questions asked of the finished recording and of the database built from the arrived bus data at the same moment: 83.1 % identical answers, 84.2 % same content (up from 67.6 / 75.4 % after the minute tier and a waking-car fix).

## 6. The test suite

* About **1,770 tests** collected on 2026-10-25 with 0 failed imports. A full run on 24 Oct: 1,587 passed, 0 failed (before data files were removed). 14 legacy rule-engine tests were deleted at the owner's request on 23 Oct.
* Test modules cover: isolation, protocol oracles, hostile input, token mask, worker queue, no cache, route policy, confidence, recordstore, release manifest and release model, corpus, boundary, bootstrap, role walkthroughs (each of the four roles through the running app with the real model), installer and updater run for real with stand-ins for systemd, car-OS floor (with OpenCV, cryptography, psutil and FTS5 each unavailable), portability scan, meaning (including `RuleBudget`), executors, digest, causes, mapper, pipe, car, linked, instant load, fleet, procpool, scaling, tiers, storage budget, archive, atrest, vault, voice (with real recorded speech through Vosk), and the UI.
* **Test data**: `tests/testdata.py` and `tools_testdata.py` define groups with SHA-256; absent data skips with the reason and **fails** in the release gate (`VGPT_REQUIRE_TESTDATA=group,...`). A small versioned corpus `tests/corpus/` holds an electric, a petrol, a diesel, a synthetic hybrid, J1939 / OBD-II / proprietary CAN cars, missing signals, injected faults, stale data and no data, so answer tests do not need the multi-gigabyte worlds. On 24 Oct the owner deleted `world_3yr.db`, the `gen_*.db` files and other data; about 38 tests then failed and 76 errored for missing data (not regressions); the corpus work of 25 Oct fixed the dependence.
* **Release gate** (`tools_release_gate.py`): the full suite with the model, corpus, driving fixture and sample cars required; release provenance (manifest and source hash); and more.

## 7. What these numbers do and do not mean

1. **We grade ourselves.** The ontology, gold frames, training questions and test questions come from the same few authors. A model can score 95 % on that and still fail a stranger. This is the single most important caveat.
2. **The unit measured is the frame, not the answer.** A right frame can produce a useless answer and a wrong frame can print an acceptable one (the review found that 134 of 283 "errors" printed an equivalent answer or had a wrong gold label).
3. **No confidence intervals or repeated seeds**; sets of 25 to 290 questions (about 3 to 6 points of uncertainty per set).
4. **No human evaluation of free-form answers, no user study, no real vehicle, no long soak** (the Pi ran for hours, not weeks).
5. The test sets that were read for failures and fed back into training (hard, platform, adv v1) are optimistic. The sealed sets and the newer sets are the honest ones.
