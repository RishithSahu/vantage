# 06. The AI model, the meaning frame and how it was trained

## 1. The idea in one paragraph

A 0.5-billion-parameter model cannot reliably remember numbers, write database queries or reason over thousands of rows. It *can* be trained to decide **what a question is asking**, if the answer is a short sentence in a closed vocabulary. So the model's whole job is to write a **meaning frame**: eight words chosen from a fixed list. Everything else (the data, the arithmetic, the wording, the checks) is ordinary code that can be tested. The model's output is also what makes it safe: it can only write words the vehicle ontology allows.

## 2. The meaning frame

An 8-word line, always in this order (frame version 2; version 1 was the first six):

| # | Word | Meaning | Values |
|---|---|---|---|
| 1 | `act` | the kind of answer wanted | 21: now, at, stat, extreme, trend, compare, events, context, why, spec, define, howto, advice, health, estimate, social, offtopic, unsupported, control, unsafe, ambiguous (+ `summary`, `anomaly`, `explain` in v2) |
| 2 | `concept` | the thing it is about | about 175: signals, events, derived quantities, specs, topics, and the special `unlisted` (a thing no signal exists for) and `none` |
| 3 | `op` | operator | none, mean, median, max, min, sum, count, share, duration, spread, first, last, change, drop, rise, list, exists (+ `corr`) |
| 4 | `cond` | a driving condition that restricts the data (highway, city, cold, hot...) or `threshold` | closed list |
| 5 | `vs` | the second operand of a comparison: a second period or condition | closed list (+ `typical` in v2) |
| 6 | `ent` | a selected unit: trip, charge, day, week, source, place | 7 |
| 7 | `c2` | a second concept ("A and B", "A against B") | any concept, else none |
| 8 | `ref` | what the question points back to | none, answer, period, concept, focus, item |

Example: "why has my range dropped lately" is written `why range_estimate drop none none none none none`. A question that needs none of the new words ends in `none none`, so every v1 frame is a prefix of a v2 frame.

**Values never take a slot.** Numbers, units, dates, thresholds and fault codes are parsed from the text by code (`vgpt/meaning/literals.py`), because parsing "above 60 mph" is parsing, not understanding.

The ontology (`vgpt/meaning/ontology_data.json` and the signal registry) has 175 concepts: 60 declared by individual vehicles, 42 signals, 30 events, 13 specs, 13 derived, 13 topics, 2 entities, 2 special. The **glossary** (`glossary.jsonl`, 112 entries) is the one sanctioned place for a human to help the model: "this word means that". It never says which operation an answer gets.

## 3. Why Qwen2.5-0.5B (the selection rationale)

### 3.1 What was tried

| Candidate | Role tried | Result |
|---|---|---|
| Qwen2.5-0.5B-Instruct + LoRA | reads the question into a closed frame | **chosen**: 84 to 95 % exact frame depending on the set; fits 2 GB |
| LFM2.5-350M and 1.2B (frozen) | planner / selector | 12 to 18 % correct as a planner; function selection 38 to 45 % |
| Granite 4.0 1B | planner with generated demonstrations | 34 / 60 lenient; about 91 s per question; needs n_ctx 8192; peak 2.85 GB |
| xLAM-2-1B (tool-calling model) | native tool call | 0 to 1 of 12 |
| Qwen2.5-1.5B (frozen) | planner | 49 % but resident 1.9 to 2.4 GB; far too slow per prompt token |
| Qwen2.5-7B (research reference) | upper bound | plain description 67 %; typed tool route only 9 / 60 |
| Qwen3-0.6B (downloaded with approval) | query-IR compiler | superseded by the frame approach |
| GLiNER2, GLiNER2.5, Decide (extractors) | evidence and candidate scoring | Decide 78 % top-1 but 9.8 s per question; wording-sensitive |
| DeBERTa NLI, bge-small, e5-small | candidate scoring, retrieval | 53 to 76 %; none detect "no correct candidate" (AUROC at most 0.64) |

### 3.2 The reasons that decided it

1. **Memory.** About 400 MB at 4-bit; about 640 MB of int8 weights in the ONNX build; about 700 MB resident for the whole assistant on the Pi.
2. **Latency.** One short bounded decode (8 words) instead of a long prompt and long output. On a 2-core Cortex-A72 the cost is about 36 ms per prompt token and about 150 ms per decode pass; a 1.5 B model would be three times worse and over the memory limit.
3. **Safety.** A closed output cannot contain SQL, a path or a command, so prompt injection and hostile input cannot steer it.
4. **Measured behaviour, not size.** Frozen models were weak at choosing the operation; a *small trained* model with a closed vocabulary beat much larger frozen ones on this task.
5. **The case study's own hardware.** The A55 has dot-product instructions; llama.cpp may be fine there. The Pi 4's A72 does not, which is why the product uses ONNX Runtime int8 on the Pi.

## 4. How the model is run

* **Prompt**: a cached system prefix (instruction, vocabulary hints, and a one-line description of the car type: electric, petrol or diesel, taken from what the database records), then per-question tokens: the question, glossary hint lines, and optionally `Previous: <last frame>`. The system prefix is a saved model state, so the car-type line costs no time per question.
* **Token mask** (`TokenMask` in `vgpt/meaning/interface.py`): a trie over the ontology's word token sequences; at each step only tokens that can continue a valid word are allowed. It gives frames and confidences **identical** to llama.cpp's grammar sampler on 200 of 200 questions, and it removed the 62 % of parse time that the grammar sampler spent walking the 151,000-token vocabulary (laptop 905 ms to 348 ms).
* **Bounded decode and failing closed**: the decode stops at the longest valid frame; a decode that is not a frame returns nothing and the system asks again; a dead mask state fails closed. The confidence is the lowest probability over the tokens the mask allowed.
* **ONNX Runtime backend** (`vgpt/meaning/ort_backend.py`, built by `tools/build_ort_model.py`): Qwen2.5-0.5B Q4_K_M base with the LoRA added at its scale, every matrix stored as per-channel int8, per-token int8 activations, a cached system-prefix KV, and an output head that computes logits only for the allowed tokens (stored as rows of a `[tokens, hidden]` matrix, which cut a decode pass from about 166 to about 135 ms on the Pi). On the Pi: 90.9 % exact for both backends on 220 frozen questions (218 of 220 identical frames); parse median 1.44 s against 3.18 s with llama.cpp. On the laptop 99.1 % identical frames to llama.cpp.
* **Speculative drafts, lossless**: the frame is written one token per pass, and each pass over 640 MB of weights costs about 150 ms on the Pi, so a frame written one token at a time takes 3 to 5 passes. `vgpt/meaning/cues.py` proposes likely tokens (the concept words found in the question, and a word-order table learned from the training frames); the model verifies several in one pass and accepts only tokens it would itself have chosen. An extra drafted row costs about 25 ms.
* **No cache.** There is no frame cache, no answer cache and no read-ahead. A repeated question is a real run (owner's rules; see [11](11_decisions_and_rules.md)).
* **llama.cpp path** is kept as the reference and as the fallback (`VGPT_BACKEND=llama`, `VGPT_DECODER=grammar`).

## 5. The calibrated confidence

The raw score (lowest token probability) is not a probability. `vgpt/meaning/calibration.json` (made by `bench/calibrate_confidence.py` from the **development** sets only, never from sealed or acceptance sets, with the fit and the check on different halves of the questions) maps it to the share of questions that were actually read right at that score.

* Below **0.50** the answer is withheld: "I am not sure I understood that, so I will not guess. I read it as: ...". On the check half of 2,989 development questions this withholds 2.7 % and catches 17 % of wrong readings; none of the owner's real questions is withheld.
* The 2026-10-25 audit confirms the shape: pmin below 0.50 is right 25.6 % of the time; 0.50 to 0.80 is right 53.6 %; 0.80 to 0.95 is right 76.0 %; at or above 0.95 is right 95.7 %. But 58 % of the wrong answers had pmin at or above 0.8, so confidence cannot catch most errors.
* Ask-back ("did you mean 1 or 2") is implemented and turned **off** (owner's order); `VGPT_MEANING_TAU=0.8` restores it.

## 6. Training: data, recipe, gates

### 6.1 What the training data is

* **Generated rows** from templates over the ontology and glossary (`training/meaning/build_meaning.py`, `templates.py`, `noise.py`): consistent labels, wide coverage, plain wording.
* **Hand-written "natural" banks** (`training/meaning/natural_*.txt`): free wording written by people and by the assistant in the loop. Banks added over time: `natural_a/b/c` (the first 1,178 rows), `feedback`, `honesty` (what the car does not have), `hard1` to `hard3`, `adv1` to `adv5`, `platform`, `v10a/b`, `v10_ice`, `v11a-d`, `v12a/b`, `v13a`, plus `natural_v2` for the new frame-2 slots.
* **Auxiliary word tasks** (what a word means), and rehearsal rows of earlier banks.
* **Evaluation leakage control**: the builder filters training rows against the frozen evaluation questions (`--eval-leak vN`), and sealed sets are never read for failures and never trained on.
* A key finding of the ablation (2026-10-19): generated-only data gave 78 % exact; hand-written natural rows alone gave 90.6 %, about equal to everything combined (93.0 %); words alone gave 0 %. The natural banks are what teach real wording; the generated rows add coverage.

### 6.2 The recipe

* An own LoRA trainer (the installed `peft` was incompatible with the installed `transformers`), assistant-only loss, gradient checkpointing (9.8 GB peak otherwise on the 6 GB RTX 3050 development GPU), trained on the development laptop only. Nothing is ever trained on the vehicle.
* Typical runs: rank 16 to 32, alpha 2x rank, 3 to 4 epochs; a 35 MB adapter; 1 h 45 min of training plus about 25 min of evaluation for the largest rounds.
* The adapter is converted to a separate GGUF file (`training/convert_lora_gguf.py`); for the Pi it is merged into an int8 ONNX build.
* Adoption is **gated and pre-registered**: adopt only if the target sets improve by the stated margin and no frozen set loses more than 1.5 points; otherwise keep the old adapter. Gates were never loosened after the fact. Adoption is manual (`models/product.json`).

### 6.3 The lineage of adapters

| Adapter | Date | What it was | Outcome |
|---|---|---|---|
| `semantic-lora` v1/v2 | early Oct | 267 rows, 58 s of training | superseded; shipped as the legacy path |
| `full` to `full_v3` | 18 to 19 Oct | first meaning-frame adapters; hand-written banks and the honesty bank | novel 94.4, messy 95.7, sealed 91.3 to 94.0 (corrected scorer) |
| `full_v4` to `full_v6` | 19 Oct | adversarial banks `hard1` to `hard3`, feedback | **v6 default** until 20 Oct; training stopped by the owner |
| `full_v7`, `v8` | 20 Oct | `platform` and `adv1` banks | v7 rejected; **v8 adopted** (platform_v1 93.3, platform_v2 88.7, adv sets 86 to 88) |
| `full_v10` | 20 Oct | rank 32, 4 epochs, adv1 to adv3 | not adopted (user_v1 89.6, 2.9 below v6; platform_v2 85.0) |
| `full_v11` | 20 Oct | + `adv4` (174 rows) | target sets 92.2 to 95.0 % (target met, sealed sets never trained on); missed one gate by 2.2 points on a contested set; made the default at about 11:00 on 20 Oct when the owner stopped training; later the documented fallback adapter |
| `frame2`, `frame2b`, `frame2c`, `frame2m`, `frame2d` (and `frame2f`) | 21 Oct | frame version 2 (8 words): rounds on new hand-written data, and weight averages of two of them | **`frame2d` the product** on 21 Oct; owner then froze the model |
| `frame2w` | 21 to 23 Oct | adapter in use through 23 Oct | fresh questions 60.3 % strict (the figure to beat) |
| `frame2r10` | 24 Oct | 4 epochs on 28k rows, new label conventions R1 to R7 and C1 to C4, car-type line | step to r11 |
| `frame2r11` | 24 Oct | + 1 epoch lr 1e-4, + banks v11a to v11d | step to r12 |
| **`frame2r12`** | 24 Oct 12:00 | + 1 epoch lr 5e-5, + `natural_v12a` | **adopted**: older 3,038 questions 95.75 / 91.80 (reviewed / strict, was 94.96 / 90.72); fresh 677 strict 67.65 to 85.38; sealed_v7 (95 unseen, written after r12 was chosen) 70.5 to 84.2; fixed 98 of 149 genuine errors |
| `frame2r13` | 24 Oct 13:30 | a patch on r12 | **not adopted** (older 94.83, fresh 85.08); "stop patching the same adapter" |
| `frame2r1011` | 24 Oct | weight average 0.5 r10 + 0.5 r11 | evaluated, not adopted |
| per-car-type adapters (EV-only, ICE-only, random mix) | 24 Oct | a test of three adapters against one adapter plus a car-type line | one adapter plus the line won (ice_v1: 76.0 % with the hint against 66.9 % without; ICE-only fine-tune 74.4 %; random-mix control 77.7 %) |

### 6.4 Lessons recorded along the way

* **Training answers a different question than the product answers.** Frozen "development" sets overstate quality because they were used while developing; on questions written fresh the same adapter scored far lower (60.3 % strict on 368 fresh questions for `frame2w`, against 90.7 % on the frozen sets). Always measure on newly written, outside-the-loop questions.
* **A fine-tune flips about 100 old answers each way.** Judge by pooled sets, not by one set.
* **One meaning, one label encoding.** Two spellings of the same label collapse the model's confidence.
* **Your own rows can contradict the dominant convention** (for example a trips count using `ent=trip`); scan for conflicts before training.
* **The gold labels were not independently checked.** Reviewing the 283 frozen "errors" by what the person is shown showed that 149 were genuine and 134 were wrong gold or equivalent answers: the real error rate on frozen sets was 4.9 %, not 9.3 %.
* **Rule layers had been carrying the model.** In the 2026-10-20 ablation, model-alone accuracy on diesel and petrol sets was 37 to 40 %, and the vehicle-vocabulary layer lifted it to 89.5 and 75.7 %; after narrowing the layer and retraining it is now 95 % and 97 % on those sets, with every rule layer off changing 2 of 2,176 executed frames. The first "arbitration" rules changed 65 frames, fixed 2 and broke 56, so they were retired.
* **Vocabulary size is a latency matter, not an accuracy matter.** Only 14 of 175 concepts are single tokens, but in 81 % of confused concept pairs the two words already differ at their first token, so new tokens would help the number of decode steps, not the confusion.

## 7. Where the model still fails

See [09](09_results_and_evaluation.md) section 4 for the audit of 2026-10-25 and [10](10_research.md) for the methods proposed because ordinary training has stalled. In short: confident mistakes between near-synonym concepts (throttle against accelerator position, regen power against regen energy, fuel rate against Wh/km), the boundary between answer kinds (now against health), operator conventions, "unlisted" (things the car does not have), and unseen wording (slang, Hindi-English, figures of speech).
