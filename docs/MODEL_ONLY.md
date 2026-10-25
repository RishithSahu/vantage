# The model, and everything around it: what is learned and what is a rule (2026-10-20)

Written in answer to the owner's question: *the principle says code must not decide what a question means, but the handoff still contains `vgpt/meaning/arbitrate.py`, and the training queue admits rules for follow-ups, investigation routing and log-search terms.  How do we know a gain comes from better learning and not from another layer of phrase-based corrections?*

## 0. Short answer

* **The gains of v6 -> v11 are the model's.**  They are measured on the model alone (no layer of any kind): on the five sets about the new vocabulary and hard wording the exact frame went from **44.5 % to 93.9 %**; on the eleven older sets from 88.6 % to 89.9 % (`docs/V11_RESULTS.md` section 1; `bench/meaning_eval/eval_frames.py` does not call a single layer).
* **The rare-word arbitration (`arbitrate.arbitrate`) is retired and measured as harmful**: switched on over the 2,176 frozen questions it changes 65 frames, fixes 2 and breaks 56 (frame exact 90.2 % -> 87.7 %).  It is off by default and only reachable with `VGPT_ARBITRATE=1`.  The *file* `arbitrate.py` still exists because the layer that is on (below) lives in it; the name is misleading and is listed as debt.
* **Two rule layers *were* steering the model on the default path, and both hurt (both fixed today).**  (1) The "vehicle's own vocabulary" (`arbitrate.own_vocabulary`) was written to supply words only one vehicle brings (AdBlue, lambda), but it also let a generic word ("engine", "wheels", "coolant") replace the model's correct *"this car does not record that"*: **13 questions changed, 0 fixed, 10 broken** on the electric world (the product scored 89.8 % against the model's 90.2 %).  It is now limited to the names the vehicle's own profile declares.  Where it is meant to work the model cannot: on the diesel truck's own vocabulary the model alone reads **36.8 %** of 38 questions and the layer **89.5 %**; on the petrol car **40.5 % -> 75.7 %** (the first version of the layer: 84.2 % and 67.6 %).  (2) The investigation router sent 5 of 70 ordinary questions, wrapped in long chatty text, to an analysis template because one cue word appeared (answers matching a perfect reader fell from 90.0 % to 82.9 %); it now needs two cues, or a model that is not confident (section 3.2).
* **Every remaining code path that influences meaning can now be switched off one by one, or all at once** (`VGPT_LAYERS_OFF=all`), and a harness measures each against the model alone (`bench/meaning_eval/layer_ablation.py`).  After the fix the electric-world product equals the model alone, frame for frame (section 3.2).
* **What is still not the model's, and cannot yet be**: follow-up completion ("and yesterday?"), the investigation templates for long multi-part requests, the conversion of a meaning into a legacy handler's canonical sentence, and the safety guard.  For the first two there is no learned alternative today (v11 was trained on single questions), so they are not "corrections" of the model but the only implementation of those capabilities; the training queue holds the plan to learn them.

## 1. What the rule says and what the build actually enforces

`docs/semantic_method.md` 2.1: *code may parse values, enforce types and safety, and calculate; it may never decide what a question means.*  `tests/test_meaning.py::RuleBudget` enforces **less** than that: it counts calls of the form `re.<function>(...)` in the modules of `vgpt/meaning/` and fails above a small budget.  It does **not** see a word list consulted by plain string comparison, a cue dictionary in a JSON file, a rule outside `vgpt/meaning/` (`vgpt/investigate/`, `safety_policy.py`) or an old handler's regular expressions.  "Passes the budget" therefore never meant "no rule steers the model"; this file is the missing accounting.

## 2. Every place code still influences meaning

| layer (`vgpt/meaning/layers.py` name) | where | what it does | default | what the model alone would do |
|---|---|---|---|---|
| `arbitrate` | `arbitrate.arbitrate` | replaces the model's concept from word spans of the glossary (rules A / B / C) | **off** (retired) | the model's frame stands |
| `own_vocab` | `arbitrate.own_vocabulary`, `pipeline.py` | a signal that **this vehicle's own profile declares** (AdBlue, lambda ...), named in the question, replaces `unlisted` or a clearly different concept | on, **declared names only** | cannot name a signal it never saw: 37 % (diesel), 41 % (petrol) |
| `unsafe_floor` | `pipeline.py` | an `unsafe` reading with p < 0.70 is asked again, not refused (keyboard noise read as a request to tamper) | on | refuses nonsense input |
| `investigate` | `vgpt/investigate/parse.py`, `engine.py`, `pipeline.py` | a request of at least 14 words that is long (>= 24 words) or has several sentences, with **two or more** analysis cues (`lexicon.json`, 844 lines), is decomposed and answered by one of 32 analysis templates; with exactly one cue the model's own reading decides (p >= 0.95 keeps a single question on the frame path) | on | squeezes the request into one frame (the earlier behaviour; "one answer for many different asks") |
| `discourse` | `meaning/discourse.py` + `discourse.json` | completes a fragment from the previous turn ("and the week before that", "which one was the longest", "how did you get that") from closed-class cue words | on | reads the fragment as `ambiguous` (v11 was trained on single questions only) |
| `fuel_remap` | `meaning/fuelx.py` | for a vehicle whose data shows an engine and no battery, executes the model's electric reading ("power", "energy left") in engine terms; one text-dependent step (`power_asked`, from glossary tags) | on | answers a diesel in battery terms (the wrong-answer bug the owner reported) |
| safety guard | `vgpt/safety_policy.py` | a decision to keep driving with a hazard is made before and without the model | always on, not switchable | not allowed to depend on a model's confidence (the case study) |
| legacy adapter | `resolve._legacy`, `bindings.json` (43 entries) | a topic meaning (health, advice, estimate) is executed by an older handler that is handed a **machine-written canonical sentence** from a table, not the user's words | on | n/a (an executor, not a reader) |

The legacy adapter deserves its own sentence: the user's wording cannot steer it (it receives a fixed phrase chosen by the frame), but the handler it calls is old keyword code; the share of questions that reach it is a number, not an impression (section 3.4).

## 3. Measurements (v11, the electric 105-day world, 2,176 frozen questions; `layer_ablation.py`)

### 3.1 The model alone, across versions

From `docs/V11_RESULTS.md` section 1 (exact six-word frame, no layer): platform_v1 21.1 -> 93.3 -> **92.2** (v6, v8, v11), platform_v2 25.0 -> 88.7 -> **95.0**, adv_v1 58.0 -> 86.6 -> **94.6**, adv_v2 (sealed) 60.7 -> 87.5 -> **92.9**, adv_v3 (sealed) 57.5 -> 86.3 -> **95.0**; the eleven older sets 88.6 -> **89.9** on average.  No arbitration, no dialogue rule, no vehicle vocabulary was used in these numbers: the improvement is the model's.

### 3.2 The whole assistant with and without each layer

| configuration (2,176 frozen questions, electric world, v11) | frame exact | act + concept | same answer as a perfect reader |
|---|---|---|---|
| **the model alone** (every optional layer off) | **90.2 %** | 91.5 % | 91.3 % |
| the product as first measured (vehicle-vocabulary layer matching any signal by a generic word) | 89.8 % | 91.0 % | 90.8 % |
| **the product now** (vehicle vocabulary limited to declared names; investigation router trusts a confident model) | **90.2 %** | 91.5 % | identical to the model alone: 0 of 2,176 frames differ |
| the product with the retired arbitration switched back on | 87.7 % | 89.0 % | 88.8 % |

| the same, for 70 ordinary questions inside long chatty wrappers (`longwrap_v1`, 27-41 words) | frame exact | same answer as a perfect reader | taken over by the investigation router |
|---|---|---|---|
| the model alone | 87.1 % | 90.0 % | - |
| the product before the fix (any single cue word in a long text sent it to an analysis template) | 80.0 % | **82.9 %** | **5 of 70**, all five read correctly by the model |
| **the product now** | 87.1 % | 90.0 % | 0 of 70 |

The investigation router now needs **two or more analysis cues** to take a long request on its own; with exactly one cue the **model decides** (a confident reading, p >= 0.95, of an executable question keeps it on the single-frame path).  On the 62 analysis requests of `tests/investigation_prompts.tsv` that still reach the investigator: **61 of 62** (the one lost is read with p = 0.996 as a service-log question), and none of the 23 prompts that must not route does.

Reading it: frame exact is the share of questions whose executed frame equals the gold frame; "same answer" compares the answer with the one a perfect reader gets with every layer off.  *Changed / helped / hurt* count the questions whose executed frame differs from the model's, and whether the change made a wrong frame right (helped) or a right one wrong (hurt).

* `default+arbitrate`: 65 changed, **2 helped, 56 hurt**, 7 wrong-to-another-wrong.  Confirms the retirement.
* The first version of `own_vocab`: 13 changed, **0 helped, 10 hurt** (e.g. *"engine oil temperature"*, which this car does not record, became *motor temperature* because both contain "engine"; *"is the wheel alignment ok"* became *tyre pressure*; *"how is the heated steering wheel"* became *steering angle*).  Every one of the 13 had the model saying `unlisted`.
* `investigate`, `discourse`, `unsafe_floor`, `fuel_remap` changed **no** frame of these questions: they are short, single-turn, about an electric car.  This is *not* evidence for or against them: the sets cannot exercise them (see section 5).

### 3.3 Where the model cannot know: the vehicle's own vocabulary (new sets, `icevocab_diesel_v1`, `icevocab_petrol_v1`)

Written after v11 was final and before the layer was measured on these vehicles (75 questions: 26 + 25 questions naming a signal only that vehicle declares, in two or three wordings each, plus 12 controls per vehicle that the layer must leave alone).  Run on `world_rhino.db` (diesel truck) and `world_kestrel.db` (petrol hatchback).

| configuration | diesel (38) | petrol (37) |
|---|---|---|
| the model alone | 36.8 % | 40.5 % |
| the layer as first written (any signal, matched by a generic word) | 84.2 % | 67.6 % (helped 12, hurt 2) |
| **the layer limited to the vehicle's declared names** (the product now) | **89.5 %** (helped 20, hurt 0) | **75.7 %** (helped 13, hurt 0) |

The remaining misses are cases where the model reads a *known* concept (the long-term fuel trim read as a definition or as longitudinal acceleration, the air intake temperature as motor condition): only training can fix those (queued: the words of the diesel / petrol signals).  On the electric world the narrowed layer changes none of the 13 questions the first version broke (`tests/test_layers.py`).

### 3.4 What computes the answers (`executor_census.py`, gold frame, every layer off)

| | questions | share |
|---|---|---|
| typed or purpose-built executors | 1,901 | 87.4 % |
| the legacy adapter (older keyword handlers, fed a canonical sentence) | 268 | 12.3 % |
| no executor needed (a refusal or a greeting) | 7 | 0.3 % |

The legacy share is almost all topic questions: *health* 124 of 140 (88.6 %), *estimate* 77 of 78, *advice* 50 of 55.  They are the part of section 8 of `semantic_method.md` ("typed-argument versions are separate work") and not a way for the user's words to steer a result.

## 4. What changed in the product because of this

1. `vgpt/meaning/layers.py` (new): one switch per layer; `VGPT_LAYERS_OFF=name,name`, `VGPT_LAYERS_OFF=all` or `VGPT_MODEL_ONLY=1`; the pipeline, the investigator route, the discourse completion and the fuel remap consult it (`tests/test_layers.py`).
2. `own_vocab` limited to the vehicle's declared names (`VGPT_OWNVOCAB_ALL=1` restores the first version for comparison only).
3. The investigation router needs two analysis cues, or one cue and a model that does not confidently read the request as a single question (`pipeline.py`, `INVESTIGATE_TRUST`; `Investigator.strength`).
4. `bench/meaning_eval/layer_ablation.py`, `executor_census.py` and three new frozen sets: `icevocab_diesel_v1`, `icevocab_petrol_v1`, `longwrap_v1` (hashes in `*.sha256`).

## 5. What this does **not** show (read before quoting)

* **Follow-ups and investigations cannot be compared with a model alone**, because there is no model alone for them: v11 never saw a `Previous:` turn, and a long multi-part request is not a six-word frame.  Their value is measured by scripted dialogues (`tests/test_followups.py`) and by `tests/test_investigate.py` (32 analyses against injected ground truth), not by the frozen frame sets.  What would make it a *learned* capability is in `docs/TRAINING_QUEUE.md`: a continuation frame and a multi-slot investigation frame.  Until then they are rule layers, labelled as such, and can be switched off.
* **The investigation router is still a keyword scorer for the long requests it keeps**: it took over 0 of the 2,176 frozen questions (all short); the missing measurement was long single questions, now `longwrap_v1` (70, same author as the rest) - it found the 5 thefts, which the two-cue gate removed.  The 0.95 trust level and the two-cue minimum come from 62 analysis requests and 70 wrapped questions, i.e. they are interim numbers, not tuned on a large set.
* **One author wrote the training banks and most evaluation sets**; the two new vehicle-vocabulary sets were also written by the same author.  The vehicle-vocabulary result (37-41 % -> 76-90 %) is large enough that authorship does not change its direction, but its size is an upper bound.
* **Training is stopped** (owner's order).  Every item "to be learned" above is a plan, not work in progress.
* The electric-world numbers are for the aster world; the ICE sets are 75 questions.  One model run; no seed variance.

## 6. Reproduce

```bash
python bench/meaning_eval/layer_ablation.py --name v11_aster --oracle                     # 2,176 questions, 9 configurations, ~40 min on two cores
python bench/meaning_eval/layer_ablation.py --name ice_rhino --db world_rhino.db --sets icevocab_diesel_v1 --configs model_only default own_vocab_first_version
python bench/meaning_eval/layer_ablation.py --name ice_kestrel --db world_kestrel.db --sets icevocab_petrol_v1 --configs model_only default own_vocab_first_version
python bench/meaning_eval/layer_ablation.py --name longwrap --sets longwrap_v1 --oracle --configs model_only default
python bench/meaning_eval/executor_census.py
VGPT_LAYERS_OFF=all python ui/server.py --db data/world_aster.db                          # run the application with the model alone
```
