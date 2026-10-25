# Meaning from data: how VehicleGPT understands a question without a rule per question

*(2026-10-18. Companion to `results/meaning/` and `bench/meaning_eval/`. The Case Study 5 PDF is the binding specification; section 3 maps every constraint in it to this design.)*

## 1. What was wrong (measured, not assumed)

The owner's complaint: every failing question was fixed by writing another rule, which is "no better than humans writing the answers". The audit confirms the diagnosis and shows its cause.

| finding | evidence |
|---|---|
| The rule layer is the real language understanding | `vgpt/` holds **1,234 regex call sites and 1,734 long raw patterns in 24,600 lines**. `semantic_ops.py` alone has 189 regex calls (`_driver_semantics`: "people" + "fit" -> seating capacity, "motor" + "power" -> motor power, ...), `brain.ask` has a further 148. |
| The "0.5B semantic interface" never had enough to learn from | The adapter in the product (`semantic-lora-qwen25-05b-v2.gguf`) was trained on **267 synthetic examples in 58 seconds** (`training/out/semantic_qwen25_05b_v2/train_log.json`). Its own gate report: 100 % on synthetic validation, **75 % on 16 real failures** - a synthetic-to-natural gap that was never closed. |
| The architecture encodes "rules outrank the model" | `tests/test_driver_semantic_families.py::test_coarse_model_frame_cannot_preempt_exact_semantics` asserts that the rule layer must pre-empt the model. A better model could therefore never help. |
| The system answers 54 % of unpatched questions correctly | 357 frozen, never-patched questions (novel 287, messy 70) run through the real product (0.5B + all rule layers): **194 Y / 16 partial / 147 wrong**, plain wording only 58 %, typos 52 %, Hinglish 17 %, "how-to" 1/9, trend 3/15, at-a-time 3/12 (`results/meaning/current_product_judgments.json`; one judge, not blind). |
| Each fix cannot generalise | A rule encodes one phrase pattern. "how many ppl can fit" was fixed; "does it have 5 seats" and "gaadi mein kitne log baith sakte hain" were not (`N205`, `N208`). |

The root cause is not "too few rules". It is that **the system has no place where a word's meaning can be taught as data**, so a human has to teach it as code.

## 2. The method

### 2.1 The one rule that governs everything

> **Code may parse *values*, enforce *types and safety*, and *calculate*. It may never decide what a question *means*.**

* *Values* (numbers, units, dates, clock times, durations, numeric limits) are typed out of the text by `vgpt/meaning/literals.py`, the only text-pattern module in the meaning path.
* *Meaning* - which quantity, which operation, whether it is a fact about the car, a definition, advice, an instruction to operate the vehicle - is a prediction by the learned model into a closed, ontology-derived vocabulary.
* *Types* - whether a predicted meaning is well-formed (a live signal is not a vehicle specification; a comparison needs two operands) - are checked against the ontology, deterministically.
* *Human help* is limited to the two things only a human knows: **what the car's data can answer** (the ontology) and **what a word means** (the glossary). Both are data files. Neither contains a question, a trigger phrase or a route.

`tests/test_meaning.py::RuleBudget` enforces this: it counts text-pattern call sites per module and fails the build if any module other than the literal parser uses one, and it fails if a meaning-path module imports a rule layer.

### 2.2 The pipeline

```
question
  -> deterministic safety guard           (decision-under-hazard; never depends on a model's confidence - PDF: safety isolated from the AI)
  -> glossary lookup (dictionary, O(1))    adds "Terms: ppl = people (passengers)" ONLY when the question contains a flagged rare word
  -> learned frame (Qwen2.5-0.5B + LoRA, GBNF-constrained, ~11 tokens)     act concept op cond vs ent
  -> ontology typing                       ill-typed -> a specific clarification, never an executed guess
  -> resolver                              frame -> typed Query (existing engine) | instant lookup | profile fact | definition | method | outcome
  -> evidence-backed answer                (existing composer/verifier)
```

The **frame** is six words from a closed vocabulary generated from the ontology:

| slot | values |
|---|---|
| `act` (what kind of question) | now, at, stat, extreme, trend, compare, events, context, why, spec, define, howto, advice, health, estimate, social, offtopic, unsupported, control, unsafe, ambiguous |
| `concept` (what it is about) | 118 ids: 73 from the registry (40 recorded signals, 13 derived metrics, 20 known-but-often-unrecorded), 16 events, 13 vehicle specifications, 14 whole-subject topics / entities, plus `unlisted` (a measurable thing this vehicle does not model) and `none` |
| `op` | none, mean, median, max, min, sum, count, share, duration, spread, first, last, change, drop, rise, list, exists |
| `cond` | none, threshold, one of 15 driving conditions, or `not_` + one |
| `vs` | none, period, or a second condition (comparisons) |
| `ent` | none, trip, charge, day |

This decomposition is what separates the distinctions the previous approach hard-coded per question: *state of charge* vs *capacity* vs *charging power* vs *cycles* are different `concept`s; *current speed* vs *rated speed* are `now speed` vs `spec rated_max_speed`; *motor power* vs *battery power* are two concepts that share a quantity and differ in component; *exact-time lookup* vs *aggregate* is `at` vs `stat`. The ontology derives these relations from structure (`Ontology.relations`), so contrast training data is generated, not hand-picked.

### 2.3 Files

| file | role |
|---|---|
| `vgpt/meaning/ontology_data.json` | acts, operators, spec and topic concepts, event words, where each profile fact is stored |
| `vgpt/signal_registry.py` | signals, derived metrics, events, named conditions (unchanged; the ontology reads it) |
| `vgpt/meaning/lexicon.json` | the noun phrases people use for each concept (vocabulary, 762 phrases over 116 concepts) |
| `vgpt/meaning/glossary.jsonl` | what words, abbreviations, units and slang *mean* (70 entries); the only sanctioned human help |
| `vgpt/meaning/bindings.json` | which executor serves a meaning the query engine does not express as one query (legacy adapter phrases, formulas, spec counterparts) |
| `vgpt/meaning/{ontology,frame,glossary,interface,compile,resolve,pipeline,literals}.py` | the code |
| `training/meaning/` | data builder, templates, natural banks, trainer wrapper, ablation driver |
| `bench/meaning_eval/` | frozen evaluation sets, evaluators, perf probe |
| `tools_triage.py` | turns a failing question into a data change |

## 3. The Case Study 5 PDF, clause by clause

| PDF requirement | how this design meets it |
|---|---|
| Understand natural-language vehicle queries; accuracy | learned frame over a typed ontology; measured on frozen unseen sets (section 6) |
| The five example queries (range reduction; largest efficiency drop; ADAS disengagements yesterday; before the overheating warning; city vs highway) | each is a frame (`why range_estimate drop`, `trend wh_per_km drop`, `events adas_disengagement list`, `context battery_overheat_warning`, `compare wh_per_km mean city highway`); the 63-question CS5 end-to-end set is a separate evaluation |
| Trends, anomalies, historical events, explain causes | `trend` / `events` / `context` / `why` acts executed by the existing typed engine |
| No cloud | nothing in the runtime path leaves the device. Training data was authored offline (disclosed in section 5); no cloud model is called at run time |
| ~2 GB RAM, ~2 ARM A5x cores | one 0.5B Q4 model (~400 MB) + 9 MB LoRA; glossary and ontology are KB-sized; no second model, no embedding model, no retrieval index. Measured on 2 pinned x86 cores (section 6); ARM not measured |
| < 2 s for common queries | one short constrained decode (about 11 tokens) over a cached system prompt; deterministic typed execution. Latency is reported honestly; the ARM figure is an open item |
| Weeks of history searchable | unchanged: the existing SQLite store and typed engine |
| Safety-critical functions isolated | read-only executors; `control` and `unsafe` are non-executing acts; the hazard-decision guard runs before the model and does not depend on it |
| Local-only processing, privacy | no network call in the path; the glossary / ontology / frames contain no user data |
| Deliverable headings (architecture, collection, SLM selection, retrieval workflow, storage, edge deployment, UI) | this document updates *SLM selection rationale* (section 4) and *search and retrieval workflow* (section 2.2); the other headings in `DESIGN.md` are unchanged |

## 4. Why this training method (research basis)

The question the owner raised: *train on words and their correlations rather than on questions?* The evidence says: **partly.**

* **Word meanings help most when they are supplied, not memorised.** Retrieving WordNet glosses into the prompt nearly doubled out-of-distribution concept prediction in [RASP (2025)](https://arxiv.org/abs/2412.10207); computing a rare word's representation from its definition on the fly beat trained-per-word embeddings in [Bahdanau et al.](https://arxiv.org/pdf/1706.00286); [Dict-BERT](https://arxiv.org/pdf/2110.06490) adds definitions of rare words during pre-training. This is the glossary: a one-line meaning added to the prompt only when a rare word occurs.
* **Memorising correlations by LoRA is the wrong tool.** [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2) finds the gap to full fine-tuning is largest for continued pre-training on unlabeled domain text and closes for instruction-style fine-tuning. Domain-adaptive pre-training helps most when the domain is distant ([Gururangan et al.](https://aclanthology.org/2020.acl-main.740)), but it needs far more text and compute than a 6 GB laptop GPU offers. Our own earlier "question-free" experiment (ALMA, `results/ALMA_FEASIBILITY_VERDICT.md`) raised task recognition from 27.8 % to 59.8 % but left attachment/scope at 33-50 % and *damaged* signal linking (66.7 % -> 39.1 %) when a shared encoder was fine-tuned on fragments: words alone do not teach composition.
* **Synthetic data from an ontology works, and has a known ceiling.** Grammar/ontology-generated questions plus paraphrase train usable parsers (Overnight; [AutoQA](https://aclanthology.org/2020.emnlp-main.31/)), but real user language differs from canonical paraphrase - the *language gap* ([Yin et al. 2022](https://aclanthology.org/2022.acl-long.103/)); closing it needs structurally diverse natural text ([Niu et al. 2023](https://aclanthology.org/2023.tacl-1.22)).
* **Contrastive hard negatives and abstention are what make selection robust.** [Octopus](https://arxiv.org/abs/2404.01549) trains with three semantically similar functions per example and a curriculum; [Hammer](https://arxiv.org/html/2410.04587v2) adds ~10 % "irrelevance" examples (the right function removed -> no call) and masks identifiers so the model reads meaning, not names; [few-shot intent detection](https://aclanthology.org/2021.emnlp-main.144) uses supervised contrastive learning for fine-grained confusable intents.
* **Template diversity is not natural diversity.** LLM-written data collapses to typical phrasing unless diversity is engineered ([SynthesizRR](https://arxiv.org/html/2405.10040v1)).
* **Confidence should drive clarification, not answers** ([Calibrated Interpretation](https://ar5iv.labs.arxiv.org/html/2211.07443)); here the frame is also human-readable, so a low-confidence turn can show what it understood.

**What this implies, and what was built.** Training mixes four sources (each switchable, so the claim can be tested): word-level tasks (`Term: phrase -> concept`, `Pair: a | b -> same_concept / same_quantity / same_component / unrelated`, generated from the ontology's structure - the "words and their correlations" part); compositional sentences generated from the ontology with hard contrasts and abstention (~10 %); hand-written natural sentences; and glossary-hint rows (the model sees `Terms:` lines in 85 % of rows that contain a flagged word). Surface noise (typos, abbreviations, fillers, code-mixed forms) is part of training rather than a repair rule at run time. Ablations (section 6) measure which source matters.

## 5. Honest limits of the method

* The same author (Claude, offline) wrote the natural training banks and the primary evaluation sets. The training banks were written *after* the sets were frozen (SHA-256 in `bench/meaning_eval/*.sha256`), rows overlapping any frozen set (exact or Jaccard >= 0.8) were removed (`manifest.json` counts them), and two sets by other authors (the owner's real failing questions, `user_v1`; the Case Study 5 bundle by a different session) are reported separately. This reduces, but does not remove, vocabulary leakage between author-written training and evaluation text.
* No cloud model is used at run time. Training data was written by a cloud LLM (Claude) at development time; a local 7B teacher was not available (the llama.cpp build is CPU-only and the 7B file is not on disk).
* The 0.5B model decides; it can be wrong. Safety does not depend on it (guard before it, read-only executors after it); correctness is measured, not assumed (section 6).
* Executors for *topic* meanings (overall health, advice, charging time/cost, service) are the older keyword route handlers called with a machine-written canonical request (`bindings.json` "legacy"). They no longer decide *which* handler runs (the frame does), but they still contain keyword logic inside. Migrating them to typed arguments is separate work; the coverage list of meanings with no executor is a data output (`facts.coverage_gap`).
* Latency and memory were measured on x86 pinned to two cores, not on an ARM A5x board (QEMU runs remain paused by the owner's instruction).

## 6. Results (all measured on this machine; frozen sets; one judge for answer correctness)

### 6.1 The training run
`training/meaning/data/full`: **10,175 training rows** = 7,725 question -> frame rows (6,221 ontology-generated, 1,166 hand-written natural, 338 abstention) + 1,005 `Term` + 1,445 `Pair` rows; 426 validation rows; 218 held-out-combination rows (20 (act, concept, operator) combinations removed from training entirely). 60 candidate rows were dropped for overlapping a frozen evaluation set. LoRA r16 on Qwen2.5-0.5B-Instruct Q4, 3 epochs, **55 minutes on the laptop RTX 3050 (2.2 GB peak with activation checkpointing)**, adapter 17.6 MB. Teacher-forced exact frame on validation 97.0 %, on the held-out combinations 81.2 % (`training/out/meaning_full/train_log.json`).

### 6.2 Understanding: exact six-word frame (`bench/meaning_eval/eval_frames.py`)
| set (author) | n | exact frame | act | concept | act + concept | false execution | false abstention |
|---|---|---|---|---|---|---|---|
| novel_v1 (Claude, frozen before training data) | 287 | **93.0 %** | 96.2 | 96.9 | 94.4 | 1 / 39 | 0 / 248 |
| messy_v1 (typos, slang, voice, Hinglish) | 70 | **92.9 %** | 95.7 | 98.6 | 94.3 | 0 / 7 | 0 / 63 |
| sealed_v2 (Claude, sealed before any model existed) | 149 | **91.3 %** | 97.3 | 96.6 | 96.0 | 0 / 19 | 1 / 130 |
| user_v1 (the owner's real failing questions, never trained on) | 67 | **74.6 %** | 86.6 | 86.6 | 79.1 | 0 / 4 | 0 / 63 |
| Case Study 5 end-to-end bundle (a different author; act in an allowed set + concept) | 62 | 87.1 % | | | | | |

*False execution* = a request that must not run (social, off-topic, unsupported, control, unsafe, ambiguous) read as an executable question: **1 of 69**. *False abstention* = an answerable question read as one of those: **1 of 504**. `sealed_v2` has 8 of 149 questions whose wording overlaps a training row at Jaccard >= 0.8 (templated phrases such as "what preceded the tyre warning"); this is reported, not hidden.

### 6.3 Answers: what the owner experiences (end to end, `bench/meaning_eval/run_pipeline.py`, judged by reading)
| | old product (0.5B + all rule layers) | meaning pipeline |
|---|---|---|
| novel_v1 (287) | 160 correct (55.7 %) | **260 (90.6 %)** |
| messy_v1 (70) | 34 (48.6 %) | **66 (94.3 %)** |
| sealed_v2 (149) | 64 (43.0 %) | **138 (92.6 %)** |
| **the three never-patched sets (506)** | **258 (51.0 %)** | **464 (91.7 %)** |
| user_v1 (67; the old product's rules were written for exactly these) | ~67 (patched) | 50 (74.6 %), never seen |

By kind of question, **on the same 506 questions** (correct / asked, old -> new): how-to 1/13 -> 13/13; trend 5/24 -> 23/24; value at a time 5/17 -> 17/17; live value 29/64 -> 63/64; extremes 12/29 -> 27/29; specifications 17/39 -> 36/39; why 10/29 -> 22/29; events 29/46 -> 42/46; statistics 31/49 -> 43/49; advice 5/19 -> 13/19; context 4/13 -> 10/13; definitions 14/21 -> 20/21; health 24/33 -> 31/33. By register: Hinglish 2/12 -> 7/12; typos 12/23 -> 22/23; slang 18/43 -> 42/43; terse 45/73 -> 68/73; plain wording 117/202 -> 184/202; the sealed set's fresh phrasing 64/149 -> 138/149.

**Reading the user_v1 row correctly.** The old product answers those 67 questions almost perfectly because each one was fixed with a pattern: that is the problem, and on fresh wording it falls to 43-56 %. The meaning pipeline had never seen those questions and gets 75 %. The failures are concept-boundary calls the ontology leaves ambiguous (odometer vs distance driven; "how fast can we go" as a rating vs a range; "motor power" now vs a specification), plus two forms the executors cannot yet compute (acceleration between two speeds; an explicit "from 3 to 5 days ago" interval). They are closed by data (section 7), not by code.

With **perfect understanding** the executors answer 408 of 424 (96.2 %) of the novel / messy / user gold-frame questions correctly (`results/meaning/oracle_judgments.json`); the remainder are explicit gaps (trip averages, per-day averages of accumulating quantities, intervals, event context when no samples exist). Executor limits and understanding errors are therefore separable: the model is not the bottleneck for most of the remaining wrong answers.

### 6.4 The glossary mechanism works as designed
A hint line was shown on 164 of 573 questions. Exact frames on those: **146 with hints, 143 without** (inference-only ablation, same weights): hints fixed 4 (the Hindi "gaadi ... kitne", "adaptive cruise", "ADAS" twice) and broke 1 ("engine"). Per set (hints off): novel 92.3 (93.0 with), messy 92.9 (same), user_v1 74.6 (same), sealed_v2 90.6 (91.3). A missing word meaning is therefore a one-line data fix that takes effect at once.

### 6.5 Confidence and abstention
The lowest emitted-token probability separates correct from wrong frames (AUROC 0.79-0.89 per set). Over all 573 questions: threshold 0.8 answers 90 % of questions at 94.2 % exact (30 wrong answered, 35 correct held back); 0.95 answers 80 % at 96.9 %. The gate is off by default (`VGPT_MEANING_TAU`); with it on, a held-back question shows the plain-words reading ("I read it as: ...") instead of guessing.

### 6.6 Latency and memory (the PDF's budgets; x86, **not ARM**)
Two logical E-cores pinned (`tools_pin.py 8,9`), Q4 0.5B + LoRA + typed engine, steady state after the first question: **frame parse p50 1.08 s, p95 1.69 s, max 2.28 s; total answer p50 1.10 s, p95 1.71 s; mean 8.2 decoded tokens; resident set 624 MB after load, 679 MB peak.** A repeated question hits a frame cache. The first question after boot pays one prefill of the ~200-token prompt (the earlier product keeps a persistent KV state file for this; not yet added here). An ARM A55 core is slower than these cores, so the < 2 s target is **not demonstrated on target hardware**; the decode length (8 tokens) is the lever to shorten if it is missed.

### 6.7 Tests
`tests/test_meaning.py`: 19 tests, all pass, including the rule budget and the guarantee that a typed frame never reaches the legacy router. The pre-existing suite (`python -m pytest tests`) has 42 failing tests (legacy rule routing, UI, round-52 fixtures, `test_bug_layers`); none touches the new modules and the only existing file I changed is `vgpt/product.py` (an opt-in branch). I did not triage those 42.

### 6.8 Second round (2026-10-18 night): ablations, probe on unseen wording, bugs found

* **Ablation (frame exact, sealed_v2)**: words_only 0.0 % (act+concept 1.4 %); natural_only 89.9 %; full 91.3 %; **full_v2 (+ feedback rows) 91.9 %**. On the dev sets full_v2 scores novel 94.8, messy 97.1, user 82.1 (these sets taught it, so they are development numbers); CS5 corrected scorer 90.3 %. The remaining ablation arms were run afterwards; all of them are in section 6.9, the v1 / v2 / v3 comparison is in 6.10.
* **Probe on wording nobody authored for this model** (`bench/meaning_eval/probe_answers.py`, the legacy everyday / held-out / messy banks, 270 questions, read by eye, not a measurement): most answers are right, but it exposed what the frozen sets could not: (a) a typed calendar date was ignored (`17 March 2025 09:13:27` was answered with today's 09:13: **a hallucinated 0.0 km/h**), now parsed (`literals.date_of`); (b) counterfactuals ("would the fault still have occurred on a flat road?") were refused as tampering; predictions of faults were answered with the service interval; blame / part / CAN-id / fleet / radar / passenger questions were mis-read as other concepts; (c) executor gaps: charging state reported as "not recorded", trips could not be described, a fault code was not looked up, `firmware version when the fault occurred` was a typing error. (b) is training data (`training/meaning/natural_honesty.txt`, source `honesty`, used by full_v3 and later), (c) is fixed in the resolver / bindings.
* **Code bugs fixed (not model)**: average speed refused when the 1 Hz tier did not span the period (`analytics.summary`); `AttributeError` on trend of energy / distance / Wh per km beyond 7 days (`query_executor._finish` with no sample frame); `no such table` crash in `semantic_ops`; legacy-adapter exceptions now become an honest sentence; the units change did not re-show the last answer; a tyre-pressure question answered with a tyre definition; a mAh-with-voltage question answered as a capacity statement; `driving score` pre-empted by a later rule; present speed answered outside the typed engine (dashboard / assistant contract); camera staleness wording; product factory test and the retired interpretation-review prototype (skipped with the reason).
* **Test suite**: 41 failing tests at the start of the night, 13 now (863 + passing). All 13 are the legacy rule router (`qa`, `qa2`, `routing`, `round52`): with `semantic_ops.handle` disabled most pass again, i.e. the hand-written pre-emption layer (not the meaning path) is what broke the older router; the DBC-decode / fleet answers are intercepted by the typed capability layer. They are not chased, because the meaning path replaces that front end.

### 6.9 Ablations: which training data does the work (answers the owner's question)

The owner asked whether the model should be trained on **words and their correlations** instead of on questions. Each arm below was trained from its own data directory with everything else equal (same ontology, glossary, lexicon, seed 7 data seed, LoRA r16 / alpha 32, lr 2e-4, batch 16, `max(3, 600 steps' worth)` epochs, one training run per arm). The arms are built with `build_meaning.py --eval-leak v1`, which reproduced the `full` model's data byte for byte (train / val / test_comp hashes identical) when I checked it before the first arm trained, so an arm should differ from `full` only in the sources it keeps. **That held for `words_only` and `natural_only` but not fully for the later arms: see "Input drift" below.** Evaluation is the same frozen sets, 2 CPU threads. `training/meaning/run_arms.py` rebuilds every row of the table.

| arm | trains on | rows | epochs | novel_v1 | messy_v1 | user_v1 | cs5 (act+concept) | sealed_v2 | answerable questions refused (3 dev sets, of 374) |
|---|---|---|---|---|---|---|---|---|---|
| **full** | words + generated sentences + hand-written sentences + abstention | 10,175 | 3 | **93.0** | **92.9** | **74.6** | 87.1 | **91.3** | 0 |
| words_only | `Term` / `Pair` word tasks only; no question is ever seen | 2,447 | 4 | 0.0 | 0.0 | 0.0 | 3.2 | 0.0 | 151 |
| natural_only | the hand-written sentences only (with their own abstain rows) | 1,178 | 9 | 90.6 | 90.0 | 70.1 | 82.3 | 89.9 | 2 |
| template_only | ontology-generated sentences + abstain pool, no hand-written sentence | 6,564 | 3 | 78.0 | 72.9 | 52.2 | 77.4 | 77.2 | 28 |
| no_aux | `full` without the word tasks | 7,761 | 3 | 91.3 | 88.6 | 59.7 | 83.9 | 89.9 | 7 |
| no_hints | `full`, but the model never sees a glossary line in training or at test | 10,213 | 3 | 92.7 | 90.0 | 73.1 | 87.1 | 94.0 | 5 |
| full, glossary hints off at inference only (6.4) | the same weights | | | 92.3 | 92.9 | 74.6 | 87.1 | 90.6 | |

Exact six-slot frame in percent (cs5: the coarse act-and-concept score). 287 / 70 / 67 / 149 questions, so one question is 0.35 / 1.4 / 1.5 / 0.7 points. Differences below about three questions per set are inside the noise of a single training run (see 6.10); the table gives the sign-test p-value of the questions that changed between two arms where it matters.

**Plain-language conclusions**
1. **Word-level training alone does not work: it scores 0 % exact on every set** (act + concept right on 1.4 % of novel questions). It was trained on 1,005 word -> concept examples and 1,445 word-pair relations and never saw a question, so it cannot read one: on the 287 novel questions it emits 251 different frames, essentially a guess each time, and it refuses 151 of 374 answerable questions. This matches the earlier ALMA result (section 4): words alone do not teach how a question is composed.
2. **Words help as an addition, modestly.** Removing the word tasks from `full` (no_aux) costs 1.7 points on novel, 4.3 on messy, 14.9 on user_v1 and 1.4 on sealed_v2. Only the user_v1 loss is statistically firm (13 questions lost, 3 gained, p = 0.02; novel p = 0.23, messy p = 0.25, sealed p = 0.75) and user_v1 is the set written by someone else, which is the register the word tasks are meant to help. So: worth keeping, not the main source of understanding.
3. **Hand-written sentences matter far more than generated ones.** 1,178 hand-written rows (natural_only, about an eighth of the data) already reach 90.6 / 90.0 / 70.1 / 89.9 against 93.0 / 92.9 / 74.6 / 91.3 for everything together, and none of the four gaps is significant (p = 0.23, 0.63, 0.63, 0.79). Six and a half thousand ontology-generated sentences alone (template_only) are much worse than both `full` (p < 0.001 on novel, messy and sealed_v2, 0.001 on user_v1) and natural_only (p < 0.001 novel, 0.004 messy, 0.017 user_v1), and refuse 28 answerable questions where `full` refuses none. Generated sentences cover the ontology but not how people phrase things. The same ordering holds on user_v1, written by a different author (natural_only 70.1, template_only 52.2), so it is not only the shared style of one author.
4. **What the generated sentences and word tasks add on top of hand-written ones** is small and not individually significant: `full` minus natural_only is +2.4 / +2.9 / +4.5 / +1.4 points. The practical recipe is: collect natural questions with their frames first (they are cheap: one line each), use generation to widen coverage and the contrast pairs, and keep the word tasks and glossary as a small extra.
5. **The glossary hint is not what the average score rests on.** A model trained and tested without any glossary line (no_hints) is within noise of `full` on every set (novel 92.7 vs 93.0, messy 90.0 vs 92.9, user_v1 73.1 vs 74.6, cs5 87.1 vs 87.1; sealed_v2 94.0 vs 91.3; all p >= 0.29 by the sign test), which agrees with 6.4 (hints off at test time: about +3 questions net on the 164 hinted ones). The glossary's value is operational, not statistical: a word meaning is one data line that takes effect without retraining. (`no_hints` was built after the ontology and glossary edits described in the input-drift note below, so its data differ from `full`'s in more than the hint lines: 10,213 rows against 10,175, with 2,464 word-task rows against 2,450 and 6,256 generated rows against 6,221.)

**Input drift (found afterwards, from the SHA-256 values each arm's `manifest.json` records).** While the arms ran, `vgpt/meaning/ontology_data.json` (changed at least twice) and `vgpt/meaning/glossary.jsonl` (70 -> 89 entries, 01:47 on 2026-10-19) were edited by another working session, and `build_meaning.py` was extended. So the arms were not all built from identical inputs: `full`, `words_only`, `natural_only` and `full_v2` used the original ontology file (`2fdfa15d`) and glossary (`ec14be19`); `template_only` and `full_v3` the first edited ontology (`ce6a1c5d`) with the original glossary; `no_aux` the first edited ontology and the larger glossary (`37b4187c`); `no_hints` a second edited ontology (`761ff33d`) and the larger glossary. The concept vocabulary still has 118 ids (as documented before) and the system prompt is built only from the act / operator / condition / entity lists, so as far as I can tell the model's input and output format did not change; the differences are in what the generated word-task and sentence rows contain (a few dozen rows, for example 2,447 -> 2,464 word-task rows) and, for `template_only`, in the hint lines shown at test time (it was trained with the smaller glossary and evaluated with the larger one). I cannot quantify the effect (no copy of the earlier files exists to rebuild from), but it is small next to the 15-25 point gaps that carry the conclusions (words only, generated only, hand-written only), and it is **not** small next to the 1-4 point gaps, which is one more reason not to read those as effects. A clean repeat needs all arms rebuilt from one frozen ontology / glossary (the manifests now make any drift visible).

Caveats that change how far to trust this: one training run (seed) per arm; the same author (Claude) wrote the natural banks and three of the four evaluation sets, which could favour natural_only a little (user_v1 is the guard against that); the small arms train for more epochs (natural_only 9, words_only 4) by a fixed rule that gives every arm at least about 600 optimizer steps, not by tuning; cs5 here is the coarse score of `eval_frames.py` (the corrected scorer in 6.8 gives higher numbers). The `natural_only` arm needed a data-builder fix: the old code dropped its hand-written abstain rows, contradicting the arm's own definition (new source `natural_abstain`; no other arm's data changed).

### 6.10 The failure-to-data loop run once (v1 -> v2 -> v3)

*What was done.* The 42 failing rows of the first `full` model on novel_v1 (20), messy_v1 (5) and user_v1 (17) showed these confusable families: odometer vs distance driven; "how fast can we go" (rated speed vs range, fuel words); home charger power vs export power; the 12 V battery (why / define / health); `unsafe` vs `control` (disabling a warning light); Hinglish event / health / "why" forms; maintenance timing (estimate) vs maintenance advice; range as an estimate vs a live value; compare two conditions read as a definition; best / worst efficiency polarity; period sums read as means; and a few typing slips. `tools_triage.py` refuses a question that is in a frozen set, so the failing rows themselves could not be used. Instead 187 **new-wording** questions for those families were written (none a copy; `build_meaning.py` additionally drops any row at Jaccard >= 0.8 to any frozen set and dropped none of the final 187; one draft row that did overlap an evaluation question was removed before the run) and passed through `tools_triage.py --batch`: the first model already read 97 correctly and misread 90. The 18 it classed `model` were appended by the tool; the other 72 it classed `word` only because a rare English word was not in the glossary (a heuristic false alarm: "lifetime", "wallbox"), so they were added as training rows too and no glossary meaning was invented. The 97 correct ones were kept as the other side of each pair. All 187 rows are in `training/meaning/natural_feedback.txt`; `sealed_v2` was never opened while writing them.

*What it exposed beyond the model.* (a) The first model had **never seen one `spec home_charger_power` training row**: that combination is one of the 20 held-out combinations, so every such row had gone to the held-out test, which alone explains "how powerful is my home charger" -> export power. Feedback rows are therefore never held out (11 rows touch that combination). (b) The generated data labels "current driving range" as `now range_estimate` (11 rows), while the evaluation convention is `estimate range_estimate`: the data contradicts itself; not fixed here. (c) The natural bank labels "longest trip last month" with `distance_driven`, while the evaluation set labels "longest trip" with `trip_record`; likewise open.

*Builder changes.* `--eval-leak v2` adds `sealed_v2.tsv` to the leakage filter (10 candidate rows removed from the existing data, none of them feedback; every other row is unchanged); `--eval-leak v1` keeps the original filter; new sources `feedback` and `natural_abstain`. `full_v2` = `full` + feedback (10,394 rows, 3 epochs, 59 minutes alone). `full_v3` = `full_v2` + `natural_honesty.txt` (257 rows on predictions, counterfactuals, blame, fleet and unrecorded content, written by another session; section 6.8) with `sealed_v3` added to the filter.

| exact frame, % | status of the set | full (v1) | full_v2 | full_v3 |
|---|---|---|---|---|
| novel_v1 (287) | development-exposed | 93.0 | 94.8 | 94.4 |
| messy_v1 (70) | development-exposed | 92.9 | 97.1 | 95.7 |
| user_v1 (67) | development-exposed | 74.6 | 82.1 | 85.1 |
| cs5 (62, act + concept) | coarse score | 87.1 | 83.9 | 83.9 |
| sealed_v2 (149) | clean for v2; v3's honesty rows were written by another session, so I cannot vouch it stayed unseen for v3 | 91.3 | 91.9 | 94.0 |
| sealed_v3 (149) | clean, the final set (not read by me) | 92.6 | 91.3 | 91.3 |
| answerable questions refused (3 dev sets, of 374) | | 0 | 6 | 3 |

*How to read it.* The gains on novel, messy and user_v1 are **development numbers**: the rows were written from those sets' failures, so they show that the loop fixes the wordings it was shown, not that the model generalises better. On the two clean sets the changes are one to four questions (sealed_v2: 136 / 137 / 140 of 149, v3 against v1 p = 0.22; sealed_v3: 138 / 136 / 136, v3 against v1 p = 0.73) and the sign is not consistent, so **there is no measurable gain, and no measurable loss, on unseen wording.** A single retraining moves 5 to 8 questions in each direction by itself: v2 against v3 on sealed_v3 is 136 against 136 with 5 questions each way; on user_v1, v2 fixed 12 and broke 7 of v1's answers (it fixed "how fast can we go", the petrol-left, city-versus-highway, acceleration-between-speeds and "engine overheat" forms; it broke "when to service" -> advice, "tell tyres" -> ambiguous, "ADAS" -> control; v3 won the "when to service" ones back but lost "did betry get hoy" and gained other wrong ones). Targeted rows do not always win: "how many miles have we gone" (odometer) and "what is our motor power" (now, not a specification) are still wrong in v2 and v3 although rows were written for exactly those families. Two things worth keeping: false executions of requests that must not run stayed at 0-1 of 50, and v2 refused 6 answerable dev questions where v1 refused none (v3: 3).

What this says about the method: the loop is cheap (one sitting of writing rows and about an hour of training) and fixes many of the wordings it is shown, but a hand-written row is a weak steering signal against thousands of generated rows; to make a family stick, contradicting generated data has to change too ((b) above), and a fix must be judged on a set it did not come from, with a margin of more than the 5-8 questions a retrain moves by itself. Two retrains cannot be compared question by question without more than one seed.

### 6.11 The hard-set loops (2026-10-19): what generalises and what does not

Hard sets are written by me *before* the model sees them, frozen with a SHA-256, and become development material the moment their failure families enter training; each loop gets a fresh set.  Numbers are exact six-word frame (act, concept, op, cond, vs, ent all right).

| adapter | data added | novel | messy | user_v1 | sealed_v2 | sealed_v3 | CS5 (corrected) | hard_v1 | hard_v2 | hard_v3 | hard_v4 | hard_v5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_v3 | feedback + honesty bank | 94.4 | 95.7 | 85.1 | 94.0 | 91.3 | 90.3 | 55.5 | 80.9 | - | - | - |
| full_v4 | + hard1 (families of hard_v1) | 95.1 | 95.7 | 86.6 | 96.0 | 91.3 | 91.9 | **85.0 (dev)** | 80.9 (clean) | 87.3 | - | - |
| full_v5 | + hard2 (families of hard_v2) | 95.1 | 95.7 | 86.6 | 94.6 | 91.3 | 90.3 | 81.4 | **88.2 (dev)** | 87.3 (clean) | 70.0 (clean) | 99.0 (clean) |
| full_v6 | + hard3 (families of hard_v3 / hard_v4), x2 weight, politeness / hedging wrappers on 25 % of noised rows | 93.4 | 94.3 | **94.0 (dev)** | 94.0 | **93.3** | 90.3 | 79.5 | 88.6 | 89.1 (dev) | 79.5 (dev) | 95.0 (clean) |

* **Reading it honestly.** A set that taught a round jumps (hard_v1 55 -> 85, hard_v2 81 -> 88); a set the round never saw does not move (hard_v2 under full_v4 = 80.9 = full_v3; hard_v3 under full_v5 = 87.3 = full_v4).  The 0.5 B model generalises well to plain unseen wording (hard_v5: 99 / 100, 99 % exact) and poorly to adversarial wording (terse one-word asks, hedged long asks, idioms, mixed language, all-caps): hard_v4, written as five personas, is 70 %.  Several hard_v3 / hard_v4 gold labels are themselves contestable (`status`, `summary`, `report`, idioms such as "is the battery feeling happy"); they are reported, not retrained.
* **What the probes found that frames could not**: see 6.8 (ignored typed dates giving a made-up 0.0 km/h; KeyErrors on topic concepts; wrong period for "between three and five days ago" and "since monday"; display units ignored; no definition for derived concepts; "which places do I visit most" wrongly labelled unsupported in my own honesty rows (the product can answer it)).
* **Resolver improvements from the loops**: `between` time type (bounded days / hours ago, never absolute timestamps), display units parsed in `literals.display_units`, trips as dated events, ratio quantities asked "at" a time read over the period, binary rain sensor durations, per-day extremes of any 1-minute signal, bare numbers after a comparator word ("when above 50"), mAh -> Wh calculation, glossary definitions for events and derived concepts, a resolver safety net (an executor failure is an honest sentence).
* **Default switched (owner's approval, 2026-10-19)**: the meaning front end is the product default (`make_qwen05_brain`, `app.py`, the shipped UI); `VGPT_MEANING=0` is the way back.
* **Final adapter**: `models/meaning-lora-full_v6.gguf` is the default of `make_meaning_brain`.  Pooled over the three sets no round ever trained on (sealed_v2, sealed_v3, hard_v5; 398 questions) v5 and v6 are equal within noise (376 vs 374 correct), v6 is clearly better on the owner's own phrasing (user_v1 86.6 -> 94.0, a development set) and on sealed_v3 (91.3 -> 93.3); v5 is better on the plain hard_v5 set (99 -> 95).  A reader who prefers the cleaner-set evidence may set `VGPT_MEANING_ADAPTER=models/meaning-lora-full_v5.gguf`.
* **End to end (strict answer equality with the answer the same resolver gives for the gold frame, `e2e_auto.py`; fresh display units per question)**: full_v6 751 / 822 = 91.4 % (novel 90.2, messy 92.9, user_v1 94.0, sealed_v2 93.3, sealed_v3 89.9, hard_v5 91.0); full_v5 751 / 822 = 91.4 %.  The test is strict (whitespace, date style and 600-character truncation inside long answers still differ in a few rows), so it under-counts equivalent answers.
* **Executor coverage with perfect understanding**: 3 of 822 gold frames on the five ordinary sets and 14 of 880 on the four hard sets had no executor before the last fixes (counts of crossings, a peak inside a trip, three definitions, "a day and a half ago"); those are fixed, the remaining ones are quarters, weekday-versus-weekend and context on non-event signals.
* **Latency / memory** (pinned to 2 x86 cores, `tools_pin.py 8,9`, full_v6): load 1.1 s, RSS 625 MB after load, 679 MB peak; answer p50 1.11 s, p95 1.73 s, max 2.13 s.  x86, not the ARM target.
* **Training cost**: one full run alone takes about 60 minutes on the RTX 3050 (11 k rows, 3 epochs, gradient checkpointing); two runs sharing the card are ~4x slower each, so they must run one at a time.

## 7. How a failure is fixed now (no code)
```
python tools_triage.py "<failing question>" --expect "act concept [op=.. cond=.. vs=.. ent=..]"
```
The tool runs the model with and without hints and prints exactly one data change: **word** (the question contains words nobody has defined -> a `glossary.jsonl` line; effective immediately), **ontology** (the concept does not exist -> an `ontology_data.json` skeleton, retrain), **model** (words known, concept exists, frame wrong -> a training row appended to `training/meaning/natural_feedback.txt`, picked up by the next training run), or **executor** (frame right, no executor -> a coverage-gap record). It refuses a question that is in a frozen evaluation set (novel, messy, user_v1 and sealed_v2). Every call is logged to `training/meaning/triage_log.jsonl`. Training = `python training/meaning/run_arms.py full_v2` (about an hour on the laptop GPU when it has the GPU to itself); evaluation = `bench/meaning_eval/eval_frames.py` and `run_pipeline.py`; `bench/meaning_eval/compare_runs.py A B <sets>` lists the questions two runs disagree on with a sign test, and `summarize_arms.py` prints the arm table.

**Learning from a failing frozen question.** The tool refuses the question itself, so write a few new questions with the same meaning in different words and run them in one pass: `python tools_triage.py --batch rows.txt --apply` (each line `question | act concept [op=..] ...`; the model is loaded once). Rows the model gets wrong are appended to `natural_feedback.txt`; rows it already gets right are the other side of the confusable pair and are worth keeping by hand. Three details learned the first time (6.10): the `word` verdict fires for any rare English word and is mostly a false alarm; rows for a combination that was held out of training (`HELD_COMBOS` in `build_meaning.py`) must be feedback rows, which are never held out; and a row that contradicts generated data (a template that labels the same wording differently) will not stick until the template is fixed. Never add a regex to `vgpt/meaning/` for this (`tests/test_meaning.py::RuleBudget`).

## 8. Not done / open (read this)
* **2026-10-20 accounting of every rule that still touches meaning, measured against the model alone: `docs/MODEL_ONLY.md`** (two layers that hurt were narrowed; the frame's six slots and what they cannot hold: `docs/FRAME_SLOTS.md`).  `RuleBudget` below counts only `re.` calls in `vgpt/meaning/`; it does not see word lists, JSON cue files or rules outside that folder.
* **Ablations: all five measured** (6.9: words alone 0 %, hand-written sentences carry most of the result, generated sentences alone are far behind, word tasks add a little, glossary hints are not needed for the average score). Every arm is one training run; no seed variance has been measured, and a retrain alone moves 5-8 questions per set (6.10), so none of the small differences should be quoted as an effect.
* **Failure-to-data loop: no demonstrated gain on unseen wording yet.** v2 and v3 fix the families they were written for on the sets those failures came from, but the clean sets (`sealed_v2`, `sealed_v3`) move by 1-4 questions with inconsistent sign (6.10). Open data conflicts that cap it: generated `now range_estimate` rows against the `estimate range_estimate` convention; "longest trip" labelled `distance_driven` in the natural bank and `trip_record` in the evaluation; the odometer and "motor power now" questions remain wrong after rows for them.
* **Operational note.** Two trainings on the 6 GB card at once ran at roughly a quarter of the speed of one at a time (an ablation arm that takes 45 minutes alone ran for more than six hours and was lost when the jobs were stopped). Run GPU jobs one at a time.
* The frame model is stateless. Follow-ups ("and yesterday?", "why" after "how is the battery") need `Previous:` context rows in training; until then they are read as `ambiguous`.
* Executors for topic meanings still call the older keyword route handlers with a machine-written request (`bindings.json`, 32 entries); typed-argument versions are separate work.
* Evidence caveats: one author wrote training banks and three of the four evaluation sets; answer judgments are one reader's; `sealed_v2` has 8 overlapping rows; ARM was not measured; the Case Study 5 bundle is scored on a coarse act-set + concept rule.
* The `unsafe`/`control` distinction is imperfect (1 of 6 "disable the airbag warning light" read as `control`; both refuse). Hazard-decision questions are intercepted by the deterministic safety guard before the model, by design.
