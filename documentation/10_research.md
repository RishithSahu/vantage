# 10. Research: what we read, what we tested, what we learned, and what is still open

This page collects the research behind the decisions: literature that shaped the design, the project's own experiments (including the negative results), and the open methods proposed now that ordinary training has stalled. References marked "recorded" come from the project's earlier documents and were not re-checked on 2026-10-26.

## 1. The research questions the project asked

1. **Where should a small model sit in the system?** (planner, tool caller, query compiler, selector, reader of meaning)
2. **How can a 0.5 B model understand free wording without a rule per question?**
3. **How do we make it fast on a CPU with no dot-product instructions?**
4. **How do we show cause, not just correlation?**
5. **How do we prove the AI cannot interfere with the car?**
6. **How should history be stored so that weeks are cheap and exact questions are still possible?**

## 2. Experiments and their verdicts

### 2.1 Where the model sits (Oct 5 to 8)

| Design | Verdict |
|---|---|
| Rules first, model as a fallback slot filler (prototype) | worked, but every new failure needed a new rule: rejected as the long-term design |
| Model as a Query IR compiler (v1.1) | good typed back end; the compiler step itself needed too much training and a long grammar |
| Model as a planner over a catalog (VIR/VehicleFM) | rules baseline (88 % test) beat small planners; 1.5 B planners over the memory limit |
| One frozen model plus retrieval plus tool call (VANTAGE model route) | frozen models 12 to 67 %, function selection the bottleneck; prompts of hundreds of tokens incompatible with 2 s |
| Model writes a whole typed plan (qp-lora) | held-out template 0.88 but natural wording 39 to 83 of 104; not ready |
| **Model writes a short closed-vocabulary frame (meaning frame)** | **adopted**: small output, high accuracy on trained distribution, safe by construction |

### 2.2 The research basis of the meaning-frame training (`docs/semantic_method.md` section 4)

* **Supply word meanings; do not make the model memorise them.** Retrieving WordNet glosses into the prompt nearly doubled out-of-distribution concept prediction ([RASP 2025](https://arxiv.org/abs/2412.10207)); computing a rare word's representation from its definition on the fly beat per-word embeddings ([Bahdanau et al.](https://arxiv.org/pdf/1706.00286)); [Dict-BERT](https://arxiv.org/pdf/2110.06490) adds rare-word definitions. This became the glossary.
* **LoRA is the wrong tool to memorise correlations.** [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2): the gap to full fine-tuning is largest for continued pre-training on unlabeled text and closes for instruction-style fine-tuning. Domain-adaptive pre-training needs more text and compute than a 6 GB laptop GPU offers ([Gururangan et al.](https://aclanthology.org/2020.acl-main.740)). The project's own "question-free" experiment (ALMA) raised task recognition 27.8 to 59.8 % but damaged signal linking (66.7 to 39.1 %): words alone do not teach composition.
* **Ontology-generated synthetic data works and has a known ceiling.** Grammar and paraphrase generation trains usable parsers (Overnight, [AutoQA](https://aclanthology.org/2020.emnlp-main.31/)), but real language differs from canonical paraphrase: the language gap ([Yin et al. 2022](https://aclanthology.org/2022.acl-long.103/)); closing it needs structurally diverse natural text ([Niu et al. 2023](https://aclanthology.org/2023.tacl-1.22)).
* **Contrastive hard negatives and abstention make selection robust.** [Octopus](https://arxiv.org/abs/2404.01549) trains with three semantically similar functions per example; [Hammer](https://arxiv.org/html/2410.04587v2) adds about 10 % "irrelevance" examples and masks identifiers; [few-shot intent detection](https://aclanthology.org/2021.emnlp-main.144) uses supervised contrastive learning for confusable intents.
* **Template diversity is not natural diversity**: LLM-written data collapses to typical phrasing unless diversity is engineered ([SynthesizRR](https://arxiv.org/html/2405.10040v1)).
* **Confidence should drive clarification, not answers** ([Calibrated Interpretation](https://ar5iv.labs.arxiv.org/html/2211.07443)).

The project's ablation (2026-10-19) tested this: words-only training 0 %, template-only 78 %, natural-only 90.6 %, everything 93.0 %: natural diversity does the work.

### 2.3 Speed on a CPU without dot-product instructions

Root cause found 2026-10-22: the Cortex-A72 has no dot-product instruction, so llama.cpp tops out near 11 prompt tokens/s; an int8 ONNX Runtime kernel reaches about 30 GOPS on the same two cores. Everything else tried is in [08](08_performance_and_scaling.md): 4-bit weights are 2.4 times slower on this CPU; fusion, threads and S8S8 do not help; a row-layout head helps by about 30 ms per pass. The floor is memory bandwidth.

### 2.4 Explaining causes

Based on the case study's explicit evaluation line ("explain causes behind observed vehicle behavior") and the 6 Oct research note that CS5 is judged as an architecture submission. The cause engine (`vgpt/cause/`) combines: change-point detection with a permutation test against the series' own noise (and de-trending against recorded conditions only if they explain at least a quarter of the variance); a physical trip model (energy per km as a function of speed squared, climate power, cold, highway share); competing hypotheses from a library, one of which is always "no known cause"; calibrated confidence on a sealed benchmark. Single events use a small causal graph (a supply dip can cause a shutdown within N seconds; a controller reset can cause an ADAS drop) over events, logs and signals. Evaluated on a sealed set (87.8 % first answer; 0 % confident false causes); see [09](09_results_and_evaluation.md).

### 2.5 Protocol verification

Every decoder was compared with an independent implementation written by somebody else (cantools, can-j1939, python-can-isotp, udsoncan, scapy, pynmea2, ldfparser, python-doipclient, python-canopen) and, for ISO-TP and J1939, with the Linux kernel's own stacks on a Raspberry Pi (53 of 53). FlexRay has no independent oracle available offline: spec-only evidence.

### 2.6 What is searched and why not a vector database

Structured vehicle data is queried by typed, exact queries (SQL on whitelisted names) because numbers must be exact and cheap; retrieval of free text is used only for log lines (FTS5 when available, a substring scan otherwise). Retrieval-augmented prompts were measured on small models in October and rejected: too many prompt tokens for 2 s.

### 2.7 Things considered and not done

* **Spider 1.0 as a gold standard**: no. It is text-to-SQL over generic databases; our model never writes SQL. Its method (execution-based truth, difficulty tiers) could be borrowed; nothing downloaded.
* **CARLA**: not used; simulation by our own virtual car and world builder, with recorded captures. The owner is to be asked before it is added.
* **One new token per concept** (to cut decode steps from about 2.5 to 1): measured as a latency matter only, and needs training; not done. Recorded references: Token Distillation (arXiv 2505.20133), KL self-distillation (arXiv 2508.15807), Grounded Token Initialization (arXiv 2604.02324).

## 3. Why ordinary training has stalled, and what is open

Evidence (2026-10-24 and 2026-10-25):

* The patch r13 on r12 gained nothing (older 94.83 reviewed against 95.75; fresh 85.08 against 85.38).
* On unseen wording the model is at about 84 %, against 91.5 % on older sets.
* 58 % of errors are made with confidence of at least 0.8, so more of the same loss does not push on them: the model is already confident about the (wrong) answer, so ordinary cross-entropy gives these rows almost no gradient.
* About 40 % of the failures were overlapping labels, and 134 of 283 "errors" printed an equivalent answer or had a wrong gold label: the ceiling is partly in the labels.

### 3.1 The methods proposed (no change to any code; a training round needs the owner's approval)

| # | Method | What it targets | Evidence and cautions | Cost |
|---|---|---|---|---|
| 1 | **Repair the labels first**: a label-noise check ([cleanlab / confident learning](https://web3.arxiv.org/abs/1911.00068v4), [docs](https://docs.cleanlab.ai/v2.6.0/cleanlab/classification.html)) on training and test rows; a second annotator on the failures; merge or define the near-synonym pairs with a decision rule; score near-synonyms as equivalent (accept a *set* of right labels) | label coin-flips; the largest fixable share | test sets average about 3 % label errors (Northcutt et al., NeurIPS 2021, arXiv 2103.14749, recorded); ours had no second annotator | no training |
| 2 | **Preference optimisation on the model's own confident mistakes** (DPO-style pairs of right frame against the model's wrong frame, mined from the audit's 385 errors and from generated confusable pairs) | confident errors that ordinary loss cannot reach | a related study ([Marco-o1 v2](https://arxiv.org/pdf/2503.01461)) shows DPO can hurt other tasks, so it needs the old-set gate; no study found on closed-vocabulary frame writing | a small training run |
| 3 | **Nearest labelled examples as a second opinion, only when confidence is low** (about 8 % of questions) | new wording; tie-breaks in twin pairs | works in task-oriented parsing ([kNN-ICL](https://arxiv.org/pdf/2312.10771), [Generate-and-Retrieve](https://arxiv.org/pdf/2209.14899), [Few-shot intent classification and slot filling with retrieved examples](https://ar5iv.labs.arxiv.org/html/2104.05763), [Dr.ICL](https://arxiv.org/pdf/2305.14128)); no study found for small models on unseen paraphrases. In the prompt it would add about 36 ms per token on the Pi (breaks the latency rule); placed *after* the model it adds time only to those few questions | no training |
| 4 | **A stronger teacher writes diverse wordings, filtered by agreement** (slang, Hinglish, figures of speech, long preambles) | the "unseen wording" gap | written data collapses to typical phrasing unless diversity is engineered (SynthesizRR); needs a teacher available offline at build time (needs approval to download) | training plus a teacher |
| 5 | **Classification heads instead of word-by-word writing**: one read of the question, then a separate choice for act, concept and operator | per-slot calibrated confidence; saves 3 to 5 decode passes (about 0.5 s on the Pi) | may not fix accuracy by itself; a real change to the decoder | training of a small head |
| 6 | **Hard-pair contrastive training and out-of-scope ("unlisted") examples** | twin pairs; the "unlisted" class (69 %) | supervised contrastive training on confusable intents (NAACL 2022, recorded); generated negatives can be false negatives, so each row needs a check | training |
| 7 | **Merging or ensembling adapters** (r10, r11, r12) | little here | averaging two was already tried; ensembles help when members disagree | cheap, probably not worth it |
| 8 | **Equivalence classes in the gold labels and in the loss** (several acceptable labels per question) | removes label noise from both the score and the training | design change in the test sets and the trainer | no training for scoring |

**Recommended order:** 1 first (no training), then 2 together with 6 on the cleaned labels, then 3 as a low-confidence net; 5 is the code-level alternative.

### 3.2 What a trustworthy next measurement looks like

The failure analysis (section 6 of `docs/FAILURE_ANALYSIS_2026-10-24.md`) and the audit agree: a **new set written by someone outside the team** is the honest test. Gold on it should be double-annotated, and the score should be reported both as exact frame and as "what the person is shown" (frame to final answer, sampled and judged by a person).

## 4. Datasets and external sources used

| Source | Use |
|---|---|
| opendbc (commaai, MIT) | 25 real vehicle CAN databases (77 real DBC files in protocol tests) |
| AOSP `VehicleProperty.aidl` and friends | the Android Vehicle HAL property table, generated |
| VIF-AEGIS (35 real OBD-II trips with accelerometer) | driving traces and hard events for the life generator |
| EPA UDDS, HWFET, US06, SC03, FTP cycles | speed traces |
| Open-Meteo hourly weather | real weather in the simulated life |
| HCRL Car-Hacking dataset | real CAN attack recordings for the CAN health monitor |
| comma2k19 | one real camera clip for the camera monitor |
| Loghub (Android and Linux logs) | log parser validation |
| Vosk small English model (`vosk-model-small-en-us-0.15`) | offline speech recognition (installed 2026-10-20 with approval) |
| Qwen2.5-0.5B-Instruct (Q4_K_M GGUF) | the base model |
| bge-small-en-v1.5 (legacy retrieval only) | embeddings for the retired planner route |
| Unused datasets noted: SynCAN, CAN-Intrusion (OTIDS), survival-IDS | not used |

Licences of some datasets (VIF-AEGIS, HCRL, comma2k19, Open-Meteo) were marked "to verify" in the release checklist and need confirming before redistribution.
