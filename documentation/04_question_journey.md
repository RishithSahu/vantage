# 04. The journey of a question

This document follows one question through the system in the order the code runs it (`ui/server.py` then `vgpt/meaning/pipeline.py` `MeaningBrain.ask`, then `vgpt/meaning/resolve.py`). For each step: **what happens, when, why, where, and what comes out.** The same story is drawn in `docs/VANTAGE_PIPELINE.drawio`.

Running example: *"Why has my range dropped over the last two weeks?"*

## Stage A: arriving

| Step | What happens | Why | Where | Output |
|---|---|---|---|---|
| 1. The question arrives | typed in the Ask page, or spoken ("Vantage, ..."), recognised offline | hands-free while driving | page or `vgpt/voice.py`; reaches the server through an in-process bridge, no socket | question text |
| 2. Route policy | is this role allowed this call? deny by default | roles are access control, not a view | `vgpt/routepolicy.py` | allowed, or a refusal logged without content |
| 3. Queue | one worker, a bounded queue; a full queue answers "busy" at once; a job whose deadline passed is never run | no growing memory, no stale work | `BrainWorker` in `ui/server.py` | a job with a deadline and a session |
| 4. No cache | the same words asked twice are two runs; only the *same request id* in flight is joined | owner's rule: every latency number must be a real run; answers must reflect data as it is now | `tests/test_no_cache.py` | - |

## Stage B: guards and the pre-parser (no model yet)

| Step | What happens | Why | Output |
|---|---|---|---|
| 5. Clean | markup and control characters removed, text cut to 600 characters | hostile or garbled paste must not crash anything | clean text |
| 6. Safety guard | "should I keep driving with this red brake warning?" is classified and answered by fixed policy | safety must not depend on a model's confidence; a warning *count* never answers a safety question | a safety answer, or continue |
| 7. Repair | `canon.py` (8 patterns) repairs typos and shorthand (`avg acc` becomes `average acceleration`) and shows the change; a shorthand with exactly two meanings (`acc` alone) asks which | a 0.5 B model should be helped, not trusted with everything; every repair is visible: "you wrote ..., I read it as ..." | repaired text plus a note |
| 8. No-model questions | `askme.py` (24 patterns): who are you, what do you run on, what can you do, the time, what is outside the car's data (law, prices, weather), the service record, a trip energy estimate, "is the car charging" | no reason to spend 1.4 s of model on these; every number comes from the car's own tables | an answer, or continue |
| 9. Pending replies | if the last turn asked "did you mean 1 or 2" or "which time", the short reply completes it | follow-ups in fragments | resolved question |
| 10. Long analysis request? | two or more analysis cues (long, multi-part) route to the **investigation engine**; one cue is left to the model (measured: 5 of 70 ordinary questions in a long wrapper had exactly one) | long requests are not one 8-word frame | a report, or continue |
| 11. "What data do you hold?" | counted from the store's tables | a question about the store itself is not a signal | an inventory answer, or continue |

## Stage C: the model reads the question

| Step | What happens | Why | Where | Output |
|---|---|---|---|---|
| 12. Typed values | dates, times, numbers, units, wheel position, "before my last service" read by `literals.py` / `anchor.py` | a number is parsed, never guessed | code (the only text-pattern module in the meaning path) | typed values |
| 13. Previous-turn gate | the last answered frame is added to the prompt only when the wording points back ("and yesterday?") or the bare reading is incomplete | an unrelated previous line cost tokens (about 11 of 25 prompt tokens) and pulled the reading towards it | `meaning/prevgate.py` | prompt with or without a "Previous:" line |
| 14. The model | Qwen2.5-0.5B + LoRA r12 (int8, ONNX Runtime) reads the question, glossary hint lines, the car type (EV, petrol, diesel) and optionally the previous frame; one **bounded** decode under a **token mask** writes the frame | free wording cannot be a rule list; the mask makes hostile text harmless and bounds the decode | `meaning/interface.py`, `ort_backend.py` | an 8-word frame (act, concept, operator, condition, comparison, entity, second concept, reference) and a confidence score |
| 15. Speculative drafts | likely next tokens (from concept words found in the question and word-order tables learned from training frames) are verified by the model in the same pass | each pass costs about 150 ms on the Pi; an extra drafted row costs only about 25 ms; lossless because only tokens the model would choose are accepted | `meaning/cues.py` | fewer passes (median 2 to 3) |

For the example: `why range_estimate drop none none none none none none`.

## Stage D: can we trust the reading?

| Step | What happens | Why |
|---|---|---|
| 16. Known misreads | glossary words the model gets wrong, signals only this vehicle declares ("AdBlue"), "fuel" asked of an EV mapped to charge | small data fixes beat retraining; each correction is recorded in the answer's facts |
| 17. Unsafe floor | an `unsafe` reading with p < 0.70 is asked again, not refused | keyboard noise read as an attack request must not be refused |
| 18. Grounding floor | a why/define/how-to reading with p < 0.50 whose concept no word of the question names is treated as not about this vehicle | off-topic questions were being answered as vehicle health |
| 19. Calibrated confidence | the raw score is mapped to the share of development questions read right at that score; **below 0.50 the answer is withheld**: "I am not sure I understood that, so I will not guess. I read it as ..." | an honest "not sure" beats a confident wrong answer; this withholds about 2.7 % and catches about 17 % of wrong readings |
| 20. Ask-back (off by default) | below 0.80 the assistant could offer "did you mean 1 or 2" with the model's own two readings | the owner turned it off on 2026-10-24; `VGPT_MEANING_TAU=0.8` restores it |
| 21. Follow-up completion | "and yesterday?", "which one was the longest?", "what about the rear left?", "how did you get that?" are completed from the last turn (period shift, concept swap, ordinal, referent, provenance) | people speak in fragments |
| 22. Type-check | act, concept and operator must fit together under the ontology; otherwise a specific clarification is produced | an ill-typed frame must never execute |

## Stage E: code calculates the answer

| Step | What happens | Why | Output |
|---|---|---|---|
| 23. Resolver order | (1) non-executing acts (social, off-topic, unsupported, control, unsafe, ambiguous) and unmodelled concepts return fixed outcome texts and never touch data; (2) typing; (3) a frame the typed engine can express goes to compile and the query engine; (4) instant lookups, specs, definitions, how-tos; (5) topics (health, advice, estimate) through a small machine-written adapter to the older route handlers, reported as `legacy_adapter`; (6) otherwise say what was understood and that no executor exists | the order depends only on the frame, never on the words | a result or a stated gap |
| 24. Compile and validate | frame plus typed values become a typed `Query`; validated against the registry; **capability check**: does this car have this data? missing data is explained, never turned into zero | no silent substitution of one signal for another | a valid typed query, or "this car does not record that" |
| 25. Plan | the planner builds the **only SQL in the system** from fixed templates, whitelisted table and column names and bound values; picks the cheapest tier that is exact enough (seconds for short recent windows, minutes for history, the daily tier for long windows, the archive for exact old seconds, the trip ledger for whole-trip accounting) | the model can never write SQL; read-only handle | a read-only plan |
| 26. Execute | executors do the arithmetic: statistics, trend verdicts (Kendall, a yes or no first), event counts and rates (exact binomial tests), trips, charging, fuel and economy, correlations, per-day and per-week series, weekend against weekday, patterns in when events happen, summaries, "unusual for this car", derived power | all numbers come from here | numbers with sample counts |
| 27. Why | for the act `why`, the **cause engine** detects when a quantity changed (a permutation test against the series' own noise), attributes the change with a physical trip model (drag grows with speed squared, rolling resistance is flat in speed, cold follows temperature, climate follows HVAC power), judges competing causes from a library, and allows the answer **nothing changed enough to need a cause** | explanations must come with evidence and what was ruled out | a cause, evidence, alternatives, how to confirm, a checked confidence |
| 28. Investigations | long requests use one of 32 analyses and keep recorded facts, associations and hypotheses apart, state the data resolution and what is missing | a long request is not one frame | a structured report |

## Stage F: words, checks and privacy

| Step | What happens | Why |
|---|---|---|
| 29. Evidence | claim, numbers, units, sample counts, tier, quality flags, gaps, evidence class (measured, derived, estimated) | an answer is only as strong as its data |
| 30. Composer | templates write the words from the numbers; verdict first, working under "How I got this" | a language model must not state a result |
| 31. Verifier | every number in the text must be an evidence item or a count, threshold or window from the query; units and claims are checked; unsupported answers are withheld with a safe fallback | the last gate |
| 32. Privacy and role | location, saved places and pictures only for the driver; the engineer also sees the working and open codes; the fleet operator sees fleet numbers; validation sees the evidence pack; an audit entry records what happened, never the content | the same truth for different readers |
| 33. Out | the answer on screen with "I read it as...", confidence and evidence chips; optional local spoken answer; thumbs up or down kept as a small encrypted, expiring record | - |

## What the model decides, and what it does not

| Decided by the model | Decided by code |
|---|---|
| which kind of answer is wanted (act) | safety, permissions, who may see what |
| which quantity or event (concept) | spelling repair, no-model questions (rule lists, visible and counted) |
| the operator, condition, comparison, entity, second quantity, reference | dates, numbers, units, "last service" periods |
| nothing else | the database query, all arithmetic, the cause analysis, the wording, the checks |

`docs/BOUNDARY.md` lists every stage as learned, code that calculates, or "code that influences meaning" (the pre-parser, a few correction layers, follow-up completion, investigation routing, the fuel mapping). Each of those can be switched off to measure what the model does alone: `VGPT_MODEL_ONLY=1`, `VGPT_PREPARSER=0`, `VGPT_LAYERS_OFF=...`. A test (`RuleBudget`) counts the text patterns in each module and fails if one grows.
