# What is learned and what is code (the boundary, as built - 2026-10-25)

Vantage is a **hybrid**.  This page says honestly which stage is the model and which is code, in the order a question meets them.  It replaces every older sentence that called the product "model-only"
(`docs/MODEL_ONLY.md` of 2026-10-20 is the measurement behind the first half of the table; `DESIGN.md` section 4d describes the retired two-stage design).

The rule the project holds itself to (`docs/semantic_method.md` 2.1): **code may parse typed values, enforce types, permissions and safety, and calculate; it may not decide what a question means.**
Where the build still breaks that rule, the stage is listed below as *code that influences meaning* - it is debt, it can be switched off, and what it changes can be measured.

## The stages, in order

| # | stage | module | kind | what it does | switch |
|---|---|---|---|---|---|
| 1 | safety guard | `vgpt/safety_policy.py` | **safety** (allowed) | "can I keep driving with this hazard": decided before and without the model | none, on purpose (the case study: safety does not depend on a model's confidence) |
| 2 | pre-parser: spelling and shorthand | `vgpt/meaning/canon.py` (8 patterns) | **influences meaning** (asked for 2026-10-24) | repairs typos and shorthand in place ("batety" -> "battery", "avg" -> "average") and shows the repair; the one ask-back: a shorthand with exactly two meanings | `VGPT_PREPARSER=0` |
| 3 | pre-parser: questions that need no model | `vgpt/meaning/askme.py` (24 patterns) | **influences meaning** (asked for 2026-10-24) | who the assistant is, the time, the service record, a trip estimate, what is outside the car's data (law, prices, weather) | `VGPT_PREPARSER=0` |
| 4 | analysis routing | `vgpt/investigate/`, layer `investigate` | **influences meaning** | a long multi-part request with two or more analysis cues goes to one of 32 analysis templates | `VGPT_LAYERS_OFF=investigate` |
| 5 | typed values | `vgpt/meaning/literals.py`, `vgpt/nl2ir.py`, `vgpt/meaning/anchor.py` | **typed values** (allowed) | dates, times, numbers, units, "before and after my last service" read as periods the vehicle's record names | none (values, not meaning) |
| 6 | **the frame model** | `vgpt/meaning/interface.py`, `ort_backend.py` | **learned** | Qwen2.5-0.5B + adapter writes the frame (act, concept, op, condition, comparison, entity, second concept, reference), decoded under a token mask that can only emit the ontology's words, bounded to the longest valid frame, failing closed | - |
| 7 | confidence | `vgpt/meaning/calibration.py` | **permission / safety** (allowed) | the model's score is mapped to the share of development questions read right at that score; below 0.50 the answer is withheld ("I will not guess"), above it the calibrated number is reported | `VGPT_MEANING_REFUSE_BELOW=0` |
| 8 | corrections of the frame | layers `hint_terms`, `own_vocab`, `unsafe_floor`, `grounding_floor`; `evmap.guard`; `arbitrate` (retired, measured harmful: off unless VGPT_ARBITRATE=1) | **influences meaning** | a word the model is known to misread, a signal only this vehicle declares, an `unsafe` reading with p < 0.70, a concept no word of the question names | `VGPT_LAYERS_OFF=...` (`evmap.guard` is part of the executors' powertrain mapping) |
| 9 | follow-ups | `vgpt/meaning/discourse.py`, layer `discourse` | **influences meaning** | "and yesterday?", "which one was the longest" completed from closed-class cue words | `VGPT_LAYERS_OFF=discourse` |
| 10 | typed execution | `vgpt/meaning/resolve.py` and the executors (`calx`, `eventx`, `tripx`, `thresholdx`, `trendv`, `digest`, `binx`, `records`) | **code that calculates** (allowed) | type-checks the frame, runs plain SQL on the vehicle's own tables, does the arithmetic | - |
| 11 | powertrain mapping | `evmap.py`, `fuelx.py`, layer `fuel_remap` | **influences meaning** | "fuel" asked of an electric car is read as the state of charge (and the reverse for a petrol / diesel car, battery health = the 12 V battery); the first line of the answer says so | `VGPT_LAYERS_OFF=fuel_remap` |
| 12 | wording | `vgpt/answer_composer.py`, `plain.py`, `vgpt/answer_verifier.py` | **code that calculates** (allowed) | composes the answer from the numbers and rejects any number the evidence does not hold | - |
| 13 | permissions | `vgpt/routepolicy.py`, `vgpt/access.py`, `vgpt/privacy.py`, `vgpt/rolelens.py` | **permissions** (allowed) | which role may call which route, what is removed for which role, what each role is shown under an answer | - |

`tests/test_meaning.py::RuleBudget` counts the text patterns of each module of `vgpt/meaning/` and fails when one grows (`literals.py`, `canon.py`, `askme.py` and the small helper modules have a budget; every other module has zero).  A new
stage that inspects the user's words must be listed here and in that budget.

## How to measure what the model does alone

```
VGPT_MODEL_ONLY=1   (or VGPT_LAYERS_OFF=all)       every layer of stage 4, 8, 9 and 11 and the pre-parser (stages 2 and 3) switched off
VGPT_PREPARSER=0                                   only stages 2 and 3
python bench/meaning_eval/layer_ablation.py        each layer against the model alone, on the frozen sets
```

The safety guard, the typed values, the executors and the permissions stay on: they are not about what a question means.

## What this means for a claim

* "The model reads what the question means" is true of stage 6.  It is **not** the whole story of what an answer says: stages 2, 3, 4, 8, 9 and 11 can change which question is answered.
* Gains of an adapter are measured with the layers off (`docs/MODEL_ONLY.md`, `docs/V11_RESULTS.md`).
* The pre-parser exists because the owner asked for it; its rules are listed one by one in `docs/CANON.md`.
