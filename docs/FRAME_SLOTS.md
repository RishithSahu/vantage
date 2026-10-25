# Is a six-slot frame enough?  Evidence and a proposal (2026-10-20)

> The proposal became a built design: **see `docs/FRAME_V2.md`** (eight words, data, evaluation sets and gate ready, training awaiting approval).

Written after the owner asked whether real questions might need more fields than `act concept op cond vs ent`.  **No training was done and none is proposed without the owner's approval** (the order of 2026-10-20 stands: no training after `full_v11`).  Everything below is a count over existing files; the patterns used for counting are analysis tools in the scratchpad, not product code.

## 1. What the six slots hold, and what already travels outside them

| slot | holds | example |
|---|---|---|
| `act` | the kind of question | `extreme`, `compare`, `why`, `events` |
| `concept` | the thing | `battery_temperature`, `hard_braking` |
| `op` | what is done to it | `max`, `count`, `mean`, `change` |
| `cond` | one named driving condition, or a threshold | `highway`, `not_highway`, `threshold` |
| `vs` | a second period or a second condition | `period`, `city` |
| `ent` | the unit a question is about | `trip`, `charge`, `day` |

**Values never need a slot**: numbers, units, dates, durations, thresholds ("above 50 km/h"), "last 3 days", "between three and five days ago", codes ("P0A80"), the top-n number ("top 3") are parsed from the text by `vgpt/meaning/literals.py` (typed values, allowed by the method) and handed to the executor next to the frame.  So "six slots" is the *meaning*; the *values* are a separate, deterministic channel.

## 2. How much of what people ask fits (counted, not argued)

* **The 2,176 frozen questions** (all sets): 96.6 % have a gold frame that is an answerable question or a defined non-answer (`social`, `offtopic`, `control`, `unsafe`, `ambiguous`).  **75 (3.4 %) are gold-labelled `unsupported`**: the frame has no way to say them, so the sets expect an honest refusal.  Of those 75: 25 are forecasts ("how long until the next breakdown"), 13 are hypotheticals ("what would my range be at 130 km/h", "what if the coolant pump fails"), 5 ask for something a car never records (weather, tolls, prices), 32 are requests to act or about things outside the data ("book a service appointment", "who else has driven this car").  The model gets these right (it refuses), which is correct behaviour - but it is also the evaluation hiding the gap: **a question the frame cannot express is scored right when it is refused.**
* **A different author's free-wording bank** (counts are upper bounds: they are pattern hits, and many hits are general-knowledge or advice questions the frame already routes to `define` / `advice`) (`VehicleGPT_500_QA.csv`, 500 questions, never written for the frame): questions that compare two named things 5.2 % (many are definitions, "difference between GPS speed and wheel speed"), ranking 0.8 %, hypotheticals / what-ifs 1.8 %, "before / after an event" 3.0 %, a negation or exclusion 1.2 %, "than usual" 1.2 %, several questions in one 0.4 %, a named place 0.8 %.  None groups by a unit; none has two driving conditions at once.
* **What the frame already does for these**: ranking ("which day / trip had the most ...") is `extreme` with `ent` = day / trip (16 of 16 right); "before the warning" is `context`; "consumption excluding the motorway" is `cond = not_highway`; "per trip / per day" is `ent`.  Counts of questions the frame **cannot** hold: two signals compared or correlated ("correlation of speed and consumption": the model answers `stat speed mean`, wrong), "A or B" over two events ("did the battery overheat or did the motor": one concept only), a hypothetical with a parameter, a multi-part request (handled by the investigator, outside the frame, by a length-and-cue rule: see `docs/MODEL_ONLY.md`).

## 3. Verdict

The six slots cover roughly **95 %** of what the free-wording bank and the owner's own phrasing ask, **by construction of the frozen sets** (written for the frame) and by direct count on the bank written by someone else.  The remaining ~5 % is not noise: it is a small number of recurring *kinds* of question, each missing one specific field.  The sets under-measure it, because refusing is scored as correct.

## 4. Proposal: grow the frame only where there is evidence, one optional field at a time

Backward compatible (a question that needs none of them writes `none`, exactly as today):

| change | what it makes expressible | how many questions would use it (bank / frozen) | executor |
|---|---|---|---|
| `c2` (a second concept) | "correlation of speed and consumption", "did the battery or the motor overheat", "range versus efficiency" | <= 5 % (mostly definitions) / 0.2 % | `compare` / `stat` over two series; `op = corr` |
| `vs` gains `typical` | "than usual", "unusually high", "abnormal" - compared with the vehicle's own history | 1.2 % / 0 % | exists as the investigator's baseline analysis; needs a typed argument |
| `ent` gains `week`, `hour`, `unit`, `reason`, `source`, `code` | "by reason", "per ECU", "each week" | 0 % / < 0.4 % | `eventx` group-by exists |
| `act = whatif` with its values as literals | "what would my range be at 130 km/h", "if I drive 120 km tomorrow" | <= 1.8 % (about half are advice, "what should I do if I smell burning") / 0.4 % | needs a physical estimate (road load, consumption model) - a new executor, and an honest "estimate" label |
| `act = forecast` | "when will the battery reach 20 %" | <= 1.6 % (most are general knowledge) / 1.1 % | extrapolation with an uncertainty statement; refuse when the data cannot support it |

What would **not** become a slot: any number, unit, date, name or place (literals); "and then" chains of independent questions (the dialogue splits them); anything that is a request to act on the vehicle (refused by design).

## 5. How to find out before spending a training run

1. Write a **beyond-frame evaluation set** from the free-wording bank and the owner's transcripts (about 150 questions), labelled in the extended frame by hand and frozen with a hash before any model sees it.  It measures the demand honestly: how often a real question needs the new fields, and how often today's model fakes an answer instead of refusing.
2. Score today's `full_v11` on it with no layer (`VGPT_LAYERS_OFF=all`): the **false-answer rate** (a wrong frame executed) matters more than accuracy, because a question the frame cannot hold should be refused, not squeezed into the nearest slot.
3. Only if the demand is real and the false-answer rate is not negligible: add the fields to the ontology and grammar, generate training rows for them, and train **one** arm, judged on the frozen gate already in `docs/TRAINING_QUEUE.md` plus the new set.

Training is not started by this document.
