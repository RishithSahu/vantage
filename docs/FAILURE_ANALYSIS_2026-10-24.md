# Why the question reader gets 283 of 3,038 frozen questions wrong (2026-10-24)

Source: `results/clarify_eval.json` (the shipped model `frame2w`, int8, all 26 frozen sets, 3,038 rows; 266 distinct wrong questions). Gold = the frame the project's own evaluation sets say is right. Causes were assigned by explicit rules over the wrong words and then read by a person; the full list is the appendix. A question gets exactly one cause (the first rule that fits).

## 1. The count

| Root cause | questions | share | of them confident (p>=0.8) |
|---|---|---|---|
| A. The label set overlaps (two names or two answer-kinds for nearly the same thing) | 113 | 40% | 75 |
| B. Things the car may not have (the ontology edge: listed vs unlisted vs unsupported) | 35 | 12% | 15 |
| C. Wording the model has not learned (slang, figures of speech, Hindi-English, typos, long preambles) | 59 | 21% | 25 |
| D. Operator and optional-slot conventions (mean/sum, usual, rise/drop, driving/trip/day) | 65 | 23% | 39 |
| E. Not a model error (the car in the test lacks the signal; the model read it right) | 11 | 4% | 10 |
| **all** | **283** | | **164** (58%) |

### Finer split

| cause | n | confident |
|---|---|---|
| A: near-synonym concepts | 42 | 31 |
| A: near-synonym concept, with act/operator shifting along | 31 | 16 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | 29 | 19 |
| A: refuse / off-topic / control boundary | 11 | 9 |
| B: listed vs unlisted thing | 35 | 15 |
| C: figure of speech / slang | 30 | 11 |
| C: abbreviation / typo / telegraphic | 12 | 7 |
| C: long polite preamble | 12 | 6 |
| C: Hindi-English | 5 | 1 |
| D: answer-kind and operator shift together | 29 | 18 |
| D: optional slot (condition / unit / compared-with) | 22 | 12 |
| D: operator word | 14 | 9 |
| E: test-label artifact | 11 | 10 |

Reading the table honestly: **A is the largest and it is mostly a label-design problem, not a spelling or vocabulary problem.** In 126 of the 156 wrong concept words the gold and the predicted word already differ at their FIRST token (measured with the model's own tokenizer, section 4), so giving each label its own single token would not by itself separate them: the model confuses what they MEAN.

## 2. Per cause: what exactly fails and why

**A. Overlap in the label set.** The ontology has pairs that people use interchangeably and that the evaluation sets nonetheless score as different: `time_to_full` / `charge_time`; `regen_power` / `regen_energy` ("what does regen mean"); `maintenance` / `service_reminder` / `service_interval`; `battery_soh` / `battery_condition` / `battery_soc`; `overtemp_warning` / `battery_overheat_warning` / `motor_temperature`; `throttle_position` / `accelerator_position`; `aux_battery_voltage` / `aux_battery_condition`; `odometer` / `distance_driven` / `trip_record`. The same happens one level up between answer-kinds: `now` vs `estimate` ("estimated range right now", "when will the battery be full"), `now` vs `health` ("how worn are the brake pads"), `health` vs `summary` vs `anomaly`, `unsupported` vs `offtopic` ("what is the weather tomorrow"), `control` vs `unsafe`. When two labels are near-duplicates, whichever the training set used more wins, and the model is confident because both readings were seen often. This is the cause of most **confident** errors and ask-back cannot catch it.

**B. The edge of what the car has.** Questions about things no signal exists for ("how many cup holders", "drive modes", "boot space", "USB ports", "how loud is the engine") must come out as `unlisted` or `unsupported`; the model instead picks the nearest listed signal (`charge_port_state`, `engine_rpm`, `gear`), because in training almost every question had a real answer. The opposite also occurs (`dpf_soot_load` read as `unlisted`).

**C. Wording the model has not learned.** Figures of speech ("the car's pulse", "heartbeat of the pack", "does the car have a fever", "mood of the coolant", "thirsty", "juice"), Hindi-English ("kitna km chala hoon is hafte", "service kab hai", "motor kitna garam hai"), telegraphic and misspelled text ("soh pls", "gm", "pax last trip", "whperkm this mnth", "y did batry overheat"), and a polite preamble ("Hello! Two things: first ... and second, thank you very much") that the model reads as a comparison. A 0.5-billion-parameter model has seen little of this; this IS partly a vocabulary-and-training-data gap, and it is where more data helps most. Most of these are NOT confident, so this is the cause ask-back used to catch.

**D. Operator and optional-slot conventions.** The model writes `sum` where the gold says `mean` ("fuel consumption this month", "typical regen per week"), `rise` where it says `drop`, `count` for `list`, `exists` for `count`; and it fills or leaves empty the optional words `cond`, `ent`, `vs` differently from the gold ("average speed on trips" gold `trip`; "how many trips per day" gold `day`; "when was I last above 100" gold `speed ... threshold`). Several of these are conventions that no reader could infer from the wording; they need an explicit rule in the training data, not more of the same data.

**E. Not a model error.** Eleven rows say `unlisted` in the gold because the test car has no gear / oil-temperature / AdBlue / turbo signal; the model's `gear`, `engine_oil_temperature` ... is the right reading and the vehicle catalogue already turns it into "this car does not report that". These should leave the error count.

## 3. Are the 'wrong' answers really wrong? (two caveats)

* **The gold labels have not been re-checked.** Published work on test sets finds about 3 % label errors on average (up to 6 % on one image set) (Northcutt et al., NeurIPS 2021, arXiv 2103.14749). Ours are written by the same team that wrote the training data, with no second annotator, so the real figure is unknown. Reading the 283 rows, at least 40 are cases where both readings are defensible ("estimated range right now", "what is the weather tomorrow", "how worn are the brake pads", "soh pls"). Some of the 9.3 % error is therefore the test, not the model.
* **A wrong frame is not always a wrong answer.** `unsupported` vs `offtopic`, or `charge_time` vs `time_to_full`, may print the same reply. Nobody has yet measured how many of the 283 change what the person is shown. That measurement (frame -> final answer, compared by a person on a sample) is step 3 of the plan below.

## 4. Can the model's vocabulary be improved?

Measured with the model's own tokenizer (Qwen2.5): the ontology has 175 concepts; **only 14 are one token**, the average is 2.5 tokens (`battery_soh` -> ` battery`, `_s`, `oh`; `regen_power` -> ` reg`, `en`, `_power`). Acts and operators are already almost always single tokens. Two different things are called 'vocabulary':

1. **The model's token vocabulary** (one new token per concept, its embedding built from the concept's description or from its parts). Benefit: fewer decoding steps (about 1.5 fewer per frame; on the Pi each step is about 140 ms) and a clean unit for the model to be sure or unsure about. Risk: the common default of initialising new tokens to the mean embedding makes them nearly identical and hard to separate later (Grounded Token Initialization, arXiv 2604.02324); better is to initialise from the parts or the description and to distil the old multi-token behaviour into the new token (Token Distillation, arXiv 2505.20133; KL self-distillation, arXiv 2508.15807). It needs a training run, and it will NOT by itself fix cause A, because 81 % of the confused pairs differ already at the first token.
2. **The project's word list (glossary and ontology descriptions).** This is where meaning is defined. Fixing causes A and B means editing THIS vocabulary: merge the near-duplicate labels, or give each a one-line definition and a decision rule, and keep a list of what is NOT in the car. Cheap, no training to change the labels, but the training data must follow the change.

## 5. Methods from the literature, and what each would fix here

| Method | Fixes | Evidence / caution | Needs training |
|---|---|---|---|
| Find and fix label errors with confident learning (cleanlab) on the evaluation AND training sets | E, part of A | NeurIPS 2021: about 3.3 % label errors in 10 benchmark test sets; confident learning cut human review by up to 90 % (arXiv 2103.14749) | no |
| Redefine overlapping labels: a decision rule per borderline pair, merge pairs people cannot tell apart, add explicit unknown / not-applicable classes | A, B | taxonomy design, not labeling, is the usual cause of confusable intents; measure the confusion between neighbouring labels, then merge or redefine (practitioner guidance and a CHIIR 2021 study; weaker evidence than a paper on exactly our case) | no for the labels, yes for the model afterwards |
| Targeted hard-negative training: for every confused pair, corrected failures plus contrasting examples | A, B, D | Closest published case: a small intent classifier on CLINC150 whose errors were overlapping intents, fixed with corrected failures plus hard negatives per confusable pair (Pioneer Agent, arXiv 2604.09791); supervised-contrastive training on in-scope intents the model already confuses (NAACL 2022, 2022.naacl-main.307). LLM-generated negatives can be too close to the positives and become false negatives (arXiv 2504.21015), so each generated row needs a check | yes |
| More varied wording: paraphrases, slang, Hinglish, typos, preambles, written by a larger model and filtered | C | Diverse paraphrase augmentation helped a small classifier on 12 text datasets (DoAug); for Hinglish, small models fine-tuned on good code-mixed data can be competitive and transliteration mismatches hurt (WMT22 MixMT, arXiv 2210.11309). I found no paper that tests slang or figurative language for a classifier like ours, so the size of the gain is unknown | yes |
| Retrieval of the nearest labelled examples at question time (nearest-neighbour intent detection) | A, C, new wording without retraining | 10-shot nearest-neighbour matched 50-shot classifiers (Zhang et al., EMNLP 2020); works by changing the index, no retraining (arXiv 2104.05763). **Costs time at question time**, and you ruled out added latency, so it would have to be measured first | no (needs an index) |
| Out-of-scope training: examples of things the car does not have, labelled `unlisted` | B | Generating outlier examples so a model is less sure on unseen classes (CoNAL, ACL 2023) | yes |
| Ensembles of two or three adapters (we already average two) | confident errors | Ensembling helps when members disagree a lot compared with their error (NeurIPS 2023); a 2026 vision study notes that high-confidence disagreement often means an ambiguous class or a wrong label, not a model mistake. Costs 2-3x the compute on a Pi already near 1.3 s | yes |
| Calibrate or abstain | none of the confident errors | Calibration and selective prediction are different goals; abstaining only helps if wrong answers are less confident than right ones, and 58 % of ours are not (arXiv 2607.03528). It also removes the answer, which you do not want | no |

## 6. What to solve, in order

1. **Make the test trustworthy first (no training, about 1 day).** Run confident learning over all 3,038 rows and the training rows; have a second person re-label the 283 failures and 100 confident-and-right rows as a control; remove the 11 artifact rows. Result: the real error rate and the list of pairs whose gold is a coin-flip.
2. **Fix the label set (no training, about 2 days).** Merge or define each overlapping pair of section 2.A with a one-sentence rule and examples; define `unsupported` vs `offtopic` and `control` vs `unsafe` by what the car does (if the reply is the same, one label); decide whether `now` and `estimate` are one act. Fewer, sharper labels is the only fix that removes the confident errors at the source.
3. **Measure what the person sees (no training).** For the 283, compare the final printed answer under the gold frame and the predicted frame. Anything identical is not an error in the product.
4. **Then one training round (needs your approval: training was stopped on 2026-10-21).** New data in four groups: corrected and re-labelled failures with hard negatives per confusable pair; unlisted / unsupported things; wording variety (figures of speech, Hinglish, typos, preambles); explicit operator and slot conventions. Check on a NEW set written by someone outside the team. Optional in the same round: single-token labels for the concepts, initialised from their descriptions (latency, not accuracy).
5. **Ask-back is off** (done today: `CLARIFY_BELOW = 0` in `vgpt/product.py`; `VGPT_MEANING_TAU=0.8` brings it back). Until step 4 the 7 % of questions that used to be asked back are answered with the first reading, which catches fewer wrong readings: expect the accuracy of what is answered to fall from 94.2 % back to the model's own 90.7 % on our sets.

## 7. What cannot be promised

'So good that it has no confusion' is not reachable for words that mean the same thing to people: if two labels are interchangeable, a perfect reader still has to pick one, and the right fix is to remove the distinction (step 2), not to train harder. A realistic target for a 0.5-billion-parameter reader after steps 1-4 is the high 90s on a fresh set written by someone else; that number cannot be shown until that set exists.

## Appendix: all 283 wrong readings

| cause | set | question | gold -> said (only the words that differ) | p |
|---|---|---|---|---|
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v4 | brake pad life remaining | now -> health | 0.76 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | user_v1 | driver assistant state zero mean | define -> now | 0.75 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | novel_v1 | estimated range right now | estimate -> now | 0.97 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | express the battery temperature in kelvin | now -> define | 0.76 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | adv_v1 | how do i read the state of health number | define -> howto | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | digest_v1 | how is my driving style looking | summary -> health | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | followup_fresh_v4 | how is my driving style looking | summary -> health | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v5 | how is the estimated range obtained | howto -> spec | 0.60 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | how many 5000 mah batteries could my pack charge | howto -> estimate | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | novel_v1 | how much boot space is there | spec -> now | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | novel_v1 | how much does the car weigh | spec -> now | 0.95 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | novel_v1 | how worn are the brake pads | health -> now | 0.68 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | i am planning a long drive this weekend and just want to be sure there are no warnings or problems at the mome | advice -> health | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v3 | i would like a summary of the condition of the traction battery | health -> summary | 0.92 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | digest_v1 | is anything wrong with the way the battery is behaving | anomaly -> health | 0.98 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | followup_fresh_v4 | is anything wrong with the way the battery is behaving | anomaly -> health | 0.98 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v4 | longest idle period this week | stat -> extreme | 0.99 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | range at 50 percent charge | estimate -> at | 0.56 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | the soh did not change did it | trend -> events | 0.58 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v1 | what happened before the last overheating and why did it happen | context -> why | 0.90 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | user_v1 | what is our motor power | now -> spec | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | catalog_v1 | what is the egr for | define -> now | 0.93 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | followup_fresh_v4 | what is the egr for | define -> now | 0.93 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v2 | what is the home charger power | spec -> now | 0.53 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v2 | what is the plan for maintenance | advice -> estimate | 0.91 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | adv_v3 | what is the range at the moment | estimate -> now | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | adv_v3 | whats the point of the voltage dip event | define -> why | 1.00 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | adv_v2 | when did hard braking drop off | trend -> events | 0.78 |
| A: answer-kind boundary (now / estimate / health / summary / stat / events) | hard_v2 | when should i service | estimate -> advice | 0.76 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | at 80 percent how many kwh is that | howto -> now; battery_capacity -> energy_used | 0.66 |
| A: near-synonym concept, with act/operator shifting along | hard_v3 | coolant flow dropping | trend -> events; coolant_flow -> coolant_flow_drop; drop -> list | 0.91 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | correlation of speed and consumption | wh_per_km -> speed; mean -> corr | 1.00 |
| A: near-synonym concept, with act/operator shifting along | slots_v2 | did regeneration give less than it usually does last week | regen_energy -> regen_power; sum -> mean | 0.60 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | did the camera clean itself or is it still blocked | health -> events; camera_sharpness -> camera_degraded; none -> exists | 0.87 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | everyone says electric cars lose range in winter so did mine lose range when it was cold | stat -> events; wh_per_km -> range_estimate; mean -> drop | 0.99 |
| A: near-synonym concept, with act/operator shifting along | platform_v2 | give me the picture of my diagnostic codes | health -> summary; diagnostic_status -> diagnostic_code | 0.64 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | how long have we been driving today | trip_record -> drive_mode; sum -> duration; none -> driving; trip -> none | 0.55 |
| A: near-synonym concept, with act/operator shifting along | hard_v2 | how long have we driven in total | now -> stat; odometer -> drive_mode; none -> duration; none -> driving | 0.54 |
| A: near-synonym concept, with act/operator shifting along | sealed_v3 | how long until the battery is full | estimate -> now; charge_time -> time_to_full | 0.81 |
| A: near-synonym concept, with act/operator shifting along | novel_v1 | how many km before i hit 10 percent | estimate -> stat; range_estimate -> distance_driven; none -> count; none -> threshold | 0.51 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | how much petrol is left | estimate -> now; range_estimate -> fuel_level | 0.53 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | how often should i change the engine oil | spec -> advice; service_interval -> engine_oil_temperature | 0.53 |
| A: near-synonym concept, with act/operator shifting along | adv_v2 | i guess the lane assist stayed on the whole time | events -> stat; adas_disengagement -> adas_state; exists -> duration | 0.99 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | is the 12 v battery holding its voltage | health -> now; aux_battery_condition -> aux_battery_voltage | 0.99 |
| A: near-synonym concept, with act/operator shifting along | catalog_v1 | is the 12 volt battery low | now -> health; aux_battery_voltage -> aux_battery_condition | 0.64 |
| A: near-synonym concept, with act/operator shifting along | followup_fresh_v4 | is the 12 volt battery low | now -> health; aux_battery_voltage -> aux_battery_condition | 0.64 |
| A: near-synonym concept, with act/operator shifting along | catalog_v1 | minutes until fully charged | now -> estimate; time_to_full -> charge_time | 0.80 |
| A: near-synonym concept, with act/operator shifting along | followup_fresh_v4 | minutes until fully charged | now -> estimate; time_to_full -> charge_time | 0.80 |
| A: near-synonym concept, with act/operator shifting along | hard_v3 | service due when | estimate -> events; maintenance -> service_reminder; none -> last | 0.97 |
| A: near-synonym concept, with act/operator shifting along | hard_v2 | total time spent driving this week | trip_record -> drive_mode; sum -> duration; none -> driving; trip -> none | 1.00 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | was every charge a fast charge | compare -> events; charging_power -> charging_session; mean -> exists; charging -> none | 0.77 |
| A: near-synonym concept, with act/operator shifting along | hard_v1 | was the coolant flow never below normal | events -> compare; coolant_flow_drop -> coolant_flow; exists -> mean; none -> typical | 0.94 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | what does that warning mean | why -> health; any_warning -> briefing | 0.59 |
| A: near-synonym concept, with act/operator shifting along | hist_v2 | what is the total distance since the start | stat -> now; distance_driven -> odometer; sum -> none | 1.00 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | what was the pack doing when the flow dropped | coolant_flow_drop -> battery_current; none -> drop | 0.77 |
| A: near-synonym concept, with act/operator shifting along | hard_v4 | when was i last above 100 | events -> stat; hard_acceleration -> speed; none -> threshold | 0.89 |
| A: near-synonym concept, with act/operator shifting along | novel_v1 | when will the battery be full | estimate -> now; charge_time -> time_to_full | 0.56 |
| A: near-synonym concept, with act/operator shifting along | messy_v1 | whens service due | estimate -> events; maintenance -> service_reminder; none -> last | 0.95 |
| A: near-synonym concept, with act/operator shifting along | adv_v2 | why did the motor get so hot on the hill | overtemp_warning -> motor_temperature; none -> rise | 0.54 |
| A: near-synonym concept, with act/operator shifting along | hard_v2 | why did the tyre pressure drop | tyre_pressure_warning -> tyre_pressure; none -> drop | 0.95 |
| A: near-synonym concepts | hard_v1 | any faults before last month | diagnostic_code -> any_warning | 0.92 |
| A: near-synonym concepts | hard_v4 | are the pads worn | brake_condition -> brake_pad_life | 0.98 |
| A: near-synonym concepts | hard_v4 | average cost per trip in energy | energy_used -> trip_record | 0.93 |
| A: near-synonym concepts | slots_v2 | correlation between the accelerator and battery power | battery_power -> accelerator_position; accelerator_position -> battery_power | 1.00 |
| A: near-synonym concepts | adv_v1 | did the battery overheat or did the motor | overtemp_warning -> battery_overheat_warning | 1.00 |
| A: near-synonym concepts | adv_v1 | did the car hit the brakes for me without my foot on the pedal | aeb_activation -> hard_braking | 0.96 |
| A: near-synonym concepts | adv_v3 | has the fault memory ever been reset | diagnostic_code_cleared -> ecu_reset | 0.89 |
| A: near-synonym concepts | hard_v2 | how do i keep the battery healthy | battery_condition -> battery_soh | 0.75 |
| A: near-synonym concepts | sealed_v2 | how do i keep the battery healthy over the years | battery_condition -> battery_soh | 0.71 |
| A: near-synonym concepts | sealed_v3 | how do i turn kwh into km | range_estimate -> energy_used | 0.99 |
| A: near-synonym concepts | catalog_v1 | how do throttle position and speed relate | throttle_position -> accelerator_position | 0.79 |
| A: near-synonym concepts | followup_fresh_v4 | how do throttle position and speed relate | throttle_position -> accelerator_position | 0.79 |
| A: near-synonym concepts | platform_v1 | how hard is the driver turning the wheel | steering_torque -> steering_angle | 0.53 |
| A: near-synonym concepts | hard_v4 | how is the car after the service | vehicle_health -> maintenance | 0.74 |
| A: near-synonym concepts | hist_v2 | how long was the car parked last month | drive_mode -> speed | 0.91 |
| A: near-synonym concepts | platform_v1 | how many network errors have been logged | can_error -> service_log | 0.87 |
| A: near-synonym concepts | hard_v2 | how much energy did the motor use today | energy_used -> motor_power | 0.78 |
| A: near-synonym concepts | digest_v1 | how much fuel does the engine normally burn per hour | fuel_rate -> engine_load | 0.85 |
| A: near-synonym concepts | followup_fresh_v4 | how much fuel does the engine normally burn per hour | fuel_rate -> engine_load | 0.85 |
| A: near-synonym concepts | adv_v2 | how often did the pack go above 45 | battery_temperature -> battery_soc | 0.66 |
| A: near-synonym concepts | novel_v1 | longest trip | trip_record -> distance_driven | 1.00 |
| A: near-synonym concepts | hard_v5 | meaning of regenerative braking | regen_power -> regen_energy | 0.88 |
| A: near-synonym concepts | hard_v4 | number of samples below 20 degrees | battery_temperature -> ambient_temperature | 0.60 |
| A: near-synonym concepts | hard_v2 | peak power on the last trip | motor_power -> battery_power | 1.00 |
| A: near-synonym concepts | hard_v3 | regen meaning | regen_power -> regen_energy | 0.94 |
| A: near-synonym concepts | hard_v1 | tell me it is safe to drive with a brake warning at highway speed | vehicle_health -> brake_condition | 0.97 |
| A: near-synonym concepts | followup_v2 | the most recent of these | adas_disengagement -> any_warning; item -> none | 0.88 |
| A: near-synonym concepts | sealed_v2 | the wheel angle please | steering_angle -> wheel_slip | 0.56 |
| A: near-synonym concepts | hard_v3 | time pack above 40c | battery_temperature -> battery_soc | 0.84 |
| A: near-synonym concepts | novel_v1 | tips to save energy | wh_per_km -> range_estimate | 0.99 |
| A: near-synonym concepts | hard_v2 | what does regen mean | regen_power -> regen_energy | 0.95 |
| A: near-synonym concepts | slots_v2 | what is a regen | regen_power -> regen_energy | 0.82 |
| A: near-synonym concepts | hard_v4 | what is my range in winter compared to summer | wh_per_km -> range_estimate | 1.00 |
| A: near-synonym concepts | followup_v2 | what is regen | regen_power -> regen_energy | 0.85 |
| A: near-synonym concepts | adv_v2 | what power is going into the pack right now | charging_power -> battery_power | 1.00 |
| A: near-synonym concepts | platform_v1 | what was happening before the motor got hot | overtemp_warning -> motor_temperature | 0.94 |
| A: near-synonym concepts | hard_v4 | which day was the quietest this month | distance_driven -> source_dropout | 1.00 |
| A: near-synonym concepts | messy_v1 | why did battry get so hot | battery_overheat_warning -> battery_temperature | 0.97 |
| A: near-synonym concepts | user_v1 | why did the engine overheat | motor_temperature -> overtemp_warning | 0.58 |
| A: near-synonym concepts | hard_v5 | why did the pack get so hot | battery_overheat_warning -> battery_temperature | 0.99 |
| A: near-synonym concepts | adv_v1 | why did the pack get so warm after the charge | battery_overheat_warning -> battery_temperature; none -> charging | 0.91 |
| A: near-synonym concepts | adv_v2 | why does the speed differ between two sources | sensor_conflict -> speed | 0.98 |
| A: refuse / off-topic / control boundary | hard_v3 | by the way what will the weather be like in mumbai this weekend | offtopic -> unsupported | 1.00 |
| A: refuse / off-topic / control boundary | hard_v2 | call the service centre | control -> unsupported | 0.93 |
| A: refuse / off-topic / control boundary | hard_v1 | my friend said the speed limit is 200, so is 180 fine | unsupported -> ambiguous | 0.23 |
| A: refuse / off-topic / control boundary | hard_v2 | send my trips to my email | unsafe -> unsupported | 1.00 |
| A: refuse / off-topic / control boundary | hard_v2 | share my location with my friend | unsafe -> unsupported | 0.77 |
| A: refuse / off-topic / control boundary | adv_v1 | switch off the abs light for me | unsafe -> control | 0.84 |
| A: refuse / off-topic / control boundary | hard_v1 | switch off the brake warning permanently | control -> unsafe | 1.00 |
| A: refuse / off-topic / control boundary | novel_v1 | take me to the nearest charger | unsupported -> control | 0.94 |
| A: refuse / off-topic / control boundary | hard_v5 | what did you mean | ambiguous -> explain | 1.00 |
| A: refuse / off-topic / control boundary | sealed_v3 | what is the traffic like downtown | unsupported -> offtopic | 0.80 |
| A: refuse / off-topic / control boundary | slots_v2 | what is the weather tomorrow | unsupported -> offtopic | 1.00 |
| B: listed vs unlisted thing | user_v1 | Which selectable drive modes are fitted? | spec -> unsupported; drive_modes -> none | 0.87 |
| B: listed vs unlisted thing | sealed_v3 | can the car run a small fridge | export_power -> unlisted | 0.75 |
| B: listed vs unlisted thing | hard_v4 | does it use petrol | now -> spec; unlisted -> fuel_rate | 0.87 |
| B: listed vs unlisted thing | sealed_v3 | does the car record the cabin humidity | now -> spec; unlisted -> humidity | 0.72 |
| B: listed vs unlisted thing | hard_v4 | give me the full database | unsafe -> spec; none -> unlisted | 0.58 |
| B: listed vs unlisted thing | catalog_v1 | how is the dpf doing | now -> health; dpf_soot_load -> unlisted | 0.97 |
| B: listed vs unlisted thing | followup_fresh_v4 | how is the dpf doing | now -> health; dpf_soot_load -> unlisted | 0.97 |
| B: listed vs unlisted thing | icevocab_diesel_v1 | how is the heated steering wheel | health -> now; unlisted -> steering_angle | 0.53 |
| B: listed vs unlisted thing | icevocab_petrol_v1 | how is the heated steering wheel | health -> now; unlisted -> steering_angle | 0.53 |
| B: listed vs unlisted thing | sealed_v3 | how is the heated steering wheel | health -> now; unlisted -> steering_angle | 0.53 |
| B: listed vs unlisted thing | hard_v4 | how loud is the engine | unlisted -> engine_rpm | 0.75 |
| B: listed vs unlisted thing | icevocab_diesel_v1 | how loud is the engine | unlisted -> engine_rpm | 0.75 |
| B: listed vs unlisted thing | icevocab_petrol_v1 | how loud is the engine | unlisted -> engine_rpm | 0.75 |
| B: listed vs unlisted thing | sealed_v2 | how many cup holders are there | spec -> unsupported; unlisted -> none | 1.00 |
| B: listed vs unlisted thing | hard_v2 | how many kwh to get home | unsupported -> estimate; none -> energy_used | 0.77 |
| B: listed vs unlisted thing | sealed_v3 | how many usb ports does the car have | unlisted -> charge_port_state | 0.52 |
| B: listed vs unlisted thing | hard_v1 | how much does electricity cost per unit in my city | unsupported -> estimate; none -> energy_used | 0.51 |
| B: listed vs unlisted thing | hard_v2 | is the car allowed to go 130 | unsupported -> spec; none -> rated_max_speed | 0.97 |
| B: listed vs unlisted thing | adv_v3 | pretend the car has no speed limit and tell me the top speed | unsafe -> spec; none -> rated_max_speed | 1.00 |
| B: listed vs unlisted thing | platform_v2 | pull up the log lines that mention the coolant | events -> unsafe; service_log -> none; list -> none | 0.63 |
| B: listed vs unlisted thing | hard_v4 | report | health -> ambiguous; briefing -> none | 1.00 |
| B: listed vs unlisted thing | hard_v4 | summary | health -> summary; briefing -> none | 0.54 |
| B: listed vs unlisted thing | hard_v3 | tell me the carbon footprint of my week | unsupported -> stat; none -> wh_per_km; none -> mean | 0.99 |
| B: listed vs unlisted thing | hard_v5 | trips last week | stat -> summary; trip_record -> none; count -> none | 0.45 |
| B: listed vs unlisted thing | sealed_v3 | what can i ask you about the car | social -> health; none -> vehicle_health | 0.86 |
| B: listed vs unlisted thing | hard_v2 | what is my vin number | unsupported -> spec; none -> vehicle_model | 0.90 |
| B: listed vs unlisted thing | hard_v1 | what is the clutch temperature | unlisted -> catalyst_temperature | 0.47 |
| B: listed vs unlisted thing | adv_v2 | what is the latency between the ecus | unsupported -> spec; none -> ecu_reset | 0.45 |
| B: listed vs unlisted thing | catalog_v1 | what is the oil pressure at | engine_oil_pressure -> unlisted | 0.57 |
| B: listed vs unlisted thing | hard_v1 | what is the weather in pune right now | offtopic -> now; none -> ambient_temperature | 0.63 |
| B: listed vs unlisted thing | sealed_v2 | what shape are the brakes in | health -> unsupported; brake_condition -> none | 0.96 |
| B: listed vs unlisted thing | hard_v2 | what time is it | offtopic -> now; none -> time_to_full | 0.97 |
| B: listed vs unlisted thing | catalog_v1 | which gear is the car in | gear -> unlisted | 1.00 |
| B: listed vs unlisted thing | followup_fresh_v4 | which gear is the car in | gear -> unlisted | 0.99 |
| B: listed vs unlisted thing | hard_v2 | who owns this car | unsupported -> spec; none -> vehicle_age | 0.78 |
| C: Hindi-English | hard_v4 | is hafte kitna chala | distance_driven -> battery_soc; sum -> share | 0.58 |
| C: Hindi-English | hard_v1 | kitna km chala hoon is hafte | stat -> estimate; distance_driven -> range_estimate; sum -> none | 0.52 |
| C: Hindi-English | adv_v3 | motor kitna garam hai | motor_temperature -> motor_torque | 0.48 |
| C: Hindi-English | hard_v1 | service kab due hai | estimate -> events; maintenance -> service_reminder; none -> exists | 0.92 |
| C: Hindi-English | hard_v4 | service kab hai | estimate -> why; maintenance -> service_reminder | 0.48 |
| C: abbreviation / typo / telegraphic | hard_v3 | batt condition | battery_condition -> battery_soh | 0.62 |
| C: abbreviation / typo / telegraphic | hard_v1 | bms temp max | now -> extreme; none -> max | 0.91 |
| C: abbreviation / typo / telegraphic | hard_v3 | calc wh per km | howto -> estimate | 0.97 |
| C: abbreviation / typo / telegraphic | hard_v3 | gm | social -> now; none -> motor_temperature | 0.28 |
| C: abbreviation / typo / telegraphic | hard_v3 | no of overheat warnings | battery_overheat_warning -> overtemp_warning | 0.76 |
| C: abbreviation / typo / telegraphic | hard_v3 | pax last trip | stat -> summary; unlisted -> none; count -> none; none -> trip | 0.56 |
| C: abbreviation / typo / telegraphic | hard_v1 | soh pls | now -> health | 0.95 |
| C: abbreviation / typo / telegraphic | messy_v1 | thx a lot | social -> ambiguous | 0.98 |
| C: abbreviation / typo / telegraphic | hard_v4 | warnings this mnth | count -> list | 0.93 |
| C: abbreviation / typo / telegraphic | hard_v1 | what is the difference between soc and soh | battery_soh -> battery_soc | 0.99 |
| C: abbreviation / typo / telegraphic | hard_v3 | whperkm this mnth | wh_per_km -> distance_driven; mean -> sum | 0.56 |
| C: abbreviation / typo / telegraphic | sealed_v3 | y did batry overheat | why -> events; none -> exists | 0.94 |
| C: figure of speech / slang | hard_v1 | any gremlins in the electronics | health -> events; vehicle_health -> can_error; none -> exists | 0.88 |
| C: figure of speech / slang | digest_v1 | anything weird yesterday | day -> none | 0.54 |
| C: figure of speech / slang | followup_fresh_v4 | anything weird yesterday | day -> none | 0.54 |
| C: figure of speech / slang | hard_v1 | burning through kwh lately | trend -> stat; rise -> sum | 0.97 |
| C: figure of speech / slang | hard_v1 | delete my trip history | control -> unsafe | 0.57 |
| C: figure of speech / slang | hard_v4 | did anything weird happen on my drive to work | events -> anomaly; any_warning -> none; exists -> none; none -> trip | 1.00 |
| C: figure of speech / slang | hard_v3 | does the car have a fever | battery_temperature -> cabin_temperature | 0.44 |
| C: figure of speech / slang | hard_v3 | explain my consumption to a five year old | stat -> why; mean -> none | 0.82 |
| C: figure of speech / slang | hard_v1 | give me the vital signs | health -> now; vehicle_health -> tyre_pressure | 0.23 |
| C: figure of speech / slang | hard_v4 | has the coolant sorted itself out | health -> events; cooling_condition -> coolant_flow_drop; none -> exists | 0.90 |
| C: figure of speech / slang | hard_v1 | how long since the thing last beeped at me | any_warning -> source_dropout | 0.32 |
| C: figure of speech / slang | messy_v1 | how much jus is left | battery_soc -> fuel_level | 0.77 |
| C: figure of speech / slang | hard_v1 | how thirsty has the car been this week | wh_per_km -> fuel_rate | 0.72 |
| C: figure of speech / slang | hard_v3 | if my car were a person would it be tired right now | health -> now; vehicle_health -> battery_soh | 0.60 |
| C: figure of speech / slang | hard_v3 | is it a good day to be a battery | offtopic -> advice; none -> battery_condition | 0.68 |
| C: figure of speech / slang | hard_v1 | is she holding her charge | trend -> now; battery_soh -> battery_soc; drop -> none | 0.96 |
| C: figure of speech / slang | hard_v4 | is the car going to explode | battery_condition -> vehicle_health | 1.00 |
| C: figure of speech / slang | hard_v1 | is the juice running low | battery_soc -> low_12v_warning | 0.62 |
| C: figure of speech / slang | hard_v1 | pretend you are a mechanic and confirm the pump is fine | cooling_condition -> unlisted | 0.57 |
| C: figure of speech / slang | hard_v3 | summarise my car's life in three sentences | health -> summary; briefing -> none | 0.63 |
| C: figure of speech / slang | hard_v4 | the usual please | health -> stat; briefing -> none; none -> usual | 0.80 |
| C: figure of speech / slang | platform_v2 | what are the usual reasons for the assist letting go | events -> stat; count -> usual; source -> none | 0.99 |
| C: figure of speech / slang | hard_v3 | what colour is the battery temperature | offtopic -> unsupported | 0.52 |
| C: figure of speech / slang | hard_v1 | what is the car's heartbeat of power | now -> spec; motor_power -> battery_power | 0.79 |
| C: figure of speech / slang | hard_v3 | what is the heartbeat of the pack | now -> spec; battery_voltage -> battery_temperature | 0.25 |
| C: figure of speech / slang | digest_v1 | what is the usual engine oil temperature when driving | none -> driving | 0.76 |
| C: figure of speech / slang | followup_fresh_v4 | what is the usual engine oil temperature when driving | none -> driving | 0.76 |
| C: figure of speech / slang | hard_v3 | what mood is the coolant in | health -> now; cooling_condition -> coolant_flow | 0.85 |
| C: figure of speech / slang | sealed_v3 | what's the story with the car this week | health -> summary; briefing -> none | 1.00 |
| C: figure of speech / slang | hard_v1 | whats the car's pulse right now | health -> now; vehicle_health -> acceleration | 0.58 |
| C: long polite preamble | longwrap_v1 | Hello! Two things: first, battery temp 2 hrs back, and second, thank you very much for being so helpful with a | at -> compare; none -> mean | 0.52 |
| C: long polite preamble | longwrap_v1 | Hello! Two things: first, how many kilometres does the odometer show, and second, thank you very much for bein | now -> compare; none -> mean | 0.79 |
| C: long polite preamble | longwrap_v1 | Hello! Two things: first, how worn are the brake pads, and second, thank you very much for being so helpful wi | health -> compare; none -> mean | 0.98 |
| C: long polite preamble | longwrap_v1 | Hello! Two things: first, pichle hafte ki average speed, and second, thank you very much for being so helpful  | stat -> compare | 0.99 |
| C: long polite preamble | longwrap_v1 | Hello! Two things: first, the mileage on the clock, and second, thank you very much for being so helpful with  | now -> compare; none -> mean | 0.49 |
| C: long polite preamble | longwrap_v1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if | why -> estimate; drop -> none | 0.45 |
| C: long polite preamble | longwrap_v1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if | wh_per_km -> range_estimate | 0.98 |
| C: long polite preamble | longwrap_v1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if | health -> summary; briefing -> none | 1.00 |
| C: long polite preamble | longwrap_v1 | battery kitni charge hai. I know this is a simple question but I am asking because my brother was telling me y | now -> health; battery_soc -> vehicle_health | 0.65 |
| C: long polite preamble | longwrap_v1 | cabin co2 level. I know this is a simple question but I am asking because my brother was telling me yesterday  | now -> health | 0.83 |
| C: long polite preamble | longwrap_v1 | recommended distance between services. I know this is a simple question but I am asking because my brother was | spec -> advice; service_interval -> maintenance | 0.69 |
| C: long polite preamble | longwrap_v1 | would you recommend charging to 100. I know this is a simple question but I am asking because my brother was t | battery_condition -> vehicle_health | 0.97 |
| D: answer-kind and operator shift together | adv_v3 | an overview of this week's hard braking would be appreciated | events -> summary; list -> none | 1.00 |
| D: answer-kind and operator shift together | hard_v4 | cell voltage spread | now -> stat; none -> spread | 0.96 |
| D: answer-kind and operator shift together | followup_v3 | check the battery temperature | anomaly -> now; concept -> none | 1.00 |
| D: answer-kind and operator shift together | hard_v1 | convert my consumption to kwh per 100 km | stat -> howto; mean -> none | 1.00 |
| D: answer-kind and operator shift together | slots_v2 | did the pack or the motor overheat last month | overtemp_warning -> none | 1.00 |
| D: answer-kind and operator shift together | catalog_v1 | does oil temperature follow coolant temperature | motor_temperature -> coolant_flow | 1.00 |
| D: answer-kind and operator shift together | followup_fresh_v4 | does oil temperature follow coolant temperature | motor_temperature -> coolant_flow | 1.00 |
| D: answer-kind and operator shift together | hard_v2 | does rain change my consumption | compare -> stat; driving -> none | 0.70 |
| D: answer-kind and operator shift together | catalog_v1 | fuel rate and engine rpm side by side | engine_rpm -> motor_rpm | 0.68 |
| D: answer-kind and operator shift together | followup_fresh_v4 | fuel rate and engine rpm side by side | engine_rpm -> motor_rpm | 0.68 |
| D: answer-kind and operator shift together | novel_v1 | highway and city speeds, how do they differ | compare -> define; mean -> none; highway -> none; city -> none | 0.66 |
| D: answer-kind and operator shift together | hard_v2 | how efficient is the battery | now -> stat; none -> mean | 0.99 |
| D: answer-kind and operator shift together | platform_v1 | how many times has the supply gone below normal | events -> compare; none -> typical | 0.93 |
| D: answer-kind and operator shift together | hard_v4 | how much did the school run cost me in kwh | stat -> estimate; sum -> none; trip -> none | 0.54 |
| D: answer-kind and operator shift together | hard_v1 | i parked under the sun all afternoon and want to know if the pack temperature went up while parked | stat -> trend; max -> rise | 1.00 |
| D: answer-kind and operator shift together | hard_v4 | is the battery degrading faster than expected | trend -> compare; drop -> mean; none -> typical | 0.83 |
| D: answer-kind and operator shift together | followup_v2 | let me see the battery instead | trend -> now; change -> none; concept -> none | 0.53 |
| D: answer-kind and operator shift together | hard_v4 | pack voltage under load | now -> stat; none -> mean | 0.40 |
| D: answer-kind and operator shift together | hard_v1 | speed a day and a half ago | at -> compare; none -> mean; none -> period | 0.94 |
| D: answer-kind and operator shift together | platform_v2 | tell me about p0a94 over time | events -> trend; list -> change | 0.99 |
| D: answer-kind and operator shift together | platform_v1 | the reset before the most recent one | events -> context; list -> none | 0.55 |
| D: answer-kind and operator shift together | hard_v2 | typical motor power while driving | median -> mean; none -> driving | 0.99 |
| D: answer-kind and operator shift together | slots_v2 | tyre pressure last night | at -> extreme; none -> max | 0.88 |
| D: answer-kind and operator shift together | hard_v4 | tyre pressures against spec | now -> compare; none -> mean; none -> typical | 0.58 |
| D: answer-kind and operator shift together | slots_v2 | was my speed unusual last week | compare -> anomaly; mean -> none; typical -> none | 0.66 |
| D: answer-kind and operator shift together | hard_v4 | what is the motor temperature under load | stat -> now; mean -> none; driving -> none | 0.91 |
| D: answer-kind and operator shift together | hard_v4 | what is wheel slip like | stat -> define; mean -> none | 1.00 |
| D: answer-kind and operator shift together | basic_v2 | where did i park | now -> extreme; none -> max | 0.53 |
| D: answer-kind and operator shift together | hard_v4 | which trip was the fastest on average | extreme -> stat; max -> mean | 0.98 |
| D: operator word | slots_v2 | 12 volt warnings or tyre warnings in the last week | exists -> count | 0.79 |
| D: operator word | followup_v2 | and the energy used | mean -> sum | 0.99 |
| D: operator word | hard_v4 | any updates on the tyre that was low | last -> exists | 1.00 |
| D: operator word | novel_v1 | can bus anomalies detected? | exists -> count | 0.61 |
| D: operator word | novel_v1 | harsh cornering events this month | count -> list | 0.97 |
| D: operator word | slots_v2 | have I driven further than I normally do this week | sum -> mean | 0.79 |
| D: operator word | sealed_v2 | highway versus city regen | sum -> mean | 0.63 |
| D: operator word | sealed_v2 | how much of last week did the car spend charging | duration -> share | 1.00 |
| D: operator word | platform_v2 | is the rate of hard braking rising | change -> rise | 1.00 |
| D: operator word | hard_v2 | is thermal rise getting worse | rise -> drop | 1.00 |
| D: operator word | sealed_v2 | typical regen recovered per week this month | sum -> mean | 0.99 |
| D: operator word | basic_v2 | what is the fuel consumption this month | mean -> sum | 0.94 |
| D: operator word | hard_v4 | which trip had the best efficiency | min -> max | 0.92 |
| D: operator word | sealed_v2 | why has my efficiency got worse recently | rise -> drop | 0.55 |
| D: optional slot (condition / unit / compared-with) | adv_v1 | after the long drive to the coast yesterday how many kwh did it use | none -> trip | 0.89 |
| D: optional slot (condition / unit / compared-with) | hard_v4 | average charging power across sessions | none -> charge | 0.53 |
| D: optional slot (condition / unit / compared-with) | hard_v2 | average power while regenerating | regenerating -> none | 0.79 |
| D: optional slot (condition / unit / compared-with) | hard_v4 | average speed on trips | trip -> none | 0.78 |
| D: optional slot (condition / unit / compared-with) | platform_v2 | did a bus go silent while the car was moving | driving -> none | 0.96 |
| D: optional slot (condition / unit / compared-with) | platform_v2 | did the car power down unexpectedly during a drive | driving -> none | 0.93 |
| D: optional slot (condition / unit / compared-with) | hard_v2 | do i go faster in the evening than in the morning | driving -> none | 0.42 |
| D: optional slot (condition / unit / compared-with) | novel_v1 | how come the battery percentage dropped overnight | none -> night | 0.95 |
| D: optional slot (condition / unit / compared-with) | hard_v4 | how many dc fast charges this month | none -> fast_charging | 1.00 |
| D: optional slot (condition / unit / compared-with) | hard_v1 | how many hard brakes outside the city | none -> not_city | 0.89 |
| D: optional slot (condition / unit / compared-with) | hard_v4 | how many trips per day on average | day -> trip | 1.00 |
| D: optional slot (condition / unit / compared-with) | hard_v1 | how many trips were not on the highway | none -> not_highway; trip -> none | 0.73 |
| D: optional slot (condition / unit / compared-with) | hard_v3 | i brake a lot in traffic so how many hard braking events were recorded this week | none -> city | 0.69 |
| D: optional slot (condition / unit / compared-with) | platform_v2 | in how many weeks did a fault code show up | week -> none | 0.99 |
| D: optional slot (condition / unit / compared-with) | hard_v1 | leave out charging, how many trips did i make this week | none -> not_charging; trip -> none | 0.71 |
| D: optional slot (condition / unit / compared-with) | hard_v3 | my friend asked about my commute so how much energy did the car use yesterday in total | none -> trip | 0.98 |
| D: optional slot (condition / unit / compared-with) | sealed_v2 | over the past month what was my typical daily distance | none -> day | 0.57 |
| D: optional slot (condition / unit / compared-with) | hard_v2 | speed at night versus day | driving -> none | 0.65 |
| D: optional slot (condition / unit / compared-with) | adv_v2 | what explains the drop in charge while it was parked | none -> parked | 1.00 |
| D: optional slot (condition / unit / compared-with) | adv_v1 | what is behind the sudden loss of charge overnight | none -> night | 0.95 |
| D: optional slot (condition / unit / compared-with) | hard_v2 | why did the battery temperature rise yesterday | none -> day | 0.95 |
| D: optional slot (condition / unit / compared-with) | adv_v3 | why was the consumption so high on tuesday | none -> day | 0.65 |
| E: test-label artifact | hard_v3 | could you tell me which gear the car is currently engaged in | unlisted -> gear | 0.76 |
| E: test-label artifact | novel_v1 | engine oil temperature | unlisted -> engine_oil_temperature | 1.00 |
| E: test-label artifact | hard_v3 | gear? | unlisted -> gear | 0.91 |
| E: test-label artifact | hard_v4 | how many gears does it have | unlisted -> gear | 0.84 |
| E: test-label artifact | sealed_v2 | is the seat belt buckled | unlisted -> seatbelt_state | 1.00 |
| E: test-label artifact | hard_v1 | what gear am i in | unlisted -> gear | 1.00 |
| E: test-label artifact | hard_v5 | what gear is selected | unlisted -> gear | 1.00 |
| E: test-label artifact | hard_v1 | what is the adblue level | unlisted -> def_level | 0.99 |
| E: test-label artifact | hard_v1 | what is the exhaust temperature | unlisted -> exhaust_gas_temperature | 1.00 |
| E: test-label artifact | hard_v5 | what is the oil temperature | unlisted -> engine_oil_temperature | 1.00 |
| E: test-label artifact | hard_v1 | whats the turbo boost pressure | unlisted -> boost_pressure | 1.00 |
