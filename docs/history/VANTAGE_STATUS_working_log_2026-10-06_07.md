# VANTAGE status log

## Audit of 6 Oct 2026 (experiment validity and semantic correctness) — what was found and fixed

| # | Finding | Category | Verified how | Fix | Guard |
|---|---|---|---|---|---|
| 1 | Retrieved examples (`bank_k`) never reached the prompt. Saved `lfm_full` and `lfm_full_bank4` predictions are identical on 120/120 questions. | experiment setup | read `_user()`; compared the two result files | prompt now includes "SOLVED EXAMPLES" | `PlannerWiring.test_retrieved_examples_reach_the_prompt` (stub backend) |
| 2 | The parser draft (`draft=`) never reached the prompt. | experiment setup | read `_user()` | prompt now includes the draft | `test_the_parser_draft_reaches_the_prompt` |
| 3 | A demonstration taught "do you record tyre temperature" -> tyre *pressure* when the signal was absent (the substitution the design forbids). | experiment setup / semantics | read `make_examples` | example only emitted when the asked signal exists | `test_no_example_substitutes_a_different_signal...` |
| 4 | `explain_change` / `detect_change` dropped the query's conditions and selection (a "highway only" question analysed all driving). | execution | read `_series`; test | conditions/selection now apply to every series; the trip-level contribution model (fitted on all trips) is not offered for a restricted scope | `ChangeOperatorsKeepTheScope` |
| 5 | Model fallback could substitute a signal the question did not name (`steering torque` -> `motor_torque`) or a condition that only resembles a word (`fastest` -> `fast_charging`); the grounding guard accepted both. | interpretation (model) | classified the 19 wrong answers | grounding guard: the signal must be one the question names; every named condition must be mentioned | cases checked directly; wrong-among-answered 21.4% -> 16.7% (abstentions rose) |
| 6 | 12 of the final-set gold queries could not be verified by the engine (numbers in "too few days" explanations, default 60 s event window, rank durations in minutes). | execution / evidence | ran every gold query through the engine | engine-produced reasons and window labels pass the number verifier; durations carry minute evidence | gold execution now: 0 unverified on dev (120) and final (308) |

### What the earlier numbers do and do not mean
* **Void**: every result labelled `bank4` or `draft` (LFM and Qwen cascades). They measured the static prompt only.
* **Still valid** (static schema prompt, no retrieved examples, no draft): frozen LFM2.5-350M alone 12% outcome on dev; frozen Qwen-0.5B/1.5B alone 31% / 49%; rules -> Qwen-0.5B+LoRA cascade 66.7% (the LoRA was trained in an earlier session, not in this one).
* The question files `bench/indep_dev.tsv` / `indep_final.tsv` were written by the same author as the planner prompts (not independent until the owner edits them); the final file has not been used for any tuning or selection.

### The 19 wrong-but-answered cases of the cascade on dev (before the fixes), by source
* 12 from the model fallback: operation/signal confusion (`explain_change` read as `explain_event`, capability instead of count, a statistic dropped), the two substitutions above, and follow-ups the model cannot see (it gets one question).
* 7 from the rules at the time: 3 are ambiguous gold (`last wk` read as the calendar week vs 7 days; no period stated; `energy use` vs consumption), the rest were real (a comparison read as a single aggregate, a count read as a list).
* 0 from the executor: re-executing the gold queries gives the same result as the evaluator's gold outcome by construction; independent checks of the executor are in `tests/test_vantage.py` (largest day-to-day move vs the generator's own daily truth).

## Progress on the revised sequence (6 Oct 2026, evening) - what was run and what it shows

| Step | What was done | Result |
|---|---|---|
| 2a Independent executor check | `tests/test_reference.py`: statistics, event counts, trip totals/rankings and period consumption recomputed with plain SQL/numpy straight from the tables, sharing no code with the planner/executor | 4 tests pass (speed statistics are skipped on purpose: the catalog defines them over driving samples) |
| 2b Retention | `tests/test_retention.py`: delete the 1 Hz tier, a driven day, or a whole window and re-ask | answers narrow to "one-minute averages" with the limit stated; a lost day is reported ("readings on N of M days with trips"); an aged-out window is declined with a reason; trip-ledger totals are unchanged. **Fix found by this test:** the coverage rule tolerated one lost day out of 14 and could not tell a lost day from a parked day; it now compares against days on which the trip ledger shows driving (0 false flags on 84 intact queries, 4 vehicles) |
| 2c Explanations vs injected causes | `bench/counterfactual.py` builds twin histories (same routes and synthetic weather, the roof box exists only in one); `bench/eval_attribution.py` scores `explain_change` | 13 pairs, 12 with a box effect >= 5 Wh/km: the unrecorded cause stays in the *unexplained* part by a median 76% (mean 77%); recorded factors absorb a median 30% of it (>50% in 1 of 12 pairs). Negative control (no box): median |unexplained|/|gap| 0.49 but small in absolute terms (about 2-3 Wh/km). Change-point location: 2 days or less on step causes (tyre leak, roof box), 8 days late on a gradual cooling-pump fault. **Not a proof**: simulated causes, one cause type, synthetic weather |
| 2d Interpretation contract | `vgpt/interpretation.py`: each field of a query is labelled question / prior turn / default (disclosed) / inferred / ungrounded; ambiguities ("last week" = 7 days vs calendar week; "energy use" = per-km consumption) are recorded and disclosed; used as a second guard for model-written queries and to disclose defaults in answers | On the historical dev predictions (all earlier planner runs), abstaining on ungrounded fields lowers wrong-among-answered: hybrid rules->Qwen-LoRA 18.6% -> 13.8% (1 correct answer refused), hybrid rules->LFM 27.8% -> 17.4% (0 refused), frozen LFM alone 82.6% -> 51.6%. **Caveat**: the checker's vocabulary was tuned while reading the dev and final gold queries, so these numbers are optimistic and say nothing about a fresh set; 37 of 395 gold queries (9%) are themselves flagged (mostly vocabulary gaps, relative follow-ups such as "the day before", and one gold query that looks wrong: f260 "same for the previous week" -> last:14:day) |
| 3a Prompt capacity | `bench/check_prompt.py` with 4 retrieved examples + parser draft, LFM2.5-350M, n_ctx 4096 | 0 of 428 questions overflow; worst case 4059 of 4096 tokens (thin margin: consider n_ctx 4608); no silent truncation anywhere (an overflow abstains) |
| 3b Prefix-state equivalence | restored prefix state vs ordinary full-prompt inference, same grammar, greedy | identical output on 30 of 30 sampled questions |
| 3c Complete-answer latency, local | `bench/run_guest_v2.py`, first 40 dev questions without history, LFM2.5-350M Q4_K_M, pinned to 2 cores (x86 E-cores, **not ARM**); prompt state restored from disk | static prompt: total per question mean 461 ms, p50 23 ms, p95 1.79 s (rules path 103 ms mean; model path 1.41 s mean, p95 2.16 s); peak process memory 509 MB. With 4 retrieved examples + parser draft: model path 3.08 s mean (p95 3.78 s), overall mean 910 ms, p95 3.77 s, peak 529 MB. Outcome counts are not accuracy (answered 29 / 31, refused 7 / 5, no_data 4 / 4 of 40) |
| 3d ARM (QEMU) | `vm/driver.py` is written and boots the guest, loads the model with llama-cpp-python 0.3.36 (the guest's) and starts building the prompt state; **stopped at the owner's request after ~20 minutes of the first prompt-state build, before any result** | no ARM number exists yet; run again later with `python vm/driver.py smoke` |

Regression suite after these changes: 309 tests; the only failure was a backspace character in a new test file (fixed); the inherited 289 pass.

### Corrections to the table above (reviewer follow-ups, same evening)
* **13 pairs constructed, 12 scored for the headline numbers:** 13 twin pairs were built (6 `cold` seeds, 6 `roadwarrior` seeds and the original `roadwarrior` preset). All 13 were run. The coverage/leak statistics use only pairs in which the roof box changed consumption by at least 5 Wh/km over the window; `cf_roadwarrior_s16` (true box effect 4.1 Wh/km) is below that threshold and is excluded from those two statistics (it is still in the negative-control line). Nothing else was dropped. Its inclusion would add a case where the engine over-assigned to the unexplained part (R 14.3 vs box 4.1).
* **What the attribution result supports:** a qualified claim - an unrecorded step-like cause mostly stays in the unexplained part - not causal accuracy. Recorded factors do absorb part of an unseen cause (confounding remains; in `cold` the weather also changes); the explanation text must keep saying "associated / estimate".
* **Context window:** raised to 4608 tokens (default in `PrefixLlama`, `SchemaPlanner`, the benchmark scripts; the prefix-state cache key now includes it). Worst case over all 428 questions with 4 retrieved examples, the parser draft, the previous turn and the 80-token output allowance: 4059 of 4608 (549 spare). Memory/latency to be re-measured with it.
* **Evaluator defect fixed:** planner-only runs with `--draft rules` never received the draft (`run()` passes `draft=None`; only the cascade supplied one). The planner now builds the draft itself from the same rules parse.
* **Final set:** `indep_final.tsv` is regression data only (it influenced the interpretation guard's vocabulary). A fresh acceptance set, independently reviewed, is to be written and kept out of development.

### Pre-registered comparison on DEV only (written before any of these runs)
Same questions, same history (gold), same schema prompt for every planner variant; planner-only results and complete-cascade results are reported separately; accuracy (correct of all, correct of answerable), answer coverage, wrong-among-answered, latency (mean/p50/p95) and peak memory are reported together.
1. **Planner-only, frozen LFM2.5-350M Q4_K_M:** static | + retrieved examples (k=4) | + parser draft | + both.
2. **Retained over `static` only if** dev correct-of-all rises by at least 3 points AND wrong-among-answered does not rise. Between two qualifying variants take the cheaper unless the dearer adds at least 3 further points. The cost (latency, prompt tokens) is reported either way.
3. **Cascade (rules -> frozen planner):** guard none | numbers | interpretation | both. The interpretation guard is kept only if it lowers wrong-among-answered and correct-of-all falls by no more than 1 point versus the number guard.
4. **Reference points (not candidates):** frozen Qwen2.5-0.5B with the same prompt (static), and the earlier Qwen-LoRA cascade, labelled historical.
5. No training, no new downloads, no phrase handlers. One run per configuration; differences under about 3 points on 120 questions are within noise (95% CIs are in the output) and are not reported as wins.

## Results of the pre-registered dev comparison (23:00-23:45, 6 Oct 2026; 120 dev questions, gold history, 2 pinned x86 cores, one run each)
Full tables: `results/table_planner_only.txt`, `results/table_cascade.txt` (per-run JSON in `results/abl_*.json`). Columns: correct of all (95% CI), correct of answerable, answered share, wrong among answered, false action on refusal cases, false refusals, latency.

**Planner only (frozen model, same schema prompt)**
| variant | correct/all | answered | wrong among answered | mean ms |
|---|---|---|---|---|
| LFM2.5-350M static | 18.3% (12-26) | 81.5% | 84.1% | 1161 |
| + retrieved examples (k=4) | 16.7% (11-24) | 82.4% | 89.9% | 2699 |
| + parser draft | 49.2% (40-58) | 90.7% | 46.9% | 1591 |
| + both | 48.3% (40-57) | 94.4% | 52.0% | 3275 |
| Qwen2.5-0.5B static (reference) | 26.7% (20-35) | 83.3% | 74.4% | 2576 |

**Complete cascade (rules, then the model with the rules' draft, then a guard)**
| system | correct/all | answered | wrong among answered | false refusal |
|---|---|---|---|---|
| rules alone | 54.2% (45-63) | 57.4% | 14.5% | 43% |
| rules -> LFM+draft, no guard | 56.7% | 95.4% | 40.8% | 0% |
| ... number guard | 55.8% | 79.6% | 32.6% | 18% |
| ... **interpretation guard** | **59.2% (50-68)** | 80.6% | **29.9%** | 19% |
| ... both guards | 56.7% | 75.9% | 29.3% | 23% |
| rules -> Qwen-0.5B+LoRA (earlier training, historical) + number guard | 62.5% | 77.8% | 22.6% | 20% |
| ... + interpretation guard | 64.2% (55-72) | 77.8% | 20.2% | 20% |

**Pre-registered decisions applied**
* Retrieved examples: **dropped** (no gain, +1.5 s, wrong rate not lower). The earlier hope for catalog retrieval is not supported with this model.
* Parser draft: **kept** (+31 points planner-only, wrong rate 84% -> 47%, +0.4 s). Caveat: the draft is the rules' own partial parse, so this is largely "the model edits the rules' answer"; it also explains why the planner alone with a draft (49%) is below the rules alone (54%) on this set.
* Interpretation guard over the number guard: **kept** (lower wrong-among-answered, higher correct/all).
* Memory/latency with the larger context not yet re-measured end to end.

**What this does and does not show (read before quoting any number)**
* The frozen LFM2.5-350M adds little to the rules: +5 points correct (59.2 vs 54.2; the intervals overlap heavily) at the price of wrong-among-answered rising from 14.5% to 29.9%. On the answered questions the rules are more reliable than the model. Of the model's 48 attempts at questions the rules refused: 8 correct, 17 wrong, 20 abstained (guard), 2 false actions, 1 other.
* The earlier Qwen-0.5B+LoRA cascade is still better (64.2%, 20.2% wrong) than any frozen variant. A training-free route to that level is **not demonstrated**; the remaining gap is not explained by anything fixed so far.
* Where the model's 17 wrong answers come from (fields): operation 14, statistic 7, period 7, event 5, signal 5, conditions 5. Recurring patterns: a dropped follow-up condition ("only on the highway" -> condition lost), a specific signal widened to a general one (front-left tyre -> tyre_pressure), count read as list, a period replaced by the default, and a statistic changed (fastest -> mean). Two are scoring questions rather than model errors (d081 adds `unit=psi` when the user said psi; d052 drops a threshold that the user stated - this one is a real error).
* The rules' own 9 wrong answers are 5 operation, 3 statistic, 2 period, 2 group/selection, 1 event/signal/condition (some questions have several wrong fields); the rules refuse 46 of 108 answerable questions - this refusal share, not the 9 wrong answers, is where most of the remaining dev score is lost.
* The dev set was written by the same author as the prompts and lexicon; 120 questions; one run each. Differences under about 3 points are noise.
* Sealed acceptance set: `bench/acceptance_v1.tsv` (114 questions, unreviewed draft; all gold queries validate and execute: 87 answered, 9 no data, 9 missing data, 2 unsupported). Not run on anything yet.

## Controlled query completion, generic rule composition, paired statistics (7 Oct 2026, overnight)
Paired comparisons (`bench/paired.py`: per-question differences, bootstrap interval, exact McNemar test) replace the earlier "under 3 points is noise" remark. Tables: `results/table_r6.txt`; per-run JSON `results/abl_*.json`.

**1. Constrained completion of the model's query (`vgpt/completion.py`) - hypothesis NOT supported.** The model's query is repaired against what the question states: fields of the rules' partial parse that the interpretation contract traces to words of the question are locked, omitted stated conditions are restored, unsupported fields fall back to the previous turn. Dev, same questions: LFM cascade identical on all 120 questions with and without it; Qwen-LoRA cascade 77 -> 76 correct, wrong answers 17 -> 20 (p=0.25). Same with the system's own history instead of gold history (no difference from gold history). The semantic check on the rules' own answers made no difference on dev and lost 2 correct on the diagnostic set. **Why:** the information that completion needs is not in the draft - in the cases that matter the rules had not parsed it ("faster than 60 mph", "front left", the follow-up "only on the highway"), so there was nothing to lock. The failures sit upstream, in the rules' composition. (The code is kept behind `--complete`; it is not used by default.)

**2. Generic compositional steps in the rule parser (no question-specific handlers):**
* wheel position ("front left", "rear right") specialises a tyre or wheel signal and is never dropped silently; a follow-up "and the front right" changes only the wheel;
* comparatives with a number and unit (faster / slower / hotter / colder than N) are conditions; the unit names the signal;
* follow-ups: discourse words (same, actually, only, just, instead) no longer block a follow-up; "the day / week before", "the previous week" shift a calendar period; "without X", "actually all trips" remove the matching condition; a leading "now" before "same" is not a time;
* superlatives and "how hot/fast ... did it get" are maxima ("how cold ... get" minima); "variable" is variability; existence questions ("was there", "did ... at all") are counts;
* unique **describe-by-parts** fallback: when the strict lexicon finds no signal and no event, a signal or event is accepted if all words of one of its phrases occur (synonym-folded) and exactly one candidate has the greatest overlap; events additionally need an event-shaped question (count / list / when / any / ...). First attempt (no gating) added 5 wrong answers for 2 correct and was gated; the gated version added none.
Rules alone on dev: 54.2% -> 64.2% correct (paired 67 -> 77 correct of 120 after the first batch of steps, McNemar p=0.012 for the largest step), wrong answers unchanged at 9; refusals 43% -> 31%. **Contamination warning:** these steps were designed while reading the dev failures and the synonym table was first built while reading dev/final gold queries, so the dev gain is optimistic. Diagnostic set (30 conversation/condition/sensor/default questions; three gold periods were corrected to the documented convention after the first run): rules 18 -> 20 of 30 correct. The sealed acceptance set has not been touched and is the only number that can confirm any of this.
Regression suite after these changes: 318 tests (including 8 new composition tests), see results/tests_audit6.log. A last fix (how hot did it get = maximum even when the words were consumed by another match) came after the cascade table above: rules alone on dev 78 of 120 correct, 8 wrong.

**3. Complete cascade on the improved rules (dev, gold history, interpretation guard, draft, no retrieval)**
| system | correct/all | answered | wrong among answered | false action (of 12) |
|---|---|---|---|---|
| rules alone | 64.2% | 68.5% | 12.2% (9/74) | 0 |
| rules -> frozen LFM2.5-350M | 68.3% | 87.0% | 23.4% (22/94) | 2 |
| rules -> Qwen-0.5B + existing LoRA (historical) | 70.8% | 84.3% | 17.6% (16/91) | 2 |
Paired: LFM over rules +5 correct (p=0.18, not significant) but +13 wrong answers (p=0.0002); Qwen-LoRA over rules +8 correct (p=0.04) and +7 wrong (p=0.016); LFM vs Qwen-LoRA +3 correct (p=0.61), -6 wrong (p=0.11). On the diagnostic set: rules 23/30, LFM 23/30, Qwen-LoRA 25/30. The LoRA's interface ignores history, so its follow-up behaviour comes entirely from the rules. **Reading:** the model fallback buys coverage (+15-19 points answered) at the price of confident errors; neither model is a convincing default. The rules are the more reliable component; whether a fallback should answer at all is a product decision (risk vs coverage), not a modelling one.

**4. Complete-answer latency, local x86, 2 pinned cores, context 4608, draft on, retrieval off (60 dev questions)**: mean 396 ms, p50 9 ms, p95 1.9 s; rules path mean 75 ms (p95 0.6 s), model path mean 1.68 s (p95 2.5 s, model 1.6 s of it); peak process memory 533 MB; prefix state restored in 0.07 s. ARM numbers: none (QEMU deferred).

**Not done (stated plainly):** unfamiliar signal identifiers with schema descriptions (renamed-schema test through the full harness), missing-specific-sensor variants on a vehicle that truly lacks the wheel sensors (this vehicle records all four tyres, so the diagnostic only tested that the specific one is used), and a model-side change that uses the draft as a constrained prefix rather than a hint. Remaining rule failures on dev by kind: causal/trend questions phrased unusually (why/when did ... get worse), event phrasings outside the described-by-parts reach (driver assist switched off, fault code appeared), unit conditions with a context word ("95 degrees fahrenheit" for the pack, "20 percent" for charge), selection ("longest charging session"), pronoun follow-ups ("those", "then", "that"), and gold-convention disagreements (count vs list for "any ... warnings").

## Not yet validated (stated so nobody reads more into the tests than they show)
* The contract and explanation layers have unit tests and 144 stress queries without a verifier failure; that shows the wording is consistent with the evidence, not that the *interpretation* of a user's question is right.
* `explain_change` contributions are a trip-level regression on a simulated fleet; attribution against injected causes is now measured on counterfactual twins (see above) but only for one cause type (roof box) and synthetic weather.
* Gold outcomes for `explain_event` are not executed in the evaluator (no legacy engine attached); those rows are scored by exact match.

## Proposed corrections and verification plan (nothing below has been run)
1. **Re-run the planner experiments validly, once**, on the DEV set only: static prompt, + retrieved examples (k=4), + parser draft, + both; models: the existing Qwen-0.5B (frozen and with the earlier LoRA), LFM2.5-350M. Pre-registered decision: a variant is kept only if it improves dev outcome AND does not raise wrong-among-answered; ties go to the smaller memory footprint. The final file is evaluated exactly once, after the configuration is frozen.
2. **Report error classes separately**: setup / interpretation / execution / missing evidence, plus confident-wrong rate as its own headline.
3. **Add a second, independent reference for execution**: recompute a sample of gold answers (sums, means, counts, extremes) directly with numpy/SQL from the raw tables and compare with the executor.
4. **Retention tests**: delete raw 1 Hz rows for a window and check that answers degrade to the narrower claim instead of keeping the fine-resolution wording.
5. **Latency / memory on the QEMU A55 guest** with the corrected pipeline (complete-answer time, p50/p95, resident memory of the whole guest), labelled as emulation.
6. Then UI integration and the submission document.

## PRE-REGISTRATION of the bounded model-role experiment (written 7 Oct 2026 before the experiment was implemented or run)
**Variant under test: candidate selection with the frozen LFM2.5-350M.** A deterministic generator proposes candidate queries from the question, the actual accepted conversation state and the runtime schema (every signal / event / period / condition / statistic the question supports, with the supporting phrases kept); the frozen model only chooses among numbered candidates (output constrained to an index). It does not write queries. A chosen candidate is answered only if it passes the support + coverage checks; otherwise the top readings are offered as a clarification.
**Measured separately:** (1) candidate availability = share of answerable dev questions for which the gold query is among the candidates (generator quality; if below 70% the model cannot be judged and the result is reported as a generator limitation); (2) selection accuracy given availability; (3) end-to-end outcomes.
**Compared on the same dev and diagnostic questions, same history policy (the system's own state), against:** rules + policy (no model), rules + policy + whole-query LFM (already run), rules + policy + the earlier Qwen-0.5B + LoRA (historical baseline, no new training).
**Success criterion (all must hold):** wrong answers do not exceed those of rules + policy; correct answers plus clarifications that contain the intended reading exceed rules + policy by at least 4 of 120 dev questions; model-path p95 latency under 2.5 s on the pinned laptop cores; the context needed fits the 4608-token window with the 80-token output allowance.
**Transfer criterion:** on the new-signal fixture (a signal the registry never saw, described only by its label and synonyms), the selector must choose the correct candidate on at least 8 of 10 questions written for that signal.
**Stopping rule:** one run per configuration. If the criteria fail, the result is reported as "SLM role unproven for this variant" and the experiment stops; the generator or prompt is not tuned against the same failures.

## PRE-REGISTRATION of the final performance measurement (written before it was run)
**"Common queries"** (the case study's under-2-second target) are defined as every dev-set question in the families LK (lookups, statistics, extremes), EVT (event lists and counts), THR (thresholds), CMP (comparisons), SEL (trip / charge selections), CAP (capability), FU (follow-ups), MSY (messy wording) and REF (refusals) - 100 of 120 questions (the first draft of this paragraph said 86; the family list is unchanged, the count was wrong). **Analytic queries** (EXPL explanations, CHG change detection / trends, CTX event context) are reported separately with their own latency, not folded into the common figure and not excluded from the report.
**Reported per query:** time to a clarification and time to a final answer are separate; the stages (rules, candidates, model selection, execution + composition + verification); by family and by path (rules / candidate+model / clarification).
**Configuration measured:** the frozen final configuration - interpretation guard, clarification-first policy, candidate selection with the frozen LFM2.5-350M (Q4_K_M), context 4608, retrieved examples off, pinned to 2 x86 cores, prompt state restored from disk. Cold start (model load + prompt-state construction) is measured separately.
**Also measured:** warm inference; concurrent ingestion at real-time pace while querying; peak process memory and the sum for the whole deployment (assistant + ingest daemon + model); a 90-day history and a 300-signal schema; the cost of the attribution calculation.
**Reading rule:** a path that exceeds 2 s is reported as exceeding it; the overall percentile is never presented as every path meeting two seconds. All numbers are x86 laptop numbers; ARM is not measured.
