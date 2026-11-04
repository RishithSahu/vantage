# 17. Questions over their limit on the Pi (full pass 2026-10-28)

Source: `results/full_pass_pi9.json` (4,096 questions, native engine, 2 cores, 4-year electric car, compact hints). Limit: 2 s for a question under 37 words, 2.5 s for longer. **312 of 4096 questions are over (7.6 %)**: 308 short, 4 long. Median over all questions 1.05 s, p95 2.25 s, p99 3.08 s. A question can have more than one cause, so the counts below add to more than 312.

## Causes

| cause | questions | what it is and the fix |
|---|---|---|
| M1 long prompt tail | 134 | The tail (question + glossary hint line + template) is 40 or more tokens; the Pi prefills about 35 ms a token. Done: compact and pruned hints (-39 % hint tokens). Next, no training: (a) skip a hint when the question already spells the term out (needs the audit); (b) keep only the first hint when it alone decides the frame (the per-term ablation lost 1-5 questions per term, so only the safest); (c) one shared short form for the 'in an electric vehicle ...' phrases. With training: a model that reads the term without a hint line. |
| M3 prefill+passes together | 74 | Tail 20-39 tokens and 2-3 passes: each part is ordinary but the sum reaches the limit. Same levers as M1 and M2; each saves a few hundred ms. |
| M2 many decode passes | 36 | The frame has many non-'none' words, so the draft-and-verify decoder needs 4 or more passes at 156 ms. Next: a better draft (copy the words that follow from the question and the previous frame, accept more rows per pass), measured under the no-accuracy-change rule; a head that predicts two words at once (training) is the real fix. |
| A5 other analysis / first use of a lake signal | 32 | Mostly the first question after a start that needs a lake signal (tyre, 12 V, brake): its column copy is made on the spot (about 2 s on the Pi). Fix: build the lake copies in the background at start-up after the three real tables, or only for the signals the car really has. |
| A1 cause engine (why / what changed) | 26 | `cause/engine.build` runs 15 change-point and trend analyses (a permutation test with 300 shuffles each, plus regressions) for every 'why' or 'what changed' question: 0.7 s on the Pi. Fix: run the permutation test with 100 shuffles first and go to 300 only when the p-value is within 3x of 0.05 (same decision); compute the series once. Expected -0.3 s. |
| A2 health summary (diag_range, eff_fit) | 11 | `qa_c.findings` -> `diag_range` (Welch test at every split of 14 days, `eff_fit` regression over 2,800 trips, tyre batching, six aggregate queries): 0.8 s on the Pi. Done: array regression, C-speed `sd`. Next: one shared per-day table per question and a vectorised Welch scan over all splits. Expected -0.3 s. |
| A4 seasonal compare (four years in Python) | 6 | 'Compare hot and cold days / summer and winter' loads the minute frame for four years and filters it in Python (3.7 s on the Pi; the loader also cuts every load at 1.8 M rows, which drops the newest 14 %). Fix: aggregate in the column copy with the condition masks (one pass per mask, about 40 ms) and return per-day means directly. Some printed numbers change (all rows, not 1.8 M), so it needs the owner's decision. |
| A3 range / trend analysis | 3 | Daily series of range or consumption from trips plus Welch/OLS in Python. Fix: the shared per-day table and vectorised scan of A2. |

First cause is model time for 244 questions and analysis or data for 68.

## Fixes made to these questions (2026-10-28, no training)
Measured by running the 309 slowest questions again on the Pi, once with the new code and once as a control with the new paths switched off (`VGPT_SHARE=0 VGPT_GATE_ABORT=0`), because slow questions look faster on any second run by chance alone.
| step | what | exact? | result |
|---|---|---|---|
| Two readings (174 of the 312) | A question the parser is unsure about is read a second time with the previous turn's frame. The second reading now starts from the first one's work (same tokens up to the end of the question and its hint, so same keys and values) and is guessed from the first reading's words; the first reading stops as soon as the gate has decided (a step with probability under 0.5, or act and concept that already say 'incomplete'). | yes: 268 of 268 follow-up audit questions give the identical frame, largest pmin difference 0.0000 (`bench/share_check.py`) | about 0.25 to 0.3 s per such question |
| Hints | kWh, kW and Wh/km added to the terms that get no hint (audit of the 44 questions affected: lost 0, won 3, `bench/hint_delta.py`). 'efficiency' stays: dropping it lost 2 (direction of 'got worse'). | audit | fewer prompt tokens on word-problem questions |
| Lake signals | Their column copies are built at start-up after the three real tables. | yes | first lake question after a start no longer pays 2 s |
| Result on the 309 | questions over their limit: **264 (control) -> 179** (85 more under the limit); median model time 2.16 -> 1.88 s; slowest 5.7 s | | |
The remaining 179 are the long prompt tails of word-problem and multi-term questions (hint line is 56 % of the tail for them), the second reading's 'Previous:' line (about 20 tokens that cannot be shortened without retraining), and the analysis families A1-A4 above.

## Measured and rejected (2026-10-29, no training)
- **Skip the Previous line** (`bench/gate_study.py`, Pi): of 268 audit questions with a previous frame, 89 point back by their wording; for 69 of them the bare reading is wrong, even when it looks complete and sure (35 of 54). The gate is right and stays. Only 9 of the 268 are read twice; the other slow 'two reading' cases of the full pass are mostly short questions that always get the line.
- **Trim the trailing `none` words of the Previous line** (`bench/prev_trim.py`, Pi): 28 of 268 frames change and 23 correct answers are lost (251 -> 229): the last slots tell the model which earlier item to point at. Rejected.
- **KV-copy cost** (`kvcost`, Pi): a decode pass is about 100 ms for one row and 181 ms for five (20 ms a row), prefill about 30 ms a token. Passes are limited by reading the weights, not by copying the cache, so the cache is not worth restructuring.
- **Copy refresh**: the column copy re-read its last 30 rows 17 times in one health question (about 0.1 s on the Pi); it now refreshes at most every 0.5 s per table (`REFRESH_MIN_S`).
- The remaining model-bound questions have a 40+ token tail with 3 or more passes; no further change that keeps every frame identical was found. Fewer tokens needs a hint the model was trained on (training); a shorter template or fewer passes needs training or a faster CPU.

## Round 3 (2026-10-29): only the over-limit questions were re-run on the Pi
Method: `bench/full_pass.py --only-slow-in <list>` on the Pi, never all 4,096. **Correction:** three intermediate runs (pi15-pi17) were invalid: a misplaced `@dataclass` made 52 answers errors that looked fast, so their "151" and "144" are wrong. The honest runs count error answers (0 now). 174 questions that were over the limit: **174 -> 152 still over** after these exact changes (each has a test against the old code):
- frame loader uses the column copy's columns directly, skips the timestamp search for tables on the same minutes and otherwise searches from the smaller side (`query_executor._Cols`, `_match_idx`, `tests/test_time_alignment.py`)
- `change_ops._split`: 400 permutation tests as one batch, bit-identical (`tests/test_change_split_batch.py`)
- `sensor_quality.stuck` / `spikes`: Python loop over 1.8 M samples -> vectorised (`tests/test_sensor_quality_vector.py`); `temporal.runs` vectorised; `Frame.take` scans the mask once; day grouping of sorted days without `np.unique`
- Pi numbers (after the model): cold-vs-hot comparison 3.76 -> 1.54 s, highest power ever 1.38 -> 0.83, braking trend 1.17 -> 0.69, charging hotter in summer 1.42 -> 1.14.
Reading the `rest` column: it is total minus the LAST model reading, so compound questions ("the soc and also the soh", "Thanks ... Quick one ...") that are parsed twice show about 0.9 s of "rest" that is really a second model reading. Per-thread sampling on the Pi (`sample_pi.py` idea: sample the worker thread's stack every 4 ms) gave the true analysis time of the health question: 0.32 s. Python 3.13's cProfile records all threads, so the live simulator's sleeping appears inside a question: do not use it there.
Still over 0.3 s of real analysis: the seasonal comparisons (about 1.1-1.5 s: `_value_over`, loading and masking 1.8 M rows), `assess_frame` on large frames, the cause engine (0.7 s). Next step for those: compute the per-condition statistics from the column copy without building the full frame.

## Context completion (2026-10-29): previous-turn information without prompt tokens
Why: 16 % of the questions in the 4,096-question pass carry the previous frame in the prompt (about 12 tokens, 0.4-0.7 s on the Pi) and 172 of the 309 over-limit questions are among them. A 3-5 token form is not possible without training (cutting the trailing `none` words alone lost 23 of 268 correct frames).
What: `vgpt/meaning/context.py`. For an **ellipsis** ("and the motor", "same again for last week": an opener word and no pointer word, decided from the wording before the model is used) the question is read WITHOUT the previous line and the slots the previous frame supplies are copied: new period -> the whole previous frame, `ref=period`; new quantity -> the previous act/op/cond/vs/ent with the new concept, `ref=concept`. Pointer questions ("why did that happen", "which one", "how did you get that") keep today's reading with the previous line: rules for focus/item/answer were wrong too often when tried (12 of 17).
Measured on the Pi, 268 follow-ups of the audit sets (`bench/context_check.py`): strict 249 today, 249 now (one gained, one lost); 25 completed from the previous frame, 519 ms saved each; 9 ellipsis questions the rules cannot finish read the question bare and then with the previous line (which reuses the bare reading's work), 534 ms more each; all 268 together 302 s -> 286 s. Unit tests `tests/test_context_completion.py`. `VGPT_CONTEXT=0` switches it off.
Limits: it helps ellipsis questions only (about a quarter of the follow-ups); pointer questions still pay the previous line. The unfinished rule gaps are listed by `bench/context_collect.py` + the offline evaluator. A real fix for the rest (a model that reads the context in 3-5 tokens) needs training.

## Other defects found on the way

- `legacy adapter 'is the vehicle healthy overall' failed: TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` on a car with a few minutes of history (linked car on the Pi): the health adapter adds a value that is None while a signal has no data yet. To fix with a None-safe sum, then test with a fresh car.
- The Pi's `/bin/sh` (dash) refused file descriptors above 9, so the Chromium shell could not start there. Fixed: `vgpt/shell/pipe_shell.py` places the pipe ends in Python.
- The start-up self-check refused a linked car because there is no `data/*.db` (by design there is none). Fixed with `VGPT_LINKED`.

## Every question over its limit

Seconds: total / model / after the model. w = words, tail = prompt tail tokens, passes = decode passes. Sorted by total time.

| s | model | rest | w | tail | passes | causes | question |
|---|---|---|---|---|---|---|---|
| 8.06 | 7.98 | 0.08 | 24 | 87 | 5 | M1 | The charging log reports 30 kWh delivered, while SoC rose from 20% to 60% on a 70 kWh usable pack. Are these v |
| 5.56 | 5.51 | 0.05 | 29 | 73 | 2 | M1 | Hello! Two things: first, the mileage on the clock, and second, thank you very much for being so helpful with  |
| 5.44 | 5.39 | 0.04 | 16 | 77 | 2 | M1 | A trip has 0 km distance but 2 kWh energy use. What should the calculation do? |
| 5.34 | 1.60 | 3.73 | 10 | 30 | 3 | A4 | compare the motor temperature on cold days and hot days |
| 5.03 | 4.97 | 0.06 | 31 | 64 | 2 | M1 | Hello! Two things: first, how many kilometres does the odometer show, and second, thank you very much for bein |
| 4.84 | 4.79 | 0.05 | 8 | 70 | 2 | M1 | convert my consumption to kwh per 100 km |
| 4.82 | 2.31 | 2.51 | 30 | 58 | 2 | M1; A5 | Thanks for helping me out with the car so far. Quick one before I forget: city vs motorway consumption. Please |
| 4.10 | 4.07 | 0.02 | 10 | 45 | 3 | M1 | Why should timestamps be synchronized across camera and vehicle logs? |
| 4.05 | 4.02 | 0.04 | 17 | 57 | 3 | M1 | A user asks, 'Was my battery hot last Friday?' but the logs use UTC. What should happen? |
| 3.97 | 3.92 | 0.05 | 7 | 51 | 2 | M1 | does the 12 volt show anything strange |
| 3.97 | 3.90 | 0.07 | 20 | 46 | 4 | M1 | The system says 0 km/h, but GPS points show 300 metres of movement over 10 minutes. What checks come first? |
| 3.81 | 3.78 | 0.03 | 22 | 79 | 5 | M1 | The car used 22 kWh over 110 km in rain, and 18 kWh over 100 km on a dry day. Compare consumption. |
| 3.79 | 3.77 | 0.02 | 9 | 32 | 7 | M2 | Does charging to 100% always damage an EV battery? |
| 3.74 | 3.72 | 0.02 | 21 | 57 | 1 | M1 | A recipe for 4 people uses 300 g of flour. How much is needed for 10 people at the same ratio? |
| 3.74 | 3.72 | 0.02 | 7 | 38 | 3 | M3 | how many cylinders does the engine have |
| 3.71 | 3.69 | 0.02 | 15 | 43 | 3 | M1 | If a 20% discount is followed by a 10% discount, is the total discount 30%? |
| 3.71 | 3.69 | 0.02 | 7 | 38 | 3 | M3 | how many cylinders does the motor have |
| 3.56 | 3.54 | 0.03 | 16 | 43 | 2 | M1 | i want to know how many people were riding in the car on the last trip |
| 3.54 | 3.27 | 0.28 | 20 | 45 | 2 | M1 | A warning timestamp is 12:00, but the database clock may be five minutes fast. How should the timeline be repo |
| 3.52 | 3.50 | 0.02 | 12 | 49 | 2 | M1 | Can you tell me the exact current weather from last month's logs? |
| 3.51 | 1.43 | 2.08 | 9 | 14 | 7 | A5 | do the hottest cell and the pack temperature agree |
| 3.49 | 3.47 | 0.02 | 9 | 26 | 5 | M2 | Will turning off the cabin heater always maximize range? |
| 3.47 | 3.45 | 0.02 | 10 | 27 | 6 | M2 | Does a larger battery automatically make a car more efficient? |
| 3.46 | 3.44 | 0.02 | 6 | 43 | 2 | M1 | what lit the check engine lamp |
| 3.46 | 1.58 | 1.87 | 3 | 37 | 2 | A5 | minimum 12v voltage |
| 3.45 | 3.40 | 0.04 | 18 | 41 | 3 | M1 | A code maps A to 1, B to 2, and so on. What is the sum for CAT? |
| 3.44 | 3.38 | 0.05 | 14 | 43 | 4 | M1 | If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? |
| 3.41 | 1.39 | 2.01 | 10 | 15 | 6 | A5 | do voltage and current in the pack move in step |
| 3.38 | 3.37 | 0.02 | 19 | 74 | 6 | M1 | If I add 12 kWh to a battery while charging at 4 kW, how long does it take ideally? |
| 3.30 | 1.25 | 2.05 | 7 | 12 | 5 | A5 | correlation between the accelerator and battery power |
| 3.29 | 1.67 | 1.62 | 17 | 37 | 3 | A5 | i wonder what the highest power the motor was ever asked to deliver in the last week |
| 3.29 | 3.27 | 0.01 | 3 | 31 | 3 | M3 | pax last trip |
| 3.23 | 3.22 | 0.01 | 8 | 44 | 2 | M1 | explain my consumption to a five year old |
| 3.20 | 3.17 | 0.03 | 12 | 35 | 4 | M2 | A value rises from 50 to 60. What is the percentage increase? |
| 3.19 | 1.17 | 2.02 | 7 | 12 | 5 | A5 | is picture brightness related to the daylight |
| 3.19 | 1.28 | 1.91 | 8 | 28 | 2 | A5 | what was the peak motor power last month |
| 3.17 | 3.14 | 0.03 | 8 | 44 | 2 | M1 | whats the point of the voltage dip event |
| 3.16 | 3.13 | 0.03 | 10 | 41 | 4 | M1 | What comes next: 1, 1, 2, 3, 5, 8, __? |
| 3.14 | 3.12 | 0.02 | 5 | 35 | 2 | M3 | how loud is the engine |
| 3.10 | 3.07 | 0.03 | 5 | 40 | 3 | M1 | is my efficiency getting worse |
| 3.08 | 1.23 | 1.85 | 8 | 13 | 4 | A4 | is charging hotter in summer than in winter |
| 3.08 | 3.06 | 0.02 | 6 | 36 | 2 | M3 | what is the engine oil level |
| 3.06 | 3.00 | 0.06 | 25 | 74 | 3 | M1 | A vehicle drove 120 km at 200 Wh/km. The next trip drove 80 km at 150 Wh/km. What was total energy use and com |
| 3.05 | 3.04 | 0.02 | 4 | 36 | 3 | M3 | is the engine knocking |
| 3.04 | 0.93 | 2.11 | 7 | 12 | 4 | A5 | pressure drop trend on the rear left |
| 3.03 | 3.00 | 0.03 | 10 | 29 | 5 | M2 | did the camera get blocked or just degraded this week |
| 3.02 | 2.47 | 0.55 | 6 | 25 | 5 | M2; A1 | No, I meant the other battery. |
| 3.02 | 2.98 | 0.04 | 21 | 71 | 3 | M1 | Two trips used 12 kWh each. Trip A was 60 km and Trip B was 48 km. Which was more efficient? |
| 3.02 | 2.98 | 0.04 | 7 | 25 | 4 | M2 | Why could lane assistance disengage in rain? |
| 2.99 | 2.98 | 0.01 | 10 | 35 | 3 | M3 | Why is the average of 10 and 20 not 20? |
| 2.99 | 2.97 | 0.02 | 14 | 37 | 2 | M3 | A farmer has 17 sheep and all but 9 run away. How many remain? |
| 2.98 | 2.89 | 0.10 | 23 | 74 | 4 | M1 | A car uses 15 kWh for a 100 km trip, but the charger later draws 17 kWh to replenish it. Explain the differenc |
| 2.98 | 2.96 | 0.02 | 11 | 29 | 3 | M3 | Is a high battery temperature during fast charging always a fault? |
| 2.98 | 1.01 | 1.97 | 10 | 15 | 2 | A5 | for how much of the time was lane keeping active |
| 2.97 | 2.38 | 0.59 | 20 | 63 | 2 | M1; A1 | Energy use rose from 170 to 204 Wh/km. Distance stayed similar, but the weather became colder. What can you co |
| 2.97 | 0.93 | 2.04 | 6 | 11 | 2 | A5 | what range is the dashboard claiming |
| 2.96 | 2.94 | 0.02 | 9 | 27 | 5 | M2 | Was the camera lens dirty during the rainy trip? |
| 2.96 | 2.94 | 0.02 | 3 | 41 | 2 | M1 | avg consumption weekdays |
| 2.96 | 0.89 | 2.07 | 6 | 11 | 3 | A5 | does braking pressure scale with speed |
| 2.95 | 1.11 | 1.84 | 11 | 19 | 2 | A5 | average hvac power when the cabin was hotter than 30 degrees |
| 2.93 | 2.89 | 0.04 | 16 | 40 | 2 | M1 | A shop has 3 red, 4 blue, and 5 green balls. How many balls are there? |
| 2.92 | 2.85 | 0.06 | 14 | 76 | 2 | M1 | A car uses 18 kWh over 100 km. What is its efficiency in Wh/km? |
| 2.91 | 2.89 | 0.02 | 14 | 37 | 2 | M3 | If 40% of a class of 35 students are absent, how many are absent? |
| 2.89 | 2.24 | 0.65 | 41 | 60 | 2 | A3 | range over the last two weeks has it gone down. I know this is a simple question but I am asking because my br |
| 2.88 | 2.24 | 0.64 | 35 | 55 | 2 | M1; A1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.88 | 2.87 | 0.01 | 7 | 29 | 4 | M2 | how often should I charge to 100 |
| 2.86 | 0.95 | 1.91 | 10 | 17 | 2 | A5 | has charging power been dropping over the last 10 days |
| 2.85 | 1.43 | 1.43 | 6 | 29 | 2 | A4 | hot weather vs cold weather efficiency |
| 2.85 | 2.83 | 0.02 | 7 | 39 | 2 | M3 | what is the motor temperature under load |
| 2.85 | 2.81 | 0.04 | 6 | 23 | 5 | M2 | was every charge a fast charge |
| 2.84 | 2.22 | 0.61 | 13 | 54 | 2 | M1; A1 | If efficiency improves from 220 to 190 Wh/km, what is the percentage reduction? |
| 2.82 | 2.23 | 0.59 | 37 | 57 | 3 | A1 | how come the charging is slow. I know this is a simple question but I am asking because my brother was telling |
| 2.82 | 2.61 | 0.21 | 9 | 27 | 5 | M2 | what was the pack doing when the flow dropped |
| 2.82 | 2.72 | 0.10 | 41 | 61 | 4 | M1 | typical electricity use per kilometre while driving in the city. I know this is a simple question but I am ask |
| 2.81 | 2.77 | 0.04 | 14 | 57 | 4 | M1 | The estimated range dropped 15%, while energy consumption rose 15%. Is that mathematically inconsistent? |
| 2.80 | 2.77 | 0.03 | 12 | 37 | 2 | M3 | A 1.5-litre bottle is filled to 60%. How much liquid is inside? |
| 2.80 | 2.78 | 0.02 | 9 | 28 | 3 | M3 | Forget the last question; what is a prime number? |
| 2.79 | 2.72 | 0.07 | 16 | 75 | 2 | M1 | The car has 50 kWh remaining and uses 200 Wh/km. Estimate range, then name two limitations. |
| 2.77 | 1.44 | 1.34 | 6 | 29 | 2 | A4 | compare hot and cold weather consumption |
| 2.76 | 2.74 | 0.03 | 7 | 26 | 3 | M3 | Does regenerative braking recover all braking energy? |
| 2.76 | 2.75 | 0.02 | 8 | 38 | 2 | M3 | how many people were in the car yesterday |
| 2.75 | 2.71 | 0.04 | 8 | 25 | 3 | M3 | How much did cold weather affect my range? |
| 2.75 | 2.71 | 0.05 | 10 | 29 | 4 | M2 | Does a camera-quality warning prove the camera hardware is broken? |
| 2.74 | 2.69 | 0.05 | 15 | 35 | 4 | M2 | The user asks for 'last week', and today is Monday. What period should be used? |
| 2.74 | 2.72 | 0.02 | 14 | 31 | 3 | M3 | Can you promise a used EV battery is healthy based on its range estimate? |
| 2.73 | 0.80 | 1.94 | 7 | 12 | 2 | A5 | peak braking intensity in the last month |
| 2.72 | 1.52 | 1.20 | 7 | 26 | 4 | A5 | did charging ever go over 100 kw |
| 2.71 | 2.69 | 0.03 | 4 | 22 | 5 | M2 | Was I speeding yesterday? |
| 2.71 | 2.69 | 0.02 | 8 | 25 | 5 | M2 | why was yesterday so tough on the battery |
| 2.71 | 0.72 | 1.98 | 5 | 10 | 2 | A5 | peak lateral acceleration in turns |
| 2.70 | 2.62 | 0.07 | 7 | 26 | 3 | M3 | How can a dirty windscreen affect ADAS? |
| 2.69 | 2.63 | 0.06 | 4 | 23 | 6 | M2 | Repeat the last value. |
| 2.67 | 2.60 | 0.08 | 30 | 58 | 3 | M1 | Hello! Two things: first, hot weather vs cold weather efficiency, and second, thank you very much for being so |
| 2.66 | 0.65 | 2.01 | 3 | 8 | 2 | A5 | cell voltage spread |
| 2.65 | 2.11 | 0.55 | 7 | 26 | 3 | M3; A1 | Do the same calculation for 90 km. |
| 2.65 | 2.61 | 0.04 | 20 | 69 | 2 | M1 | Two weeks have 500 km at 180 Wh/km and 700 km at 210 Wh/km. What is the combined average efficiency? |
| 2.65 | 2.63 | 0.02 | 12 | 30 | 3 | M3 | Can the assistant use a webcam if the vehicle camera is unavailable? |
| 2.64 | 0.79 | 1.85 | 5 | 10 | 2 | A5 | average speed in the rain |
| 2.64 | 2.00 | 0.64 | 36 | 47 | 2 | M1; A1 | Hello! Two things: first, why has my driving range gone down over the last two weeks, and second, thank you ve |
| 2.64 | 2.61 | 0.03 | 14 | 34 | 2 | M3 | A rectangle is 8 cm long and 5 cm wide. What is its area? |
| 2.63 | 2.57 | 0.07 | 26 | 57 | 4 | M1 | A warning occurred at 14:10. Coolant flow fell at 14:03 and battery temperature rose from 31Â°C to 48Â°C by 14 |
| 2.63 | 1.69 | 0.95 | 31 | 41 | 2 | A5 | Thanks for helping me out with the car so far. Quick one before I forget: the strongest braking this week. Ple |
| 2.62 | 0.62 | 2.00 | 3 | 8 | 2 | A5 | steering angle distribution |
| 2.61 | 2.55 | 0.07 | 35 | 66 | 3 | M1 | battery kitni charge hai. I know this is a simple question but I am asking because my brother was telling me y |
| 2.61 | 2.59 | 0.02 | 12 | 29 | 3 | M3 | Why can the wall meter show more energy than the battery gained? |
| 2.61 | 2.58 | 0.03 | 12 | 31 | 3 | M3 | The car says 'stop safely'. Should I continue to the next exit? |
| 2.59 | 2.57 | 0.02 | 7 | 24 | 3 | M3 | did the cold weather cut my range |
| 2.59 | 0.73 | 1.85 | 8 | 14 | 1 | A5 | average battery voltage in the last 3 days |
| 2.59 | 2.56 | 0.02 | 11 | 33 | 3 | M3 | Is the statement 'This sentence is false' a simple true-or-false statement? |
| 2.57 | 2.50 | 0.07 | 39 | 59 | 3 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.57 | 2.54 | 0.03 | 11 | 30 | 4 | M2 | A circle has radius 3 cm. What is its area approximately? |
| 2.56 | 2.50 | 0.06 | 20 | 65 | 2 | M1 | A trip's average efficiency is 180 Wh/km, but a short segment shows 600 Wh/km. Is the whole trip necessarily i |
| 2.56 | 2.51 | 0.05 | 18 | 66 | 2 | M1 | Efficiency is 160 Wh/km over 50 km and 220 Wh/km over 150 km. What is the combined efficiency? |
| 2.56 | 2.52 | 0.04 | 14 | 67 | 2 | M1 | A vehicle uses 16 kWh for 80 km. What is consumption per 100 km? |
| 2.56 | 2.52 | 0.03 | 10 | 43 | 6 | M1 | 12 volt warnings or tyre warnings in the last week |
| 2.55 | 0.72 | 1.83 | 8 | 14 | 1 | A5 | mean tyre temperature during the past 3 days |
| 2.55 | 2.36 | 0.19 | 6 | 24 | 5 | M2 | what was going on just before |
| 2.54 | 2.46 | 0.08 | 34 | 66 | 2 | M1 | current motor rpm. I know this is a simple question but I am asking because my brother was telling me yesterda |
| 2.54 | 2.52 | 0.02 | 8 | 26 | 3 | M3 | What is the difference between RAM and storage? |
| 2.54 | 2.52 | 0.02 | 8 | 25 | 3 | M3 | how many usb ports does the car have |
| 2.54 | 2.51 | 0.03 | 14 | 34 | 2 | M3 | A warning occurred after a hard acceleration event. Is the acceleration necessarily the cause? |
| 2.54 | 0.66 | 1.88 | 5 | 11 | 1 | A5 | mean yaw rate during cornering |
| 2.54 | 2.48 | 0.06 | 12 | 62 | 2 | M1 | What does 90% charging efficiency mean for 10 kWh from the wall? |
| 2.53 | 2.48 | 0.05 | 5 | 23 | 3 | M3 | show my 4 shortest trips |
| 2.53 | 0.72 | 1.81 | 4 | 9 | 2 | A5 | max rpm last week |
| 2.52 | 1.85 | 0.67 | 31 | 42 | 2 | M1; A3 | Hello! Two things: first, when did the range fall the most, and second, thank you very much for being so helpf |
| 2.52 | 2.38 | 0.15 | 20 | 59 | 2 | M1 | A 12 V battery reads 11.6 V once, but no other data exists. Can the assistant diagnose a failed battery? |
| 2.52 | 1.90 | 0.62 | 10 | 45 | 2 | M1; A1 | why does the consumption go up when it is cold |
| 2.52 | 2.43 | 0.09 | 16 | 59 | 2 | M1 | A 60 kWh usable battery is at 40% state of charge. How much usable energy remains? |
| 2.50 | 2.45 | 0.05 | 9 | 33 | 2 | M3 | What is the median of 3, 5, 8, 10? |
| 2.49 | 2.46 | 0.03 | 11 | 32 | 2 | M3 | Can I open the bonnet immediately after a severe overheating warning? |
| 2.49 | 2.45 | 0.04 | 6 | 24 | 3 | M3 | was my driving efficient this week |
| 2.49 | 1.77 | 0.72 | 15 | 38 | 3 | A5 | how does my consumption on the highway compare with the city over the last year |
| 2.49 | 2.44 | 0.05 | 14 | 65 | 2 | M1 | If the car travels 240 km using 48 kWh, what is its average consumption? |
| 2.47 | 2.33 | 0.14 | 11 | 52 | 4 | M1 | How long would 24 kWh take at a steady 6 kW? |
| 2.46 | 2.43 | 0.03 | 9 | 59 | 2 | M1 | was there a brownout on the 12 volt supply |
| 2.46 | 2.44 | 0.02 | 10 | 29 | 3 | M3 | What is the difference between object detection and object tracking? |
| 2.45 | 1.40 | 1.05 | 23 | 28 | 3 | A2 | i am planning a long drive this weekend and just want to be sure there are no warnings or problems at the mome |
| 2.45 | 2.35 | 0.10 | 35 | 58 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.44 | 2.40 | 0.04 | 10 | 54 | 3 | M1 | did the 12 volt battery warn me it was low |
| 2.44 | 2.40 | 0.04 | 14 | 56 | 3 | M1 | after the long drive to the coast yesterday how many kwh did it use |
| 2.43 | 2.41 | 0.02 | 10 | 29 | 3 | M3 | Can I use water to extinguish an EV battery fire? |
| 2.43 | 2.28 | 0.15 | 8 | 25 | 3 | M3 | would the battery survive a month without charging |
| 2.42 | 2.39 | 0.03 | 13 | 64 | 2 | M1 | If a car gets 250 km from 50 kWh, what is its efficiency? |
| 2.42 | 2.30 | 0.12 | 3 | 20 | 4 | M2 | the usual please |
| 2.42 | 2.24 | 0.18 | 9 | 45 | 3 | M1 | did my efficiency get better after the software update |
| 2.42 | 2.39 | 0.03 | 9 | 44 | 4 | M1 | when did the 12 volt last drop below 12 |
| 2.41 | 2.27 | 0.15 | 35 | 55 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.41 | 2.31 | 0.10 | 32 | 56 | 3 | M1 | Hi there, I hope you are doing well today. I was just wondering, and please take your time with this, could yo |
| 2.41 | 0.89 | 1.52 | 7 | 12 | 3 | A5 | has the braking got harder over time |
| 2.41 | 2.38 | 0.03 | 10 | 28 | 3 | M3 | What is the difference between an EV and a hybrid? |
| 2.41 | 1.75 | 0.66 | 7 | 30 | 3 | A1 | Did tyre pressure explain my lower efficiency? |
| 2.41 | 2.37 | 0.04 | 6 | 26 | 5 | M2 | Yesterday's data, but in local time. |
| 2.40 | 2.37 | 0.03 | 11 | 52 | 3 | M1 | how often did a voltage dip or a power derate happen |
| 2.40 | 2.38 | 0.01 | 9 | 33 | 2 | M3 | I typed 'peflol'â€”can you figure out what I meant? |
| 2.39 | 2.32 | 0.07 | 36 | 56 | 4 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.38 | 2.09 | 0.28 | 29 | 40 | 3 | M1 | Hello! Two things: first, how is the cooling loop, and second, thank you very much for being so helpful with a |
| 2.38 | 1.74 | 0.64 | 6 | 26 | 4 | A5 | what do the fuel trims say |
| 2.38 | 1.79 | 0.58 | 7 | 40 | 2 | A1 | why is the 12 v battery weak |
| 2.37 | 1.04 | 1.32 | 8 | 13 | 3 | A4 | was i more efficient in summer or winter |
| 2.36 | 2.28 | 0.08 | 25 | 43 | 5 | M1 | last time i charged it was at a fast charger and i think it got very hot, how hot did the pack get during char |
| 2.36 | 2.35 | 0.01 | 13 | 33 | 1 | M3 | A number doubled and increased by 6 equals 20. What is the number? |
| 2.35 | 1.79 | 0.57 | 13 | 36 | 3 | A1 | An answer reports 220 Wh/km but omits the time range. What is missing? |
| 2.35 | 2.11 | 0.24 | 32 | 53 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.35 | 2.28 | 0.07 | 36 | 56 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.35 | 2.32 | 0.03 | 9 | 27 | 3 | M3 | How should the assistant describe an uncertain camera event? |
| 2.35 | 2.33 | 0.02 | 6 | 23 | 4 | M2 | what is eating the battery overnight |
| 2.35 | 2.33 | 0.02 | 7 | 24 | 2 | M3 | Why can a roof box reduce range? |
| 2.35 | 1.72 | 0.63 | 27 | 38 | 2 | A3 | Hello! Two things: first, is range droping, and second, thank you very much for being so helpful with all of m |
| 2.34 | 2.28 | 0.06 | 14 | 60 | 2 | M1 | A battery has 45 kWh usable and the car consumes 180 Wh/km. Estimate range. |
| 2.34 | 2.29 | 0.05 | 8 | 49 | 5 | M1 | what is the soc and also the soh |
| 2.33 | 2.27 | 0.06 | 13 | 59 | 2 | M1 | A car has 36 kWh available and uses 240 Wh/km. Estimate its range. |
| 2.33 | 0.87 | 1.46 | 7 | 12 | 3 | A2 | is the battery getting worse with age |
| 2.33 | 1.62 | 0.70 | 8 | 28 | 3 | A5 | is the motor temperature typical at the moment |
| 2.33 | 2.31 | 0.01 | 6 | 23 | 3 | M3 | has the car been running hot |
| 2.32 | 1.11 | 1.21 | 9 | 14 | 3 | A4 | what is my range in winter compared to summer |
| 2.32 | 2.23 | 0.09 | 30 | 54 | 2 | M1 | Hello! Two things: first, what does state of health mean, and second, thank you very much for being so helpful |
| 2.32 | 2.24 | 0.08 | 36 | 55 | 2 | M1 | so yesterday i was driving back from the office and it was raining quite hard and i noticed the range dropping |
| 2.31 | 2.30 | 0.02 | 11 | 29 | 2 | M3 | how has washer fluid level changed over the last 14 days |
| 2.30 | 2.21 | 0.08 | 21 | 40 | 4 | M1 | given that i drive mostly in the city can you say how much energy i use per kilometre in the city |
| 2.29 | 2.26 | 0.04 | 7 | 25 | 2 | M3 | what is the car's heartbeat of power |
| 2.29 | 2.27 | 0.02 | 6 | 23 | 2 | M3 | which gear is the car in |
| 2.29 | 2.24 | 0.06 | 7 | 27 | 3 | M3 | heater power versus motor power on average |
| 2.29 | 2.08 | 0.21 | 32 | 43 | 4 | M1 | Hello! Two things: first, is the battery getting warmer over the weeks, and second, thank you very much for be |
| 2.29 | 2.25 | 0.04 | 5 | 25 | 4 | M2 | What is a black-box model? |
| 2.29 | 2.27 | 0.01 | 7 | 25 | 2 | M3 | How many sides does a hexagon have? |
| 2.28 | 2.26 | 0.02 | 8 | 51 | 3 | M1 | at 80 percent how many kwh is that |
| 2.28 | 2.23 | 0.05 | 15 | 38 | 4 | M2 | please tell me the average efficiency of the last trip only not the whole week |
| 2.27 | 2.20 | 0.07 | 35 | 53 | 3 | M1 | recommended distance between services. I know this is a simple question but I am asking because my brother was |
| 2.27 | 2.22 | 0.06 | 7 | 25 | 3 | M3 | when was the performance last cut back |
| 2.27 | 2.25 | 0.02 | 7 | 29 | 2 | M3 | What is 7 squared minus 3 cubed? |
| 2.26 | 2.20 | 0.06 | 8 | 41 | 4 | M1 | how fast is the engine turning when idling |
| 2.26 | 1.67 | 0.59 | 14 | 37 | 2 | A1 | tell me why the motor temperature has crept up in the last 10 days |
| 2.26 | 2.22 | 0.04 | 13 | 54 | 2 | M1 | My efficiency rose from 160 to 200 Wh/km. What is the percentage increase? |
| 2.26 | 2.21 | 0.05 | 17 | 45 | 4 | M1 | A temperature sensor reports 25Â°C, then 90Â°C one second later, then 26Â°C. What should the assistant say? |
| 2.26 | 2.22 | 0.04 | 6 | 23 | 4 | M2 | What is a vehicle telemetry signal? |
| 2.25 | 2.24 | 0.02 | 6 | 23 | 2 | M3 | what is the coolant reservoir level |
| 2.25 | 2.19 | 0.06 | 5 | 22 | 3 | M3 | typical battery temperature while driving |
| 2.25 | 2.22 | 0.03 | 6 | 24 | 5 | M2 | this time over the last fortnight |
| 2.25 | 1.69 | 0.56 | 13 | 29 | 3 | A1 | Why can two chargers with the same advertised kW charge at different speeds? |
| 2.25 | 2.16 | 0.09 | 5 | 41 | 4 | M1 | fuel economy city versus highway |
| 2.25 | 2.22 | 0.02 | 10 | 29 | 2 | M3 | What is the difference between a virus and a bacterium? |
| 2.24 | 2.15 | 0.10 | 7 | 51 | 2 | M1 | the 12 volt battery is it fine |
| 2.24 | 1.14 | 1.11 | 5 | 22 | 3 | A2 | gaadi ka kya haal hai? |
| 2.24 | 2.20 | 0.04 | 5 | 49 | 3 | M1 | and the 12 volt battery |
| 2.24 | 2.17 | 0.07 | 32 | 52 | 3 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.24 | 1.20 | 1.04 | 5 | 10 | 4 | A2 | any faults on the car |
| 2.24 | 2.22 | 0.01 | 5 | 24 | 2 | M3 | what is the oil pressure |
| 2.22 | 0.92 | 1.30 | 7 | 12 | 3 | A2 | is the battery getting weaker over time |
| 2.22 | 2.20 | 0.02 | 10 | 29 | 2 | M3 | What is the difference between a meteor and a meteorite? |
| 2.22 | 2.16 | 0.05 | 24 | 55 | 2 | M1 | A car travels 60 km at 30 km/h and then 60 km at 60 km/h. What is its average speed for the whole trip? |
| 2.21 | 2.19 | 0.03 | 9 | 48 | 3 | M1 | how many ECU resets or unexpected shutdowns this month |
| 2.21 | 1.67 | 0.54 | 5 | 23 | 4 | A1 | What if it was raining? |
| 2.21 | 2.18 | 0.04 | 10 | 28 | 2 | M3 | What is the boiling point of water at sea level? |
| 2.19 | 2.14 | 0.05 | 10 | 56 | 2 | M1 | At 150 Wh/km, what distance could 30 kWh theoretically cover? |
| 2.19 | 2.12 | 0.07 | 22 | 56 | 2 | M1 | A tyre warning is timestamped 08:00, but the pressure sample is from 07:20. Can it be reported as the pressure |
| 2.19 | 1.46 | 0.72 | 9 | 31 | 2 | A5 | is my battery temperature higher than it usually is |
| 2.19 | 2.12 | 0.06 | 34 | 54 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.19 | 2.16 | 0.03 | 15 | 43 | 3 | M1 | Rain is recorded at 09:05 and a warning occurred at 09:00. Did rain cause it? |
| 2.18 | 2.16 | 0.03 | 2 | 48 | 3 | M1 | 12v ok |
| 2.18 | 2.12 | 0.06 | 35 | 53 | 3 | M1 | which model is this. I know this is a simple question but I am asking because my brother was telling me yester |
| 2.18 | 1.55 | 0.62 | 8 | 32 | 2 | A1 | why did my consumption go up in 2025 |
| 2.17 | 2.11 | 0.07 | 13 | 47 | 3 | M1 | A charger delivers 18 kWh in 30 minutes. What is its average power? |
| 2.17 | 1.97 | 0.20 | 8 | 24 | 2 | M3 | Can braking events prove which person was driving? |
| 2.17 | 2.13 | 0.03 | 13 | 47 | 3 | M1 | A 75 kWh battery is at 20%. Approximately how much energy is stored? |
| 2.16 | 2.13 | 0.03 | 10 | 31 | 6 | M2 | which ran hotter this week, the pack or the motor |
| 2.16 | 2.07 | 0.09 | 33 | 53 | 2 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.16 | 2.14 | 0.01 | 10 | 28 | 2 | M3 | Can a model infer a person's identity from GPS history? |
| 2.15 | 0.90 | 1.26 | 10 | 15 | 2 | A2 | is the battery health declining over the last three years |
| 2.15 | 1.52 | 0.64 | 12 | 36 | 2 | A1 | what is behind the rise in my energy use per kilometre lately |
| 2.14 | 2.11 | 0.03 | 6 | 23 | 2 | M3 | what is the oil pressure at |
| 2.14 | 2.12 | 0.02 | 9 | 27 | 2 | M3 | what is the pressure of the windscreen washer fluid |
| 2.14 | 1.14 | 0.99 | 3 | 22 | 3 | A2 | How's it doing? |
| 2.13 | 2.03 | 0.10 | 21 | 39 | 4 | M2 | A vehicle reports 280 km range at 80% charge and 210 km at 60%. Is the range estimate proportional to charge? |
| 2.13 | 1.83 | 0.31 | 21 | 42 | 2 | M1; A5 | The latest tyre pressure is 240 kPa, the target is 250 kPa, and the earlier reading was 260 kPa. What changed? |
| 2.13 | 2.12 | 0.01 | 7 | 24 | 2 | M3 | does the car record the cabin humidity |
| 2.13 | 1.72 | 0.41 | 9 | 29 | 6 | A5 | how strongly are battery and motor temperature tied together |
| 2.13 | 2.06 | 0.07 | 11 | 31 | 5 | M2 | why did the motor get hot on my most recent trip |
| 2.13 | 2.10 | 0.03 | 4 | 21 | 5 | M2 | only for this morning |
| 2.13 | 2.08 | 0.04 | 8 | 42 | 3 | M1 | any oddities in the 12 volt battery readings |
| 2.13 | 2.08 | 0.04 | 8 | 41 | 3 | M1 | why did the 12 volt battery warning appear |
| 2.12 | 2.10 | 0.02 | 8 | 54 | 2 | M1 | Convert 210 Wh/km to kWh per 100 km. |
| 2.12 | 2.09 | 0.03 | 4 | 22 | 3 | M3 | lsit teh volatge dpis |
| 2.12 | 2.09 | 0.03 | 10 | 43 | 4 | M1 | how much fuel does the engine normally burn per hour |
| 2.12 | 1.47 | 0.65 | 10 | 16 | 5 | A1 | why did the pack get so warm after the motorway |
| 2.12 | 2.05 | 0.07 | 34 | 53 | 2 | M1 | meaning of adas. I know this is a simple question but I am asking because my brother was telling me yesterday  |
| 2.12 | 2.10 | 0.02 | 3 | 21 | 2 | M3 | wiper fluid left |
| 2.12 | 1.47 | 0.64 | 7 | 30 | 2 | A1 | Why compare energy use over similar routes? |
| 2.11 | 1.85 | 0.26 | 4 | 22 | 6 | M2 | why did that occur |
| 2.11 | 2.08 | 0.03 | 8 | 26 | 2 | M3 | how many dips in voltage did we get |
| 2.11 | 1.45 | 0.66 | 6 | 26 | 3 | A1 | why is the engine warning on |
| 2.10 | 1.54 | 0.56 | 12 | 41 | 1 | A1 | Why is charging cost not always just kWh multiplied by one price? |
| 2.10 | 1.84 | 0.25 | 5 | 23 | 6 | M2 | and the first of them |
| 2.09 | 0.83 | 1.26 | 7 | 12 | 3 | A2 | the car feels slow what is wrong |
| 2.09 | 2.05 | 0.04 | 34 | 53 | 2 | M1 | cabin co2 level. I know this is a simple question but I am asking because my brother was telling me yesterday  |
| 2.09 | 1.90 | 0.19 | 8 | 44 | 3 | M1 | is fuel economy getting worse over the months |
| 2.09 | 2.04 | 0.05 | 5 | 22 | 4 | M2 | how much jus is left |
| 2.09 | 2.00 | 0.08 | 33 | 55 | 1 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.09 | 2.07 | 0.02 | 6 | 24 | 3 | M3 | What is a GPS fix timestamp? |
| 2.08 | 2.05 | 0.03 | 2 | 36 | 4 | M2 | engine revs |
| 2.08 | 1.06 | 1.03 | 8 | 14 | 3 | A2 | Can you tell me if this is serious? |
| 2.07 | 1.41 | 0.66 | 8 | 32 | 2 | A1 | why was the consumption so high on tuesday |
| 2.07 | 2.01 | 0.06 | 35 | 53 | 1 | M1 | whats d coolant flow. I know this is a simple question but I am asking because my brother was telling me yeste |
| 2.07 | 2.02 | 0.04 | 10 | 27 | 6 | M2 | other than the last trip which one was the longest |
| 2.07 | 1.87 | 0.20 | 6 | 42 | 3 | M1 | is the fuel consumption going up |
| 2.06 | 2.03 | 0.03 | 36 | 57 | 1 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.06 | 2.02 | 0.04 | 8 | 39 | 3 | M3 | how many kilowatts is the motor delivering now |
| 2.06 | 2.01 | 0.05 | 19 | 50 | 2 | M1 | the service centre said the aux battery was weak last time so how is the twelve volt battery now |
| 2.06 | 1.89 | 0.17 | 6 | 42 | 3 | M1 | is my diesel consumption getting worse |
| 2.06 | 2.02 | 0.04 | 12 | 42 | 3 | M1 | first the odometer then forget it, how many hard brakes this week |
| 2.06 | 1.83 | 0.23 | 27 | 47 | 2 | M1 | my wife says the car feels slower than before and i want to know if the motor power has been going down over t |
| 2.06 | 2.03 | 0.03 | 6 | 27 | 2 | M3 | what is the octane rating needed |
| 2.06 | 2.02 | 0.03 | 10 | 39 | 3 | M3 | tell me how many kW the motor is putting out |
| 2.05 | 1.42 | 0.64 | 5 | 28 | 2 | A1 | what made the efficiency drop |
| 2.05 | 2.00 | 0.05 | 30 | 40 | 4 | M1 | Hi there, I hope you are doing well today. I was just wondering, and please take your time with this, could yo |
| 2.05 | 2.02 | 0.04 | 5 | 26 | 1 | M3 | what is the adblue level |
| 2.05 | 1.99 | 0.06 | 28 | 52 | 2 | M1 | Hi there, I hope you are doing well today. I was just wondering, and please take your time with this, could yo |
| 2.05 | 2.04 | 0.01 | 6 | 44 | 3 | M1 | the engine light, is it on |
| 2.05 | 1.88 | 0.17 | 8 | 44 | 2 | M1 | when did my fuel efficiency drop the most |
| 2.05 | 2.02 | 0.03 | 8 | 41 | 4 | M1 | how long was the engine idling this week |
| 2.05 | 1.79 | 0.25 | 3 | 21 | 6 | M3 | the latest one |
| 2.05 | 1.99 | 0.06 | 34 | 44 | 3 | M1 | Thanks for helping me out with the car so far. Quick one before I forget: what is the software build on the ca |
| 2.05 | 2.04 | 0.01 | 11 | 29 | 1 | M3 | What should the assistant do if a sensor value is missing? |
| 2.04 | 2.03 | 0.02 | 5 | 23 | 3 | M3 | What is a prime number? |
| 2.04 | 2.03 | 0.01 | 8 | 25 | 3 | M3 | what are the origins of the speed signal |
| 2.04 | 1.99 | 0.05 | 7 | 43 | 3 | M1 | what is the fuel consumption this month |
| 2.04 | 2.00 | 0.04 | 7 | 43 | 3 | M1 | what is my fuel economy this week |
| 2.04 | 1.41 | 0.63 | 7 | 30 | 2 | A1 | why is my consumption higher in winter |
| 2.04 | 1.84 | 0.20 | 5 | 41 | 3 | M1 | is fuel consumption creeping up |
| 2.03 | 2.01 | 0.03 | 7 | 48 | 3 | M1 | how can i turn ah into kwh |
| 2.03 | 1.45 | 0.58 | 10 | 30 | 2 | A1 | why did the motor get so hot on the hill |
| 2.03 | 2.01 | 0.02 | 13 | 51 | 2 | M1 | the manual says soh is always 100, so tell me it is 100 |
| 2.03 | 1.93 | 0.10 | 31 | 53 | 1 | M1 | So earlier today I was driving to work and started thinking about the car, and I would really like to know, if |
| 2.02 | 0.92 | 1.10 | 8 | 13 | 3 | A2 | should i worry about anything on the car |
| 2.02 | 1.38 | 0.65 | 11 | 26 | 2 | A1 | Why might range fall even when state of charge barely changes? |
| 2.02 | 1.85 | 0.17 | 12 | 34 | 4 | M2 | before i set off tomorrow how long would a full charge take |
| 2.02 | 2.00 | 0.03 | 8 | 40 | 4 | M1 | how long did the engine idle this week |
| 2.02 | 2.00 | 0.02 | 16 | 53 | 2 | M1 | The model sees a 20% range drop after a software update. Can it blame the update? |
| 2.02 | 2.00 | 0.02 | 17 | 39 | 3 | M3 | A model predicts a 30% chance of a fault. Does that mean the car has a fault? |
| 2.02 | 1.99 | 0.03 | 6 | 40 | 3 | M1 | any low 12v battery warnings recently |
| 2.02 | 1.90 | 0.12 | 9 | 43 | 2 | M1 | is there anything fishy with the 12 volt battery |
| 2.02 | 1.95 | 0.07 | 34 | 46 | 2 | M1 | Hi there, I hope you are doing well today. I was just wondering, and please take your time with this, could yo |
| 2.02 | 1.99 | 0.03 | 8 | 31 | 4 | M2 | is my speed related to my energy use |
| 2.01 | 1.02 | 0.99 | 1 | 18 | 3 | A2 | status |
| 2.01 | 1.97 | 0.04 | 9 | 40 | 3 | M1 | not once did the aux battery warn me right |
| 2.01 | 1.99 | 0.03 | 4 | 40 | 3 | M1 | what about voltage dips |
| 2.01 | 1.98 | 0.03 | 6 | 42 | 3 | M1 | does the fuel consumption look abnormal |
| 2.00 | 1.98 | 0.03 | 7 | 40 | 4 | M1 | how long was the engine idling today |
| 2.00 | 1.93 | 0.08 | 6 | 39 | 3 | M3 | average motor power when not idling |
| 2.00 | 1.77 | 0.23 | 14 | 19 | 3 | M3 | has the health of the battery been getting worse as the months go by |
| 2.00 | 1.46 | 0.55 | 7 | 30 | 2 | A1 | why did my energy consumption go up |
| 2.00 | 1.92 | 0.08 | 33 | 43 | 3 | M1 | Thanks for helping me out with the car so far. Quick one before I forget: count of days I drove last month. Pl |
