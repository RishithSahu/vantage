# The pre-parser: every rule, in one place

Written by `tools_canon_doc.py` from the tables in `vgpt/meaning/canon.py` and `vgpt/meaning/askme.py`; `tests/test_canon.py` fails if this file is out of date.

The owner asked (2026-10-24) for a rule layer in front of the language model, so that a 0.5 B model is not expected to know everything. It is **visible** (the answer shows `you wrote ..., I read it as ...`), **small** (this page lists all of it) and **counted** (`tests/test_meaning.py::RuleBudget` fixes how many pattern calls each module may contain). What a question *means* in general stays with the model.

## 1. Shorthand and spelling (`canon.normalise`)

A shorthand is replaced by its plain word. A word of five or more letters that is neither English nor in the vehicle's vocabulary is corrected only to the one nearest vocabulary word (one edit for 5-7 letters, two for 8 or more).

| written | read as |
|---|---|
| `abt` | about |
| `avg` | average |
| `batery` | battery |
| `batety` | battery |
| `batry` | battery |
| `batt` | battery |
| `battrey` | battery |
| `batty` | battery |
| `btry` | battery |
| `bttry` | battery |
| `chg` | charge |
| `chging` | charging |
| `chrg` | charge |
| `chrging` | charging |
| `chrgng` | charging |
| `coz` | because |
| `cuz` | because |
| `diag` | diagnostic |
| `diags` | diagnostics |
| `dis` | this |
| `gonna` | going to |
| `gotta` | have to |
| `hr` | hour |
| `hrs` | hours |
| `hw` | how |
| `kinda` | kind of |
| `kmph` | km/h |
| `kph` | km/h |
| `lemme` | let me |
| `lvl` | level |
| `motr` | motor |
| `mth` | month |
| `mths` | months |
| `odo` | odometer |
| `pct` | percent |
| `perc` | percent |
| `pls` | (dropped) |
| `plz` | (dropped) |
| `pres` | pressure |
| `press` | pressure |
| `r` | are |
| `regn` | regen |
| `rn` | right now |
| `spd` | speed |
| `teh` | the |
| `temp` | temperature |
| `temps` | temperatures |
| `th` | the |
| `tho` | though |
| `thr` | the |
| `thx` | (dropped) |
| `tire` | tyre |
| `tires` | tyres |
| `tmp` | temperature |
| `tmrw` | tomorrow |
| `trq` | torque |
| `tyr` | tyre |
| `u` | you |
| `ur` | your |
| `veh` | vehicle |
| `wanna` | want to |
| `wat` | what |
| `wen` | when |
| `wht` | what |
| `whts` | whats |
| `wich` | which |
| `wk` | week |
| `wks` | weeks |
| `wut` | what |
| `yday` | yesterday |
| `yest` | yesterday |

### `acc`: the only question the assistant asks back

`acc` means *acceleration* or *adaptive cruise control*. It is read from the other words: active, adaptive, cruise, disabled, distance, enabled, engage, engaged, following, lane, off, on, radar, setting, standby choose cruise control; 0-100, average, avg, brake, braking, cornering, fast, force, g, hard, harsh, lateral, launch, longitudinal, m/s, max, maximum, mean, my, peak, pedal, quick, quickly, rate, sec, seconds, slow, sudden, throttle, top choose acceleration. If nothing in the question chooses, the assistant asks which is meant, once, and takes the answer (`1`, `2`, or the words). No other question is ever asked back.

## 2. Awkward wordings read as plain questions (`canon.REWRITES`)

| pattern (on the lower-case question) | plain question |
|---|---|
| `which (?:tyre|tire)s? (?:is|are) (?:bad|worst|flat|weak|soft|deflated|losing air|going down|the problem)` | which tyre is low |
| `(?:is|are) (?:my |the )?(?:tyre|tire)s? (?:bad|flat|ok|okay|fine|alright)` | how are my tyres |
| `(?:is|are) (?:the |my |this )?(?:car|vehicle|it|ev) (?:currently |now |right now |still )?(?:charging|plugged in|on charge|on the charger)(?: now| right now| currently)?` | is the car charging |
| `(?:charging|charge) (?:state|status)` | is the car charging |
| `how (?:much|many) (?:km|kilometres|miles|range) (?:do i|can i|will i) (?:have|get|go|drive)` | how far can i go |
| `what (?:is|are|s) (?:you|your|ur) name` | what is your name |
| `(?:tell|show) me about (?:yourself|you)` | who are you |
| `what (?:do|can) you (?:do|help (?:me )?with)` | what can you do |

## 3. Questions answered without the model (`askme.HANDLERS`, first match wins)

| handler | what it answers | where the answer comes from |
|---|---|---|
| `that_date` | that_date |  |
| `charge_state` | is the car charging | latest battery row and the charge table |
| `data_catalog` | which data this vehicle records | the signal catalogue of the store |
| `how_it_works` | what the assistant runs on | the loaded adapter's name |
| `identity` | who or what the assistant is | the vehicle's model name |
| `capabilities` | what can be asked | fixed text |
| `clock` | the time and date | the vehicle's clock |
| `trip_energy` | how much battery or fuel a distance uses | the recent consumption of the vehicle |
| `trip_energy2` | same, other word order | same |
| `trip_can` | can I drive N km | same |
| `service` | last service, distance since, interval, next due | meta veh.last_service_*, the odometer |
| `law` | traffic law and licences: out of scope | fixed text |
| `buy` | where to get the car: out of scope | fixed text |
| `price` | prices: out of scope | fixed text |
| `weather` | weather: out of scope | fixed text |
| `route` | directions: out of scope | fixed text |

Every number such an answer contains is computed from the vehicle's own tables; what cannot be known says so. Pattern text of each handler: see `askme.HANDLERS`.

## 4. What it measurably does (bench/typo_stress.py, word list built WITHOUT the evaluation sets it is tested on)

300 questions of the held-out sets, each with one typo in one word: the model alone reads 90.3 % of them as it reads the clean question; with the pre-parser 94.3 %; the pre-parser broke 2 (a plural turned singular, an unknown abbreviation). Ordinary clean questions are changed only by the shorthand table (tests/test_canon.py: at most 1.2 % get any spelling change, and those are typos the sets contain on purpose). The model is robust to most typos by itself; the pre-parser's job is the shorthand it cannot know (`avg`, `acc`, `srevicing`) and the few wordings it keeps missing.
