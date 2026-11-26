# VANTAGE - Vehicle Analytics, Navigation, Telemetry & Guidance Engine (Case Study 5)

> VANTAGE is the successor of VehicleGPT 1.1 (this folder started as a copy of it; the Python package is still called `vgpt` so every earlier test keeps working). New in VANTAGE: schema-conditioned planner (`vgpt/schema_*.py`, `vgpt/slm_backend.py`), evidence contracts, composable explain/detect-change operators, independent evaluation sets (`bench/indep_*.tsv`). Current report: `VANTAGE_STATUS.md` (measured results, one source revision); architecture: `DESIGN.md`; the product configuration (what the UI runs) is `vgpt/product.py::make_brain`: rules + clarification-first policy, no language model on the answer path.

## (inherited text from VehicleGPT 1.1 follows)

# VehicleGPT - on-device vehicle assistant (Case Study 5)

Ask the car questions in plain language (typos and all); get explained, number-backed answers - **offline, inside a ~2 GB / 2 x ARM Cortex-A55 budget shared with the vehicle OS**.
**Just want to use it?** Extract the release zip, double-click `Install-VehicleGPT.bat` once, then `Start-VehicleGPT.bat` (macOS / Linux: `./install.sh`, `./start.sh`). Details: **[USER_GUIDE.md](USER_GUIDE.md)** - fixes: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - honest limits: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) - data handling: [PRIVACY.md](PRIVACY.md) - licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Everything is driven by one launcher: `python vehiclegpt.py` (`run`, `doctor`, `smoke`, `feedback`, `version`).
**Building a release (developers):** `python tools_release.py` makes `dist/VehicleGPT-<version>.zip` from an allow-list and self-tests the extracted copy in a clean environment; `python tools_release_gate.py` runs every release gate (tests, accuracy floors, package, security, offline) and prints PASS / FAIL. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

Full design (the case study's 7 Expected-Output headings + justification): **[DESIGN.md](DESIGN.md)**. Diagram: [docs/architecture.svg](docs/architecture.svg).

```
"Why has my driving range reduced over the last two weeks?"
 -> Your estimated range fell about 26% (298 -> 222 km) because energy use rose 34% (from 201.2 to 270.3 Wh/km). Main reasons: colder weather (15.6 -> 9.7 C), which raised
    climate-control use by 7 Wh/km, and more highway driving (47% -> 74% of distance). There was also a sudden jump on 2026-09-21 (+33.9 Wh/km) - check for added drag such as a roof box.
"how much peflol di i use yesterdey"
 -> Yesterday you drove 148.5 km in 2 trips, using 40.4 kWh (272.2 Wh/km, about 220 km per full charge). (This is an electric vehicle, so I report energy rather than fuel.)
"what happened before the battery overheating warning?"
 -> Battery over-temperature warning at 2026-09-30 13:14. ... battery temperature went 20.9 -> 52.0 C, coolant flow went 10.1 -> 4.0 L/min. A DC fast charge was in progress. First low-flow
    log at 13:06 (thermal_mgr). Probable cause: coolant flow collapsed while the pack heated - cooling-circuit or pump degradation (check pump / coolant level).
"how are my tyres" (generated vehicle)
 -> Tyre-pressure warning: the rear left tyre is low at 202 kPa (2.0 bar, spec about 260). Please check it for a puncture and inflate it. Last reading 14:31: ... kPa at 20 C (spec 240 front / 260 rear).
"is it rainy" / "was it raining last week"
 -> In the last 7 days the wipers/rain sensor registered rain for 8 h 13 min of the 32 h 29 min the car was in use (25%), on 6 of 6 days; 29 min of it heavy. In daylight the sky was on average partly cloudy ...
"is my car ok"
 -> 2 problems need attention.  - cooling: P1A00 Battery coolant flow low on 2026-08-20 11:37 [PROBLEM]  - cooling: coolant flow is down 50% against two weeks ago [PROBLEM]
"how will you rate my driving"
 -> In the last 7 days: 20 trips, 750.5 km in 19 h 36 min. Hard braking 37 times (4.9 per 100 km; typical is 2-6) ... I don't give a driving score, but by these numbers your driving was typical.
```

## What the last night added (2026-10-21) - read `docs/SCORECARD.md` first
* **Any data, no per-car setup**: `python vehiclegpt.py ingest <file>` reads candump / ASC / TRC / BLF / CSV / JSON / OpenXC / VSS / ELM327 / NMEA / logs, recognises each signal from the standards (OBD-II, J1939, VSS, VHAL) and its own name, unit and values, and writes the vehicle's profile pack (`docs/DATA_SOURCES.md`).
* **Car and Vantage on two devices** over a TLS link with a pinned certificate, an encrypted-at-rest option, and a one-command installer with update-and-rollback (`docs/VANTAGE_LINK.md`, `docs/DEPLOYMENT.md`).
* **Safety isolation measured, not promised**: `vgpt/isolation.py` + `tests/test_isolation.py` (`docs/SAFETY_ISOLATION.md`).
* **Car operating systems** (the PDF's Linux boards, QNX, Android Automotive / Cuttlefish): what the app needs from an OS, what it does on each, and what was and was not run - `docs/PLATFORMS.md`; checked by `tests/test_portability.py` and `tests/test_car_os_floor.py` (the real product with OpenCV, cryptography, psutil and SQLite FTS5 all unavailable).
* **Safety argument, hardware reference, attack tests**: `docs/SAFETY_CASE.md` (claim, argument, evidence marked tested / argued / not shown), `docs/HARDWARE_REFERENCE.md` (measured vs proposed), `tests/test_http_security.py` (29 attacks over real HTTP), `tests/test_deploy.py` (installer, updater and rollback run for real), `tests/test_role_walkthroughs.py` (each of the four roles through the running app).
* **Summaries, what changed, unusual readings, usual behaviour, why a trip or a parked-night battery loss happened** (`vgpt/meaning/digest.py`, `tests/test_digest.py`), and an eight-word meaning frame (`docs/FRAME_V2.md`).

## Run it as the car's application (2026-10-20)
`python vehiclegpt.py app` (add `--kiosk` on the Raspberry Pi) opens the assistant in a window of its own - no address bar, no tabs - on a private loopback port that only that window can talk to. The vehicle's data arrives as a vehicle's would (a recording is a source; the application stores what arrives; the last ~20 seconds are read live, anything older from the dataset: `docs/ARCHITECTURE.md` 8c), roles give a service engineer, a fleet operator and the validation team their own pages with the privacy rules of 8d, and voice input listens for *"Vantage ..."* with an offline recogniser (8e; the speech model is not installed yet). Raspberry Pi launcher files: `deploy/raspberrypi/`.  **Two devices**: the laptop as the car sends a chosen vehicle's data over a signed link to the Raspberry Pi (Vantage), which stores it, runs the assistant and hosts the website - `docs/VANTAGE_LINK.md` (`Start-Car.bat`, `Open-Vantage.bat`).

## The default screen: a car-cockpit interface
`start_ui.bat` lands on `/`, a dark and light cockpit display: instrument cluster (speed arc, battery arc, top-down car with tyre pressures, gear), vehicle readings, warning lamps that double as shortcuts, quick-question tiles, the assistant, a charge card, a route map and the live camera.
Vanilla JS and CSS, no build step, fully offline: the Outfit and Michroma fonts and the Phosphor icons are bundled in `ui/modern/` (licences in `ui/modern/LICENSES.txt`). `/classic` and `/headunit` keep the earlier screens.

## Try it yourself: car head-unit simulator (camera + assistant)
```bash
start_ui.bat              # opens http://127.0.0.1:8765  (python ui/server.py; local only)
start_ui.bat webcam       # start on your webcam: cover the lens with your hand -> "Front camera blocked"
start_ui.bat none         # rules only, no language model
```
A 1920x720 infotainment-style UI: Drive screen (speed / charge / range, live camera view, assistant), Camera health page (metrics, history, event thumbnails), Assistant, Alerts, System (memory, model state, ms/frame).
The head unit works out **what kind of vehicle it is looking at from the data itself** (battery-electric, diesel, petrol, hybrid, or "energy source not reported") and shows a battery or a fuel gauge, the right readings, lamps and questions accordingly (`docs/ARCHITECTURE.md` section 8b); `data/world_aster.db` and `world_meridian.db` are electric, `world_rhino.db` is a diesel truck, `world_kestrel.db` a petrol hatchback.  The **Test track** tab of the settings has one choice, the **dataset** (the 105-day worlds, the generated vehicles in different cities and the original scripted one).  The car then drives by itself: it replays the recorded trips of that dataset in real time, one after the other, and skips the parked time in between; there is nothing to set (the part of the history stored once per second, the last week, is what plays, and it begins again when the last drive is over).  The camera is the recorded dashcam clip (`--camera webcam` uses your webcam)
and jump to interesting moments of the vehicle's history (morning commute, driving in rain, a tyre warning appearing, DC fast charging, camera blocked ...). Drive screen shows tyre pressures (they turn amber/red), a rain chip, telltales and banners.
Say "I want it in psi / miles / Fahrenheit" (or use the Units switches on the System page) and both the assistant and the Drive screen follow. Ask "what can I ask you?", "what is a CAN bus?" or, after an answer, "what do you mean by more?" - the conversation layer
(`vgpt/dialogue.py`) answers those deterministically, remembers the last answer, and politely declines things it should not do (parking search, how-to-attack instructions).
GPS: "where am I?", "which direction am I heading?", "how far is work from here?", "how long to get home?", "what are my GPS coordinates?" - answered from the live fix (distance and time to saved places use your own earlier trips).
Ask: "How are my tyres?", "Is it raining?", "Is my car healthy?", "When is my next service?", "How many times did I brake hard last week?", "Where did I go yesterday?", "Is the camera blocked right now?", "How much did I charge last week?" ...
Real in the UI: the assistant, the camera analyser, the camera -> SQLite ingest path every answer reads. Stand-ins: OpenCV video decode (a car's ISP/VPU would do it), telemetry replayed from the simulation (not synchronised with the video),
browser speech recognition (off by default; may use a cloud service), the lane-assist indicator (derived from camera state, not a real ADAS function). Runs on the host, not in the ARM guest.

## Realistic random data: the generator (`tools_datagen.py`)
The original simulator was one scripted story, so most real-world questions (tyres, rain, service, places ...) had no data behind them. The generator makes **random, reproducible, realistic vehicle histories** from real data:
| Source | What it supplies |
|---|---|
| **Open-Meteo** hourly archive (downloaded, cached in `data/ref/weather/`) | real temperature, rain, cloud, sunlight for the chosen city and dates -> ambient temperature, rain-sensor/wiper state, ambient light, climate load, sun glare on the camera, rainy-day behaviour |
| **VIF-AEGIS** (your Downloads zip): 31 real trips, 24 Hz accelerometer | real speed traces, and **hard braking / acceleration / cornering from the real accelerometer** (axes identified per trip) |
| **EPA** dynamometer schedules UDDS, HWFET, US06, SC03, FTP (downloaded, public domain) | city / highway / aggressive speed traces |
| **GPS stream** | a 1 Hz GNSS track (3 m correlated noise, course, speed, HDOP, occasional dropouts) along each trip between the saved places, 1 fix a minute while parked; the live ingest path decodes real NMEA 0183 (RMC/GGA), checked against `pynmea2` and real GPS fixes from a VIF drive. The UI has a Map page that draws it live |
| physics + profiles | mass, drag, rolling resistance vs tyre pressure and temperature, regeneration, heat-pump HVAC, battery thermal model, charging curves vs temperature; 5 drivers x 4 vehicles x 8 cities; home / work / shops and the trips between them; random faults: cooling-pump degradation, slow tyre leak, weak 12 V battery, dirty windscreen, roadworks, roof box, a harsh-driving week |
```bash
make_data.bat                                         # builds the 4 demo vehicles: data/gen_demo.db (Bengaluru commuter), gen_cold (Oslo winter), gen_courier (London delivery), gen_roadwarrior (Graz, long distance)
python tools_datagen.py --list                        # cities, drivers, vehicles, stories
python tools_datagen.py --seed 7                      # a random vehicle/driver/city/faults chosen from the seed (reproducible); add --city/--driver/--vehicle/--stories/--days/--end to pin things
python tools_datagen.py --seed 7 --offline            # synthetic weather instead of downloading
```
Each run writes the on-device database plus `*_truth.json`, ground truth computed from the generator's arrays; `tests/test_datagen.py` checks determinism, energy conservation, the hard-event rate against the real accelerometer data, and that **the assistant's
answers equal the truth** (km, hard-braking counts, rain minutes, charged kWh, odometer, service km, places per trip, tyre warnings, the planted cooling fault). Honest limits are in DESIGN.md section 9 (routes are not real roads, tyre/12 V/brake-wear/battery-health are engineering models, the extra
signals are bulk-loaded history without live adapters yet).

## Analysis engine: "show me the evidence" questions (`vgpt/qa*.py`)
The case study asks for more than lookups: *search the history, find trends and anomalies, explain what happened, say what it cannot know*. The analysis engine answers a 120-question stress list (basic retrieval, temporal reasoning, trends, multi-signal reasoning,
explain-why with evidence, cross-source joins, numerical questions, missing data, false premises, vague human questions, long-range history, safety boundaries, hallucination attacks, prompt injection, "brutal" case-report questions + a 6-question demo sequence) on every generated vehicle:
```
Q1 "Why has my driving range reduced over the last two weeks?"   -> the change, whether it is statistically real (Welch p), what the recorded conditions explain (trip regression), the biggest single-day jump, confidence
Q2 "What evidence supports that conclusion?"                       -> daily Wh/km before/after with dates and t-test, regression shares, other signals that changed
Q3 "Show me the exact time periods where the change occurred."    -> change-point date, step vs drift, largest daily jumps, the most energy-hungry trips (timestamps)
Q4 "What other vehicle signals changed during those periods?"     -> 12 signals compared with significance (coolant flow -31%, p=0.02, ... the rest not significant)
Q5 "Could there be another explanation?"                           -> ranked alternatives with data for and against (weather, driving mix, tyres, ageing, things the car does not measure)
Q6 "What data are you missing to distinguish between those explanations?" -> payload, wind, gradient, cabin set-point, accessories, a capacity test ... and what each would settle
```
How it works (all deterministic - the language model is not involved and never writes a number): keyword routes (`qa_final.py`) -> otherwise a **fuzzy router** (similarity to ~800 example phrasings in `vgpt/data/qa_protos.json`, built by `tools_build_protos.py`; the older
questions are negative examples so they stay on the older pipeline) -> a handler computes the answer from the SQLite store (1 Hz tier for the last 7 days, per-minute tier for 90 days, frozen 1 Hz context around every warning/fault, trips, events, places, GPS) -> templated text with timestamps, units and p-values.
Rules it follows: **RPM is derived** (an EV has no RPM sensor: speed x fixed 9:1 ratio, always labelled); **cabin temperature, throttle, steering angle, radar, rear/side cameras are not recorded and are said to be missing** (nearest real signal offered); a date outside the data is refused, not invented; **"you said ..." premises are
checked against the conversation and the data** (and corrected); causal questions get "association, not proof" with the numbers; **safety-critical commands (ABS, speed limit, battery protection, ADAS thresholds, airbag, "override the safety controller") are refused** - the assistant is read / analyse / explain only; text stored in logs and events is treated as **data, never as an instruction**
(an injected "ignore all previous instructions ..." event is quarantined and reported; `tests/test_qa.py` plants one); a date that does not exist ("31 February"), a time that does not exist ("99:99") or a date outside the recorded history is refused rather than answered for some other day; the older domain questions now also accept a named day ("how were my tyres on 13 September?"). `bench/eval_torture.py` measures wording robustness: on **unseen paraphrases** of the stress list the first measurement was 84% / 83% routed as intended (two held-out sets of 119 and 66; after fixing what they exposed they are in the example pool, so they are no longer held out - new wording needs a new blind set).

### Hard-reasoning round: causal, counterfactual, statistical, look-ahead, protocol, poisoning (`qa_d.py`, `qa_e.py`)
A second stress list (`bench/torture2_questions.txt`, ~129 questions; tests in `tests/test_qa2.py`) is answered by the same engine: "prove X caused Y", the causal effect of speed on pack temperature controlling for ambient and load (OLS with standard errors, partial correlation), a causal graph and a root-cause ordering for the last overheating, counterfactuals (slower by 20 km/h, 15 C instead of 30 C, better cooling, gentler braking - always labelled **model estimates**, or refused when the quantity is not recorded: road gradient, wind, cargo, individual cells), hidden-variable and fleet questions ("only this car's data is held"), long-horizon degradation (rate, linear vs accelerating by AIC, bootstrap interval, change points, periodicity), statistics (Welch / Mann-Whitney, CI, Cohen's d, Holm and Benjamini-Hochberg, Pearson vs Spearman), **look-ahead-free diagnosis** (the same diagnosis re-run with only data up to N seconds before the event, plus what a live system could have known), sensor quality (drift, impossible values, glitch vs real, reconstruction), protocol (CAN payload decoding from `vehicle.dbc`, message / scale / ECU mapping), ontology (same quantity vs same phenomenon, incident grouping, dependency graph), "explain your reasoning" (measured vs inferred, assumptions, what would prove the answer wrong), poisoned logs (a log line saying "all normal" does not change the health answer) and impossible precision ("exactly 17 min 32 s from now" gets a historical range, not a number). The three-part "nuclear" test (full investigation -> strongest evidence it is wrong -> re-run with only pre-incident evidence) is supported.
**Wording robustness, honestly:** the first blind measurement of these new categories on unseen paraphrases was **9% (6/61)**; after adding examples and arbitrating weak keyword routes with the fuzzy router the labelled sets route 100%, but those sets are now used up - they no longer measure generalisation, so a fresh blind set is needed for a new number.

### Driving use: short answers, voice in, voice out
Every answer comes back as three strings: `answer` (full, with the evidence), `brief` (one or two plain sentences - what the driver sees and hears first) and `speech` (the brief with dates and units spelled out for text-to-speech: "21 September at 14:53", "kilometres per hour", "degrees Celsius"). The UI shows the brief with a *Details* link for the evidence; the microphone is on by default (browser speech-to-text -> `/api/ask` -> the assistant sees only text) and answers to spoken questions are read aloud with the browser's speech synthesis. Speech recognition / synthesis are stand-ins for the car's own voice stack.
Polish: dates are human ("21 Sep, 14:53"), temperatures are in degrees, statistics in brackets (p-values, R2, CI) stay in the details and are dropped from the brief, and the brief is written per handler wherever the first sentence would be too technical.

### Co-pilot features (demo-able in the head-unit simulator)
* **Pre-drive briefing** - "Brief me", "Good morning", "Is it safe to drive?", "anything I should know before I set off" (and it runs by itself when the assistant opens). One verdict (*OK to drive / Drive with care / Service advised before any long drive*), battery and range, last trip, what needs attention, one plain suggestion, read aloud in full. Read-only: it reports and advises, it never changes a setting.
* **Proactive alerts** - when a battery over-temperature warning, a new trouble code or a tyre warning appears while you drive (try *Test bench -> Battery overheating warning* at 60x), the assistant raises it by itself, explains the most likely cause in one sentence and reads it out; no question needed.
* **Evidence chart** - any answer about a warning or fault has "Show the signals around it": battery temperature, coolant flow, power and speed for the 15 minutes before to 3 minutes after, with the event and the moment coolant flow fell below 75% of normal marked. Drawn from the recorded 1 Hz context (kept for every warning even after the 7-day 1 Hz tier is gone); nothing in the chart is estimated.
* **Hands-free conversation** (Test bench -> Hands-free) - after each answer the microphone reopens by itself; say "stop" / press Esc to interrupt, "hands free off" to leave. Speech-to-text / text-to-speech are the browser's (stand-ins for the car's voice stack); there is no wake word and no barge-in while it is speaking (echo would feed back into the recogniser) - interrupt with the stop button, Esc, the mic button or a new prompt.
* Answers are short by design: one or two plain sentences first, the evidence under *Details*, lists read out in full.

### The app (`start_ui.bat` -> http://127.0.0.1:8765) and everyday questions
* **Responsive assistant UI** (`ui/app.html`, the default page): a chat with the short spoken-style answer first and *Show details* for the evidence, live vehicle panel (speed, charge and range, pack temperature, coolant flow, tyres, alerts you can tap, front camera, trip map), voice in / voice out with a stop button, hands-free mode, settings drawer with the demo controls (replay moments, speed, camera faults, units, vehicle). It adapts from phone to wide screens; nothing is clipped (long answers scroll, details wrap). The original fixed 1920x720 head-unit simulator is still at `/headunit`.
* **Everyday questions** (`vgpt/qa_g.py`, `qa_h.py`; `bench/everyday_questions.txt`, `tests/test_everyday.py`): "how's my battery", "how efficient was I yesterday", "when did I last charge", "what did charging cost (at 0.30 per kWh)", "how long to charge to 80%", "do I need to charge / can I make it 150 km / will I get home", "am I driving too fast / what is my driving score", "how can I improve my range", "was the battery ever hot", "is the coolant pump OK", "is it safe to keep driving", "what does that code mean", "summary of the week", "compare this month with last month", "longest trip", "which places do I visit most" ... 111 such questions all get a real answer on every generated vehicle with the language model switched off; where the car does not record something (electricity price, speed limits, battery age) the answer says so instead of guessing.

### Understanding everyday wording (honest numbers)
Three layers decide which answer a question gets: keyword rules, a nearest-example router, and (new) a **linear question classifier** (`vgpt/qa_clf.py`: tf-idf of synonym-mapped words and character 3-5-grams, one score per route, trained offline with scikit-learn by `tools_train_clf.py`, 1.5 MB, numpy only on the car, ~5 ms). The classifier overrules a keyword route only when it is clearly more sure, answers questions no keyword matched, and recognises off-topic questions. Training wordings come from hand-written paraphrases plus a small grammar (`tools_everyday_grammar.py`, ~12,000 generated wordings, weighted lower).

History of the first version (keyword rules only), each set measured *before* it was used for tuning: **19%** (67 unseen phrasings), then **74%**, **78%**, **66%** on three more sets.
Now measured on sets that are **not training data** (`bench/eval_everyday.py`, language model off, 36 everyday routes + small talk + Hindi-English + off-topic):

| held-out set | questions | answered as intended |
|---|---|---|
| `heldout_dev.tsv` (used to tune thresholds) | 93 | **98%** |
| `heldout_dev2.tsv` (used to tune thresholds) | 93 | **97%** |
| `heldout_final.tsv` (written once; never used for tuning or training) | 89 | **94%** |
| `heldout_messy.tsv` (typos, speech-to-text slips, filler words; written once) | 67 | **94%** |

With ~90 questions per set the uncertainty is about +/-5 points, so read the last two rows as "roughly 89-98%". The four older sets are now training data and score 96-100%, which proves nothing about unseen wording. The same results hold on the other generated vehicles and the scripted `vehicle.db`. 25 off-topic questions ("tell me a joke", "what time is it in Tokyo", "my phone is not charging") are never answered as if about the car. The older analysis routes are protected: `eval_torture` stays at 100% of 119 + 246 labelled questions.
`tests/test_routing.py` turns these numbers into regression floors (tuning sets >= 95%, final / messy / earlier sets >= 90%); `tests/test_clf.py` and `tests/test_handlers.py` test the classifier (no scikit-learn on the car, deterministic, fast), the grammar (no held-out leakage), small talk / Hindi-English / help phrases, every handler's fallbacks on the table-less scripted car, number parsing in questions, and that no source file contains a stray backspace character (a shell heredoc once silently disabled regexes that way).

## Query compiler: a typed, compositional query IR (new in 1.1)

The language understanding was refactored from "one intent per question shape" into **compile, then execute**. A question is compiled into a small typed query (`vgpt/query_ir.py`); everything after that is deterministic, tested code.

```
question -> nl2ir (rules, ~1 ms) -> [IrSlm: grammar-constrained 0.5B model + LoRA, only if the rules are unsure]
         -> Query IR -> validator -> capability -> planner (trusted SQL compiler) -> executor (read-only) -> evidence -> composer -> verifier -> answer
```

* **Compositional**: "average battery temperature during my three least efficient highway trips last month" is one query, not a new intent. Time x selection x conditions x statistic x signal combine freely.
* **Capability-aware**: it says *unknown signal* / *known but this vehicle does not record it* / *recorded signals are insufficient to derive it*, with the exact missing signals and the formula that would compute it. Electrical power is never silently used as motor power; estimates are labelled.
* **Verified answers**: every number and unit in an answer is checked against the evidence; no bare causal claims; no vehicle-control instructions.
* **Safe by construction**: the model's output grammar is generated from the registry (no free text, SQL, paths or commands), the SQL compiler is whitelist + bound parameters + `SELECT` only, and the connection is read-only.
* **Backwards compatible**: the older routes still answer everything they answered before (`VGPT_QC=auto` hands the new path only what they cannot express; `VGPT_QC=off` disables it). Existing databases work unchanged; the simulator extension (`tools_datagen_ext.py`) is additive.
* **Trainable and measurable**: `training/` builds an execution-verified dataset with compositional held-out splits, trains a LoRA (assistant-only loss), converts it to a separate GGUF adapter, and evaluates rules / base model / base+LoRA / hybrid with 13 metrics. Results: **[docs/evaluation.md](docs/evaluation.md)**.

Docs: [query_compiler.md](docs/query_compiler.md) · [signal_ontology.md](docs/signal_ontology.md) · [training.md](docs/training.md) · [evaluation.md](docs/evaluation.md). Commands:

```
python -m unittest discover -s tests                                    # whole suite (incl. tests/test_qc.py, test_parity.py, test_ext_training.py)
python tools_datagen_ext.py data/gen_courier.db --out data/gen_courier_ext.db      # extended signals + fault scenarios with ground truth
python -m training.build_dataset --n 26000 --adv 900                    # execution-verified dataset with compositional splits
python -m training.train_lora --epochs 2                                # LoRA (GPU)
python -m training.convert_lora_gguf training/out/qwen25_05b --out models/ir-lora-qwen25-05b.gguf
python -m training.evaluate --backend hybrid --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora models/ir-lora-qwen25-05b.gguf
```

## Run it
```bash
# dev environment (cross-validation libraries + numpy for the simulator; not part of the in-car footprint)
python -m venv .devenv && .devenv/Scripts/pip install cantools python-can udsoncan can-isotp someipy numpy
.devenv/Scripts/python tools_seed.py                 # build the simulated 28-day vehicle + DB + ground truth     (tools_record.py: real-wire-format replay)
python app.py                                        # chat on the laptop (Qwen2.5-0.5B, lazy-loaded, prompt state cached);  --persona engineer for evidence
.devenv/Scripts/python -m unittest discover -s tests -v          # 113 tests (analysis engine; protocols vs libraries / real data; analytics; safety; camera; head-unit server; generator vs truth; NLU)
# the case study's ARM virtualised environment (QEMU, Cortex-A55, 2 cores, 2 GB):
python vm/driver.py refresh                          # boots the Debian arm64 guest, installs nothing new, runs bench/run_guest.py inside it, saves results/guest_*.json
```
Not in the repo (large / third party): `models/*.gguf` (Qwen2.5-0.5B-Instruct Q4_K_M, Apache-2.0), `vm/` images (QEMU, Debian arm64), the datasets in your Downloads folder (read in place, never copied).

## What was measured (inside the ARM guest, `results/guest_q05_secondboot.json`)
| | |
|---|---|
| Platform | `aarch64`, Cortex-A55 (part 0xd05), 2 cores, 1,977 MB RAM visible, no swap |
| Memory used, whole system | OS idle **132 MB** -> +model loaded **282 MB** -> after ~750 questions (136 original + 112 extended + 694 analysis) 445 MB; min available **1,532 MB**; assistant peak 791 MB resident (649 MB right after loading the model; 185 MB rules-only) |
| Boot to ready | **538 s** first boot (one-time prompt-state build; varies with host load, earlier runs 393-697 s) -> **17 s** afterwards |
| Answers under 2 s | original 136: **130 / 136 (96%)**, median 46 ms (rules only 136/136); extended set: 111/112, median 20 ms. **Analysis engine, 694 history / causal / statistical questions on the generated courier vehicle: 630 under 2 s (91%), 673 under 5 s, median 171 ms, p95 3.1 s, slowest 10-12 s** (the 16 s outlier is a vague question sent to the model) (the heavy ones scan a week of 1 Hz data - speed vs temperature correlation, largest temperature rises - under emulation; earlier 503-question run: 460 under 2 s, max 9 s) |
| Model path | consulted for 1/136 original questions and a handful of the 694 analysis questions (the rest never need it); **17-32 s per call under QEMU software emulation** (0.3-0.4 s on the laptop) - emulator speed, not an A55 number |
| Live ingestion of real wire formats | 502 msg/s real time at 15% of one emulated core; **~6,200 msg/s unpaced** |
| Accuracy (same as host) | dev 22/24, test 24/24, blind 23/24*, your transcript 16/16, typo-corrupted 47/48; **extended + conversation + GPS set 112/112**; analysis stress list + paraphrase sets: 361/365 labelled questions routed as intended, 664/694 answered by the analysis engine (the rest are the older pipeline's questions) |
\* the "blind" set was run once untuned (rules-only 20/24); two rule gaps it exposed were then fixed, so it is no longer fully held out. See DESIGN.md section 8.

## Protocols - each validated against an independent implementation and/or real recordings
CAN/CAN-FD + DBC (cantools: 0 mismatches / 187,373 real ROAD frames; python-can: identical on real CAN-FD files) - ISO-TP (can-isotp: 300/300) - UDS/DTC (udsoncan) - OBD-II (270,005 real samples) -
SOME/IP + SD (someipy: 2,000/2,000) - Android VHAL (real AOSP property table, 280 properties) - QNX/Android/Linux logs (Android 2,000/2,000, Linux 1,999/2,000 real lines).

## CAN-bus intrusion monitors on real datasets (extra to the document; results in `results/*.json`)
HCRL (16.6 M frames): DoS/fuzzing F1 1.00, RPM spoof 0.96, gear spoof 0.94, no false alarms - OTIDS: 0 false alarms / 947,760 unseen frames, DoS 1.00, impersonation **undetected** - Autohack2025: F1 0.71 -
ROAD (real vehicle): correlated-signal F1 ~0.89, max-speedometer 0.76-0.92, reverse-light flips **undetected** - SynCAN: flooding F1 0.75, other four attack types **near-zero recall**.

## Datasets
Used (in place): HCRL Car-Hacking, OTIDS, Autohack2025, ROAD, SynCAN, VIF-AEGIS (OBD driving cycles, OBD formula test, **and now accelerometer-derived hard-driving events**). Downloaded: python-can CAN-FD ASC samples, AOSP VHAL AIDL files, Loghub Android/Linux logs,
**Open-Meteo real weather**, **EPA drive schedules**, a 10,000-word English frequency list (so ordinary words are never "corrected" by the typo repair).
comma2k19 (one real dashcam clip, used to evaluate the camera monitor). Not used: the password-protected "Survival Analysis" IDS archive, `18801994.zip`. Respect each dataset's own licence; nothing is redistributed.

## Not met / not measured (details in DESIGN.md section 9)
No real-A55 measurement (QEMU timings only; the model path is **not** shown to meet 2 s) - camera feed = health monitor evaluated on one real clip with injected faults, not measured in the ARM guest - SOME/IP/VHAL/slog2info validated against specs and libraries,
not live buses - simulated vehicle and faults - question set is mine - no encryption at rest, no engineer-mode access control - CAN monitors weak on subtle value/impersonation attacks.
