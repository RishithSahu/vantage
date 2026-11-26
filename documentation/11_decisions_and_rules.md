# 11. Decisions and the owner's rules

This is the register of the rules the owner set and the design decisions taken, each with its date and its reason. It exists so that nobody (human or assistant) re-opens a closed question by accident.

## 1. The owner's standing rules (in force now unless stated)

| Rule | Set | Why / what it means in practice |
|---|---|---|
| The Case Study 5 PDF is the highest authority ("the word of god"); do not relax its constraints | 2 Oct, repeated 8 Oct | no cloud; about 2 GB and 2 ARM cores; under 2 s for common queries; weeks of history; safety isolated |
| Enforce the limits for real, and say where a number was measured | 2 Oct | a laptop core is faster than an A55; every latency figure must name where it came from |
| **Ask before downloading any model or dataset** (name, source, size) | 2 Oct | disk space and provenance; approved downloads are on record (Qwen3-0.6B, LFM2.5-1.2B, bge-small, Granite 4 1B, xLAM-2 1B, GLiNER2.5, DeBERTa NLI, Qwen2.5-7B research reference, Vosk) |
| The model decides what a question *means*; code may only parse values, type-check, calculate and guard safety. **Never fix a failing question by adding a phrase pattern.** | 18 Oct | the owner objected to fixing failures with regexes ("no better than humans writing answers"). Failures become data (`tools_triage.py`: glossary line, ontology entry or training row). `RuleBudget` counts `re.` calls per module and fails on growth |
| Where rules exist they are visible, listed and switchable | 24 Oct, 25 Oct | the pre-parser shows "I read it as...", every rule is in `docs/CANON.md`; `docs/BOUNDARY.md` lists every stage as learned or code; `VGPT_PREPARSER=0`, `VGPT_LAYERS_OFF`, `VGPT_MODEL_ONLY=1` measure the model alone |
| **No cache of any kind for answers or frames**; a repeated question is a real run | 22 Oct, 25 Oct | every latency figure must be a real model run; answers must reflect the data as it is now |
| **No read-ahead or prediction of the user** | 23 Oct | plain pipeline cuts only |
| **No added query latency** for storage savings or features; ask first | 23 Oct | "we cannot afford even 1 ms more latency"; a cold-minute archive was reverted for this reason |
| Training needs the owner's approval; adoption is gated and pre-registered; sealed sets are never trained on or read for failures | rules since 19 Oct; approvals: 20 Oct 02:00 (asleep, autonomous), 24 Oct (overnight); stops: 20 Oct 10:00, 21 Oct | training is allowed only when the owner says; a new adapter replaces the old only if target sets rise and no frozen set drops over 1.5 points |
| **Ask-back is off** except for an exactly ambiguous shorthand | off 24 Oct; narrowed 24 Oct evening | "no confusion"; `acc` alone (acceleration or adaptive cruise control) may ask; everything else is answered or refused |
| Honest claims only: say what is measured and what is proposed; do not claim "perfect", "model-only" or numbers the data does not support | repeatedly; 25 Oct | the product is a hybrid and says so; confidence below 0.50 is refused rather than answered |
| The car must drive by itself following the dataset; the interface adapts to the data (EV, petrol, diesel); no sliders for the person to set | 20 Oct | "it must feel like sitting in a car"; the Live page (sliders, buttons) was removed from the interface |
| Cars stream as they really stream; the receiver adapts to any car with no per-car setup; Vantage must not be told the car, protocol or channels | 20 to 21 Oct | the car is a separate program joined to Vantage only by a local pipe |
| **English-named signals only** (no VW MQB or PSA German/French databases for now) | 21 Oct | the mapper reads English names |
| "Any OS" means any **car** OS (Linux, QNX, Android Automotive, AUTOSAR), not laptops or phones | 21 Oct | the earlier laptop/phone work (PWA, mac bundles) is kept and labelled, not extended |
| It is an **app**, not a website | 21 Oct | application window, kiosk mode, in-process bridge, no port |
| Keep the Raspberry Pi under 2 GB; leave the other Pi projects alone | 20 Oct | `MemoryMax=2G`; do not touch the CS3 "aqe" project, the hotspot or the firewall rules. The Pi may be rebooted (allowed 20 Oct) |
| **Nothing is uploaded**: the git repository is local only | 25 Oct | tag `v1.3.4`; no remote |
| **Explain every step in plain language**; no phone-size UI checks | 20 to 21 Oct | the owner could not follow terse progress notes |
| Never touch the real Vantage store when testing; use a separate port and the dev harness | 24 Oct | a test car once appended six hours of rows to the real store; `bench/dev_web.py` and `bench/dev_up.sh` isolate |
| Leave the model alone unless told | 21 Oct, 25 Oct | the 25 Oct instruction for the accuracy audit was "do not change any code, just find other methods" |

## 2. Design decisions and the reasons

| Decision | When | Reason, with the evidence |
|---|---|---|
| Small model writes a closed-vocabulary **frame**; code executes | 18 Oct | frozen models were 12 to 67 %; a trained 0.5 B reading into a closed vocabulary reached 90 %+; safe by construction |
| **Qwen2.5-0.5B** as the only model | 18 Oct | fits 2 GB; larger models over the memory or latency limit ([06](06_ai_model_and_training.md)) |
| Typed Query IR and the **only SQL in one file** | 5 Oct | the model can never write SQL; read-only handle |
| One **profile pack** per vehicle as the only integration artefact | 19 Oct | plug-and-play any vehicle; a car is described by data, not code |
| Vehicle type **read from the data**, never a setting | 20 Oct | a real vehicle does not announce its type |
| **Single writer**, many read-only readers | 19 Oct | no two writers can corrupt the history; the assistant cannot write |
| Receive-only, listen-only collectors; the wire writer exists only in car code | 21 Oct | safety isolation |
| Local **pipe**, not a network link, between Car and Vantage | 21 Oct | offline by construction; replaced the HTTP link |
| **int8 ONNX Runtime** for the Pi | 22 Oct | the A72 has no dot-product instructions; 2.2 times faster than llama.cpp at the same accuracy |
| **Token mask** instead of a grammar sampler | 21 Oct | 62 % of parse time was the sampler walking 151k tokens; frames identical on 200 of 200 questions |
| One adapter with a **car-type line** instead of per-type adapters | 24 Oct | measured: ice_v1 76.0 with the hint against 66.9 without; ICE-only fine-tune 74.4 |
| Frame **version 2** (8 words) | 21 Oct | `c2` (a second quantity) and `ref` (what the question points back to) replace cue-word lists for follow-ups |
| **Confidence calibrated** on development sets only; refuse below 0.50 | 25 Oct | the raw token probability is not a probability |
| **Daily tier and archive**; five years of minutes in a 6 GB budget | 23 Oct | long history cheaply, exact old seconds when asked |
| **Cause engine** replaces association-only explanations | 23 Oct | "explain causes" is a judged criterion; scored on a sealed set |
| **Route policy**, deny by default | 25 Oct | a request that is not in the table does not exist |
| Records are **minimised, sealed and expiring** | 25 Oct | privacy |
| A pre-parser **in front of the model**, visible and counted | 24 Oct | "do not expect a 0.5 B model to know everything; help it" |
| The word "model-only" **withdrawn** | 25 Oct | `docs/BOUNDARY.md`: stages 2, 3, 4, 8, 9, 11 can change which question is answered |
| Retired: arbitration rules, the vehicle-vocabulary layer in its first form, frame cache, answer cache, HTTP link, per-car adapters | various | each was measured to help little or hurt (arbitration changed 65 frames, fixed 2, broke 56) |

## 3. Process rules we hold ourselves to

* **Pre-register**: write the comparison, the adoption rule and the noise threshold before running (done for the October 6 comparison and every training gate).
* **One source revision per report**: figures in a report come from a named source hash (`results/final_manifest.json` in early October; `RELEASE.json` since 25 Oct).
* **Never write benchmark-ID handling** or question-specific branches.
* **Do not select a configuration on the acceptance or sealed set.**
* **Do not count simulated clarification or author-run tasks as user success.** "Correct after one extra turn" is an oracle upper bound.
* **Report assisted and autonomous results separately.**
* **Contradictions between the gold labels and the model are first a data question.** `tools_triage.py` says whether the fix is a glossary line, an ontology entry or a training row; code is the last resort and counted.
* **Test where the problem lives.** Testing only on the Pi found two bugs that the laptop hid (a Python 3.13 thread-attribute clash; a full-table MIN/MAX scan on every question). Latency on a laptop is not latency on the target.

## 4. Practical gotchas recorded along the way

* The shell tool turns a typed `\b` into a backspace character and `\n` into a real newline inside heredoc Python: write patches with the file tools, or build the characters with `chr(92)` and `chr(10)`. A test (`test_clf` hygiene) catches `\x08` in source.
* Tests that import `vehiclegpt.py` or call `_adopt_product_model` must use `tests/_envguard.py` or they leak `VGPT_FRAME=2` into later tests.
* Run long jobs as plain background commands or with PowerShell `Start-Process`; `nohup` children die when a tool call times out.
* `bench/probe_ask.py` uses the ONNX build named in `product.json` and ignores `--adapter` unless `VGPT_BACKEND=llama`.
* Run Python benchmarks on the Pi with `~/vehiclegpt2/.venv/bin/python` (it has onnxruntime and llama.cpp); system Python lacks one of them and the model "could not be loaded".
* Never delete a trained adapter you may need: old models were removed on 22 Oct to free 30.5 GB and cannot be recovered except by retraining.
