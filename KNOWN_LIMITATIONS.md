# Known limitations (please read before relying on it)

We would rather you hear these from us than discover them.

**State of 2026-10-21 (this block is current; where an older line below disagrees, this block wins - see `docs/SCORECARD.md`)**
* **No real vehicle has ever been attached.**  Every vehicle is a recording or a simulation replayed as its real wire traffic.  The SocketCAN collector has run only with a stand-in socket; the MQTT / VISS / HTTP collectors only against servers written for their tests, not against Mosquitto or a production gateway.
* **The numbers about understanding are measured on question sets we wrote ourselves** (the model carries the meaning: with every rule layer switched off the answers change for 2 of 2,176 questions; pooled exact-frame accuracy on the older sets is about 90 %, on the newer ones 85-93 %).  Nobody outside the project has used it; there is no user study.
* **The Raspberry Pi was last measured before the faster decoder** (new wordings 3.7-6.2 s); on the development laptop an answer now takes about 0.4 s at the median.  The installer, the update-with-rollback script and the hardened service unit have been run for real only on the development PC (with stand-ins for systemd: `tests/test_deploy.py`), the container file not at all; none has run on a device.
* **The safety boundary is software on one host** (measured by `vgpt/isolation.py`); there is no partition or hypervisor.  A safety **argument** with its evidence exists (`docs/SAFETY_CASE.md`) but no ISO 26262 / SOTIF work product.  The assistant cannot operate a car because no code path does, whatever the model says.
* **It has run on one car-class operating system only (Linux, on a Raspberry Pi 4).**  QNX, Android Automotive / Cuttlefish, a Yocto image and a hypervisor guest were not run; the Android display app, a Binder client for the Vehicle HAL, a reader for the QNX CAN driver and a Yocto recipe were not written (QNX and Android logs have a tested forwarder, never run on a real device) (`docs/PLATFORMS.md`).  The code is checked for Linux-only and POSIX-only calls and runs with OpenCV, cryptography, psutil and SQLite FTS5 all missing, but that is a simulation of a sparse system on a Windows PC, not a QNX run.
* A question the model reads as a follow-up when it was a fresh one (or the reverse) still happens: 87 % right on a held-out set of 167 complete questions asked after another answer.
* "Battery power" asked of a diesel is answered with the engine's power (the model cannot tell the two apart); it is the one known substitution left in the answerability audit.

**What the product is**
* A demonstration of an offline vehicle assistant over a **simulated** 28-day vehicle history (five vehicles). It does not connect to a real car. The "live" panel is a replay.
* **Advisory only.** It reads recorded data. It does not monitor a car in real time, cannot control anything and must not be used to decide whether a vehicle is safe. Its go / no-go answers say so every time.

**Understanding questions**
* About **9 in 10** unseen everyday questions get the intended answer with the rules and classifier alone (measured 94-98% on fresh test sets of ~90 questions; the uncertainty is about +/-5 points). The rest get a "did you mean ..." with related suggestions, a closest-question guess marked *read as ...*, or - when the optional language model is installed - a model-assisted reading. Very long, multi-part or heavily slangy questions are the most likely to miss. English only, plus a handful of Hindi-English phrases.
* It sometimes answers the nearest question it knows instead of the one you meant. It always says what it understood when it is guessing. Please press the thumbs-down; it is how we improve it.
* Questions about the future ("will it rain tomorrow", "when will it break") and about other places are declined. Numbers the car does not record (electricity price, speed limits, battery age, cargo weight, road gradient) are never invented.

**Query compiler (1.1)**
* "Last week" / "last month" mean the last 7 / 30 days; "this week" / "this month" are calendar periods. Windows the older routes align to local midnight and the new engine counts back from "now" can differ by a partial day (about 1-2 %).
* The model path needs the LoRA adapter file; without it only the rules compile questions and everything else falls back to the older routes. Follow-ups are handled by rules only (the model sees one question).
* Causal ("why is my range ...") and hypothetical ("does cold weather affect ...") questions about signals stay with the older analysis engines; the new engine explains events only, by reusing their diagnosis.
* Extended signals (torque, steering, wheel speeds, ...) exist in the simulator as physics-consistent derived data; a real vehicle needs them recorded by the ingest daemon (the capability engine reports them as not recorded until then). Older history keeps only per-minute means, so sub-minute dynamics (acceleration, yaw, steering) are not available before the 7-day 1 Hz window.
* A comparison of two selections ("how much more energy did my worst trip use than my best") is not expressible as one query and is left to the older routes. Training and evaluation data are template-generated from the registry vocabulary: the hand-written gold set (`bench/ir_gold.tsv`, 70 questions) is small, so its percentages carry roughly +/- 10 points of uncertainty.
* Training uses a GPU on a development machine; nothing is trained on the vehicle.

**Voice**
* Speech-to-text and spoken answers use the **browser's** features: Chrome or Edge only for input, quality and accent support depend on the browser, and the browser may use an online service for recognition (see PRIVACY.md). Hands-free mode cannot be interrupted by speaking over the answer (the speaker would be heard by the microphone); use Esc, the stop button, the microphone button or just ask the next question.

**Data and analytics**
* Statistics (trends, correlations, change points) are computed on at most four weeks of data per vehicle; small samples give wide uncertainty, and the answers say "not statistically clear" rather than overstate. Counterfactual answers ("if I had driven slower ...") are labelled model estimates.
* Fault diagnosis names the most consistent explanation from the recorded signals - there is no part-level diagnostic data, so "the pump or cooling circuit" is as specific as it can honestly be.

**Performance**
* Typical answers take a few milliseconds to a fraction of a second on a laptop. Heavy statistical questions can take several seconds; on the emulated 2 GB / 2-core ARM target a few took 10-16 s. The first question that needs the language model loads it (a few seconds).
* Memory: about 150 MB for the app plus ~0.5 GB while the language model is loaded (it is released after two idle minutes).

**Platform**
* Tested on Windows 11 with Python 3.11, Chrome / Edge. macOS / Linux and Firefox should work (without voice input) but are not part of regular testing. The camera panel decodes a recorded dashcam clip; a webcam can be selected but the "faults" it detects (covered lens, mud, blur) are simple image-quality checks, not a safety function.
