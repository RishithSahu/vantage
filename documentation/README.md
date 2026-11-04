# VANTAGE / VehicleGPT: the complete project documentation

Written 2026-10-26. This folder is the one place to read to understand the whole project end to end: what it is, why it is built the way it is, how it works, what was tried and thrown away, what was measured, and what is still weak.

Everything here was written from the project's own records (code, `docs/`, `CHANGELOG.md`, result files, git history and the working notes kept through the project). Where a number comes from a measurement, the file or the date is given. Where something was only argued or planned, it says so. If a statement here and a statement in an older document disagree, this folder is newer; the older document is kept for history.

## What the project is, in three sentences

TATA ELXSI's Teliport Season 4, Case Study 5 asks for an assistant that runs **inside a vehicle**, with no cloud, on about **2 GB of RAM and 2 ARM cores**, that answers natural-language questions about the car's data in **under 2 seconds**, keeps **weeks of history**, and **never interferes with safety-critical functions**. We built it as **Vantage** (code name VehicleGPT): a small language model (Qwen2.5-0.5B with a trained adapter) reads each question into a short fixed-form "meaning frame", and ordinary, checkable code then fetches and calculates the answer from the car's own stored data. The model never writes a number, a query or a command.

## Reading order

| If you want to... | Read |
|---|---|
| Understand the project in 10 minutes | [01 Executive summary](01_executive_summary.md) |
| Know how we got here, day by day, including dead ends | [02 Project history](02_project_history.md) |
| See the whole system and how the parts connect | [03 Architecture](03_architecture.md) |
| Follow one question from the keyboard to the answer | [04 The journey of a question](04_question_journey.md) |
| Understand the data side: protocols, cars, storage, retention | [05 Data, protocols and storage](05_data_protocols_storage.md) |
| Understand the language model, training and why this model | [06 The AI model and its training](06_ai_model_and_training.md) |
| Answer a judge's question on safety, security or privacy | [07 Safety, security and privacy](07_safety_security_privacy.md) |
| Know how fast it is, why, and what we tried to make it faster | [08 Performance and scaling](08_performance_and_scaling.md) |
| See every measured result and how it was measured | [09 Results and evaluation](09_results_and_evaluation.md) |
| See the research behind the decisions and the methods still open | [10 Research](10_research.md) |
| Know the owner's rules and every major decision, with the reason | [11 Decisions and rules](11_decisions_and_rules.md) |
| Run, install, test, deploy or debug it | [12 Operations guide](12_operations_guide.md) |
| Know what is not done and what comes next | [13 Limitations and roadmap](13_limitations_and_roadmap.md) |
| Find which folder or module does what | [14 Repository map](14_repository_map.md) |
| Look up a term | [15 Glossary](15_glossary.md) |

## The diagrams

Two diagrams were drawn for this project. Both are native draw.io files and open directly in draw.io (diagrams.net). They live in `docs/`.

| File | What it shows |
|---|---|
| `docs/VANTAGE_ARCHITECTURE.drawio` | The one-page system architecture, laid out against the case study's own headings and constraints |
| `docs/VANTAGE_PIPELINE.drawio` | The pipeline in eleven bands: what happens, when, why, where, and what comes out |

An older generated architecture page is `docs/architecture.svg` (made from result files by `tools_architecture_svg.py`).

## The most important facts to know before reading further

1. **It is a hybrid, and we say so.** The language model decides only *what a question means*. Code parses typed values (dates, numbers, units), checks types and safety, runs the database, does all arithmetic, and writes the wording. `docs/BOUNDARY.md` lists every stage as "learned" or "code". An earlier claim of "model-only" was an overclaim and was withdrawn on 2026-10-25.
2. **No real vehicle has ever been attached.** Every car is a recording or a simulation replayed as its real wire traffic (49 car models, 25 of them from real open CAN databases). This is the largest gap against the case study and it is stated in every honest summary.
3. **Everything about understanding is measured on question sets we wrote ourselves.** Exact-frame accuracy of the shipped adapter is **90.0 % overall, 91.5 % on older sets and 84.3 % on newer unseen-wording sets** (audit of 2026-10-25). No outside person has been the main tester, and there is no user study.
4. **Latency is good on average but not for every question.** On the Raspberry Pi 4 (2 cores) the median question takes about 1.4 s; about a quarter of questions are still above 2 s, mostly because the Pi's CPU has to read a long prompt. See [08](08_performance_and_scaling.md).
5. **The assistant cannot drive the car.** This is enforced by an automated source scan, a read-only database handle, a receive-only data link and a model whose output is a closed list of words, not by trust in the model. It was also measured on a Pi 4 against a real-time task.
6. **Nothing is uploaded anywhere.** The git repository (tag `v1.3.4`) is local only, at the owner's instruction.

## Status at the time of writing

| Item | State |
|---|---|
| Version | 1.3.4 (tag `v1.3.4`, installer `Vantage-Setup-1.3.4-x64.exe`); later work is uncommitted |
| Shipped model | Qwen2.5-0.5B (Q4_K_M) + adapter `meaning-lora-frame2r12`, int8 ONNX Runtime build `models/ort_frame2r12` |
| Test suite | about 1,770 tests collected (2026-10-25) |
| Cars | 49 models, 4 powertrain families (EV, petrol, diesel, hybrid) |
| Protocols | CAN, CAN-FD, LIN, FlexRay, CANopen, J1939, OBD-II/UDS (ISO-TP), SOME/IP, DoIP, Android VHAL, QNX/Android/Linux logs, NMEA GNSS, camera |
| Platforms run | Windows (installer), Raspberry Pi 4 (systemd service), Android Automotive on Cuttlefish (as a data source) |
| Not run | QNX, Yocto image, hypervisor guest, a real vehicle, a Pi 5, a vehicle HPC, a GPU |
- [16 Native engine](16_native_engine.md) — hand-written int8 C engine (ships, ONNX fallback), Pi/batching numbers, accuracy like for like, and the database speed-up (column copy, 2026-10-27/28)
- [17 Slow questions on the Pi](17_slow_questions_pi.md) — every question over its limit in the 2026-10-28 full pass, grouped by cause, with the fix for each
