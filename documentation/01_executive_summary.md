# 01. Executive summary

## The brief

Case Study 5 (Teliport S4, TATA ELXSI), "Natural Language Vehicle Information using On-Device SLM/LLM (VehicleGPT)". A vehicle produces huge streams of data (CAN, SOME/IP, Android VHAL, QNX logs, ADAS, cameras, diagnostics). The task is to design a practical AI-on-edge system that runs a small language model in the vehicle, consumes that data continuously, and lets a driver or service engineer ask questions such as:

* "Why has my driving range reduced over the last two weeks?"
* "When did fuel efficiency show the largest drop?"
* "Show all ADAS disengagement events from yesterday."
* "What happened before the battery overheating warning?"
* "Compare energy consumption between city and highway driving."

**Constraints the case study sets:** no cloud dependency; about 2 GB RAM and about 2 ARM A5x cores (a Raspberry Pi 4/5, Cuttlefish, QEMU or AWS Graviton may be used to demonstrate); under 2 seconds for common queries; weeks of searchable history; safety-critical functions isolated from the AI.

**Expected output:** a high-level architecture with hardware and software components, engineering justification, the data collection strategy, the model rationale, the search and retrieval workflow, the historical storage approach, the edge deployment architecture and the user interaction design.

**Judged on:** system architecture, AI capability (understanding and accuracy), data integration, search and analytics (events, trends, anomalies, **explaining causes**), edge performance (memory, latency, **scalability from reference board to vehicle HPC**), privacy and security, and innovation and practicality (value for drivers, service engineers, fleet operators, validation teams).

## What we built

**Vantage** is an on-device vehicle assistant in two cooperating programs:

* **The Car** (a separate program): a simulated or recorded vehicle that puts its data on a local pipe exactly the way a real vehicle's buses carry it (raw CAN frames, SOME/IP messages, Android VHAL parcels, diagnostics, log lines, GPS sentences, camera pictures). In a real car this program would not exist; the buses would simply be there.
* **Vantage** (the assistant): it listens to that pipe, works out by itself what car it is talking to, stores everything in a tiered database, and answers questions about it.

Inside Vantage, a question travels through this path:

1. **Guards first.** A deterministic safety check, typo and shorthand repair, and a handful of questions that need no model at all (who are you, what time is it, what is outside the car's data).
2. **The model reads the question.** Qwen2.5-0.5B with a LoRA adapter, run as int8 on ONNX Runtime, writes a fixed-form 8-word "meaning frame" (act, concept, operator, condition, comparison, entity, second concept, reference). A token mask lets it write only words from the vehicle ontology: never SQL, never a number about the car, never a command.
3. **Trust checks.** The model's confidence is calibrated; below 0.50 it says "I will not guess".
4. **Code calculates.** A resolver turns the frame into a typed, validated query; the only SQL in the system is built from fixed templates and run on a read-only handle; executors do the statistics, trends, event counts, trips, charging, fuel economy, correlations and summaries. A cause engine explains "why", and a separate engine runs 32 deterministic analyses for long multi-part requests.
5. **Words and checks.** Templates write the answer from the numbers; a verifier rejects any number the evidence does not hold; privacy and role rules decide who sees what.

## Why this design

* A 0.5 B model cannot be trusted to remember numbers or write queries, but it can be trained to *classify what is asked* into a closed vocabulary. That split keeps answers correct by construction and keeps memory small (about 640 MB of int8 weights).
* Free-form generation is replaced by a closed output: this is what makes hostile input harmless and the decode short enough for a Raspberry Pi.
* Everything the model does not do is plain code, which is testable against independent implementations (SQL, numpy, cantools, scapy, can-j1939 and others).

## Headline results (all measured; see [09](09_results_and_evaluation.md))

| Area | Result |
|---|---|
| Understanding (exact 8-word frame, product adapter r12, 3,842 questions in 38 sets) | **90.0 %** overall; 91.5 % on the 3,038 older questions; 84.3 % on 804 newer unseen-wording questions |
| Honesty about confidence | questions the model is at least 95 % sure of are right 95.7 % of the time; below 0.50 confidence it refuses |
| Latency on Raspberry Pi 4 (2 cores) | median about 1.4 s end to end; about a quarter of questions still above 2 s (long prompts) |
| Memory on Raspberry Pi 4 | about 700 MB resident; kernel cap 2 GB |
| History | seconds 14 days, minutes 5 years, daily forever; lossless archive; 6 GB budget |
| Protocols | nine families, each cross-checked against an independent implementation |
| Explaining causes (sealed test, 74 scenarios) | first answer right 87.8 %, in top three 95.9 %, 0 % confident false causes on negative controls |
| Safety isolation (Raspberry Pi 4, real kernel) | a 100 Hz / 1 ms-deadline task missed 0 deadlines in 10 stress scenarios (worst wake-up 390 µs); 22 of 22 escape attempts refused |
| Scaling | worker processes scale 5.2 to 12.8 answers/s from 1 to 6 workers on a 12-thread laptop; 16 cars into one hub at 1,977 records/s, no drops |
| Own scorecard against the case study's criteria (edge performance excluded, 2026-10-21) | 8.65 / 10, with the model rows deliberately held at 7.5 |

## What it honestly lacks

* No real vehicle, QNX, hypervisor or user study. Every car is a simulation or recording.
* Understanding is measured on questions our own team wrote; the one outside-authored set (500 questions) has no gold labels.
* 84 % exact on unseen wording means roughly one question in six is read differently from what was intended; 58 % of those errors are made with high confidence.
* The 2-second target is met on average on the Pi 4 but not for every question (about 22 to 27 % are above 2 s depending on the configuration).

## What to look at first (for a reviewer)

1. `docs/VANTAGE_ARCHITECTURE.drawio` and `docs/VANTAGE_PIPELINE.drawio`
2. [03 Architecture](03_architecture.md) then [04 The journey of a question](04_question_journey.md)
3. [09 Results](09_results_and_evaluation.md) and [13 Limitations](13_limitations_and_roadmap.md): read the second before relying on the first.
