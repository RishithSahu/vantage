# ARM evidence (2026-10-22): what was run, where, and what it proves

The brief asks for about 2 GB of RAM, about 2 ARM cores, under 2 s for common queries, and lists a Raspberry Pi 4/5, Cuttlefish, QEMU and Graviton as acceptable ways to demonstrate.

## 1. Raspberry Pi 4 (real hardware, Cortex-A72, 2 cores pinned with `taskset -c 0,1`, 3.7 GB board)

The language model is unchanged (Qwen2.5-0.5B + the `frame2d` meaning LoRA).  It is run by ONNX Runtime with int8 weights (`tools/build_ort_model.py`, `vgpt/meaning/ort_backend.py`) because the Pi 4's CPU has no dot-product instruction, which is what llama.cpp's quantised kernels need: llama.cpp reached 11 prompt tokens/s and 7.5 output tokens/s on every build and format tried; ONNX Runtime's int8 kernel reaches about 30 prompt tokens/s.

| Measurement | llama.cpp | ONNX Runtime int8 |
|---|---|---|
| Exact frames, 220 frozen-set questions | 90.9 % | 90.9 % (218 of 220 frames identical) |
| Frame parse, median / worst | 3.18 s / 8.08 s | 1.44 s / 2.67 s |
| Whole app, 464 mixed questions, median | 4.7 s | 2.19 s (33 % within 2 s, 85 % within 3 s, 97 % within 4 s) |
| Process memory | about 750 MB | 746 MB resident, 775 MB peak |

### Whole-application latency, re-measured 2026-10-23 (cache-free, 514 questions, `bench/pi_latency.py`)

Every question is a real model run (no question or frame cache). The product model is `frame2w`. The database tier (`agg_1d`), vectorised statistics and question-cue drafting are in.

| Setting | Median | p90 | p95 | p99 | Max | Over 2 s |
|---|---|---|---|---|---|---|
| 2 cores, `ondemand` governor (the spec setting) | 1.63 s | - | 2.78 s | - | 3.72 s | 129 of 514 (25 %) |
| 4 cores, `performance` governor | 1.40 s | 2.24 s | 2.45 s | 3.12 s | 3.60 s | 87 of 514 (17 %) |
| **2 cores, `ondemand`, previous-turn gate + better guesses** (final, ask-back off) | **1.34 s** | - | **2.55 s** | - | 4.36 s | **91 of 514 (18 %)** |
| before this work (2 cores, first 130 questions) | 1.91 s | - | 6.31 s | - | 7.16 s | - |

**What the gate is** (`vgpt/meaning/prevgate.py`, `bench/prev_gate_eval.py`): frame version 2 reads `Previous: <frame of the last question>` after every question, which is about 11 of the 25 tokens read after the cached prefix (32 ms each on the Pi). Measured on the frozen sets, an unrelated previous frame also *hurts*: the frame is right for 90.8 % of questions without it and 88.2 % with it. So the question is read without the line first, and the line is added only when the wording points back ("and", "what about", "which one", "it", "too", a one- or two-word question) or the model's own reading of the bare question is incomplete (ambiguous, no quantity named, p < 0.5). Result on 1,483 frozen-set questions: ordinary questions 89.8 % -> 90.6 % right, the line is shown for 12 % of them; real follow-ups 94.0 % -> 94.0 % (the cue list was completed against the follow-up sets, so that number is not independent of them). On the Pi the median fell from 1.63 s to 1.36 s. `VGPT_PREV=always` restores the old behaviour.

**Guess-ahead fixes (`bench/passes_eval.py`, 398 questions, frames identical 398/398):** when no cue names the concept, the usual concept after that act is guessed ("ambiguous", "unsafe", "social" are followed by `none`), and "the last 14 days" is no longer read as the operation `last`: passes over the weights per question 2.27 -> 2.15 (one pass for 25 % of questions, was 15 %).

**Measured and not kept:** ONNX Runtime session settings (spinning, memory arena, optimisation level, one thread for decode: a decode pass is 141 ms whatever is set, the 24-token prefill 689 ms); reading the words of a question while it is typed (built, exact, removed on the owner's instruction: no read-ahead); raising the number of tokens guessed ahead per pass from 8 to 12 and 16 changed nothing (parse median 1.12 / 1.17 / 1.16 s; the passes are limited by how often the guess is right, not by its length); an embedding re-ranker of the concept (bge-small, local) is right for only 49.9 % of questions alone and would overrule the model's correct answer in 48 % of cases, so it was not built.

Where the time goes (2 cores): reading the question 1.45 s median (prefill 0.83 s for 25 tokens after the cached prefix at 32 ms per token, decode 0.55 s in 2 passes), everything after it 0.11 s median. The tail of questions over 2 s is the "why" questions (cause engine, 0.5-0.8 s on the Pi) and the long ones (40+ tokens). Four cores helped the median by 0.23 s only: the pass over the weights is memory-bound. **Under 2 s for every question is not met**; with the model frozen it needs fewer tokens per question or a cheaper reader, not more tuning. Memory: 880 MB peak resident.

An off-topic "why" question ("why does the Moon have phases") was read by the int8 model as a vehicle signal at p = 0.31 and answered with a tyre-leak cause. A grounding floor (`layers.py: grounding_floor`, `Support` in `vgpt/meaning/cues.py`) now refuses a why / define / how-to reading with p < 0.50 whose question shares no word with the vehicle's vocabulary and carries no previous turn; on the 2,493 stored frame2w readings it changes none (`tests/test_grounding_floor.py`), and on the Pi the five off-topic probes are refused.

Re-run 2026-10-23 night, after the protocol and storage work (Pi 4, 2 cores, real model, ONNX Runtime int8 in `~/vehiclegpt2/.venv`, no question cache; `bench/linked_e2e.py --drive 60`, 8 questions each):

| Car | Buses sent | Recognised | Signals | Median answer | Worst |
|---|---|---|---|---|---|
| aster_ev | CAN, SOME/IP, VHAL, UDS, log, GNSS, camera | decoder file | 41 | 1.33 s | 2.46 s |
| nova_ev | CAN, LIN, SOME/IP, VHAL, UDS/DoIP, log, GNSS, camera | decoder file | 39 | 1.27 s | 2.35 s |
| prestige_gt | CAN, FlexRay, LIN, UDS, log, GNSS | decoder file | 41 | 1.48 s | 2.45 s |
| volta_bus | CAN (incl. CANopen), UDS, log, GNSS | decoder file | 21 | 0.95 s | 1.88 s |
| kestrel_petrol | CAN, UDS, log, GNSS | decoder file | 31 | 1.15 s | 2.02 s |

Every car was recognised in 2 to 3 s from its raw traffic. The table below is the earlier run (before the gate and the guess-ahead fixes); it is kept for the other cars, which were not re-run.

Cars to answers, on the Pi, through the local pipe (`bench/linked_e2e.py`; each car drives 60 s of its own clock, then 8 questions):

| Car | Buses sent | Recognised | Signals | Median answer | Worst |
|---|---|---|---|---|---|
| aster_ev | CAN, SOME/IP, VHAL, UDS, log, GNSS, camera | decoder file | 41 | 2.60 s | 3.48 s |
| aaos_reference_ev | VHAL, log, GNSS, camera | learned from the standards | 13 | 2.25 s | 3.01 s |
| rhino_diesel | CAN, UDS, log, GNSS | decoder file | 30 | 2.08 s | 2.88 s |
| heavy_truck_j1939 | CAN (J1939), UDS | learned from the standards | 20 | 1.74 s | 2.37 s |
| generic_obd_dongle | OBD-II/UDS, log, GNSS | learned from the standards | 17 | 1.93 s | 2.58 s |
| proprietary_demo | proprietary CAN | decoder file | 15 | 1.89 s | 2.60 s |
| meridian_ev | CAN, SOME/IP, VHAL, UDS, camera | decoder file | 37 | 2.26 s | 3.13 s |
| tesla_model_3 | CAN, UDS, camera | decoder file | 29 | 3.12 s | 3.97 s |

Not met: the brief's under-2 s is met by some of these medians and not others.  A new wording costs about 1 s to read the question (34 tokens when a previous question is carried), about 0.8 s to write the frame, and 0.2-0.5 s to read the data.  Getting every question under 2 s needs fewer tokens per question (a shorter prompt format), which needs retraining, or a smaller reader in front of the model.

## 2. Emulated ARM guest on the development laptop (QEMU 11.1, `-cpu cortex-a55 -smp 2 -m 2048`, Debian 13 arm64)

An x86 laptop cannot run an ARM guest with hardware acceleration, so this is software emulation (TCG).  What it shows and what it does not:

* Shows: the same ARM programs and the same int8 model run in a 2-CPU, 2 GB Cortex-A55 guest (`asimddp` present); aster_ev (CAN + SOME/IP + VHAL + UDS), aaos_reference_ev (VHAL) and rhino_diesel (CAN + UDS) were driven through the pipe, were recognised (decoder file, learned from the standards, decoder file), 41 / 13 / 30 signals were learned, and every question was answered.  Peak memory 746 MB (limit 2 GB).
* Does not show: speed.  Answers took a median of 9.7-11 s in the guest, which is the emulator's speed (about 5-7 times slower than the real Pi), not an A55's.  No latency claim for an A55 is made from it.
* Cuttlefish does not run in this guest (it needs hardware virtualisation, which an emulated guest does not have).  Cuttlefish was run on the Pi under KVM (`docs/CAR_AND_VANTAGE.md`).

## 3. What would settle the A55 question

Run `bench/linked_e2e.py` and `bench/pi_latency.py` on a board with real A55 cores (for example a Rockchip RK3566 board with four A55, pinned to two with `taskset`), with onnxruntime installed.  Those cores have the dot-product instruction, so both ONNX Runtime and llama.cpp have faster kernels there than on the Pi 4; no measurement of that exists yet.

## Known answer gap seen in these runs

"What was my highest speed in this drive" returns "nothing to rank" for every car: the model reads it correctly (`extreme speed max ... trip`), but the drive in progress is not yet a completed trip in the ledger, so there is nothing to rank.
