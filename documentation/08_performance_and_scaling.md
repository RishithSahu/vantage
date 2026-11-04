# 08. Performance, latency and scaling

The case study asks for under 2 seconds for common queries, about 2 GB and 2 ARM cores, and scalability from a reference board to a vehicle HPC. This page records every measurement, in the order it was made, and what was learned. **Every latency number says where it was measured.** A laptop core is much faster than a Cortex-A55 or A72, and the early numbers were laptop numbers.

## 1. The reference board and what limits it

Raspberry Pi 4 Model B, Cortex-A72, 4 GB RAM (used as 2 cores and under 2 GB), Raspberry Pi OS (Debian 13), Python 3.13.

* **No dot-product instructions** (the case study's A55 has them). This is the reason llama.cpp's quantised kernels reach only about 11 prompt tokens/s and about 7.5 output tokens/s on this CPU, and why the product uses ONNX Runtime int8 there.
* **Memory bandwidth bound.** A decode pass reads about 358 to 640 MB of weights. Measured bandwidth is about 3.4 to 3.9 GB/s (a floor of about 68 ms per pass at 5.3 GB/s); the matrix multiplications take about 110 to 116 ms per pass, near memory speed.
* A second model context **lowers** throughput (0.46 down to 0.34 questions/s); more threads help latency of the first context (meaning step 2.2 s on 1 core, 1.4 s on 2, 1.1 s on 4).

## 2. The cost model of one question on the Pi

A question's model time is roughly:

`prompt tokens after the cached prefix x 25 to 36 ms  +  decode passes (median 2 to 3) x about 150 ms (+ about 25 ms per extra drafted row)  +  data work`

Median tail is 14 to 16 prompt tokens, of which 44 % are glossary hint lines. This is why the **28 % of questions above 2 s** are mostly the ones with long prompts (about 41 tokens against about 18).

## 3. History of latency, in order

| Date | Where | Measurement | Notes |
|---|---|---|---|
| 2 Oct | QEMU Cortex-A55 | model call 33 to 44 s; 503 analysis questions 91 % under 2 s (p95 2.5 s, slowest 9 s), process peak 788 MB | most questions never reach the model; first boot 538 s |
| 6 Oct | x86, 2 pinned E-cores | static-prompt model path 1.41 s mean (p95 2.16 s); with retrieved examples 3.08 s | rules path 103 ms |
| 8 Oct | x86 | Granite about 83 s for 811 uncached tokens; about 10 tokens/s prefill | long prompts are incompatible with 2 s |
| 18 Oct | x86, 2 cores | meaning frame p50 1.1 s, p95 1.7 s; 679 MB | first design that fits |
| 20 Oct | x86 | p50 745 ms, p95 984 ms | v11 |
| 21 Oct | x86 | the grammar sampler took 62 % of parse time (913 ms with it, 351 ms without) | token mask: 905 to 348 ms per parse; whole answer 431 ms median |
| 21 Oct | **Pi 4** | new wording 3.7 to 6.2 s | first Pi measurement of the meaning route; the old "1 s on the Pi" numbers were laptop numbers |
| 22 Oct | Pi 4 | int8 ONNX Runtime: parse median 1.44 s (was 3.18 s); whole app median 2.19 s, 33 % at most 2 s, 85 % at most 3 s, max 6.7 s (was median 4.7 s, none under 2 s) | same accuracy, no training |
| 23 Oct | Pi 4 | 514 real model runs: median **1.34 s**, p95 2.55 s, 91 over 2 s (before: 1.63 / 2.78 / 129); four cores 1.40 s before the gate | previous-turn gate, guess-ahead fixes, passes 2.27 to 2.15 |
| 23 to 24 Oct | Pi 4 | five protocol cars end to end: medians 0.95 to 1.48 s, worst 2.46 s | |
| 25 Oct | Pi 4 | first testing only on the Pi by car type; slow analyses fixed; petrol 120 questions median 1.58 s; A/B of the new output head 1.49 to 1.42 s median (40 questions); about 25 to 28 % of questions above 2 s | |
| 26 Oct | Pi 4 | tool-stage work (below): over 2 s 26.9 % to 22.1 % with no answer changed | parallel session |

## 4. The decode-pass investigation (25 October, no training)

Goal asked: take the output from about 200 ms per token to about 100 ms.

| Idea | Result on the Pi 4 |
|---|---|
| 4-bit weights (MatMulNBits) | **2.4 times slower** (400 against 166 ms per pass) on the A72; rejected |
| Fuse operators (SimplifiedLayerNormalization etc.) | no gain |
| More threads at 1 row | no gain (168 ms at 1 row); at 9 rows 262 against 380 ms |
| S8S8 variant (a parallel session, same finding) | bit-identical but no faster (149.7 against 146.7 ms) |
| **Output head as rows of a `[tokens, hidden]` matrix** (gather 3,000 candidate rows contiguously instead of 896 scattered bytes per row) | **gather 46.5 ms down to 8.4 ms; a pass from about 166 to about 135 ms**; logits bit-identical; adopted (`models/ort_frame2r12`) |
| End to end | median 1.49 to 1.42 s (about 70 ms): the real mask offers fewer candidates than the 3,000 used in the benchmark |

The per-operator profile of the new graph: `MatMulInteger` 116 ms per pass (97 nodes), Mul 35, Add 9, Reshape 8, Cast 8, Concat 7, Transpose 6. The matrix multiplications are the floor; 100 ms per pass is not reachable with these weights on this CPU. A Pi 5 (dot-product instructions, faster memory) plausibly could.

**Lossless savings that remain:** more drafted rows per pass (3 rows cost about 203 ms for up to 3 tokens, about 68 ms per token if all are accepted); fewer prompt tokens; stopping the first reading of a twice-read question early.

## 5. The tool stage and the over-2-second share (26 October, a parallel session)

Measured on the Pi 4, with the three real 4-year databases and 793 probe questions (petrol, diesel, EV):

| Configuration | Over 2 s | Answers different from original |
|---|---|---|
| Original code | 213 (26.9 %; petrol 27.9 %) | n/a |
| **Default (all fixes below)** | **175 (22.1 %**; petrol 25.0, EV 18.1, diesel 22.8) | **0 of 793** |
| `VGPT_HINT_POLICY=fast` (opt-in) | 77 (9.7 %) | 26 (3.3 %) |

What was slow and what changed:

* **A hidden 1.8 s per question**: a lake view (the four tyre pressures) was aggregated over the whole 1-second history just to answer "is there any data?", on every question, because a new profile was built per question and lost its cache. Reading the answer through the primary key takes 0.4 ms against 0.5 to 2.7 s, with the same result. Digest, changes, anomaly and detect-change routes went from about 1.8 s to about 0.15 s. (Related: known-present columns are now preserved when the profile is rebuilt.)
* **Long-window fuel economy and the engine-temperature trend** read the daily tier: the trend 4.2 s to about 0.1 s; fuel economy 0.4 to 1.2 s to about 0.05 s (sums differ from the old ones only at 1e-14, with identical displayed values).
* **A per-day loop** built a full-length mask per day: bit-identical grouped version; battery-health trend 7.3 s to 3.0 s on the laptop.
* **Questions needing the previous turn are read twice**: the first reading now stops once the outcome is certain: about 200 ms saved, 0 of 268 frames changed, also verified on the Pi's ARM CPU.
* **Prefix reuse for the second reading** is *not* bit-exact on ARM (1 of 268 frames changed on the Pi, 5 confidences changed), so it stays opt-in as `VGPT_CASCADE_REUSE=1`.
* **Hint policy**: the first version was presented as accuracy-neutral and was not; it changed 26 answers (about half improved, half got worse: "how much fuel is left" turned into a clarification). The default now keeps only 8 hint terms that changed no frame on any of 3,389 questions; the 22-term version is opt-in as `fast`.

**What blocks "all below 2 s"** (at the default configuration): 140 of the 175 slow questions (17.7 % of all) spend their time reading a long prompt (median parse 2.4 s); only fewer tokens (which changes answers), a faster CPU, or a cheaper per-token kernel can fix those. 35 (4.4 %) are database-bound, mostly vehicle-health and EV compare/detect routes that load millions of minute rows into Python; answering them from a tier would change their data-quality wording.

## 6. Memory

| Measurement | Value |
|---|---|
| Whole assistant on the Pi (RSS) | about 690 to 750 MB; peak 978 MB (cgroup) over five vehicles; kernel limit 2 GiB |
| Receiver process alone | about 37 to 52 MB |
| Model weights (int8 ONNX) | about 640 MB (`model.onnx.data` 635 MB) |
| First model context | about 0.6 GB; each additional context about 1.1 GB (the LoRA is applied per context), which is why the 2 GB board has one |
| Cuttlefish guest + host | about 2.8 GB of 3.7 GB: the guest and the assistant do not fit together, so they are not run together |

## 7. Ingest and history performance (laptop, `tools_scaling.py`)

* **Ingest**: one runtime process decodes and stores 32.9 k messages/s; a vehicle sends about 87 per simulated second, so one process could serve about 380 vehicles' real-time traffic. 8.5 CPU-seconds per simulated hour of driving; decode 18 µs per message. N independent runtimes: 1.76x (2), 3.63x (4), 5.27x (8).
* **History length**: 14 to 105 days: database 60 to 92 MB; the data side of a standard question p50 25 to 47 ms, worst 80 to 376 ms; investigations p50 26 to 39 ms. Long-range history queries (730 to 1,095 days) still had a data-side maximum of 2 to 3 s on the laptop before the later tier work.
* **Hydration of the archive**: one day of the lake 127 ms to decode (archive codec v2), faster than the old format.

## 8. Scalability from a board to a vehicle HPC

`vgpt/scale/`: **probe** (read cores, memory, ISA, accelerators from the OS, overridable) then **plan** (one rule set from a 2-core board to a many-core machine, a reason for every number) then **calibrate** (measure how many cores one context still gains from on this machine) then **serve** and **autoscale** (an elastic model pool; the controller adds a context only if a measured throughput gain of at least 20 % results).

| Finding | Number |
|---|---|
| Threads in one process do not scale; **worker processes do** (the part of an answer that is not the model is Python, which shares one interpreter lock) | laptop 5.2 to 12.8 answers/s from 1 to 6 workers |
| Concurrent people with real model, 2 threads each | 1 / 2 / 4 / 8 sessions: p50 0.75 / 1.2 / 2.4 / 4.6 s; three contexts: 4.2 answers/s at 8 sessions, p50 1.7 s |
| Pi 4 is memory-bandwidth bound | a second context lowered throughput; boards of four cores or fewer get one wide context |
| Fleet hub: 16 different cars into one machine | 1,977 records/s, 52 MB per receiver, 0 drops, a killed receiver restarted without disturbing the rest |
| Tiers | `edge2g`, `pi5`, `vm8`, `hpc` (see [03](03_architecture.md)) |

**Not measured:** any real vehicle HPC, a GPU, a Pi 5, a second ARM board, or a hypervisor guest. The "HPC" figures are a 12-thread laptop standing in.

## 9. Levers still open for the 2-second target

1. **Shorten the prompt** (the main lever): trim glossary hint lines (44 % of tail tokens), under an accuracy gate against the 2026-10-25 baseline (90.0 % overall, 84.3 % on unseen sets). Tested 2026-10-28: first-clause hints save 1.9 tokens on average and would fix a fifth of the 310 model-bound slow questions, but fail one product test, so they stay off.
2. **Fewer decode passes**: drafted rows are cheap (25 ms against 150 ms); improve draft acceptance.
3. **A faster CPU**: a Pi 5 or an A55-class core with dot-product instructions. Untested.
4. **Per-question profiling** to split prompt, passes and data time for the 10 to 15 slowest question shapes. Done 2026-10-27/28 for the data time: see [16](16_native_engine.md) (column copy, `fastcols`, `nearest`; the database part of the slow families is now 0.04 to 0.1 s on the Pi, what remains is Python arithmetic in `diag_range`, `eff_fit` and the cause engine).
5. **Real-car overheads**: live ingest added about 0.5 s and the Android guest about 0.3 s in a combined test; give the feed a lower priority or its own core.
6. *(Needs training approval)* drop a few of the model's 24 layers and re-tune, or shrink the label vocabulary.

Things not to try again: 4-bit weights on the A72, operator fusion, more threads at one token, S8S8 graphs, a frame cache, a question cache, read-ahead.
