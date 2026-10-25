# Scalability: from a 2-core board to a vehicle HPC

One build, one code path, any machine. The plan for a machine is computed from what it is, then checked by measurement, and corrected while running.

## 1. The mechanism (`vgpt/scale/`)

| Stage | Module | What it does |
|---|---|---|
| Probe | `probe.py` | cores (honouring CPU affinity and container limits), RAM, CPU instruction set (NEON, dot-product, AVX2, AVX-VNNI), GPU, free disk; every value can be overridden |
| Plan | `plan.py` | from the probe: which backend (int8 ONNX Runtime, GPU graph if one is installed, llama.cpp), how many model contexts and threads, the people served at once, how much history is kept at one-second and one-minute resolution, database cache, ingest workers, camera frame rate, what to give up under overload; every decision records its reason (`Plan.explain()`) |
| Calibrate | `calibrate.py` | measures how many threads one context can use before it stops gaining on *this* machine |
| Serve | `procpool.py`, `service.py` | one worker **process** per model context (priority queue: driver, then service, then fleet, then batch; sticky sessions; a crashed worker is replaced and its question retried) |
| Autoscale | `autoscale.py` | adds a context when the 95th-percentile wait threatens the 2 s target, **measures whether the new context really raised throughput and takes it away again if it did not** (scale-out is then capped for that machine), releases contexts when idle or when memory is tight, and when no more contexts can be added gives things up in a fixed order: pause background analytics, lower the camera analysis rate, queue service and fleet questions, refuse non-driver questions, shorten answers. A driver's question is never shed |
| Fleet | `fleet.py` | one receiver process per vehicle (its own pipe, database and failure domain), restarted if it dies |

## 2. What was measured, and what it taught

Threads inside one Python process do **not** scale: the interpreter lock and spinning ONNX threads serialise the work. Worker processes do. On the 12-thread laptop (the HPC stand-in), 12 people asking at once:

| worker processes | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| answers per second | 5.2 | 9.0 | 12.0 | 12.8 |
| median wait | 2.26 s | 1.33 s | 0.98 s | 0.87 s |

(`results/scaling/laptop_procs.json`). The plan therefore uses processes.

The Raspberry Pi 4 behaves differently, and the system found that out by measuring (`results/scaling/pi_*.json`):

| cores given to one context | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| meaning step, median | 2.21 s | 1.41 s | 1.25 s | 1.13 s |

A second context on the same Pi **lowered** throughput (0.46 → 0.34 answers/s with two askers): the model's weights stream through one memory bus (about 640 MB of int8 weights per decoding pass; the Pi 4's memory bandwidth is the limit), so extra contexts only share it. For boards of four cores or fewer the plan therefore gives *all* AI cores to one context, which is what lowers latency; the autoscaler's self-check is the safety net for any machine where the plan's assumption is wrong.

Many vehicles into one machine (`bench/fleet_scale.py`, `results/scaling/laptop_fleet.json`): 1, 2, 4, 8, 16 different cars (16 different wirings: CAN, CAN-FD, J1939, OBD-II, SOME/IP, Android VHAL, logs, GNSS) each streaming in real time into its own receiver:

| cars | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| records stored per second | 88 | 180 | 400 | 691 | 1977 |
| memory of all receivers | 50 MB | 97 MB | 202 MB | 407 MB | 837 MB |
| recognised with no setup | 1 | 2 | 4 | 8 | 16 |
| records dropped | 0 | 0 | 0 | 0 | 0 |

Memory is about 52 MB per vehicle and throughput is linear in the number of vehicles. Killing one receiver: the hub restarted it and the other receivers kept growing their databases (`after_kill` in the result file).

## 3. The plan for four machines (same code)

Printed by `tools_architecture_svg.py` from `make_plan`: in-car board (2 cores, 2 GB): 1 context × 2 threads; Raspberry Pi 4: 1 × 3; 12-thread laptop: 2 contexts growing to 5; 32-core 64 GB vehicle HPC with 16 cars: 7 contexts growing to 15 and 14 ingest workers. The last row is a plan, not a measurement: no machine of that size was available.

## 4. Verified by tests

`tests/test_autoscale.py` (25: probe, plan grid never overcommits cores or memory, more hardware never gives less, decisions, hysteresis, closed loop under load, the verification and cap), `tests/test_procpool.py` (8: pool, crash retry, sticky sessions, priority), both pass on the laptop and on the Pi.

## 5. Not shown

* No run on a machine with 32 or more cores, or with a GPU/NPU: the GPU tier exists in the plan and is not measured.
* The 2 s target is not met on the Pi for every question (median ≈ 2 s; see `docs/ARM_PROOF.md`); the scaler cannot make one question faster than the machine's single-question latency, only keep the wait from growing.
* Long histories: data-side time grows with the span asked about (`results/scaling/laptop_history.json`).
