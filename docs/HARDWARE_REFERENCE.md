# Hardware reference: what to build it on, what was measured, what is only proposed

The case study asks for a solution that runs in about 2 GB of memory and 2 ARM A5x cores, and says it may be shown on "Automotive grade arm boards, raspberry pi 4/5 with AI HAT, ARM virtualized environment like cuttlefish, qemu, Amazon AWS Graviton or equivalent", and that it should scale "from Reference board/Raspberry Pi/Virtual Machine to vehicle HPC".  This page is the bill of materials and the sizing arithmetic.  Every row says whether it was **measured** on hardware here, **calculated** from measured figures, or only **proposed**.

## 1. What was actually used

| Hardware | What was run on it | Status |
|---|---|---|
| **Raspberry Pi 4 Model B** (aarch64, Debian 13, Python 3.13, 4 cores, 3.7 GB; the service limited to **2 GB and 2 cores** by the kernel) | The Vantage service with the real model, five vehicles streamed one after the other | **Measured** (earlier code): 697 MB resident, peak 978 MB, 19 % of one core streaming with the camera, 0.1-0.5 s for repeated or known wordings, 3.7-6.2 s for new wordings (`docs/VANTAGE_LINK.md`).  **Not measured on the Pi** with the current decoder, which halved the answer time on the laptop |
| Windows 11 laptop, x86-64, 16 GB (the stand-in for the car) | everything, pinned to 2 cores for benchmarks | **Measured**: end-to-end p50 431 ms for a new wording with the current token-mask decoder (`docs/ARCHITECTURE.md` section 6) |
| QEMU ARM64 guest (Cortex-A55 x2, 2048 MB) | a harness exists (`vm/driver.py`); an earlier prompt-building run was stopped after 20 minutes | **Not measured**; QEMU runs are on hold until the owner says.  Emulation timings would not be real timings anyway |
| Raspberry Pi 5, an automotive board, an AI HAT, Graviton, Cuttlefish | nothing | **Not run** |

## 2. Reference rig for a demonstration (proposed, none of it built or bought)

| Part | Choice | Why | Status |
|---|---|---|---|
| Computer | **Raspberry Pi 4 (4 GB) or Pi 5 (4-8 GB)**, aarch64 Raspberry Pi OS, 64-bit | the board the case study names; the Pi 4 is the one measured | Pi 4 **measured**; Pi 5 **proposed** (the `pi5` tier in `deploy/tiers.json`) |
| Storage | industrial-grade microSD or an NVMe SSD, 32 GB or more | about **1 MB per day** of history is kept (measured: 105 days of a hatchback = 92 MB; the 1 Hz tier 7 days, one-minute data 120 days), so capacity is not the issue; **write endurance** is, which is why the service uses SQLite write-ahead logging and why `--encrypt-at-rest` keeps the live database in RAM and writes a snapshot every 5 minutes | **calculated** |
| Model files | 491 MB base (Qwen2.5-0.5B Q4_K_M) + a 35 MB adapter | measured sizes of the files in `models/` | **measured** |
| CAN | a CAN controller that supports **listen-only mode**: for example an MCP2515-based HAT (SPI), or a PEAK or Kvaser USB adapter; check that the driver supports `ip link set can0 type can listen-only on` | the product can refuse to attach unless the controller is listen-only (`SocketCanCollector(require_listen_only=True)`), so that the hardware neither transmits nor acknowledges | **proposed**; the collector was run only against a stand-in socket and candump text |
| Ethernet (SOME/IP, UDS over DoIP) | a 100BASE-T1 to 100BASE-TX media converter or a managed switch with a **mirror port**; the Pi's port receives and never transmits into the vehicle network | the assistant needs a read-only copy | **proposed** |
| GNSS | a USB or UART receiver speaking NMEA | the GNSS decoder reads NMEA (`vgpt/protocols`) | **proposed**; replayed only |
| Display | 7-inch touch display, 1024x600 or 800x480 | the layout was checked in a browser window at 800x480, 1024x600, 1280x720 and 1912x990; touch targets 48-52 px | layout **checked on a PC**, never on that display |
| Microphone | a USB microphone or a small array | voice input runs offline (Vosk, 40 MB model) | **measured on the laptop** (137 MB, 5.8 % of a core); **not on the Pi**; no real-noise error rate |
| Power | 5 V 3 A supply, or an automotive 12 V to 5 V converter with load-dump protection; a supercapacitor or small UPS HAT gives a clean shutdown | the encrypted-at-rest mode loses at most 5 minutes on a power cut; the plain mode relies on SQLite WAL | **proposed**; no power-cut test |
| Accelerator ("AI HAT") | none used | the model runs on the CPU cores through llama.cpp.  A neural accelerator would need the model converted to its format and a different runtime: **not done**, and no claim is made that it would be faster | **not done** |

## 3. Sizing arithmetic (calculated from the measured figures)

| Item | Figure | Source |
|---|---|---|
| Process memory in service | 697 MB resident when streaming, 978 MB peak over five vehicles | Pi measurement |
| Budget | 2048 MB: the service is capped at `MemoryMax=2G`, `MemoryHigh=1800M`, and the process exits itself at 1950 MB (`--memory-limit-mb`) | `deploy/raspberrypi/vehiclegpt-vantage.service.in`, `tests/test_memory_guard.py` |
| Left for the operating system and the rest of the board | about 1.0 GB at the peak (2048 - 978 = 1070 MB; the RAM database of the encrypted-at-rest mode, about 100 MB for 105 days, counts inside the 2 GB) | calculated |
| Voice (if used) | +137 MB, +6 % of a core (laptop) | measured on x86 |
| History | ~1 MB per vehicle-day; 120 days of one-minute data, 7 days of 1 Hz | calculated from a 105-day history of 92 MB |
| Cores | 2 threads for the model (`edge2g`); 4 on a Pi 5 (`pi5`) | `deploy/tiers.json` |
| Time for a new wording | 431 ms p50 on the laptop (x86); 3.7-6.2 s on the Pi 4 with the older grammar decoder; the token-mask decoder took the laptop's parse step from 905 ms to 348 ms; **the Pi figure for it is not measured** | `docs/ARCHITECTURE.md`, `docs/VANTAGE_LINK.md` |

## 4. From the reference board to the vehicle computer

The same code, four resource profiles chosen by `VGPT_TIER` (`deploy/tiers.json`):

| Tier | Machine | Model | Threads / workers | Status |
|---|---|---|---|---|
| `edge2g` | 2 ARM A5x cores, 2 GB (the case study's budget) | Qwen2.5-0.5B Q4 + adapter | 2 / 1 | **measured on a Pi 4 limited to this** |
| `pi5` | 4 x A76, 8 GB | same | 4 / 1 | **proposed** |
| `vm8` | validation rig or VM, 8 vCPU, 16 GB | same | 2 / 3 contexts | laptop only |
| `hpc` | vehicle HPC, 8-16 Cortex-A7x, 16-32 GB | Qwen2.5-1.5B Q4 (**needs its own adapter**: the shipped adapter was trained for the 0.5B) | 4 / 3 | **proposed**; the larger model was not trained for here |

## 5. What is unknown

* Automotive-grade parts: temperature range, vibration, the clean-shutdown behaviour of the specific board.
* Any measurement on a real CAN controller, a mirror port or a real vehicle.
* The Pi 5, the AI HAT, an automotive ARM board, Graviton, Cuttlefish and QNX hardware.
* Power-cut behaviour of the plain (non-encrypted) mode.
* The answer time on the Pi with the current decoder: the next Pi session should run `bench/meaning_eval/perf.py` there (`deploy/raspberrypi/sync-to-pi.sh PI_HOST --adapter --restart` first).
