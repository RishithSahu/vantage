# Car operating systems: where the app runs and how it reads the car

Rewritten 2026-10-21 after the owner explained that "any OS" means **any car OS** (Linux-based car systems, QNX, Android Automotive ...), not laptops or phones.  An earlier version of this page read it the other way; what was built under that reading is listed in section 9 and is not part of the answer.

## 1. What the case study document says

The case-study PDF (Case Study 5, pages 2-3) never writes "any OS".  It names the operating systems and platforms the solution has to live among, and asks for the reasoning behind the OS choice:

| Where in the PDF | What it says (short quotes) | What it means for the app |
|---|---|---|
| Data collection | data from "CAN / CAN-FD, SOME/IP services, Android Vehicle HAL, QNX Services/logs, ADAS, Camera feeds, and diagnostic logs" | The car has **several operating systems at once** (QNX services, an Android head unit with its Vehicle HAL, Linux or AUTOSAR boxes on the Ethernet backbone).  The assistant has to read all of them. |
| Constraints | "~2 GB RAM, ~2 ARM A5X Cores"; to demonstrate: "Automotive grade arm boards, raspberry pi 4/5 with AI HAT, ARM virtualized environment like cuttlefish, qemu, Amazon AWS Graviton or equivalent" | The assistant has to run on small **ARM Linux** boards and in virtual machines, including **Cuttlefish**, which is Android's virtual car. |
| Constraints | "Vehicle safety-critical functions must remain isolated from the AI system." | Safety functions stay on their own OS (typically QNX or an automotive RTOS); the assistant sits beside them and only reads. |
| Evaluation: Edge Performance | "Scalability from Reference board/Raspberry Pi/Virtual Machine to vehicle HPC." | The same program from a small board to the vehicle's high-performance computer. |
| Expected output | "reasoning for AI model, OS, architecture choices, and communication protocols" | Section 4 is the reasoning for the OS. |

My reading, stated so it can be argued with: "works on any OS" means **the assistant runs on the OS families a modern car is built from, and reads data from each of them**.  It does not mean one binary for every OS: a QNX box, an Android head unit and a Linux HPC are different build targets, and what the app can promise is a small, checked list of what it needs from an OS (section 3).

## 2. The car operating systems and the app's job on each

| Car OS | Typical job in the car | What the app does there | Status |
|---|---|---|---|
| **Linux** (Yocto / Automotive Grade Linux / Ubuntu on the HPC / Raspberry Pi OS / ARM board / Graviton / QEMU guest) | HPC, gateway, infotainment partition, validation rig | **The assistant runs here**, as a service that starts at boot, with its memory and CPU budget enforced by the kernel (`deploy/raspberrypi/install-vantage.sh`, systemd) | **Ran on a Raspberry Pi 4** (aarch64, Debian 13).  Other Linux builds: not run |
| **QNX** (Neutrino 7.1 / 8.0, POSIX) | instrument cluster, ADAS, safety-critical services | **Data source**: its services and `slog2info` logs are read by `vgpt/protocols/logs.py`; running `slog2info -w` piped into `python3 deploy/forwarders/forward_lines.py HOST PORT` sends the lines (standard library only, nothing of the assistant on the QNX box).  **Could also host the assistant**: the code uses no Linux-only call (checked), does not need `/proc`, OpenCV, psutil or FTS5 (checked in section 3) | Log path: the forwarder is **tested** into the collector and parser with the documented `slog2info` line layout (never a real QNX device); assistant on QNX **not run** |
| **Android Automotive OS** (AAOS, including **Cuttlefish**) | head unit, driver display, apps | **Data source**: Vehicle HAL properties are decoded with the real AOSP property ids and units (`vgpt/protocols/vhal.py`).  **Display**: shows the assistant's interface.  The assistant itself belongs in a Linux partition or guest beside it, not inside Android (see below) | Decoder **replayed**; Binder client, Android display app and anything on Cuttlefish **not built / not run** |
| **AUTOSAR Adaptive** (runs on QNX or Linux) | service-oriented ECUs | **Data source**: SOME/IP services and events decoded from the service catalogue (`vgpt/protocols/someip.py`) | **Replayed**; a live Ethernet tap not run |
| **Hypervisor guests** (QNX Hypervisor, Xen, KVM) | separate partitions on one SoC | The assistant as its own guest with a one-way data path; this is the vehicle form of the isolation in `docs/SAFETY_ISOLATION.md` | Argued and partly tested in one host; **no hypervisor run** |

**Why the assistant is not "inside Android".**  Android apps are Kotlin or Java programs with native parts built by the NDK.  The assistant is Python plus llama.cpp.  Putting it inside Android would mean rewriting the glue in Kotlin and building llama.cpp with the NDK, which is a different product.  What a car does today is run such services in a Linux partition and show them on the Android display; the display part is a web view (`docs/ARCHITECTURE.md`: the interface is plain HTML) that must obey Android Automotive's driver-distraction rules.  The interface already locks typing while the car moves and offers suggested-question buttons instead (`restrict`; `tests/test_http_security.py` checks that a typed question gets HTTP 423, a suggested-question button is answered, and nobody can sign in to a service, fleet or validation page while the car moves), which is the behaviour those rules ask for.  Whether it satisfies a particular head unit's certification is not tested.  The wrapper app that hosts the web view was **not written**.

## 3. What the app needs from an operating system

This is the contract; each line is checked by a test that fails if it stops being true.

| Need | Used for | If it is missing | Checked by |
|---|---|---|---|
| Python 3.10+ and **numpy** | everything | does not start (the one hard Python dependency) | `vgpt/doctor.py` |
| **llama.cpp** (built with CMake and a C++ toolchain for that OS, through `llama-cpp-python`) | the language model, which reads every question | the server starts but answers every question with a plain "the model could not be loaded, so I cannot interpret questions": the model **is** the product, there is no rules-only fallback by design (checked by starting it with llama.cpp unavailable) | `doctor`, `tests/test_car_os_floor.py` (runs the real model) |
| SQLite 3 | the history database | does not start | standard library |
| POSIX sockets, threads, a monotonic clock, a writable data folder | the server and the store | does not start | `vgpt/portability.py`, `tests/test_portability.py` |
| *no Linux-only call*: no `epoll`, netlink, `memfd`, `/proc`, `/dev/shm` without a guard | QNX and Android lack them | -- | `tests/test_portability.py` reads every module and fails on an unguarded one; it also proves it can fail |
| SQLite **FTS5** (optional) | fast full-text search of log lines | log search becomes a substring match: same lines for whole words, slower on a very long log | `tests/test_car_os_floor.py` blocks FTS5 and compares both ways |
| **cryptography** (optional) | making a certificate, the encrypted at-rest vault | the link is still TLS with a certificate **the vehicle supplies** (`--tls-cert`, `--tls-key`; its fingerprint is read with the standard library only); at-rest encryption then comes from the vehicle's encrypted file system | `tests/test_car_os_floor.py` serves and pins a supplied certificate with `cryptography` blocked |
| **psutil** (optional) | listing network addresses, memory reading | the memory guard falls back to the C library's peak-memory figure (`vgpt/memuse.py`); where even that is missing the OS's own limit is the only enforcer | `tests/test_car_os_floor.py` |
| **OpenCV** (optional) | the test rig's camera pictures only | camera pictures say "need OpenCV" (HTTP 503); everything else works | `tests/test_car_os_floor.py` |
| cgroups / systemd (optional) | kernel-enforced 2 GB / 2 cores, start at boot | the process's own limit (`--memory-limit-mb`) is the guard; the OS's service manager restarts it | `deploy/raspberrypi/` (ran on the Pi) |

`tests/test_car_os_floor.py` runs the shipped product, with its real model, in a subprocess where OpenCV, cryptography and psutil cannot be imported and SQLite cannot create an FTS5 table.  It still starts, answers a question through the meaning-frame model ("The last start of the coolant-flow drop was on 16 Sep, 14:44 ..."), searches its logs, serves TLS with a supplied certificate whose fingerprint matches what is on the wire, and says in words which pieces are off.  This **simulates** a sparse car OS on a Windows PC; it is not a QNX run.

## 4. Why the assistant itself belongs on Linux (the reasoning the PDF asks for)

1. **Every demonstration target the PDF lists is Linux underneath**: ARM boards, Raspberry Pi, Graviton, QEMU guests, and the Linux host of Cuttlefish.  One OS for the assistant means one build to test.
2. **The 2 GB / 2-core budget can be enforced by the kernel, not promised by the program.**  On the Pi the service runs under systemd with `MemoryMax=2G` and `CPUQuota=200%`; the measured result was 697 MB resident and a 978 MB peak (`docs/VANTAGE_LINK.md`).  QNX has its own mechanism (memory partitions) but there is no tested equivalent here.
3. **Isolation from the safety functions.**  The assistant runs in its own partition or VM and only reads: a CAN controller in listen-only mode, a mirror port or a one-way forwarder (`docs/SAFETY_ISOLATION.md`).  The safety functions stay on QNX or an automotive RTOS where they are today.
4. **The AI toolchain.**  llama.cpp, Python and prebuilt ARM64 packages are mature on Linux.  On QNX they have to be built and verified by the integrator; on Android the NDK route is a rewrite of the glue.
5. **Updates and rollback.**  `deploy/update-release.sh` and a systemd unit give an atomic switch and an automatic rollback with ordinary tools.

## 5. How the car's data reaches the assistant from each OS

| Source (PDF) | Lives on | How it gets in | Status |
|---|---|---|---|
| CAN / CAN-FD | the buses of any OS | Linux: SocketCAN, receive-only wrapper, controller in listen-only mode (`vgpt/live/collect.py`); QNX: a reader written against the QNX CAN driver that sends the collector's JSON record (`id`, `data` as hex, `ch`) - **not written** (the format is in `vgpt/live/collect.py`) | SocketCAN with a stand-in socket and candump text; **never a real controller** |
| SOME/IP | Ethernet backbone (AUTOSAR Adaptive, Linux, QNX) | tap or mirror port, decoded with the service catalogue | replayed |
| Android Vehicle HAL | AAOS / Cuttlefish | properties forwarded by a small Android-side agent, decoded by `vgpt/protocols/vhal.py` | replayed; **Binder client not built** |
| QNX services / logs | QNX | `slog2info` lines through `deploy/forwarders/forward_lines.py` (UDP or TCP, standard library only), parsed by `vgpt/protocols/logs.py` (also logcat, syslog, DLT) | forwarder **tested** (`tests/test_forwarders.py`) with the documented line layout; never on a real QNX device |
| Diagnostics (UDS / OBD-II) | any | decoded from captures and diagnostic frames | replayed |
| ADAS, camera | SOME/IP object services, camera ECUs | metadata records; pictures are analysed on the box and only metadata is stored | replayed |
| Anything else | gateway, cloud-less brokers | MQTT 3.1.1 / 5 (QoS 0-2, user/password, TLS), W3C VISS over WebSocket, HTTP polling (`vgpt/live/netcollect.py`); any capture file with `vehiclegpt.py ingest` | tested against the protocol specifications and independent servers, **not against a real vehicle gateway** |

## 6. Putting it on each car OS

| Car OS | Steps |
|---|---|
| Linux (Raspberry Pi OS, Debian, Ubuntu, a Yocto image with Python) | `sh deploy/raspberrypi/install-vantage.sh` (systemd user service, 2 GB / 2 cores); for the vehicle's own certificate add `--tls-cert`/`--tls-key`; the Pi's display runs the interface in a full-screen window (`deploy/raspberrypi/README.md`) |
| Yocto / AGL production image | the same service file; a recipe that packages the code, Python, numpy and llama.cpp **was not written** |
| QNX | expected, **not run**: `python3 vehiclegpt.py vantage --bind <ip> --port 8765 --tls on --tls-cert <pem> --tls-key <pem> --work-dir <ram-backed folder, /dev/shmem> --memory-limit-mb 1950`; the integrator builds Python, numpy and llama.cpp with the QNX toolchain and encrypts the file system.  If the assistant stays on a Linux partition instead, the QNX box only runs the forwarder (`slog2info -w` piped into `python3 forward_lines.py <assistant address> <port>`) |
| Android Automotive | assistant in a Linux guest or partition as above; the head unit shows its web interface through a distraction-optimised wrapper app (**not written**) |
| Container on a hypervisor guest | `deploy/docker/Dockerfile` (written, never built) |

## 7. What has been run and what has not

| What | Status |
|---|---|
| Raspberry Pi 4 (aarch64 Linux) | The Vantage service **ran** there earlier this week, 2 GB limit enforced by the kernel (older code).  Today's code was **not** run on it: the Pi was off.  `deploy/raspberrypi/sync-to-pi.sh` brings it up to date. |
| Windows 11 (the development PC) | **Run**: all tests; the product started with OpenCV, cryptography, psutil and FTS5 made unavailable (the sparse-system test); the portability scan (no Windows-only, POSIX-only or Linux-only call outside a guard) passes. |
| QNX, Android Automotive, Cuttlefish, QEMU, Graviton, a Yocto image, a hypervisor | **Not run.**  `vm/driver.py` is a QEMU ARM64 harness from earlier; QEMU runs are on hold until the owner says. |
| Real vehicle buses | Never connected. |

## 8. What stops this from being a plain "10"

Running it on the other car OSes.  The code is written and checked so that it should run there, and the checks are real (they fail when a Linux-only call or a hard library dependency is introduced), but "should run" is a prediction, not a result.  The cheapest next evidence, in order: the Raspberry Pi (already installed there: sync and start), a Yocto or Debian ARM64 image under QEMU, a Graviton instance, a Cuttlefish guest running the assistant beside Android, and a QNX target.  Each one that passes moves a "not run" above to "run".  Not built at all: the Android display wrapper, a Binder client for the Vehicle HAL, a reader for the QNX CAN driver, a Yocto recipe.

## 9. Things in the folder made under the earlier reading (laptops and phones)

Not part of the car-OS answer; harmless, tested, and can be ignored: `vehiclegpt.py install-app` (a Windows shortcut, a macOS bundle, a Linux menu entry that start the interface in its own window), the fallback to the default browser when no Chromium-family browser exists, and the installable web-app layer (`vgpt/pwa.py`: manifest and service worker).  The Windows shortcut is what puts the Vantage icon on the owner's own PC, which plays the part of the car in the demonstration.
