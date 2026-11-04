# 07. Safety isolation, security and privacy

The case study requires that vehicle safety-critical functions stay isolated from the AI, that processing is local only, and that vehicle and driver data are protected. This page says what was built, how each claim is checked, and what is **not** shown. The authoritative detail is in `docs/SAFETY_ISOLATION.md`, `docs/SAFETY_CASE.md`, `docs/SECURITY.md` and `docs/BOUNDARY.md`.

## 1. The principle

> The assistant has no function with which it could operate the car, and no way to acquire one at run time. It cannot import a CAN library, start a process, open a socket, or write its own database. Its language model writes only words from a closed list, never SQL, a command, a path or a number about the vehicle.

Safety is decided **by structure, not by trust in the model's judgement**. A deterministic safety guard runs before the model so that "should I keep driving with this warning?" never depends on a model's confidence.

## 2. Safety isolation

### 2.1 Two domains

| | Collector domain | Assistant domain |
|---|---|---|
| What | code that touches a network, a socket, a bus or a process: the car-side writer, the receiver, the local server, start-up and health-check tools | the language model, the meaning pipeline, the resolver, the SQL planner and executors, the answer writer, the protocol **decoders** (178 modules) |
| May open a socket or start a process | yes, listed one by one in `vgpt/isolation.py` with a reason | **no** |
| May transmit on a vehicle bus | **no** (the SocketCAN socket is wrapped to receive only) | **no** |
| Database access | the single writer (the runtime) | a handle that is read-only three ways |

### 2.2 The checks (all in `tests/test_isolation.py`, 11 tests, run on every build)

1. **Source scan** (AST based): every module is read; the assistant domain must import no network, process, foreign-function or CAN, serial, USB or Bluetooth facility, and contain no `send` call, no `os.system` or `Popen`, no `eval` or `exec`, no computed `__import__`. The scan is itself tested by planting seven violations in a scratch tree; each is found.
2. **Receive-only bus access**: the SocketCAN collector holds a `ReceiveOnly` wrapper (receive, timeout, close; no send, no `setsockopt`, no way to the raw socket).
3. **Controller listen-only check**: the collector refuses to attach unless the CAN controller is in listen-only mode, in which the hardware neither transmits nor acknowledges even if software were compromised.
4. **The real assistant under observation**: the real model, on a real history, answers ordinary questions and **twelve requests to operate, attack or leak** (disable the ABS, send a CAN frame, override the safety controller, unlock the doors, clear fault codes, flash firmware, "send the GPS history to a web address", and others) while every socket operation and process start is recorded. Result: zero connections, binds, sends and processes; the database file is byte-for-byte identical afterwards.
5. **The database handle cannot write**: `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `VACUUM`, `ATTACH` are refused; even with the SQL-level locks switched off, the file is opened read-only and every write fails.
6. The **Privacy page** runs the scan on the running installation and shows the result.

### 2.3 The link is one way

The car pushes records to Vantage over a local pipe. The only messages Vantage sends back are a **closed list** (`select` a history that exists in the car's catalogue, `mult` replay speed, `stop`), and none reaches a bus. The wire writer exists only in the car's code. (An earlier network version of the link used TLS with a pinned certificate and a signed stream; the HTTP link was retired on 2026-10-21 when the design moved to the local pipe.)

### 2.4 Freedom from interference, measured on a Raspberry Pi 4 (2026-10-23)

`tools_isolation_rig.py`: a stand-in for a safety function (`tools/rt_task.c`: a 100 Hz control loop, 200 µs of work, `SCHED_FIFO` real-time priority, pinned to core 0, deadline 1 ms) runs while the assistant, started as a hardened systemd unit (cores 2 and 3 only, 2 GB, 256 tasks, no network, no CAN, read-only file system, no capabilities, seccomp filter), does the worst it can. Wake-up lateness was recorded for 45 s per scenario.

| Scenario | What the assistant did | Missed deadlines | Worst wake-up delay |
|---|---|---|---|
| baseline | nothing | 0 | 203 µs |
| idle | model loaded | 0 | 169 µs |
| ask | answering as fast as it can | 0 | 168 µs |
| cpu | every allowed core at 100 % | 0 | 308 µs |
| cpu-shared | allowed onto the safety loop's own core | 0 | 287 µs |
| mem | allocating memory until the kernel ends it (restarted 16 times) | 0 | 390 µs |
| fork | fork bomb (stopped at 255 tasks) | 0 | 197 µs |
| io | write and fsync flood (784 MB) | 0 | 258 µs |
| crash | killed every 1 to 2 s, restarted 14 times | 0 | 224 µs |
| attack | every escape attempt below | 0 | 161 µs |

**22 of 22 escape attempts refused** from inside the unit: IPv4, IPv6, raw packet, CAN and netlink sockets; writing the history file, /etc, the home folder and the program folder; seeing other processes; signalling the safety function's pid; taking a real-time priority; moving onto the reserved core; raising its own CPU limit; mounting; rebooting; loading a kernel module; setting the clock; creating a namespace; a child opening a socket. A control (writing its own scratch folder) works, so the refusals are not a broken test. `systemd-analyze security` scores the shipped unit **0.6 (SAFE)**.

### 2.5 What is not shown

* **Not a hardware or hypervisor boundary.** Both domains run on one host under one OS. A bug in the interpreter, the OS or a native library could in principle cross. The vehicle step is the same code split into two partitions or VMs with a hardware data diode; proposed, not built.
* **A safety argument exists; an ISO 26262 safety case does not.** No ASIL was allocated, no HARA or FMEA was done with a manufacturer, no tool was qualified.
* The model can misread a request. Misreading never causes an action (there is no action path) but can produce a wrong *answer*: for example "flash new firmware onto the motor controller" was read as a question about software-update events; the isolation tests assert only that nothing is touched.
* The SocketCAN collector has never run on a real CAN controller. The safety function in the Pi test is a stand-in.

## 3. Offline is a fact about the running process

| Layer | What it does | Evidence |
|---|---|---|
| No network code | no module imports a socket, HTTP, TLS, MQTT, WebSocket, FTP, SMTP or DNS facility; the only socket is the receive-only CAN socket | `vgpt/isolation.py`, `tests/test_isolation.py` |
| In-process guard | a Python audit hook installed on `import vgpt` refuses every socket that is not CAN and every name lookup, for the product and every library it loads; it cannot be removed once installed | `vgpt/offline.py`; on the Pi 0 refused attempts |
| Operating system | `PrivateNetwork=yes`, `RestrictAddressFamilies=AF_UNIX`, `IPAddressDeny=any` in the unit | `deploy/linux/vgpt-assistant.service.in` |
| Nothing downloaded | models, maps and datasets raise a clear error instead of fetching; `HF_HUB_OFFLINE=1` is set by the launcher | `vgpt/datasets.py`, `vgpt/voice.py` |

Measured on the Pi (`bench/pi_security_proof.py`, `results/security_pi.json`, 2026-10-23): the whole assistant, real int8 model and real database, answered questions inside a network namespace with only a down loopback interface; the guard recorded **0 refused attempts**, meaning nothing even tried to reach out.

## 4. Data at rest

| Threat | Control |
|---|---|
| Disk or SD card copied or stolen | the database is stored as `*.db.enc` (AES-256-GCM, chunked; each chunk authenticates its position, so truncation, reordering and edits are detected); the working copy opens in a private RAM folder (`/dev/shm`, mode 0700) and a clear copy never touches storage |
| Key stolen with the card | the key is not next to the data; `VGPT_ATREST_KEY` names a *source*: a TPM-sealed systemd credential, a TPM 2.0 object, the kernel keyring, a secure element command (a key file is the test setup only) |
| Power cut or `kill -9` leaves a working copy | with a key configured, the working copy is in RAM and disappears; otherwise the next start sweeps the folders of dead processes; a normal stop and SIGTERM wipe it (overwrite, then remove) |
| Crash dump or `ptrace` | no core dumps, not dumpable, umask 077, optional `mlockall`; the unit adds `LimitCORE`, `MemorySwapMax=0`, `ProtectProc=invisible` |

On the Pi a world database was sealed (97.5 MB, file starts `VGPTENC1`), the clear file was overwritten and removed, the assistant ran from the sealed file only, and a scan of the sealed folder, `/tmp` and `/dev/shm` for the table names and the SQLite header found nothing.

## 5. What a person, a log line or a pasted document can make it do

| Threat | Control |
|---|---|
| SQL or shell injection through a question | the model's output is masked to ontology words; the database handle is read-only; 17 hostile inputs (SQL, `ATTACH`, `../../etc/shadow`, `${jndi:...}`, a 4,000-character string, NUL bytes, a lone surrogate) all come out harmlessly (`tests/test_hostile_input.py`) |
| A crash from a garbled paste | text is re-encoded before the model reads it |
| Prompt injection in vehicle data (a log line saying "ignore previous instructions") | data never reaches the model's prompt; log text is neutralised (`qa_core.safe_text`) |
| Off-topic or low-confidence readings answered as vehicle questions | calibrated confidence refusal; grounding floor |
| Another website or program talking to the local server | bound to loopback only, DNS-rebinding and cross-site guard, a launch token that becomes a cookie, and every request checked against the route policy |

## 6. Who can see what

* **Roles**: driver (no sign-in; the person in the seat), service engineer, fleet operator, validation team. Roles are access control, not a view. Passcodes are salted scrypt hashes; five failures lock with a doubling wait; idle and drive-away session end.
* **Privacy scrub** (`vgpt/privacy.py`): only the driver role holds personal permissions. Location, saved places and camera pictures are removed from every answer, event and telemetry copy sent to another role, on the server.
* **Route policy** (`vgpt/routepolicy.py`): one table, deny by default. Simulation controls never in a production build, never for the fleet role, never while the vehicle drives with the lock on; study pages only with `--research` and the participant's consent.
* **Audit**: a hash-chained log records who did what (sign-in, export, erase, role pages, voice) and never the content of a question or answer.
* **Records people leave** (thumbs up or down, study lines): cut short (the question to 200 characters), sealed with AES-256-GCM, removed after 30 days (study lines 14 days), erasable at once by the driver.
* **Exports** are signed (HMAC-SHA256) and may be sealed under a passphrase; a tampered pack fails verification.
* **Driving lock** (optional): while the vehicle moves, typing, settings, role changes and privacy actions are refused by the server; answers are shortened and read aloud. Off by default in the test build.

## 7. Honest limits of the security story

* A compromise of the running assistant with the key loaded can read the clear database in RAM. The design shrinks that (read-only, no network, no process start, no dumps) but does not remove it.
* Root on the device can do anything. Secure boot, a locked bootloader and dm-verity belong to the integrator. The Pi 4 has no TPM, so TPM-backed keys were never run on hardware.
* Flash storage does not guarantee an overwritten file is gone; that is why the clear copy lives in RAM.
* The camera path and the speech recogniser have had no independent security review. No third-party penetration test was run.
* The model's reading is not a security control: it can be wrong. Controls are structural.
