# Safety case: why the assistant cannot compromise the vehicle's safety functions

This is a **safety argument with its evidence**, written for an integrator or assessor to challenge.  It is **not** an ISO 26262 or SOTIF work product: no ASIL is allocated, no HARA was done with a vehicle manufacturer, no tool was qualified, and nothing here was assessed by anyone but the author.  What it does give is a claim, the argument for it, and for every step the test or file that backs it, marked **tested**, **argued** or **not shown**.  `docs/SAFETY_ISOLATION.md` has the detail of the isolation checks; this page puts them in one argument and adds what that page does not cover (answers that mislead, resource starvation, distraction).

## 1. The item and its boundary

* **The item**: the assistant (language model, meaning pipeline, executors, answer writer) and the web interface that shows its answers.
* **What goes in**: a read-only *copy* of the vehicle's data (decoded CAN / SOME/IP / VHAL / logs / camera metadata) and the driver's or engineer's question (typed, chosen, or spoken).
* **What comes out**: text and speech shown or played on a display.  **Nothing else.**  No bus frame, no diagnostic request, no actuator command, no file outside its own data folder.
* **What it is not**: a safety function, an ASIL item, or a source of decisions for other functions.  No vehicle function reads its output.

## 2. The claim

> **C0.** Running the assistant, whether it works correctly or fails in any way, introduces no new way for a vehicle's safety-critical functions to be commanded, starved or corrupted; and what it says about safety is conservative: it never says the vehicle is safe, and it says plainly when it does not know.

C0 is deliberately narrower than "the assistant is safe".  It cannot be: a wrong answer is possible, and section 4 says what is done about it.

## 3. The argument

| # | Sub-claim | Why it holds | Evidence | Strength |
|---|---|---|---|---|
| **C1** | **The assistant has no way to command anything** | | | |
| C1.1 | The code that thinks has no way out of the process: no network, no process start, no foreign-function call, no CAN / serial / USB / Bluetooth library | A source scan of every module sorts them into a *collector domain* (named one by one, each with a reason) and the *assistant domain* (everything else, 181 modules when this was written).  The assistant domain may import none of those facilities and may contain no `send`-type call, no `os.system`, no `eval` | `vgpt/isolation.py`; `tests/test_isolation.py` fails the build if the boundary moves without the list being edited on purpose; the scan itself is tested by planting seven different violations (each is found) | **tested** |
| C1.2 | Bus access is receive-only | The SocketCAN collector holds a wrapper with receive, timeout and close, and no send, no `setsockopt`, no way to the raw socket; and may refuse to attach unless the controller is in *listen-only* mode (the hardware then neither transmits nor acknowledges) | `tests/test_isolation.py` (stand-in socket records everything: 5 frames in, nothing out; listen-only parser tested on real `ip -details` output) | **tested** with stand-ins; never a real controller |
| C1.3 | The history cannot be written by the assistant | The database handle is read-only at the file, at the SQL setting and through an authorizer: `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `VACUUM`, `ATTACH` are refused, even after the SQL-level locks are switched off | `tests/test_isolation.py` | **tested** |
| C1.4 | The model cannot write a command, SQL, a path or a number | Its output is a closed list of words (an 8-word frame from lists generated from the signal registry), enforced token by token while it generates; fixed code does everything after that | `vgpt/meaning/interface.py` token mask; `tests/test_token_mask.py` (identical to the grammar sampler on 200/200 questions) | **tested** |
| C1.5 | A request to operate the vehicle does nothing | The real model, on a real history, is asked twelve things (disable the ABS, send a CAN frame, override the safety controller, unlock the doors, clear fault codes, flash firmware, write over UDS, ...) while *every socket operation and process start of the whole run is recorded*: zero connections, zero binds, zero sends, zero processes, database byte-for-byte unchanged | `tests/test_isolation.py` (real-model observation test) | **tested**; 8 of 12 are refused in words, for the other 4 only "nothing is touched" is asserted (model debt: `docs/TRAINING_QUEUE.md`) |
| C1.6 | The data link from the car cannot be used to send commands to the vehicle | The only messages Vantage returns are a closed list (`select` a history, `mult` replay speed, `stop`); none reaches a bus; the car-side program has no transmit path and is in the scan; the link is HMAC-signed and, on the vehicle network, TLS with a pinned certificate | `tests/test_link.py`, `tests/test_link_tls.py` | **tested** |
| C1.7 | Even a compromised interpreter cannot open a CAN socket or reach another process | The hardened unit (`deploy/linux/vgpt-assistant.service.in`): no network namespace, only `AF_UNIX` allowed, no capabilities, seccomp filter, private pid namespace, read-only file system, no real-time priority | **run on a Raspberry Pi 4**: 22 of 22 escape attempts refused, including a CAN socket, writing the vehicle's files, signalling the safety process and moving onto its core (`results/isolation/pi_attack.json`, `docs/SAFETY_ISOLATION.md` section 4); unit text also checked by `tests/test_deploy.py` | **measured on a device** (system-service form) |
| **C2** | **The assistant cannot starve a safety function** | | | |
| C2.1 | Memory is capped by the kernel | `MemoryMax=2G`, `MemorySwapMax=0`, `CPUQuota=200%` in the service; measured on a Raspberry Pi 4: 697 MB resident, 978 MB peak over five vehicles | `docs/VANTAGE_LINK.md`; unit text in `tests/test_deploy.py` | **measured on the Pi** (older code) |
| C2.2 | Where the kernel cannot cap it, the process ends itself | The process reads its own memory every 3 s and exits with code 75 past the limit; the service manager restarts it and the history is on disk; if the memory cannot be read the guard stays quiet and the operating system's limit is the only enforcer | `vehiclegpt.py _memory_guard`; `tests/test_memory_guard.py` (crossing the limit, staying under it, an unreadable reading); `vgpt/memuse.py` for systems without `/proc` | **tested** |
| C2.3 | The assistant's failure does not stop anything else | It is its own process (own partition or VM in a vehicle); no vehicle function waits for it; a crash is a restart | argued from the architecture | **argued**; no hypervisor run |
| C2.4 | On a shared SoC it does not take the CPU a safety function needs | cpuset (cores 2-3 only), CPU quota, memory and task caps, no real-time priority; the safety loop keeps core 0 with a fixed real-time priority | **measured on a Raspberry Pi 4** with a 100 Hz, 1 ms-deadline stand-in loop: 0 missed deadlines in 10 scenarios (answering at full speed, all cores at 100 %, a hog allowed onto the safety core, memory exhaustion, fork bomb, disk flood, repeated crashes, the attack list); worst wake-up delay 390 us (`results/isolation/pi_full.json`) | **tested** (stand-in safety function) |
| **C3** | **Hostile or broken input cannot make the assistant misbehave or leak** | | | |
| C3.1 | Text that arrives from the car (log lines, names) cannot steer the model | Log text is stored as data and neutralised before it can reach an answer; the validator refuses injection in every text field | `tests/test_qc.py` | **tested** |
| C3.2 | The web interface cannot be driven by another page, another host name or an oversized or malformed request, and tells nothing about its insides | 29 attacks over real HTTP: another site's page, DNS rebinding, form posts, oversize / negative / non-numeric lengths, 50 000-deep JSON, request smuggling after a refused POST, path traversal, torn-off requests, a burst of hostile requests; protective headers; cookies `HttpOnly`, `SameSite=Strict`, `Secure` over TLS | `tests/test_http_security.py` | **tested** |
| C3.3 | Garbage in the data does not crash it | random bytes, emoji, scripts, SQL, HTML, huge words, many questions at once | `tests/test_robustness.py`, `tests/test_protocols.py` | **tested** |
| **C4** | **What it says about safety is conservative** | | | |
| C4.1 | A question like "is it safe to keep driving with this warning?" is not left to the model | A deterministic policy runs *before* the model: a decision-about-continued-operation frame plus a hazard word gets a conservative recommendation ("stop safely", "that needs a workshop"), quotes at most the last recorded related event as *data*, never names a failed component, never says the car is safe; retrospective questions ("how many warnings") are not mistaken for decisions | `vgpt/safety_policy.py`; `tests/test_semantic_integrity.py` (positive and negative question sets; no "you can keep driving", no "is fine") | **tested**; the policy is a rule on purpose (a safety decision should not be learned) |
| C4.2 | Missing data is never "fine" | Each component check carries its evidence; a component with no data is *unknown* | `tests/test_roles_privacy.py` | **tested** |
| C4.3 | Answers cite what they rest on | the evidence pack (signed) for any answer; time stamps; the figures come from fixed code over the stored data, not from the model | `tests/test_roles_privacy.py` | **tested** |
| **C5** | **It does not take the driver's attention while the car moves** | | | |
| C5.1 | While driving, typing is locked; only suggested-question buttons are answered; no one can sign in to a service, fleet or validation page | `423` for a typed question, `200` for a button, sign-in refused with the reason | `tests/test_http_security.py` | **tested**; no head-unit certification is claimed |
| **C6** | **An update cannot leave the car without it** | | | |
| C6.1 | A release that does not start healthy is undone | atomic switch, health check, automatic return to the previous release; data and keys live outside the release | `deploy/update-release.sh`; `tests/test_deploy.py` (healthy, unhealthy with rollback, not-a-release, missing archive, data survives, old releases pruned) | **tested** with a stand-in restart; not against systemd |

## 4. Hazards of the answers themselves

The assistant can be wrong.  Wrong answers do not move the vehicle, but a driver may rely on one.  These are the ones considered, what stops each and what remains.  **Severity, exposure and controllability are not rated**: that needs the vehicle's context and is the integrator's step.

| ID | Hazard | Cause | Control in the product | What remains |
|---|---|---|---|---|
| H1 | The driver is told, or understands, that it is safe to continue when it is not | a safety question answered from statistics; the model misreading | C4.1 policy before the model; never says "safe"; names the last recorded event as data | a safety question worded so that neither the policy nor the model recognises it is answered like any other question: the answer is then about the data, not a recommendation.  No measurement of how often |
| H2 | "No data" is read as "nothing wrong" | an unplugged sensor, a missing signal | *unknown* is a state; the Service page shows it as such (C4.2) | the driver page speaks in plain sentences: how well "I have no data for that" is understood is not studied |
| H3 | An old value is read as the current one | a stalled source | the live snapshot carries how old its newest sample is (`age_s`); event answers carry the time of the event; the redundant-source router flags stale and conflicting sources (`SENSOR_CONFLICT`) | that the sentence shown to the driver *always* states the age of a current value was **not checked**; no measure of whether people notice it |
| H4 | A request to *do* something is answered as a request to *tell* | the model reads "flash new firmware" as a question about update events | nothing is operated in any case (C1); the isolation test asserts that | the wording can mislead ("0 software updates"); model debt, not retrained (the model is frozen) |
| H5 | The assistant distracts the driver | typing, long answers | C5.1; short answer first, details behind a tap | no driver-distraction study |
| H6 | The assistant slows or stops another function | memory or CPU use | C2; measured contention test on the Pi (C2.4) | the safety function in the test is a stand-in; the integrator must re-run `tools_isolation_rig.py` with the real one |
| H7 | Personal data leaves the vehicle or reaches a role that should not see it | cloud use, wrong role | no cloud path exists (C1.1); roles scrub place, driver and camera data; audit trail | `PRIVACY.md`; no external penetration test |

## 5. Conditions the integrator must keep true

These are the safety-related application conditions of this item.  If one is not kept, C0 is not supported.

1. The assistant runs in its **own partition, VM or process space**, not inside a safety function's partition.
2. Its data comes through a **read-only path**: a CAN controller in listen-only mode, a mirror port or data diode on the Ethernet backbone, a one-way forwarder for Android / QNX data.  The service runs under the hardened unit or an equivalent mandatory-access policy.
3. **No vehicle function consumes the assistant's output**, and no output is wired to a bus.
4. The display marks the answers as **information**, not an instruction from the vehicle, and enforces the head unit's driver-distraction rules (the product's own typing lock is an aid, not the certification).
5. The memory and CPU budget is **enforced by the operating system** of that partition (cgroups, a QNX memory partition, a hypervisor budget).
6. Updates come through the **integrator's own signed channel**; the product's rollback is a second line.

## 6. What this case does not show

* **No hardware or hypervisor boundary was built or run.**  Both domains run in one process tree on one host: a defect in the interpreter, the operating system or a native library could in principle cross the software boundary.
* **No ASIL allocation and no manufacturer-reviewed HARA / FMEA; no ISO 26262 / SOTIF / ISO 21434 work product** (section 7 is a working failure-mode table of our own, not a work product), and no tool qualification for the language model or the interpreter.
* **The model's behaviour is statistical.**  The tests show what it does on the cases tested, not that it never misreads.  Its output cannot act, which is why C0 still holds; its answers can mislead, which is why section 4 exists.
* **The hardened unit was run on a Raspberry Pi 4** (system-service form; the opt-in user-service form was not).  The listen-only check was never run on a real CAN controller; the receive-only collector was run on a virtual CAN interface in `tests/test_socketcan_vcan.py`.
* **No driver study** (distraction, understanding of "unknown"), **no penetration test by a third party**, **no vehicle**.

## 7. Failure modes of the interference paths (working table)

How the assistant could interfere with a safety function, what stops it, and which measurement shows it. S = severity if it happened (3 = a safety function misses a deadline or receives a wrong command), D = how well it is detected.

| Path | Failure | S | Control | Evidence | Detection |
|---|---|---|---|---|---|
| CPU | the assistant runs all cores flat out | 3 | cpuset keeps it off the safety core; real-time priority for the safety function only; quota | `cpu`, `cpu-shared` scenarios: 0 misses | the rig; production: a deadline monitor in the safety function |
| Memory | the assistant grows until the machine is out of memory | 3 | `MemoryMax`, no swap; the process ends itself past its limit; the unit restarts | `mem` scenario: killed and restarted 16 times, 0 misses | service manager restart count |
| Tasks | a fork bomb | 3 | `TasksMax=256` | `fork` scenario: stopped at 255, 0 misses | EAGAIN in the unit's log |
| Disk | the assistant floods storage | 2 | only its own scratch folder is writable; IO weight 20 | `io` scenario: 784 MB written, 0 misses | free-space monitor |
| Crash loop | the assistant dies repeatedly | 2 | nothing waits for it; restart delay | `crash` scenario: 14 restarts, 0 misses | restart count |
| Bus | the assistant transmits on a vehicle bus | 3 | no CAN or raw socket possible (address-family filter, no capabilities); the collector is receive-only; controller listen-only mode | `attack` scenario, `tests/test_isolation.py`, `tests/test_socketcan_vcan.py` | source scan on every build |
| Signals | the assistant signals or ptraces another process | 3 | private pid namespace (0 other processes visible), seccomp | `attack` scenario | none needed: not possible |
| Priority | the assistant raises its own priority or moves onto the safety core | 3 | no real-time priority, cpuset, `RestrictRealtime` | `attack` scenario | none needed |
| Data | a wrong answer misleads the driver | 2 | section 4 controls | section 4 | not a path to a safety function |

## 8. How an assessor re-runs the evidence

```
python -m unittest tests.test_isolation tests.test_http_security tests.test_deploy tests.test_memory_guard tests.test_semantic_integrity tests.test_car_os_floor tests.test_portability
python vehiclegpt.py doctor
# on the board, with passwordless sudo:
python tools_isolation_rig.py --seconds 45 --out results/isolation/pi_full.json
```

The Privacy page of the running application repeats the isolation scan on the installed code and turns red if the boundary is broken.
