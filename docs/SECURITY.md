# Security and privacy: offline, and safe even when something goes wrong

Vantage has no network. Everything below is something a test or a measurement on the Raspberry Pi 4 checks; a row with no evidence is marked.

## 1. Offline is a fact about the running process

| Layer | What it does | Evidence |
|---|---|---|
| No network code | No module of the product imports a socket, HTTP, TLS, MQTT, WebSocket, FTP, SMTP or DNS facility. The only socket is a **CAN bus** socket (`vgpt/live/collect.py`, receive-only). The loopback web server that `app.py --web` used to start was removed on 2026-10-23, and `app.py` is now in the scan. | `vgpt/isolation.py` (AST scan of `vgpt/`, `ui/`, `vehiclegpt.py`, `app.py`), `tests/test_isolation.py` |
| In-process guard | A Python audit hook installed on `import vgpt` refuses every socket that is not a CAN socket and every name lookup, for the product and every library it loads. It cannot be removed once installed. | `vgpt/offline.py`; the Pi run below shows `refused_attempts: 0` |
| Operating system | `PrivateNetwork=yes`, `RestrictAddressFamilies=AF_UNIX`, `IPAddressDeny=any` in the systemd unit. | `deploy/linux/vgpt-assistant.service.in` |
| Nothing is downloaded | Models, maps and weather come from the folder or not at all; the language model, speech recogniser and datasets raise a clear error instead of fetching. `HF_HUB_OFFLINE=1` is set by the launcher. | `vgpt/datasets.py`, `vgpt/voice.py`, `vehiclegpt.py` |

**Measured on the Pi (`bench/pi_security_proof.py`, `results/security_pi.json`, 2026-10-23):** the whole assistant, with the real int8 model and a real database, answered questions inside a Linux network namespace that has only a down loopback interface (`unshare -n`): "how fast am I going", "what was the average battery temperature last week" (21.2 degC, 1.0 s) and an off-topic question (refused, 2.9 s). The guard recorded 0 refused attempts: nothing even tried to reach out.

## 2. Data at rest

| Threat | Control | Evidence |
|---|---|---|
| SD card or disk copied, stolen device, a backup, a disk image | The vehicle database is stored as `*.db.enc` (AES-256-GCM, chunked; each chunk authenticates its position and whether it is the last, so truncation, reordering and edits are detected). The product opens it into a private folder in **RAM** (`/dev/shm`, mode 0700) and never writes a clear copy to the storage. | `vgpt/atrest.py`, `vgpt/runtime_vault.py`, `tests/test_atrest.py`, `tests/test_runtime_vault.py` (8) |
| Key stolen with the card | The key is not next to the data. `VGPT_ATREST_KEY` names a *source*: a systemd credential sealed to the TPM, a TPM 2.0 object, the kernel keyring, or a command (secure element). A key file is the test setup only. | `vgpt/atrest.py` docstring; only the file source runs against real storage in the tests |
| Power cut or `kill -9` leaves a working copy | With a key configured the working copy is in RAM and disappears with the power. Without a key it is in the temp folder; every run's folder is named with its pid and **the next start sweeps the folders of dead processes**. A normal stop and `SIGTERM` wipe it (overwrite, then remove). | `RV.sweep_stale`, `hardening.harden()` (SIGTERM runs the exit handlers), `tests/test_runtime_vault.py` |
| Crash dump or `ptrace` reads the database out of memory | `RLIMIT_CORE=0`, `prctl(PR_SET_DUMPABLE, 0)`, umask 077, optional `mlockall` (no swap of the clear database); the unit adds `LimitCORE`, `MemorySwapMax=0`, `ProtectProc=invisible`. | `vgpt/hardening.py`; Pi readback: `Max core file size 0` |

**Measured on the Pi (second run, after two fixes):** a world database was sealed (97.5 MB, file starts `VGPTENC1`), the plaintext was overwritten and removed, the assistant was started from the sealed file only with the key from a 0600 file, answered three questions, and exited. A scan of the sealed folder, `/tmp` and `/dev/shm` for the database's table names and the SQLite header found **nothing** (`files_containing_them_outside_the_key: []`). The first run of the same test had found the database in two `/tmp/vehiclegpt_*` folders that earlier, killed benchmark runs had left behind (no key was configured for those runs) and an empty RAM folder left by the sealing tool; the pid-named folders with a start-up sweep, the SIGTERM handler and the tool's cleanup are the fixes.

## 3. What a person (or a log line, or a pasted document) can make the assistant do

| Threat | Control | Evidence |
|---|---|---|
| SQL or shell injection through a question | The language model's output is masked token by token to the ontology's words: eight words from closed lists, never SQL, a path or a command. The assistant's database handle is `mode=ro`. | `tests/test_hostile_input.py`: 17 hostile inputs (SQL, `ATTACH`, `../../etc/shadow`, `${jndi:...}`, a 4,000-character string, NUL bytes, a lone surrogate) all come out as closed frames; every write statement on the handle raises |
| A crash from a garbled paste | Text is re-encoded before the model reads it (a lone surrogate used to raise `UnicodeEncodeError` inside the request). | `tests/test_hostile_input.py` |
| A prompt-injection in vehicle data (a log line that says "ignore previous instructions") | Data never reaches the model's prompt: the model reads the question and, at most, the previous frame. Log text is neutralised before it is shown (`qa_core.safe_text`). | `docs/SAFETY_CASE.md` |
| Off-topic or low-confidence readings answered as if they were vehicle questions | The model's confidence is calibrated. Below 0.8 the assistant asks "did you mean 1 or 2" with the model's own two readings; a why / define / how-to reading below 0.5 whose question shares no word with the vehicle's vocabulary is refused as not about the vehicle. | `docs/ARM_PROOF.md`, `tests/test_grounding_floor.py`, `bench/clarify_eval.py` |

## 4. Who can see what

Roles (driver, workshop, fleet, validation) with salted-hash passcodes, five-strikes lock with doubling wait, idle and drive-away session end, and a **hash-chained audit trail** that records what happened and never the content (`vgpt/access.py`, `tests/test_roles_privacy.py`, `tests/test_role_walkthroughs.py`). Only the driver role holds the personal permissions: location, saved places and camera pictures are removed from every answer, event and telemetry copy sent to another role (`vgpt/privacy.py`). Evidence packs that leave the head unit are signed (HMAC-SHA256) and can be sealed with a passphrase (AES-256-GCM, scrypt) (`vgpt/securestore.py`).

## 4b. Which role may call which route (deny by default), and the records people leave

Every request the interface answers is first checked against one table, `vgpt/routepolicy.py`: a route that is not in the table does not exist (404).  Each route has a rule - open to any role (the questions, the live state, voice), a permission of
`vgpt/access.py` (service, fleet, validation, audit, erase), a list of roles (the raw feedback export belongs to the service and validation roles only), **sim** (the simulated car and test track: dataset switch, time speed, fault injection, camera source,
reset, the driving lock - never in a production build, never for the fleet role, never while the vehicle drives with the lock on) or **research** (the interpretation-review study: off unless switched on with `--research`, never in a production build,
and every call that records anything needs the participant's consent in the request).  `--production` (or `VGPT_PRODUCTION=1`) is the car's own build.  `tests/test_route_policy.py` checks that every route of the dispatcher is in the table.

The records a person chooses to leave (a thumbs up / down; a line of a study they joined) go through `vgpt/recordstore.py`: cut short (the question to 200 characters), sealed with AES-256-GCM under a key derived from the installation's own secret where the
`cryptography` package exists (said plainly in `status()` where it does not), removed after 30 days (study lines after 14), and erasable at once by the driver (`POST /api/privacy/erase` with `what` = `feedback` or `study`).  There is **no answer cache and no
frame cache** (`tests/test_no_cache.py`): the only de-duplication is of the same request id while it is running.

## 5. The service unit, scored by systemd itself

`systemd-analyze security` on the shipped unit, on the Pi: **exposure 0.6 - SAFE** (scale 0 safest, 10 most exposed). It denies all network, all capabilities, new privileges, namespaces, realtime, kernel tunables/modules/logs, other processes' view (`ProtectProc=invisible`, `PrivatePIDs`), writes outside its state folder, and every privileged system call, and it caps memory (2 GB, no swap), tasks and CPU. A hostile-load rig on the Pi ran 22 of 22 escape attempts to a refusal with a 100 Hz real-time loop missing no deadline (`docs/SAFETY_ISOLATION.md`).

## 6. What this does not protect against (said plainly)

* **A compromise of the running assistant process with the key loaded** can read that process's memory: the clear database is in RAM while the assistant answers. What the design does is shrink that: the assistant can only read (`mode=ro`), cannot reach the network, cannot start processes, cannot be dumped, and sees only what the role allows. Keeping the key out of the process entirely needs a TPM-unsealed credential or a separate key service, which this repository supports through `VGPT_ATREST_KEY` sources but did not run on real TPM hardware (the Pi 4 has none).
* **Root on the device** can do anything, including reading `/dev/shm`. Secure boot, a locked bootloader and dm-verity are the integrator's part.
* **Flash storage** does not guarantee that an overwritten file is gone; that is why the clear copy lives in RAM and not there.
* The camera path and the speech recogniser have no independent security review; nothing in them touches the network.
* No third-party penetration test was run.
