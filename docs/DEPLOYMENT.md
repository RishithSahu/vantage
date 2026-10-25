# Deploying VehicleGPT / Vantage in a software-defined vehicle

One application, four resource profiles (`deploy/tiers.json`), three ways to run it.  This page says what to buy, what to type, how an update is applied (and undone), and what has and has not been run.

## 1. Reference hardware

| | Reference board (the case study's budget) | Larger board | Validation rig / VM | Vehicle HPC |
|---|---|---|---|---|
| tier name | `edge2g` | `pi5` | `vm8` | `hpc` |
| CPU / RAM | 2 ARM cores (A53/A72 class) / **2 GB** | 4 x A76 / 8 GB | 8 vCPU / 16 GB | 8-16 A7x / 16-32 GB |
| storage for weeks of history | **about 1 MB per day** all tiers included (measured: 105 days of the hatchback = 92 MB; the 1 Hz tier is kept 7 days, one-minute data 120 days), so 32 GB flash is more than enough; leave 4 GB for the model files | same | same, 12 sessions | same, 60 days of 1 Hz |
| model | Qwen2.5-0.5B Q4_K_M + the meaning adapter, 2 threads | same, 4 threads | same, 3 contexts | same or 1.5B (needs its own adapter) |
| measured here | **yes**, on a Raspberry Pi 4 (2 GB cgroup limit): 697 MB resident, peak 978 MB, 0.1-0.5 s for known wordings, 3.7-6.2 s for new ones (latency is the open item, see the scorecard) | no | laptop only | no |

Gateway: the assistant needs only a **read-only copy** of the vehicle's traffic.  Best: a CAN controller in listen-only mode (`ip link set can0 type can listen-only on`; the SocketCAN collector can refuse to start otherwise,
`docs/SAFETY_ISOLATION.md`), or a mirror port / diode on the Ethernet backbone for SOME/IP; Android Automotive offers the Vehicle HAL through a Binder client; QNX/Linux logs arrive over DLT, syslog or `slog2info`.
Any of these can be fed to Vantage by a small forwarder (`UdpSender`, `candump -L | nc`); and a capture file can be read directly (`docs/DATA_SOURCES.md`).

## 2. Three ways to run it

| Mode | Command | Use |
|---|---|---|
| **Vantage service** (the car's display runs a browser against it) | `sh deploy/raspberrypi/install-vantage.sh [--encrypt-at-rest] [--enable-cgroups]` | the vehicle: a systemd user service, 2 GB / 2 cores enforced by the kernel, restarts itself, starts at boot |
| **Single process** (all in one, on a laptop or a validation rig) | `python vehiclegpt.py run` (or `Start-VehicleGPT.bat`) | development, workshop laptop, validation |
| **Container** | `deploy/docker/Dockerfile` | an SDV with a container runtime on a hypervisor guest.  **Written, not built or run here** (no Docker on the development machine); the application has no dependency a container hides |

`--harden` also restricts what the service may do (a read-only file system apart from its data folder, IP sockets only, no privileged system call): `deploy/raspberrypi/vehiclegpt-vantage.hardening.in`, **not run on a device**.  Which operating systems the app is meant for, and what it does on each, is `docs/PLATFORMS.md`; the bill of materials and sizing is `docs/HARDWARE_REFERENCE.md`; why it cannot compromise the vehicle's safety functions is `docs/SAFETY_CASE.md`.

What the installer does (and prints, step by step): checks Python, memory and the kernel's memory controller; creates the environment; makes the link key (the one secret shared with the car) and, if asked, the at-rest
key; writes the service from a template; waits until the port listens.  `--dry-run` prints without changing anything.  It is run for real by `tests/test_deploy.py` on the development PC (22 tests, with stand-ins for `systemctl`, `loginctl` and `ss`: it writes the service and the keys, is idempotent, refuses to start when the at-rest key cannot be made, removes the service and keeps the data); **its first real run on a Pi is still to
be done** (the Pi was switched off while this was written).

## 3. Data from a car nobody configured

* A car with a profile pack (the four shipped ones) is plug-and-play.
* Any other car: `python vehiclegpt.py ingest <capture-file>` reads candump/ASC/TRC/BLF/CSV/JSON/OpenXC/VSS/ELM327/NMEA/logs, works out what each signal is from the standards (OBD-II PIDs, SAE J1939 SPNs, COVESA VSS, Android VHAL names) and from the signal's own name, units and values, and writes a profile pack and a database; the car side can play the file as a vehicle (`data/captures/`).  A signal it cannot name with confidence is **listed and left out**, never guessed (`docs/DATA_SOURCES.md`).

## 4. Updating a deployed device, and going back

`sh deploy/update-release.sh release.tar.gz` unpacks the new code beside the old one (`releases/<time>`), links the shared folders (models, data, the Python environment) into it, switches `current` atomically, restarts
the service and waits for the port; **if the new release does not come up healthy, it switches back to the previous one and restarts it** (exit code 1).  The last three releases are kept.  Vehicle data and keys are
never inside a release, so neither an update nor a rollback can lose them.  Tested automatically on the development PC (`tests/test_deploy.py`) with a healthy release, an unhealthy release (rollback verified), an archive that is not a release, a missing archive, data surviving updates and rollbacks, and old releases pruned; not yet run against a systemd service.
An over-the-air channel (signature check of the archive, staged rollout) is **not built**; the archive is trusted because the integrator delivers it.

From a development computer, the quick way to bring a device that is already installed up to date is `sh deploy/raspberrypi/sync-to-pi.sh PI_HOST --adapter --restart` (code, and the adopted adapter named in `models/product.json`; key-based ssh; `--dry-run` prints
what would happen).  The Raspberry Pi of this project holds the code of 2026-10-21 morning (before the TLS link, the token-mask decoder and frame 2) until this is run; the Pi was off when the work was done.

## 5. Protecting the data on the device

* **In transit**: the car-to-Vantage link is TLS with a self-signed certificate pinned under the shared key, so a man in the middle with his own certificate is refused (`tests/test_link_tls.py`).
* **At rest**: `--encrypt-at-rest` keeps the database in RAM (`/dev/shm/vantage`; about 100 MB for 105 days, counted in the 2 GB) and writes AES-256-GCM snapshots every 5 minutes and at stop; pulling the SD card or the power leaves
  ciphertext and at most five minutes of lost data (`tests/test_atrest.py`).  In a vehicle the key belongs in a TPM / secure element / keyring, not beside the data (`--atrest-key`); the alternative is full-disk encryption (LUKS), which is the integrator's standard tool and covers everything else on the disk too.
* **Roles and audit**: passcodes (scrypt), lock-out, personal data removed from the service / fleet / validation views, hash-chained audit trail (`docs/ARCHITECTURE.md` 8d).
* **Isolation**: `docs/SAFETY_ISOLATION.md`.

## 6. Who gets what

| Role | Opens | Gets | Where it is tested |
|---|---|---|---|
| Driver | Ask (voice or typing), Car, Energy, Map | plain answers, summaries of a trip or a day, "why" explanations in words, pre-drive briefing, range | `tests/test_everyday.py`, `tests/test_dialogue.py`, `tests/test_meaning.py` |
| Service engineer | Service page | per-system status (codes, warnings, tyres, 12 V, brakes, traction battery or engine, DPF / AdBlue / fuel trim) with evidence and the next service step; components with no data say *unknown*, never *fine* | `tests/test_roles_privacy.py` |
| Fleet operator | Fleet page | per-vehicle trips, distance, utilisation, efficiency against the fleet median of its type, harsh events per 100 km, codes, service overdue; no driver, place or route | `tests/test_roles_privacy.py` |
| Validation team | Validation page, evidence pack | injected-fault recall against ground truth, latency, signal coverage, a signed evidence pack for any answer | `tests/test_roles_privacy.py` |

**Not done:** no user study, no workshop or fleet has used it; every role page runs on simulated or recorded data.  That is the largest remaining gap in "value for the user".

## 7. One start-up, a production build, and the provenance of a release

* **One start-up for every door.**  `vehiclegpt.py run`, `python ui/server.py` and `app.py` all begin with `vgpt/bootstrap.py`: hardening, the product model from `models/product.json` (adapter, frame version, ONNX Runtime build), the memory guard (`--memory-limit-mb`
  or `VGPT_MEMORY_LIMIT_MB`) and the doctor.  `/api/health` says which model answers (adapter file and SHA-256, frame version, backend) and which release this is.
* **`--production`** (or `VGPT_PRODUCTION=1`) is the car's own build: no simulation or test controls, no study pages.  The test build (the default) keeps the Settings > Test track section.
* **Camera analysis** runs only while the camera picture is on screen (or a fault is being injected); ONNX Runtime busy-waits only on a machine with cores to spare (`VGPT_ORT_SPIN` decides explicitly).
* **A release is described by `RELEASE.json`** (`tools_release_manifest.py`): the SHA-256 of every runtime source file and one digest over them, the models, the vocabulary, the test data, the runtime, the git commit and tag when there is one, and - beside the installer - the installer's hash.  The
  installed program's doctor refuses a copy whose files differ from the manifest; `python tools_release_manifest.py verify --root <dir>` proves a device runs the release it claims.  A tree that is not a tagged git commit says so in the manifest (`releasable_from_tag: false`).
* **Test data** is its own versioned bundle (`tests/testdata_manifest.json`, `python tools_testdata.py check`): the small corpus (electric, petrol, diesel, hybrid and the situations that matter), the driving fixture and the sample cars.  The release gate runs with them required: an absent bundle fails the release, it does not skip tests.
