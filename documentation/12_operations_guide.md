# 12. Operations guide: run, install, test, evaluate, deploy

All commands are run from the project folder (`vehiclegpt2/`). Paths and ports are the ones recorded in the project; verify before relying on them.

## 1. Prerequisites

* Python 3.11 or newer (3.13 on the Pi); `numpy`; SQLite; for the model path `onnxruntime` (preferred on the Pi) or `llama-cpp-python`; optional `opencv`, `cryptography`, `psutil`, `vosk` (voice), `pywebview` (window).
* Models in `models/`: `qwen2.5-0.5b-instruct-q4_k_m.gguf`, the adapter `meaning-lora-frame2r12.gguf`, and for ONNX `models/ort_frame2r12/` (`model.onnx`, `model.onnx.data`, `draft.json`). `models/product.json` names them. Speech: `models/asr/vosk-model-small-en-us-0.15`.
* Data: car histories in `data/` (for example `data/world4/<car>/<car>_4yr.db`); test corpus in `tests/corpus/`.
* The git repository is local only; the tag `v1.3.4` marks the release; `.gitignore` keeps data, models, results and build output out.

## 2. The launcher

```
python vehiclegpt.py                  start the app in its own window (same as: run)
python vehiclegpt.py run --kiosk --db <history>.db       full screen, for the car's display
python vehiclegpt.py doctor           check this computer and the install, saying exactly what to fix
python vehiclegpt.py smoke            start the app in-process, ask a set of questions, report pass or fail (exit 0 = healthy)
python vehiclegpt.py ingest FILE --db data/mycar.db      read any capture (candump, ASC, TRC, BLF, CSV, JSON, VSS, OpenXC, ELM327, NMEA) into a history
python vehiclegpt.py ingest FILE --report-only           only say what the capture carries and how sure it is
python vehiclegpt.py receive          Vantage's end of the one-way pipe
python vehiclegpt.py car MODEL [--dataset f.db] [--instant]   the separate simulated car end of the pipe
python vehiclegpt.py car-aaos --adb <adb> --serial 0.0.0.0:6520   use an Android Automotive device as the car
python vehiclegpt.py car-ui           the car and dataset selector window
python vehiclegpt.py voice-test       listen for a few seconds and say which step of voice fails
python vehiclegpt.py build-ui         pack the interface into its single self-contained page
python vehiclegpt.py install-app      put a Vantage icon on the desktop and in the start menu
python vehiclegpt.py feedback [--tsv out.tsv]            summary of thumbs up or down, optionally a triage list
python vehiclegpt.py version
options: --linked (read the car on the local pipe), --production (no simulation or test controls), --research, --shell auto|webview|pipe, --memory-limit-mb N
```

Every door (launcher, `python ui/server.py`, `python app.py`) starts through `vgpt/bootstrap.py`: harden, adopt `models/product.json`, memory guard, doctor.

## 3. Running the two programs (Car and Vantage)

1. Start Vantage: `python vehiclegpt.py run --linked` (or the installed Vantage app).
2. Start the car in another process: `python vehiclegpt.py car aster_ev [--dataset data/world4/aster_ev/aster_ev_4yr.db] [--instant]` (or `car-ui`). Speeds: real time, 120x, 300x, 600x, pause, restart, or **instant load** (copies a whole finished history in one step).
3. The Vantage link view shows how the car was recognised and the loading progress.

**For development, never use the default link home.** Use `bench/dev_web.py` and `bench/dev_up.sh` (`VGPT_LINK=devlink`, `VEHICLEGPT_HOME=../_dev_home`, `--link-home ../_dev_link`). A test car once appended six hours of simulated rows to the real store.

## 4. Environment switches

| Variable | Effect |
|---|---|
| `VEHICLEGPT_HOME` | the private home folder (logs, keys, vehicles); default per OS |
| `VGPT_FRAME` | frame version (2 for the product); set from `models/product.json` |
| `VGPT_MEANING_ADAPTER`, `VGPT_ORT_DIR`, `VGPT_VEHICLE_HINT` | adapter, ONNX build and the car-type line; set from `product.json` unless set explicitly |
| `VGPT_BACKEND=llama`, `VGPT_DECODER=grammar` | use llama.cpp; use the grammar sampler instead of the token mask |
| `VGPT_MEANING=0` | the older rule front end (legacy, for baselines) |
| `VGPT_MEANING_TAU=0.8` | turn ask-back on |
| `VGPT_MEANING_REFUSE_BELOW=0` | do not withhold low-confidence readings |
| `VGPT_PREPARSER=0`, `VGPT_MODEL_ONLY=1`, `VGPT_LAYERS_OFF=all\|investigate,discourse,unsafe_floor,own_vocab,fuel_remap,hint_terms...` | measure the model alone |
| `VGPT_ARBITRATE=1` | the retired arbitration rules |
| `VGPT_TIER=edge2g\|pi5\|vm8\|hpc` | resource tier |
| `VGPT_CORES`, `VGPT_MEM_MB`, `VGPT_ISA`, `VGPT_ACCEL` | override what the probe reads (to make a laptop behave as a 2-core board) |
| `VGPT_MEMORY_LIMIT_MB` | process ends itself above this (the service manager restarts it) |
| `VGPT_ACCEPT_HISTORY` | allow the instant load of a finished history |
| `VGPT_RESTRICT=1` | the driving lock (typing refused while the car moves) |
| `VGPT_ATREST_KEY` | where the at-rest key comes from (a TPM credential, a keyring, a command; a file is test only) |
| `VGPT_LINK` | the pipe name |
| `VGPT_HINT_POLICY`, `VGPT_KEYED_VIEWS`, `VGPT_DATA_FAST`, `VGPT_CASCADE_REUSE` | the 26 Oct speed-work switches (see [08](08_performance_and_scaling.md)) |
| `VGPT_NOW=<epoch>` | freeze "now" for probes on a recorded history |
| `VGPT_REQUIRE_TESTDATA=group,...` | make absent test data fail instead of skipping (release gate) |
| `HF_HUB_OFFLINE=1` | set by the launcher: never fetch a model |

## 5. Installing

### Windows (the installer)
`deploy/windows/build.ps1` runs the tests, builds the bundle with PyInstaller (`Vantage.spec`), checks the bundle (`VantageCli.exe doctor`, car list, speech model, microphone library), and compiles `dist/Vantage-Setup-<version>-x64.exe` with Inno Setup (`Vantage.iss`). It installs per user without administrator and contains the model (Qwen2.5-0.5B plus the adapter), the speech model, the simulated cars with their decoder files and a four-year sample history. The silent install needs `/DIR="quoted path"` as one argument string. Rebuild the installer only from a clean tagged tree so `RELEASE.json` says the tree is releasable from the tag. Release gate: `python tools_release_gate.py`; manifest: `python tools_release_manifest.py make|verify`.

### Raspberry Pi
The reference deployment (as recorded; verify the paths):
* Login: key only, `ssh -i ~/.ssh/vantage_pi vantage@<pi address>`; passwordless sudo; user services start at boot (linger on).
* The current code tree used for testing is `~/vantage134`; the older live server tree is `~/vehiclegpt2` (its `.venv` has onnxruntime and llama.cpp: **run benches with `~/vehiclegpt2/.venv/bin/python`**).
* The assistant runs as a systemd user service (`vehiclegpt-vantage.service`, `MemoryMax=2G`, `CPUQuota=200%`); a hardened system-service form is `deploy/linux/vgpt-assistant.service.in` (no network, `AF_UNIX` only, no swap, no capabilities; scored 0.6 SAFE by `systemd-analyze security`).
* Sync with tar over ssh (excluding `__pycache__` and the vehicle packs), then restart the unit.
* For latency benches stop the service first (only about 900 MB of RAM stays free beside it) and pin to the two cores: `taskset -c 0,1 ...`.
* Do not touch the other project on the Pi (`~/aqe*`, ports 8787/8788), the hotspot or the firewall.
* Android Automotive on Cuttlefish (`~/cf`): start with `nohup sg cvdnetwork -c ~/cf/launch.sh`, boot takes 5 to 7 minutes, adb at `0.0.0.0:6520`; the tap devices and ACLs are lost on reboot; the guest and the assistant do not fit in RAM together.

## 6. Tests

```
python -m pytest -q tests                              # the suite (about 1,770 tests); heavy modules are best run one at a time
python -m unittest tests.test_meaning                  # includes RuleBudget
python tools_run_tests.py [tests.test_x]               # run under the target's limits: 2 cores, memory reported against the 2 GB budget
python tools_testdata.py check                         # which test-data groups are present and whether their hashes match
python tools_release_gate.py [--quick]                 # every release gate in one command
```

* Protocol oracle tests need the test-only libraries (cantools, scapy, udsoncan, can-j1939, doipclient, ldfparser, python-canopen); they live in `.oracle_venv`. When Hypothesis is missing, run with `--ignore=tests/test_protocol_oracles.py`.
* Tests that start a real `App` must restore `os.environ`. Tests that call `_adopt_product_model` must use `tests/_envguard.py`.
* On the Pi: `bench/pi_socketcan_oracle.py` (kernel ISO-TP and J1939 oracle on `vcan0`), `bench/pi_security_proof.py` (offline in an empty network namespace, sealed database), `tools_isolation_rig.py` (freedom from interference), `bench/pi_cartype.py` (per car type latency of every probe question).

## 7. Evaluating the model and the answers

```
python bench/eval_parallel.py models/meaning-lora-frame2r12.gguf <name> [--no-hint] [--procs 3] [--threads 4]   # all frozen and new sets
python bench/meaning_eval/rescore.py <name> [other names]         # the table (reviewed / strict, per set)
python bench/meaning_eval/gate.py ...                              # the adoption gate
python bench/meaning_eval/layer_ablation.py                        # each rule layer against the model alone
python bench/meaning_eval/pipeline_fidelity.py                     # live ingest against the finished recording
python bench/calibrate_confidence.py                               # rebuild vgpt/meaning/calibration.json (development sets only)
python bench/causal_bench.py ; python bench/event_cause_bench.py   # causes against planted truth
python bench/mapper_benchmark.py                                   # signal mapper, 831 reviewed signals
python tools_triage.py "a failing question" --expect "act concept ..."   # says whether the fix is a glossary line, an ontology entry or a training row
```

Run an evaluation as a plain background command; a `nohup ... &` job died once with no output.

## 8. Training (only with the owner's approval)

```
python training/meaning/build_meaning.py ...       # build the training set (generated rows plus natural banks; --eval-leak vN filters evaluation questions)
python -m training.train_lora --data ... --epochs 4 --bs 16     # own LoRA trainer; GPU; ~1 h 45 min for the largest rounds
python -m training.convert_lora_gguf ...           # to a separate GGUF adapter
python tools/build_ort_model.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/<adapter>.gguf models/ort_<name>   # merge and store as int8 for ONNX (delete the target folder first; external data appends)
python tools_autotrain.py                          # an unattended chain: train, evaluate on every frozen set, apply the gate, write a report
```

Adoption is manual: set the adapter (and `ort`) in `models/product.json`, re-run the gate, update the docs. Keep the previous adapter and a backup of `product.json`. Never train on a sealed set. Stop runners by PID, never `python.exe` by name.

## 9. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "the language model could not be loaded" on the Pi | the wrong Python; use `~/vehiclegpt2/.venv/bin/python` |
| First question slow | the model loads on first use; the app warms it with one question |
| Every question fails on Python 3.13 with a `_ThreadHandle` error | a method named `_handle` on a `Thread` subclass (fixed: `_run_job`) |
| Questions on a 4-year database take 0.6 s per MIN/MAX | full-table scans (fixed: separate MIN and MAX statements, span cached by `data_version`) |
| Repeated availability probes on a live car | the profile is rebuilt when `data_version` changes; known-present columns are now preserved |
| A test fails with missing data | the corpus-based tests do not; others skip or fail if a data group is absent; `tools_testdata.py check` |
| ONNX rebuild directory has 1.2 GB | external data appends on rebuild; delete the folder before building |
| The installer's silent install exits 4 | pass `/DIR="..."` as one argument string |
| Voice does nothing | run `voice-test`; the microphone needs the app window or a secure origin; Vosk misreads "AdBlue" as "a blue" |
| Pi reboot lost the Cuttlefish taps | re-create the tap devices, ACLs and the `cvdnetwork` setup (see the notes of 21 Oct) |
