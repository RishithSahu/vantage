# 05. Data, protocols, cars and storage

## 1. What data the assistant works on

Everything the assistant knows about a vehicle comes from **bytes that arrived on a pipe** (or, for testing, from a recorded history loaded in one step). Nothing is typed in by hand per car. The data reaches the database in four moves:

1. **Arrival**: raw records on the pipe.
2. **Recognition**: sniff what each channel carries, fingerprint the car against installed decoder files, or learn a new decoder file from the standards.
3. **Decoding**: the profile router turns bytes into named signals in standard units, events, log lines and thumbnails.
4. **Storage**: one writer stores them in tiers; trips and charging sessions are derived once, in one place.

## 2. Protocols: what is read, and how it was checked

Vantage reads thirteen protocol families. The decoders are written in the standard library only and are **listen-only**: nothing in them can transmit. `docs/PROTOCOL_EVIDENCE.md` is the authoritative page; it says for each family what independent implementation it was compared with.

| # | Family | Compared with (test-only library) | What was compared |
|---|---|---|---|
| 1 | CAN and CAN-FD (DBC) | cantools | every signal of every message of 77 real vehicle DBC files, random payloads: Intel and Motorola bit order, sign, scale, multiplexing, value tables |
| 2 | SAE J1939 (transport and diagnostics) | can-j1939 | identifier fields on 3,000 random ids; messages of 9 to 1,785 bytes by broadcast announce, rebuilt by us and by the reference; DM1 faults |
| 3 | OBD-II over ISO-TP | python-can-isotp, SAE J1979 formulas | segmentation and reassembly of every length 1 to 4,095; PID formulas |
| 4 | UDS (ISO 14229) | udsoncan | service and negative-response codes (65-entry table) |
| 5 | SOME/IP and service discovery | scapy | header, both directions |
| 6 | Android Vehicle HAL | AOSP `VehicleProperty` table | property ids and units (the table is *generated* from the AOSP files, nothing typed from memory) |
| 7 | NMEA 0183 GNSS | pynmea2 | our parser on its examples and its parser on ours |
| 8 | Logs (logcat, syslog 3164 and 5424, QNX slog2) | published line formats | parse tests with fixtures |
| 9 | Camera (JPEG, ADAS object records) | decode and schema tests | not an independent check |
| 10 | LIN 2.x | ldfparser | LDF reader on up to 600 random payloads, PID parity of all 64 ids, checksums |
| 11 | DoIP (ISO 13400-2) | python-doipclient, scapy | bytes equal for routing activation, diagnostic message, ack, NACK, alive check, announcement; streams cut at 50 random places |
| 12 | CANopen (CiA 301) | python-canopen | SDO reads and abort, emergency, heartbeat, NMT, SYNC |
| 13 | FlexRay (spec 2.1) | **none available offline** | only the specification's constants, a second separately written CRC and CRC properties. This is the weakest evidence on the page. |

**The Linux kernel as an oracle** (Raspberry Pi 4, 2026-10-23, `bench/pi_socketcan_oracle.py`): the kernel's own ISO-TP and J1939 stacks on a virtual CAN interface agree with ours in both directions on 53 of 53 transfers (lengths 1 to 5,000 bytes; J1939 broadcast and RTS/CTS). A first run had failures that were test mistakes, not decoder faults, and each was fixed in the test.

**Real attack data**: CAN attack recordings from the HCRL Car-Hacking dataset were used to score the CAN health monitor (learned on clean traffic; unknown ids and rate spikes): F1 0.94 to 1.00 with no false alarms on the clean run (early prototype).

### Collectors and formats

* Live collectors: UDP and TCP records, file tail (candump, ASC, JSON lines, NMEA, logs), NMEA and syslog over UDP, a Linux SocketCAN interface in **listen-only** mode (the socket is wrapped so it can only receive), and three subscribers (MQTT, W3C VISS over WebSocket, HTTP polling).
* File readers: candump (`-L` and default), Vector ASC and BLF (CAN and CAN-FD), PEAK TRC, CSV and TSV (the unit in a header such as `Speed (OBD)(km/h)` is split and converted), JSON and JSON lines (OpenXC, COVESA VSS paths), NMEA, ELM327 traces, DLT, text logs.
* `python vehiclegpt.py ingest capture.csv --db data/mycar.db` reads a capture, says which quantity each signal is and how sure it is, writes the profile pack it worked out, and stores the data through the same runtime a bus would use (`docs/DATA_SOURCES.md`).

## 3. The signal mapper: refusing is better than guessing

`vgpt/ingest/mapper.py` is the one engine that says which registry quantity a signal is (a DBC signal, a table column, a VHAL property, a SOME/IP field). It decides from the name, unit, bit width, scale, range, sibling signals and observed statistics, and it **refuses** when unsure.

* Benchmark (`bench/mapper_benchmark.py`): 831 hand-reviewed signals over 76 real opendbc files: **0 wrong mappings**, recall 0.997 (two non-English refusals accepted).
* Rules: unknown non-noise units refuse; state signals (door, belt, wiper, headlamp, parking brake) refuse if any word is unread; controls (stalk, switch, handle) are not the thing controlled.
* Every held-out round of new DBCs still found 2 to 4 wrong guesses in 200 to 300 mappings before fixes, so "perfect" is claimable only on the reviewed set. Only English signal names are read.

## 4. The cars

| Group | Count | Notes |
|---|---|---|
| Real vehicle models from opendbc (MIT licence) | 25 | Acura, Honda, Hyundai, Kia, Toyota, Tesla, Nissan, Volvo, Mazda, BMW, Mercedes, Chevrolet, Chrysler, Ford, and others; each with the bindings the mapper drafted and a person reviewed |
| Synthetic maker "test bench" cars | Aster (EV), Kestrel (petrol), Meridian (US CAN-FD EV), Rhino (J1939 diesel truck), Nova, Volta (bus), Prestige (GT) and others | written for the bench; they differ in buses, names, units, byte order and powertrain |
| Standards cars | OBD-II dongle car, J1939 heavy vehicle, Android Automotive car (VHAL only), proprietary-CAN car with a clearly labelled **dummy** decoder file | prove the "learn from standards" and "unidentified" paths |

The generated catalogue `docs/CAR_CATALOGUE.md` lists 45 models from 34 makers as of its last generation (2026-10-23); further protocol cars were added after. Protocol coverage across them: logs and GNSS on all, CAN on 44, UDS/OBD 44, camera 30, CAN-FD 4, SOME/IP 3, VHAL 3, LIN 2, DoIP 1, FlexRay 1, CANopen 1. English-named signals only: VW MQB (German) and PSA (French) databases were dropped at the owner's rule.

## 5. Where histories come from (simulation and recorded data)

The case study asks how driving scenarios and error sequences are recreated. The answer is simulation and recordings, on real data where possible:

* **The life generator** (`vgpt/datagen*.py`, `vgpt/world/`): a seeded, physics-driven simulation built on **real weather** (Open-Meteo hourly), **real driving** (VIF-AEGIS 35 trips with accelerometer, and EPA UDDS / HWFET / US06 / SC03 / FTP dynamometer cycles), five driver styles and eight cities. It lives 105 days (or four years, built as four one-year pieces with their own seeds and joined) and plants faults with a **ground-truth file**: cooling-pump failure, tyre slow leak, roof box, weak 12 V battery, dirty windscreen, roadworks, plus 13 injectable fault scenarios in the virtual car (coolant pump, ECU reset, unexpected shutdown, voltage dip, sensor conflict, camera blocked, and others).
* **The virtual car** (`vgpt/live/car.py`): a physical model that knows only quantities (speed, state of charge, tyre pressures), never a bus. The **emitter** (`vgpt/world/emit.py`) walks the car's profile in reverse and writes CAN, SOME/IP, VHAL, UDS, log, camera and NMEA traffic.
* **Combustion overlay** (`vgpt/world/ice.py`): the same life lived by a diesel truck or a petrol hatchback: gear, engine speed, torque, load, fuel rate and level with refuelling, coolant and oil warm-up, and diesel- or petrol-specific signals.
* **Recorded captures**: the car can play any capture (candump, ASC, BLF, CSV, JSON) to Vantage as a vehicle. The four recordings supplied for Case Study 5 hold only a few signals (for example speed) and are the test for "refuse almost everything honestly".
* **Android Automotive**: a real Android 17 Automotive guest (Cuttlefish `aosp_cf_arm64_auto`) runs on the Pi under KVM. The `car-aaos` bridge reads its Vehicle HAL values with adb and puts them on the pipe; Vantage learned 19 quantities with no decoder file. Limit: the stock fake VHAL has fixed read-only speed and gear, so the car does not move.
* **CARLA** was not used.

Honest limit: encoder and decoder were written inside the project (the emitter and the decoders agree by construction, though the decoders are checked against independent libraries). No real vehicle bus has ever fed the system.

## 6. Storage, retention and the size budget

### The tiers (as shipped)

| Tier | Where | Kept | Size |
|---|---|---|---|
| Hot | database: 1-second rows (`sig_1s`, `lk_1s`) | 14 days | about 100 MB |
| Minutes | database: 1-minute rows (core and lake); a 31-byte row | 5 years (1,830 days) | about 175 MB per car-year at 41 quantities |
| Daily | `agg_1d`: per-day mean, max, min of every column | forever | 12 MB for 3 years of 41 quantities |
| Records | events, trouble codes, logs, trips, charges, service history, frozen context around events | 5 years | tens of MB |
| Lossless archive | `archive/*.vga`: the expired seconds exactly | up to the budget | 0.4 bytes per sample |

* **Self-describing**: the database carries its own profile and a `sig_catalog` with one row per mapped signal (concept, unit, plausible range, source, rate), plus per-signal SQL views. A new car's signals are readable with no schema change.
* **Archive codec v2**: byte-plane integer deltas, and floats that are really fixed point stored as integers (checked bit for bit). One day of 41 signals: 317 KB became 189 KB, and it decodes faster. Files are SHA-256 hashed and hash-chained; a damaged file is reported, not used.
* **Size budget** (`vgpt/storage_budget.py`): `ret.budget_gb` (default 6) is the most the database plus the archive may take; the nightly housekeeping removes the oldest archive files first until the total is 10 % under it and writes a `source_log` row.
* **Four-year worlds**: 389 MB (Aster EV, 41 quantities), 627 MB (Kestrel petrol), about 900 MB (Meridian) for four years; under 1.3 GB projected for five. These come from generated data, which is smoother than a real car's.
* **A rejected design**: a cold-minute archive hydrated on demand cut size but cost 0.9 to 10 s per question; reverted. Not even 1 ms of added latency for storage.

### Retention for the case study

"Searchable history of at least several weeks" is met many times over: minutes for five years answer every ordinary question; exact seconds for 14 days; older exact-second questions (the highest speed on a day last year) are answered from the archive.

### Concurrency

One writer, many read-only readers (SQLite WAL). The assistant's handle is read-only three ways: the file mode, the SQL, and an authorizer callback. The writer runs in the receiver's store thread; the collector thread decodes without touching the disk.

### Instant load (for testing)

The Car app's speed list has "Instant: load the whole dataset now" (and 120x, 300x, 600x). The car sends a control record naming a finished history; Vantage checks it (switch `VGPT_ACCEPT_HISTORY`, absolute `.db` path, at most 16 GB, same tables, concepts a subset of the car's, no existing history above 1 MB) and copies it with an SQLite backup into the car's database. A 4-year history (389 MB, 2.1 million minute rows) loads in 1.2 s from the installed program; the car then continues from the last 10 minutes.

## 7. Where data and files live on a device

| What | Where |
|---|---|
| Per-vehicle database | `<home>/vehicles/<id>[-VIN]/vantage.db` (the default home is private to the user; `VEHICLEGPT_HOME` overrides) |
| Archive | `archive/` next to the database |
| Keys, hashes, logs, feedback | the private home folder; sealed or encrypted where the platform can |
| Models | `models/` next to the program; `models/product.json` names the adapter and the ONNX build |
| Decoder files | `vgpt/vehicle/packs/<id>/` and `vgpt/car/models/<id>/` |
