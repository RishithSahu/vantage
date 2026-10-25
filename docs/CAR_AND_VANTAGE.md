# The Car and Vantage: two programs, one local pipe

Everything below is offline. The only thing the two programs share is a one-way operating-system pipe on the same machine (Windows named pipe `\\.\pipe\VehicleGPT-<user>-link`, Linux FIFO). There is no network socket anywhere in the link (`tests/test_pipe.py::OneWay` checks the source; `vgpt/offline.py` would refuse a socket at run time).

```
 CAR (vgpt/car, `Vantage.exe car-ui` / `vehiclegpt.py car <model>`)           VANTAGE (vgpt/pipe + ui, `Vantage.exe run --linked`)
 a vehicle model + optional recorded dataset                                   COLLECTOR   look at the bytes, work out what they are, decode
   -> each bus as that car's bus really carries it                              STORE       one writer thread per vehicle database (SQLite, WAL)
      (SocketCAN frames, SOME/IP, Android VHAL parcels, log lines,              QUERY       the assistant, read-only handles
       NMEA, JPEG pictures, OBD-II/UDS over ISO-TP)                            
   -> in real time, with channel numbers of its own  ------- pipe ------->      knows no make, model, bus or channel number in advance
```

## What Vantage does with a stream it has never seen

1. **Wire** (`vgpt/pipe/wire.py`): SYNC, length, CRC-32, a time, a channel number and bytes. Nothing about vehicles. A reader that joins in the middle or meets damage skips to the next good record and counts what it skipped.
2. **Sniff** (`sniff.py`): every block is classified from its structure (CAN / CAN-FD frame, SOME/IP, VHAL parcel, picture, NMEA, JSON, log line). A channel is read the way most of its blocks read.
3. **Fingerprint** (`fingerprint.py`): the CAN identifiers, SOME/IP services and VHAL properties seen are compared with every installed *decoder file* (a profile pack: its DBCs, service catalogue and bindings). The best fit is used, with the reasons; which wire is which bus is decided from the same identifiers.
4. **Learn** (`autoprofile.py`, `mapper.py`): if no decoder file fits, what the standards allow (OBD-II, SAE J1939, Android VHAL, NMEA, named values) is read from the traffic and written down as a new decoder file.
5. **Unidentified**: if even that finds nothing (a proprietary network), the link says so and lists the unexplained identifiers. Adding a decoder file (see `vgpt/car/models/proprietary_demo/` for a dummy proprietary CAN database) makes Vantage recognise the car at once, without a restart.
6. **Adapt**: the inventory keeps growing; a clearly better fitting decoder file replaces the current one; a new connection is a new session; two cars of one model are told apart by the VIN they announce.

The signal mapper (`vgpt/ingest/mapper.py`) decides which quantity a signal is. It refuses rather than guesses: measured on 831 reviewed signals of 76 real vehicle databases it makes no wrong mapping (`python bench/mapper_benchmark.py`, `tests/test_mapper.py`). Names are read in English only for now.

## Where the data goes (retention)

| Tier | What | Kept |
|---|---|---|
| Hot | every signal once per second (`sig_1s`, `lk_1s`) in the vehicle's SQLite database | `fine_days` (7) |
| Per minute | duration-weighted averages and extremes (`sig_1m`, `lk_1m`, `cam_1m`, `ext_1m`) | 1 year |
| Lossless archive | the expired seconds, exactly (`archive/*.vga` next to the database, indexed in `archive_idx`, SHA-256 per file, hash-chained) | 1 year (`ret.archive_days`) |
| Records | events, diagnostic trouble codes, logs, trips, charging sessions, service history, context around events | 5 years |

The archive is read only for a question that is exact over the seconds themselves (a highest speed, how long a limit was exceeded) and only for windows up to 45 days; it is loaded for that one question into the assistant's own temporary tables, the database is never written by the assistant, and a damaged file is reported instead of being used (`vgpt/archive.py`, `tests/test_archive.py`). Everything else is answered from the minute tier.

## Voice

The microphone is read by the program itself (`sounddevice`/PortAudio), recognised offline (Vosk, `models/asr/vosk-model-small-en-us-0.15`) and the page only shows the words that come back. No browser captures audio and nothing is sent anywhere. "Vantage, how is my battery?" fills the question box; "Vantage stop" ends listening.

## Windows installer

`deploy/windows/build.ps1` runs the tests, builds the bundle (PyInstaller), checks the bundle (`VantageCli.exe doctor`, car list, speech model and microphone library present) and compiles `dist/Vantage-Setup-<version>-x64.exe` (Inno Setup). It installs per user, needs no administrator, and contains the one language model (Qwen2.5 0.5B + the `frame2d` adapter), the speech model, the simulated cars (20 models) with their decoder files and a recorded sample history. Start-menu entries: Vantage (reads the car on the pipe), Car Simulator, Vantage with the recorded sample history, Vehicle Receiver (text window), Check this installation.

## Android Automotive as the car (tested on the Raspberry Pi 4)

`vehiclegpt.py car-aaos --adb <adb> --serial <device>` makes an Android Automotive device the car: it reads the device's Vehicle HAL values (`cmd car_service get-property-value`), puts them in the form the HAL really uses and writes them to the same pipe. Vantage is not told it is Android; it sees VHAL parcels, learns the standard properties (speed, gear, odometer, tyre pressure per wheel, battery energy as a percentage of the pack's capacity, temperatures ...) and stores them.

Measured on the Pi (aarch64 Debian 13, 3.7 GB, KVM): a Cuttlefish `aosp_cf_arm64_auto-userdebug` guest (Android 17 Automotive, build 16644040, 2 GB, 2 cores) boots, and Vantage learned 19 quantities from it and answered questions about them. Limits that are real: a stock Cuttlefish Vehicle HAL keeps speed, gear and the like at fixed values (read-only fake hardware), so it shows the data path but not driving; a moving Android car needs a vehicle simulator behind the HAL (CARLA, a replay). The guest and the assistant's language model do not fit in the Pi's RAM together, so the assistant was run with the guest stopped.
