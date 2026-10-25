# Third-party software and data

VehicleGPT bundles or uses the following. **Items marked VERIFY need a licence check by the project owner before public distribution** - we could not confirm them from inside the project and will not guess.

## Software (installed from PyPI by the installer, not bundled in the zip)
| Component | Licence | Notes |
|---|---|---|
| NumPy | BSD-3-Clause | required |
| OpenCV (`opencv-python-headless`) | Apache-2.0 (OpenCV); the Python wheels also include FFmpeg libraries under LGPL-2.1 | required; decodes the camera clip |
| llama-cpp-python / llama.cpp | MIT | optional; runs the language model |
| scikit-learn, SciPy | BSD-3-Clause | development only (training the classifier); not needed to run |

## Model (optional, not in the zip unless built with `--with-model`)
| Component | Licence | Notes |
|---|---|---|
| Qwen2.5-0.5B-Instruct (GGUF, Q4_K_M) and Qwen2.5-1.5B-Instruct | Apache-2.0 (Alibaba Cloud / Qwen team) | the 0.5B and 1.5B sizes are Apache-2.0; other Qwen2.5 sizes have different terms, do not swap them in without checking |

## Data used to build the simulated vehicle histories (the shipped `data/*.db` are derived from these)
| Source | Licence / terms | Status |
|---|---|---|
| EPA dynamometer drive schedules (UDDS, HWFET, US06, SC03, FTP) | US Government work, public domain | OK |
| Open-Meteo historical weather API | CC BY 4.0 - **attribution required**: "Weather data by Open-Meteo.com". Free API tier is for non-commercial use | VERIFY if distribution is commercial |
| VIF-AEGIS automotive research dataset (35 real trips: OBD speed, accelerometer) | not stated in the project | **VERIFY** - used to shape driving patterns in the simulation; derived values only, no raw trips are shipped |
| comma2k19 (front-camera clip `camera_clip.hevc`) | repository states MIT | VERIFY the dataset terms; the clip is shipped as-is |
| HCRL Car-Hacking dataset (CAN-bus intrusion examples) | research dataset, terms not confirmed | **VERIFY**; used only for development benchmarks, **not shipped** |
| AOSP `VehicleProperty.aidl` (Android Vehicle HAL property list) | Apache-2.0 | property names / IDs only |
| Open-source CAN / OBD / UDS protocol knowledge | public standards (ISO 15765, SAE J1979, ...) | implemented from the public descriptions |

## This project
`LICENSE.txt` is intentionally **not supplied yet**: the choice of licence (or an "all rights reserved, evaluation use only" notice) belongs to the project owner and must reflect the VERIFY items above. Until it is chosen, treat the package as *not licensed for redistribution*.

## Added 2026-10-20 (voice input)
| Component | Licence | Status |
|---|---|---|
| Vosk (`vosk` Python package, Kaldi based offline speech recognition) | Apache-2.0 | installed with the owner's approval; optional |
| `vosk-model-small-en-us-0.15` (Alpha Cephei small English model, ~40 MB) | Apache-2.0 | downloaded from alphacephei.com/vosk/models with the owner's approval, kept in `models/asr/` (not part of the repository's source) |

## Added 2026-10-21 (car models and the signal mapper)
| Component | Licence | Status |
|---|---|---|
| **opendbc** (comma.ai, https://github.com/commaai/opendbc): the open CAN databases (`*.dbc`) in `vgpt/car/models/<model>/` that describe the real Hyundai, Toyota, Tesla, BMW, Mercedes-Benz, Volvo, Nissan, MG, BYD, GWM, Hongqi and Chevrolet buses used by the simulated cars, and the 70-odd databases in `tests/fixtures/opendbc/` used to measure the signal mapper | MIT (Copyright (c) 2020, Comma.ai, Inc.); the licence text is `tests/fixtures/opendbc/LICENSE` and must travel with any copy of these files | OK - the notice and the licence text are kept with the files. Brand names in model names are only the names of the vehicles the databases describe; no vehicle maker has endorsed this product |
| AOSP `hardware/interfaces/automotive/vehicle` (`VehicleProperty.aidl`, used for the Android Automotive reference model) | Apache-2.0 | property names and identifiers only |
| SAE J1939-71 / J1979 parameter definitions (heavy-truck and OBD-II dongle models) | public standards | implemented from the published scalings; no copyrighted tables are copied |

## Added 2026-10-21 (the program's own microphone reader and the installer)
| Component | Licence | Status |
|---|---|---|
| `sounddevice` (Python bindings) and the PortAudio library it ships on Windows | MIT (sounddevice) / MIT-style (PortAudio) | the program reads the microphone itself, so no web view or browser has to capture audio; bundled in the installer |
| Inno Setup (build tool only, not shipped) | Inno Setup licence (free, including commercial use) | used to compile the Windows setup file |
| PyInstaller (build tool; its bootloader is in the bundle) | GPL-2.0 with the bootloader exception (the exception permits any licence for the bundled program) | OK |
| The opendbc licence text travels with the installer in `licenses/opendbc/LICENSE` | MIT | see above |
