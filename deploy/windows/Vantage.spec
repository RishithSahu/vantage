# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_all

root = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas = [
    # the interface the car shows: its pages and the modern interface.  The study pages (research.html, review.html) and the study's dashboard module (ui/research.py) are NOT part of the product build.
    (os.path.join(root, "ui", "app.html"), "ui"),
    (os.path.join(root, "ui", "index.html"), "ui"),
    (os.path.join(root, "ui", "modern"), "ui/modern"),
    (os.path.join(root, "deploy", "tiers.json"), "deploy"),
    (os.path.join(root, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), "models"),
    (os.path.join(root, "models", "meaning-lora-frame2r12.gguf"), "models"),
    (os.path.join(root, "models", "product.json"), "models"),
    # the offline speech model: voice input is part of the product
    (os.path.join(root, "models", "asr", "vosk-model-small-en-us-0.15"), "models/asr/vosk-model-small-en-us-0.15"),
    (os.path.join(root, "data", "world_aster.db"), "data"),
    (os.path.join(root, "data", "world4", "aster_ev", "aster_ev_4yr.db"), "data/world4/aster_ev"),     # four years of the sample car (tools_build_worlds.py): history questions, and the Car Simulator's "Instant" load
    (os.path.join(root, "data", "ref", "camera_clip.hevc"), "data/ref"),
    (os.path.join(root, "README.md"), "."),
    (os.path.join(root, "PRIVACY.md"), "."),
    (os.path.join(root, "THIRD_PARTY_NOTICES.md"), "."),
    (os.path.join(root, "USER_GUIDE.md"), "."),
    (os.path.join(root, "docs", "CAR_CATALOGUE.md"), "docs"),
    (os.path.join(root, "docs", "STORAGE.md"), "docs"),
    (os.path.join(root, "tests", "fixtures", "opendbc", "LICENSE"), "licenses/opendbc"),
]
# Every non-code runtime contract (ontology/catalog JSON, classifiers, DBCs, profile packs, templates) is
# package data.  Listing known folders hid newly added contracts and made the installed simulator fail while
# source runs passed.  Discover package data as a class instead.
datas += collect_data_files("vgpt", include_py_files=False)
binaries = []
hidden = []
# the offline speech recogniser and the program's own microphone reader (PortAudio): no browser captures audio
# llama_cpp: the language model's native libraries (llama.dll, ggml*.dll) sit in llama_cpp/lib; without them the app starts and every question says the model could not be loaded
for pkg in ("vosk", "sounddevice", "_sounddevice_data", "llama_cpp"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

dynamic = ["vgpt.nlu", "vgpt.nl2ir", "vgpt.composed", "vgpt.temporal_ops",
           "vgpt.temporal_misc", "vgpt.temporal_events", "vgpt.semantic_ops"]
a = Analysis([os.path.join(root, "vehiclegpt.py")], pathex=[root], binaries=binaries, datas=datas,
             hiddenimports=dynamic + hidden + ["webview", "tkinter", "sounddevice", "vosk"],
             hookspath=[], runtime_hooks=[],
             # Development/training packages are installed on this workstation but are not runtime
             # dependencies.  Excluding them is essential: otherwise PyInstaller follows optional
             # statistical fallbacks and accidentally bundles PyTorch, transformers, notebooks and tests.
             excludes=["torch", "transformers", "pandas", "pytest", "IPython", "ui.research",
                       "matplotlib", "scipy", "pygame", "notebook", "jupyter",
                       "PyQt6", "pyarrow", "zmq", "fsspec"],
             noarchive=False)
pyz = PYZ(a.pure)
icon = os.path.join(root, "ui", "modern", "vantage.ico")
# Vantage.exe is the windowed program (no console window behind the app).  VantageCli.exe is the same
# program with a console, for the commands that talk in text (doctor, car, receive, ingest ...).
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Vantage", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False, icon=icon)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VantageCli", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True, icon=icon)
coll = COLLECT(exe, cli, a.binaries, a.datas, strip=False, upx=False, name="Vantage")
