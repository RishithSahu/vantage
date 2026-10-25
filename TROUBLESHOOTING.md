# Troubleshooting

First, always: `python vehiclegpt.py doctor`. It checks Python, the libraries, the data, the model, the camera clip, the folder for logs and the network port, and tells you the fix for each problem.

| Symptom | Cause and fix |
|---|---|
| `python` is not recognised | Python is not installed or not on PATH. Install it from python.org and tick *Add python.exe to PATH*; open a new console. |
| `ModuleNotFoundError: cv2` / `numpy` | Run **Install-VehicleGPT.bat** (or `python -m pip install -r requirements.txt`). |
| "port 8765 is already in use" | VehicleGPT (or something else) is already running. The launcher normally picks the next free port; with `server.py` directly use `--port 8766`. |
| The page is blank or says *Lost the connection* | The console window was closed or the app crashed. Start it again; if it keeps happening send the newest `vehiclegpt.log` (see PRIVACY.md for its location). |
| The microphone button is greyed out | Your browser has no speech recognition. Use Chrome or Edge, or type. |
| Microphone does nothing / "Microphone blocked" | Allow the microphone in the browser's address-bar icon, and in Windows *Settings > Privacy > Microphone*. |
| Answers are not read aloud | Turn the volume up; browsers only speak after you have clicked somewhere on the page. Check *Settings > Read every answer aloud*. |
| "rules only" in `doctor` | The language model or `llama-cpp-python` is missing. That is fine - see USER_GUIDE.md if you want it. |
| Camera panel shows an error | `data/ref/camera_clip.hevc` is missing (re-extract the release), the webcam is used by another program, or Windows blocks camera access (*Settings > Privacy > Camera*). |
| It answered something wrong | Press the thumbs-down under the answer and type what you expected. That is the fastest way to get it fixed. |
| "That doesn't look like a question about this car" | Rephrase with the car in mind ("How's my battery?"), or tap one of the suggestion chips under the answer. |
| Slow first answer | The first question that needs the language model loads it (a few seconds); later ones are fast. Start with `--model none` to avoid it. |
| Antivirus or the company firewall complains | The app listens on `127.0.0.1` only and opens no outgoing connections. If your tool still blocks it, allow `python.exe` for local use. |

### Starting by hand, with options
```
python vehiclegpt.py run --port 8800 --model none --db gen_courier.db --no-browser
python vehiclegpt.py smoke        # self-test, exit code 0 = healthy
python vehiclegpt.py feedback     # what people rated down
```
Environment: `VEHICLEGPT_HOME` (where logs and feedback go), `VGPT_LOG_QUESTIONS=1` (also write question text to the log - off by default), `VGPT_VERBOSE=1` (log to the console too).
