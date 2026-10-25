# VehicleGPT - user guide

VehicleGPT is an assistant that answers questions about a vehicle's **recorded data**: range and efficiency, battery and cooling, charging, tyres, trips, warnings and fault codes. It runs entirely on your computer and works offline.

> **What it is not.** It is an advisory tool, not a safety system. It cannot see the car live, cannot control anything and cannot diagnose a fault by itself. When it says "get the cooling checked" it means *ask a workshop*. The data in this release is a **simulated** vehicle history made for demonstration.

## 1. Install (once)

You need **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick *Add python.exe to PATH*). Then, in the extracted folder:

| System | Do this |
|---|---|
| Windows | double-click **Install-VehicleGPT.bat** (needs internet once, a few minutes), then **Start-VehicleGPT.bat** |
| macOS / Linux | `./install.sh` then `./start.sh` |
| Any | `python -m pip install -r requirements.txt` then `python vehiclegpt.py` |

The installer ends with a self-check. If it prints **READY**, you are done. If it prints **NOT READY**, each `[FAIL]` line says exactly what to fix. You can run the check any time: `python vehiclegpt.py doctor`.

### Optional: the language model
Without it the assistant answers with its rules, which understand roughly nine in ten everyday questions. A small language model (Qwen2.5-0.5B-Instruct, ~490 MB) helps with unusual wording. To use it: put `qwen2.5-0.5b-instruct-q4_k_m.gguf` and the question-reading adapter `meaning-lora-full_v6.gguf` in the `models` folder and run `python -m pip install llama-cpp-python`. The model reads each question into a small structured form and the program answers from your car's recorded data; `doctor` tells you what is missing. To go back to the previous question reader set the environment variable `VGPT_MEANING=0` before starting (it needs `semantic-lora-qwen25-05b-v2.gguf` instead).

## 2. Use it

Start it as an application with `python vehiclegpt.py app` (add `--kiosk` for a full-screen car display such as the Raspberry Pi: see `deploy/raspberrypi/README.md`): the assistant opens in a window of its own, with no address bar, and stops when you close it. Or start it the older way and your browser opens at `http://127.0.0.1:8765` (the number changes if that port is busy). Click **Got it** on the first-run notice.

* **Type or speak.** Press the **microphone** button at the top: the app listens for its name. Say *"Vantage, how is my battery?"* and the question appears in the text box as you say it (press Enter to ask; while the car moves it is asked at once). Say *"Vantage"* alone and then your question, and *"Vantage stop"* - or press **Stop** - to end listening. Everything is recognised on this computer by an offline speech model (Vosk, installed in `models/asr/`; see `docs/ARCHITECTURE.md`, section 8e); anything said that is not for the assistant is thrown away at once. Answers can be read aloud with the speaker button, using a voice installed on this computer. While the car moves, typing can be locked (Settings -> Test track); in this test build it is off.
* **Short answer first.** Every answer starts with one or two plain sentences. *Show details* gives the evidence; *Signals around it* draws a chart of the recorded signals around a warning.
* **Tell us when it is wrong.** The thumbs up / down under each answer saves your rating **on this computer**. On thumbs-down you can type what you expected. Nothing is sent anywhere.
* **Left panel.** The vehicle's current state in the replay, tyre pressures, alerts (tap one to ask about it), the front camera and a trip map.
* **Who is using it** (the person button): the driver sees everything about their own car; a **service engineer**, a **fleet operator** and the **validation team** each sign in with a passcode and get their own page (service report, fleet overview, data-source and audit view) - and never see positions, saved places or camera pictures. The **Privacy** page shows what stays on the device and what was checked on this computer just now.
* **Settings** (the sliders icon): read every answer aloud, ask spoken questions at once, units (km/mi, °C/°F, kPa/psi), choose another dataset and the **time speed** (Test track; the default is real time, as in a car): the page then reshapes itself to the vehicle - fuel gauge, engine readings and diesel / petrol specifics for a combustion vehicle, battery and charging for an electric one - because it reads what the vehicle sends, not a setting.  The car then drives the recorded trips of that dataset by itself, in real time, and skips the parked time; the assistant answers about whatever moment it has reached, and raises a warning by itself when the history contains one.

### Testing with the Car Simulator (start menu: *Car Simulator*)
The Car Simulator is a separate program that plays a vehicle and sends what its buses carry to Vantage over the local one-way pipe. Start Vantage first or after, it does not matter.

* **Vehicle** and **Recorded history**: choose a car; if a four-year history of that car is on this computer (the installer puts one for the sample car) it is chosen for you. *Browse...* takes any other history file. Leave it empty for a simulated drive.
* **Speed**: 1 is the car's own pace; 2 to 600 play the history that many times faster, and it can be changed while the car is sending. **Instant - load the whole dataset now** is for testing: the car tells Vantage where the history is, Vantage checks it (it must be a vehicle history of the same car, and the vehicle must not have a history yet), copies it into the vehicle's own store in one go and the car then drives on from the end of it. Nothing is played back; the data link in Vantage shows *Loading history*, and the **Validation** page says how long it took. Command line: `VantageCli car <car> --dataset <file> --instant`. Vantage can be told to refuse this with `VGPT_ACCEPT_HISTORY=0`.
* **Pause**, **Stop car**, **Restart**; the bar shows how much of the history has been played, the line below it says whether the pipe is keeping up or Vantage is the limit.

### A walk through for each person (each one is run on the real application by `tests/test_role_walkthroughs.py`)
* **Driver** (no sign-in): try *"give me a summary of the week"* (state, range, last trip, what needs attention), *"why did my coolant flow drop"* (a hedged explanation: what the data supports, "not a proven cause"), *"what happened before the last warning"* (the ten minutes around it), *"compare city and highway driving"* (both figures, the difference and whether it is more than chance), *"was anything unusual lately"* and *"where am I"* (only the driver is ever told).
* **Service engineer**: person button, *Service engineer*, passcode (set once, when the car is parked).  The **Service** page lists every component with its evidence and, where something needs a look, the next step; a component with no data says *unknown*, never *fine*.  Ask *"what faults were recorded"*: codes with dates, and a plain note when the records are illustrative.  **Export** gives a signed service report, or a signed (and, with a passphrase, encrypted) evidence pack for any one answer.
* **Fleet operator**: the **Fleet** page shows each vehicle by model and an anonymous id, which ones need attention and why, with medians and totals - no driver, no place, no route.
* **Validation team**: the **Validation** page shows signal coverage and rates, the buses and silent sources, injected faults scored against what was recorded (on the generated histories, which have ground truth), and which model the accuracy figures belong to; the **Audit** page is a chained trail of who did what (never what was asked); any answer can be exported as an evidence pack.
* **While the car moves**: typing is locked, so tap a suggested question; the service, fleet and validation pages can only be signed in to when the car is parked.

### The cockpit screen (the page you land on)

The page at `/` is laid out like a car's display. **Top row:** the time of the replay, seven warning lamps (battery, motor, cooling, tyres, camera and lane, charging, warnings; a lamp lights green, amber or red, and pressing one asks the matching question), voice, light or dark, the question browser (**Ctrl K**) and settings.
**Left:** the vehicle readings with a trend line each, and six quick tiles (battery, range, tyres, trips, charging, health). A blue tile means that topic needs attention or is happening now; the outlined tile is the one you asked last.
**Middle:** the instrument cluster (speed on the left arc, battery on the right arc, the car with its four tyre pressures, the gear) and the assistant. **Right:** the battery and charging card, the map of the last half hour, and the camera and events.
Everything is drawn by this program: the fonts (Outfit, Michroma) and the icons (Phosphor) ship in the `ui/modern` folder, and nothing is fetched from the internet. The screen is made for a car display or a laptop; it is not laid out for phones.
The earlier screens are still there: `/classic` (the first assistant page) and `/headunit` (the 1920x720 head-unit simulator).

### Things to ask
```
Brief me                              Is it safe to keep driving?          How's my battery?
When did I last charge?               What did charging cost at 0.30 per kWh?   How long to charge to 80%?
Can I make it 150 km?                 How can I improve my range?          Why is my battery power negative?
What happened before the last warning?   Is the coolant pump OK?           What does that code mean?
Give me a summary of the week         What is my driving score?            Which day was most efficient?
Compare this month with last month    Where do I go most?                  Explain the last warning simply
```
It also understands a few Hindi-English phrases (*"gaadi ka kya haal hai"*), and politely declines what a car cannot know (the weather elsewhere, navigation, music, the future).

### New in 1.1: precise, combinable questions
You can now combine a signal, a period, a condition and a selection in one question, for example:
* "What was the average battery temperature during my three least efficient highway trips last month?"
* "What share of the time was my speed above 100 km/h last week?"
* "Average speed during hard braking yesterday" - then just "and for the highway?" or "what about last week?"
* "Does the car record motor torque?" - if it does not, the answer says exactly what is missing and what would be needed; it never substitutes another signal.
Ambiguous wording ("temperature last week") gets a short question back ("battery, motor, cabin or outside?"). Every number in an answer is checked against the data before it is shown.

### What it will not do
It will not change any vehicle setting, override safety systems, guess numbers it does not have (electricity price, speed limits, battery age) or follow instructions hidden inside logs. If the data does not contain the answer it says so.

## 3. Stopping, updating, removing
Close the console window or press **Ctrl+C**. To update, extract the new release over the old folder (your logs and feedback live elsewhere, see PRIVACY.md). To remove, delete the folder and the `VehicleGPT` folder shown by `python vehiclegpt.py doctor`.

## 4. Getting help
Run `python vehiclegpt.py doctor` and see **TROUBLESHOOTING.md**. If the screen says *"Something went wrong ... (reference abc12345)"*, the reference is in the log file (path in PRIVACY.md); send us that line.
