# Privacy

**Short version:** VehicleGPT runs on your computer, makes no internet connections and sends nothing to us. The one thing that can leave your computer is your voice, if you use the microphone button in a browser whose speech recognition is online (see below).

## What the app does and does not do
* It listens on `127.0.0.1` (this computer only) and refuses requests that do not come from its own page (protection against other websites poking at it).
* It opens **no outgoing network connections**. The assistant, the language model, the classifier and all data are local files.
* The vehicle data is **simulated**; no real person's driving is in this release.

## What is stored, and where
All of it in one folder - on Windows `%LOCALAPPDATA%\VehicleGPT`, on macOS `~/Library/Application Support/VehicleGPT`, elsewhere `~/.local/share/VehicleGPT` (or `VEHICLEGPT_HOME` if you set it). `python vehiclegpt.py doctor` prints the exact path.

| File | Contents | When |
|---|---|---|
| `logs/vehiclegpt.log` (1 MB x 3, rotated) | start-up facts, errors with a reference id, answer timings. **Not the text of your questions** unless you set `VGPT_LOG_QUESTIONS=1`. | always |
| `feedback.jsonl` | your thumbs up / down, the question (first 200 characters), the first line of the answer and your note - **sealed with this installation's key** where the platform allows, **removed after 30 days**, and erasable at once from the privacy page (`what` = `feedback`) | **only when you press a thumbs button** |
| `interaction_log.jsonl` | only if the interpretation-review study was switched on (`--research`) and you agreed to it: what was asked, proposed and decided; sealed, removed after 14 days | **only in a study, only with your consent** |
| the browser's local storage | a single flag that you read the first-run notice, and your option switches | always |
| a temporary folder | this run's working copy of the vehicle history; deleted when the app stops | while running |

To delete everything: close the app and delete that folder. Nothing is uploaded; to give us your feedback you would export it from the service or validation role (`python vehiclegpt.py feedback` shows what is in it).

## Voice
The microphone button uses your **browser's** built-in speech recognition. In Chrome and Edge this normally sends the audio to the browser vendor's speech service. That is the browser's feature, not ours, and it is the only part of the product that may use the internet. Typing never leaves the computer. The spoken answers use the browser's built-in voices and stay local. If you need a guarantee, do not use the microphone button.

## Children, accounts, tracking
No accounts, no analytics, no advertising identifiers, no cookies set by the app.
