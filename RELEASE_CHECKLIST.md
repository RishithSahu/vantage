# Release checklist (internal)

"No complaints" cannot be promised for any product. What can be done is: remove every avoidable failure before people see it, be honest about what remains, and make the remaining complaints cheap to receive and fix. Status as of 1.0.0.

## A. Automated - `python tools_release_gate.py` (all must PASS)
| Gate | What it proves |
|---|---|
| unit / integration tests (226) | behaviour of the assistant, UI server, doctor, feedback, release builder, robustness |
| understanding floors | unseen everyday wording: tuning sets >= 95%, final held-out >= 90%, messy speech-to-text style >= 90% |
| older analysis routes | the 365 labelled analysis questions and the 112-question extended eval are unchanged |
| off-topic | 25 off-topic questions never answered as if about the car |
| package | allow-list zip, no dev paths / secrets, manifest hashes, extracted copy passes `doctor` and `smoke` with a clean environment |
| fresh environment | a brand-new virtual environment with only `requirements.txt` installed runs the extracted package |

## B. Built in this release (done)
- [x] One launcher (`run / doctor / smoke / feedback / version`), free-port handling, readable failure messages, optional model degrades to rules automatically (also if the model fails to load at run time).
- [x] Install scripts (Windows / macOS / Linux), `requirements.txt` with only what is used (numpy, opencv-headless); no scikit-learn / scipy at run time (tested).
- [x] Security: 127.0.0.1 only; Host / Origin / Sec-Fetch-Site / JSON-only checks (DNS-rebinding and cross-site requests); 64 KB request cap; camera source limited to video files; security headers + CSP; no stack traces or paths in errors (reference id instead); question text stripped of markup and control characters; XSS-safe rendering (tested in the browser).
- [x] Privacy: no outgoing connections (tested by blocking every non-loopback socket during the self-test); logs without question text; working copy of the data in a private temp folder; nothing written into the install folder; feedback only on button press.
- [x] Feedback loop: thumbs up / down + "what did you expect?" on every answer, stored locally, `vehiclegpt.py feedback` ranks routes and writes a triage list (`--tsv`) to add to the training wordings.
- [x] Honest UI: first-run notice (advisory only, simulated data, voice caveat), footer disclaimer, connection-lost banner, retry button, "busy" and "something went wrong (reference ...)" messages, microphone disabled with an explanation where unsupported, keyboard-operable controls, reduced-motion support.
- [x] Documents: user guide, troubleshooting, privacy, known limitations, third-party notices, changelog.

## C. Needs a person (NOT done - these block a public release)
- [ ] **Choose and add `LICENSE.txt`** (or an "evaluation use only" notice). Not our decision to make.
- [ ] **Resolve the VERIFY lines in `THIRD_PARTY_NOTICES.md`**: VIF-AEGIS and HCRL dataset terms, comma2k19 dataset terms, Open-Meteo commercial-use terms and attribution. The shipped databases are derived from some of these.
- [ ] **Run it on a second, ordinary computer** (not the developer's) from the zip, following only the USER_GUIDE. Time it; note every stumble; fix the guide.
- [ ] **Try three real users** for ten minutes each, watching (not helping). Collect thumbs-down feedback with `vehiclegpt.py feedback`, fix the top routes, retrain (`python tools_train_clf.py --ship`), repeat.
- [ ] Decide on the language model: bundle it (`tools_release.py --with-model`, +490 MB, Apache-2.0 notice included) or leave it optional (current default).
- [ ] Re-run the ARM-VM benchmark (`python vm/driver.py refresh`, ~35 min) after the last code change if the 2 GB / 2-core claims are part of the release; the README numbers predate the classifier.
- [ ] Code-sign the Windows scripts / zip if the audience's IT blocks unsigned downloads.

## D. Risks we know and accept (also in KNOWN_LIMITATIONS.md)
- About 1 in 10 unseen everyday questions will not get the intended answer (they get suggestions, not a wrong confident answer, in most cases).
- Voice input depends on the browser (Chrome / Edge) and may use the vendor's online service.
- The data is simulated; it demonstrates the assistant, not a particular real car.
- Statistical answers on 4 weeks of data carry wide uncertainty and say so.
