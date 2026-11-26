# Contributing to VANTAGE

Thank you for helping. Bug reports with a real question and the answer you got are the most valuable contribution.

## The three rules

1. **The model decides what a question means; code calculates.** Do not fix a misread question by adding a phrase pattern. Report it as a failing question: it becomes a glossary line, an ontology entry or a training row (`python tools_triage.py "the question" --expect "act concept ..."` tells you which). Code that influences meaning is visible, listed in `docs/CANON.md`, and counted by the `RuleBudget` test, which fails if a module gains text patterns.
2. **Honest claims only.** Say what was measured, on which machine, and what is only proposed. Latency numbers must name where they were measured (a laptop core is not a Cortex-A72).
3. **No caches of answers or frames, no read-ahead.** A repeated question is a real run, so every latency figure is real.

## Setup and checks

```bash
pip install -r requirements.txt -r requirements-dev.txt
python vehiclegpt.py doctor
python -m pytest -q tests --ignore=tests/test_protocol_oracles.py
```

Protocol oracle tests need extra libraries (cantools, scapy, udsoncan, can-j1939 and others). The model, the adapter and the large car histories are not in the repository; tests that need them skip with the reason.

## Good first contributions

- A new question set for `bench/meaning_eval/` (never edit a frozen set; add a new one and its hash).
- A decoder file for a car with English signal names (`vgpt/vehicle/packs/`, see `documentation/05_data_protocols_storage.md`).
- A fix for something listed in `documentation/13_limitations_and_roadmap.md`.

## Pull requests

Keep changes small and say what you ran. A change that touches model behaviour should include the before and after on the evaluation sets. By contributing you agree your work is released under the MIT licence.
