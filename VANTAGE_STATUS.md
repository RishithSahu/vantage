# VANTAGE - status (8 Oct 2026)

The previous status report (rule/hybrid era, 7 Oct) is archived in `docs/history/VANTAGE_STATUS_2026-10-07_hybrid_era.md` and must not be quoted: it recommends a hybrid route that the product no longer has.

**Current state, in one paragraph.** The product is a model-only assistant: question + conversation state -> capability retrieval -> one frozen-model typed plan -> structural validation -> read-only execution with the evidence contract -> answer, plus an optional editable interpretation preview (`/review`) in which a person can inspect and correct the typed interpretation before anything runs. There is no rule parser and no hybrid fallback in the product (`vgpt/product.make_brain`; the old rule implementation remains only as `make_legacy_brain`, a measured baseline). The 90% autonomous-accuracy target and the 2 GB / two-core / 2 s deployment target are **not met** by any measured configuration; the evidence is in `results/LEDGER.md` and the decision in `results/DECISION_REPORT.md`.

* Constraints kept: no training or fine-tuning, no AWS, QEMU stopped, sealed acceptance file never read, every model download approved first with hashes recorded.
* Reproduce the best autonomous configuration: `python tools_pin.py 8,9 bench/diag_format.py --model models/granite-4.0-1b-Q4_K_M.gguf --n-ctx auto --stride 2 --format intent --decode grammar --shots core --demos 8 --demo-cards no`.
* Run the prototype: `python ui/server.py` then open `/review`; layout check without a model: `python bench/serve_review_stub.py`.
* Tests: `python -m pytest tests -q` (model-dependent tests are opt-in; the UI and concurrency tests use the scripted test double `tests/stub_planner.py`).
* Open items: user study with real participants; acceptance freeze/review; ARM measurement (QEMU on hold); a model-based demonstration of the analytical features (event context, history, comparison, missing data); requirement-to-evidence table refresh.
