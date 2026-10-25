# Evaluation of the query compiler

Every number below was produced by running the command next to it on this repository's data. Nothing is estimated. Raw results: `results/qc_eval_*.json` (with the 25 first errors per split) and `results/qc_runtime.json`.

**Conditions.** Evaluation and benchmarks run **pinned to 2 CPU cores with 2 inference threads** (`training.evaluate --cores 2 --threads 2`) to resemble the target's budget. Only LoRA *training* used the whole development machine (RTX 3050 6 GB). The machine's cores are much faster than a Cortex-A55, so absolute latencies on the target are higher (see "What was not measured").

## Method

* **Sets** (`training/data/*.jsonl`, `python -m training.build_dataset`): `test_iid`, `test_comp` (held-out *combinations* of familiar parts), `test_template` (sentence shapes never trained on), `test_phrasing` (signal names never seen by the product), `test_adv` (typos, noise, STT style, off-topic questions, SQL / shell / prompt-injection / vehicle-control attacks), plus **`bench/ir_gold.tsv`**: 70 hand-written everyday questions with hand-written IR (independent of the generator). All gold IR is *executed* to obtain gold results. See `training.md`.
* **Backends**: `rules` (the deterministic parser alone); `Qwen2.5-0.5B base` and `Qwen3-0.6B base` (few-shot prompt, grammar-constrained, no adapter); `Qwen2.5-0.5B + LoRA` (one-line prompt, grammar-constrained); `hybrid` = rules first, the LoRA model only when the rules are unsure (the production design when the model path is enabled).
* **Subsets**: SLM runs use the first 300 examples of each split (150 for the two base models, which are slow); rules use 800. The gold set is always all 70. With n = 300 a percentage is good to roughly +/- 5 points; with n = 70 to about +/- 10 points.
* **Metrics**: `ir_exact` (predicted query equals gold query, compared as data); `op / signal / time / condition / aggregation` accuracy of the parts; `unit` accuracy of thresholds that carry a non-registry unit (mph, °F, psi ...); `executable` (the prediction validates and runs through the whole pipeline with a verified answer); `numeric` (executing the prediction gives the same status and value as executing the gold query); `refusal` (off-topic / attack text becomes `op=none`); `missing` (gold queries about unrecorded signals remain recognised as such); `halluc` (predicted query names a number / signal / event / group that the question does not contain); `verified` (composed answers that pass the answer verifier); latency and output tokens per query.

## Results (percent; latency in ms)

### Which backend compiles the question

`ir_exact` / `numeric` accuracy:

| backend | test_iid | test_comp | test_template | test_phrasing | test_adv | ir_gold (hand-written) |
|---|---|---|---|---|---|---|
| Rules only (n 800) | 85.1 / 87.4 | 77.6 / 81.6 | 80.6 / 84.8 | 14.0 / 18.2 | 59.5 / 57.0 | 70.0 / 73.0 |
| Qwen2.5-0.5B base, few-shot (n 150) | 2.7 / 15.3 | 0.7 / 9.3 | 1.3 / 16.7 | 6.0 / 19.3 | 5.3 / 17.5 | 14.3 / 23.8 |
| Qwen3-0.6B base, few-shot (n 150) | 0.0 / 6.0 | 0.0 / 4.7 | 1.3 / 12.0 | 0.0 / 4.7 | 1.3 / 10.2 | 11.4 / 12.7 |
| **Qwen2.5-0.5B + LoRA** (n 300) | **97.7 / 98.7** | **89.3 / 91.0** | **88.7 / 93.3** | **72.3 / 80.0** | **90.7 / 93.2** | **75.7 / 79.4** |
| Hybrid: rules + LoRA (n 300) | 91.7 / 94.0 | 83.7 / 88.0 | 91.0 / 96.0 | 63.7 / 73.7 | 87.7 / 90.5 | 80.0 / 85.7 |

Reading it honestly:

* The **base models cannot do this task** even with 12 worked examples and a grammar that guarantees well-formed output (0-6 % exact match; their ops are right about half the time, their numbers and signals mostly wrong; Qwen3-0.6B Q8_0 additionally loops inside the grammar and is ~3x slower). Grammar-constraining makes the output *safe*, not *right*.
* The **LoRA adapter** is what makes the model usable: 97.7 % on in-distribution questions, 89 % on held-out *combinations* it never saw (the point of the compositional split), 91 % on noisy / adversarial text.
* **Held-out wording is the weak spot**: with signal names the product never saw ("twisting force at the motor") the exact match falls to 72 % (op, time and aggregation stay at 99 %; the signal choice is what fails, 79 %). The rules fall to 14 % there, which is expected - they only know the registry vocabulary.
* **Hybrid is not better than the model alone on these sets**, because the rules are *confidently wrong* on some sentences they only half understand (coverage 72 % is accepted). It is better on the hand-written gold set (80 vs 76 %) and about 10x faster on average. Tightening the rules' acceptance threshold would trade speed for accuracy; this has not been tuned.
* On the **hand-written gold set** every backend is lower than on generated data (rules 70 %, LoRA 76 %, hybrid 80 %): generated text is easier than people. n = 70, so +/- 10 points.
* `unit` accuracy of the LoRA model is **0 %** on the examples that carry a spoken non-registry unit (mph, °F): it copies the number and drops the unit. The grounding check now rejects such a query ("the threshold 40 was spoken in mph but the query does not say so"), and the hybrid path gets these right through the rules (hybrid 83 % on iid), but the model alone is unreliable for unit-bearing thresholds.

### Safety and hallucination

| backend | refusal of off-topic / attacks (test_adv) | hallucination rate (test_iid) | verified answers (test_iid) |
|---|---|---|---|
| Rules only | 97.9 | 9.7 | 92.2 |
| Qwen2.5 base | **0.0** | 25.3 | 74.0 |
| Qwen3 base | **0.0** | 70.0 | 38.7 |
| Qwen2.5 + LoRA | 97.3 | 8.0 | 99.7 |
| Hybrid | 94.6 | 8.0 | 99.0 |

* The base models turn **every** off-topic question and attack into some query (0 % refusal). With the adapter 97 % become `op=none`; the remaining ~3 % were still stopped by the grammar (no free text), the validator and the read-only executor - see the security tests.
* The hallucination column is the grounding check's verdict on the *predicted* query. The check was tightened during the benchmark (implied numbers such as "last week" = 7 days, the default period, and a spoken-unit check were added because the first version flagged 13 % of *correct* gold queries; it now flags 2.6 %). The Qwen2.5+LoRA, rules and hybrid rows were measured with the earlier, stricter version, so their rates are upper bounds; the base-model rows are not affected in substance (they are 25-78 %).
* `verified` is 99-100 % for LoRA / hybrid: when a query is produced it executes and its answer passes the verifier; the failures are in choosing the query, never in stating unsupported numbers.

### Latency, tokens and memory

Model path, pinned to 2 cores / 2 threads on the development machine (mean / p95 ms, output tokens):

| backend | test_iid | test_comp | gold | tokens |
|---|---|---|---|---|
| Qwen2.5-0.5B + LoRA | 1580 / 2560 | 1798 / 2343 | 1325 / 2052 | 16-23 |
| Hybrid (rules first) | 92 / 1095 | 163 / 1600 | 245 / 1452 | 1-1.6 average |
| Qwen2.5-0.5B base | 1695 / 2423 | 1681 / 2191 | 1383 / 1721 | 22-25 |
| Qwen3-0.6B base (Q8_0) | 4436 / 8003 | 4582 / 7916 | 3624 / 8745 | 43-67 |

Rules only: < 1 ms parse. Process memory during the model benchmarks: 720-750 MB (Qwen2.5, includes the 0.5 GB model), 997 MB (Qwen3 Q8_0). The adapter file is 17.6 MB (f16).

Rules + engine only (no model), 2 cores, one `Brain`, 153 gold questions (`bench/qc_runtime.py`):

| stage | mean ms |
|---|---|
| parse | 0.4 |
| validate | 0.05 |
| execute | 10.0 |
| compose | 0.04 |
| verify | 0.2 |
| **end to end** | **p50 4.0, p95 94.5, max 110** |

Peak process memory **89 MB** (36 MB at start). The full test suite (297 tests, 226 older + 71 new) passes pinned to 2 cores in ~4 minutes; the test process peaks at 1.8 GB because it builds many `Brain` objects and imports PyTorch for the LoRA unit tests - that is the test harness, not the assistant.

### Other measured results

* **Execution-verified dataset**: of 26 900 generated questions every one validated and executed with a verified answer or refusal after the engine defects this process uncovered were fixed (listed in the CHANGELOG). Outcomes of the executed gold queries: 18 778 answered, 7 064 no data in the period, 385 missing data (signal not recorded), 271 unsupported, 402 refused.
* **Sensor-quality engine vs injected ground truth** (`tests/test_ext_training.py`): all 5 injected fault scenarios (stuck wheel sensor, wheel dropout bursts, steering offset, yaw spikes, brake-pressure over-range) are detected with the right flag; a clean wheel signal and the untouched databases produce no stuck / range / bias flags.
* **Parity** with the older routes (`tests/test_parity.py`): yesterday's energy, the 14-day battery temperature peak, the lowest state of charge, and the live reading are identical; rolling-week energy and consumption agree within 5 % / 3 % (a partial day); hard-braking counts within 5 %.
* **Security tests** (`tests/test_qc.py::SecurityTests`): SQL / path / shell / script payloads in every text field are rejected by the validator; the SQL compiler refuses non-`SELECT`, `PRAGMA`, `ATTACH`, `load_extension`, comments and `;`; vehicle-control phrases never become queries; the verifier refuses answers that instruct a control action, state ungrounded numbers or make bare causal claims.

## What was not measured / not done

* **Qwen3-0.6B was benchmarked as a base model only.** A LoRA for it was started and **stopped at your request** (you asked to stop fine-tuning), so there is no Qwen3 + LoRA row. Qwen2.5 + LoRA was trained once on the final dataset (2 epochs, 17 650 examples, validation token accuracy 99.95 %, 16 min on the GPU).
* **The ARM / QEMU environment was not re-run for this work.** The numbers above are from an x86 laptop pinned to 2 cores; no ARM latency or memory figure exists for the current configurations and none is estimated here. To measure on the emulated target use the existing harness (`vm/`, `bench/run_guest.py`); results from it are emulation timings. Current configuration-specific figures are in `VANTAGE_STATUS.md`.
* The hand-written gold set is small (70). The generated sets share a generator with the training data, so their high numbers are an upper bound on real-world accuracy (the gold set is the better guide).
* No A/B with real users; thumbs-up/down feedback would feed `training.feedback_to_training`.

## Reproduce

```
python -m training.build_dataset --n 26000 --adv 900 --seed 0
python -m training.evaluate --backend rules  --sets test_iid,test_comp,test_template,test_phrasing,test_adv,bench/ir_gold.tsv
python -m training.evaluate --backend hybrid --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora models/ir-lora-qwen25-05b.gguf --max 300 --sets ...
python -m training.evaluate --backend slm    --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --max 150 --sets ...        # base model
python bench/qc_runtime.py
python tools_run_tests.py                                                                                                   # whole suite, 2 cores
```
