# VehicleGPT 1.1 - Query compiler: final report

## 1. What changed, in one paragraph
Language understanding was split into **compiling** a question into a typed Query IR and **executing** it. Rules parse first (~0.4 ms); a small model with an optional LoRA adapter is an opt-in fallback. Validation, capability analysis, SQL planning, read-only execution, evidence, wording and verification are deterministic code. The older routes remain the default for everything they already answered (`VGPT_QC=auto|first|shadow|off`).

## 2. How it works now
`question -> nl2ir (rules) -> [IrSlm + LoRA, opt-in] -> Query IR -> validator -> capability -> planner -> executor -> evidence -> composer -> verifier -> answer`
* **Capability-aware**: unknown signal / known but not recorded / not derivable from what is recorded, with exact missing signals and the formula. No electrical-for-mechanical substitution; estimates labelled with assumptions.
* **Compositional**: time x entity selection x conditions x statistic x signal in one query.
* **Verified**: every number/unit checked against evidence; no bare causal claims; no control instructions.
* **Safe by construction**: grammar with no free text, validator string scan, whitelist SQL compiler (SELECT only, bound parameters), read-only connection.
* **Follow-ups and clarification**: "what about yesterday?", "and for city?", "which temperature?".
* **Sensor quality**: dropouts, stuck values, range, spikes, steering bias, cross-sensor disagreement.
* **Simulator extension**: 23 extended signals + 5 fault scenarios with ground truth, additive schema (`xsig_1s`, `xsig_1m`).
* **Training pipeline** (`training/`): formal-template generation, execution-verified dataset, compositional splits, LoRA (assistant-only loss), GGUF adapter conversion, evaluation, reviewed feedback workflow.

## 3. What was fixed
Defects exposed by executing 26,900 generated queries: tyre pressure failed to load beside other columns; verifier mistook "12 V battery" / "95th percentile" for measurements; no-data estimates mislabelled; SQL guard missed `load_extension`; stuck detection needed 2 h runs; live reading used minute means. Parity fixes: "last week/month" = last 7/30 days; live reading from newest second. Grounding check false positives cut from 13% to 2.6%.

## 4. Files
Created: vgpt/{signal_registry, query_ir, query_validator, capability, derived, frame, sensor_quality, query_planner, query_executor, evidence, answer_composer, answer_verifier, qc, nl2ir, query_grammar, ir_slm, datagen_ext}.py; training/*; tools_datagen_ext.py; tools_run_tests.py; bench/ir_gold.tsv, bench/qc_runtime.py; tests/test_qc.py, test_parity.py, test_ext_training.py; docs/{query_compiler, signal_ontology, training, evaluation}.md.
Modified: vgpt/brain.py (hook + state), vgpt/version.py, README, DESIGN, USER_GUIDE, KNOWN_LIMITATIONS, CHANGELOG.
Untouched: old routes, datagen, existing databases, old tests.

## 5. Migration, IR, registry, model role, training
See docs/query_compiler.md, docs/signal_ontology.md, docs/training.md. Migration is idempotent and never alters existing tables. The model only chooses registry names and copies numbers; it never sees data.

## 6. Benchmark (2 cores, 2 threads; details in docs/evaluation.md)
Exact match, hand-written 70 questions (about +/-10 points): rules 70%, Qwen2.5-0.5B base 14%, Qwen3-0.6B base 11%, Qwen2.5-0.5B + LoRA 76%, rules + LoRA fallback 80%.
LoRA on generated sets: 97.7% in-distribution, 89% held-out combinations, 91% noisy/adversarial, 72% unseen wording. Base models refuse 0% of attacks/off-topic; with LoRA 97%. LoRA model drops mph/F units (guard rejects; rules handle them).
Runtime without model: p50 4 ms, p95 95 ms, peak 89 MB. Model path: ~1.3-1.8 s per query (16-23 tokens). Qwen3 benchmarked as base only (LoRA stopped at your request). ARM/QEMU not re-run.

## 7. Tests
297 tests pass pinned to 2 cores (226 older + 71 new): `python tools_run_tests.py`.

## 8. Commands
```
python tools_run_tests.py
python tools_datagen_ext.py data/gen_courier.db --out data/gen_courier_ext.db
python -m training.build_dataset --n 26000 --adv 900
python -m training.train_lora --epochs 2            # dev machine only
python -m training.convert_lora_gguf training/out/qwen25_05b --out models/ir-lora-qwen25-05b.gguf
python -m training.evaluate --backend hybrid --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora models/ir-lora-qwen25-05b.gguf
python bench/qc_runtime.py
```
ARM guest: run the last two inside the guest with the existing `vm/` harness.

## 9. Limitations
See KNOWN_LIMITATIONS.md ("Query compiler (1.1)") and the "not measured" section of docs/evaluation.md.

## 10. About this zip
Contains the full project except: the base model files (re-download or copy into `models/`; the 17.6 MB LoRA adapter is included), the VM images, old zips/dist, caches and WAL files. Qwen base models: qwen2.5-0.5b-instruct-q4_k_m.gguf, Qwen3-0.6B-Q8_0.gguf.
