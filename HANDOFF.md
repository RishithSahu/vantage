# VANTAGE hand-off (2026-10-14)

Goal: a Qwen2.5-0.5B-Instruct-Q4 + LoRA **learned query-plan parser** ("learn the meaning, validate the structure, calculate deterministically") for the vehicle-data assistant. No Pi deployment has been done; do not deploy without the owner's go-ahead. Never type or store the Pi password.

## Status
| item | state |
|---|---|
| frozen, never overwrite | `models/qp-lora.gguf` (v12), `qp-lora-v13.gguf`, `qp-lora-v14d_s14{1,2,3}.gguf` (v14d, **reference**), `qp-lora-v15.gguf` (not promotable) |
| candidate | `models/qp-lora-v16.gguf` + `qp-lora-v16.gguf.contract.json` (`outcome_contract: 2`), HF adapter `training/out/qp_v16/`, init from `training/out/qp_v14d_s143`. evaluated: `results/semantic_recovery/major_semantic_recovery_report.md` (not approved for promotion) |
| 500-QA benchmark | `VehicleGPT_500_QA.csv`: evaluation only, never train on it; v14d hybrid 97/500, v15 hybrid 95/500 (frozen rubric `results/semantic_recovery/v15/rubric_v1.json`). v16: learned-only 112, hybrid 120 (see major_semantic_recovery_report.md; in-scope telemetry NOT better than v14d) |
| tests | `python -m pytest -q tests` -> 758 passed, 2 failed (headwind test needs `VGPT_QP_LORA=models/qp-lora-v14d_s143.gguf`; torture-set route-label oracle) |

## Read first
1. `results/semantic_recovery/semantic_contract.md` - the single semantic contract
2. `results/semantic_recovery/semantic_pipeline_map.md`, `semantic_root_cause_report.md`, `semantic_data_audit.md`
3. `results/semantic_recovery/v15_final_comparison.md` - why v15 was rejected
4. `results/semantic_recovery/evaluation_contract.md`, `v14_training_report.md`

## What changed in this phase
* `vgpt/outcomes.py` (typed outcome categories, contract versions), `vgpt/lexicon.py` (central lexicon), grammar `why` enum gained `greeting`, `unsafe`
* `vgpt/learned_front.py` grounding repairs (negative thresholds, `k=1`, compound/hyphenated durations, fortnight, equal-length windows, no fuzzy "right"->eight); `temporal_ops.py` / `temporal_misc.py` fixes; `schema_view.py` op help
* tests: `test_outcome_contract.py`, `test_validator_msr.py`, `test_msr_repairs.py`, `test_lexicon.py`
* data/training: `training/qp/v16_semantic/` (`relabel.py`, `gen_v16.py`, `build_v16.py`, `independent_eval.py` + `.jsonl`, `evaluate_v16.py`, `evaluate_e2e.py`, `data_v16/`)

## Rules that must stay
no phrase-trigger rules; rules may normalise and validate, never choose the operation or write the answer; do not tune anything against the 500-QA or the independent set (`independent_eval.jsonl`, sha256 `1ad1808e38b8a798...`); frozen adapters immutable (hashes in `results/semantic_recovery/msr/baseline_hashes.txt`); report true denominators; use `Query.to_text()` for every label; in this shell write regex files with an editor tool, not heredocs (backslashes get halved -> 0x08).

## Setup on the new machine
```
pip install -r requirements.txt   # llama-cpp-python, numpy; training also needs torch + transformers + safetensors + gguf
python -m pytest -q tests/test_outcome_contract.py tests/test_validator_msr.py tests/test_lexicon.py
```
Large models not included (re-download if needed): LFM2.5, granite, xLAM, qwen2.5-1.5b, qwen25_7b_ref. `data/fixtures` (270 MB) is not included; a few legacy tests that use it will skip.

## Commands
Rebuild data: `python training/qp/v16_semantic/build_v16.py`
Train (GPU, ~35 min): `python training/train_lora.py --data training/qp/v16_semantic/data_v16 --out training/out/qp_v16 --init-adapter training/out/qp_v14d_s143/adapter.safetensors --epochs 3 --lr 1e-4 --seed 161`
Convert: `python -m training.convert_lora_gguf training/out/qp_v16 --out models/qp-lora-v16.gguf` (+ `.contract.json` sidecar)
Plan-level eval: `python training/qp/v16_semantic/evaluate_v16.py --adapter models/qp-lora-v16.gguf --name v16 --sets independent,holdout,val,dev`
End-to-end: `python training/qp/v16_semantic/evaluate_e2e.py --adapter models/qp-lora-v16.gguf --name v16`
500-QA (once): `python bench/qa500_run.py --mode learned_only --adapter models/qp-lora-v16.gguf --name v16_lo --out results/semantic_recovery/v16/runs` (and `--mode learned` for hybrid), then `bench/qa500_score.py`, `bench/qa500_compare.py` (add the new configs)

## Next steps
1. read the v16 plan-level + e2e results; compare with v14d on the independent set (per category: outcome, time, negation, context) and on contract-corrected dev/challenge/holdout
2. only if v16 does not regress in-scope telemetry or any outcome category: run the 500-QA once, blind-audit a sample, write `major_semantic_recovery_report.md`, `semantic_model_comparison.csv`, `semantic_training_manifest.json`, `semantic_failure_taxonomy.csv`
3. open issues: gate abstentions on natural phrasing (28/32 low-confidence in v14d learned-only), legacy fuzzy "follow-up" routes, typo'd period words
