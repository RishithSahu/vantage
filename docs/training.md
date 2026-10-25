# Training the query-compiling model

> **Status.** Fine-tuning is an optional, developer-side tool, not part of the shipped behaviour: the model path is **opt-in** (`VGPT_IR_SLM=1`) and the product works on the rules plus the older routes without any adapter. One adapter (Qwen2.5-0.5B) was trained for the benchmark in `evaluation.md`; no further training is planned (a Qwen3-0.6B adapter was started and stopped, so Qwen3 is reported as a base model only). Only training may use the whole development machine; tests, evaluation and benchmarks run pinned to 2 cores (`tools_run_tests.py`, `training.evaluate --cores 2`).

The language model's whole job is *question → one line of Query IR* (see `query_compiler.md`). That is a small, closed, verifiable task, which is why a 0.5 B model with a LoRA adapter can do it on a 2-core, 2 GB target.

```
training/
  phrases.py            surface vocabulary (signal names, times, conditions, events, off-topic and attack text)
  gen_queries.py        formal-template generator: sample a valid IR first, render the question from it
  build_dataset.py      execute every IR against a real database; keep only what runs and verifies; write the splits
  lora.py               minimal LoRA (no peft dependency)
  train_lora.py         assistant-only-loss training on a GGUF base
  convert_lora_gguf.py  adapter -> GGUF LoRA, shipped next to the unmodified base model
  evaluate.py           metrics for rules / base model / base+LoRA / hybrid
  feedback_to_training.py   reviewed user feedback -> approved training examples
```

## 1. Data: formal templates, labels correct by construction

`python -m training.build_dataset --n 26000 --adv 900 --seed 0`

The generator samples the **IR first** (valid by construction: statistic matches signal type, conditions are not contradictory, units are real) and renders the question from it with 5 sentence templates per operation, several phrasings per signal / time / condition, number words vs digits, units (km/h, mph, °C, °F, psi, newton metres …), fillers, and light noise. There is no human or model labelling noise. Off-topic questions and **attacks** (SQL, shell, prompt-injection, vehicle-control commands) are labelled `op=none`.

Every IR is then **executed** (`build_dataset.py`): it must round-trip through text and JSON, validate, run through planner → executor → evidence → composer → verifier on `data/gen_courier_ext.db`, and produce a verified answer or a verified refusal. Examples that do not are dropped and counted. This found and fixed real defects in the engine while the dataset was being built (the virtual tyre-pressure concept failed to load next to other `ext_1m` columns; the verifier mistook "12 V battery" and "95th percentile" for measurements; estimates that returned no data were mislabelled) — a label that does not run is a bug, never training data.

The default build keeps ~17.7 k training examples, ~1 k validation, and 5 held-out test sets of up to 800 examples each. Outcomes of the executed gold queries are recorded (answered / no_data / missing_data / unsupported / refused) together with a compact numeric gold result, so evaluation can compare **execution results**, not only text.

## 2. Compositional and out-of-distribution splits

| split | what it holds out |
|---|---|
| `train`, `val`, `test_iid` | random 90 / 5 / 5 of what is not held out |
| `test_comp` | **combinations of familiar parts**: {coolant flow, climate power, motor temperature} × {highway, charging} conditions; the pairs of groups {night, cold}, {raining, highway}, {hot weather, city} in comparisons; rankings of {city, rainy} trips; 95th percentile × last month, std × last week, median × this month. Each part appears elsewhere in training; the *combination* never does. |
| `test_template` | sentence templates 3 and 4 (never used in training) |
| `test_phrasing` | signal names the product has never seen (`UNSEEN_TARGET_PHRASES`: "twisting force at the motor", "juice left", …) |
| `test_adv` | typos, upper case, run-together words, speech-to-text style, fillers, 1–3 noise operations on fresh questions, off-topic and attack text |
| `bench/ir_gold.tsv` | 70 **hand-written** questions in everyday wording with hand-written IR (executed for gold results) — independent of the generator |

`tests/test_ext_training.py::Generator` asserts that no held-out combination, template or unseen phrase occurs in the training split, that every label is a valid round-tripping query, that generation is deterministic, and that texts are unique across splits.

## 3. LoRA training

`python -m training.train_lora --epochs 2` (≈ 20–30 min on a laptop RTX 3050 6 GB; 5.5 GB peak)

* Base: the shipped GGUF (`qwen2.5-0.5b-instruct-q4_k_m.gguf`), de-quantised by transformers on load, so the adapter learns to correct the *quantised* weights llama.cpp will actually run (QLoRA-style).
* LoRA r = 16, α = 32 on q/k/v/o/gate/up/down of every block (8.8 M parameters, 1.4 %), bf16 base, fp32 adapter, AdamW 2e-4 with warm-up and cosine decay, batch 16, 2 epochs.
* **Loss on assistant tokens only** (the query line and the end-of-turn token); the system prompt and the question are context. The vocabulary is 152 k wide, so logits are computed only at answer positions.
* Prompt: one-line system prompt (`query_grammar.SYSTEM_TUNED`) — the adapter has learned the format, so no few-shot text is needed; that keeps the prompt prefix tiny and cacheable.
* The environment's `peft` was incompatible with its `transformers`; `lora.py` is ~50 lines of PyTorch with peft-compatible tensor names, so the adapter also loads in peft tooling.

## 4. Shipping the adapter

`python -m training.convert_lora_gguf training/out/qwen25_05b --out models/ir-lora-qwen25-05b.gguf`

The adapter is a separate ~35 MB GGUF LoRA file (f16) loaded with `lora_path=` on top of the unmodified base. The base file never changes; the adapter can be replaced over the air; deleting it restores the original model (and `autodetect()` then disables the model path entirely — the base model alone is not trusted to compile queries). `tests/test_ext_training.py::LoRA` round-trips an adapter through the converter and reads it back with the `gguf` reader.

## 5. Evaluation

See `evaluation.md`. All numbers there come from running the commands there on this repository's data.

## 6. Feedback → reviewed training data

Thumbs-down questions and questions the assistant could not place become **candidates**, never training data:

```
python -m training.feedback_to_training collect                       # scrub PII, quarantine attacks, propose an IR
#  a person edits training/data/feedback_candidates.jsonl: fill "ir", set "status": "approved" | "rejected"
python -m training.feedback_to_training approve training/data/feedback_candidates.jsonl --db data/gen_courier_ext.db
python -m training.feedback_to_training merge                         # appends approved examples to train.jsonl
python -m training.build_dataset ... && python -m training.train_lora ... && python -m training.evaluate ...
```

`approve` re-checks mechanically: the IR parses, validates, **executes**, is **grounded** in the question, is not a duplicate of any existing train/test text (approved data only ever goes to training — no leakage into held-out sets), and attack-like text is refused. Unreviewed lines are ignored. The reviewer is the safety boundary: the model is never retrained on its own output or on raw user text.

## 7. Updating for a new signal

1. Add the concept (and its storage mapping, if recorded) to `signal_registry.py`; add paraphrases to `EXTRA_SYNONYMS`.
2. Rebuild the dataset (`build_dataset`) — the generator, the parser and the model grammar pick the new concept up automatically.
3. Retrain, convert, evaluate, compare with the previous adapter on the same test sets before shipping.
