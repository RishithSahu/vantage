# The query compiler

The assistant's language understanding is split into **compiling** a question into a typed query and **executing** it. The language model (when one is used at all) only does the first part. Everything after it is deterministic code you can test.

```
question
  │  1. nl2ir.parse           fast deterministic rules, ~1 ms, reports how much of the sentence it explained
  │  2. ir_slm.IrSlm          only if the rules are unsure: grammar-constrained model, then a grounding check
  ▼
Query IR  ──  query_validator ──  capability ──  query_planner ──  query_executor ──  evidence ──  answer_composer ──  answer_verifier
   (data,        structure &       what this      trusted SQL       read-only SQLite,   numbers,       template wording,    refuses what the
   never code)   vocabulary        vehicle can    compiler          tiered data,        quality flags, never free text     evidence cannot
                                   answer, and    (whitelists)      derived metrics     causal graph                       support
                                   why not
```

`qc.QueryCompiler.execute(query)` runs stages 3–9 and times each one; `Brain.qc_try` wires stages 1–2 in front of it. Anything that cannot be handled returns `None` and the previous engine (the "legacy" routes) answers exactly as before.

## The IR (`vgpt/query_ir.py`)

A `Query` is *data*: registry names, numbers and ISO dates — no field is ever executed.

| field | meaning |
|---|---|
| `op` | `current_value` `aggregate` `extreme` `trend` `compare` `count` `list_events` `event_context` `rank` `correlation` `percent_time` `explain_event` `capability` |
| `target`, `target2` | registry concepts (signals or derived metrics); `target2` only for correlations |
| `agg` | `mean min max median std first last delta count p95 sum percent_time mode` (type-checked: no mean of an accumulating quantity, no sum of a temperature) |
| `scope.time` | `current` · `relative(n, minute/hour/day/week/month)` · `calendar(yesterday, last_week, last_month, …)` · `date(YYYY-MM-DD)` · `all` |
| `scope.conditions` | named predicates (`highway`, `charging`, `hard_braking`, …) and/or signal thresholds `(signal, gt/lt/ge/le/eq, number, unit)` |
| `scope.select` | entity selection: `trip` or `charge_session`, optional class, ranked by an entity metric, ascending/descending, top-k |
| `scope_b`, `groups` | the second period, or the two named groups, of a `compare` |
| `event` | event type, selector (first/last), context window before/after in seconds |
| `unit` | the unit the person asked for |
| `refs` | which parts a follow-up inherited from the previous query |

Two equivalent encodings: **JSON/dict** (`to_dict` / `from_dict`, unknown fields are rejected, not ignored) and a **compact text DSL** (`to_text` / `from_text`) that the language model writes:

```
op=aggregate tgt=battery_temperature agg=mean t=cal:last_month c=sig:speed:gt:60:mph sel=trip:wh_per_km:desc:3:highway
op=percent_time t=cal:last_week c=sig:speed:gt:100
op=compare tgt=wh_per_km agg=mean t=last:14:day grp=city,highway
op=count t=last:14:day ev=HARSH_BRAKE
op=none                                   # not a question about this vehicle / not a data query: the model refuses
```

Numbers and units are **copied from the question** (`60:mph`); conversion to the registry unit is deterministic code (`Query.normalized()`), never the model's arithmetic. Defaults are not written (`evs=last`, `aft=0`), which keeps the model's output to ~15–35 tokens.

### Compositionality

"Average battery temperature during my three least efficient highway trips last month" is **one** query — `aggregate(battery_temperature, mean)` over `Scope(last_month, select=trip[highway] ranked by wh_per_km desc, k=3)` — not a new intent. Time, selection and conditions combine freely with any target and statistic; the executor intersects them (trip intervals ∩ condition mask) before aggregating.

## Stages

* **Validator** (`query_validator.validate`) — closed vocabulary, type rules (statistic vs signal type, unit family, calendar names, real dates, `k ≤ 50`), and a string scan: any field that is not a plain registry name or date is refused (SQL, paths, shell, control characters, 5 kB blobs). Structure errors have stable codes (`BAD_AGG`, `UNKNOWN_SIGNAL`, …).
* **Capability** (`capability.assess`) — runs against the data profile: `OK` / `MISSING_DATA` / `INSUFFICIENT` / `UNKNOWN`, with the exact reason and the signals or formula that would make it computable. A valid query about missing data is an *explained answer*, not an error.
* **Planner** (`query_planner`) — the only place SQL text is made: whitelisted tables and columns, bound parameters, `SELECT`-only (`assert_read_only` rejects `;`, comments, DML/DDL, `PRAGMA`, `ATTACH`, `load_extension`, …). Chooses the tier (1 Hz for windows ≤ 6 h inside the 7-day window, per-minute otherwise).
* **Executor** (`query_executor`) — read-only connection, `PRAGMA query_only=ON`, row caps. Derived metrics come from `derived.py` (pointwise: rpm, torque, horsepower, turn radius, wheel slip, braking intensity…; accumulated: energy, distance, Wh/km, regen — integrals over the selected samples, matched against the trip ledger so the answers agree with the legacy engine). Event predicates such as *hard braking* use the events table, because sub-minute dynamics cannot be recovered from minute data.
* **Evidence** (`evidence.py`) — every number with metric, unit, role, period, tier, sources, quality flags, assumptions, method; plus an **evidence graph** for causal questions whose edges are limited to `preceded`, `followed`, `associated_with`, `likely_contributor`.
* **Composer** (`answer_composer.py`) — wording from templates over the evidence. Estimates say so; measured values never do; time phrases are consistent; counts and zeros are formatted honestly.
* **Verifier** (`answer_verifier.py`) — the last gate: every number in the text is in the evidence or the query; units belong to the evidence; named signals belong to the query/evidence; estimates are labelled with assumptions; unsupported answers state no value; no bare causal claim ("caused by", "due to", "proves"); no vehicle-control instruction; timestamps inside the history. A failure falls back to a conservative evidence-only answer.

## Parsing

**Rules first** (`nl2ir.parse`, ~0.3–1 ms): longest-phrase matching for time, signals, aggregations, conditions (named and thresholds with units, mph → km/h etc.), entity selections, events and comparison groups, then an operation decision from cue words. It reports a *coverage confidence* — the share of content words it explained. Below the threshold the parse is not trusted. It never guesses an ambiguity: bare "temperature" asks "Which temperature do you mean — battery, motor, cabin or outside?" and the next reply ("battery") completes the pending query. Causal ("why is my range…"), advice and hypothetical ("does cold weather affect…") questions are left to the legacy analysis engines.

**Follow-ups** (`ConvState`): "what about yesterday?", "and for city?", "how about the motor temperature" inherit what was not restated (target, time, conditions, selection); a mode condition replaces a mode condition; a fragment containing words the parser did not understand is *not* treated as a follow-up. An answer from the legacy routes resets the state, so a stale follow-up cannot attach to the wrong topic.

**Model fallback** (`ir_slm.py`): Qwen2.5-0.5B (Q4_K_M) with a LoRA adapter, GBNF-constrained to the registry (`query_grammar.py`, generated — adding a signal extends the grammar), greedy decoding, ≤ 96 tokens. The output must then pass `Query.from_text`, the **grounding check** (every number, signal, event and group in the query is traceable to words in the question — the hallucination guard), the validator and the capability engine. The base model with a few-shot prompt is available for benchmarking but is **not** used in the product without the adapter.

## Routing policy (`VGPT_QC`)

| mode | behaviour |
|---|---|
| `auto` (default) | the new engine answers what the legacy intents cannot express — sample-level conditions, entity selections, period/group comparisons other than city-vs-highway consumption, correlations, share-of-time, other statistics (median/std/p95), signals the legacy engine never knew (torque, steering, wheel speeds, voltages…), capability questions — and signals this vehicle does not record (an explained answer instead of a guess). Plain questions the legacy routes already answer well (and their tests cover) stay there. |
| `first` | the new engine answers everything it can parse. |
| `shadow` | the legacy engine answers; the new one runs silently (`Brain.last_qc`) for comparison. |
| `off` | legacy only. |

`VGPT_NO_QC=1` is an alias for `off`; `VGPT_IR_SLM=1` opts in to the model fallback (off by default: the rules alone are the shipped behaviour); `VGPT_QC_TRACE=<file>` logs every question the new path answered.

## Security boundary

The model's output grammar contains no rule for free text, SQL, paths or commands. The validator refuses any non-name string. The SQL compiler uses whitelists and bound parameters and refuses non-`SELECT` text. The executor connection is read-only. Vehicle-control phrases ("turn off the pump", "unlock the doors") never parse into a query, and the verifier rejects any answer that instructs a control action outside an explicit refusal. Text inside the database (DTC descriptions, log lines) is *data*: it is quoted verbatim in answers and is never parsed as a question or a query. See `tests/test_qc.py::SecurityTests`.

## Observability

Every `QCResult.timings` carries per-stage milliseconds (`validate`, `execute`, `compose`, `verify`, `total`); `Brain.ask` reports them in `facts["timings_ms"]` with the query text, the plan (`facts["plan"]`), whether the answer was verified, and which parser produced the query (`rules` or `slm` with its latency and token count).
