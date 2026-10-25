# How much room a car's history takes, and how it is kept inside a budget

**Rule (owner, 2026-10-23): storage work must not add latency to a question.** Nothing in this page sits between a question and its data except what was already there.

**Result (measured):** five years of dense history of a car with 31 to 41 recorded quantities takes **0.2 to 1.2 GB** (four-year builds: 389 MB, 627 MB and about 900 MB; five years is the same plus a quarter), against a budget of **6 GB** per car. The budget is enforced by the nightly housekeeping.

## Where the bytes are

`world_3yr.db` (a three-year history of one 41-quantity car, 524 MB), table by table (each table copied to a file of its own):

| Table | Size | Rows | Bytes per row |
|---|---|---|---|
| `lk_1m` (one-minute lake tier: mean, min, max, count per signal per minute) | **319 MB (61 %)** | 10.2 M | 31.4 |
| `sig_1m` (one-minute core tier) | 49 MB | 1.6 M | 31.4 |
| `lk_1s` (one-second lake tier, 7 days) | 46 MB | 2.6 M | 17.3 |
| `event_ctx`, `event_ctx_v2`, `event_ctx_x` (the seconds around each event) | 48 MB | 8 k | ~5,000 |
| `logs`, `logs_fts_*` | 25 MB | 243 k | |
| `agg_1d` (the daily tier) | 12 MB | 240 k | 48.6 |

## What was tried and thrown away because it costs latency

The minute tier is 70 % of the file and an archive file holds it in 1.3 to 2.2 bytes per row instead of 31. Moving the minute rows older than a year into the archive (and unpacking them for a question that reaches back that far) made a four-year history 2 to 2.4 times smaller (389 MB to 167 MB, 627 MB to 320 MB), and every answer was identical. **It was removed**, because unpacking is time added to the question, measured on the 4-year kestrel history (laptop; a Pi 4 is about 4 to 5 times slower):

| Question reaches into the archived minutes for | Time to make them readable |
|---|---|
| 1 day | 0.9 s (a month of every signal was read for one day) |
| 30 days | 1.3 s |
| 100 days | 2.7 s |
| 400 days | 10.3 s |

Per-day files and loading one signal would have made one day cost tens of milliseconds, not nothing; the rule is nothing. So the minute tier stays in the database for the whole history (`ret.coarse_days` = 1,830), a question about 2023 is answered by the same SQL as a question about last week, and the space is paid for with a budget that is large enough (below) instead of with a codec in the question's path. An integer-encoded minute table (scaled integers instead of REAL) was also measured: 319 MB became 206 MB; it would not slow a query but touches 26 places that read the table, so it is not done.

## The layout that remains

| Tier | Where | Kept for | Cost |
|---|---|---|---|
| Hot | database, one-second rows | `ret.fine_days` = 14 days | ~100 MB |
| Minutes | database, one-minute rows (core and lake) | `ret.coarse_days` = **1,830 days (5 years)** | the 31-byte rows: ~175 MB per car-year at 41 quantities |
| Daily | database (`agg_1d`) | forever | 12 MB for 3 years of 41 quantities |
| Records | events, trips, charges, logs, the context around events | 5 years | tens of MB |
| Expired seconds | archive files, lossless, read only by a question that is exact over the seconds themselves | `ret.archive_days` = 1,830 | 0.4 bytes per sample |

### Things that make it smaller without touching the question's path
* **Archive codec, version 2** (`vgpt/archive.py`): byte-plane integer deltas, and floats that are really fixed-point stored as integers (checked bit for bit). One day of the 41-signal lake: 317 KB became 189 KB (0.40 bytes per row), **and it decodes faster** (127 ms against 179 ms for the old format on the same rows), so the one path that does read the archive got quicker, not slower. Old files still decode.
* **A fault fixed on the way:** the per-signal views `lkv_*` of the main schema could not see the temporary rows that `archive.hydrate` loads, so archived *seconds* of lake signals were never readable through them. The views are now made again in the TEMP schema when their table is shadowed (`tests/test_archive.py`).
* **A size budget** (`vgpt/storage_budget.py`): `ret.budget_gb` (default 6) is the most the database file plus the archive folder may take. `store.compact` runs `enforce` once a night, after the archive has expired its old files: over the budget, it removes the **oldest archive files first** (seconds before minutes) until the total is 10 % under it, and writes a `source_log` row. The database, the daily tier and the records are never touched. This runs at night, never in a question. `tests/test_storage_budget.py`.
* **The build tool** (`tools_build_worlds.py`) no longer keeps every one-year piece and its copies on disk at once: the pieces are joined, the daily tier is rolled, the file is compacted and the pieces are deleted, one car at a time.

## Results per car

`python tools_storage_report.py` (the world builder writes `data/world4/<car>/<car>_4yr.db`, plus `<car>/archive/` when seconds have expired):

| Car | Quantities | 4-year file |
|---|---|---|
| aster_ev (BEV, every interface) | 41 | 389 MB |
| kestrel_petrol | 31 | 627 MB |
| meridian_ev (CAN-FD) | 37 | ~900 MB |

All of them under 1 GB for four years and under 1.3 GB projected for five.

## What this does not claim

* The sizes are measured on this test bench's generated data, which is smoother than a real car's; a real bus with more noise is bigger. The budget enforcement is what makes the 6 GB bound true regardless.
* Dense here means every quantity the decoder file binds at one-minute resolution for the whole history, one-second resolution for the last 14 days, and lossless one-second history for everything that has expired in a *running* car. The generated worlds hold the minute tier for four years; they do not carry four years of one-second samples (the generator writes the minute tier directly; generating them would take hours per car).
