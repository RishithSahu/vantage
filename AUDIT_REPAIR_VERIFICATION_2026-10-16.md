# VANTAGE repair verification — 2026-10-16

This snapshot contains the repair work for the Round 10, Round 12, and Round
13 audit findings that could be exercised from the supplied source tree.

## Repairs additionally verified in this pass

- Strict compact-event selector parsing rejects malformed selectors without
  rejecting valid `all` selectors.
- A camera cannot establish a healthy baseline from a flat or textureless
  startup frame, and the accepted calibration baseline no longer gradually
  adapts away a worsening blur.
- Camera capability/privacy questions remain answerable when current camera
  telemetry is stale, without presenting that telemetry as current.
- Incomplete trip energy, tyre specifications, charge-session energy, GPS,
  vehicle metadata, DTC details, and driving-extension data are treated as
  incomplete evidence rather than zero-valued measurements or crash inputs.
- GNSS freshness permits an overnight parked vehicle (up to 12 hours) but
  refuses to treat days-old location as current.
- The first external live-database copy in the Windows UI uses a safe create
  path; later swaps retain staged replacement and source/destination checks.
- The shipped CLI and UI now construct the Qwen2.5-0.5B typed-plan product
  route rather than the disconnected rule-parser baseline.  The production
  route disables BGE embedding retrieval and the optional semantic-support
  model: capability retrieval is deterministic SQLite FTS, so Qwen is the
  only learned model it loads.
- If that model cannot be constructed, the UI worker signals readiness and
  returns an explicit model-load error instead of silently serving a
  rule-based fallback or leaving the server apparently healthy with a dead
  assistant thread.
- Follow-up R13-A161–A165: the answer verifier no longer derives a numeric
  `1` from a nonzero period start, and a non-percent measurement cannot
  support a percentage claim.  An unverified fallback remains explicitly
  `unverified` with reduced confidence.  Open clarification state persists
  through failed plans/execution and is cleared only after a completed
  answer; planner construction failures are returned as explicit failed-plan
  outcomes.

## Verified test evidence

The following completed successfully in this environment:

- Round regression / GNSS group: 38 passed, 1 skipped.
- Focused UI-independent regression group: 85 passed, 3 skipped.
- Semantic, protocol, retention, evaluator, paging, and VANTAGE contracts:
  101 passed, 5 skipped.
- Composition, interpretation, validation, time-window, and formal fixtures:
  133 passed.
- Dialogue and semantic integrity: 50 passed.
- Product and outcome-contract checks: 35 passed.
- Round 13 targeted regressions: 11 passed.
- Final combined Round 10/12/13, GNSS, and dialogue recheck: 55 passed,
  1 skipped.
- Updated Round 10/12/13 regression and QC group: 75 passed, 1 skipped.
- Updated model-route, format, and model-budget group: 124 passed, 4 skipped.
- Product-route checks after the deployment-route repair: 21 passed.
- Python source compilation after the deployment-route repair: passed.
- Broad Round 10/12/13 semantic, QC, and fixture recheck: 205 passed, 1
  skipped.
- R13-A161–A165 targeted follow-up checks: 19 passed.
- Model-route and evaluator recheck after the follow-up repairs: 100 passed,
  4 optional skips.
- Round 10/12/13 regression and QC recheck after the follow-up repairs: 102
  passed, 1 optional skip.
- Python source compilation: passed.

## Environment limitations, not suppressed

- Three UI HTTP integration tests cannot open loopback sockets because this
  sandbox returns Windows error 10013. Their app setup reaches the server test
  stage; the blocked operation is the test client's local socket connection.
  The same three tests were rerun after the 0.5B deployment-route change and
  failed only at that blocked connection point.
- The LoRA GGUF converter integration check cannot create a temporary
  `.safetensors` file because the sandbox denies the third-party library's
  file operation before the converter is called.
- Optional protocol tests skip when their external parser packages or real
  capture fixtures are unavailable.
- Full-suite collection was started twice in this sandbox, but the harness
  returned no terminal pytest summary or exit status.  It is therefore not
  represented as a pass; the completed focused groups above are the evidence
  for this package.

These limits need rerunning on the target Windows/Raspberry Pi environment
before claiming a complete release-gate pass.
