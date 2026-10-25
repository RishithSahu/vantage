# VOLT-Q/ALMA feasibility prototype

This prototype asks a narrow question: can a compact pretrained encoder improve
language-to-semantics alignment after training only on algebra-generated
declarative fragments and semantic contrasts?

It does not train on complete vehicle question-to-program pairs and it never
opens bench/indep_final.tsv.

## Run

From the vehiclegpt2 directory:

    python -m unittest tests.test_alma -v
    python -m alma.experiment --epochs 5

Outputs are written to results/alma_probe:

- metrics.json contains all measurements and errors.
- REPORT.md contains the readable comparison.
- checkpoint contains the fine-tuned encoder.

## What this first probe covers

- Dynamic task and scope cards.
- Question-free declarative microsemantic training.
- Natural development-question transfer.
- Scope/attachment minimal pairs.
- Exact exhaustive decoding for a bounded single-target grammar.
- Denotational counterexample generation for scope mutations.

It is not a full parser. Time, event, dialogue-state, numeric-span, and arbitrary
graph composition heads are later milestones and should be added only if this
probe shows useful transfer.

The completed first verdict is in `results/ALMA_FEASIBILITY_VERDICT.md`. Current
evidence supports another experiment using span-to-algebra edges; it does not
support a 95% accuracy claim.

That bounded ALMA-v2 experiment has now been completed in
`results/alma_v2_edge`. It failed the continuation gate: the best frozen
backbone reached 51.7% gold-span attachment accuracy and 17.7% strict
full-structure accuracy on the 300-case construction-separated challenge.

The later, separate natural-supervision branch found one component worth
retaining: `time_curriculum.py`. Its controlled replacement ablation is in
`results/time_integration_ablation`; predicted time improves mean structural
exactness from 8.3% to 13.3% and passes the predefined three-seed gates.
