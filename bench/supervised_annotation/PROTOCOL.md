# Natural Structural Annotation Protocol: 100 → 250

## Purpose

Collect 150 independently phrased natural questions that target the measured structural failures. These are training/development data, never acceptance or final-test data.

## Quotas

| Structural target | New questions |
|---|---:|
| Time and event-relative structure | 45 |
| Operator minimal contrasts | 35 |
| Comparison and modifier attachment | 25 |
| Threshold and exclusion structure | 15 |
| Nested selection/ranking | 10 |
| Dialogue edits and inheritance | 10 |
| Messy wording, capability and refusal boundaries | 10 |
| **Total** | **150** |

Within each quota, use different underlying programs—not several paraphrases of one program. Keep every paraphrase or dialogue continuation from the same semantic seed in one split.

## Collection

1. Show an author a plain-language goal and relevant fictional dialogue state, never a canonical question or parser prediction.
2. The author writes the question they would naturally ask.
3. A separate annotator assigns executable IR, semantic spans and typed relations.
4. A reviewer executes or validates the IR and resolves disagreements without showing the author model output.
5. Record `author_id`, `program_family` and `semantic_seed` so evaluation can hold out authors and families.

## Required annotations

- `question` and optional `previous_question`;
- canonical executable `gold_ir`;
- character spans for target, time, event, restriction, number/unit and selection when present;
- typed relations such as `restriction->comparison`, `restriction->left_scope`, `time->selection`, `before->event`;
- ambiguity flag and reviewer note.

Do not add phrase-specific rules after reviewing an error. Corrections enter the annotated corpus and are evaluated through family-separated splits.

## 250-example stopping gate

- Compare against the frozen 100-example checkpoint on the same development set.
- Require a clear increase in strict skeleton and complete-program accuracy, not only individual fields.
- Report author-held-out and program-family-held-out results.
- Stop this architecture if operator/time fields improve but strict exact accuracy remains near zero.
