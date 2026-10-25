> **2026-10-24: OFF by default (owner's decision: the model must not ask back).** `CLARIFY_BELOW = 0` in `vgpt/product.py`; `VGPT_MEANING_TAU=0.8` restores it. Why the readings are wrong in the first place: `docs/FAILURE_ANALYSIS_2026-10-24.md`.

# Asking back instead of guessing: "did you mean 1 or 2?"

**Why.** The frame model reads about 9 questions in 10 correctly on the frozen evaluation sets (3,038 questions, 283 read wrongly). The wrong readings are not random: a long tail of near-synonyms the ontology separates on purpose (battery health / battery condition, a service reminder / a maintenance plan, "how worn" as a live value / as a condition) and of questions that really have two readings. A rule per pair would fix a handful and break others. What *is* reliable is the model's own confidence: it is well calibrated, so the questions it is unsure about are the ones it gets wrong.

**What it does.** The decoder keeps the least certain step of each frame and the runner-up token there (`OrtFrameModel._alternative`). When the lowest token probability is below 0.8 (`vgpt/product.py CLARIFY_BELOW`, `VGPT_MEANING_TAU`, 0 turns it off) and the runner-up names exactly one other word of the same slot, the assistant asks:

> I want to be sure I understood. Did you mean:
> 1. Battery health
> 2. Battery condition
> Say 1 or 2, or ask it another way.

"1" / "2" (also "first", "second", "yes", "no") runs that frame for the original question; anything else is a new question. Nothing in the choice is a rule about meaning: both readings are the model's own, the words are the ontology's labels, and the threshold is a probability.

**Measured** (`bench/clarify_eval.py`, `results/clarify_eval.json`, all frozen sets, follow-up sets with their real previous frame):

| threshold | questions asked back | wrong readings caught | correct readings asked needlessly | accuracy of the questions answered at once |
|---|---|---|---|---|
| none (before) | 0 % | 0 % | 0 | 90.7 % |
| 0.7 | 4.6 % | 29 % | 58 | 93.1 % |
| **0.8 (shipped)** | **7.0 %** | **42 %** | **95** (3.1 % of all questions) | **94.2 %** |
| 0.9 | 9.6 % | 53 % | 142 | 95.2 % |

Two readings were offered for 74 % of the questions asked back (for the rest the runner-up does not name a single word, and the older one-reading message is used); in 68 % of those the right frame is one of the two offered. So of the wrong readings, about 42 % are no longer answered wrongly, and roughly a third of those turn into a correct answer after one tap.

**What it does not do.** 58 % of the wrong readings are *confident* (p >= 0.8): they are answered, wrongly, as before. Lowering them needs the model itself to learn the distinction, i.e. more and better-targeted training data for the confusable pairs and a new adapter. That is a training run, which the owner stopped on 2026-10-21; `bench/clarify_eval.py` and `results/clarify_eval.json` list the exact pairs and questions to train on.
