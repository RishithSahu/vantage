# Brief for the independent reviewer (someone who has not seen any model output)

Two things to review WITHOUT seeing any model prediction, and to freeze before anything is evaluated:

1. **The study goals** (`bench/user_study/tasks.tsv`): is each goal clearly worded, answerable on the study vehicle (`data/gen_courier_ext.db`; task t17 uses the cold-climate vehicle), and is the gold typed interpretation (the IR text in the last column) the intended meaning? Mark any goal that genuinely permits more than one meaning (then list the acceptable alternatives) or whose gold is wrong. Do not change a gold to match what the system does.
2. **The sealed acceptance interpretations** (`bench/acceptance_v1.tsv`): this file has not been read or used by the developer or the assistant and must stay that way. The reviewer should read each question with its gold interpretation and note disputed ones (ambiguity, wrong operation, a defaulted period that the question does not state, a restriction missing from the gold). Record the decisions in a new file `bench/acceptance_review.tsv` (id, verdict, note) and tell the developer only how many were changed, not which. After that the set is frozen; nothing is tuned against it.

Why: the development numbers were obtained on questions the developer saw, with a bank designed after reading development failures; only an independent, frozen acceptance review can give an honest figure.
