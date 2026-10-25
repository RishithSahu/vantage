# Usability and end-to-end check of the interpretation-review flow (`/review`)

Purpose: find out whether people other than the author can use the editable interpretation preview to get the answer they intended, how long they wait for the model, how long they spend editing, and where it breaks. NOT a measure of autonomous accuracy.

## Who and how many
People who did not write the questions or the tasks; 6 to 10 is enough to expose usability defects, not to estimate a rate precisely. Record role (driver / engineer / student) only. No personal data beyond a participant code (p01, p02 ...). Everything is stored locally in `interaction_log.jsonl`.

## Setup (once)
1. `python ui/server.py --model none --port 8765 --db data/gen_courier_ext.db --planner granite --threads 8` - the server loads the language model first (about 4 minutes the first time; later starts reuse the cached prefix). `--planner lfm` selects the other measured configuration. Laptop timings are laptop timings, not ARM.
2. Open `http://127.0.0.1:8765/review`.
3. Before the first participant do one dry run yourself (`python bench/dry_run_review.py --port 8765`) to catch broken controls and to read the wait times; write them in the report.

## Session (about 25-35 minutes per participant)
1. Read the task list in `bench/user_study/tasks.tsv` (GOALS, never questions to copy). Give each participant the goals in a shuffled order but keep follow-up pairs together. Include: the five case-study scenarios (cs1-cs5), t01-t16 (everyday wording, restriction, threshold, selection, comparison, explanation, event context, missing sensor, capability, follow-up, refusal), t17 (changed vehicle mapping) and the three own goals (own1-own3: the participant chooses).
2. The participant types their OWN question for each goal (they see the goal, not the question), presses Interpret, reads the proposed interpretation, edits anything wrong, runs it, answers "Is this what you asked for?" and gives a 1-5 satisfaction rating. "I give up" is always available; use it rather than leaving a task silently.
3. Set the participant code and the task code in the two boxes at the top of the page before each task. Tick "this continues my previous question" ONLY for the second half of a follow-up task (t14b, t15b); every other task starts a new topic.
4. Do not help. Note anything the participant says aloud and any control they do not understand. Do not explain the model.
5. After the session ask three open questions: what was confusing, what would they change, would they use it instead of a menu.

## Analysis
`python bench/summarize_interactions.py <home>/interaction_log.jsonl` prints, separately: first-proposal-right (no edit and the participant says yes), goals met after CORRECTING, goals met after REBUILDING the query (changed the operation or three or more fields - not counted as the model's help), gave up, satisfaction, waiting time versus editing time, and agreement between the gold and the participant's own verdict. First-attempt accuracy, assisted success and satisfaction are never merged.

## Decision rule (written before the study)
* Usable: most participants meet their goals after at most one or two corrections, the wait is not the dominant part of the time, and satisfaction is not low. Then present the review flow as the demonstrated product with the assistance disclosed.
* Not usable: many give up, goals are only met by rebuilding, or the wait dominates. Then stop polishing the UI and report the unmet targets plainly.
