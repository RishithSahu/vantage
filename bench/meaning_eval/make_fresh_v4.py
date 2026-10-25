"""Write the frozen set `followup_fresh_v4.tsv`: complete questions asked right AFTER another answer, which must inherit nothing from it.

    python bench/meaning_eval/make_fresh_v4.py

Each row is a self-contained question of digest_v1 or catalog_v1 (their gold frame is unchanged) preceded by a randomly chosen frame of a previously answered question (seeded, so the file is reproducible).  The
frame model sees `Previous: <frame>` and must still answer exactly the gold frame, with `ref none`.  The first frame-2 runs inherited the topic of the previous turn in a few of these cases ("anything unusual
lately" after a battery question became `anomaly battery_condition`).  Written 2026-10-21 AFTER seeing those failures, BEFORE the training data that addresses them (`v4_fresh` in training/meaning/build_v2.py) was
used; its questions are not in any training file.  Never trained on.  sha256 beside it."""
import hashlib
import io
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PREV = ["now speed none none none none none none none", "now battery_soc none none none none none none none", "stat battery_temperature mean none none none none none none", "events hard_braking count none none none none none none",
        "summary none none none none none none none none", "anomaly none none none none none none none none", "trend motor_temperature change none none none none none none", "extreme speed max none none none none none none",
        "stat wh_per_km mean none none none none none none", "events any_warning count none none none none none none", "why wh_per_km rise none none trip none none none", "compare speed mean highway city none none none none",
        "health battery_condition none none none none none none", "stat tyre_pressure mean none none none none none none", "summary none change none none none none none none", "now range_estimate none none none none none none none",
        "stat battery_temperature usual none none none none none none", "events charging_session list none none none none none none", "estimate range_estimate none none none none none none none", "social none none none none none none none none"]


def main():
    r = random.Random(41); rows = []
    for name in ("digest_v1", "catalog_v1"):
        for line in io.open(os.path.join(HERE, name + ".tsv"), encoding="utf-8"):
            if line.startswith("#") or not line.strip(): continue
            p = line.rstrip("\n").split("\t"); rows.append((p[1], p[2]))
    r.shuffle(rows); out = ["# FOLLOWUP_FRESH_V4: complete questions after another answer; they inherit nothing.  Written 2026-10-21 after the first frame-2 runs were read, before the data that addresses the failure was used.  Never trained on.",
                            "# Format: id <TAB> previous frame (8 words) <TAB> text <TAB> gold frame (8 words) <TAB> tag"]
    for i, (text, gold) in enumerate(rows, 1): out.append(f"X{i:03d}\t{r.choice(PREV)}\t{text}\t{gold}\tfresh_v4")
    path = os.path.join(HERE, "followup_fresh_v4.tsv"); io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest(); io.open(os.path.join(HERE, "followup_fresh_v4.sha256"), "w").write(h + "  followup_fresh_v4.tsv\n"); print(len(rows), "rows", h[:16])


if __name__ == "__main__":
    main()
