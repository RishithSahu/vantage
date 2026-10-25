"""Experiment ledger: one compact row per measured configuration, appended to results/LEDGER.md, so that an unsuccessful later experiment never hides the best configuration and every number can be traced to a source revision, a model file, a prompt/schema identity and an exact change.
    python bench/ledger.py add results/x.json --change "what changed" --decision keep|revert|baseline [--note ...]
    python bench/ledger.py rev                    # the current source revision
A row records: source revision (hash of vgpt/*.py), model identity, prompt/schema identity (written by bench/eval_model.py), the exact change, complete accuracy counts (correct / answerable / wrong among answered / clarifications / refusals), stage-level errors, runtime and memory when present, and the keep/revert decision."""
import argparse, glob, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(HERE, "results", "LEDGER.md")
HEAD = "| when | source rev | file | model | prompt/schema id | change | correct (all) | correct (answerable) | wrong / answered | clarif. | refusals ok / set | earliest failing stage | latency ms (mean) | decision |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"


def source_revision():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(HERE, "vgpt", "*.py"))):
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


def add(path, change, decision, note=""):
    d = json.load(open(path, encoding="utf-8")); s = d["summary"]; ca, cv, an, wa = s["correct_all"], s["correct_answerable"], s["answered_coverage"], s["wrong_among_answered"]
    ref = s["outcomes"].get("refusal", 0); fr = s["false_action_on_refusal_set"]; stages = s.get("earliest_failing_stage", {}); lat = (s.get("timing_ms") or {}).get("total_ms", {}).get("mean")
    row = (f"| {time.strftime('%Y-%m-%d %H:%M')} | {s.get('source_rev') or 'n/r (before ledger)'} | {os.path.basename(path)} | {s.get('system', '')} ({s.get('dialect', '')}) | {s.get('prompt_id', '?')} | {change} | {ca['k']}/{ca['n']} | {cv['k']}/{cv['n']} | {wa['k']}/{wa['n']} | "
           f"{s['clarification']['asked']} (answerable {s['clarification']['on_answerable']}) | {fr['n'] - fr['k']}/{fr['n']} | {', '.join(f'{k} {v}' for k, v in stages.items() if v)} | {lat} | {decision}{(' - ' + note) if note else ''} |\n")
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", encoding="utf-8") as f:
        if new: f.write("# Experiment ledger (model-only product, development set of 120 questions)\n\nDevelopment evidence only; the acceptance set is sealed and has not been read. Target: at least 109/120 correct (all) and 98/108 (answerable).\n\n" + HEAD)
        f.write(row)
    print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["add", "rev"]); ap.add_argument("path", nargs="?"); ap.add_argument("--change", default=""); ap.add_argument("--decision", default="")
    ap.add_argument("--note", default=""); a = ap.parse_args()
    if a.cmd == "rev": print(source_revision())
    else: add(a.path, a.change, a.decision, a.note)
