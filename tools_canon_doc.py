"""Write docs/CANON.md from the rule tables of vgpt/meaning/canon.py and askme.py, so the list of rules can never differ from the rules.

    python tools_canon_doc.py"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from vgpt.meaning import canon, askme   # noqa: E402


def main():
    L = ["# The pre-parser: every rule, in one place", "",
         "Written by `tools_canon_doc.py` from the tables in `vgpt/meaning/canon.py` and `vgpt/meaning/askme.py`; `tests/test_canon.py` fails if this file is out of date.", "",
         "The owner asked (2026-10-24) for a rule layer in front of the language model, so that a 0.5 B model is not expected to know everything. It is **visible** (the answer shows `you wrote ..., I read it as ...`), **small** (this page lists all of it) and **counted** (`tests/test_meaning.py::RuleBudget` fixes how many pattern calls each module may contain). What a question *means* in general stays with the model.", "",
         "## 1. Shorthand and spelling (`canon.normalise`)", "", "A shorthand is replaced by its plain word. A word of five or more letters that is neither English nor in the vehicle's vocabulary is corrected only to the one nearest vocabulary word (one edit for 5-7 letters, two for 8 or more).", "",
         "| written | read as |", "|---|---|"]
    for k, v in sorted(canon.SHORT.items()):
        if v != k: L.append(f"| `{k}` | {v or '(dropped)'} |")
    L += ["", "### `acc`: the only question the assistant asks back", "", "`acc` means *acceleration* or *adaptive cruise control*. It is read from the other words: " + ", ".join(sorted(canon.CRUISE_WORDS)) + " choose cruise control; " + ", ".join(sorted(canon.ACCEL_CUES)) + " choose acceleration. If nothing in the question chooses, the assistant asks which is meant, once, and takes the answer (`1`, `2`, or the words). No other question is ever asked back.", "",
          "## 2. Awkward wordings read as plain questions (`canon.REWRITES`)", "", "| pattern (on the lower-case question) | plain question |", "|---|---|"]
    for p, r in canon.REWRITES: L.append(f"| `{p}` | {r} |")
    L += ["", "## 3. Questions answered without the model (`askme.HANDLERS`, first match wins)", "", "| handler | what it answers | where the answer comes from |", "|---|---|---|"]
    src = {"charge_state": ("is the car charging", "latest battery row and the charge table"), "data_catalog": ("which data this vehicle records", "the signal catalogue of the store"), "how_it_works": ("what the assistant runs on", "the loaded adapter's name"),
           "identity": ("who or what the assistant is", "the vehicle's model name"), "capabilities": ("what can be asked", "fixed text"), "clock": ("the time and date", "the vehicle's clock"),
           "trip_energy": ("how much battery or fuel a distance uses", "the recent consumption of the vehicle"), "trip_energy2": ("same, other word order", "same"), "trip_can": ("can I drive N km", "same"),
           "service": ("last service, distance since, interval, next due", "meta veh.last_service_*, the odometer"), "law": ("traffic law and licences: out of scope", "fixed text"), "buy": ("where to get the car: out of scope", "fixed text"),
           "price": ("prices: out of scope", "fixed text"), "weather": ("weather: out of scope", "fixed text"), "route": ("directions: out of scope", "fixed text")}
    for name, rx, fn in askme.HANDLERS:
        a, b = src.get(name, (name, "")); L.append(f"| `{name}` | {a} | {b} |")
    L += ["", "Every number such an answer contains is computed from the vehicle's own tables; what cannot be known says so. Pattern text of each handler: see `askme.HANDLERS`.", ""]
    L += ["## 4. What it measurably does (bench/typo_stress.py, word list built WITHOUT the evaluation sets it is tested on)", "",
          "300 questions of the held-out sets, each with one typo in one word: the model alone reads 90.3 % of them as it reads the clean question; with the pre-parser 94.3 %; the pre-parser broke 2 (a plural turned singular, an unknown abbreviation). Ordinary clean questions are changed only by the shorthand table (tests/test_canon.py: at most 1.2 % get any spelling change, and those are typos the sets contain on purpose). The model is robust to most typos by itself; the pre-parser's job is the shorthand it cannot know (`avg`, `acc`, `srevicing`) and the few wordings it keeps missing.", ""]
    open(os.path.join(ROOT, "docs", "CANON.md"), "w", encoding="utf-8").write("\n".join(L)); print("wrote docs/CANON.md", len(L), "lines")


if __name__ == "__main__": main()
