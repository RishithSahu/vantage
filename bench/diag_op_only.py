"""Diagnostic: how well does a frozen language model pick the OPERATION when that is its only task (no arguments to write, no output format to follow beyond one operation name)? Compared with the joint plan this separates "does not understand the request" from "cannot express a long structured plan".
The prompt lists every operation with its descriptive label (from the plan-language definition) and the capability cards of the message; the answer is constrained to the operation names. Variants: --shots none | one (one generic example per operation, written from the specification, checked against the evaluation sets by tests)
    python tools_pin.py 8,9 bench/diag_op_only.py --model PATH [--shots none|one] [--slice bench/slice_v1.txt]"""
import argparse, collections, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_function import EQUIV, gold_function
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import toolspec as T
from vgpt.modelplan import Planner, State
from vgpt.slm_backend import PrefixLlama

OPS = [fn for fn in T.FUNC_NL]
ONE = {"current_value": "How much charge is left in the battery?", "statistic": "What was the average cabin temperature over the last 3 days?", "extreme": "What was the highest motor temperature yesterday?", "count_events": "How many hard cornering events were there last week?",
       "count_entities": "How many charging sessions did I have this month?", "list_events": "Show me every tyre warning from the last 5 days.", "did_event_happen": "Did the camera get blocked yesterday?", "did_condition_hold": "Did the pack temperature ever go above 45 degrees this week?",
       "time_share": "What share of last week was the car parked?", "compare": "Is the battery cooler on the highway than in the city?", "when_changed": "When did the cabin temperature change the most?", "event_context": "What was going on just before the last 12 volt warning?",
       "explain_change": "Why is my range lower than it was a month ago?", "explain_event": "Why did the traction battery warning come on?", "trend": "Is the brake pad life slowly falling?", "rank_trips": "Which were my three most efficient trips?",
       "correlate": "How does outside temperature relate to energy use?", "is_recorded": "Does the car record steering angle?", "use_tool": "How are my tyres doing?", "decline": "Please turn the heating up.", "say": "Thanks a lot!", "ask_user": "And what about it?", "unresolved": "What is the engine oil temperature?"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--shots", default="none", choices=["none", "one", "knn_final"], help="knn_final: UPPER-BOUND ESTIMATE ONLY - the k nearest questions of the old 308-question final set are shown as demonstrations (contaminated by construction; it measures what retrieved demonstrations could add, and is never a product configuration)"); ap.add_argument("--k", type=int, default=6); ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); ap.add_argument("--n-ctx", type=int, default=4096); a = ap.parse_args()
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; rows = [rows[i] for i in ids if not rows[i]["prev_text"]]
    P = Planner(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True), load_model=False)
    system = ("You classify what a person asks about one vehicle's recorded data. Choose the ONE operation that answers the message. Operations:\n" + "\n".join(f"- {fn}: {T.FUNC_NL[fn]}" for fn in OPS) + "\nAnswer with the operation name only.")
    shots = [(f"Message: {ONE[fn]}", fn) for fn in OPS if a.shots == "one"]
    lm = PrefixLlama(a.model, n_ctx=a.n_ctx, threads=2, mmap=True); lm.set_prefix(system, shots, name="p"); g = "root ::= " + " | ".join(f'"{fn}"' for fn in OPS)
    bank = None
    if a.shots == "knn_final":
        import numpy as np
        fin = [r for r in load(os.path.join(HERE, "bench", "indep_final.tsv")) if not r["prev_text"]]; bank = [(r["text"], "decline" if r["ir"].startswith("op=none") else gold_function(r["ir"])) for r in fin]; bank = [(q, f) for q, f in bank if f]; bv = P.emb.encode([q for q, _ in bank])
    ok = n = 0; conf = collections.Counter(); t0 = time.time()
    for r in rows:
        gf = "decline" if r["ir"].startswith("op=none") else gold_function(r["ir"]); n += 1
        cards = P.reg.card_text(P.build_message(r["text"], State(), DEFAULT_NOW)[2]); msg = f"Capabilities:\n{cards}\nMessage: {r['text']}"
        hist = None
        if bank is not None:
            sims = bv @ P.emb.encode([r["text"]], query=True)[0]; idx = [int(i) for i in np.argsort(-sims)[:a.k]][::-1]; hist = [("Message: " + bank[i][0], bank[i][1]) for i in idx]
        out, _ = lm.generate(msg, grammar=g, max_tokens=8, stop_text=None, prefix="p", history=hist); fn = out.strip(); good = fn == gf or fn in EQUIV.get(gf, ()) or (gf == "decline" and fn in ("decline", "unresolved", "ask_user", "say")); ok += good
        if not good: conf[(gf, fn)] += 1
    print(f"{os.path.basename(a.model)} shots={a.shots}: operation chosen correctly {ok}/{n} = {100 * ok / n:.0f}%  ({(time.time() - t0) / n:.1f} s each)")
    for (g_, f_), c in conf.most_common(8): print(f"   gold {g_:18s} -> {f_} x{c}")


if __name__ == "__main__": main()
