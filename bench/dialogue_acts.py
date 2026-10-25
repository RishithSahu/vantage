"""How reliable is the permitted frozen model (Qwen2.5-0.5B) at choosing the dialogue act of a follow-up among the acts the typed state supports? Cases: (typed topic as a query text, message, gold act). Used to choose the prompt of vgpt/dialogue_state.decide_act; the cases are paraphrases and neighbours of the campaign sequences, not the sequences themselves.
    python bench/dialogue_acts.py [--variant a|b]"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import dialogue_state as D
from vgpt.brain import LazySlm
from vgpt.query_ir import Query

CASES = [
 ("op=current_value tgt=motor_temperature t=now", "and the battery?", "swap_entity"),
 ("op=current_value tgt=motor_temperature t=now", "how about the pack", "swap_entity"),
 ("op=current_value tgt=ambient_temperature t=now", "what about the cabin one", "swap_entity"),
 ("op=current_value tgt=battery_temperature t=now", "is that normal", "assess_value"),
 ("op=current_value tgt=battery_temperature t=now", "is this a usual reading for it", "assess_value"),
 ("op=current_value tgt=battery_temperature t=now", "put it next to the motor temperature", "compare_with"),
 ("op=current_value tgt=battery_temperature t=now", "how does it compare with the outside temperature", "compare_with"),
 ("op=aggregate tgt=battery_soh agg=mean t=last:7:day", "how much has it moved over the past 60 days", "change_window"),
 ("op=aggregate tgt=battery_soh agg=delta t=last:30:day", "what data is that based on", "evidence"),
 ("op=aggregate tgt=battery_soh agg=delta t=last:30:day", "show me the readings behind it", "evidence"),
 ("op=list_events ev=DTC_SET evs=all t=all", "does that still apply", "fault_status"),
 ("op=list_events ev=DTC_SET evs=all t=all", "is the fault still there", "fault_status"),
 ("op=compare tgt=wh_per_km agg=mean t=last:14:day grp=city,highway", "now do it for distance", "switch_metric"),
 ("op=compare tgt=wh_per_km agg=mean t=last:14:day grp=city,highway", "are both over the same days", "same_window"),
 ("op=current_value tgt=motor_temperature t=now", "how much energy did I use last week", "new_question"),
 ("op=current_value tgt=motor_temperature t=now", "how can I drive more smoothly", "new_question"),
 ("op=list_events ev=DTC_SET evs=all t=all", "what is my tyre pressure", "new_question"),
 ("op=list_events ev=DTC_SET evs=all t=all", "what should I look at first", "new_question"),
 ("op=list_events ev=DTC_SET evs=all t=all", "which warning matters the most", "new_question"),
 ("op=current_value tgt=motor_temperature t=now", "what about the pack", "swap_entity"),
 ("op=current_value tgt=speed t=now", "and the outside temperature", "new_question"),
 ("op=event_context ev=ANY_WARNING evs=last bef=600 t=all", "is that still going on", "fault_status"),
]
FEWSHOT = [("Previous answer was about: the current speed", "And the battery?", "swap_entity"), ("Previous answer was about: the average coolant flow over the last 7 days", "What was the range of my trips yesterday", "new_question"), ("Previous answer was about: the number of harsh brake events", "Is that high?", "assess_value")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="a"); a = ap.parse_args()
    slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words"); ok = 0; t0 = time.time()
    class B: pass
    b = B(); b.slm = slm
    for topic, msg, gold in CASES:
        q = Query.from_text(topic); feas = [x for x in D.ACTS if x != "new_question"] + ["new_question"]
        if a.variant == "b":
            sysm = "You decide how a vehicle-data assistant should read a message in a conversation. Choose exactly one option."; ex = chr(10).join(f"Previous: {p}{chr(10)}Message: {m}{chr(10)}Answer: {g}" for p, m, g in FEWSHOT)
            user = ex + chr(10) + "Previous: " + D._summary(q) + chr(10) + "Message: " + msg + chr(10) + "Options: " + ", ".join(feas) + chr(10) + "Answer:"
            act, _ = slm.choose(sysm, user, feas, max_tokens=6)
        else: act = D.decide_act(b, msg, q, feas, "")
        ok += act == gold; print(("ok  " if act == gold else "MISS"), gold, "<-", act, "|", msg)
    print(f"{ok}/{len(CASES)} correct, {(time.time() - t0) / len(CASES):.1f} s per decision")


if __name__ == "__main__": main()
