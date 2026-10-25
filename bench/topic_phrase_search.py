import os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ["VGPT_FRONTEND"] = "legacy"
from vgpt.product import make_legacy_brain
b = make_legacy_brain(os.path.join(HERE, "data", "gen_demo.db"), slm=None, now=1787625900)
C = {
 "current_soc": ["what is my battery percentage right now", "what is the charge level of the battery right now", "how full is the battery right now", "how much charge does the battery have now"],
 "current_inventory": ["show me all the current readings", "give me every current value from the car", "list all the current signal values", "what values can you see right now"],
 "motion_state": ["is the car moving right now", "is the vehicle moving at the moment", "is the car currently moving", "is it driving right now", "is the car stationary or moving"],
 "battery_equiv": ["how many 5000 mAh cells at 3.7 V equal the remaining usable energy", "how many 7000 mAh batteries at 3.7 V equal the remaining usable energy of the car", "how many phone batteries equal the remaining energy"],
 "daily_series": ["show me the trend in battery temperature day by day for the last seven days", "show the battery temperature day by day over the last 7 days", "give me the battery temperature for each day last week"],
 "diagnostic_priority": ["what should I investigate first", "what needs attention first", "what should I fix first", "what should the workshop check first", "which problem should be handled first"],
 "confirmed_active_faults": ["what are the three most serious confirmed active faults", "which faults are confirmed and active", "list the confirmed active faults"],
 "fault_recency_severity": ["which fault is most recent and which is most severe", "what is the most recent fault and what is the most severe fault", "which fault is the latest and which is the worst"],
 "warning_comparison": ["is the coolant flow warning more important than the tyre pressure warning", "which is more important the coolant warning or the tyre pressure warning", "compare the importance of the coolant warning and the tyre pressure warning"],
 "safety_assessment": ["can I safely drive 200 km with this warning", "is it safe to drive with this warning", "can I drive 100 km with this warning on"],
 "cooling_evidence": ["what evidence suggests that battery cooling has a problem", "what evidence is there of a battery cooling problem", "is there evidence that the battery cooling system has a problem"],
 "tyre_status": ["is my tyre pressure problem still present", "is the tyre pressure problem still there", "is my tyre pressure still low"],
 "briefing": ["brief me", "give me a briefing", "give me a quick summary of the car", "summary please"],
 "battery_soh": ["what is my battery health", "battery health please", "how healthy is the battery", "what is the state of health of my battery"],
 "change_over_window": ["how much has the battery health changed over 30 days", "how much has it changed over 30 days", "how much has my speed changed over 30 days"],
 "charging_now": ["how fast is it charging right now", "what is the charging power right now", "how fast is the car charging now"],
 "fault_status": ["is the last fault still active", "is the fault still active", "is that warning still active"],
 "normality_check": ["is the motor temperature normal", "is that value normal", "is the battery temperature normal"],
 "entity_swap": ["what about the battery temperature", "and the battery temperature", "what about the battery"],
 "side_by_side": ["compare the motor temperature with the battery temperature", "compare battery temperature and motor temperature side by side"],
 "group_compare_edit": ["compare city and highway energy consumption", "compare city and highway driving on energy consumption"],
 "comparison_window": ["do both comparisons use the same dates", "are both comparisons over the same period"],
 "change_evidence": ["what evidence supports that change", "what is the evidence for that change", "what supports the change you reported"],
}
ACC = {" ".join(x["q"].lower().split()).rstrip("?!. ") for x in json.load(open(os.path.join(HERE, "results", "round52", "acc_legacy_ref.json"), encoding="utf8"))}
res = {}
for tid, cands in C.items():
    cands = sorted(cands, key=lambda c: " ".join(c.lower().split()).rstrip("?!. ") in ACC)           # phrases that are not acceptance questions first
    hit = None; seen = []
    for ph in cands:
        b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
        r = b.ask(ph, use_slm=False); f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
        route = (f.get("route") or "").replace("qc:", ""); seen.append(route)
        if route == tid: hit = ph; break
    res[tid] = hit; print(("ok  " if hit else "BAD ") + f"{tid:24s} {hit or seen}")
json.dump(res, open(os.path.join(HERE, "results", "round52", "topic_phrases.json"), "w"), indent=1)
