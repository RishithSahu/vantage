"""Multi-turn development conversations (author-written, unreviewed). Each turn: utterance and the expected outcome - a gold IR, CLARIFY (a targeted clarification is the right behaviour) or REFUSE.
The evaluation uses the system's OWN accepted state after each turn (bench/convo_eval.py); no gold history is supplied."""
CONVOS = {
 "state_kept": [("average speed while on the highway yesterday", "op=aggregate tgt=speed agg=mean t=cal:yesterday c=highway"), ("and for the day before", "op=aggregate tgt=speed agg=mean t=cal:day_before_yesterday c=highway"),
                ("actually city only", "op=aggregate tgt=speed agg=mean t=cal:day_before_yesterday c=city"), ("without the city restriction", "op=aggregate tgt=speed agg=mean t=cal:day_before_yesterday"),
                ("what about this week", "op=aggregate tgt=speed agg=mean t=cal:this_week")],
 "events_and_correction": [("how many hard brakes this week", "op=count t=cal:this_week ev=HARSH_BRAKE"), ("list those", "op=list_events t=cal:this_week ev=HARSH_BRAKE evs=all"), ("how many of those were there", "op=count t=cal:this_week ev=HARSH_BRAKE evs=all"),
                           ("what about harsh accelerations", "op=count t=cal:this_week ev=HARSH_ACCEL"), ("how charged is the battery right now", "op=current_value tgt=battery_soc t=now"), ("and the motor temperature", "op=current_value tgt=motor_temperature t=now")],
 "thresholds": [("share of time above 60 mph this week", "op=percent_time t=cal:this_week c=sig:speed:gt:60:mph"), ("make that faster than 80 mph", "op=percent_time t=cal:this_week c=sig:speed:gt:80:mph"), ("and last week", "op=percent_time t=last:7:day c=sig:speed:gt:80:mph")],
 "ambiguous_then": [("how many warnings yesterday", "op=count t=cal:yesterday ev=ANY_WARNING"), ("what was the temperature then", "CLARIFY")],
 "no_antecedent": [("how many of those were there", "CLARIFY")],
 "specific_sensor": [("what is the tyre pressure on the front left", "op=current_value tgt=tyre_pressure_fl t=now"), ("and the rear right", "op=current_value tgt=tyre_pressure_rr t=now"), ("average over the last 3 days", "op=aggregate tgt=tyre_pressure_rr agg=mean t=last:3:day")],
 "event_followups": [("when was the last fault code", "op=list_events t=all ev=DTC_SET"), ("how many this month", "op=count t=cal:this_month ev=DTC_SET"), ("and last month", "op=count t=last:30:day ev=DTC_SET")],
 "refusal_keeps_state": [("average battery temperature this week", "op=aggregate tgt=battery_temperature agg=mean t=cal:this_week"), ("disable the speed limiter on this car", "REFUSE"), ("and the maximum", "op=extreme tgt=battery_temperature agg=max t=cal:this_week")],
 "topic_switch": [("highest speed on the highway yesterday", "op=extreme tgt=speed agg=max t=cal:yesterday c=highway"), ("how charged is the battery", "op=current_value tgt=battery_soc t=now"), ("what is the average coolant flow last week", "op=aggregate tgt=coolant_flow agg=mean t=last:7:day")],
 "explicit_replacement": [("average speed in the city this week", "op=aggregate tgt=speed agg=mean t=cal:this_week c=city"), ("change that to raining instead", "op=aggregate tgt=speed agg=mean t=cal:this_week c=raining")],
}
