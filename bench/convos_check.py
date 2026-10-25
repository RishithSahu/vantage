"""Second set of scripted conversations (author-written, unreviewed), written AFTER the first set had been used to fix the follow-up logic and run ONCE before any change: its first-run score is the generalisation measurement.
Any later fix made because of these conversations makes them regression data too (recorded in the status report)."""
CONVOS = {
 "tyre_chain": [("what was my lowest tyre pressure yesterday", "op=extreme tgt=tyre_pressure agg=min t=cal:yesterday"), ("and the highest", "op=extreme tgt=tyre_pressure agg=max t=cal:yesterday"),
                ("now for last night", "op=extreme tgt=tyre_pressure agg=max t=cal:last_night"), ("only on the motorway", "op=extreme tgt=tyre_pressure agg=max t=cal:last_night c=highway"),
                ("forget the motorway part", "op=extreme tgt=tyre_pressure agg=max t=cal:last_night")],
 "camera_events": [("how many times did the camera get blocked this month", "op=count t=cal:this_month ev=CAMERA_BLOCKED"), ("and this week", "op=count t=cal:this_week ev=CAMERA_BLOCKED"),
                   ("list them", "op=list_events t=cal:this_week ev=CAMERA_BLOCKED evs=all"), ("what was the cabin temperature then", "CLARIFY")],
 "threshold_chain": [("average hvac power when it is colder than 5 degrees outside last week", "op=aggregate tgt=hvac_power agg=mean t=last:7:day c=sig:ambient_temperature:lt:5"), ("make that colder than 0 degrees", "op=aggregate tgt=hvac_power agg=mean t=last:7:day c=sig:ambient_temperature:lt:0"),
                     ("and this week", "op=aggregate tgt=hvac_power agg=mean t=cal:this_week c=sig:ambient_temperature:lt:0")],
 "energy_chain": [("how much energy did i use yesterday", "op=aggregate tgt=energy_used agg=sum t=cal:yesterday"), ("and the distance", "op=aggregate tgt=distance_driven agg=sum t=cal:yesterday"), ("what about consumption", "op=aggregate tgt=wh_per_km agg=mean t=cal:yesterday"),
                  ("over the last 5 days", "op=aggregate tgt=wh_per_km agg=mean t=last:5:day")],
 "wheel_chain": [("what is the front right tyre pressure", "op=current_value tgt=tyre_pressure_fr t=now"), ("rear left", "op=current_value tgt=tyre_pressure_rl t=now"), ("average over the last week", "op=aggregate tgt=tyre_pressure_rl agg=mean t=last:7:day")],
 "offtopic_then_back": [("tell me a joke", "REFUSE"), ("average speed today", "op=aggregate tgt=speed agg=mean t=cal:today"), ("and highest", "op=extreme tgt=speed agg=max t=cal:today")],
 "misunderstanding": [("how many harsh brakes yesterday", "op=count t=cal:yesterday ev=HARSH_BRAKE"), ("no i meant harsh accelerations", "op=count t=cal:yesterday ev=HARSH_ACCEL"), ("and the day before", "op=count t=cal:day_before_yesterday ev=HARSH_ACCEL")],
 "mode_chain": [("average battery temperature while charging this week", "op=aggregate tgt=battery_temperature agg=mean t=cal:this_week c=charging"), ("city only", "op=aggregate tgt=battery_temperature agg=mean t=cal:this_week c=city"), ("actually all trips", "op=aggregate tgt=battery_temperature agg=mean t=cal:this_week")],
}
