"""Writes bench/diag_v1.tsv: a small DIAGNOSTIC set (development use) aimed at the failures seen in the dev comparison: state kept across follow-ups, contradictory updates, specific sensors, omitted conditions, unwanted defaults."""
import os
R = []
def add(fam, q, ir, pq="-", pir="-"): R.append((fam, q, ir, pq, pir))
A = "op=aggregate"; X = "op=extreme"
# gold periods follow the dev convention (last week = last:7:day, last month = last:30:day); three rows were corrected to it after the first run (author inconsistency, not tuning of the system)
# KEEP: a follow-up changes one field and must preserve the rest
add("KEEP", "same for last week", f"{A} tgt=tyre_pressure_fl agg=mean t=last:7:day c=highway", "average front-left tyre pressure on highway trips this week", f"{A} tgt=tyre_pressure_fl agg=mean t=cal:this_week c=highway")
add("KEEP", "and for the day before", f"{X} tgt=speed agg=max t=cal:day_before_yesterday c=highway", "highest speed on highway trips yesterday", f"{X} tgt=speed agg=max t=cal:yesterday c=highway")
add("KEEP", "what about this month", f"{A} tgt=battery_temperature agg=mean t=cal:this_month c=charging", "average battery temperature while charging this week", f"{A} tgt=battery_temperature agg=mean t=cal:this_week c=charging")
add("KEEP", "and the highest", f"{X} tgt=battery_soc agg=max t=last:3:day", "lowest battery charge in the last 3 days", f"{X} tgt=battery_soc agg=min t=last:3:day")
add("KEEP", "only last week", "op=count t=last:7:day ev=HARSH_BRAKE", "how many harsh brakes in the last 14 days", "op=count t=last:14:day ev=HARSH_BRAKE")
add("KEEP", "and this week", f"{A} tgt=speed agg=mean t=cal:this_week c=raining", "average speed in the rain last week", f"{A} tgt=speed agg=mean t=cal:last_week c=raining")
add("KEEP", "and yesterday", f"{X} tgt=motor_temperature agg=max t=cal:yesterday c=sig:speed:gt:60:km_per_h", "peak motor temperature when above 60 km/h this month", f"{X} tgt=motor_temperature agg=max t=cal:this_month c=sig:speed:gt:60:km_per_h")
add("KEEP", "and the front right", f"{A} tgt=tyre_pressure_fr agg=mean t=last:7:day", "mean rear right tyre pressure in the last 7 days", f"{A} tgt=tyre_pressure_rr agg=mean t=last:7:day")
add("KEEP", "now the same for last month", f"{A} tgt=wh_per_km agg=mean t=last:30:day c=city", "average energy per km on city trips this month", f"{A} tgt=wh_per_km agg=mean t=cal:this_month c=city")
add("KEEP", "and how many were there", "op=count t=cal:yesterday ev=HARSH_ACCEL", "list the hard accelerations from yesterday", "op=list_events t=cal:yesterday ev=HARSH_ACCEL evs=all")
# CONTRA: a later turn replaces or removes an earlier condition
add("CONTRA", "actually city only", f"{A} tgt=wh_per_km agg=mean t=cal:this_month c=city", "average energy per km on highway trips this month", f"{A} tgt=wh_per_km agg=mean t=cal:this_month c=highway")
add("CONTRA", "without the rain restriction", f"{A} tgt=speed agg=mean t=cal:this_week", "average speed while raining this week", f"{A} tgt=speed agg=mean t=cal:this_week c=raining")
add("CONTRA", "make that colder than 0", f"{A} tgt=hvac_power agg=mean t=cal:this_week c=sig:ambient_temperature:lt:0", "average hvac power when it is colder than 5 degrees outside this week", f"{A} tgt=hvac_power agg=mean t=cal:this_week c=sig:ambient_temperature:lt:5")
add("CONTRA", "actually all trips", f"{A} tgt=speed agg=mean t=cal:this_week", "average speed on highway trips this week", f"{A} tgt=speed agg=mean t=cal:this_week c=highway")
add("CONTRA", "change that to raining instead", f"{A} tgt=speed agg=mean t=cal:this_week c=raining", "average speed on highway trips this week", f"{A} tgt=speed agg=mean t=cal:this_week c=highway")
# SENS: a specific sensor must stay specific
add("SENS", "average front-left tyre pressure last week", f"{A} tgt=tyre_pressure_fl agg=mean t=last:7:day")
add("SENS", "what is the rear right wheel speed now", "op=current_value tgt=wheel_speed_rr t=now")
add("SENS", "front right tyre pressure this week", f"{A} tgt=tyre_pressure_fr agg=mean t=cal:this_week")
add("SENS", "lowest rear left tyre pressure yesterday", f"{X} tgt=tyre_pressure_rl agg=min t=cal:yesterday")
add("SENS", "do you record the minimum cell temperature", "op=capability tgt=cell_temperature_min t=all")
# OMIT: every stated condition must survive
add("OMIT", "average speed on highway trips last week", f"{A} tgt=speed agg=mean t=last:7:day c=highway")
add("OMIT", "how many times did it brake hard while raining this week", "op=count t=cal:this_week ev=HARSH_BRAKE c=raining")
add("OMIT", "peak battery temperature while charging yesterday", f"{X} tgt=battery_temperature agg=max t=cal:yesterday c=charging")
add("OMIT", "average hvac power when it was below 5 degrees outside last week", f"{A} tgt=hvac_power agg=mean t=last:7:day c=sig:ambient_temperature:lt:5")
add("OMIT", "average speed in the city this month", f"{A} tgt=speed agg=mean t=cal:this_month c=city")
# DEFAULT: a stated period must not be replaced by the default
add("DEFAULT", "average battery temperature over the last 3 days", f"{A} tgt=battery_temperature agg=mean t=last:3:day")
add("DEFAULT", "peak speed in the last 10 days", f"{X} tgt=speed agg=max t=last:10:day")
add("DEFAULT", "total energy used in the last 2 weeks", f"{A} tgt=energy_used agg=sum t=last:2:week")
add("DEFAULT", "how many harsh accelerations in the last 20 days", "op=count t=last:20:day ev=HARSH_ACCEL")
add("DEFAULT", "average speed on 15 September", f"{A} tgt=speed agg=mean t=date:2026-09-15")
hdr = ["# DIAGNOSTIC set v1 (author-written by Claude; development use only; unreviewed). Targets: state kept across follow-ups (KEEP), contradictory updates (CONTRA), specific sensors (SENS), omitted conditions (OMIT), stated period replaced by a default (DEFAULT).",
       "# Columns (TAB separated): id, family, question, gold IR, previous question (or -), previous gold IR (or -)"]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_v1.tsv")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(hdr) + "\n")
    for i, (fam, q, ir, pq, pir) in enumerate(R, 1): f.write("\t".join([f"g{i:03d}", fam, q, ir, pq, pir]) + "\n")
print(len(R), "rows")
