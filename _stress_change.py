import sys
sys.path.insert(0, '.')
from vgpt.brain import Brain
from vgpt.query_ir import Query
bad = 0; tot = 0
for db, now in (("gen_courier_ext", 1790406000), ("gen_demo", None), ("gen_cold", None), ("gen_roadwarrior", None)):
    qc = Brain(f"data/{db}.db", slm_ir=None, now=now).qc()
    for tg in ("range_estimate", "wh_per_km", "battery_temperature", "coolant_flow", "tyre_pressure", "hvac_power", "motor_temperature", "speed"):
        for t in ("op=explain_change tgt=%s t=last:14:day", "op=explain_change tgt=%s t=last:7:day", "op=detect_change tgt=%s t=all", "op=explain_change tgt=%s t=last:14:day c=highway"):
            r = qc.execute(Query.from_text(t % tg)); tot += 1
            if r.status in ("unverified", "invalid"): bad += 1; print(db, tg, t, r.status, str(r.verification)[:160])
print(tot, "queries", bad, "problems")
r = Brain("data/gen_roadwarrior.db", slm_ir=None).qc().execute(Query.from_text("op=explain_change tgt=wh_per_km t=last:14:day")); print(r.answer.text[:1800])
r = Brain("data/gen_courier_ext.db", slm_ir=None, now=1790406000).qc().execute(Query.from_text("op=detect_change tgt=range_estimate t=all")); print(r.answer.text[:900])
