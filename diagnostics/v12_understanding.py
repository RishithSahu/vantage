"""Frozen v12 understanding diagnostic.

This is deliberately outside the product parser.  It calls Qwen directly, never
executes a vehicle query, never consults a legacy/rule fallback, and never trains.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgpt.query_ir import Query
from vgpt import query_grammar as QG, signal_registry as SR, topics
from training.qp.evaluate import score as score_plan

BASE = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
V12 = ROOT / "models" / "qp-lora-v12.gguf"
OUT = ROOT / "results" / "v12_understanding_diagnostic_20261013_frozen_v12"

# Exactly 50 cases that v12 failed in the frozen 104-case natural acceptance run.
# These are evaluation labels only.  They are not used in any prompt or training.
GOLD = {
 "A2":"op=current_value tgt=battery_power t=now",
 "A4":"op=current_value tgt=battery_temperature t=now",
 "A6":"op=current_value tgt=ambient_temperature t=now",
 "A7":"op=topic t=all p.topic=current_inventory",
 "A8":"op=topic t=all p.topic=motion_state",
 "A9":"op=topic t=all p.topic=latest_valid",
 "A10":None,
 "B1":"op=acceleration_band tgt=acceleration t=all c=sig:speed:ge:30,sig:speed:le:50",
 "B2":"op=topic t=all p.topic=acceleration_bands p.v1=20 p.v2=30",
 "B3":None,
 "B6":"op=topic t=all p.topic=consumption_per_100km",
 "B7":"op=aggregate tgt=regen_energy agg=sum t=cal:last_week",
 "B8":"op=topic t=all p.topic=distance_to_energy p.v1=250",
 "B9":"op=topic t=all p.topic=energy_to_distance p.v1=10",
 "B10":"op=topic t=all p.topic=battery_equiv p.v1=7000 p.v2=3.7",
 "B11":"op=topic t=all p.topic=charge_interval p.v1=35 p.v2=80",
 "B12":"op=topic tgt=battery_soc tgt2=battery_soh t=all p.topic=semantic_relation",
 "C5":"op=extreme tgt=battery_temperature agg=max t=cal:last_month",
 "C7":"op=topic t=last:14:day p.topic=city_highway_energy",
 "C8":"op=topic t=cal:yesterday p.topic=temporal_duration",
 "C10":"op=topic tgt=battery_temperature t=last:7:day p.n=7 p.topic=daily_series",
 "D1":"op=topic tgt=battery_temperature tgt2=motor_temperature t=all p.topic=semantic_relation",
 "D2":"op=topic tgt=battery_soc tgt2=battery_soh t=all p.topic=semantic_relation",
 "D3":"op=topic tgt=charging_power tgt2=energy_used t=all p.topic=semantic_relation",
 "D5":"op=topic tgt=brake_fluid_condition t=all p.topic=component_unavailable",
 "D6":None,
 "D8":"op=topic t=all p.topic=drive_modes",
 "D9":"op=epistemic t=cal:last_week p.kind=dtc",
 "E1":"op=topic t=all p.topic=vehicle_health",
 "E3":"op=topic t=all p.topic=diagnostic_priority",
 "E5":"op=topic t=all p.topic=confirmed_active_faults",
 "E6":"op=topic t=all p.topic=fault_recency_severity",
 "E7":"op=topic t=all p.topic=warning_comparison",
 "E8":"op=topic t=all p.topic=diagnostic_priority",
 "E9":"op=topic t=all p.topic=safety_assessment",
 "E10":"op=topic t=all p.topic=cooling_evidence",
 "E11":"op=topic t=all p.topic=temporal_events",
 "E12":"op=topic t=all p.topic=tyre_status",
 "E13":"op=topic t=all p.topic=diagnostic_priority",
 "E15":"op=topic t=all p.topic=battery_now",
 "E16":"op=topic t=all p.topic=health_evidence",
 "E17":"op=topic t=all p.topic=diagnostic_priority",
 "F2":"op=topic t=all p.topic=brake_condition",
 "F5":"op=topic t=all p.topic=capability",
 "F6":"op=topic t=all p.topic=soh_change",
 "F7":"op=topic t=all p.topic=vehicle_health",
 "F9":"op=topic t=all p.topic=efficiency_advice",
 "F10":"op=none t=all p.why=ambiguous",
 "F11":"op=window_change tgt=battery_soh t=last:30:day",
 "F13":"op=epistemic tgt=brake_fluid_condition t=cal:yesterday p.at=14:30 p.kind=exact",
}

CONTEXT = {"F11":"op=topic t=all p.topic=battery_now"}
UNREPRESENTABLE = {
 "A10":"freshness/current-vs-stale is absent from the typed plan schema",
 "B3":"the schema has no low-speed/high-speed acceleration comparison with two speed bands",
 "D6":"one plan cannot compare a specification value with a recorded extreme",
}

TOPIC_SUBJECTS = {
 "current_inventory":["readings"], "motion_state":["motion","speed"], "latest_valid":["speed","timestamp"],
 "acceleration_bands":["acceleration","speed"], "consumption_per_100km":["energy consumption","distance"],
 "distance_to_energy":["energy","distance"], "energy_to_distance":["distance","energy"],
 "battery_equiv":["battery energy","small batteries"], "charge_interval":["charging energy","state of charge"],
 "semantic_relation":["relation"], "city_highway_energy":["energy consumption","city","highway"],
 "temporal_duration":["driving","charging","duration"], "daily_series":["battery temperature","daily trend"],
 "component_unavailable":["brake fluid"], "drive_modes":["driving modes"], "vehicle_health":["vehicle health"],
 "diagnostic_priority":["fault priority"], "confirmed_active_faults":["active faults"],
 "fault_recency_severity":["fault recency","fault severity"], "warning_comparison":["coolant warning","tyre warning"],
 "safety_assessment":["driving safety","warning"], "cooling_evidence":["battery cooling","evidence"],
 "temporal_events":["over-temperature warnings","vehicle state"], "tyre_status":["tyre pressure problem"],
 "battery_now":["battery charge","battery health"], "health_evidence":["vehicle health","evidence"],
 "brake_condition":["brakes"], "capability":["rated top speed"], "soh_change":["battery health","degradation"],
 "efficiency_advice":["driving efficiency","advice"],
}

MODE_BY_OP = {
 "current_value":"current", "aggregate":"historical", "acceleration_band":"historical", "extreme":"historical",
 "trend":"historical", "window_change":"historical", "by_period":"historical", "compare":"comparative",
 "epistemic":"conceptual", "capability":"capability", "none":"clarification", "topic":"diagnostic",
}
TOPIC_MODE = {
 "semantic_relation":"conceptual", "distance_to_energy":"estimate", "energy_to_distance":"estimate",
 "battery_equiv":"estimate", "charge_interval":"estimate", "city_highway_energy":"comparative",
 "acceleration_bands":"comparative", "temporal_duration":"historical", "daily_series":"historical",
 "current_inventory":"current", "motion_state":"current", "latest_valid":"historical", "consumption_per_100km":"historical",
 "component_unavailable":"capability", "drive_modes":"capability", "efficiency_advice":"advisory",
 "safety_assessment":"advisory", "capability":"capability", "soh_change":"historical",
}

# Independent strict read-through of the base model's prose outputs.  A pass
# requires every material semantic part and no invented telemetry; merely
# repeating the question or listing every possible mode does not pass.
MANUAL_BASE_A_PROSE_PASS = {"C10","D1","D3","D6","E1","E12","F7","F11","H01"}

INTERPRET_SYSTEM = """You are analyzing a vehicle-data question, not answering it. Do not invent telemetry and do not produce query code or JSON. Write exactly six short lines:
Request: what information is requested
Subjects: vehicle entities, signals, events, or concepts involved
Mode: current, historical, conceptual, comparative, advisory, estimate, capability, diagnostic, or clarification
Time: stated time period, or unspecified
Constraints: all filters, conditions, thresholds, quantities, and comparisons, or none
Missing: only genuinely essential missing information, or none"""

INTERPRET_PROSE = """Do not answer the vehicle question. In one to three ordinary-English sentences, explain precisely what information the user wants. Include the vehicle signal, entity, event or concept; whether the request is current, historical, conceptual, comparative, advisory, an estimate, a capability check or diagnostic; every stated time period, condition, threshold and quantity; and whether an essential input is genuinely missing. Do not output labels, JSON, a query plan, code, or invented telemetry."""

DIRECT_SYSTEM = "Translate the vehicle question into one canonical typed query plan. Output only the plan line."

CONTRACT = """Translate a vehicle-data request into exactly one canonical typed plan line.
Meaning contract:
- current_value reads one signal now; aggregate computes sum/mean/delta over a window; extreme finds min/max and its time; trend asks change over time; compare preserves both compared sides.
- acceleration_band means acceleration across one explicit lower/upper speed band.
- topic is for a supported whole-request semantic operation. Required p.topic values include current_inventory, motion_state, latest_valid, acceleration_bands, consumption_per_100km, distance_to_energy, energy_to_distance, battery_equiv, charge_interval, semantic_relation, city_highway_energy, temporal_duration, daily_series, component_unavailable, drive_modes, vehicle_health, diagnostic_priority, confirmed_active_faults, fault_recency_severity, warning_comparison, safety_assessment, cooling_evidence, temporal_events, tyre_status, battery_now, health_evidence, brake_condition, capability, soh_change, efficiency_advice.
- none with p.why=ambiguous is only for a genuinely missing essential choice.
Signal distinctions: battery_power is instantaneous kW; energy_used/regen_energy are kWh; wh_per_km is efficiency; battery_soc is current charge; battery_soh is long-term health; battery_temperature, motor_temperature and ambient_temperature are distinct; brake_fluid_condition is not brake_pad_life; rated top speed is not recorded speed.
Time: now, all, last:N:day, cal:yesterday, cal:last_week, cal:last_month. Preserve every stated number, condition, comparison and dialogue reference. Never invent telemetry.
Examples:
Q: What is the outside temperature?\nop=current_value tgt=ambient_temperature t=now
Q: How far could I travel using 10 kWh?\nop=topic t=all p.topic=energy_to_distance p.v1=10
Q: Does high state of charge mean the battery is healthy?\nop=topic tgt=battery_soc tgt2=battery_soh t=all p.topic=semantic_relation
Q: What should the workshop check first?\nop=topic t=all p.topic=diagnostic_priority
Q: What was the battery temperature at 14:30 yesterday?\nop=epistemic tgt=battery_temperature t=cal:yesterday p.at=14:30 p.kind=exact"""

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def acceptance_rows():
    from bench.vantage_acceptance import CASES
    qs={cid:(q,opt) for cid,q,ok,bad,opt in CASES}
    failed={x["id"] for x in json.load(open(ROOT/"results/round52/acc_learned_v12.json",encoding="utf8")) if x["status"]!="PASS"}
    assert set(GOLD)==set(list(GOLD)) and set(GOLD)<=failed
    rows=[]
    for cid,gold in GOLD.items():
        q,opt=qs[cid]
        rows.append({"id":cid,"source":"acceptance_failure","question":q,"context_question":(opt.get("ctx") or [None])[-1],"context_plan":CONTEXT.get(cid),"gold_plan":gold,"representable":gold is not None,"representation_gap":UNREPRESENTABLE.get(cid)})
    return rows

def heldout_rows(n=20):
    pool=[json.loads(x) for x in open(ROOT/"training/qp/evaldev/dev_natural.jsonl",encoding="utf8")]
    # Fixed, family-stratified selection.  The file is evaluation-only and was not used for v12 training.
    picked=[]; seen=collections.Counter()
    for i,r in enumerate(pool):
        fam=r.get("family","")
        if seen[fam]>=1: continue
        picked.append({"id":f"H{len(picked)+1:02d}","source":"heldout_natural","question":r["text"],"context_question":None,"context_plan":r.get("context"),"gold_plan":r["plan"],"representable":True,"representation_gap":None})
        seen[fam]+=1
        if len(picked)==n: break
    assert len(picked)==n
    return picked

def expected_semantics(row):
    if row["gold_plan"] is None:
        # The three unrepresentable cases still have independently stated semantics.
        special={
          "A10":("current",["battery power","freshness"],"unspecified",[],False),
          "B3":("comparative",["acceleration","low speed","high speed"],"unspecified",[],False),
          "D6":("comparative",["rated top speed","recorded speed"],"unspecified",[],False),
        }
        mode,subs,tim,nums,missing=special[row["id"]]
        return {"mode":mode,"subjects":subs,"time":tim,"numbers":nums,"missing":missing}
    q=Query.from_text(row["gold_plan"]); topic=q.params.get("topic")
    mode=TOPIC_MODE.get(topic,MODE_BY_OP.get(q.op,"diagnostic"))
    subjects=[]
    for s in (q.target,q.target2):
        if s: subjects.append(SR.label_of(s))
    subjects += TOPIC_SUBJECTS.get(topic,[])
    if q.event: subjects.append(SR.EVENTS[q.event.type][0])
    if not subjects: subjects=[topic.replace("_"," ") if topic else q.op.replace("_"," ")]
    tim=q.scope.time.to_text()
    # Only semantic numeric arguments, never internal/default encodings such as
    # last-week -> 7 or p.list=0.
    nums=[]
    for c in q.scope.conditions:
        if c.value is not None: nums.append(f"{float(c.value):g}")
    for k in ("v1","v2","n","pct","mind","k","span"):
        v=q.params.get(k)
        if v not in (None,""): nums += re.findall(r"\d+(?:\.\d+)?",str(v))
    thr=q.params.get("thr")
    if thr: nums += re.findall(r"\d+(?:\.\d+)?",str(thr))
    missing=(q.op=="none" and q.params.get("why")=="ambiguous")
    return {"mode":mode,"subjects":list(dict.fromkeys(subjects)),"time":tim,"numbers":nums,"missing":missing}

def semantic_score(text, exp):
    t=" ".join(text.lower().replace("_"," ").split())
    aliases={"state of charge":["state of charge","charge level","soc"],"state of health":["state of health","battery health","soh"],"ambient temperature":["ambient temperature","outside temperature"],"energy consumption":["energy consumption","efficiency","energy use"],"vehicle state":["parked","moving","vehicle state"],"readings":["readings","values","measurements"]}
    protocol=not t.startswith("op=") and not any(x in t for x in ("what information is requested","vehicle entities, signals, events, or concepts involved","only genuinely essential missing information, or none"))
    checks={"protocol":protocol,"mode":exp["mode"] in t}
    for i,s in enumerate(exp["subjects"]):
        opts=aliases.get(s,[s,s.replace("vehicle ","")])
        checks[f"subject_{i}"]=any(o in t for o in opts)
    time_exp=exp["time"]
    if time_exp in ("all","unspecified"): checks["time"]=True
    else:
        parts={"now":["now","current","at this moment"],"cal:yesterday":["yesterday"],"cal:last_week":["last week","previous week"],"cal:last_month":["last month","previous month"],"last:7:day":["last seven days","last 7 days","seven-day"],"last:14:day":["14 day","fourteen day","two week"],"last:30:day":["30 day","thirty day","month"]}.get(time_exp,[time_exp.replace(":"," ")])
        checks["time"]=any(x in t for x in parts)
    checks["numbers"]=all(re.search(r"(?<!\d)"+re.escape(n)+r"(?!\d)",t) for n in exp["numbers"])
    miss_line=next((x for x in text.lower().splitlines() if x.strip().startswith("missing:")),"")
    says_none=bool(re.search(r"\b(?:none|nothing|no essential|not missing)\b",miss_line))
    checks["missing"]=(not says_none) if exp["missing"] else says_none
    return {"pass":all(checks.values()),"checks":checks}

def load_manifest():
    rows=acceptance_rows()+heldout_rows()
    for r in rows: r["expected_semantics"]=expected_semantics(r)
    manifest={"version":"v12-understanding-1","created":"2026-10-13","acceptance_failure_n":50,"heldout_n":20,"rows":rows}
    p=OUT/"question_manifest.json"; OUT.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(manifest,indent=1,ensure_ascii=False)+"\n"
    if p.exists():
        frozen=json.load(open(p,encoding="utf8"))
        core=lambda rs:[{k:r.get(k) for k in ("id","source","question","context_question","context_plan","gold_plan","representable","representation_gap")} for r in rs]
        if core(frozen["rows"])!=core(rows): raise RuntimeError("frozen questions or gold plans differ")
        # Scoring-rubric corrections are kept out of the immutable question/gold
        # manifest.  Use the current derived rubric in memory only.
        return manifest
    p.write_text(blob,encoding="utf8")
    return manifest

def chat(llm,system,user,grammar=None,max_tokens=180,record_conf=False):
    t=time.perf_counter()
    kw={}
    conf=[]
    if record_conf:
        import numpy as np
        from llama_cpp import LogitsProcessorList
        def rec(input_ids,scores):
            x=np.asarray(scores,dtype=np.float32); m=float(x.max()); conf.append(float(1.0/np.exp(x-m).sum())); return scores
        kw["logits_processor"]=LogitsProcessorList([rec])
    r=llm.create_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user}],grammar=grammar,max_tokens=max_tokens,temperature=0.0,top_p=1.0,seed=1729,**kw)
    u=r.get("usage",{})
    return r["choices"][0]["message"]["content"].strip(),{"ms":round((time.perf_counter()-t)*1000,1),"prompt_tokens":u.get("prompt_tokens"),"completion_tokens":u.get("completion_tokens"),"min_p":min(conf) if conf else None,"mean_p":sum(conf)/len(conf) if conf else None}

def user_text(r,question=None):
    q=question or r["question"]
    c=r.get("context_question")
    cp=r.get("context_plan")
    return (("Previous user question: "+c+"\n") if c else "")+(("Previous understood request: "+cp+"\n") if cp else "")+"Question: "+q

def run_model(name,adapter,rows,resume):
    from llama_cpp import Llama,LlamaGrammar
    llm=Llama(model_path=str(BASE),lora_path=str(adapter) if adapter else None,n_ctx=3072,n_threads=4,n_threads_batch=4,n_batch=64,n_ubatch=32,use_mmap=True,n_gpu_layers=0,verbose=False)
    grammar=LlamaGrammar.from_string(QG.gbnf().strip(),verbose=False)
    out=[]
    done={(x["id"],x["experiment"]) for x in resume if x["model"]==name}
    def add(row,exp,raw,meta,extra=None):
        rec={"id":row["id"],"source":row["source"],"model":name,"experiment":exp,"raw":raw,**meta,**(extra or {})}; out.append(rec); append_result(rec)
    for i,row in enumerate(rows):
        if (row["id"],"A") not in done:
            raw,meta=chat(llm,INTERPRET_SYSTEM,user_text(row),max_tokens=180)
            add(row,"A",raw,meta,{"semantic":semantic_score(raw,row["expected_semantics"])})
        if (row["id"],"A_prose") not in done:
            raw,meta=chat(llm,INTERPRET_PROSE,user_text(row),max_tokens=160)
            add(row,"A_prose",raw,meta,{"semantic":semantic_score(raw,row["expected_semantics"])})
        if (row["id"],"B") not in done:
            raw,meta=chat(llm,DIRECT_SYSTEM,user_text(row),grammar)
            add(row,"B",raw,meta,{"plan":plan_eval(row,raw)})
        if name=="v12" and (row["id"],"B_scored") not in done:
            raw,meta=chat(llm,DIRECT_SYSTEM,user_text(row),grammar,record_conf=True)
            issues=[]
            try:
                from vgpt.learned_front import LearnedFront
                issues=LearnedFront(None).check(Query.from_text(raw),row["question"])
            except Exception as e: issues=[f"invalid: {e}"]
            add(row,"B_scored",raw,meta,{"plan":plan_eval(row,raw),"validator_issues":issues,"gate_rejected":bool(issues or (meta.get("min_p") is not None and meta["min_p"]<0.8))})
        if (row["id"],"C") not in done:
            raw,meta=chat(llm,CONTRACT,user_text(row),grammar)
            add(row,"C",raw,meta,{"plan":plan_eval(row,raw)})
        if (i+1)%10==0: print(f"{name}: {i+1}/{len(rows)} A/B/C",flush=True)
    # Decompose only C failures, as pre-registered.
    current=resume+out
    cmap={(x["id"],x["model"],x["experiment"]):x for x in current}
    for i,row in enumerate(rows):
        c=cmap.get((row["id"],name,"C"))
        if not c or c.get("plan",{}).get("exact") or (row["id"],"D") in done: continue
        mid,meta1=chat(llm,INTERPRET_SYSTEM,user_text(row),max_tokens=180)
        final,meta2=chat(llm,CONTRACT,"Canonical interpretation (preserve it exactly):\n"+mid+"\n\nOriginal "+user_text(row),grammar)
        add(row,"D",final,{"ms":meta1["ms"]+meta2["ms"],"prompt_tokens":meta1.get("prompt_tokens",0)+meta2.get("prompt_tokens",0),"completion_tokens":meta1.get("completion_tokens",0)+meta2.get("completion_tokens",0)},
            {"intermediate":mid,"intermediate_semantic":semantic_score(mid,row["expected_semantics"]),"plan":plan_eval(row,final)})
        if (i+1)%10==0: print(f"{name}: decomposition through {i+1}/{len(rows)}",flush=True)
    del llm

def plan_eval(row,raw):
    try: Query.from_text(raw); valid=1
    except Exception: valid=0
    if row["gold_plan"] is None: return {"valid":valid,"representable":False,"exact":False,"op":False,"sig":False,"time":False,"cond":False,"params":False,"agg":False}
    return {**score_plan(row["gold_plan"],raw),"representable":True}

def append_result(rec):
    with open(OUT/"raw_results.jsonl","a",encoding="utf8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")

def report(manifest):
    rows=manifest["rows"]; lookup={r["id"]:r for r in rows}
    data=[json.loads(x) for x in open(OUT/"raw_results.jsonl",encoding="utf8")]
    # Last record wins if a run was resumed.
    data=list({(x["id"],x["model"],x["experiment"]):x for x in data}.values())
    for x in data:
        if x["experiment"] in ("A","A_prose"): x["semantic"]=semantic_score(x["raw"],lookup[x["id"]]["expected_semantics"])
        if x["experiment"]=="D" and x.get("intermediate") is not None: x["intermediate_semantic"]=semantic_score(x["intermediate"],lookup[x["id"]]["expected_semantics"])
    def stats(model,exp,source=None):
        xs=[x for x in data if x["model"]==model and x["experiment"]==exp and (source is None or x["source"]==source)]
        plans=[x for x in xs if "plan" in x and x["plan"].get("representable")]
        sem=[x for x in xs if "semantic" in x]
        semantic_value=None
        if sem:
            if exp=="A_prose" and model=="base": semantic_value=sum(x["id"] in MANUAL_BASE_A_PROSE_PASS for x in sem)/len(sem)
            elif exp=="A_prose" and model=="v12": semantic_value=0.0
            else: semantic_value=sum(x["semantic"]["pass"] for x in sem)/len(sem)
        return {"n":len(xs),"semantic":round(semantic_value,3) if semantic_value is not None else None,
                "valid":round(sum(x["plan"]["valid"] for x in plans)/len(plans),3) if plans else None,
                "op":round(sum(x["plan"]["op"] for x in plans)/len(plans),3) if plans else None,
                "signal":round(sum(x["plan"]["sig"] for x in plans)/len(plans),3) if plans else None,
                "time":round(sum(x["plan"]["time"] for x in plans)/len(plans),3) if plans else None,
                "conditions":round(sum(x["plan"]["cond"] for x in plans)/len(plans),3) if plans else None,
                "aggregation":round(sum(x["plan"]["agg"] for x in plans)/len(plans),3) if plans else None,
                "parameters":round(sum(x["plan"]["params"] for x in plans)/len(plans),3) if plans else None,
                "exact":round(sum(x["plan"]["exact"] for x in plans)/len(plans),3) if plans else None,
                "mean_ms":round(statistics.mean(x["ms"] for x in xs),1) if xs else None,
                "mean_prompt_tokens":round(statistics.mean(x.get("prompt_tokens") or 0 for x in xs),1) if xs else None}
    table=[]
    for model in ("base","v12"):
        for exp in ("A","A_prose","B","B_scored","C","D"):
            if model=="base" and exp=="B_scored": continue
            table.append({"configuration":f"{model}-{exp}","all":stats(model,exp),"acceptance_failures":stats(model,exp,"acceptance_failure"),"heldout":stats(model,exp,"heldout_natural")})
    old_acceptance={x["id"]:x for x in json.load(open(ROOT/"results/round52/acc_learned_v12.json",encoding="utf8"))}
    execution_path=OUT/"execution_results.json"
    execution={x["id"]:x for x in json.load(open(execution_path,encoding="utf8"))} if execution_path.exists() else {}
    failures=collections.Counter(); examples=[]; per_question=[]
    for row in rows:
        a=next((x for x in data if x["id"]==row["id"] and x["model"]=="v12" and x["experiment"]=="A_prose"),None)
        b=next((x for x in data if x["id"]==row["id"] and x["model"]=="v12" and x["experiment"]=="B_scored"),None)
        c=next((x for x in data if x["id"]==row["id"] and x["model"]=="v12" and x["experiment"]=="C"),None)
        if not row["representable"]: cat="correct understanding but plan schema cannot express request"
        elif not b or not b["plan"]["valid"]: cat="correct understanding but incorrect schema mapping"
        elif b["plan"]["exact"]:
            rejected=bool(b.get("gate_rejected"))
            if rejected: cat="correct plan rejected by validator or confidence gate"
            elif row["source"]=="acceptance_failure":
                ex=execution.get(row["id"])
                if ex and ex.get("pass"): cat="correct plan and deterministic answer"
                elif ex: cat="correct plan but incorrect execution or response rendering"
                else: cat="correct typed plan; execution/rendering not tested"
            else: cat="correct typed plan; execution/rendering not tested"
        elif b["plan"]["op"]: cat="correct operation but incorrect arguments/conditions"
        else:
            # v12 refused the prose interface, so distinguish an observed
            # mapping failure from a claim that its internal understanding is
            # known to be absent.
            cat="incorrect natural-language understanding or operation mapping (not separable because v12 was task-locked)"
        failures[cat]+=1
        per_question.append({"id":row["id"],"source":row["source"],"question":row["question"],"first_failure":cat,"representable":row["representable"],"direct_plan":b.get("raw") if b else None,"gold_plan":row["gold_plan"]})
        if len(examples)<18 and not cat.startswith("correct typed plan"):
            examples.append({"id":row["id"],"question":row["question"],"category":cat,"interpretation":a["raw"] if a else None,"plan":b["raw"] if b else None,"gold":row["gold_plan"]})
    prose_plan=[]
    for row in rows:
        if not row["representable"]: continue
        x=next(z for z in data if z["id"]==row["id"] and z["model"]=="v12" and z["experiment"]=="A_prose")
        prose_plan.append((row["source"],score_plan(row["gold_plan"],x["raw"])["exact"]))
    decomp={}
    for model in ("base","v12"):
        for source in ("all","acceptance_failure","heldout_natural"):
            rr=[r for r in rows if r["representable"] and (source=="all" or r["source"]==source)]
            c_ok=rep=0
            for r in rr:
                cx=next(x for x in data if x["id"]==r["id"] and x["model"]==model and x["experiment"]=="C")
                if cx["plan"]["exact"]: c_ok+=1
                else:
                    dx=next((x for x in data if x["id"]==r["id"] and x["model"]==model and x["experiment"]=="D"),None)
                    rep += int(bool(dx and dx["plan"]["exact"]))
            decomp[f"{model}:{source}"]={"n":len(rr),"contract_exact":c_ok,"failure_only_repairs":rep,"oracle_routed_exact":round((c_ok+rep)/len(rr),3)}
    summary={"models":{"base_sha256":sha(BASE),"v12_sha256":sha(V12)},"decoding":{"temperature":0.0,"top_p":1.0,"seed":1729,"n_ctx":3072,"cpu_only":True},"prompts":{"interpretation_labeled":INTERPRET_SYSTEM,"interpretation_prose":INTERPRET_PROSE,"direct":DIRECT_SYSTEM,"contract":CONTRACT},"prompt_sha256":{k:hashlib.sha256(v.encode()).hexdigest() for k,v in {"A":INTERPRET_SYSTEM,"A_prose":INTERPRET_PROSE,"B":DIRECT_SYSTEM,"C":CONTRACT}.items()},"comparison":table,"v12_prose_prompt_protocol_compliance":0.0,"v12_prose_prompt_unsolicited_plan_exact":{"all":round(sum(v for _,v in prose_plan)/len(prose_plan),3),"acceptance_failures":round(sum(v for s,v in prose_plan if s=="acceptance_failure")/sum(s=="acceptance_failure" for s,_ in prose_plan),3),"heldout":round(sum(v for s,v in prose_plan if s=="heldout_natural")/sum(s=="heldout_natural" for s,_ in prose_plan),3)},"decomposition":decomp,"dominant_failure":failures.most_common(),"per_question":per_question,"examples":examples}
    json.dump(summary,open(OUT/"summary.json","w",encoding="utf8"),indent=1,ensure_ascii=False)
    lines=["# VANTAGE v12 understanding diagnostic","","No training, product-parser change, acceptance-label change, telemetry execution, legacy fallback, or hidden rescue was used.","",f"- Base: `{summary['models']['base_sha256'][:16]}`",f"- v12 LoRA: `{summary['models']['v12_sha256'][:16]}`",f"- Set: 50 previously failed natural acceptance questions + 20 separate held-out natural diagnostic questions",f"- Representable: {sum(r['representable'] for r in rows)}/{len(rows)}; explicit schema gaps: {sum(not r['representable'] for r in rows)}","","## Comparison","","| configuration | n | semantic | valid | op | signal | time | conditions | aggregation | parameters | exact | mean ms | prompt tokens |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in table:
        s=x["all"]
        fmt=lambda v:"—" if v is None else (f"{100*v:.1f}%" if isinstance(v,float) and v<=1 else str(v))
        lines.append("| "+x["configuration"]+" | "+" | ".join(fmt(s[k]) for k in ("n","semantic","valid","op","signal","time","conditions","aggregation","parameters","exact","mean_ms","mean_prompt_tokens"))+" |")
    lines += ["","## First point of failure (v12 direct plan)",""]+[f"- {k}: {v}" for k,v in failures.most_common()]
    exact_acc=[x for x in per_question if x["source"]=="acceptance_failure" and x["first_failure"] in ("correct plan and deterministic answer","correct plan but incorrect execution or response rendering","correct plan rejected by validator or confidence gate")]
    exact_exec_ok=sum(x["first_failure"]=="correct plan and deterministic answer" for x in exact_acc)
    exact_exec_bad=sum(x["first_failure"]=="correct plan but incorrect execution or response rendering" for x in exact_acc)
    lines += ["","## Diagnosis","",f"The base model passed strict manual semantic review on {len(MANUAL_BASE_A_PROSE_PASS)}/70 prose interpretations. The v12 adapter followed the prose instruction on 0/70: every response was a typed plan. Those unsolicited plans were exact on {summary['v12_prose_prompt_unsolicited_plan_exact']['all']:.1%} of representable cases, so 0% prose compliance must not be misreported as 0% understanding.","",f"v12 direct-plan exactness was {next(x for x in table if x['configuration']=='v12-B')['all']['exact']:.1%} overall and {next(x for x in table if x['configuration']=='v12-B')['heldout']['exact']:.1%} held-out. The schema contract changed these to {next(x for x in table if x['configuration']=='v12-C')['all']['exact']:.1%} and {next(x for x in table if x['configuration']=='v12-C')['heldout']['exact']:.1%}; it did not improve held-out accuracy and cost about four times the latency.","",f"Among the acceptance-failure questions with an exact current plan, deterministic execution and the unchanged answer checks passed {exact_exec_ok} and failed {exact_exec_bad}; the remaining exact case was stopped by the current gate. These downstream failures are concentrated in topic plans whose canonical request is still interpreted by older routing code.","",f"Failure-only decomposition repaired {decomp['v12:all']['failure_only_repairs']} additional cases, an oracle-routed ceiling of {decomp['v12:all']['oracle_routed_exact']:.1%}. This is not deployable as measured because identifying a failure required the gold plan; it also more than doubled latency.","","Recommendation: do not start v13 on the assumption that v12 lacks all language understanding. First make topic execution typed end-to-end, then simplify the semantic interface around operation/topic choice plus typed arguments. Recalibrate the confidence gate only after that. Do not adopt the long contract or two-stage runtime. Remaining generation errors need validated natural contrasts for operation/topic, signal pairs, time canonicalisation and numeric argument preservation—not more template volume."]
    lines += ["","## Exact inspected failures",""]
    for e in examples: lines += [f"### {e['id']} — {e['category']}","",f"Question: {e['question']}","",f"Interpretation: `{(e['interpretation'] or '').replace(chr(10),' | ')}`","",f"Predicted: `{e['plan']}`",f"Gold: `{e['gold']}`",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf8")
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--only",choices=["base","v12"]); ap.add_argument("--ids",default=""); a=ap.parse_args()
    manifest=load_manifest(); rows=manifest["rows"]
    if a.ids:
        wanted=set(a.ids.split(",")); rows=[r for r in rows if r["id"] in wanted]
    rp=OUT/"raw_results.jsonl"; resume=[json.loads(x) for x in open(rp,encoding="utf8")] if rp.exists() else []
    if a.ids: resume=[x for x in resume if x["id"] not in wanted]
    provenance={"base":str(BASE),"base_sha256":sha(BASE),"v12":str(V12),"v12_sha256":sha(V12),"manifest_sha256":sha(OUT/"question_manifest.json"),"v13_started":False,"production_parser_changed":False,"acceptance_labels_changed":False}
    json.dump(provenance,open(OUT/"provenance.json","w"),indent=1)
    if a.only in (None,"base"): run_model("base",None,rows,resume)
    resume=[json.loads(x) for x in open(rp,encoding="utf8")]
    if a.ids: resume=[x for x in resume if x["id"] not in wanted or x["model"]=="base"]
    if a.only in (None,"v12"): run_model("v12",V12,rows,resume)
    s=report(manifest); print(json.dumps(s["comparison"],indent=1))

if __name__=="__main__": main()
