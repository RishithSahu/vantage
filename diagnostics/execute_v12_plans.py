"""Execute current v12 diagnostic plans directly through deterministic code.

The user's wording is never reparsed.  This is downstream diagnosis only: the
typed plan produced by Experiment B is handed to the existing executor and its
answer is checked with the unchanged acceptance predicates.
"""
import json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from bench.vantage_acceptance import CASES, judge
from vgpt.product import make_legacy_brain
from vgpt.learned_front import LearnedFront
from vgpt.query_ir import Query

OUT=ROOT/"results/v12_understanding_diagnostic_20261013_frozen_v12"
NOW=1787625900

def main():
    manifest=json.load(open(OUT/"question_manifest.json",encoding="utf8"))
    rows={r["id"]:r for r in manifest["rows"]}
    cases={cid:(q,ok,bad,opt) for cid,q,ok,bad,opt in CASES}
    raw=[json.loads(x) for x in open(OUT/"raw_results.jsonl",encoding="utf8")]
    preds={(x["id"],x["model"],x["experiment"]):x for x in raw}
    results=[]
    for cid,row in rows.items():
        if row["source"]!="acceptance_failure": continue
        pred=preds[(cid,"v12","B_scored")]["raw"]
        qtext,ok,bad,opt=cases[cid]
        t=time.perf_counter(); answer=""; error=None; route=None
        try:
            b=make_legacy_brain(str(ROOT/"data/gen_demo.db"),slm=None,now=NOW)
            result=LearnedFront(b).execute(Query.from_text(pred),qtext)
            answer=(result or {}).get("answer") or ""
            route=((result or {}).get("facts") or {}).get("route")
            b.c.close()
        except Exception as e: error=f"{type(e).__name__}: {e}"
        flags=judge(answer,ok,bad,opt) if not error else [error]
        results.append({"id":cid,"question":qtext,"predicted_plan":pred,"gold_plan":row["gold_plan"],"answer":answer,"route":route,"flags":flags,"pass":not flags,"error":error,"ms":round((time.perf_counter()-t)*1000,1)})
    json.dump(results,open(OUT/"execution_results.json","w",encoding="utf8"),indent=1,ensure_ascii=False)
    print(f"{sum(x['pass'] for x in results)}/{len(results)} current plans pass deterministic execution+render checks")
    for x in results:
        if not x["pass"]: print(x["id"],x["predicted_plan"],x["flags"],repr(x["answer"][:160]))

if __name__=="__main__": main()
