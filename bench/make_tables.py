"""Markdown tables from results/*.json -> results/summary.md (so README numbers are copied, not typed)."""
import glob, json, os, statistics as st
rows = []
for f in sorted(glob.glob("results/*.json")):
    n = os.path.basename(f)[:-5]
    if n in ("can_real_eval",) or n.startswith("t_"): continue
    r = json.load(open(f)); q = r["queries"]; c = r["config"]; ms = sorted(x["ms_total"] for x in q); slm = [x["ms_total"] for x in q if x["path"] == "slm"]
    ws = r["working_sets_mb"]; assistant = ws.get("assistant", 0); total_ws = sum(ws.values())
    nums = [x["numbers_ok"] for x in q if x["numbers_ok"] is not None]; bl = [x for x in q if x["split"] == "blind"]
    rows.append(dict(name=n, model=("rules only" if c["model"] == "none" else ("Qwen2.5-0.5B" if "0.5b" in c["model"] else "Qwen2.5-1.5B")), mmap=c["mmap"], k=c["slowdown"], os=c["os_mb"],
                     commit=r["mem_peak_job_mb"], ws_total=round(total_ws), ws_assistant=round(assistant), parse=f"{sum(x['parse_ok'] for x in q)}/{len(q)}", blind=f"{sum(x['parse_ok'] for x in bl)}/{len(bl)}",
                     nums=f"{sum(nums)}/{len(nums)}", p95=round(ms[int(.95 * (len(ms) - 1))]), mx=round(ms[-1]), slm_med=round(st.median(slm)) if slm else "-", under2=sum(1 for m in ms if m < 2000)))
L = ["| config | model | mmap | OS MB | slowdown K | commit peak MB (enforced, limit 2048) | resident total MB (physical RAM need) | parse (72) | blind (24) | numbers (60) | latency p95 / max ms | SLM-path median ms | answers < 2 s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['name']} | {r['model']} | {r['mmap']} | {r['os']} | {r['k']:g} | {r['commit']:.0f} | {r['ws_total']} | {r['parse']} | {r['blind']} | {r['nums']} | {r['p95']} / {r['mx']} | {r['slm_med']} | {r['under2']}/72 |")
open("results/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
