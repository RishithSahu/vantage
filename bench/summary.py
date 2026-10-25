import json, glob, os, statistics as st
rows = []
for f in sorted(glob.glob("results/*.json")):
    n = os.path.basename(f)[:-5]
    if n.startswith("t_"): continue
    r = json.load(open(f))
    if r.get("did_not_fit"): rows.append((n, "DID NOT FIT", r["config"].get("slowdown"), "", "", "", "", "", "")); continue
    q = r["queries"]; ms = sorted(x["ms_total"] for x in q); slm = sorted(x["ms_total"] for x in q if x["path"] == "slm"); test = [x for x in q if x["split"] == "blind"]
    nums = [x["numbers_ok"] for x in q if x["numbers_ok"] is not None]
    c = r["config"]; ing = r.get("ingest") or {}
    rows.append((n, f"{os.path.basename(c['model'])[:16]} mmap={c['mmap']} os={c['os_mb']}", c["slowdown"], f"{sum(x['parse_ok'] for x in q)}/{len(q)}", f"{sum(x['parse_ok'] for x in test)}/{len(test)}",
                 f"{sum(nums)}/{len(nums)}", f"{st.median(slm):.0f}/{slm[-1]:.0f}" if slm else "-", f"{r['mem_peak_job_mb']:.0f}", f"lag {ing.get('max_lag_s')} cpu {ing.get('cpu_s')}s/{ing.get('wall_s')}s"))
print(f"{'config':16s} {'model/os':42s} {'K':>3s} {'parse':>6s} {'blind':>6s} {'nums':>6s} {'slm med/max ms':>15s} {'peakMB':>7s}  ingest")
for r in rows: print(f"{r[0]:16s} {r[1]:42s} {r[2]!s:>3s} {r[3]:>6s} {r[4]:>6s} {r[5]:>6s} {r[6]:>15s} {r[7]:>7s}  {r[8]}")
