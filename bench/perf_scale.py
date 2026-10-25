"""Scale cases (x86, 2 pinned cores): (1) a 90-day history (3x the 28-day fixtures); (2) a large schema: 300 extra registered signals with their own storage table; (3) the cost of the attribution calculation.
    python tools_pin.py 8,9 bench/perf_scale.py
Timings are wall-clock milliseconds for the complete engine (parse -> plan -> execute -> compose -> verify) unless stated; memory is the process peak."""
import json, os, random, shutil, sqlite3, statistics as st, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import nl2ir, schema_overlay as SO, signal_registry as R
from vgpt.brain import Brain
from vgpt.query_ir import Query


def peak_mb():
    try:
        import psutil; mi = psutil.Process().memory_info(); return round(getattr(mi, "peak_wset", mi.rss) / 1e6)
    except Exception: return None


def time_q(qc, ir, reps=3):
    xs = []
    for _ in range(reps):
        t = time.perf_counter(); r = qc.execute(Query.from_text(ir)); xs.append((time.perf_counter() - t) * 1000)
    return {"status": r.status, "ms_first": round(xs[0], 1), "ms_best": round(min(xs), 1)}


QS = ["op=aggregate tgt=speed agg=mean t=last:7:day", "op=extreme tgt=battery_temperature agg=max t=last:30:day", "op=aggregate tgt=battery_temperature agg=mean t=all", "op=count t=all ev=HARSH_BRAKE", "op=list_events t=all ev=HARSH_BRAKE evs=all", "op=percent_time t=last:30:day c=highway",
      "op=rank t=all sel=trip:distance_driven:desc:5:-", "op=compare tgt=wh_per_km agg=mean t=last:30:day grp=city,highway", "op=trend tgt=battery_temperature t=all", "op=detect_change tgt=wh_per_km t=all", "op=explain_change tgt=wh_per_km t=last:14:day", "op=explain_change tgt=coolant_flow t=last:30:day"]


def main():
    res = {"note": "x86 laptop, 2 pinned cores; not ARM"}
    for name, path, now in (("28 days (fixture)", os.path.join(HERE, "data", "gen_courier_ext.db"), 1790406000), ("90 days", os.path.join(HERE, "data", "long90.db"), None)):
        if not os.path.exists(path): res[name] = "missing: " + path; continue
        t = time.perf_counter(); b = Brain(path, slm_ir=None, now=now); qc = b.qc(); load = (time.perf_counter() - t) * 1000
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True); rows = {t_: c.execute(f"SELECT COUNT(*) FROM {t_}").fetchone()[0] for t_ in ("sig_1m", "sig_1s", "trips", "events")}
        res[name] = {"db_mb": round(os.path.getsize(path) / 1e6, 1), "rows": rows, "engine_load_ms": round(load, 1), "queries": {ir: time_q(qc, ir) for ir in QS}}
    # large schema
    tmp = tempfile.mkdtemp(); db = os.path.join(tmp, "wide.db"); shutil.copy(os.path.join(HERE, "data", "gen_courier_ext.db"), db); c = sqlite3.connect(db); rng = random.Random(3); n = 300
    c.execute("CREATE TABLE wide_1m (ts INTEGER PRIMARY KEY, " + ", ".join(f"w{i} INTEGER" for i in range(n)) + ")"); ts = [r[0] for r in c.execute("SELECT ts FROM sig_1m ORDER BY ts")]
    c.executemany("INSERT INTO wide_1m VALUES (" + ",".join("?" * (n + 1)) + ")", [[t] + [rng.randint(0, 1000) for _ in range(n)] for t in ts]); c.commit(); c.close()
    spec = {"concepts": [{"name": f"extra_signal_{i:03d}", "label": f"auxiliary channel {i}", "unit": "%", "category": "vehicle", "synonyms": [f"auxiliary channel {i}", f"aux channel {i}"], "store": {"1m": ["wide_1m", f"w{i}", 10.0]}} for i in range(n)]}
    t = time.perf_counter(); tok = SO.apply(spec); res["large_schema"] = {"signals_added": n, "overlay_apply_ms": round((time.perf_counter() - t) * 1000, 1), "concepts_total": len(R.CONCEPTS)}
    t = time.perf_counter(); b = Brain(db, slm_ir=None, now=1790406000); qc = b.qc(); res["large_schema"]["engine_load_ms"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter(); p = nl2ir.parse("average auxiliary channel 217 last week", now=1790406000, tz=3600, profile=qc.profile); res["large_schema"]["parse_ms"] = round((time.perf_counter() - t) * 1000, 2); res["large_schema"]["parsed_to"] = p.query.to_text() if p.query else None
    res["large_schema"]["queries"] = {ir: time_q(qc, ir) for ir in ("op=aggregate tgt=extra_signal_217 agg=mean t=last:7:day", "op=extreme tgt=extra_signal_042 agg=max t=last:14:day", "op=aggregate tgt=speed agg=mean t=last:7:day")}
    from vgpt import candidates as CD
    t = time.perf_counter(); cs = CD.generate("average auxiliary channel 217 last week", None, qc.profile, 1790406000, 3600); res["large_schema"]["candidate_generation_ms"] = round((time.perf_counter() - t) * 1000, 1); res["large_schema"]["candidates"] = len(cs)
    from vgpt.schema_view import SchemaView
    t = time.perf_counter(); sv = SchemaView(); txt = sv.render() if hasattr(sv, "render") else ""; res["large_schema"]["schema_prompt_chars_with_overlay"] = len(txt); res["large_schema"]["schema_render_ms"] = round((time.perf_counter() - t) * 1000, 1)
    SO.restore(tok); shutil.rmtree(tmp, ignore_errors=True); res["peak_mb"] = peak_mb()
    out = os.path.join(HERE, "results", "perf_scale.json"); json.dump(res, open(out, "w"), indent=1)
    for k in ("28 days (fixture)", "90 days"):
        v = res.get(k)
        if isinstance(v, dict): print(k, v["db_mb"], "MB", v["rows"], "load", v["engine_load_ms"], "ms"); [print("   ", ir[:70].ljust(72), q["status"], q["ms_first"], "ms first", q["ms_best"], "ms best") for ir, q in v["queries"].items()]
    print("large schema:", json.dumps({k: v for k, v in res["large_schema"].items() if k != "queries"})); [print("   ", ir[:70].ljust(72), q) for ir, q in res["large_schema"]["queries"].items()]; print("peak MB", res["peak_mb"])


if __name__ == "__main__": main()
