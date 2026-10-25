"""Evaluate a query-compiling backend on the held-out splits.

    python -m training.evaluate --backend rules
    python -m training.evaluate --backend slm --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf                        # base model, few-shot prompt
    python -m training.evaluate --backend slm --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora models/ir-lora.gguf   # base + LoRA, one-line prompt
    python -m training.evaluate --backend hybrid --model ... --lora ...                                                  # rules first, model only when the rules are unsure (the production path)

Metrics (per split, and per operation in the JSON):
  ir_exact          the predicted query equals the gold query (compared as data, not as text)
  op / signal / time / condition / aggregation accuracy     the individual parts of the query
  unit_acc          thresholds carry the right unit (only examples whose gold query has a unit)
  executable        the predicted query validates and runs through the whole pipeline with a verified answer (or a verified refusal)
  numeric_acc       executing the prediction gives the same result as executing the gold query (status and value)
  refusal_acc       off-topic questions and attacks become op=none (never a query)
  missing_acc       gold queries about signals this vehicle does not record are still recognised as such after prediction
  hallucination     a predicted query names a number / signal / event / group that the question does not contain (grounding check)
  verified_rate     composed answers that pass the answer verifier (evidence, units, no bare causal claims)
  latency ms (mean / p50 / p95) and tokens per query, peak process memory
Nothing is estimated: every number comes from running the thing."""
import argparse
import hashlib
import json
import os
import statistics
import math
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from training.build_dataset import DEFAULT_NOW, gold_of
from vgpt import nl2ir
from vgpt.ir_slm import grounding_issues
from vgpt.qc import QueryCompiler
from vgpt.query_ir import IRError, Query
from vgpt.query_validator import validate
from vgpt import signal_registry as R

SPLITS = ["test_iid", "test_comp", "test_template", "test_phrasing", "test_adv"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info(); return getattr(mi, "peak_wset", mi.rss) / 1e6
    except Exception: return None


class Rules:
    name = "rules"

    def __init__(self, conn=None): self.conn = conn

    def predict(self, text):
        t = time.perf_counter(); p = nl2ir.parse(text, now=DEFAULT_NOW, tz=3600)
        ir = p.query.to_text() if p.ok else "op=none"
        return ir, {"ms": (time.perf_counter() - t) * 1000, "tokens": 0, "source": "rules"}


class Slm:
    def __init__(self, model, lora=None, threads=4, mode=None, guard=False):
        from vgpt.ir_slm import IrSlm
        self.m = IrSlm(model, lora_path=lora, threads=threads, mode=mode); self.m.warmup(); self.guard = guard
        self.name = ("slm+lora" if lora else "slm-base") + ("+guard" if guard else "")

    def predict(self, text):
        q, info = self.m.compile(text)
        ir = info["raw"] if q is not None or info["raw"] == "op=none" else "op=none"
        if q is not None and self.guard and grounding_issues(q, text): ir = "op=none"
        return ir, {"ms": info["ms"], "tokens": info["out_tokens"] or 0, "source": "slm", "raw": info["raw"]}


class Hybrid:
    def __init__(self, slm): self.slm = slm; self.name = "hybrid(" + slm.name + ")"

    def predict(self, text):
        p = nl2ir.parse(text, now=DEFAULT_NOW, tz=3600)
        if p.ok: return p.query.to_text(), {"ms": p.ms, "tokens": 0, "source": "rules"}
        ir, info = self.slm.predict(text); info["ms"] += p.ms; return ir, info


def parts(q):
    d = q.to_dict(); sc = d.get("scope", {}); sb = d.get("scope_b", {})
    for x in (sc, sb or {}):                                                                  # conditions are a set: their order is not part of the meaning
        if x.get("conditions"): x["conditions"] = sorted(x["conditions"], key=lambda c: json.dumps(c, sort_keys=True))
    def conds(s): return sorted(json.dumps(c, sort_keys=True) for c in (s or {}).get("conditions", []))
    sig = sorted([q.target or "", q.target2 or ""] + [c.signal for sc_ in (q.scope, q.scope_b) if sc_ for c in sc_.conditions if c.signal])
    return {"op": q.op, "sig": sig, "time": json.dumps([sc.get("time"), (sb or {}).get("time")], sort_keys=True), "cond": json.dumps([conds(sc), conds(sb), sorted(q.groups)]), "agg": q.agg,
            "unit": sorted((c.signal, c.unit) for sc_ in (q.scope, q.scope_b) if sc_ for c in sc_.conditions if c.signal and c.unit) + ([("answer", q.unit)] if q.unit else []), "full": json.dumps(d, sort_keys=True), "sel": json.dumps(sc.get("select"))}


def same_gold(a, b):
    if not a or not b or a.get("status") != b.get("status"): return False
    va, vb = a.get("value"), b.get("value")
    ua, ub = a.get("unit", ""), b.get("unit", "")
    if ua != ub:
        if not ua or not ub or R.unit_family(ua) != R.unit_family(ub): return False
        try: va = R.convert(va, ua, ub) if isinstance(va, (int, float)) else va
        except ValueError: return False
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        if not math.isfinite(va) or not math.isfinite(vb): return False
        if abs(va - vb) > 1e-3 + 1e-3 * abs(vb): return False
    elif va != vb: return False
    if a.get("n") != b.get("n"): return False
    if a.get("n", 0):
        if not a.get("items_sha256") or not b.get("items_sha256"): return False
        if a["items_sha256"] != b["items_sha256"]: return False
    return True


def run(backend, rows, qc, progress=False):
    S = {"n": 0, "ir_exact": 0, "op": 0, "signal": 0, "time": 0, "condition": 0, "aggregation": 0, "executable": 0, "numeric": 0, "n_unit": 0, "unit": 0, "n_refuse": 0, "refusal": 0, "n_missing": 0, "missing": 0, "halluc": 0, "parsed": 0,
         "verified": 0, "answered": 0, "selection": 0}
    lat = []; toks = []; srcs = {"rules": 0, "slm": 0}; by_op = {}; errors = []; gold_cache = {}
    for i, r in enumerate(rows):
        pred, info = backend.predict(r["text"]); lat.append(info["ms"]); toks.append(info.get("tokens", 0)); srcs[info.get("source", "slm")] = srcs.get(info.get("source", "slm"), 0) + 1
        S["n"] += 1; gold_ir = r["ir"]; o = by_op.setdefault(r["op"], {"n": 0, "exact": 0}); o["n"] += 1
        if gold_ir == "op=none":
            S["n_refuse"] += 1; ok = pred == "op=none"; S["refusal"] += ok; S["ir_exact"] += ok; o["exact"] += ok; S["op"] += ok
            if not ok and len(errors) < 40: errors.append({"text": r["text"], "gold": gold_ir, "pred": pred, "split": r.get("split")})
            continue
        try: gq = Query.from_text(gold_ir); gp = parts(gq)
        except IRError: continue
        if gold_ir not in gold_cache:
            try: gold_cache[gold_ir] = qc.execute(gq)
            except Exception: gold_cache[gold_ir] = None
        current_gold_res = gold_cache[gold_ir]
        current_gold = gold_of(current_gold_res) if current_gold_res is not None else None
        try: pq = Query.from_text(pred) if pred != "op=none" else None
        except IRError: pq = None
        if pq is None:
            if len(errors) < 40: errors.append({"text": r["text"], "gold": gold_ir, "pred": pred, "split": r.get("split")})
            if (current_gold or r["gold"]).get("status") == "missing_data": S["n_missing"] += 1
            if gp["unit"]: S["n_unit"] += 1
            continue
        S["parsed"] += 1; pp = parts(pq)
        S["op"] += pp["op"] == gp["op"]; S["signal"] += pp["sig"] == gp["sig"]; S["time"] += pp["time"] == gp["time"]; S["condition"] += pp["cond"] == gp["cond"]; S["aggregation"] += pp["agg"] == gp["agg"]; S["selection"] += pp["sel"] == gp["sel"]
        ex = pp["full"] == gp["full"]; S["ir_exact"] += ex; o["exact"] += ex
        if not ex and len(errors) < 40: errors.append({"text": r["text"], "gold": gold_ir, "pred": pred, "split": r.get("split")})
        if gp["unit"]: S["n_unit"] += 1; S["unit"] += pp["unit"] == gp["unit"]
        if grounding_issues(pq, r["text"]): S["halluc"] += 1
        if validate(pq).codes(): continue
        try: res = qc.execute(pq)
        except Exception: continue
        okx = res.status in ("answered", "no_data", "missing_data", "unsupported") and (res.verification is None or res.verification.ok)
        S["executable"] += okx; S["verified"] += bool(res.verification and res.verification.ok)
        if okx and current_gold is not None: S["numeric"] += same_gold(gold_of(res), current_gold)
        S["answered"] += 1 if okx and res.status == "answered" else 0
        if (current_gold or r["gold"]).get("status") == "missing_data":
            S["n_missing"] += 1
            if res.status == "missing_data": S["missing"] += 1
        if progress and (i + 1) % 200 == 0: print(f"  {i + 1}/{len(rows)}", flush=True)
    n = max(S["n"], 1); nq = max(S["n"] - S["n_refuse"], 1)
    out = {"n": S["n"], "ir_exact": S["ir_exact"] / n, "op_acc": S["op"] / n, "signal_acc": S["signal"] / nq, "time_acc": S["time"] / nq, "condition_acc": S["condition"] / nq, "aggregation_acc": S["aggregation"] / nq, "selection_acc": S["selection"] / nq,
           "unit_acc": S["unit"] / S["n_unit"] if S["n_unit"] else None, "executable": S["executable"] / nq, "numeric_acc": S["numeric"] / nq, "refusal_acc": S["refusal"] / S["n_refuse"] if S["n_refuse"] else None,
           "missing_acc": S["missing"] / S["n_missing"] if S["n_missing"] else None, "hallucination": S["halluc"] / max(S["parsed"], 1), "verified_rate": S["verified"] / nq, "parsed_rate": S["parsed"] / nq,
           "latency_ms": {"mean": statistics.mean(lat), "p50": statistics.median(lat), "p95": sorted(lat)[int(0.95 * (len(lat) - 1))]}, "tokens_mean": statistics.mean(toks), "sources": srcs,
           "by_op": {k: round(v["exact"] / v["n"], 4) for k, v in by_op.items()}, "errors": errors[:25]}
    return out


def table(results, name):
    cols = ["ir_exact", "op_acc", "signal_acc", "time_acc", "condition_acc", "aggregation_acc", "unit_acc", "executable", "numeric_acc", "refusal_acc", "missing_acc", "hallucination", "verified_rate"]
    head = "| split | n | " + " | ".join(cols) + " | ms mean | ms p95 | tokens |\n|" + "---|" * (len(cols) + 5) + "\n"
    body = ""
    for sp, r in results.items():
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"
        body += f"| {sp} | {r['n']} | " + " | ".join(f(r[c]) for c in cols) + f" | {r['latency_ms']['mean']:.0f} | {r['latency_ms']['p95']:.0f} | {r['tokens_mean']:.1f} |\n"
    return f"### {name}\n\n" + head + body


def load_tsv(path, db):
    """A hand-written gold set: `question<TAB>IR` per line. Gold results are obtained by EXECUTING the gold IR, exactly as for the generated sets."""
    from training.build_dataset import execute_all
    from training.gen_queries import Example
    exs = []
    for line in open(path, encoding="utf-8"):
        if chr(9) not in line: continue
        text, ir = line.rstrip(chr(10)).split(chr(9)); op = ir.split()[0][3:]; exs.append(Example(text, ir, op, ["hand_written"], 0))
    out = []
    for ex, gold, ok, why in execute_all(exs, db, progress=False):
        d = ex.to_dict(); d["gold"] = gold; d["split"] = "ir_gold"
        if not ok: print("gold example does not execute:", ex.text, why)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["rules", "slm", "hybrid"], required=True); ap.add_argument("--model"); ap.add_argument("--lora"); ap.add_argument("--mode", choices=["fewshot", "tuned"]); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--cores", type=int, default=2, help="pin this process to N CPU cores (the target has 2); 0 = no pinning")
    ap.add_argument("--guard", action="store_true", help="reject model queries that fail the grounding check"); ap.add_argument("--data", default=os.path.join(HERE, "training", "data")); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db"))
    ap.add_argument("--sets", default=",".join(SPLITS)); ap.add_argument("--max", type=int, default=0, help="cap examples per split"); ap.add_argument("--out"); ap.add_argument("--label")
    a = ap.parse_args()
    if a.cores:                                                                                   # behave like the 2-core target: 2 cores, 2 inference threads
        try:
            import psutil
            psutil.Process().cpu_affinity(list(range(psutil.cpu_count()))[:a.cores])
        except Exception as e: print("could not pin CPU cores:", e)
    import sqlite3
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); qc = QueryCompiler(conn, now=DEFAULT_NOW)
    if a.backend == "rules": be = Rules()
    else:
        if not a.model: ap.error("--model is required for slm / hybrid")
        slm = Slm(a.model, a.lora, a.threads, a.mode, a.guard); be = slm if a.backend == "slm" else Hybrid(slm)
    results = {}; t0 = time.time(); db_sha = file_sha256(a.db)
    provenance = {"database": os.path.abspath(a.db), "database_sha256": db_sha, "evaluator_sha256": file_sha256(__file__)}
    report_path = os.path.join(a.data, "report.json")
    if os.path.exists(report_path):
        built = json.load(open(report_path, encoding="utf-8")); provenance["dataset_database_sha256"] = built.get("database_sha256")
        provenance["database_matches_dataset"] = bool(built.get("database_sha256")) and built.get("database_sha256") == db_sha
    for sp in a.sets.split(","):
        path = os.path.join(a.data, sp + ".jsonl")
        if not os.path.exists(path): path = sp                                                   # a file path (e.g. bench/ir_gold.jsonl)
        if not os.path.exists(path): print("missing", sp); continue
        if path.endswith(".tsv"): rows = load_tsv(path, a.db)
        else: rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if a.max: rows = rows[:a.max]
        results[os.path.splitext(os.path.basename(sp))[0]] = run(be, rows, qc); print(f"{sp}: ir_exact {100 * results[os.path.splitext(os.path.basename(sp))[0]]['ir_exact']:.1f}%", flush=True)
    mem = rss_mb(); name = a.label or be.name
    print(table(results, name)); print(f"peak/current process memory: {mem:.0f} MB; wall {time.time() - t0:.0f}s" if mem else f"wall {time.time() - t0:.0f}s")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"backend": name, "model": a.model, "lora": a.lora, "threads": a.threads, "rss_mb": mem, "provenance": provenance, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
