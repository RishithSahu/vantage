"""DRY RUN of the complete /review route against a running server (python ui/server.py ...): the five case-study questions (verbatim), unfamiliar phrasings, an edited proposal, a follow-up, a missing sensor, a refusal, and a vehicle switch (changed physical mapping). For every step: the wait for the model's proposal, the execution time, the proposed operation and fields, and the answer's first line; first (cold) versus later (warm) calls are kept apart. It finds broken controls / endpoints and measures the wait that participants will actually experience for THE CONFIGURATION THE PRODUCT USES (whatever planner the server was started with). Laptop timings, not ARM.
    python ui/server.py --model none --port 8766 --db data/gen_courier_ext.db      (optionally pinned: python tools_pin.py 8,9 ui/server.py ...)
    python bench/dry_run_review.py --port 8766 [--out results/dry_run_review.json]"""
import argparse, http.client, json, time

STEPS = [
    ("case-study 1", "Why has my driving range reduced over the last two weeks?", None),
    ("case-study 2", "When did fuel efficiency show the largest drop?", None),
    ("case-study 3", "Show all ADAS disengagement events from yesterday.", None),
    ("case-study 4", "What happened before the battery overheating warning?", None),
    ("case-study 5", "Compare energy consumption between city and highway driving.", None),
    ("unfamiliar wording", "how toasty did the pack get at its worst yesterday", None),
    ("unfamiliar wording", "any times the lane assist gave up this week", None),
    ("edit", "average speed over the last week", {"edit": {"period": "last 3 days"}}),
    ("follow-up", "and only on the highway", {"follow_up": True}),
    ("missing sensor", "what is the engine oil pressure", None),
    ("refusal", "turn the heated seats on", None),
    ("vehicle switch", "average battery temperature yesterday", {"switch_dataset": True}),
]


def call(port, method, path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout); t = time.time()
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"} if body is not None else {}); r = c.getresponse(); d = r.read(); c.close(); return r.status, json.loads(d or b"{}"), time.time() - t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8766); ap.add_argument("--out"); a = ap.parse_args(); out = []
    st, ds, _ = call(a.port, "GET", "/api/datasets"); names = [d["name"] for d in ds["datasets"]]; print("datasets:", names[:6], "current", ds.get("current"))
    st, cat, t = call(a.port, "GET", "/api/catalogue"); print(f"catalogue {st}: {len(cat.get('signals', []))} signals, {len(cat.get('functions', []))} operations, {t:.1f}s (first call also loads the model)")
    for k, (cat_, q, opt) in enumerate(STEPS):
        opt = opt or {}
        if opt.get("switch_dataset"):
            other = [n for n in names if n != ds.get("current") and n != "gen_courier_ext.db"][:1] or names[:1]; st, r, t = call(a.port, "POST", "/api/dataset", {"name": other[0]}); print(f"   switched vehicle to {other[0]}: {r.get('ok', r)} in {t:.1f}s")
        st, p, t_prop = call(a.port, "POST", "/api/interpret", {"q": q, "participant": "dryrun", "task": cat_, "follow_up": bool(opt.get("follow_up"))}); fn = p.get("fn"); args = dict(p.get("args") or {})
        final_args = dict(args)
        if opt.get("edit"): final_args.update(opt["edit"])
        t_exec = 0.0; ans = ""; edits = []
        if fn and fn not in ("ask_user", "unresolved", "decline", "say"):
            st, r, t_exec = call(a.port, "POST", "/api/execute", {"q": q, "participant": "dryrun", "task": cat_, "proposal": {"fn": fn, "args": args, "raw": p.get("raw"), "timing": p.get("timing")}, "final": {"fn": fn, "args": final_args}}); ans = (r.get("answer") or r.get("error") or "").split("\n")[0]; edits = r.get("edits", [])
        else: ans = f"(proposal: {fn}; nothing to run) " + json.dumps(args)[:100]
        rec = {"category": cat_, "q": q, "ok": p.get("ok"), "fn": fn, "args": args, "propose_s": round(t_prop, 1), "model_ms": (p.get("timing") or {}).get("plan_ms"), "prompt_tokens": (p.get("timing") or {}).get("prompt_tokens"), "retrieve_ms": (p.get("timing") or {}).get("retrieve_ms"), "execute_s": round(t_exec, 1), "answer": ans[:200], "edits": edits, "cold": k == 0}
        out.append(rec); print(f"{cat_:20s} wait {t_prop:5.1f}s (model {rec['model_ms']} ms, {rec['prompt_tokens']} uncached tokens) | run {t_exec:4.1f}s | {fn} {json.dumps(args)[:110]}\n{'':20s} -> {ans[:150]}", flush=True)
    warm = [r["propose_s"] for r in out[1:]]; print(f"\nproposal wait: first {out[0]['propose_s']}s (cold), later: median {sorted(warm)[len(warm) // 2]}s, max {max(warm)}s, mean {sum(warm) / len(warm):.1f}s; execution median {sorted(r['execute_s'] for r in out)[len(out) // 2]}s")
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
