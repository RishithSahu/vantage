"""Does a speed method cost accuracy?  The product adapter is run over every frozen and new evaluation set (the 3,842 questions of the 2026-10-25 audit, the same car-type tags, the same scoring as bench/meaning_eval/eval_r10.py) with ONE
change to what the model is shown, and the strict-correct count is compared with the audit baseline (3,457 of 3,842).

    python bench/speed_variants.py --variant hint2 --name sv_hint2 [--procs 3] [--threads 4]

Variants (what the model is shown; the question, the car type and the previous frame are untouched unless said):
  base         as shipped (4 glossary terms, full meaning)
  hint2        at most 2 glossary terms, full meaning
  hint1        at most 1 term, full meaning
  clause4      4 terms, only the first clause of each meaning
  clause2      2 terms, first clause
  nohint       no glossary line
  hint1c       1 term, first clause
  iceonly      the shipped hint line for petrol, diesel and hybrid cars; none for an electric car
The acceptance rule written BEFORE the runs (2026-10-26): a method is rejected if it has 15 or more fewer strict-correct questions than the baseline on the 3,842; it is preferred when it has none fewer."""
import argparse, io, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
import eval_r10 as E  # noqa: E402


def clause(meaning, words=6):
    first = meaning.split(";")[0].split(",")[0].split(" - ")[0]
    return " ".join(first.split()[:words])


def patch(G, variant):
    orig = G.hint_line
    if variant == "base": return
    if variant == "short": os.environ["VGPT_HINT_SHORT"] = "1"; return                 # compact hand-written meanings (glossary_short.json)
    def make(limit, short):
        def f(text, limit_=4, vehicle=None):
            hit = [e for e in G.lookup(text) if e.get("hint")][:limit]; ice = vehicle in ("petrol", "diesel", "ice", "hybrid")
            mean = lambda e: e["meaning_ice"] if ice and e.get("meaning_ice") else e["meaning"]
            return ("Terms: " + "; ".join(f"{e['term']} = {clause(mean(e)) if short else mean(e)}" for e in hit)) if hit and limit else ""
        return f
    if variant == "iceonly":
        G.hint_line = lambda text, limit=4, vehicle=None: orig(text, limit=limit, vehicle=vehicle) if vehicle in ("petrol", "diesel", "ice", "hybrid") else ""
        return
    G.hint_line = {"hint2": make(2, False), "hint1": make(1, False), "clause4": make(4, True), "clause2": make(2, True), "nohint": make(0, False), "hint1c": make(1, True)}[variant]


def run(a):
    from vgpt.meaning import interface
    ad = a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter)
    os.environ["VGPT_VEHICLE_HINT"] = "1"; fm = interface.FrameModel(ad, threads=a.threads); patch(fm.glossary, a.variant)       # the car-type line is switched on BEFORE the model is built (it is read at construction)
    review = {}
    rp = os.path.join(HERE, "meaning_eval", "gold_review.json")
    if os.path.exists(rp):
        for e in json.load(open(rp, encoding="utf-8")):
            for key in ((e["set"], e["text"].strip().lower()), ("*", e["text"].strip().lower())): review.setdefault(key, []).extend(x for x in (E.pad(y) for y in e["accepted"]) if x not in review.get(key, []))
    out, t0 = [], time.time()
    for name in a.sets:
        rows = E.load(name)
        for r in rows:
            veh = r["tag"].split(":")[0] if r["tag"].split(":")[0] in E.VEHICLES else ("petrol" if name == "icevocab_petrol_v1" else "diesel" if name == "icevocab_diesel_v1" else "ev")
            fm.set_vehicle(veh if (a.line == "all" or (a.line == "ice" and veh != "ev")) else None)
            try: f, info = fm.parse(r["text"], prev=r["prev"])
            except Exception as ex: f, info = None, {"raw": "EXC " + type(ex).__name__, "pmin": 0.0, "ms": 0}
            raw = info["raw"]; strict = E.ok_prefix(raw, r["gold"]); acc = review.get((name, r["text"].strip().lower())) or review.get(("*", r["text"].strip().lower()))
            out.append({"set": name, "id": r["id"], "text": r["text"], "gold": r["gold"], "pred": raw, "strict": strict, "reviewed": strict or (acc is not None and E.pad(raw) in acc), "pmin": info.get("pmin"), "vehicle": veh,
                        "hint": info.get("hint", "")})
        print(f"{name} done [{time.time() - t0:.0f}s]", flush=True)
    d = os.path.join(ROOT, "results", "meaning", a.name + "_r10"); os.makedirs(d, exist_ok=True)
    json.dump({"variant": a.variant, "predictions": out}, io.open(os.path.join(d, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="base"); ap.add_argument("--name", required=True); ap.add_argument("--adapter", default="models/meaning-lora-frame2r12.gguf"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--procs", type=int, default=3); ap.add_argument("--sets", nargs="*"); ap.add_argument("--part", type=int, default=-1); ap.add_argument("--line", default="none", choices=["none", "ice", "all"], help="who is told the car type in the system prompt: nobody (how the product is configured today), petrol/diesel/hybrid only, or everybody (how the adapter was trained)"); a = ap.parse_args()
    sets = a.sets or (E.FROZEN + E.NEW)
    if a.part >= 0: a.sets = sets; return run(a)
    sizes = {s: len(E.load(s)) for s in sets}; parts = [[] for _ in range(a.procs)]; load = [0] * a.procs
    for s in sorted(sets, key=lambda s: -sizes[s]): k = load.index(min(load)); parts[k].append(s); load[k] += sizes[s]
    procs = []
    for k, ps in enumerate(parts):
        log = open(os.path.join(ROOT, f"logs_sv_{a.name}_p{k}.txt"), "w", encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, __file__, "--variant", a.variant, "--name", f"{a.name}_p{k}", "--adapter", a.adapter, "--threads", str(a.threads), "--line", a.line, "--part", str(k), "--sets", *ps], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1", VGPT_FRAME="2")))
    for p in procs: p.wait()
    merged = []
    for k in range(a.procs): merged += json.load(open(os.path.join(ROOT, "results", "meaning", f"{a.name}_p{k}_r10", "predictions.json"), encoding="utf-8"))["predictions"]
    d = os.path.join(ROOT, "results", "meaning", a.name + "_r10"); os.makedirs(d, exist_ok=True)
    json.dump({"variant": a.variant, "predictions": merged}, io.open(os.path.join(d, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    base = json.load(open(os.path.join(ROOT, "results", "meaning", "audit1025_r10", "predictions.json"), encoding="utf-8"))["predictions"]
    key = lambda p: (p["set"], p["id"], p["text"]); bm = {key(p): str(p["strict"]) == "True" for p in base}
    c = sum(1 for p in merged if str(p["strict"]) == "True"); bc = sum(bm.values()); lost = sum(1 for p in merged if bm.get(key(p)) and str(p["strict"]) != "True"); won = sum(1 for p in merged if not bm.get(key(p), True) and str(p["strict"]) == "True")
    verdict = "REJECT (15 or more fewer correct)" if c <= bc - 15 else ("PASS, no loss" if c >= bc else "PASS with a small loss (under 15) - flagged")
    print(f"VARIANT {a.variant}: strict-correct {c} of {len(merged)} (baseline {bc}); lost {lost}, won {won}, net {c - bc:+d} -> {verdict}", flush=True)


if __name__ == "__main__": main()
