"""Which glossary hints does the model actually need?  For every hinted term, the audit questions whose hint line contains it are parsed again with that ONE term left out (the others stay), and the strict score is compared with the score of the same questions with the full (compact) hint line.
    python bench/hint_ablation.py [--procs 3]      -> results/hint_ablation.json  {term: {n, base, without, lost, won, tokens}}
A term is a candidate for dropping when removing it loses nothing on the 3,842 audit questions (see docs/16_native_engine.md); the product tests decide the rest."""
import argparse, io, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["VGPT_HINT_SHORT"] = "1"
import eval_r10 as E  # noqa: E402


def rows_of(names):
    out = []
    for name in names:
        for r in E.load(name):
            veh = r["tag"].split(":")[0] if r["tag"].split(":")[0] in E.VEHICLES else ("petrol" if name == "icevocab_petrol_v1" else "diesel" if name == "icevocab_diesel_v1" else "ev")
            out.append((name, r, veh))
    return out


def run(a):
    from vgpt.meaning import interface
    os.environ["VGPT_VEHICLE_HINT"] = "1"; fm = interface.FrameModel(os.path.join(ROOT, "models/meaning-lora-frame2r12.gguf"), threads=a.threads); G = fm.glossary; orig = G.hint_line; drop = [None]
    def hint_line(text, limit=4, vehicle=None):
        hit = [e for e in G.lookup(text) if e.get("hint")][:limit]
        if drop[0]: hit = [e for e in hit if e["term"] != drop[0]]
        saved = G.lookup; G.lookup = lambda t, hit=hit: hit; G.index  # noqa
        try: return orig(text, limit=limit, vehicle=vehicle)
        finally: G.lookup = saved
    G.hint_line = hint_line
    res = {}; sets = a.sets
    for name, r, veh in rows_of(sets):
        terms = [e["term"] for e in G.lookup(r["text"]) if e.get("hint")][:4]
        if not terms: continue
        fm.set_vehicle(veh)
        for t in [None] + terms:
            drop[0] = t
            try: f, info = fm.parse(r["text"], prev=r["prev"]); raw = info["raw"]
            except Exception: raw = "EXC"
            if t is None: base = E.ok_prefix(raw, r["gold"]); continue
            ok = E.ok_prefix(raw, r["gold"]); x = res.setdefault(t, {"n": 0, "base": 0, "without": 0, "lost": 0, "won": 0}); x["n"] += 1; x["base"] += base; x["without"] += ok; x["lost"] += (base and not ok); x["won"] += (ok and not base)
    json.dump(res, io.open(os.path.join(ROOT, "results", "hint_ablation_%d.json" % a.part), "w", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=3); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--part", type=int, default=-1); ap.add_argument("--sets", nargs="*"); a = ap.parse_args()
    sets = E.FROZEN + E.NEW
    if a.part >= 0: return run(a)
    parts = [sets[k::a.procs] for k in range(a.procs)]; ps = []
    for k, p in enumerate(parts): ps.append(subprocess.Popen([sys.executable, __file__, "--part", str(k), "--threads", str(a.threads), "--procs", str(a.procs), "--sets", *p], cwd=ROOT, stdout=open(os.path.join(ROOT, "logs_abl_%d.txt" % k), "w"), stderr=subprocess.STDOUT))
    for p in ps: p.wait()
    tot = {}
    for k in range(a.procs):
        for t, x in json.load(open(os.path.join(ROOT, "results", "hint_ablation_%d.json" % k))).items():
            y = tot.setdefault(t, dict.fromkeys(x, 0))
            for q in x: y[q] += x[q]
    json.dump(tot, io.open(os.path.join(ROOT, "results", "hint_ablation.json"), "w", encoding="utf-8"), indent=1)
    for t, x in sorted(tot.items(), key=lambda kv: -kv[1]["n"]): print("%-24s n=%-4d base %-4d without %-4d lost %-3d won %d" % (t, x["n"], x["base"], x["without"], x["lost"], x["won"]))


if __name__ == "__main__": main()
