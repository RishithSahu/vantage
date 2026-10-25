"""Large oracle-checked question set for the VANTAGE semantic campaign.
Every question is GENERATED from a typed intent (family x concept x window x restriction x surface form); the expected answer is not written by hand but computed by executing the intended typed query on the same database at the same instant (the oracle). The assistant's answer is accepted only if it (a) is not a clarification / refusal for an answerable question, (b) contains the oracle's value (within the printed precision), (c) states the oracle's unit, and (d) for windows names the requested window. Surface forms vary: synonyms from the ontology lexicon, word order, contractions, case, a typo.
    python bench/vantage_stress.py [--n 900] [--seed 5] [--now 1787625900] [--out results/vantage_stress.json]"""
import argparse, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import signal_registry as R
from vgpt.query_ir import Query, TimeSpec, Condition, EventRef

SIG = {   # concept -> (phrases, unit as printed)
    "speed": (["speed", "vehicle speed"], "km/h"), "battery_temperature": (["battery temperature", "battery temp", "pack temperature"], "°C"),
    "motor_temperature": (["motor temperature", "motor temp"], "°C"), "coolant_flow": (["coolant flow", "coolant flow rate"], "L/min"), "ambient_temperature": (["outside temperature", "ambient temperature"], "°C"),
    "battery_soc": (["battery percentage", "state of charge", "battery level"], "%"), "motor_rpm": (["motor rpm", "motor speed in rpm"], "rpm"), "hvac_power": (["climate control power", "hvac power"], "W"),
}
PER = [("yesterday", TimeSpec("calendar", name="yesterday"), ["yesterday"]), ("over the last 3 days", TimeSpec("relative", 3, "day"), ["3 days"]), ("in the last 5 days", TimeSpec("relative", 5, "day"), ["5 days"]),
       ("over the last 10 days", TimeSpec("relative", 10, "day"), ["10 days"]), ("over the last 14 days", TimeSpec("relative", 14, "day"), ["14 days"]), ("last week", TimeSpec("relative", 7, "day"), ["7 days", "week"]),
       ("over the last nine days", TimeSpec("relative", 9, "day"), ["9 days", "nine days"]), ("in the last 24 hours", TimeSpec("relative", 24, "hour"), ["24 hours", "day"])]
AGG = [("average", "mean", ["average", "mean", "typical"]), ("highest", "max", ["highest", "peak", "maximum", "max"]), ("lowest", "min", ["lowest", "minimum", "min"])]
EVT = {"HARSH_BRAKE": ["hard brakes", "harsh braking events", "hard braking events"], "HARSH_ACCEL": ["hard accelerations", "harsh acceleration events"], "ADAS_DISENGAGE": ["ADAS disengagements", "lane assist disengagements"]}
COND = {"highway": ["on the highway", "while driving on the motorway"], "city": ["in the city", "while driving in town"], "charging": ["while charging", "when plugged in"]}
NOISE = [lambda s: s, lambda s: s.lower(), lambda s: s.rstrip("?") , lambda s: s.replace("What", "Whats", 1) if s.startswith("What ") else s, lambda s: s.replace("average", "avg") if "average" in s else s]


def typo(rng, s):
    w = s.split(); idx = [i for i, x in enumerate(w) if len(x) > 5 and x.isalpha()]
    if not idx: return s
    i = rng.choice(idx); x = w[i]; j = rng.randrange(1, len(x) - 2); w[i] = x[:j] + x[j + 1] + x[j] + x[j + 2:]; return " ".join(w)


def gen(rng, n):
    out = []
    def add(fam, q, query, kind, unit=None, window=None, tag=""): out.append({"family": fam, "q": q, "query": query.to_text() if query is not None else None, "kind": kind, "unit": unit, "window": window})
    while len(out) < n:
        fam = rng.choice(["current", "aggregate", "aggregate", "aggregate", "count", "cond_agg", "extreme_time", "compare_periods", "group_compare", "energy_math", "relation"])
        c = rng.choice(list(SIG)); ph, unit = SIG[c]; p = rng.choice(ph); per = rng.choice(PER); agg = rng.choice(AGG)
        if fam == "current":
            tpl = rng.choice(["What is the {p} right now?", "what's my {p} at the moment", "How is the {p} now?", "Tell me the current {p}", "{p} right now?"])
            add(fam, tpl.format(p=p), Query(op="current_value", target=c), "value", unit)
        elif fam == "aggregate":
            if c in ("battery_soc",) and agg[1] != "mean": continue
            tpl = rng.choice(["What was my {a} {p} {t}?", "{a} {p} {t}", "Tell me the {a} {p} {t}", "what is the {a} {p} {t}"]); a = rng.choice(agg[2]); w = per[0]
            q = Query(op="extreme" if agg[1] in ("max", "min") else "aggregate", target=c, agg=agg[1], scope=Query().scope.__class__(time=per[1]))
            add(fam, tpl.format(a=a, p=p, t=w), q, "value", unit, per[2])
        elif fam == "count":
            e = rng.choice(list(EVT)); ph2 = rng.choice(EVT[e]); tpl = rng.choice(["How many {e} did I have {t}?", "number of {e} {t}", "count the {e} {t}", "how many {e} {t}"])
            add(fam, tpl.format(e=ph2, t=per[0]), Query(op="count", event=EventRef(e, "all"), scope=Query().scope.__class__(time=per[1])), "count", None, per[2])
        elif fam == "cond_agg":
            if c != "speed" and c != "battery_temperature": continue
            k = rng.choice(list(COND)); phr = rng.choice(COND[k]); a = rng.choice(agg[2])
            q = Query(op="extreme" if agg[1] in ("max", "min") else "aggregate", target=c, agg=agg[1], scope=Query().scope.__class__(time=per[1], conditions=[Condition(predicate=k)]))
            add(fam, rng.choice(["What was my {a} {p} {c} {t}?", "{a} {p} {c} {t}"]).format(a=a, p=p, c=phr, t=per[0]), q, "value", unit, per[2])
        elif fam == "extreme_time":
            if c not in ("battery_temperature", "motor_temperature", "speed"): continue
            add(fam, rng.choice(["When was my {p} highest {t}?", "when did the {p} peak {t}"]).format(p=p, t=per[0]), Query(op="extreme", target=c, agg="max", scope=Query().scope.__class__(time=per[1])), "extreme_time", unit, per[2])
        elif fam == "group_compare":
            m = rng.choice([("speed", ["average speed", "speed"], "km/h"), ("wh_per_km", ["energy consumption", "efficiency"], "Wh/km")]); d = rng.choice([7, 14])
            add(fam, rng.choice(["Compare city and highway {m} over the last {d} days", "{m} in the city versus the highway over the last {d} days", "how does {m} differ between city and highway driving over the last {d} days"]).format(m=rng.choice(m[1]), d=d),
                Query(op="compare", target=m[0], agg="mean", groups=["city", "highway"], scope=Query().scope.__class__(time=TimeSpec("relative", d, "day"))), "compare_groups", m[2], [f"{d} days"])
        elif fam == "energy_math":
            km = rng.choice([50, 120, 250, 400]); kwh = rng.choice([5, 10, 20, 30])
            if rng.random() < .5: add(fam, rng.choice(["At my recent efficiency, how much energy would a {k} km trip require?", "how much energy for {k} km at my current consumption", "energy needed to drive {k} km at my recent efficiency"]).format(k=km), None, f"math_energy:{km}", "kWh")
            else: add(fam, rng.choice(["How far could I travel using {e} kWh?", "how many km can i go on {e} kwh", "distance possible with {e} kWh at my recent efficiency"]).format(e=kwh), None, f"math_dist:{kwh}", "km")
        elif fam == "relation":
            pairs = [("What is the difference between battery state of charge and state of health?", ["state of charge", "state of health"]), ("does a full battery mean it is healthy", ["charge", "health"]),
                     ("what is the difference between motor temperature and battery temperature", ["motor", "battery"]), ("Does low coolant flow prove the pack is overheating?", ["not", "coolant"]),
                     ("is charging power the same as battery energy", ["kw", "kwh"]), ("does a normal tyre pressure guarantee the tyres are safe", ["tread", "not"])]
            q, kws = rng.choice(pairs); add(fam, q, None, "relation:" + "|".join(kws), None)
    return out


def check(item, ans, oracle, now):
    a = ans or ""; low = a.lower(); flags = []
    if re.search(r"i want to be sure|do you mean|reply with a number|could not turn that|i'm not sure i understood|doesn't look like a question", low): flags.append("clarification_or_refusal")
    kind = item["kind"]
    if kind.startswith("relation:"):
        for kw in kind.split(":", 1)[1].split("|"):
            if kw not in low: flags.append("missing:" + kw)
        return flags
    nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", a)]
    if kind in ("value", "extreme_time"):
        v = oracle.get("value")
        if v is None: flags.append("oracle_empty")
        elif not any(abs(x - v) <= max(0.55, abs(v) * 0.012) for x in nums): flags.append(f"value_mismatch(expected {v:.2f})")
    elif kind == "count":
        v = oracle.get("value")
        if v is not None and not any(abs(x - v) < 0.5 for x in nums) and not (v == 0 and re.search(r"\bno\b|none|zero", low)): flags.append(f"count_mismatch(expected {v:g})")
    elif kind == "compare_groups":
        v = oracle.get("groups") or {}
        for g, x in v.items():
            if not any(abs(n - x) <= max(0.55, abs(x) * 0.012) for n in nums): flags.append(f"group_{g}_mismatch(expected {x:.1f})")
    elif kind.startswith("math_"):
        arg = float(kind.split(":")[1]); wh = oracle.get("whkm"); exp = arg * wh / 1000.0 if kind.startswith("math_energy") else arg * 1000.0 / wh
        if not any(abs(n - exp) <= max(0.15, exp * 0.012) for n in nums): flags.append(f"math_mismatch(expected {exp:.1f})")
    if item.get("unit") and item["unit"].lower().replace("°", "") not in low.replace("°", "").replace("�", "") and not kind.startswith("math"): flags.append("unit_missing:" + item["unit"])
    if item.get("window") and kind != "extreme_time" and not any(w.lower() in low.replace("-day", " days") for w in item["window"]) and item["family"] not in ("current", "energy_math"): flags.append("window_not_stated")
    return flags


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=900); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--now", type=int, default=1787625900); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vantage_stress.json")); ap.add_argument("--noise", type=float, default=0.35); a = ap.parse_args(); rng = random.Random(a.seed)
    from vgpt.product import make_legacy_brain
    from vgpt.brain import LazySlm
    slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words")
    items = gen(rng, a.n); b = make_legacy_brain(a.db, slm=slm, now=a.now); qc = b.qc(); res = []; t0 = time.time()
    from vgpt import analytics as A
    er = A.est_range(b.c, a.now); whkm = float(er["wh_km"]) if er.get("ok") else None
    for i, it in enumerate(items):
        q = it["q"]; r = rng.random()
        if r < a.noise: q = rng.choice(NOISE)(q)
        if r > 1 - 0.12: q = typo(rng, q)
        it["asked"] = q
        oracle = {}
        if it["query"]:
            Qy = Query.from_text(it["query"]) if it["kind"] != "compare_groups" else None
            try:
                x = qc.execute(Query.from_text(it["query"]))
                if x.status == "answered" and x.result is not None:
                    if it["kind"] == "compare_groups":
                        ex = x.result.extra; oracle["groups"] = {k: float(ex[k]) for k in ("a", "b") if isinstance(ex.get(k), (int, float))}
                    oracle["value"] = float(x.result.value) if isinstance(x.result.value, (int, float)) else None
                    if it["kind"] == "extreme_time": oracle["when"] = x.result.when
            except Exception as e: oracle["error"] = type(e).__name__
        else: oracle["whkm"] = whkm
        b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None
        try: ans = b.ask(q).get("answer") or ""
        except Exception as e: ans = f"EXCEPTION {type(e).__name__}"
        fl = check(it, ans, oracle, a.now); it.update(answer=ans[:400], oracle={k: v for k, v in oracle.items() if k != "groups"}, flags=fl, status="PASS" if not fl else "FAIL"); res.append(it)
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(items)} pass so far {sum(r['status'] == 'PASS' for r in res)}", flush=True)
    p = sum(r["status"] == "PASS" for r in res); print(f"{p}/{len(res)} pass in {time.time() - t0:.0f}s")
    import collections
    fam = collections.Counter((r["family"], r["status"]) for r in res)
    for f in sorted({r["family"] for r in res}): print(f"  {f:16s} {fam[(f, 'PASS')]}/{fam[(f, 'PASS')] + fam[(f, 'FAIL')]}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
