"""Third generator for the VANTAGE campaign: unusual, noisy and indirect phrasings of questions whose answer is again computed by the oracle (the intended typed query run on the same database).  Families: rambling preambles, indirect/polite forms, shouting / emoji / slang, calendar dates and weekday names, durations above a threshold, unavailable quantities (must be reported as not recorded, never substituted), longer multi-turn chains, and mixed-language openings.
    python bench/vantage_stress3.py [--n 500] [--seed 9] [--now 1787625900]"""
import argparse, collections, datetime as dt, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.query_ir import Query, TimeSpec, Condition, EventRef

SC = Query().scope.__class__
SIG = {"speed": (["speed", "vehicle speed"], "km/h"), "battery_temperature": (["battery temperature", "pack temperature"], "°C"), "motor_temperature": (["motor temperature"], "°C"), "coolant_flow": (["coolant flow"], "L/min"), "ambient_temperature": (["outside temperature"], "°C")}
PRE = ["so I was thinking about the car earlier and", "okay quick one,", "hey assistant,", "sorry to bother you but", "right so", "before I forget,", "this is probably a silly question but", "hi, my friend asked me and I couldn't answer:", "um,", "just curious,"]
POLITE = ["Could you please tell me {q}", "I would like to know {q}", "Can you check {q} for me", "Would you mind telling me {q}", "Please show {q}", "I was wondering {q}"]
SHOUT = [lambda s: s.upper(), lambda s: s + " !!!", lambda s: s + " 🙏", lambda s: "pls " + s.lower(), lambda s: s.lower().replace("what is", "wat is").replace("the ", "da ", 1)]
UNAVAIL = [("what is the fuel level", ["fuel", "not", "electric|record"]), ("what is the tyre tread depth", ["tread", "not|no ", "record|measure"]), ("what is the engine oil temperature", ["oil", "not|no ", "record|measure|know"]),
           ("how many passengers are in the car", ["passenger|seat|occup", "not|cannot|can't|doesn't"]), ("what is the cabin humidity", ["humid", "not|no ", "record|measure|know"])]
MONTH = {8: "August"}


def nums_of(a): return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", a)]
def near(nums, v, rel=0.012, ab=0.55): return any(abs(n - v) <= max(ab, abs(v) * rel) for n in nums)
CLAR = re.compile(r"i want to be sure|do you mean|reply with a number|could not turn that|i'm not sure i understood|doesn't look like a question", re.I)


def value_of(qc, q):
    x = qc.execute(q if isinstance(q, Query) else Query.from_text(q)); return float(x.result.value) if x.status == "answered" and x.result is not None and isinstance(x.result.value, (int, float)) else None


def base_question(rng):
    c = rng.choice(list(SIG)); p = rng.choice(SIG[c][0]); unit = SIG[c][1]; kind = rng.choice(["current", "agg", "agg", "ext"])
    if kind == "current": return f"the {p} right now", Query(op="current_value", target=c), unit, None
    a = rng.choice([("average", "mean"), ("highest", "max"), ("lowest", "min")]); per = rng.choice([("yesterday", TimeSpec("calendar", name="yesterday"), ["yesterday"]), ("over the last 3 days", TimeSpec("relative", 3, "day"), ["3 days"]), ("over the last 10 days", TimeSpec("relative", 10, "day"), ["10 days"])])
    return f"the {a[0]} {p} {per[0]}", Query(op="extreme" if a[1] != "mean" else "aggregate", target=c, agg=a[1], scope=SC(time=per[1])), unit, per[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=9); ap.add_argument("--now", type=int, default=1787625900); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_demo.db"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vantage_stress3.json")); a = ap.parse_args(); rng = random.Random(a.seed)
    from vgpt.product import make_legacy_brain
    from vgpt.brain import LazySlm
    slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words")
    b = make_legacy_brain(a.db, slm=slm, now=a.now); qc = b.qc(); res = []; t0 = time.time()
    today = dt.datetime.fromtimestamp(a.now + 19800, dt.timezone.utc).date()
    def fresh(): b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
    for i in range(a.n):
        fam = rng.choice(["preamble", "polite", "shout", "date", "weekday", "duration", "unavailable", "chain3", "mixed"]); fl = []; ans = ""; asked = ""
        try:
            fresh()
            if fam in ("preamble", "polite", "shout", "mixed"):
                txt, q, unit, win = base_question(rng)
                if fam == "preamble": asked = f"{rng.choice(PRE)} what is {txt}?"
                elif fam == "polite": asked = rng.choice(POLITE).format(q=txt) + rng.choice(["", "?", " thanks"])
                elif fam == "shout": asked = rng.choice(SHOUT)(f"What is {txt}?")
                else: asked = rng.choice(["bhai ", "yaar ", "arre ", "boss "]) + f"what is {txt}"
                ans = b.ask(asked).get("answer") or ""; v = value_of(qc, q); low = ans.lower()
                if CLAR.search(ans): fl.append("clarification")
                elif v is None or not near(nums_of(ans), v): fl.append(f"value_mismatch(expected {v})")
                if unit and not CLAR.search(ans) and unit.lower().replace("°", "") not in low.replace("°", "").replace("�", ""): fl.append("unit_missing")
                if win and not CLAR.search(ans) and not any(w in low.replace("-day", " days") for w in win): fl.append("window_not_stated")
            elif fam in ("date", "weekday"):
                c = rng.choice(["speed", "battery_temperature", "motor_temperature"]); nm = SIG[c][0][0]; back = rng.choice([2, 3, 4, 5, 6])
                d = today - dt.timedelta(days=back)
                if fam == "date": phr = f"on {d.day} {d.strftime('%B')}"
                else: phr = f"on {d.strftime('%A')}"
                asked = f"what was the average {nm} {phr}?"; ans = b.ask(asked).get("answer") or ""
                q = Query(op="aggregate", target=c, agg="mean", scope=SC(time=TimeSpec("date", date=d.isoformat()))); v = value_of(qc, q)
                if CLAR.search(ans): fl.append("clarification")
                elif v is None: fl.append("oracle_empty/no data that day")
                elif not near(nums_of(ans), v): fl.append(f"value_mismatch(expected {v})")
            elif fam == "duration":
                th = rng.choice([30, 50, 70]); asked = rng.choice(["for how many minutes was my speed above {t} km/h in the last 3 days", "how long was I going faster than {t} km/h over the last 3 days"]).format(t=th)
                ans = b.ask(asked).get("answer") or ""
                n_ = b.c.execute("SELECT COUNT(*) FROM sig_1m WHERE ts>=? AND ts<=? AND speed>?", (a.now - 3 * 86400, a.now, th * 10)).fetchone()[0]
                nums = nums_of(ans); hours = n_ / 60.0
                if CLAR.search(ans): fl.append("clarification")
                elif not (near(nums, n_, 0.05, 2.5) or near(nums, hours, 0.05, 0.06) or re.search(r"\d+ ?h(?:ours?)? ?\d* ?m", ans)): fl.append(f"duration_mismatch(expected {n_} min = {hours:.2f} h)")
            elif fam == "unavailable":
                q, kws = rng.choice(UNAVAIL); asked = rng.choice([q, q + " right now", "tell me " + q.replace("what is", "").strip()]); ans = b.ask(asked).get("answer") or ""; low = ans.lower()
                if CLAR.search(ans) and "not" not in low: fl.append("clarification")
                for kw in kws:
                    if not re.search(kw, low): fl.append("missing:" + kw)
                if re.search(r"\b\d+(\.\d+)? ?(%|°c|l/min|km/h|kpa|mm)\b", low) and not re.search(r"not|no |cannot|can't", low): fl.append("fabricated_value")
            elif fam == "chain3":
                c = rng.choice(["motor_temperature", "battery_temperature"]); other = "battery_temperature" if c == "motor_temperature" else "motor_temperature"; nm = {"motor_temperature": "motor", "battery_temperature": "battery"}
                turns = [f"What is the {nm[c]} temperature?", f"what about the {nm[other]}", "is that normal"]; answers = [b.ask(t).get("answer") or "" for t in turns]; ans = " || ".join(x[:90] for x in answers); asked = " > ".join(turns)
                if CLAR.search(answers[1]) or not near(nums_of(answers[1]), value_of(qc, Query(op="current_value", target=other)), 0.015, 0.06): fl.append("turn2_wrong")
                if not re.search(r"usual|normal|typical|range", answers[2], re.I): fl.append("turn3_no_assessment")
        except Exception as e: fl.append("EXCEPTION " + type(e).__name__); ans = ans or str(e)
        res.append({"family": fam, "asked": asked, "answer": ans[:400], "flags": fl, "status": "PASS" if not fl else "FAIL"})
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{a.n} pass so far {sum(r['status'] == 'PASS' for r in res)}", flush=True)
    p = sum(r["status"] == "PASS" for r in res); print(f"{p}/{len(res)} pass in {time.time() - t0:.0f}s")
    fam = collections.Counter((r["family"], r["status"]) for r in res)
    for f in sorted({r["family"] for r in res}): print(f"  {f:12s} {fam[(f, 'PASS')]}/{fam[(f, 'PASS')] + fam[(f, 'FAIL')]}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
