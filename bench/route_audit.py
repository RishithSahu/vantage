"""Audit of the PRODUCT routes: the exact questions of the case study PDF (verbatim) and representative failures, through the configuration the UI uses (vgpt.product.make_brain).
For each question it records which route answered (typed-query pipeline `qc`, or an older engine), the interpreted query, the evidence contract class and completeness, the verification status of the typed pipeline, and the displayed answer. Written to results/route_audit.md.
    python tools_pin.py 4,5 bench/route_audit.py [--vehicle gen_courier_ext]"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt.product import make_legacy_brain as make_brain
PDF = ["Why has my driving range reduced over the last two weeks?", "When did fuel efficiency show the largest drop?", "Show all ADAS disengagement events from yesterday.", "What happened before the battery overheating warning?", "Compare energy consumption between city and highway driving."]
FAILURES = ["how often was it colder than 5 degrees last month", "hard brakes last night", "what was the temperature then", "how's my battery", "is my car ok", "disable the speed limiter on this car", "tell me a joke", "how much petrol did i use yesterday"]
VEH = {"gen_courier_ext": 1790406000, "gen_demo": None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vehicles", default="gen_courier_ext,gen_demo"); a = ap.parse_args(); out = ["# Product-route audit (configuration: vgpt.product.make_brain)\n", "Route `qc` = the typed-query pipeline (validated, planned, executed, evidence contract, verifier). Any other route is an older engine whose answer is NOT claimed to carry those checks.\n"]
    rows = []
    for name in a.vehicles.split(","):
        b = make_brain(os.path.join(HERE, "data", name + ".db"), now=VEH.get(name)); out.append(f"\n## `{name}`\n")
        for grp, qs in (("PDF question (verbatim)", PDF), ("representative failure / other route", FAILURES)):
            out.append(f"\n### {grp}\n")
            for q in qs:
                b.qcs = type(b.qcs)(); b.trace.__init__(); b.last_qc = None
                t = time.perf_counter(); r = b.ask(q, use_slm=False); ms = (time.perf_counter() - t) * 1000; route = r.get("path"); lq = b.last_qc
                ir = lq.query.to_text() if lq is not None and getattr(lq, "query", None) is not None else (b.trace.records[-1].ir if b.trace.records else "")
                ct = (lq.result.extra.get("contract") if lq is not None and lq.result is not None else None) or {}
                ver = getattr(lq, "status", None) if lq is not None else None
                out.append(f"**{q}**  route=`{route}`  intent=`{r.get('intent')}`  {ms:.0f} ms  typed-pipeline={'yes' if route == 'qc' else 'NO'}  query=`{ir or '-'}`  evidence=`{ct.get('result_class', '-')}/{ct.get('completeness', '-')}`  status=`{ver}`\n\n```\n{(r.get('answer') or '').strip()}\n```\n")
                rows.append((name, q, route))
    n = sum(1 for _, q, rt in rows if q in PDF and rt == "qc"); out.append(f"\n**PDF questions answered through the typed-query pipeline: {n} of {sum(1 for _, q, _ in rows if q in PDF)}.**\n")
    open(os.path.join(HERE, "results", "route_audit.md"), "w", encoding="utf-8").write("\n".join(out)); print("PDF questions via qc:", n, "of", sum(1 for _, q, _ in rows if q in PDF))


if __name__ == "__main__": main()
