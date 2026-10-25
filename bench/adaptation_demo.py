"""Adaptation without retraining and without parser changes: the SAME frozen model, the SAME prompts and the SAME code answer questions about vehicles whose schema, signals, units or history differ - only metadata (a schema overlay) and data change.
For each fixture (data/fixtures/*, built by bench/schema_fixtures.py and bench/fuel_vehicle.py) the overlay is applied, the capability registry and retrieval index are rebuilt, and a few questions are asked. For every question the report shows the capabilities retrieved, the plan the model wrote, the answer, and the "answer receipt" (how the question was read, the interval, the coverage).
    python tools_pin.py 8,9 bench/adaptation_demo.py --out results/adaptation_demo.md
Nothing here is scored: it is a demonstration, and a failure is shown as a failure."""
import argparse, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from vgpt import schema_overlay
from vgpt.modelbrain import ModelBrain
from vgpt.modelplan import Planner

DEMO = [
    ("gen_courier_ext (the reference vehicle)", "data/gen_courier_ext.db", None, ["has the camera been blocked in the last seven days", "what is the lowest tyre pressure in the last 3 days", "when is my next service due"]),
    ("fl_missing: the front-left tyre sensor is absent", "data/fixtures/fl_missing.db", "fl_missing", ["what is the front left tyre pressure right now", "what was the lowest tyre pressure last week"]),
    ("units: speed stored in mph, battery temperature in Fahrenheit", "data/fixtures/units.db", "units", ["highest speed in the last 7 days", "what was the average battery temperature yesterday"]),
    ("opaque identifier CH_0417 (described only by its label and synonyms)", "data/fixtures/opaque_signal.db", "opaque_signal", ["what was the highest inverter coolant outlet temperature this week", "do you record the power electronics coolant temperature"]),
    ("new_signal: a washer-fluid level the built-in registry does not know", "data/fixtures/new_signal.db", "new_signal", ["what is the lowest screen wash level since the start", "when did the washer fluid level change the most"]),
    ("intermittent: battery temperature missing on 40% of days (partial history)", "data/fixtures/intermittent.db", "intermittent", ["what was the average pack temperature over the last 14 days"]),
    ("SIMULATED fuel-capable vehicle (bench/fuel_vehicle.py)", "data/fixtures/sim_fuel_demo.db", "sim_fuel_demo", ["When did fuel efficiency show the largest drop?", "what is my fuel economy this week", "how much fuel is left"]),
    ("the reference electric vehicle asked the PDF's fuel question", "data/gen_courier_ext.db", None, ["When did fuel efficiency show the largest drop?"]),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")); ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--out", default=os.path.join(HERE, "results", "adaptation_demo.md")); a = ap.parse_args()
    out = ["# Adaptation demonstration (model-only route, `%s`)\n" % os.path.basename(a.model),
           "One frozen model, the same prompt, the same code. Only the schema overlay and the data change between sections: the capability registry, the retrieval index and the grammar are rebuilt from the new metadata; the model prefix is reused. Timings are laptop timings on 2 pinned cores.\n",
           "For each question the report shows the capability cards the model was given (descriptions and availability, as in the rendered prompt), the call it wrote, the typed query it became and the answer.\n"]; P = None
    for title, db, ov, qs in DEMO:
        path = os.path.join(HERE, db); spec = json.load(open(os.path.join(HERE, db[:-3] + ".overlay.json"))) if (ov and os.path.exists(os.path.join(HERE, db[:-3] + ".overlay.json"))) else {}
        out.append(f"\n## {title}\n"); token = schema_overlay.apply(spec) if spec else None
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            if P is None: P = Planner(conn, model_path=a.model, n_ctx=a.n_ctx)
            else: P.reload_schema(conn)
            b = ModelBrain(path, planner=P)
            for q in qs:
                b.reset(); t = time.perf_counter(); r = b.ask(q); ms = (time.perf_counter() - t) * 1000; pr = b.last_plan_result
                out.append(f"**{q}**  ({ms:.0f} ms, outcome `{b.outcome}`)\n")
                if pr:
                    out.append("* cards shown to the model:\n\n```\n" + "\n".join(l for l in pr.message.splitlines() if "|" in l) + "\n```")
                    out.append(f"* call: `{pr.raw}` -> `{(b.last_qc.query.to_text() if b.last_qc is not None else r.get('intent'))}`")
                out.append("\n```\n" + (r.get("answer") or "").strip() + "\n```\n")
        finally:
            if token is not None: schema_overlay.restore(token)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w", encoding="utf-8").write("\n".join(out)); print("written", a.out)


if __name__ == "__main__": main()
