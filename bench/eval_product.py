"""Scores the PRODUCT configuration END TO END: every question is asked through vgpt.product.make_brain (the configuration the UI and app.py use), so the command / fuel / status gates, the interpretation policy, the typed pipeline AND the routing to older engines are all in the path. Scoring is the same evaluator as bench/eval_v2.py (multi-database signatures, clarification scoring), so the numbers are directly comparable with the `policyrules` rows of the baseline.
    python tools_pin.py 8,9 bench/eval_product.py --set dev --out results/f1_product_dev.json
What is recorded per question: the route that answered (typed pipeline `qc`, or an older engine), the planned query, the verifier status, and (for clarifications) the offered readings. A question that leaves the typed pipeline is scored as a non-answer here (kind refusal) and counted under `left_typed_pipeline`: its displayed answer comes from an older engine and is NOT scored for correctness by this harness (that is stated in the report).
Follow-ups: the previous turn's gold query is installed as the conversation state (as in the baseline's `--history gold`)."""
import argparse, collections, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.eval_v2 import Engines, run, summarize
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.product import make_legacy_brain as make_brain
from vgpt.query_ir import IRError, Query


class ProductPlanner:
    name = "product (vgpt.product.make_brain, end to end)"

    def __init__(self, db): self.b = make_brain(db, now=DEFAULT_NOW); self.routes = collections.Counter(); self.verifier = collections.Counter(); self.left = []; self.legacy = {}

    def plan(self, text, prev=None, draft=None):
        b = self.b; b.qcs = type(b.qcs)(); b.trace.__init__(); b.last_qc = None
        if prev and prev[1]:
            try: b.qcs.last = Query.from_text(prev[1])
            except IRError: pass
        t0 = time.perf_counter(); r = b.ask(text, use_slm=False); ms = (time.perf_counter() - t0) * 1000; path = r.get("path"); lq = b.last_qc
        info = {"ms": ms, "tok": 0, "source": f"route:{path}", "route": path, "intent": r.get("intent")}; self.routes[path] += 1
        if r.get("intent") == "CLARIFY" and path == "qc":
            ch = list(b.qcs.choices or [])
            if ch: info["clarify"] = [c.to_text() for c in ch]; info["question"] = r.get("answer")
            return "op=none", info                                                                         # a clarification with no stored readings (e.g. the 'neither' message) is a non-answer
        if path == "qc" and lq is not None and getattr(lq, "query", None) is not None and getattr(lq, "status", None) is not None:
            self.verifier[lq.status] += 1; info["status"] = lq.status; info["source"] = "route:qc"; return lq.query.to_text(), info
        self.left.append({"text": text, "route": path, "intent": r.get("intent")}); self.legacy[text] = r.get("intent"); info["source"] = f"route:{path}"; return "op=none", info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--set", default="dev"); ap.add_argument("--out"); ap.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set); rows = load(path)
    eng = Engines(); pl = ProductPlanner(a.db); t0 = time.time(); res = run(pl, rows, eng, "gold")
    from vgpt.brain import NON_ANSWER_INTENTS
    shown = 0
    for rec in res:                                                                                                      # a refusal-gold question that an OLDER engine answered with analysis is a false action even though no typed query exists
        it = pl.legacy.get(rec["text"])
        if rec["gold"] == "op=none" and it is not None and it not in NON_ANSWER_INTENTS and it not in ("STATUS", "HEALTH"):
            rec.update(kind="answer", correct=False, false_action=True, answered=True, scored_by="refusal", legacy_answer_intent=it); shown += 1
    summ = summarize(res); summ["false_actions_from_older_engines"] = shown
    summ.update({"system": pl.name, "set": os.path.basename(path), "history": "gold", "databases": [n for n, _ in eng.qcs], "wall_s": round(time.time() - t0), "routes": dict(pl.routes), "typed_pipeline_status": dict(pl.verifier), "left_typed_pipeline": pl.left})
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_family", "left_typed_pipeline")}, indent=1))
    if a.out: os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump({"summary": summ, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
