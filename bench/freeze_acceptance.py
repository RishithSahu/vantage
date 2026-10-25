"""Freezes the acceptance evaluation. Run ONLY after an independent reviewer has read every question and expected interpretation WITHOUT seeing any system prediction.
    python bench/freeze_acceptance.py --reviewer "<name>" [--review-notes <file>]
Records (results/acceptance/freeze.json): hashes of the question file, the scoring code (bench/evalcore.py, bench/eval_v2.py), the data fixtures (evaluation databases), the system source, the models, the configuration, the evaluation policy; the reviewer's name and date.
After this, bench/run_acceptance.py will run the frozen configuration ONCE. Before the freeze the only thing that touched the question file was a validity check of its gold queries (bench/check_indep.py); no system was run on it and nothing was tuned on it."""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.manifest import manifest, _file_hash

ACC = os.path.join(HERE, "bench", "acceptance_v1.tsv"); OUT = os.path.join(HERE, "results", "acceptance")
CONFIG = {"n_ctx": 4608, "retrieval": False, "complete": False, "rules_check": True, "history": "own (the system's accepted state; no gold history)", "databases": ["gen_courier_ext", "gen_demo", "gen_cold", "gen_roadwarrior", "gen_courier"],
          "scoring": "bench/evalcore.py: outcome signature on every database; explained non-answers need status AND exact plan; clarification is its own outcome (resolved_correct reported separately)", "policy": "answers only supported interpretations; ambiguity -> clarification with concrete readings; unavailable -> explained limitation"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--system", required=True, choices=["policy-rules", "policy-select", "policy-qwen"], help="the configuration to freeze (an explicit owner decision): policy-rules = deterministic candidates + clarification, no model; policy-select = + frozen-LFM candidate selection; policy-qwen = + the earlier Qwen-0.5B+LoRA whole-query model"); ap.add_argument("--reviewer", required=True); ap.add_argument("--review-notes", default=""); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); fr = os.path.join(OUT, "freeze.json")
    if os.path.exists(fr) and not a.force: sys.exit("already frozen (results/acceptance/freeze.json); refusing to overwrite without --force (a re-freeze must be disclosed)")
    if not os.path.exists(ACC): sys.exit("bench/acceptance_v1.tsv is missing")
    cfg = dict(CONFIG, system=a.system, selector_model=("models/LFM2.5-350M-Q4_K_M.gguf" if a.system == "policy-select" else None), whole_query_model=("models/qwen2.5-0.5b-instruct-q4_k_m.gguf + models/ir-lora-qwen25-05b.gguf (trained earlier)" if a.system == "policy-qwen" else None)); m = manifest(cfg); m["acceptance_questions"] = _file_hash(ACC); m["scoring_code"] = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16] for f in ("bench/evalcore.py", "bench/eval_v2.py", "bench/paired.py")}
    m["reviewer"] = a.reviewer; m["review_notes_hash"] = _file_hash(a.review_notes) if a.review_notes and os.path.exists(a.review_notes) else None; m["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(m, open(fr, "w"), indent=1); print("frozen:", json.dumps({k: m[k] for k in ("source_hash", "acceptance_questions", "reviewer", "frozen_at")}))


if __name__ == "__main__": main()
