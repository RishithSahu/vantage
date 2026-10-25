"""Reference experiment, task 1 (RESEARCH ONLY, outside the deployment envelope): a stronger frozen instruction model (Qwen2.5-7B-Instruct Q4_K_M) describes the requested computation in PLAIN structured language - no operation labels, no schema - for the same reviewed questions as the typed task (bench/diag_format.py --format qwen). Same message (today, capability cards, conversation state), one fixed instruction, greedy decoding, no demonstrations, no gold, no grammar. The outputs are saved for a READ-THROUGH grading against the independent annotation and the gold description (bench/ref_plain_grade.tsv): understanding errors versus representation errors versus evaluation ambiguity.
    python tools_pin.py 0,2,4,6 bench/ref_plain.py --model models/qwen25_7b_ref/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --threads 4 --out results/ref7b_plain.json"""
import argparse, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt.modelplan import Planner, State
from vgpt.query_ir import Query
from vgpt import plancompile as PC
from vgpt.slm_backend import PrefixLlama

SYSTEM = """You read a person's message about their vehicle's recorded data and say, in plain words, exactly what computation they are asking for. The message comes with Capabilities (what the vehicle records) and, for a follow-up, the previous request.
Write exactly these five lines and nothing else:
Measure: <the quantity or event asked about, in words>
Calculation: <what is computed: the current value, an average / total / highest / lowest, a count, a list of occurrences, a yes/no, a share of time, a comparison, when something changed, why it changed, what happened around an event, a trend, a ranking of trips or sessions, whether it is recorded, a ready-made check - or "not a vehicle-data request">
Period: <the time period exactly as the person stated it, or "not stated">
Restrictions: <conditions the data must satisfy, exactly as stated (a driving situation, a threshold), or "none">
Scope: <which trips or charging sessions are meant (for example the longest two), or "all">
Describe only what the message says. Write "not stated" or "none" instead of guessing; never add a period or restriction the person did not state."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out"); a = ap.parse_args()
    P = Planner(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True), load_model=False, dialect="qwen")
    lm = PrefixLlama(a.model, n_ctx=4096, threads=a.threads, n_batch=64, mmap=True); lm.set_prefix(SYSTEM, [], name="plain")
    rows = load(os.path.join(HERE, "bench", "indep_dev.tsv"))[:: a.stride]; rows = rows[: a.limit or None]; out = []
    for i, r in enumerate(rows):
        st = State()
        if r["prev_ir"]:
            try: st.prev_plan = PC.query_to_plan(Query.from_text(r["prev_ir"]))
            except Exception: pass
        msg = P.build_message(r["text"], st, DEFAULT_NOW)[0]; t0 = time.perf_counter(); raw, info = lm.generate(msg, grammar=None, max_tokens=170, stop_text=None, prefix="plain", history=None)
        out.append({"id": r["id"], "text": r["text"], "gold": r["ir"], "prev_text": r["prev_text"], "output": raw, "ms": round((time.perf_counter() - t0) * 1000), "prompt_tokens": info.get("prompt_tokens"), "out_tokens": info.get("out_tokens")})
        if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(rows)}", flush=True)
    print(f"{len(out)} descriptions; mean {sum(x['ms'] for x in out) / len(out):.0f} ms, {sum(x['out_tokens'] or 0 for x in out) / len(out):.0f} output tokens")
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
