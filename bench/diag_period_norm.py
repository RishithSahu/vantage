"""Diagnostic of the FACTORED period path: a pretrained extractor finds the time-expression SPAN (exact text with offsets); a frozen language model then does a much smaller task - normalise that one expression to the typed period form (relative amount + unit, calendar name, date, now, all time) given today's date. How accurate is that small task, compared with asking the model to write a whole plan?
Input spans come from results/gl_base_slice.json (bench/diag_gliner.py output, extractor only); gold is the period of the gold query. A period counts as right when its canonical text equals the gold's, or is an equivalent documented form (7 days = 1 week, 30 days = 1 month).
    python tools_pin.py 8,9 bench/diag_period_norm.py --model models/granite-4.0-1b-Q4_K_M.gguf [--spans results/gl_base_slice.json]"""
import argparse, json, os, sqlite3, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_gliner import gold_flags
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import capreg, plancompile as PC, plangrammar, prompt, signal_registry as R, toolspec as T
from vgpt.query_ir import Query
from vgpt.slm_backend import PrefixLlama

SYSTEM = ("Convert the time expression to exactly one period. Allowed forms: \"now\", \"all time\", \"last N minutes|hours|days|weeks|months\" (N a number), \"last week\", \"last month\", \"today\", \"yesterday\", \"day before yesterday\", \"this week\", \"previous week\", "
          "\"this month\", \"previous month\", \"last weekend\", \"this morning\", \"last night\", \"on YYYY-MM-DD\". \"last week\" and \"last month\" are the product's convention for 7 and 30 days. Answer with the period only.")


SHOTS = [("Expression: the last 5 days", '"last 5 days"'), ("Expression: yesterday", '"yesterday"'), ("Expression: this month", '"this month"'), ("Expression: over the past fortnight", '"last 2 weeks"'), ("Expression: since the start", '"all time"'),
         ("Expression: on the 3rd of March", '"on 2026-03-03"'), ("Expression: last night", '"last night"'), ("Expression: in the previous calendar week", '"previous week"'), ("Expression: the last 90 minutes", '"last 90 minutes"'), ("Expression: right now", '"now"')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "granite-4.0-1b-Q4_K_M.gguf")); ap.add_argument("--spans", default=os.path.join(HERE, "results", "gl_base_slice.json")); ap.add_argument("--n-ctx", type=int, default=2048); ap.add_argument("--shots", action="store_true", help="ten generic format examples as chat turns (each period form once; none is a benchmark question)"); a = ap.parse_args()
    spans = {r["id"]: r for r in json.load(open(a.spans, encoding="utf-8"))}; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); g = 'root ::= ' + plangrammar.value_rules(reg)[0]["period"] + "\n" + "\n".join(l for l in plangrammar.common_rules(reg) if l.startswith("val-"))
    g = "\n".join(["root ::= val-period"] + [l for l in plangrammar.common_rules(reg)])
    lm = PrefixLlama(a.model, n_ctx=a.n_ctx, threads=2, mmap=True); lm.set_prefix(SYSTEM, SHOTS if a.shots else [], name="p")
    ok = n = 0; res = []; t0 = time.time()
    for i, r in spans.items():
        row = rows[i]
        if row["prev_text"] or row["ir"].startswith("op=none"): continue
        gq = Query.from_text(row["ir"]); fl = gold_flags(gq)
        if not fl["explicit_period"] or not r["entities"].get("time period"): continue
        span = r["entities"]["time period"][0]; msg = f"Today: {prompt.today_line(DEFAULT_NOW)[7:]}\nExpression: {span}"
        out, _ = lm.generate(msg, grammar=g, max_tokens=14, stop_text=None, prefix="p"); canon = T.period_canonical(out.strip().strip('"')); gold = PC.query_fields(gq)["per"]
        good = canon == gold or {canon, gold} <= {"last:7:day", "last:1:week"} or {canon, gold} <= {"last:30:day", "last:1:month"}; n += 1; ok += good; res.append((span, out, gold, good))
    print(f"{os.path.basename(a.model)}: period normalisation from an extracted span: {ok}/{n} = {100 * ok / max(n, 1):.0f}%  ({(time.time() - t0) / max(n, 1):.1f} s each)")
    for x in res:
        if not x[3]: print("  wrong:", x)


if __name__ == "__main__": main()
