"""DIRECT QUESTION-TO-PATTERN RETRIEVAL, before any composer is built: does a correct reusable PROGRAM STRUCTURE appear among the first 1 / 3 / 10 DISTINCT structures retrieved for a question?
A structure (skeleton) is the typed query with every vehicle-specific name replaced by a type: the operation, the statistic, the event selector, the kinds of restriction (named / threshold with its direction), the number of groups, whether a second scope, an entity selection (entity, direction, a count?) or an event window (before / after) is present, and whether the target is a signal or an event. NOT part of the skeleton: signal / event / tool ids, units, numbers, and the period (a stated period and a defaulted one cannot be told apart in the gold IR; periods are resolved by separate, general machinery). Correct signal retrieval is NOT counted as correct program retrieval.
The index: the schema-generated demonstration bank (vgpt/demos.py), compiled to typed queries by the product's own compiler. The bank was DESIGNED AFTER READING development failures, so development figures are OPTIMISTIC and are labelled so; the final set (bench/indep_final.tsv) and the diagnostic set (bench/diag_v1.tsv) were not used to design it (their nearest-neighbour similarity to the bank is measured by bench/check_bank_independence.py).
Reported per retrieval method and per question set: structure coverage @1/@3/@10; operation-only coverage; and the CEILING - how many gold structures exist in the bank at all (a missing structure cannot be retrieved by any ranking).
    python tools_pin.py 8,9 bench/diag_pattern_retrieval.py [--per-fn 24]"""
import argparse, collections, math, os, re, sqlite3, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import capreg, demos, plancompile as PC, signal_registry as R, toolspec as T
from vgpt.query_ir import Query
from vgpt.retrieval import Embedder


def skeleton(q):
    ev = q.event; sel = q.scope.select
    conds = tuple(sorted(("named" if c.predicate else "thr:" + str(c.op)) for c in q.scope.conditions))
    return (q.op, q.agg if q.op in ("aggregate", "extreme", "compare", "count", "percent_time") else None, (ev.selector if ev is not None and q.op in ("list_events", "event_context", "explain_event") else None),
            conds, len(q.groups or []), q.scope_b is not None, (sel.entity, sel.order, sel.k is not None, sel.klass is not None) if sel is not None else None,
            ("evt", bool(ev.before), bool(ev.after)) if ev is not None else None, "sig" if q.target else ("evt" if ev is not None else None))


def bm25(docs):
    toks = [re.findall(r"[a-z0-9\[\]]+", d.lower()) for d in docs]; df = collections.Counter(w for t in toks for w in set(t)); N = len(docs); avg = sum(map(len, toks)) / N; idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
    def score(q):
        qt = re.findall(r"[a-z0-9\[\]]+", q.lower()); out = np.zeros(N)
        for i, t in enumerate(toks):
            c = collections.Counter(t)
            for w in qt:
                if w in c: out[i] += idf[w] * c[w] * 2.2 / (c[w] + 1.2 * (0.25 + 0.75 * len(t) / avg))
        return out
    return score


def rrf(*rank_lists, k=60):
    s = collections.defaultdict(float)
    for r in rank_lists:
        for pos, i in enumerate(r): s[i] += 1 / (k + pos + 1)
    return sorted(s, key=s.get, reverse=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-fn", type=int, default=24); ap.add_argument("--seed", type=int, default=11); a = ap.parse_args()
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); emb = Embedder(os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=True)
    bank = demos.generate(reg, a.seed, a.per_fn); mask = demos.mask_function(reg); entries = []
    for e in bank:
        if e["fn"] in T.NO_TURN: continue                                    # refusals / small talk / unresolved are not query structures
        try:
            o = PC.compile_plan(T.call_to_plan(e["fn"], e["args"]), None, None, reg, DEFAULT_NOW, 0)
            if o.kind == "query": entries.append((e["q"], skeleton(o.query)))
        except Exception: pass
    print(f"bank: {len(bank)} demonstrations, {len(entries)} compile to a query, {len({s for _, s in entries})} distinct structures")
    keys_m = [mask(q) for q, _ in entries]; E_m = emb.encode(keys_m); E_u = emb.encode([q for q, _ in entries]); bm = bm25(keys_m)
    sets = {}
    for name, f in (("dev (OPTIMISTIC: bank designed after dev failures)", "indep_dev.tsv"), ("final (not used to design the bank)", "indep_final.tsv"), ("diagnostic v1", "diag_v1.tsv")):
        rows = [r for r in load(os.path.join(HERE, "bench", f)) if not r["prev_text"] and not r["ir"].startswith("op=none")]; qs = []
        for r in rows:
            try: qs.append((r["text"], skeleton(Query.from_text(r["ir"]))))
            except Exception: pass
        sets[name] = qs
    def ranker(method, text):
        if method == "embedding, names masked": return list(np.argsort(-(E_m @ emb.encode([mask(text)], query=True)[0])))
        if method == "embedding, unmasked": return list(np.argsort(-(E_u @ emb.encode([text], query=True)[0])))
        if method == "BM25, names masked": return list(np.argsort(-bm(mask(text))))
        if method == "hybrid (embedding masked + BM25 masked, RRF)": return rrf(list(np.argsort(-(E_m @ emb.encode([mask(text)], query=True)[0]))), list(np.argsort(-bm(mask(text)))))
    bank_sk = {s for _, s in entries}; bank_ops = {s[0] for _, s in entries}
    for sname, qs in sets.items():
        n = len(qs); ceil = sum(s in bank_sk for _, s in qs); opceil = sum(s[0] in bank_ops for _, s in qs); gold_distinct = len({s for _, s in qs})
        print(f"\n== {sname}: {n} stand-alone answerable questions, {gold_distinct} distinct gold structures; structure present in the bank at all: {ceil}/{n} ({100 * ceil / n:.0f}%); operation present: {opceil}/{n}")
        for method in ("embedding, names masked", "embedding, unmasked", "BM25, names masked", "hybrid (embedding masked + BM25 masked, RRF)"):
            hit = {1: 0, 3: 0, 10: 0}; ophit = {1: 0, 3: 0, 10: 0}
            for text, gs in qs:
                order = ranker(method, text); seen = []; seen_ops = []
                for i in order:
                    s = entries[int(i)][1]
                    if s not in seen: seen.append(s)
                    if s[0] not in seen_ops: seen_ops.append(s[0])
                    if len(seen) >= 10 and len(seen_ops) >= 10: break
                for k in hit: hit[k] += gs in seen[:k]; ophit[k] += gs[0] in seen_ops[:k]
            print(f"  {method:48s} structure @1 {100 * hit[1] / n:3.0f}%  @3 {100 * hit[3] / n:3.0f}%  @10 {100 * hit[10] / n:3.0f}%   |  operation only @1 {100 * ophit[1] / n:3.0f}%  @3 {100 * ophit[3] / n:3.0f}%")


if __name__ == "__main__": main()
