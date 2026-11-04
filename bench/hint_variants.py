"""What the glossary hint line costs and what it buys: the frame model is run over the development questions with the hint line as it is, cut to fewer terms, cut to each term's first clause, and left out; for each variant the exact-frame accuracy
and the number of prompt tokens (the Pi pays about 36 ms per prompt token).

    python bench/hint_variants.py [--out results/hint_variants.json] [--limit 0]

Uses the same development questions as bench/calibrate_confidence.py (never the sealed, hold-out or acceptance sets)."""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import calibrate_confidence as CC


def clause(meaning, words=6):
    first = meaning.split(";")[0].split(",")[0].split(" - ")[0]
    return " ".join(first.split()[:words])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/hint_variants.json"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.product import _frame_model
    m = _frame_model(os.environ.get("VGPT_MEANING_ADAPTER"), 4); G = m.glossary; orig = G.hint_line
    qs = list(CC.questions()); seen = set(); qs = [x for x in qs if not ((x[1], x[2]) in seen or seen.add((x[1], x[2])))]
    if a.limit: qs = qs[::max(1, len(qs) // a.limit)]
    def make(limit, short):
        def f(text, limit_=4, vehicle=None):
            hit = [e for e in G.lookup(text) if e.get("hint")][:limit]
            return ("Terms: " + "; ".join(f"{e['term']} = {clause(e['meaning']) if short else e['meaning']}" for e in hit)) if hit and limit else ""
        return f
    variants = {"as shipped (4 terms, full)": lambda t, limit=4, vehicle=None: orig(t, limit=4, vehicle=vehicle), "2 terms, full": make(2, False), "1 term, full": make(1, False), "4 terms, first clause": make(4, True), "no hint line": make(0, False)}
    res = {}
    for name, fn in variants.items():
        G.hint_line = fn; ok = 0; ms = 0.0; toks = 0; t0 = time.time(); n = 0
        for _, text, prev, gold in qs:
            fr, info = m.parse(text, prev=prev) if prev is not None else m.parse(text); n += 1
            ok += int(fr is not None and CC.right(info["raw"], gold)); ms += info["ms"]; toks += m.stats.get("tail_tokens", 0)
        res[name] = {"n": n, "exact": ok / n, "tail_tokens_mean": toks / n, "laptop_ms_mean": ms / n}; print(f"{name:30} exact {100 * ok / n:5.2f} %  tail tokens {toks / n:5.1f}  laptop {ms / n:5.0f} ms  ({time.time() - t0:.0f} s)", flush=True)
    G.hint_line = orig; json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__": main()
