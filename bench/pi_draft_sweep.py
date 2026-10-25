"""Parse time against how many tokens are guessed ahead per pass (the model checks every one, so the frame never changes; only the number of passes over the weights does).

    taskset -c 0,1 python bench/pi_draft_sweep.py [--n 120]

Each question is read once per setting, settings interleaved, the previous question's frame carried as in a conversation."""
import argparse, glob, os, statistics as S, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt; vehiclegpt._adopt_product_model()
from vgpt.meaning import ort_backend as OB


def questions(n):
    out = []
    for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2: out.append(p[2])
    pool = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench", "meaning_eval", "*.tsv"))):
        if "followup" in f: continue
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) > 2: pool.append(p[1])
    out += pool[::max(1, len(pool) // n)]
    return out[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); ap.add_argument("--settings", default="8,12,16"); a = ap.parse_args()
    m = OB.OrtFrameModel(os.path.join(ROOT, "models", "ort_frame2w"), threads=2); sets = [int(x) for x in a.settings.split(",")]
    m.parse("how fast am I going"); res = {k: [] for k in sets}; frames = {k: [] for k in sets}; prev = "now speed none none none none none none"
    for q in questions(a.n):
        raws = {}
        for k in sets:
            OB.DRAFT_MAX = k; t = time.perf_counter(); f, info = m.parse(q, prev=prev); dt = time.perf_counter() - t
            res[k].append((dt, m.stats.get("passes"), m.stats.get("decode_ms"), m.stats.get("tail_tokens"))); raws[k] = info["raw"]
        if len(set(raws.values())) != 1: print("FRAMES DIFFER", q, raws)
        prev = raws[sets[0]]
    for k in sets:
        r = res[k]; print(f"draft<={k:2d}: parse median {S.median(x[0] for x in r):.3f}s mean {S.mean(x[0] for x in r):.3f}s p90 {sorted(x[0] for x in r)[int(.9 * len(r))]:.3f}s  passes mean {S.mean(x[1] for x in r):.2f}  decode ms median {S.median(x[2] for x in r):.0f}")


if __name__ == "__main__":
    main()
