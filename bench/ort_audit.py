"""The product's own reader (int8 ONNX Runtime build) over the 3,842 audit questions: accuracy, and the number of decode passes, drafted rows and prompt tokens of every question, from which the time on the Raspberry Pi is estimated
(prompt token about 33 ms, a decode pass about 135 ms, every drafted row of a pass about 25 ms - the Pi measurements of 2026-10-25).

    python bench/ort_audit.py --name ort_base [--thr 0.5] [--draft-max N] [--hint base|clause4|hint1c] [--ort models/ort_frame2r12] [--limit N]

Drafts are verified by the model, so changing the draft threshold or length cannot change a frame; this script checks it (frames must equal the --ref run).  A change of the hint line can change frames: the strict-correct count is compared with the
baseline of the same backend (the run named by --ref)."""
import argparse, io, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ort", default="models/ort_frame2r12"); ap.add_argument("--thr"); ap.add_argument("--draft-max", type=int); ap.add_argument("--hint", default="base"); ap.add_argument("--line", default="none", choices=["none", "ice", "all"]); ap.add_argument("--native", help="weight file of the native engine (models/vg_*.vgw): run it instead of ONNX Runtime")
    ap.add_argument("--ref"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    if a.line != "none": os.environ["VGPT_VEHICLE_HINT"] = "1"
    if a.thr: os.environ["VGPT_DRAFT_THR"] = a.thr
    import eval_r10 as E
    import speed_variants as SV
    from vgpt.meaning import ort_backend as OB
    if a.draft_max: OB.DRAFT_MAX = a.draft_max
    m = OB.OrtFrameModel(os.path.join(ROOT, a.ort), threads=a.threads, native=(os.path.join(ROOT, a.native) if a.native else None)); SV.patch(m.glossary, a.hint)
    rows_log = []; orig = m._fwd; t_start = time.time()
    def spy(ids, start, kv, nkeep=1, decode=False):
        if decode: rows_log.append(len(ids))
        return orig(ids, start, kv, nkeep, decode)
    m._fwd = spy; out = []; t0 = time.time()
    for s in E.FROZEN + E.NEW:
        rows = E.load(s)
        if a.limit: rows = rows[:a.limit]
        for r in rows:
            tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev"); m.set_vehicle(veh if (a.line == "all" or (a.line == "ice" and veh != "ev")) else None); rows_log.clear()
            f, info = m.parse(r["text"], prev=r["prev"]); st = dict(m.stats); raw = info["raw"]
            out.append({"set": s, "id": str(r["id"]), "text": r["text"], "gold": r["gold"], "pred": raw, "strict": E.ok_prefix(raw, r["gold"]), "pmin": info.get("pmin"), "tail": st["tail_tokens"], "passes": st["passes"], "rows": sum(rows_log), "rpp": list(rows_log)})
        print(s, f"{time.time() - t0:.0f}s", flush=True)
    n = len(out); c = sum(x["strict"] for x in out); tail = sum(x["tail"] for x in out) / n; passes = sum(x["passes"] for x in out) / n; rws = sum(x["rows"] for x in out) / n
    est = [(x["tail"] * 33 + x["passes"] * 135 + (x["rows"] - x["passes"]) * 25) / 1000 for x in out]; est.sort()
    print(f"{a.name}: strict-correct {c} of {n}; per question: prompt tokens {tail:.1f}, decode passes {passes:.2f}, rows {rws:.2f}; estimated Pi model time mean {sum(est) / n:.2f} s, median {est[n // 2]:.2f} s, share over 1.5 s {100 * sum(e > 1.5 for e in est) / n:.1f} %")
    if a.ref:
        ref = {(x["set"], x["id"], x["text"]): x for x in json.load(open(os.path.join(ROOT, "results", "meaning", a.ref + "_r10", "predictions.json"), encoding="utf-8"))["predictions"]}
        same = sum(1 for x in out if ref.get((x["set"], x["id"], x["text"]), {}).get("pred") == x["pred"]); rc = sum(1 for v in ref.values() if v["strict"]); print(f"frames identical to {a.ref}: {same} of {n}; reference strict-correct {rc}; net {c - rc:+d}")
    d = os.path.join(ROOT, "results", "meaning", a.name + "_r10"); os.makedirs(d, exist_ok=True); json.dump({"predictions": out}, io.open(os.path.join(d, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__": main()
