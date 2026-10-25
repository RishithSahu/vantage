"""Audit of the entity-extraction interface of a GLiNER2.5 checkpoint (frozen): why are numbers, selections and restrictions lost? Same questions (the stand-alone ones of the frozen slice), same labels and descriptions, strict target = the independent annotation. Variants differ in ONE interface choice each:
  combined   classification (result, statistic, turn) + entities in one schema (the earlier probe), default threshold 0.5, overlap policy `flat`
  entities   entities only, 0.5, flat
  nested     entities only, 0.5, overlap policy `nested`
  longest    entities only, 0.5, overlap policy `longest`
  nested_.3  entities only, threshold 0.3, nested   (false additions are counted: a lower threshold is only useful if it does not invent)
Saved: every extracted span with label, offsets and confidence, plus the library/model identity and the decoding settings. A DIAGNOSTIC of the interface, not a product path.
    PYTHONPATH=pylib_gliner2_v2 python tools_pin.py 10,11 bench/audit_extraction.py --model models/gliner25_small --out results/audit_ext_small.json"""
import argparse, json, os, re, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_gliner import annotations, has
from bench.indep_io import load
from vgpt import toolspec as T
from vgpt.extractor import ENTITY_LABELS, ENTITY_LABELS_V2, TURN_LABELS

VARIANTS = {"combined": dict(kind="combined", threshold=0.5, overlap="flat"), "entities": dict(kind="entities", threshold=0.5, overlap="flat"), "nested": dict(kind="entities", threshold=0.5, overlap="nested"),
            "longest": dict(kind="entities", threshold=0.5, overlap="longest"), "nested_.3": dict(kind="entities", threshold=0.3, overlap="nested")}
ROLES = {"v1": dict(num=("number with unit", "threshold condition"), res=("driving situation", "threshold condition"), sel="trip or session selection", per="time period"), "v2": dict(num=("number", "threshold"), res=("situation", "threshold"), sel="selection", per="period")}


def numbers(s): return [float(v) for v in re.findall(r"\d+(?:\.\d+)?", s or "")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); ap.add_argument("--labels", default="v2", choices=["v1", "v2", "v3"]); ap.add_argument("--out"); a = ap.parse_args(); LAB = {"v1": ENTITY_LABELS, "v2": ENTITY_LABELS_V2, "v3": list(ENTITY_LABELS_V2)}[a.labels]; R = ROLES["v2" if a.labels == "v3" else a.labels]; NUMLAB, RESLAB = R["num"], R["res"]
    import gliner2
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(a.model); m.eval(); print("library", gliner2.__version__, "model", a.model, type(m).__name__, "architecture", m.config.architecture, "params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; ann = annotations(); rows = [rows[i] for i in ids if not rows[i]["prev_text"]]
    res_lab = {v: k for k, v in T.RESULT_NL.items()}; stat_lab = {v: k for k, v in T.STAT_NL.items()}
    out = {"library": gliner2.__version__, "model": a.model, "architecture": m.config.architecture, "variants": {}, "labels": LAB, "label_set": a.labels}
    for vname, v in VARIANTS.items():
        sch = m.create_schema()
        if v["kind"] == "combined": sch = sch.classification("result", list(res_lab), multi_label=True, cls_threshold=0.0).classification("statistic", list(stat_lab), multi_label=True, cls_threshold=0.0).classification("turn", list(TURN_LABELS.values()), multi_label=True, cls_threshold=0.0)
        sch = sch.entities(LAB); c = dict(num_n=0, num=0, num_fp=0, num_neg=0, sel_n=0, sel=0, sel_fp=0, sel_neg=0, rs_n=0, rs=0, rs_fp=0, rs_neg=0, int_n=0, int_hit=0, int_fp=0, int_neg=0); raw = {}; ms = []
        for r in rows:
            an = ann[r["id"]]; t0 = time.perf_counter(); x = m.extract(r["text"], sch, threshold=v["threshold"], include_confidence=True, include_spans=True, overlap_policy=v["overlap"]); ms.append((time.perf_counter() - t0) * 1000)
            ents = x.get("entities", {}); raw[r["id"]] = {k: [{"text": e["text"], "start": e.get("start"), "end": e.get("end"), "conf": round(float(e.get("confidence", 0)), 3)} for e in lst] for k, lst in ents.items() if lst}
            texts = lambda labs: [e["text"] for l in labs for e in ents.get(l, [])]
            if has(an["number"]): c["num_n"] += 1; want = numbers(an["number"]); got = numbers(" ".join(texts(NUMLAB))); c["num"] += any(abs(w - g) < 1e-6 for w in want for g in got)
            else: c["num_neg"] += 1; c["num_fp"] += bool(texts(NUMLAB))
            if has(an["selection"]): c["sel_n"] += 1; c["sel"] += bool(ents.get(R["sel"]))
            else: c["sel_neg"] += 1; c["sel_fp"] += bool(ents.get(R["sel"]))
            if has(an["restriction"]): c["rs_n"] += 1; c["rs"] += bool(texts(RESLAB))
            else: c["rs_neg"] += 1; c["rs_fp"] += bool(texts(RESLAB))
            if has(an["interval"]): c["int_n"] += 1; c["int_hit"] += bool(ents.get(R["per"]))
            else: c["int_neg"] += 1; c["int_fp"] += bool(ents.get(R["per"]))
        print(f"[{vname:9s}] thr {v['threshold']} overlap {v['overlap']:7s} {sum(ms) / len(ms):4.0f} ms/q | number {c['num']}/{c['num_n']} (invented {c['num_fp']}/{c['num_neg']}) | selection {c['sel']}/{c['sel_n']} (invented {c['sel_fp']}/{c['sel_neg']}) | restriction {c['rs']}/{c['rs_n']} (invented {c['rs_fp']}/{c['rs_neg']}) | interval {c['int_hit']}/{c['int_n']} (invented {c['int_fp']}/{c['int_neg']})", flush=True)
        out["variants"][vname] = {"settings": v, "counts": c, "ms": round(sum(ms) / len(ms)), "raw": raw}
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
