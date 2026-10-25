"""Diagnostic of the pretrained schema-conditioned EXTRACTOR (GLiNER2, frozen, no training, offline from the local Hugging Face cache) on the frozen slice, with the FACTORED semantic representation: the requested RESULT (what the person wants back, not an internal function name), the statistic, the conversation-edit kind, and SPANS with offsets for the target quantity, the event, the time period, the restriction, numbers with units and trip / session selections.
Evaluated against the INDEPENDENT annotation of what each question says (bench/slice_v1_annotations.tsv, written from the question text alone) - not against the evaluator's normalised gold: an explicitly stated "last seven days" counts as an explicit period even though the gold normalises it to a default.
Reported: result top-1 / top-3, statistic top-1, whether the interval / restriction / number / selection spans are present exactly when the question states them (recall AND false-alarm), whether the target mentions retrieve the gold capabilities, and the COMPLETE-interpretation count (result right AND interval right AND restriction right). Nothing here excludes a result permanently: ranked alternatives are kept. An extraction-only result does not satisfy the SLM-assistant requirement.
    PYTHONUTF8=1 python tools_pin.py 4,5 bench/diag_gliner.py [--model fastino/gliner2-base-v1] [--out results/gl_base_slice.json]
Nothing is downloaded (HF_HUB_OFFLINE=1). GLiNER2.5-small needs AutoExtractor from a newer gliner2 package and is a separate, approved download."""
import argparse, json, os, re, sqlite3, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.indep_io import load
from vgpt import toolspec as T
from vgpt.extractor import ENTITY_LABELS, TURN_LABELS
from vgpt.modelplan import Planner
from vgpt.query_ir import Query
from vgpt import plancompile as PC


def annotations(path=os.path.join(HERE, "bench", "slice_v1_annotations.tsv")):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        p = l.rstrip("\n").split("\t") + [""] * 9
        out[p[0]] = {"result": p[1], "statistic": p[2], "interval": p[3], "restriction": p[4], "number": p[5], "selection": p[6], "edit": p[7]}
    return out


def has(v): return v not in ("-", "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="fastino/gliner2-base-v1"); ap.add_argument("--slice", default=os.path.join(HERE, "bench", "slice_v1.txt")); ap.add_argument("--out"); a = ap.parse_args()
    ids = [l.strip() for l in open(a.slice, encoding="utf-8") if l.strip() and not l.startswith("#")]; rows = {r["id"]: r for r in load(os.path.join(HERE, "bench", "indep_dev.tsv"))}; ann = annotations(); rows = [rows[i] for i in ids]
    from vgpt.extractor import Extractor
    ex = Extractor(a.model); m = ex.model; res_lab = {v: k for k, v in T.RESULT_NL.items()}; stat_lab = {v: k for k, v in T.STAT_NL.items()}; tlab = {v: k for k, v in TURN_LABELS.items()}
    schema = (m.create_schema().classification("result", list(res_lab), multi_label=True, cls_threshold=0.0).classification("statistic", list(stat_lab), multi_label=True, cls_threshold=0.0)
              .classification("turn", list(tlab), multi_label=True, cls_threshold=0.0).entities(ENTITY_LABELS))
    P = Planner(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True), load_model=False)
    c = dict(res1=0, res3=0, res_n=0, st1=0, st_n=0, int_hit=0, int_n=0, int_fp=0, int_neg=0, rs_hit=0, rs_n=0, rs_fp=0, rs_neg=0, num_hit=0, num_n=0, sel_hit=0, sel_n=0, tgt_hit=0, tgt_n=0, complete=0, complete_n=0, turn_ok=0); ms = []; out = []
    for r in rows:
        an = ann[r["id"]]; t0 = time.perf_counter(); x = m.extract(r["text"], schema, include_confidence=True, include_spans=True); ms.append((time.perf_counter() - t0) * 1000)
        ranked = [res_lab[e["label"]] for e in sorted(x["result"], key=lambda e: -e["confidence"])]; stats = [stat_lab[e["label"]] for e in sorted(x["statistic"], key=lambda e: -e["confidence"])]; ents = x.get("entities", {}); turn = tlab[sorted(x["turn"], key=lambda e: -e["confidence"])[0]["label"]]
        c["turn_ok"] += (turn != "new") == bool(r["prev_text"])
        spans = {k: [e["text"] for e in v] for k, v in ents.items()}; out.append({"id": r["id"], "result_scores": {res_lab[e["label"]]: float(e["confidence"]) for e in x["result"]}, "results": ranked[:3], "statistic": stats[:2], "spans": spans, "span_detail": {k: [{"text": e["text"], "start": e.get("start"), "end": e.get("end"), "conf": round(float(e.get("confidence", 0)), 3)} for e in v] for k, v in ents.items() if v}, "turn": turn, "settings": {"threshold": 0.5, "overlap": "library default", "model": a.model}})
        if not r["prev_text"]:
            c["res_n"] += 1; c["res1"] += ranked[:1] == [an["result"]]; c["res3"] += an["result"] in ranked[:3]
            if has(an["statistic"]): c["st_n"] += 1; c["st1"] += stats[:1] == [an["statistic"]]
        for key, label, ann_key in (("int", "time period", "interval"), ("rs", "restriction", "restriction")):
            present = bool(ents.get(label) or ents.get("driving situation") or ents.get("threshold condition")) if key == "rs" else bool(ents.get(label))
            if has(an[ann_key]): c[key + "_n"] += 1; c[key + "_hit"] += present
            else: c[key + "_neg"] += 1; c[key + "_fp"] += present
        if has(an["number"]): c["num_n"] += 1; nums = [float(v) for e in (ents.get("number with unit", []) + ents.get("threshold condition", [])) for v in re.findall(r"\d+(?:\.\d+)?", e["text"])]; want = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", an["number"])]; c["num_hit"] += any(abs(w - n) < 1e-6 for w in want for n in nums)
        if has(an["selection"]): c["sel_n"] += 1; c["sel_hit"] += bool(ents.get("trip or session selection") or ents.get("event or warning"))
        if not r["prev_text"] and not r["ir"].startswith("op=none"):
            gq = Query.from_text(r["ir"]); want = [y for y in (gq.target, gq.target2, gq.event.type if gq.event else None) if y]
            if want:
                c["tgt_n"] += 1; got = set(h.id for h in P.retriever.search(r["text"], k=6)[0])
                for mt in ents.get("measured quantity", []) + ents.get("event or warning", []): got |= {h.id for h in P.retriever.search(mt["text"], k=3)[0]}
                c["tgt_hit"] += all(y in got for y in want)
            c["complete_n"] += 1; ok_res = ranked[:1] == [an["result"]]; ok_int = bool(ents.get("time period")) == has(an["interval"]); ok_rs = bool(ents.get("driving situation") or ents.get("threshold condition")) == has(an["restriction"]); c["complete"] += ok_res and ok_int and ok_rs
    pct = lambda k, n: f"{k}/{n} = {100 * k / max(n, 1):.0f}%"
    print(f"{os.path.basename(a.slice)} {a.model}: {len(rows)} questions, {sum(ms) / len(ms):.0f} ms per question (result + statistic + turn + 6 entity types in one pass)")
    print(f"  requested result (stand-alone): top-1 {pct(c['res1'], c['res_n'])}, top-3 {pct(c['res3'], c['res_n'])}   statistic where stated: top-1 {pct(c['st1'], c['st_n'])}   new-vs-continuation detected (NOT the edit kind; the probe sees the current message only, without the previous query): {pct(c['turn_ok'], len(rows))}")
    print(f"  interval span present where the question states one: {pct(c['int_hit'], c['int_n'])};  invented where it does not: {c['int_fp']}/{c['int_neg']}")
    print(f"  restriction span present where stated: {pct(c['rs_hit'], c['rs_n'])};  invented where not: {c['rs_fp']}/{c['rs_neg']}   number extracted: {pct(c['num_hit'], c['num_n'])}   selection span where stated: {pct(c['sel_hit'], c['sel_n'])}")
    print(f"  target mentions + question retrieve all gold capabilities: {pct(c['tgt_hit'], c['tgt_n'])}")
    print(f"  THREE-CHECK agreement only (requested result top-1 right AND interval-span presence right AND restriction-span presence right; NOT a complete interpretation: values, targets, numbers, selections and scope are not checked), stand-alone answerable: {pct(c['complete'], c['complete_n'])}")
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
