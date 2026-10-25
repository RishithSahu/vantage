"""Exact-frame accuracy of a frame-model backend on the frozen meaning sets (a sample: every STEP-th row of every set), with the speed of each parse.

    taskset -c 0,1 python bench/ort_accuracy.py ort|llama [STEP] [OFFSET]

The two backends run the same weights (the llama.cpp one from the GGUF + LoRA file, the ONNX one from the same two files merged and stored as int8), so any difference is the cost of the int8 storage."""
import glob
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import vehiclegpt                                                       # noqa: E402
vehiclegpt._adopt_product_model()
from meaning_eval import evalio                                         # noqa: E402
from vgpt.vehicle import lake                                           # noqa: E402


def main():
    kind = sys.argv[1]; step = int(sys.argv[2]) if len(sys.argv) > 2 else 10; off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lake.adopt_path(os.path.join(ROOT, "data", "world_aster.db")); rows = []
    sets = ["platform_v1", "platform_v2", "adv_v1", "adv_v2", "adv_v3", "novel_v1", "messy_v1", "user_v1", "sealed_v2", "sealed_v3", "hard_v1", "hard_v2", "hard_v3", "hard_v4", "hard_v5"]
    for f in [os.path.join(ROOT, "bench", "meaning_eval", n + ".tsv") for n in sets]:
        for r in evalio.load_tsv(f)[off::step]: r["set"] = os.path.basename(f)[:-4]; rows.append(r)
    ver = int(os.environ.get("VGPT_FRAME", 2))
    if kind == "ort":
        sys.path.insert(0, os.path.join(ROOT, "bench")); from ort_frame import OrtFrameModel
        m = OrtFrameModel(os.environ.get("ORT_DIR", os.path.join(ROOT, "models", "ort_frame2d_c")), version=ver)
    else:
        from vgpt.meaning.interface import FrameModel
        m = FrameModel(os.environ["VGPT_MEANING_ADAPTER"], threads=2, version=ver)
    m.parse("how fast am I going"); ok = 0; ts = []; out = []
    for r in rows:
        t = time.perf_counter(); f, info = m.parse(r["text"]); ts.append(time.perf_counter() - t); good = f is not None and f.to_text() == r["gold"].to_text(); ok += good
        out.append({"set": r["set"], "id": r["id"], "ok": bool(good), "raw": info["raw"], "s": round(ts[-1], 3)})
    print("%s: %d rows, exact frame %.1f%% (%d)  parse median %.2fs mean %.2fs max %.2fs" % (kind, len(rows), 100.0 * ok / len(rows), ok, st.median(ts), st.mean(ts), max(ts)))
    json.dump(out, open(os.path.join(ROOT, "results", f"acc_{kind}_{step}_{off}.json"), "w")) if os.path.isdir(os.path.join(ROOT, "results")) else None


if __name__ == "__main__": main()
