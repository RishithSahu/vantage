"""Does the int8 ONNX Runtime build (the board's model) write the same frames as the llama.cpp adapter it was built from?

    python bench/ort_vs_llama.py models/ort_frame2r12 models/meaning-lora-frame2r12.gguf [--sets sealed_v7 sealed_v5 ...] [--limit 150]

The car type is told to both (VGPT_VEHICLE_HINT=1) with the tag of the row.  Prints identical frames, accuracy of each (strict, against the gold), and milliseconds per question."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_VEHICLE_HINT"] = "1"
import eval_r10 as E                                                  # noqa: E402  (sets VGPT_FRAME=2)
from vgpt.meaning import interface                                    # noqa: E402
from vgpt.meaning.ort_backend import OrtFrameModel                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ort_dir"); ap.add_argument("adapter"); ap.add_argument("--sets", nargs="*", default=["sealed_v7", "sealed_v5"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
    ort = OrtFrameModel(a.ort_dir if os.path.isabs(a.ort_dir) else os.path.join(ROOT, a.ort_dir), threads=a.threads)
    ll = interface.FrameModel(a.adapter if os.path.isabs(a.adapter) else os.path.join(ROOT, a.adapter), threads=a.threads)
    n = same = ok_o = ok_l = 0; t_o = t_l = 0.0
    for st in a.sets:
        rows = E.load(st)[:a.limit] if a.limit else E.load(st)
        for r in rows:
            veh = r["tag"].split(":")[0] if r["tag"].split(":")[0] in E.VEHICLES else "ev"
            ort.set_vehicle(veh); ll.set_vehicle(veh)
            t = time.perf_counter(); fo, io = ort.parse(r["text"], prev=r["prev"]); t_o += time.perf_counter() - t
            t = time.perf_counter(); fl, il = ll.parse(r["text"], prev=r["prev"]); t_l += time.perf_counter() - t
            n += 1; same += io["raw"] == il["raw"]; ok_o += E.ok_prefix(io["raw"], r["gold"]); ok_l += E.ok_prefix(il["raw"], r["gold"])
    print(f"{n} questions: identical frames {same} ({100 * same / n:.1f} %); strict accuracy  ort {100 * ok_o / n:.1f} %  llama {100 * ok_l / n:.1f} %;  ms per question  ort {1000 * t_o / n:.0f}  llama {1000 * t_l / n:.0f}  (this laptop, not the board)")


if __name__ == "__main__":
    main()
