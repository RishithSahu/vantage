"""Second check for a hint-line variant: the probe questions of the three car types (the files used for the Raspberry Pi tests, 793 questions, no gold labels) are read with the shipped hint line and with the variant, and every question whose frame
differs is listed.  A frame that differs is reviewed by hand: did the variant lose the reading, or just word it differently?

    python bench/probe_variant_diff.py --variant clause4 [--out results/probe_diff_clause4.json]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pi_cartype as PC  # noqa: E402
import speed_variants as SV  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--out"); ap.add_argument("--adapter", default="models/meaning-lora-frame2r12.gguf"); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
    from vgpt.meaning import interface
    fm = interface.FrameModel(os.path.join(ROOT, a.adapter), threads=a.threads); orig = fm.glossary.hint_line; res = []; diff = []
    qs = {k: PC.questions(k) for k in ("ev", "petrol", "diesel")}
    for kind, items in qs.items():
        fm.set_vehicle(kind)
        for src, q in items:
            fm.glossary.hint_line = orig; f0, i0 = fm.parse(q)
            SV.patch(fm.glossary, a.variant); f1, i1 = fm.parse(q); fm.glossary.hint_line = orig
            res.append((kind, src, q, i0["raw"], i1["raw"], i0.get("pmin"), i1.get("pmin")))
            if i0["raw"] != i1["raw"]: diff.append(res[-1])
        print(kind, len(items), "questions so far differ:", len(diff), flush=True)
    out = a.out or f"results/probe_diff_{a.variant}.json"; json.dump({"variant": a.variant, "n": len(res), "different": diff}, open(os.path.join(ROOT, out), "w"), indent=1)
    print(f"variant {a.variant}: {len(diff)} of {len(res)} probe questions read differently")
    for kind, src, q, r0, r1, p0, p1 in diff: print(f"  [{kind}] {q}\n      shipped {r0}  ({p0:.2f})\n      variant {r1}  ({p1:.2f})")


if __name__ == "__main__": main()
