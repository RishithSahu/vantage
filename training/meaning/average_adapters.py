"""Weight-space average of two LoRA adapters that share a start (one was trained on from the other): `out = a * first + (1 - a) * second`.

    python training/meaning/average_adapters.py --first training/out/meaning_frame2 --second training/out/meaning_frame2b --out training/out/meaning_frame2m --alpha 0.5

A checkpoint and a model fine-tuned from it lie on one path in weight space; points between them usually keep most of what the fine-tuning added and forget less of what came before (the "WiSE-FT" idea).  Nothing is
trained; the inputs are only read.  Each adapter folder holds adapter.safetensors and adapter_config.json."""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--first", required=True); ap.add_argument("--second", required=True); ap.add_argument("--out", required=True); ap.add_argument("--alpha", type=float, default=0.5); a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    r = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    f, s = load_file(os.path.join(r(a.first), "adapter.safetensors")), load_file(os.path.join(r(a.second), "adapter.safetensors"))
    if set(f) != set(s): raise SystemExit("the two adapters do not have the same tensors")
    out = {k: (a.alpha * f[k].float() + (1 - a.alpha) * s[k].float()).to(f[k].dtype).contiguous() for k in f}
    os.makedirs(r(a.out), exist_ok=True); save_file(out, os.path.join(r(a.out), "adapter.safetensors"), metadata={"format": "pt"}); shutil.copy(os.path.join(r(a.second), "adapter_config.json"), os.path.join(r(a.out), "adapter_config.json"))
    print(f"wrote {r(a.out)}: {len(out)} tensors, {a.alpha:.2f} x first + {1 - a.alpha:.2f} x second")


if __name__ == "__main__":
    main()
