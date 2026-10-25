"""Convert a trained adapter (adapter.safetensors, peft-style names) into a GGUF LoRA file that llama.cpp / llama-cpp-python load with `lora_path=` on top of the UNMODIFIED base GGUF.

    python -m training.convert_lora_gguf training/out/qwen25_05b --out models/ir-lora-qwen25-05b.gguf

The adapter is shipped as its own small file (a few tens of MB) next to the base model; the base model file is never changed, the adapter can be updated over the air, and removing it restores the original model.
Tensor naming (Qwen2 architecture):  blk.N.attn_q.weight.lora_a / .lora_b, ... ; metadata: general.type=adapter, adapter.type=lora, adapter.lora.alpha."""
import argparse
import json
import os
import re
import sys

import numpy as np

NAMES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output", "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
RX = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight")


def convert(adapter_dir, out, dtype="f16"):
    import gguf
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"))); tens = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
    w = gguf.GGUFWriter(out, arch=cfg.get("architecture", "qwen2"))
    w.add_string("general.type", "adapter"); w.add_string("adapter.type", "lora"); w.add_float32("adapter.lora.alpha", float(cfg["lora_alpha"]))
    n = 0
    for k, v in sorted(tens.items()):
        m = RX.fullmatch(k)
        if not m: raise SystemExit(f"unexpected tensor name {k}")
        layer, proj, ab = m.groups(); name = f"blk.{layer}.{NAMES[proj]}.weight.lora_{ab.lower()}"
        w.add_tensor(name, np.ascontiguousarray(v.astype(np.float16 if dtype == "f16" else np.float32))); n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return n, os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir"); ap.add_argument("--out", required=True); ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    a = ap.parse_args(); n, size = convert(a.adapter_dir, a.out, a.dtype); print(f"wrote {a.out}: {n} tensors, {size / 1e6:.1f} MB")


if __name__ == "__main__": main()
