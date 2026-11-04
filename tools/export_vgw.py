"""Exports the product model (Qwen2.5-0.5B Q4_K_M base + LoRA merged, int8 per output channel) to the weight file of the native engine (engine/vg_engine.c).

    python tools/export_vgw.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/meaning-lora-frame2r12.gguf models/vg_frame2r12.vgw

The integers are those of tools/build_ort_model.py (same loader, same per-channel scales, same clip to +-127), so the engine and the ONNX build multiply the same numbers.  Weights are written already packed in the order the kernel reads them:
[output rows / 4][input / 16][4 rows][16 inputs].  The q, k, v matrices are one matrix (1152 rows), gate and up are one (9728 rows).  Layout of the file: 64-byte header, then every array preceded by a 64-byte block whose first 8 bytes are its length."""
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ort_model as B

H, L, FF, ROPE_N = 896, 24, 4864, 4096

def qrows(w):
    """w [out, in] float -> (int8 [out, in], float32 scale [out]); the same rounding as build_ort_model.qint (which returns it transposed)."""
    s = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)

def pack(q):
    N, K = q.shape; assert N % 4 == 0 and K % 16 == 0
    return np.ascontiguousarray(q.reshape(N // 4, 4, K // 16, 16).transpose(0, 2, 1, 3))

def main(base, lora, out):
    W = B.load_weights(base, lora); f = open(out, "wb"); f.write(b"VGW1" + struct.pack("<8i", L, H, FF, ROPE_N, 0, 0, 0, 0)); f.write(b"\0" * (64 - 4 - 32))
    def put(a):
        a = np.ascontiguousarray(a); f.write(struct.pack("<Q", a.nbytes)); f.write(b"\0" * 56); f.write(a.tobytes()); f.write(b"\0" * ((-a.nbytes) % 64))
    emb = W["token_embd.weight"]; eq, es = qrows(emb); put(eq); put(es); put(W["output_norm.weight"].astype(np.float32))
    inv = 1.0 / (B.THETA ** (np.arange(0, B.HD, 2, dtype=np.float64) / B.HD)); ang = np.outer(np.arange(ROPE_N), inv); e2 = np.concatenate([ang, ang], axis=1); put(np.cos(e2).astype(np.float32)); put(np.sin(e2).astype(np.float32))
    for l in range(L):
        p = f"blk.{l}."; put(W[p + "attn_norm.weight"].astype(np.float32))
        wqkv = np.concatenate([W[p + "attn_" + n + ".weight"] for n in "qkv"], axis=0); bq = np.concatenate([W[p + "attn_" + n + ".bias"] for n in "qkv"]).astype(np.float32)
        q, s = qrows(wqkv); put(pack(q)); put(s); put(bq)
        q, s = qrows(W[p + "attn_output.weight"]); put(pack(q)); put(s)
        put(W[p + "ffn_norm.weight"].astype(np.float32))
        q, s = qrows(np.concatenate([W[p + "ffn_gate.weight"], W[p + "ffn_up.weight"]], axis=0)); put(pack(q)); put(s)
        q, s = qrows(W[p + "ffn_down.weight"]); put(pack(q)); put(s)
        print("layer", l, flush=True)
    f.close(); print("written", out, os.path.getsize(out) // 2 ** 20, "MB")

if __name__ == "__main__": main(*sys.argv[1:4])
