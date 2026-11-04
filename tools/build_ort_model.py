"""Builds an ONNX Runtime int8 copy of the product model (Qwen2.5-0.5B Q4_K_M base + the meaning LoRA merged) for CPUs without dot-product instructions.

    python tools/build_ort_model.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/meaning-lora-frame2d.gguf out_dir

Nothing is trained or changed: the base weights are the GGUF's own (dequantised), the LoRA is added with its own scale (alpha / rank), and every matrix is stored as int8 with one scale per output channel.
Activations are quantised per token at run time, inside the graph.  The whole vocabulary is scored (the output head is the tied embedding matrix)."""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

H, L, NH, NKV, HD, FF, V = 896, 24, 14, 2, 64, 4864, 151936
EPS, THETA = 1e-6, 1000000.0


def load_weights(base_gguf, lora_gguf):
    import gguf
    out = {}
    for t in gguf.GGUFReader(base_gguf).tensors:
        if t.name.startswith(("blk.", "token_embd", "output_norm", "output")):
            out[t.name] = np.asarray(t.data, np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16) else gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
    if lora_gguf:
        lr = gguf.GGUFReader(lora_gguf); alpha = float(lr.fields["adapter.lora.alpha"].contents()); parts = {t.name: np.asarray(t.data, np.float32) for t in lr.tensors}; n = 0
        for name in list(parts):
            if name.endswith(".lora_a"):
                key = name[:-7]; a = parts[name]; b = parts[key + ".lora_b"]; rank = a.shape[0]
                out[key] = out[key] + (alpha / rank) * (b @ a); n += 1
        print("LoRA merged into", n, "matrices, scale", alpha / rank)
    return out


def qnbits(w, block=32, bits=4):
    """w: [N, K] float -> (packed uint8 [N, K/block, block/2], scales float32 [N * K/block], zero points uint8 packed [N * ceil(K/block / 2)]) for the com.microsoft MatMulNBits operator: 4-bit weights, one scale and one zero point per
    block of 32 weights along K, minimum / maximum (asymmetric) rounding.  Post-training only: nothing is trained, the same merged weights as the int8 build are rounded to fewer bits."""
    N, K = w.shape; nb = K // block; assert K % block == 0 and bits == 4
    wb = w.reshape(N, nb, block); lo = wb.min(axis=2); hi = wb.max(axis=2); sc = np.maximum((hi - lo) / 15.0, 1e-8); zp = np.clip(np.rint(-lo / sc), 0, 15)
    q = np.clip(np.rint(wb / sc[:, :, None]) + zp[:, :, None], 0, 15).astype(np.uint8)
    packed = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).astype(np.uint8)                                  # element 2i in the low nibble, 2i+1 in the high one
    zq = zp.astype(np.uint8)
    if nb % 2: zq = np.concatenate([zq, np.zeros((N, 1), np.uint8)], axis=1)
    zpk = (zq[:, 0::2] | (zq[:, 1::2] << 4)).astype(np.uint8).reshape(-1)
    return np.ascontiguousarray(packed), sc.astype(np.float32).reshape(-1), zpk


def qint(w):
    """w: [out, in] float -> (int8 [in, out], float32 scale [out]), symmetric, one scale per output channel."""
    s = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8).T.copy(), s.astype(np.float32)


class G:
    def __init__(self): self.nodes, self.init, self.n = [], [], 0

    def name(self, p="t"): self.n += 1; return f"{p}{self.n}"

    def const(self, arr, p="c"):
        n = self.name(p); self.init.append(numpy_helper.from_array(np.asarray(arr), n)); return n

    def op(self, typ, ins, nout=1, domain=None, **attr):
        outs = [self.name("v") for _ in range(nout)]; self.nodes.append(helper.make_node(typ, ins, outs, domain=domain, **attr)); return outs[0] if nout == 1 else outs

    def linear(self, x, w, bias=None):
        """per-token int8 activation quantisation -> MatMulInteger with int8 weights -> rescale.  (FLOAT=1: plain float matrix multiply, for an accelerator.)"""
        if os.environ.get("FLOAT") == "1":
            y = self.op("MatMul", [x, self.const(np.ascontiguousarray(w.T.astype(np.float32)), "wf")])
            return self.op("Add", [y, self.const(bias.astype(np.float32))]) if bias is not None else y
        if os.environ.get("QBITS") == "4" and w.shape[1] % 32 == 0:                                   # 4-bit weights, float activations: half the bytes a decode pass has to read, no activation quantisation operators
            pk, sc, zpk = qnbits(w); ins = [x, self.const(pk, "w4"), self.const(sc, "s4"), self.const(zpk, "z4"), "", self.const(bias.astype(np.float32))] if bias is not None else [x, self.const(pk, "w4"), self.const(sc, "s4"), self.const(zpk, "z4")]
            return self.op("MatMulNBits", ins, domain="com.microsoft", K=int(w.shape[1]), N=int(w.shape[0]), bits=4, block_size=32, **({"accuracy_level": int(os.environ["QACC"])} if os.environ.get("QACC") else {}))
        wq, ws = qint(w); wq_n, ws_n = self.const(wq, "wq"), self.const(ws, "ws"); zp = self.const(np.array(128, np.uint8), "zp")
        if os.environ.get("QMODE") == "dyn":                                                          # one fused operator, one activation scale for the whole call (not per token)
            return self.op("DynamicQuantizeMatMul", [x, wq_n, ws_n] + ([self.const(np.zeros(len(ws), np.int8), "wz"), self.const(bias.astype(np.float32))] if bias is not None else []), domain="com.microsoft")
        amax = self.op("ReduceMax", [self.op("Abs", [x]), self.const(np.array([-1], np.int64))], keepdims=1)
        s = self.op("Div", [self.op("Max", [amax, self.const(np.array(1e-8, np.float32))]), self.const(np.array(127.0, np.float32))])
        xq = self.quant(x, s)
        y = self.op("Mul", [self.op("Mul", [self.op("Cast", [self.op("MatMulInteger", [xq, wq_n, zp])], to=T.FLOAT), s]), ws_n])
        return self.op("Add", [y, self.const(bias.astype(np.float32))]) if bias is not None else y

    def linear_cat(self, x, parts):
        """Several linears of the same input as ONE matrix multiply (one activation quantisation, one kernel call), split afterwards.  parts: [(weight, bias or None), ...]."""
        w = np.concatenate([a for a, _ in parts], axis=0); b = np.concatenate([bb if bb is not None else np.zeros(len(a), np.float32) for a, bb in parts]) if any(bb is not None for _, bb in parts) else None
        y = self.linear(x, w, b); sizes = [len(a) for a, _ in parts]
        return self.op("Split", [y, self.const(np.array(sizes, np.int64))], nout=len(sizes), axis=-1)

    def quant(self, x, s):
        """x [1,T,K] float, s [1,T,1] -> uint8 with zero point 128, one scale per token (a single fused operator)."""
        flat = self.op("Reshape", [s, self.const(np.array([-1], np.int64))]); zp = self.op("Cast", [self.op("Add", [self.op("Mul", [flat, self.const(np.array(0.0, np.float32))]), self.const(np.array(128.0, np.float32))])], to=T.UINT8)
        return self.op("QuantizeLinear", [x, flat, zp], axis=1)

    def linear_rows(self, x, w, rows):
        """like linear(), but only the output rows listed in the int64 input `rows` are computed."""
        if os.environ.get("FLOAT") == "1":
            return self.op("MatMul", [x, self.op("Gather", [self.const(np.ascontiguousarray(w.T.astype(np.float32)), "wf"), rows], axis=1)])
        wq, ws = qint(w); wq_n, ws_n = self.const(wq, "wq"), self.const(ws, "ws"); zp = self.const(np.array(128, np.uint8), "zp")
        b = self.op("Gather", [wq_n, rows], axis=1); sc = self.op("Gather", [ws_n, rows], axis=0)
        amax = self.op("ReduceMax", [self.op("Abs", [x]), self.const(np.array([-1], np.int64))], keepdims=1)
        s = self.op("Div", [self.op("Max", [amax, self.const(np.array(1e-8, np.float32))]), self.const(np.array(127.0, np.float32))])
        xq = self.quant(x, s)
        return self.op("Mul", [self.op("Mul", [self.op("Cast", [self.op("MatMulInteger", [xq, b, zp])], to=T.FLOAT), s]), sc])

    def head_rows(self, x, wq_n, ws_n, rows):
        """The output head over the listed token rows only, with the head matrix stored as [tokens, hidden]: choosing 3,000 rows reads 3,000 contiguous rows (2.7 MB, about 1 ms) instead of 896 scattered bytes per row of a [hidden, tokens] matrix (46 ms on the
        Raspberry Pi 4, measured, every pass).  The arithmetic is the same as `linear_rows`: the same int8 weights, the same scales, the same per-token activation scale."""
        zp = self.const(np.array(128, np.uint8), "zp"); b = self.op("Transpose", [self.op("Gather", [wq_n, rows], axis=0)], perm=[1, 0]); sc = self.op("Gather", [ws_n, rows], axis=0)
        amax = self.op("ReduceMax", [self.op("Abs", [x]), self.const(np.array([-1], np.int64))], keepdims=1)
        sx = self.op("Div", [self.op("Max", [amax, self.const(np.array(1e-8, np.float32))]), self.const(np.array(127.0, np.float32))]); xq = self.quant(x, sx)
        return self.op("Mul", [self.op("Mul", [self.op("Cast", [self.op("MatMulInteger", [xq, b, zp])], to=T.FLOAT), sx]), sc])

    def rms(self, x, w):
        if os.environ.get("RMSFUSE") == "1":                                                          # one operator instead of six (the per-node cost of a decode pass is most of its time on a small CPU)
            return self.op("SimplifiedLayerNormalization", [x, self.const(w.astype(np.float32))], epsilon=EPS, axis=-1)
        ms = self.op("ReduceMean", [self.op("Mul", [x, x]), self.const(np.array([-1], np.int64))], keepdims=1)
        return self.op("Mul", [self.op("Div", [x, self.op("Sqrt", [self.op("Add", [ms, self.const(np.array(EPS, np.float32))])])]), self.const(w.astype(np.float32))])


def build(W, path):
    g = G(); ids, pos, nkeep = "ids", "pos", "nkeep"; i1 = lambda *v: g.const(np.array(list(v), np.int64))
    emb = W["token_embd.weight"]; es = (np.maximum(np.abs(emb).max(axis=1), 1e-8) / 127.0).astype(np.float32); embq = np.clip(np.rint(emb / es[:, None]), -127, 127).astype(np.int8)
    emb_n, embs_n = g.const(embq, "emb"), g.const(es, "embs")
    x = g.op("Mul", [g.op("Cast", [g.op("Gather", [emb_n, ids], axis=0)], to=T.FLOAT), g.op("Unsqueeze", [g.op("Gather", [embs_n, ids], axis=0), i1(-1)])])
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD)); ang = np.outer(np.arange(4096), inv); e2 = np.concatenate([ang, ang], axis=1)
    cos = g.op("Unsqueeze", [g.op("Gather", [g.const(np.cos(e2).astype(np.float32), "cos"), pos], axis=0), i1(1)]); sin = g.op("Unsqueeze", [g.op("Gather", [g.const(np.sin(e2).astype(np.float32), "sin"), pos], axis=0), i1(1)])

    def rope(t):
        a, b = g.op("Split", [t], nout=2, axis=-1, num_outputs=2); rot = g.op("Concat", [g.op("Neg", [b]), a], axis=-1)
        return g.op("Add", [g.op("Mul", [t, cos]), g.op("Mul", [rot, sin])])
    T_ = g.op("Gather", [g.op("Shape", [ids]), g.const(np.array(1, np.int64))], axis=0); P_ = g.op("Gather", [g.op("Shape", ["pk0"]), g.const(np.array(2, np.int64))], axis=0)
    kidx = g.op("Range", [g.const(np.array(0, np.int64)), g.op("Add", [T_, P_]), g.const(np.array(1, np.int64))])
    qpos = g.op("Unsqueeze", [g.op("Squeeze", [pos, i1(0)]), i1(1)])                                                                       # [T,1]
    allowed = g.op("LessOrEqual", [g.op("Unsqueeze", [kidx, i1(0)]), qpos])                                                                 # [T,S]
    mask = g.op("Unsqueeze", [g.op("Where", [allowed, g.const(np.array(0.0, np.float32)), g.const(np.array(-1e9, np.float32))]), i1(0, 1)])  # [1,1,T,S]
    S_ = g.op("Add", [T_, P_]); mask5 = g.op("Unsqueeze", [mask, i1(2)])                                                                 # [1,1,1,T,S]
    outs = []
    for l in range(L):
        p = f"blk.{l}."; h = g.rms(x, W[p + "attn_norm.weight"])
        q, k, v = g.linear_cat(h, [(W[p + "attn_" + n + ".weight"], W[p + "attn_" + n + ".bias"]) for n in "qkv"])
        heads = lambda t, n: g.op("Transpose", [g.op("Reshape", [t, i1(1, -1, n, HD)])], perm=[0, 2, 1, 3])
        q, k, v = heads(q, NH), heads(k, NKV), heads(v, NKV); q, k = rope(q), rope(k)
        K = g.op("Concat", [f"pk{l}", k], axis=2); Vv = g.op("Concat", [f"pv{l}", v], axis=2); outs += [(K, f"ok{l}"), (Vv, f"ov{l}")]
        # grouped-query attention without copying the keys and values 7 times: the 7 query heads that share a key/value head are stacked next to each other (head h uses key/value head h // 7)
        qg = g.op("Reshape", [q, i1(1, NKV, -1, HD)])                                                                                          # [1,2,7*T,64]
        sc = g.op("MatMul", [qg, g.op("Transpose", [K], perm=[0, 1, 3, 2])])                                                                    # [1,2,7*T,S]
        sc = g.op("Reshape", [sc, g.op("Concat", [i1(1, NKV, NH // NKV), g.op("Unsqueeze", [T_, i1(0)]), g.op("Unsqueeze", [S_, i1(0)])], axis=0)])   # [1,2,7,T,S]
        sc = g.op("Add", [g.op("Mul", [sc, g.const(np.array(1.0 / np.sqrt(HD), np.float32))]), mask5]); sc = g.op("Softmax", [sc], axis=-1)
        sc = g.op("Reshape", [sc, g.op("Concat", [i1(1, NKV, -1), g.op("Unsqueeze", [S_, i1(0)])], axis=0)])                                       # [1,2,7*T,S]
        ctx = g.op("Reshape", [g.op("MatMul", [sc, Vv]), i1(1, NH, -1, HD)])                                                                     # [1,14,T,64]
        ctx = g.op("Reshape", [g.op("Transpose", [ctx], perm=[0, 2, 1, 3]), i1(1, -1, NH * HD)])
        x = g.op("Add", [x, g.linear(ctx, W[p + "attn_output.weight"])])
        h = g.rms(x, W[p + "ffn_norm.weight"]); gate, up = g.linear_cat(h, [(W[p + "ffn_gate.weight"], None), (W[p + "ffn_up.weight"], None)])
        x = g.op("Add", [x, g.linear(g.op("Mul", [g.op("Mul", [gate, g.op("Sigmoid", [gate])]), up]), W[p + "ffn_down.weight"])])
    x = g.rms(x, W["output_norm.weight"])
    start = g.op("Sub", [T_, g.op("Gather", [nkeep, g.const(np.array(0, np.int64))], axis=0)])
    last = g.op("Slice", [x, g.op("Unsqueeze", [start, i1(0)]), g.op("Unsqueeze", [T_, i1(0)]), i1(1)])                                      # the last nkeep positions
    head_w = W.get("output.weight", emb)
    if os.environ.get("HEAD", "rows") == "rows" and os.environ.get("FLOAT") != "1":
        if head_w is emb or np.array_equal(head_w, emb): outs.append((g.head_rows(last, emb_n, embs_n, "cand"), "logits"))                  # a tied head: the embedding matrix itself, stored once
        else:
            hq = np.clip(np.rint(head_w / (np.maximum(np.abs(head_w).max(axis=1), 1e-8) / 127.0)[:, None]), -127, 127).astype(np.int8); hs = (np.maximum(np.abs(head_w).max(axis=1), 1e-8) / 127.0).astype(np.float32)
            outs.append((g.head_rows(last, g.const(hq, "headq"), g.const(hs, "heads"), "cand"), "logits"))
    else: outs.append((g.linear_rows(last, head_w, "cand"), "logits"))                                                      # the output head, over the listed token rows only (the decoder allows a few thousand tokens, not 151,936)
    inputs = [helper.make_tensor_value_info("ids", T.INT64, [1, "T"]), helper.make_tensor_value_info("pos", T.INT64, [1, "T"]), helper.make_tensor_value_info("nkeep", T.INT64, [1]), helper.make_tensor_value_info("cand", T.INT64, ["C"])] + \
             [helper.make_tensor_value_info(f"p{kv}{l}", T.FLOAT, [1, NKV, "P", HD]) for l in range(L) for kv in "kv"]
    nodes = g.nodes + [helper.make_node("Identity", [a], [b]) for a, b in outs]; outputs = [helper.make_tensor_value_info(b, T.FLOAT, None) for _, b in outs]
    model = helper.make_model(helper.make_graph(nodes, "qwen2_int8", inputs, outputs, g.init), opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]); model.ir_version = 9
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(path) + ".data", size_threshold=1024)


if __name__ == "__main__":
    base, lora, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    W = load_weights(base, lora); print({k: v.shape for k, v in list(W.items())[:4]}, len(W)); build(W, os.path.join(out, "model.onnx")); print("written", out)
