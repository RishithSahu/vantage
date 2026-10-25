"""Prototype: the product's frame model on ONNX Runtime int8 (a drop-in for FrameModel.parse), to measure what a blocked int8 kernel gives on a CPU without dot-product instructions.

    taskset -c 0,1 python bench/ort_frame.py models/ort_frame2d [N]

Runs the first N acceptance questions through (a) the product's llama.cpp FrameModel and (b) this runtime, and prints speed and whether the frames are identical."""
import os
import statistics as st
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from vgpt.meaning.frame import Frame                                       # noqa: E402
from vgpt.meaning.interface import FrameModel, TokenMask, prompt_text      # noqa: E402

L, NKV, HD = 24, 2, 64


class DraftTable:
    """Which word usually follows which, per slot, counted over the TRAINING frames (never the evaluation sets).  Used only to guess ahead; the model checks every guess, so the output is exactly what plain decoding gives."""
    def __init__(self, path):
        import json, collections
        self.t1 = collections.defaultdict(collections.Counter); self.t2 = collections.defaultdict(collections.Counter); n = 0
        for ln in open(path, encoding="utf-8"):
            j = json.loads(ln)
            if j.get("kind") != "frame": continue
            w = j["answer"].split()
            if len(w) != 8: continue
            n += 1
            for i in range(1, 8): self.t1[(i, w[i - 1])][w[i]] += 1; self.t2[(i, w[i - 2] if i > 1 else "", w[i - 1])][w[i]] += 1
        self.rows = n

    def best(self, i, prev2, prev, partial="", thr=0.5):
        """-> most likely full word for slot i given the two words before it (and starting with `partial`), or None when it is not clear enough."""
        for c in (self.t2.get((i, prev2, prev)), self.t1.get((i, prev))):
            if not c: continue
            tot = sum(c.values()); cand = [(n, w) for w, n in c.items() if w.startswith(partial)]
            if cand:
                n, w = max(cand)
                if n / tot >= thr: return w
                return None
        return None


class OrtFrameModel(FrameModel):
    """FrameModel with the language model run by ONNX Runtime (int8 weights, per-token int8 activations, KV cache, cached system prefix)."""
    def __init__(self, model_dir, base=None, threads=2, version=None, use_hints=True):
        import onnxruntime as ort
        from llama_cpp import Llama
        from vgpt.meaning import glossary as _gl, ontology as _on
        self.version = int(version or _on.active_version()); self.onto = _on.load(self.version); self.glossary = _gl.load(); self.use_hints = use_hints
        t = time.perf_counter(); so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.llm = Llama(model_path=base or os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), vocab_only=True, verbose=False)          # the tokenizer only
        self.decoder = "mask"; self.mask = None; self.mask_misses = 0; self._build_mask(); self._cands(); self.base = None; self.load_s = time.perf_counter() - t
        self.stats = {}; self.table = DraftTable(os.path.join(ROOT, "training", "meaning", "data", "frame2", "train.jsonl")) if os.environ.get("VGPT_DRAFT", "table") == "table" else None; self._path = {}

    def _cands(self):
        """Every token the decoder can ever allow (all words of all slots, and the end token): the only rows of the output head that are computed.  self.cmap: token id -> row."""
        seen = {int(self.llm.token_eos())}
        def walk(node):
            for k, v in node.items():
                if k is not None: seen.add(int(k)); walk(v)
        for root in self.mask.slots: walk(root)
        self.cand = np.array(sorted(seen), np.int64); self.cpos = {int(t): i for i, t in enumerate(self.cand)}

    def refresh(self):
        from vgpt.meaning import ontology as _on
        self.onto = _on.load(self.version); self._build_mask(); self._cands(); self.base = None

    def _empty(self): return [np.zeros((1, NKV, 0, HD), np.float32) for _ in range(2 * L)]

    def _fwd(self, ids, start, kv, nkeep=1):
        feed = {"ids": np.asarray(ids, np.int64)[None], "pos": np.arange(start, start + len(ids), dtype=np.int64)[None], "nkeep": np.array([nkeep], np.int64), "cand": self.cand}
        for l in range(L): feed[f"pk{l}"] = kv[2 * l]; feed[f"pv{l}"] = kv[2 * l + 1]
        out = self.sess.run(None, feed); return out[-1][0], out[:-1]

    def _prefix(self):
        """KV of the system prompt, computed once (it is the same for every question)."""
        head = prompt_text("frame", "x", "", None, self.version).split("<|im_start|>user")[0]
        ids = self.llm.tokenize(head.encode(), add_bos=False, special=True); _, kv = self._fwd(ids, 0, self._empty()); self.base = (ids, kv)

    def _word_path(self, i, w):
        key = (i, w)
        if key not in self._path: self._path[key] = [int(t) for t in self.llm.tokenize((("" if i == 0 else " ") + w).encode(), add_bos=False, special=False)]
        return self._path[key]

    def _draft(self, states, k, gen=None):
        """A guess of the next tokens (only a guess: the model checks every one).  A token the decoder leaves no other choice for is certain; at a real choice the usual word of the slot (table) is tried."""
        if self.table is None or gen is None: return self._draft_none(states, k)
        out = []; st = states; g = list(gen)
        while len(out) < k:
            ids = self.mask.open_ids(st)
            if not len(ids): break
            tok = int(ids[0]) if len(ids) == 1 else self._guess(g, set(ids.tolist()))
            if tok is None: break
            out.append(tok); st = self.mask.advance(st, tok); g.append(tok)
        return out

    def _guess(self, g, idset):
        text = self.llm.detokenize(g).decode("utf-8", "replace"); parts = text.split(" "); i = len(parts) - 1; eos = int(self.llm.token_eos())
        if i == 0: return None
        ws = max((j for j, t in enumerate(g) if self.llm.detokenize([t]).startswith(b" ")), default=0); done = g[ws:]; partial = parts[-1]; prev = parts[-2]; prev2 = parts[-3] if i >= 2 else ""
        w = self.table.best(i, prev2, prev, partial)
        if w is not None:
            path = self._word_path(i, w)
            if path[:len(done)] == done and len(path) > len(done) and path[len(done)] in idset: return path[len(done)]
        if i >= 7: return eos if eos in idset else None
        w = self.table.best(i + 1, prev, partial)                                                      # the word may be finished: the usual word of the next slot
        if w is not None:
            t = self._word_path(i + 1, w)[0]
            if t in idset: return t
        return None

    def _draft_none(self, states, k):
        out = []; none = [self.llm.tokenize(b" none", add_bos=False, special=False)[0], self.llm.tokenize(b"none", add_bos=False, special=False)[0]]
        for _ in range(k):
            ids = self.mask.open_ids(states)
            if not len(ids): break
            tok = int(ids[0]) if len(ids) == 1 else next((t for t in none if t in set(ids.tolist())), None)
            if tok is None: break
            out.append(tok); states = self.mask.advance(states, tok)
        return out

    def parse(self, text, hint=None, prev=None):
        hint = (self.glossary.hint_line(text) if self.use_hints else "") if hint is None else hint
        t0 = time.perf_counter(); prompt = prompt_text("frame", text, hint, prev, self.version); budget = 24 if self.version < 2 else 40; spec = os.environ.get("VGPT_SPEC", "1") == "1"
        if self.base is None: self._prefix()
        bids, bkv = self.base; ids = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
        assert ids[:len(bids)] == bids, "prompt does not start with the cached system prefix"
        rest = ids[len(bids):]; t1 = time.perf_counter(); logits, kv = self._fwd(rest, len(bids), bkv, 1); t2 = time.perf_counter()
        cur = logits[-1].astype(np.float64); pos = len(ids); eos = self.llm.token_eos(); states = self.mask.start(); gen = []; steps = []; passes = 0; tdec = 0.0

        def pick(lg, st):
            o = self.mask.open_ids(st)
            if not len(o): return int(self.cand[int(lg.argmax())])
            r = np.fromiter((self.cpos[int(t)] for t in o), np.int64, len(o)); return int(o[int(np.argmax(lg[r]))])

        def conf(lg): return 1.0 / float(np.exp(lg - lg.max()).sum())
        done = False
        while not done and len(gen) < budget:
            tok = pick(cur, states); steps.append(conf(cur))
            if tok == eos: break
            gen.append(tok); states = self.mask.advance(states, tok)
            drafts = self._draft(states, min(8, budget - len(gen)), gen) if spec else []
            if os.environ.get("DEBUG_DRAFT"): print("   gen=%r drafts=%r" % (self.llm.detokenize(gen).decode(), self.llm.detokenize(drafts).decode() if drafts else ""))
            td = time.perf_counter(); out, kvn = self._fwd([tok] + drafts, pos, kv, 1 + len(drafts)); tdec += time.perf_counter() - td; passes += 1
            out = out.astype(np.float64); cur = out[0]; m = 0
            for j, d in enumerate(drafts):
                if pick(cur, states) != d: break
                steps.append(conf(cur)); gen.append(d); states = self.mask.advance(states, d); m += 1; cur = out[j + 1]
                if d == eos: done = True; break
            keep = pos + 1 + m; kv = [np.ascontiguousarray(a[:, :, :keep]) for a in kvn]; pos = keep
            if gen and gen[-1] == eos: gen.pop(); done = True
        raw = self.llm.detokenize(gen).decode("utf-8", "replace").strip(); self.stats = {"prefill_ms": (t2 - t1) * 1000, "decode_ms": tdec * 1000, "tail_tokens": len(rest), "out_tokens": len(gen), "passes": passes}
        info = {"raw": raw, "pmin": min(steps) if steps else None, "ms": (time.perf_counter() - t0) * 1000, "hint": hint, "tokens": len(steps)}
        try: return Frame.from_text(raw), info
        except ValueError: return None, info


def main():
    import vehiclegpt; vehiclegpt._adopt_product_model()
    from vgpt.vehicle import lake
    lake.adopt_path(os.path.join(ROOT, "data", "world_aster.db"))
    mdir = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20; qs = []
    for ln in open(os.path.join(ROOT, "bench", "acceptance_v1.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if not ln.startswith("#") and len(p) > 2: qs.append(p[2])
    qs = qs[:n]; ver = int(os.environ.get("VGPT_FRAME", 2)); ref = FrameModel(os.environ["VGPT_MEANING_ADAPTER"], threads=2, version=ver); ref.parse("how fast am I going")
    rr = []; tr = []
    for q in qs:
        t = time.perf_counter(); f, i = ref.parse(q); tr.append(time.perf_counter() - t); rr.append(i["raw"])
    print("llama.cpp  median %.2fs  mean %.2fs" % (st.median(tr), st.mean(tr)), flush=True)
    del ref
    m = OrtFrameModel(mdir, version=ver); print("ort load %.1fs" % m.load_s, flush=True); t = time.perf_counter(); m._prefix(); print("system prefix %.1fs" % (time.perf_counter() - t), flush=True); m.parse("how fast am I going")
    to = []; same = 0; pf = []; dc = []; tk = []; pm = []
    for q, want in zip(qs, rr):
        t = time.perf_counter(); f, i = m.parse(q); to.append(time.perf_counter() - t); pf.append(m.stats["prefill_ms"]); dc.append(m.stats["decode_ms"]); tk.append((m.stats["tail_tokens"], m.stats["out_tokens"], m.stats["passes"])); same += i["raw"] == want; pm.append(i["pmin"])
        if i["raw"] != want: print("DIFF", q, "|", want, "|", i["raw"])
    import json
    if os.environ.get("DUMP"): json.dump(pm, open(os.environ["DUMP"], "w"))
    if os.environ.get("CMP"):
        ref_pm = json.load(open(os.environ["CMP"])); d = [abs(a - b) for a, b in zip(ref_pm, pm)]; print("pmin vs full-vocabulary pmin: mean abs diff %.5f  max %.5f" % (st.mean(d), max(d)))
    print("ort        median %.2fs  mean %.2fs  | prefill %.0f ms (%.0f tokens)  decode %.0f ms (%.1f tokens in %.1f passes)  | same frame as llama.cpp: %d/%d" % (st.median(to), st.mean(to), st.mean(pf), st.mean(a for a, _, _ in tk), st.mean(dc), st.mean(b for _, b, _ in tk), st.mean(c for _, _, c in tk), same, len(qs)))


if __name__ == "__main__": main()
