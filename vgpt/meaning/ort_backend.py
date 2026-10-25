"""The frame model on ONNX Runtime: the same Qwen2.5-0.5B + meaning LoRA, merged and stored as int8 (tools/build_ort_model.py), run by a blocked int8 matrix kernel.

Why: on a CPU without dot-product instructions (the Raspberry Pi 4's Cortex-A72) llama.cpp's quantised kernels reach about 11 prompt tokens/s; ONNX Runtime's int8 kernel reaches about 30.  Measured on the Pi (2 cores): 90.9 % exact frames on 220 frozen-set questions, the
same as llama.cpp; 218 of the 220 frames are identical; median parse 1.4 s against 3.2 s, worst 2.7 s against 8.1 s.

What is exact and what is not.  The decoding is the same masked greedy decoding as `FrameModel`.  Guessing ahead (`DraftTable`: the usual word of each slot, counted over the TRAINING frames) only decides how many tokens are checked in one pass;
every token is chosen by the model's own masked argmax, so the guesses cannot change the frame.  The int8 storage can (2 of 220 frames differ, both close calls).  The confidence (`pmin`) is read over the tokens the decoder can ever allow, not over the
whole vocabulary of 151,936 (mean difference 0.0006, largest 0.004 on 20 questions)."""
import json
import os
import sys
import time

import numpy as np

from .frame import Frame
from .interface import FrameModel, prompt_text

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
L, NKV, HD = 24, 2, 64
DRAFT_MAX = int(os.environ.get("VGPT_DRAFT_MAX", "8"))                                                 # most tokens guessed ahead in one pass (each guessed row costs ~30 ms on the Pi; a saved pass ~240 ms)
THRESHOLD = float(os.environ.get("VGPT_DRAFT_THR", "0.5"))                                       # a word is guessed ahead when it followed the same words in at least this share of the training frames
ALT_BELOW = 0.9                                                                                   # below this confidence the runner-up of the least certain step is kept, to ask "did you mean ..."


def spin_default(threads):
    """Whether the worker threads of ONNX Runtime busy-wait between operators: VGPT_ORT_SPIN=1 / 0 decides; otherwise only when the machine has cores to spare beyond the model's own (at least twice its threads plus two).  Spinning
    shortens a parse on a big machine and steals the very cores the ingest, the camera and the interface need on a two- or four-core board."""
    v = os.environ.get("VGPT_ORT_SPIN")
    if v in ("0", "1"): return v
    try: cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError): cores = os.cpu_count() or 1
    return "1" if cores >= 2 * int(threads) + 2 else "0"


class DraftTable:
    """Which word usually follows which, per slot, counted over the TRAINING frames (never the evaluation sets); stored in `draft.json` next to the model."""
    def __init__(self, path):
        j = json.load(open(path, encoding="utf-8")); self.t1 = {tuple(k.split("|", 1)): v for k, v in j["t1"].items()}; self.t2 = {tuple(k.split("|", 2)): v for k, v in j["t2"].items()}

    @staticmethod
    def build(train_jsonl, out_path, keep=4):
        import collections
        t1 = collections.defaultdict(collections.Counter); t2 = collections.defaultdict(collections.Counter)
        for ln in open(train_jsonl, encoding="utf-8"):
            j = json.loads(ln)
            if j.get("kind") != "frame": continue
            w = j["answer"].split()
            if len(w) != 8: continue
            for i in range(1, 8): t1[f"{i}|{w[i - 1]}"][w[i]] += 1; t2[f"{i}|{w[i - 2] if i > 1 else ''}|{w[i - 1]}"][w[i]] += 1
        top = lambda c: dict(c.most_common(keep))
        json.dump({"t1": {k: top(v) for k, v in t1.items()}, "t2": {k: top(v) for k, v in t2.items()}}, open(out_path, "w", encoding="utf-8"), separators=(",", ":"))

    def best(self, i, prev2, prev, partial="", thr=None):
        thr = THRESHOLD if thr is None else thr
        """-> the most likely full word for slot i given the two words before it (starting with `partial`), or None when it is not clear enough."""
        for c in (self.t2.get((str(i), prev2, prev)), self.t1.get((str(i), prev))):
            if not c: continue
            tot = sum(c.values()); cand = [(n, w) for w, n in c.items() if w.startswith(partial)]
            if cand:
                n, w = max(cand)
                return w if n / tot >= thr else None
        return None


class OrtFrameModel(FrameModel):
    """FrameModel with the language model run by ONNX Runtime (int8 weights, per-token int8 activations, KV cache, cached system prefix)."""
    def __init__(self, model_dir, base=None, threads=2, version=None, use_hints=True, provider="cpu"):
        import onnxruntime as ort
        from llama_cpp import Llama
        from vgpt.meaning import glossary as _gl, ontology as _on
        self.version = int(version or _on.active_version()); self.onto = _on.load(self.version); self.glossary = _gl.load(); self.use_hints = use_hints
        t = time.perf_counter(); so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.add_session_config_entry("session.intra_op.allow_spinning", spin_default(threads))
        avail = ort.get_available_providers(); prov = [p for p in (("CUDAExecutionProvider", "DmlExecutionProvider") if provider == "gpu" else ()) if p in avail] + ["CPUExecutionProvider"]
        self.provider = prov[0]; self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=prov)
        self.llm = Llama(model_path=base or os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), vocab_only=True, verbose=False)          # the tokenizer only
        self.decoder = "mask"; self.mask = None; self.mask_misses = 0; self._build_mask(); self._cands(); self.base = None; self.load_s = time.perf_counter() - t
        self._cue = {}; self.cues = None; self._build_cues()
        self.vehicle = None; self.vehicle_hint = os.environ.get("VGPT_VEHICLE_HINT", "0") == "1"; self.stats = {}; self.table = DraftTable(os.path.join(model_dir, "draft.json")) if os.path.exists(os.path.join(model_dir, "draft.json")) and os.environ.get("VGPT_DRAFT", "table") == "table" else None; self._path = {}

    def _cands(self):
        """Every token the decoder can ever allow (all words of all slots, and the end token): the only rows of the output head that are computed.  self.cmap: token id -> row."""
        seen = {int(self.llm.token_eos())}
        def walk(node):
            for k, v in node.items():
                if k is not None: seen.add(int(k)); walk(v)
        for root in self.mask.slots: walk(root)
        self.cand = np.array(sorted(seen), np.int64); self.cpos = {int(t): i for i, t in enumerate(self.cand)}

    def _build_cues(self):
        from .cues import Cues
        self.cues = Cues(self.onto, self.glossary) if os.environ.get("VGPT_CUES", "1") == "1" else None; self.mask_words0 = set(self.onto.words()[self.onto.fields[0]])

    def refresh(self):
        from vgpt.meaning import ontology as _on
        self.onto = _on.load(self.version); self._build_mask(); self._cands(); self._build_cues()
        if self.base is not None:                                                                    # another vehicle: the decoding words change, the system prompt usually does not - then its processed state stays (9 s on the Pi to redo)
            if self.llm.tokenize(self._head().encode(), add_bos=False, special=True) != self.base[0]: self.base = None

    def set_vehicle(self, kind):
        """As FrameModel.set_vehicle; the processed system prompt is recomputed once when the car type changes, never per question."""
        kind = kind if self.vehicle_hint else None
        if kind != self.vehicle: self.vehicle = kind; self.base = None

    def _empty(self): return [np.zeros((1, NKV, 0, HD), np.float32) for _ in range(2 * L)]

    def _fwd(self, ids, start, kv, nkeep=1):
        feed = {"ids": np.asarray(ids, np.int64)[None], "pos": np.arange(start, start + len(ids), dtype=np.int64)[None], "nkeep": np.array([nkeep], np.int64), "cand": self.cand}
        for l in range(L): feed[f"pk{l}"] = kv[2 * l]; feed[f"pv{l}"] = kv[2 * l + 1]
        out = self.sess.run(None, feed); return out[-1][0], out[:-1]

    def _prefix(self):
        """KV of the system prompt, computed once (it is the same for every question)."""
        ids = self.llm.tokenize(self._head().encode(), add_bos=False, special=True); _, kv = self._fwd(ids, 0, self._empty()); self.base = (ids, kv)

    def _head(self):
        """The part of every prompt that never changes: the system prompt and the opening of the user turn (`<|im_start|>user` + newline + `Q:`)."""
        full = prompt_text("frame", "x", "", None, self.version, self.vehicle); return full[:full.index("Q: x") + 2]

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
        ws = max((j for j, t in enumerate(g) if self.llm.detokenize([t]).startswith(b" ")), default=0); done = g[ws:]; partial = parts[-1]
        if i == 0:                                                                                    # inside the first word: once it is a whole word, propose the first token of the concept the question names
            if partial not in self.mask_words0: return None
            w = self._cue.get(1) or self.table.best(1, "", partial)                                      # the concept the question names, else the usual concept after this act (for "ambiguous", "unsafe", "social" ...: none)
            if w is None: return None
            tok = self._word_path(1, w)[0]
            return tok if tok in idset else None
        prev = parts[-2]; prev2 = parts[-3] if i >= 2 else ""
        w = self._cue.get(i)                                                                          # a word the QUESTION suggests for this slot (verified token by token like every guess)
        if w is not None and w.startswith(partial):
            path = self._word_path(i, w)
            if path[:len(done)] == done and len(path) > len(done) and path[len(done)] in idset: return path[len(done)]
        w = self.table.best(i, prev2, prev, partial)
        if w is not None:
            path = self._word_path(i, w)
            if path[:len(done)] == done and len(path) > len(done) and path[len(done)] in idset: return path[len(done)]
        if i >= 7: return eos if eos in idset else None
        w = self._cue.get(i + 1) or self.table.best(i + 1, prev, partial)                              # the word may be finished: the usual / suggested word of the next slot
        if w is not None:
            t = self._word_path(i + 1, w)[0]
            if t in idset: return t
        return None

    def _alternative(self, raw, before, tok, alt_tok, alt_p):
        """The two readings the model hesitated between: the word it wrote and the word the runner-up token would have started, in the same slot.  None when the runner-up does not name exactly one word."""
        fields = list(self.onto.words()); text = self.llm.detokenize(before).decode("utf-8", "replace") if before else ""
        starts_word = (not before) or self.llm.detokenize([tok]).startswith(b" ")
        slot = len(text.split(" ")) if before and starts_word else (len(text.split(" ")) - 1 if before else 0)
        if before and not starts_word:
            ws = max((j for j, t in enumerate(before) if self.llm.detokenize([t]).startswith(b" ")), default=0); done = before[ws:] if before[ws:] and self.llm.detokenize(before[ws:ws + 1]).startswith(b" ") else before
        else: done = []
        if slot >= len(fields): return None
        words = self.onto.words()[fields[slot]]

        def reading(first):
            return sorted({w for w in words if (lambda p: p[:len(done)] == done and len(p) > len(done) and p[len(done)] == first)(self._word_path(slot, w))})
        mine, other = reading(tok), reading(alt_tok)
        chosen = raw.split()[slot] if len(raw.split()) > slot else None
        if chosen is None or chosen not in mine or len(other) != 1 or other[0] == chosen: return None
        return {"slot": fields[slot], "index": slot, "chosen": chosen, "other": other[0], "p_other": round(alt_p, 3)}

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
        """The frame of a question.  The previous turn's frame is shown to the model only when the question needs it (vgpt/meaning/prevgate.py); VGPT_PREV=always shows it every time."""
        if prev is None or os.environ.get("VGPT_PREV", "gate") == "always": return self._parse(text, hint, prev)
        from . import prevgate as PG
        spent = 0.0
        if not PG.wants_prev(text):
            f, info = self._parse(text, hint, None); spent = info["ms"]
            if f is not None and not PG.frame_needs_prev(f, info.get("pmin")): info["prev_used"] = False; return f, info
        f, info = self._parse(text, hint, prev); info["ms"] += spent; info["prev_used"] = True; return f, info

    def _parse(self, text, hint=None, prev=None):
        text = str(text).encode("utf-8", "replace").decode("utf-8", "replace")                               # a lone surrogate (a garbled paste) must not crash the request
        hint = (self.glossary.hint_line(text, vehicle=self.vehicle) if self.use_hints else "") if hint is None else hint
        t0 = time.perf_counter(); prompt = prompt_text("frame", text, hint, prev, self.version, self.vehicle); budget = 24 if self.version < 2 else 40; spec = os.environ.get("VGPT_SPEC", "1") == "1"
        self._cue = self.cues.propose(text) if self.cues is not None else {}
        if self.base is None: self._prefix()
        bids, bkv = self.base; ids = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
        if ids[:len(bids)] != bids:                                                                  # a question whose first characters join the `:` into one token: read it without the cached state
            bids, bkv = [], self._empty()
        rest = ids[len(bids):]; t1 = time.perf_counter(); logits, kv = self._fwd(rest, len(bids), bkv, 1); t2 = time.perf_counter()
        cur = logits[-1].astype(np.float64); pos = len(ids); eos = self.llm.token_eos(); states = self.mask.start(); gen = []; steps = []; passes = 0; tdec = 0.0

        def pick(lg, st):
            o = self.mask.open_ids(st)
            if not len(o): return int(self.cand[int(lg.argmax())])
            r = np.fromiter((self.cpos[int(t)] for t in o), np.int64, len(o)); return int(o[int(np.argmax(lg[r]))])

        def conf(lg): return 1.0 / float(np.exp(lg - lg.max()).sum())

        def runner_up(lg, st, tok):
            o = self.mask.open_ids(st)
            if len(o) < 2: return None, 0.0
            r = np.fromiter((self.cpos[int(t)] for t in o), np.int64, len(o)); sc = lg[r]; order = np.argsort(-sc)
            for k in order:
                if int(o[int(k)]) != tok: return int(o[int(k)]), float(np.exp(sc[int(k)] - lg.max()) / np.exp(lg - lg.max()).sum())
            return None, 0.0
        budget = min(budget, self.mask.max_tokens); dead = False                                       # decoding never runs longer than the longest valid frame
        weakest = None                                                                              # (confidence, tokens before the step, token chosen, runner-up token, runner-up probability)
        done = False
        while not done and len(gen) < budget:
            if not len(self.mask.open_ids(states)): dead = True; break                              # no token is open: fail closed, no frame
            tok = pick(cur, states); c_now = conf(cur); steps.append(c_now)
            if tok == eos: break
            if c_now < ALT_BELOW and (weakest is None or c_now < weakest[0]):
                alt_tok, alt_p = runner_up(cur, states, tok)
                if alt_tok is not None: weakest = (c_now, list(gen), tok, alt_tok, alt_p)
            gen.append(tok); states = self.mask.advance(states, tok)
            drafts = self._draft(states, min(DRAFT_MAX, budget - len(gen)), gen) if spec else []
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
        if weakest is not None and info["pmin"] is not None and info["pmin"] < ALT_BELOW and weakest[0] == info["pmin"]:
            try: info["alt"] = self._alternative(raw, *weakest[1:])
            except Exception: pass
        if dead: info["decode_error"] = "no token was open"; return None, info
        try: return Frame.from_text(raw), info
        except ValueError: info["decode_error"] = "not a frame"; return None, info
