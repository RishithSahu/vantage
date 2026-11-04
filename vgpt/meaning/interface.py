"""The learned semantic interface: question (+ glossary hints) -> meaning frame, by one grammar-constrained decode of Qwen2.5-0.5B with the meaning LoRA.

The decoder grammar is generated from the ontology, so the model can only emit valid vocabulary.  Confidence is the lowest probability of any emitted token (a cheap, calibrated-on-data
signal; see docs/semantic_method.md).  This module has NO text rules: it formats the prompt, runs the model, parses the six words."""
import os
import re
import time

from . import glossary as _gl
from . import literals as _lit
from . import ontology as _on
from .frame import Frame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
PREFIX = {"frame": "Q: ", "term": "Term: ", "rel": "Pair: "}


VEHICLE_LINES = {"ev": "This vehicle is electric: a traction battery and a motor; it has no engine, fuel, gears or exhaust.",
                 "petrol": "This vehicle is petrol: a combustion engine, fuel tank, gears and exhaust; the only battery is the 12 V one.",
                 "diesel": "This vehicle is diesel: a combustion engine with turbo, particulate filter and AdBlue, a fuel tank and gears; the only battery is the 12 V one.",
                 "ice": "This vehicle has a combustion engine, a fuel tank and gears; the only battery is the 12 V one.",
                 "hybrid": "This vehicle is a hybrid: a combustion engine and a traction battery."}


def prompt_text(kind, text, hint="", prev=None, version=None, vehicle=None):
    """The exact ChatML string the model is trained on and run with (answer excluded).  Frame version 2 adds the line `Previous: <frame of the last answered question>` when there is one."""
    if kind == "frame": text = _lit.mask_calendar(text)
    onto = _on.load(version); user = PREFIX[kind] + text + (("\n" + hint) if hint else "") + (("\nPrevious: " + prev) if prev and onto.version >= 2 and kind == "frame" else "")
    system = onto.system_prompt() + (("\n" + VEHICLE_LINES[vehicle]) if vehicle in VEHICLE_LINES and kind == "frame" and onto.version >= 2 else "")      # the car type sits in the system prompt: the cached prefix, no cost per question
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


class TokenMask:
    """The same constraint as the generated grammar, applied as a token mask instead of through llama.cpp's grammar sampler.

    The frame is `word word ... word` with every slot a closed list of words (`Ontology.words()`).  Each word is turned into the token sequence the model was trained on (the first word as it is,
    the others with the space that precedes them), the sequences of a slot form a trie, and at every step the only tokens left open are those that continue a word of the current slot, begin a word
    of the next slot after a finished word, or - after the last word - end the answer.  Several words can be alive at once (one word a prefix of another), so a SET of states is followed.
    Why: llama.cpp's grammar sampler walks the whole 151,000-token vocabulary at every step; measured here it is 62 % of the time of one parse (913 ms with it, 351 ms without) and 1.9 s on the
    Raspberry Pi.  The mask costs one vector operation per step.  The decisions are the same as the grammar's (tests/test_frame_v2.py compares them on the frozen questions)."""
    def __init__(self, llm, words_by_slot):
        self.eos = llm.token_eos(); self.slots = []; self._ids = {}
        for i, words in enumerate(words_by_slot):
            root = {}
            for w in words:
                node = root
                for t in llm.tokenize((("" if i == 0 else " ") + w).encode("utf-8"), add_bos=False, special=False): node = node.setdefault(t, {})
                node[None] = True                                                                                    # a word may end here
            self.slots.append(root)
        self.last = len(self.slots) - 1
        self.max_tokens = sum(self._depth(root) for root in self.slots) + 1                                   # the longest valid frame, in tokens, and the end token: decoding never runs longer

    @staticmethod
    def _depth(node): return max([1 + TokenMask._depth(v) for k, v in node.items() if k is not None] or [0])

    def start(self): return [(0, self.slots[0])]

    def advance(self, states, t):
        out = []
        for s, n in states:
            if t in n: out.append((s, n[t]))
            if None in n and s < self.last and t in self.slots[s + 1]: out.append((s + 1, self.slots[s + 1][t]))
        return out

    def open_ids(self, states):
        key = tuple(sorted({(s, id(n)) for s, n in states}))
        if key not in self._ids:
            ids = set()
            for s, n in states:
                ids.update(k for k in n if k is not None)
                if None in n: ids.add(self.eos) if s == self.last else ids.update(self.slots[s + 1])
            import numpy as np
            self._ids[key] = np.array(sorted(ids), dtype=np.int64)
        return self._ids[key]

    def processor(self, steps=None, state=None):
        """The logits processor: the mask, and (with `steps`) the confidence of each step.  The confidence is the probability the model gives the token it takes, read over the tokens the decoder may allow at that step - not over the whole
        151,000-token vocabulary - which is the measure the ONNX Runtime backend uses too (they differ by 0.0006 on average).  When no token is open (a state that should never be reached) the processor fails closed: only the end token
        stays open and `state["dead"]` is set, so the answer is no frame, never a frame made of whatever the logits liked."""
        import numpy as np
        mask = self; st = state if state is not None else {}; st.update(states=None, dead=False)

        def constrain(input_ids, scores):
            if st["states"] is None: st["states"] = mask.start()
            else: st["states"] = mask.advance(st["states"], int(input_ids[-1]))
            ids = mask.open_ids(st["states"]); out = np.full_like(scores, -np.inf)
            if not len(ids):
                st["dead"] = True; out[mask.eos] = 0.0; return out
            keep = scores[ids]
            if steps is not None: steps.append(1.0 / float(np.exp(keep - float(keep.max())).sum()))
            out[ids] = keep; return out
        return constrain


class FrameModel:
    def __init__(self, adapter, base=BASE, threads=2, n_ctx=1024, n_batch=128, use_hints=True, log=False, version=None):
        from llama_cpp import Llama, LlamaGrammar
        self.version = int(version or _on.active_version()); self.onto = _on.load(self.version); self.glossary = _gl.load(); self.use_hints = use_hints; self.vehicle = None; self.vehicle_hint = os.environ.get("VGPT_VEHICLE_HINT", "0") == "1"
        t = time.perf_counter()
        kw = dict(model_path=base, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads, n_batch=n_batch, n_ubatch=n_batch, use_mmap=True, verbose=log)
        if adapter: kw["lora_path"] = adapter
        self.llm = Llama(**kw)
        self._gbnf = self.onto.gbnf(); self.grammar = LlamaGrammar.from_string(self._gbnf, verbose=False)
        self.decoder = os.environ.get("VGPT_DECODER", "mask"); self.mask = None; self._build_mask()                       # "mask" (default) or "grammar" (llama.cpp's grammar sampler, the reference)
        self.load_s = time.perf_counter() - t

    def _build_mask(self):
        w = self.onto.words(); self.mask = TokenMask(self.llm, [w[k] for k in self.onto.fields]) if self.decoder == "mask" else None

    def set_vehicle(self, kind):
        """Tell the model what kind of vehicle it reads for (ev / petrol / diesel / hybrid / ice / None). Only an adapter trained with the line (VGPT_VEHICLE_HINT=1) is shown it."""
        self.vehicle = kind if (self.vehicle_hint and (kind in _gl.COMBUSTION or not _gl.ice_line_only())) else None

    def refresh(self):
        """The vehicle changed (another dataset was opened): rebuild the decoding grammar from the registered concepts, so that the vocabulary the model may emit is the trained one plus THIS vehicle's own names."""
        from llama_cpp import LlamaGrammar
        self.onto = _on.load(self.version); g = self.onto.gbnf()
        if g != self._gbnf: self._gbnf = g; self.grammar = LlamaGrammar.from_string(g, verbose=False); self._build_mask()

    def parse(self, text, hint=None, prev=None):
        """-> (Frame | None, info).  info: raw, pmin (lowest probability of a token taken, over the tokens the decoder may allow), ms, hint.  `prev`: the frame text of the last answered question (version 2 reads it; version 1 ignores it).

        With the token mask (the default) there is ONE decode, bounded by the longest valid frame; a decode that does not end in a frame fails closed (no frame, `info["decode_error"]`), it is not repeated with the grammar sampler.
        VGPT_DECODER=grammar selects llama.cpp's grammar sampler (the reference the mask is tested against), whose confidence is read over the whole vocabulary."""
        import numpy as np
        from llama_cpp import LogitsProcessorList
        hint = (("" if _gl.ice_only() and self.vehicle not in _gl.COMBUSTION else self.glossary.hint_line(text, vehicle=self.vehicle)) if self.use_hints else "") if hint is None else hint        # VGPT_ICE_ONLY=1: electric and unknown cars get no hint line (shorter prompt; 2026-10-26 audit)
        steps = []; state = {}
        t = time.perf_counter(); prompt = prompt_text("frame", text, hint, prev, self.version, self.vehicle); budget = 24 if self.version < 2 else 40
        if self.mask is not None:
            budget = min(budget, self.mask.max_tokens)
            out = self.llm.create_completion(prompt, max_tokens=budget, temperature=0.0, logits_processor=LogitsProcessorList([self.mask.processor(steps, state)]))
        else:
            def watch(input_ids, scores):
                m = float(np.max(scores)); z = float(np.sum(np.exp(scores - m))); steps.append(1.0 / z); return scores
            out = self.llm.create_completion(prompt, grammar=self.grammar, max_tokens=budget, temperature=0.0, logits_processor=LogitsProcessorList([watch]))
        raw = out["choices"][0]["text"].strip()
        info = {"raw": raw, "pmin": min(steps) if steps else None, "ms": (time.perf_counter() - t) * 1000, "hint": hint, "tokens": len(steps)}
        if state.get("dead"): info["decode_error"] = "no token was open"; self.mask_misses = getattr(self, "mask_misses", 0) + 1; return None, info
        try: return Frame.from_text(raw), info
        except ValueError:
            if self.mask is not None: info["decode_error"] = "not a frame"; self.mask_misses = getattr(self, "mask_misses", 0) + 1
            return None, info
