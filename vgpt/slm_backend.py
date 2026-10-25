"""A small-language-model backend with an explicit, restorable PROMPT-PREFIX STATE.

Why: llama.cpp reuses the KV cache of a shared prompt prefix automatically for plain transformers, but NOT for hybrid models whose layers keep a recurrent / convolution state (LFM2.x, Granite-4 hybrid ...): the state cannot be rolled back, so the whole prompt would be evaluated again for every question. The static part of the prompt (instructions, definitions, format examples) is therefore evaluated ONCE, the model state is saved (and written to disk), and restored before each question: only the question-dependent suffix is evaluated.

Persisted states are keyed by everything that determines them: the llama.cpp runtime version, the model identity (size + a hash of the first and last megabytes of the file, not just size and mtime), the context and batch settings, the chat template and the exact prefix text. A state built for any other combination is never loaded.

Per-call timing is reported in separate stages so the real bottleneck is visible: grammar preparation, state restoration, suffix PROMPT EVALUATION (prefill) and OUTPUT-TOKEN GENERATION (decode).
It never sees vehicle data: the prefix is instructions, the suffix is the message and the retrieved capability cards."""
import hashlib
import os
import pickle
import time

import llama_cpp
from llama_cpp import Llama, LlamaGrammar
import llama_cpp.llama_chat_format as F

_MARK = "\x00QUESTION\x00"
HYBRID_ARCH = ("lfm2", "granitehybrid", "mamba", "mamba2", "jamba", "falcon-h1", "rwkv6", "rwkv7", "nemotron_h", "plamo2")     # recurrent / convolution state: cannot be rolled back, so the prefix must be restored from a saved state


def model_identity(path):
    """size + SHA-256 of the first and last megabyte: cheap, and changes when the file changes (unlike size + mtime alone)."""
    h = hashlib.sha256(); size = os.path.getsize(path); h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20): f.seek(-(1 << 20), 2); h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


class PrefixLlama:
    """One loaded model, any number of NAMED prompt prefixes (each a saved model state). generate(prefix=name) restores that state and evaluates only the suffix."""
    def __init__(self, model_path, n_ctx=4608, threads=2, n_batch=64, mmap=False, cache_dir=None, max_resident=0):
        t = time.perf_counter()
        self.path = model_path; self.cache_dir = cache_dir; self.n_batch = n_batch
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads, n_batch=n_batch, n_ubatch=n_batch, use_mmap=mmap, verbose=False)
        tok = lambda i: self.llm._model.token_get_text(i)
        self.template = self.llm.metadata["tokenizer.chat_template"]
        self.fmt = F.Jinja2ChatFormatter(template=self.template, eos_token=tok(self.llm.token_eos()), bos_token=tok(self.llm.token_bos()), add_generation_prompt=True)
        self.eos = self.llm.token_eos(); self.stop_ids = {self.eos}
        for name in ("<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>"):                  # the turn terminators of the common chat templates
            try:
                ids = self.llm.tokenize(name.encode(), add_bos=False, special=True)
                if len(ids) == 1: self.stop_ids.add(ids[0])
            except Exception: pass
        self.n_ctx = n_ctx; self.load_s = time.perf_counter() - t; self.prefixes = {}; self._tail = None; self._grammars = {}; self.max_resident = max_resident; self._lru = []
        self.identity = model_identity(model_path)
        self.arch = str(self.llm.metadata.get("general.architecture", "")); self.truncatable = self.arch not in HYBRID_ARCH; self._resident = None     # a plain transformer keeps ONE live KV cache and is rolled back to the prefix length for each question (no second copy of the prefix in memory, no restore)

    # ---- the static prefixes ------------------------------------------------------------------------
    def _render(self, system, user, shots=None, tools=None):
        """`tools` are the model's own tool declarations (a list of JSON strings / dicts): the chat template of the tool-calling models writes them into the system turn in its documented form."""
        msgs = [{"role": "system", "content": system}]
        for u, a in shots or []: msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]            # worked examples as real chat turns: the format the model was trained on
        return self.fmt(messages=msgs + [{"role": "user", "content": user}], tools=tools or None).prompt

    def has_prefix(self, name): return name in self.prefixes

    def set_prefix(self, system, shots=None, name="default", tools=None):
        """Evaluate (or load from disk) the model state after the system message, any format examples and the opening of the user turn. Returns seconds spent."""
        full = self._render(system, _MARK, shots, tools); cut = full.index(_MARK); prefix = full[:cut]
        ids = self.llm.tokenize(prefix.encode(), add_bos=False, special=True); t = time.perf_counter()
        f = self._disk_file(prefix); state = None; from_disk = False
        if f and os.path.exists(f):
            try:
                with open(f, "rb") as fh: state = pickle.load(fh)
                from_disk = True
            except Exception: state = None
        if state is None:
            self.llm.reset(); self.llm.eval(ids); state = self.llm.save_state()
            if f:
                try:
                    os.makedirs(os.path.dirname(f), exist_ok=True)
                    with open(f, "wb") as fh: pickle.dump(state, fh)
                except Exception: pass
        if self.truncatable:                                                                                  # make the live cache hold this prefix; the saved copy stays on disk only
            if from_disk: self.llm.load_state(state)
            self._resident = name
            if f and os.path.exists(f): state = None
        dt = time.perf_counter() - t; self.prefixes[name] = {"state": state, "tokens": len(ids), "text": prefix, "system": system, "shots": list(shots or []), "tools": tools, "seconds": dt, "from_disk": from_disk, "file": f, "pinned": name == "A" or not self.max_resident}
        self._touch(name)
        return dt

    def _touch(self, name):
        if name in self._lru: self._lru.remove(name)
        self._lru.append(name)
        if self.max_resident:
            resident = [n for n in self._lru if not self.prefixes[n]["pinned"] and self.prefixes[n]["state"] is not None]
            while len(resident) > self.max_resident:
                old = resident.pop(0)
                if self.prefixes[old]["file"] and os.path.exists(self.prefixes[old]["file"]): self.prefixes[old]["state"] = None

    def _state_of(self, name):
        pf = self.prefixes[name]
        if pf["state"] is None:
            with open(pf["file"], "rb") as fh: pf["state"] = pickle.load(fh)
        self._touch(name); return pf["state"]

    def cache_key(self, prefix):
        return hashlib.sha256("|".join([llama_cpp.__version__, self.identity, str(self.n_ctx), str(self.n_batch), hashlib.sha256(self.template.encode()).hexdigest()[:16], prefix]).encode()).hexdigest()[:24]

    def _disk_file(self, prefix):
        return os.path.join(self.cache_dir, f"prefix_{self.cache_key(prefix)}.state") if self.cache_dir else None

    # ---- one question -------------------------------------------------------------------------------
    def grammar(self, text):
        g = self._grammars.get(text)
        if g is None:
            if len(self._grammars) > 64: self._grammars.clear()
            g = self._grammars[text] = LlamaGrammar.from_string(text, verbose=False)
        return g

    def suffix_text(self, user, prefix="default", history=None):
        """The text after the saved prefix. `history` = earlier (user, assistant) turns of THIS conversation, rendered by the chat template exactly as if they had been part of the prompt (the saved prefix is the static part and stays valid)."""
        if not history: return self._suffix(user)
        pf = self.prefixes[prefix]; full = self._render(pf["system"], _MARK, pf["shots"] + list(history), pf["tools"])
        if not full.startswith(pf["text"]): raise RuntimeError("the conversation history does not extend the saved prefix")
        return full[len(pf["text"]):].replace(_MARK, user)

    def suffix_ids(self, user, assistant_prefix="", prefix="default", history=None):
        return self.llm.tokenize((self.suffix_text(user, prefix, history) + assistant_prefix).encode(), add_bos=False, special=True)

    def generate(self, user, grammar=None, max_tokens=64, stop_text=("\n",), prefix="default", assistant_prefix="", history=None):
        """-> (text, info). `user` is the text after the static prefix. info separates grammar preparation, state restoration, prompt evaluation (prefill) and output generation (decode), in milliseconds."""
        pf = self.prefixes.get(prefix); assert pf is not None, f"call set_prefix(name={prefix!r}) first"
        t0 = time.perf_counter(); ids = self.suffix_ids(user, assistant_prefix, prefix, history)
        if pf["tokens"] + len(ids) + max_tokens > self.n_ctx:                                                # never truncate silently: a prompt that does not fit is reported, and the caller abstains
            return "", {"ms": (time.perf_counter() - t0) * 1000, "grammar_ms": 0.0, "state_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0, "prompt_tokens": len(ids), "out_tokens": 0, "prefix_tokens": pf["tokens"], "error": "context_overflow"}
        tg = time.perf_counter(); g = self.grammar(grammar) if isinstance(grammar, str) else grammar; t1 = time.perf_counter()
        if self.truncatable and self._resident == prefix: self.llm.n_tokens = pf["tokens"]                       # roll the live cache back to the end of the static prefix
        else:
            self.llm.load_state(self._state_of(prefix)); self._resident = prefix
            if self.truncatable and pf["file"] and os.path.exists(pf["file"]): pf["state"] = None
        t2 = time.perf_counter()
        if len(ids) > 1: self.llm.eval(ids[:-1])                                                              # the suffix prompt, evaluated here so it is timed on its own
        t3 = time.perf_counter(); out = []
        for tok in self.llm.generate(ids[-1:], temp=0.0, grammar=g, reset=False):                              # the last prompt token is evaluated by the first generate step: counted as decode
            if tok in self.stop_ids or len(out) >= max_tokens: break
            out.append(tok)
            if stop_text and any(s in self.llm.detokenize(out).decode(errors="ignore") for s in stop_text): break
        text = self.llm.detokenize(out).decode(errors="ignore")
        for s in stop_text or ():
            if s in text: text = text.split(s)[0]
        t4 = time.perf_counter()
        return text.strip(), {"ms": (t4 - t0) * 1000, "grammar_ms": (t1 - tg) * 1000, "state_ms": (t2 - t1) * 1000, "prefill_ms": (t3 - t2) * 1000, "decode_ms": (t4 - t3) * 1000, "prompt_tokens": len(ids), "out_tokens": len(out), "prefix_tokens": pf["tokens"]}

    def generate_uncached(self, system, user, shots=None, grammar=None, max_tokens=64, stop_text=("\n",), assistant_prefix="", tools=None, history=None):
        """Ordinary inference on the FULL prompt with no saved state (used to check that cached and ordinary inference agree)."""
        full = self._render(system, user, list(shots or []) + list(history or []), tools) + assistant_prefix; ids = self.llm.tokenize(full.encode(), add_bos=False, special=True); g = self.grammar(grammar) if isinstance(grammar, str) else grammar
        self.llm.reset(); self._resident = None; out = []
        for tok in self.llm.generate(ids, temp=0.0, grammar=g, reset=True):
            if tok in self.stop_ids or len(out) >= max_tokens: break
            out.append(tok)
        text = self.llm.detokenize(out).decode(errors="ignore")
        for s in stop_text or ():
            if s in text: text = text.split(s)[0]
        return text.strip()

    def _suffix(self, user):
        """The text of the prompt that follows the saved prefix: the user's message, the end of the user turn and the opening of the assistant turn (read from the chat template, not hard-coded)."""
        if self._tail is None:
            tail = self._render("", _MARK); self._tail = tail[tail.index(_MARK) + len(_MARK):]
        return user + self._tail

    def close(self):
        import gc
        self.llm = None; self.prefixes = {}; gc.collect()


class Probe:
    """Vocabulary-only view of a model file (no weights are loaded, about a second): its chat template and tokenizer. Used to detect the tool-call dialect and to MEASURE the static prefix in tokens, so the context is sized to the prompt (prompt tokens + a margin) instead of guessed."""
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, vocab_only=True, verbose=False); tok = lambda i: self.llm._model.token_get_text(i)
        self.template = self.llm.metadata["tokenizer.chat_template"]
        self.fmt = F.Jinja2ChatFormatter(template=self.template, eos_token=tok(self.llm.token_eos()), bos_token=tok(self.llm.token_bos()), add_generation_prompt=True)

    def prefix_tokens(self, system, shots=None, tools=None):
        msgs = [{"role": "system", "content": system}]
        for u, a in shots or []: msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        full = self.fmt(messages=msgs + [{"role": "user", "content": _MARK}], tools=tools or None).prompt; prefix = full[:full.index(_MARK)]
        return len(self.llm.tokenize(prefix.encode(), add_bos=False, special=True))

    def close(self): self.llm = None
