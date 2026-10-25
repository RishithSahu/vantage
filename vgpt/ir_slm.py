"""The language model as a query COMPILER: question -> one line of Query IR text, generated under a grammar built from the registry (query_grammar.py).

It never answers, never sees data, never writes numbers about the vehicle - it only chooses registry names and copies numbers from the question. Everything after it (validation, capability, planning,
execution, evidence, wording, verification) is deterministic code. An optional LoRA adapter (GGUF, shipped separately from the base model) teaches it the format so the prompt shrinks to one line.

IrSlm is a drop-in sibling of brain.Slm: same memory discipline (small batch, no python-side state cache, KV reuse of the shared prompt prefix), same lazy load / idle unload (LazyIrSlm)."""
import os
import re
import time

from . import query_grammar as G
from .query_ir import IRError, Query

NUM_RX = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


class IrSlm:
    def __init__(self, path, lora_path=None, threads=2, n_ctx=1024, mmap=True, mode=None):
        from llama_cpp import Llama, LlamaGrammar
        t = time.perf_counter()
        kw = dict(model_path=path, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads, n_batch=32, n_ubatch=32, use_mmap=mmap, verbose=False)
        if lora_path: kw["lora_path"] = lora_path
        self.llm = Llama(**kw)
        self.grammar = LlamaGrammar.from_string(G.gbnf().strip(), verbose=False)
        self.mode = mode or ("tuned" if lora_path else "fewshot")
        self.path, self.lora_path = path, lora_path; self.load_s = time.perf_counter() - t; self.warm = False

    def warmup(self): self.compile("what is my battery temperature right now"); self.warm = True

    def close(self):
        import gc
        self.llm = None; gc.collect()

    def compile(self, text, max_tokens=160, context=None):
        """-> (Query | None, info{ms, out_tokens, raw, error}). Greedy, grammar-constrained."""
        t = time.perf_counter()
        import numpy as np
        from llama_cpp import LogitsProcessorList
        conf = []
        def _rec(input_ids, scores):                                                                              # confidence = the model's own top-1 probability at every generated step (no extra memory)
            x = np.asarray(scores, dtype=np.float32); m = float(x.max()); conf.append(float(1.0 / np.exp(x - m).sum()))
            return scores
        r = self.llm.create_chat_completion(messages=G.messages(text, self.mode, context), grammar=self.grammar, max_tokens=max_tokens, temperature=0.0, logits_processor=LogitsProcessorList([_rec]))
        out = r["choices"][0]["message"]["content"].strip(); u = r.get("usage", {})
        info = {"ms": (time.perf_counter() - t) * 1000, "out_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"), "raw": out, "error": None}
        info["min_p"] = min(conf) if conf else None; info["mean_p"] = (sum(conf) / len(conf)) if conf else None
        try:
            q = Query.from_text(out); q.text = text[:300]
            from . import measure_grounding as MG
            if MG.plan_issues(q, text, context_plan=context):
                abstain = Query.from_text("op=none t=all p.why=ambiguous"); abstain.text = text[:300]
                info["grounding_rejected"] = q.to_text(); return abstain, info
            return q, info
        except (IRError, ValueError) as e:
            info["error"] = str(e); return None, info


class LazyIrSlm:
    """Loads on the first question the rules could not parse; releases after `idle_s` of silence so the memory returns to the vehicle."""
    def __init__(self, path, lora_path=None, idle_s=120.0, clock=time.monotonic, **kw):
        self.path, self.lora_path, self.idle_s, self.kw, self.clock = path, lora_path, idle_s, kw, clock
        self.slm, self.last, self.loads, self.load_s = None, 0.0, 0, 0.0

    @property
    def loaded(self): return self.slm is not None

    def _ensure(self):
        if self.slm is None:
            t = time.perf_counter(); self.slm = IrSlm(self.path, self.lora_path, **self.kw); self.slm.warmup(); self.load_s = time.perf_counter() - t; self.loads += 1

    def compile(self, text, context=None):
        self._ensure(); self.last = self.clock(); return self.slm.compile(text, context=context)

    def tick(self):
        if self.slm is not None and self.clock() - self.last > self.idle_s: self.unload()

    def unload(self):
        if self.slm is not None: self.slm.close(); self.slm = None


def grounding_issues(q, text, concept_hits=None):
    """Hallucination guard for model-written queries: every number in the IR must come from the question (as a digit or a number word, possibly unit-converted),
    and every signal / event / group named must be traceable to words in the question. Returns a list of problems (empty = grounded)."""
    from . import nl2ir, signal_registry as R
    t = nl2ir.normalise(text); issues = []
    words = set(re.findall(r"[a-z0-9]+", t))
    nums = {float(x) for x in NUM_RX.findall(t)} | {float(nl2ir.NUMW[w]) for w in re.findall(r"[a-z]+", t) if w in nl2ir.NUMW}
    nums |= {1.0} if re.search(r"\b(a|an|one|last|latest|previous|first|hour|day|week|month)\b", t) else set()
    if not nl2ir._times(nl2ir._S(t), None, 0): nums |= {7.0}                                            # no period in the question: the engine's default (last 7 days) applies
    if re.search(r"\bweeks?\b", t): nums |= {7.0}                                                    # "last week" = the last 7 days
    if re.search(r"\bmonths?\b", t): nums |= {30.0}                                                   # "last month" = the last 30 days
    if re.search(r"\bhours?\b", t): nums |= {24.0}                                                    # "the last 24 hours" may be phrased "the last day"
    def has(v, tol=0.01):
        return any(abs(v - n) <= max(tol, abs(n) * 0.006) for n in nums) or any(abs(v - R.convert(n, u, "km/h")) <= 0.6 for n in nums for u in ("mph",)) or any(abs(v - (n - 32) * 5 / 9) <= 0.6 for n in nums)
    nums_ir = []
    for sc in (q.scope, q.scope_b):
        if sc is None: continue
        if sc.time.value is not None: nums_ir.append(float(sc.time.value))
        for c in sc.conditions:
            if c.value is not None: nums_ir.append(float(c.value))
        if sc.select and sc.select.k: nums_ir.append(float(sc.select.k))
    for v in nums_ir:
        if not has(v) and not (v in (1.0, 7.0, 14.0, 24.0) and not nums): issues.append(f"the number {v:g} is not in the question")
    UNIT_WORDS = {"mph": "mph", "psi": "psi", "f": "F", "fahrenheit": "F", "bar": "bar", "kpa": "kPa", "kw": "kW", "w": "W", "rpm": "rpm", "nm": "Nm", "v": "V", "l/min": "L/min", "km/h": "km/h", "c": "C", "%": "%"}
    for sc in (q.scope, q.scope_b):                                                                   # a threshold must carry the unit the person spoke (60 mph is not 60 km/h)
        for c in (sc.conditions if sc else []):
            if c.signal and c.value is not None:
                num = f"{c.value:g}"
                m = re.search(r"(?<![\w.])" + re.escape(num) + r"\s*(mph|psi|fahrenheit|f|bar|kpa|kw|w|rpm|nm|v|l/min|km/h|c|%)(?![a-z])", t)
                if m:
                    spoken = UNIT_WORDS[m.group(1)]; reg = R.CONCEPTS[c.signal].unit
                    if spoken != reg and c.unit != spoken: issues.append(f"the threshold {num} was spoken in {spoken} but the query does not say so")
    named = [x for x in (q.target, q.target2) if x] + [c.signal for sc in (q.scope, q.scope_b) if sc for c in sc.conditions if c.signal]
    seen = set()
    for m in nl2ir._CONCEPT_RX.finditer(t):
        for n in R.LEXICON.get(m.group(1).replace("-", " "), ()): seen.add(n)
    for n in named:
        c = R.CONCEPTS.get(n)
        if c is None: issues.append(f"unknown signal {n}"); continue
        if seen and n not in seen and n not in ("wh_per_km", "energy_used", "distance_driven", "battery_soc", "speed", "drive_mode"):
            issues.append(f"{n} is not what the question names (it names {', '.join(sorted(seen)[:3])})"); continue        # substitution: the question names another signal
        stems = {w[:5] for w in re.findall(r"[a-z]+", (c.label + " " + n.replace("_", " ")).lower()) if len(w) > 3}
        if n not in seen and not any(w[:5] in stems for w in words if len(w) > 3) and n not in ("wh_per_km", "energy_used", "distance_driven", "battery_soc", "speed", "drive_mode"): issues.append(f"{n} is not mentioned in the question")
    if q.event is not None and q.event.type not in ("ANY_WARNING",):
        label_words = {w[:5] for w in re.findall(r"[a-z]+", R.EVENTS[q.event.type][0].lower()) if len(w) > 3}
        syn_words = {w[:5] for s in R.EVENTS[q.event.type][1] for w in re.findall(r"[a-z]+", s) if len(w) > 3}
        if not (label_words | syn_words) & {w[:5] for w in words if len(w) > 3}: issues.append(f"event {q.event.type} is not mentioned in the question")
    for sc in (q.scope, q.scope_b):                                                                    # a named condition must be something the question says (not a word that merely resembles it: "fastest" is not "fast charging")
        for c in (sc.conditions if sc else []):
            if c.predicate:
                syns = [c.predicate.replace("_", " ")] + list(R.PRED_SYN.get(c.predicate, ()))
                if not any(re.search(r"\b" + re.escape(x) + r"\b", t) for x in syns): issues.append(f"condition {c.predicate} is not mentioned in the question")
    for g in q.groups:
        syns = [g] + list(R.PRED_SYN.get(g, ()))
        if not any(re.search(r"\b" + re.escape(s) + r"\b", t) for s in syns): issues.append(f"group {g} is not mentioned in the question")
    return issues


def autodetect(models_dir, threads=2):
    """The query-compiling model for this installation, or None. Looks for the base GGUF (qwen2.5-0.5b-instruct-q4_k_m.gguf) and a LoRA adapter (ir-lora*.gguf) in `models_dir`;
    The model path is OPT-IN: set VGPT_IR_SLM=1 (and optionally VGPT_IR_LORA=<file> to name the adapter). Without an adapter the base model is NOT used for query compilation (its few-shot accuracy is too low to trust)."""
    import glob
    import importlib.util
    if os.environ.get("VGPT_IR_SLM", "0") != "1" or importlib.util.find_spec("llama_cpp") is None: return None
    base = os.path.join(models_dir, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    lora = os.environ.get("VGPT_IR_LORA") or next(iter(sorted(glob.glob(os.path.join(models_dir, "ir-lora*.gguf")))), None)
    if not (os.path.exists(base) and lora and os.path.exists(lora)): return None
    return LazyIrSlm(base, lora, threads=threads)
