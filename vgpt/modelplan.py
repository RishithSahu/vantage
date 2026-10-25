"""The planner: message + conversation state -> retrieved capability cards -> ONE frozen-model call -> canonical plan.

    retrieval (FTS5 + bge-small embeddings over the capability descriptions)  ->  cards in the prompt  ->  one tool call, grammar-constrained to types and registered identifiers  ->  parsed to a plan

Stages are timed separately: retrieve_ms (embedding + search), render_ms (building the per-message text), state_ms (restoring the saved prefix state), prefill_ms (evaluating the per-message tokens), decode_ms (generating the call), grammar_ms. Nothing here reads the vehicle's data or interprets wording by rule."""
import os
import time
from dataclasses import dataclass, field

from . import capreg, demos as D, dialects, plangrammar, prompt, signal_registry as R, toolspec as T
from .retrieval import Embedder, Retriever
from .slm_backend import PrefixLlama, Probe
from .support import SupportCheck

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TOKENS = 130


@dataclass
class State:
    prev_plan: dict = None                     # the resolved plan of the last ANSWERED turn
    pending: dict = None                       # {"ask": clarification asked, "original": the question it was about, "plan": partial plan or None}
    history: list = field(default_factory=list)   # the last turn as (user text, assistant call text): shown to the model as an earlier chat turn


@dataclass
class PlanResult:
    raw: str = ""
    fn: str = ""
    args: dict = None
    plan: dict = None
    hits: list = field(default_factory=list)
    card_ids: list = field(default_factory=list)
    message: str = ""
    error: str = ""
    timing: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class Planner:
    def __init__(self, conn, model_path=None, embedder_dir=None, threads=2, n_ctx="auto", cache_dir=None, int8_embedder=True, mmap=True, max_resident=2, registry=None, load_model=True, dialect=None, decode="grammar", schema_level="rich", core_shots=False, extractor=None, evidence_ops=False, demo_k=0, demo_per_fn=3, demo_cards=True, demo_mask=True, semantic_retrieval=True):
        self.cache_dir = cache_dir or os.path.join(HERE, "results", "_cache")
        self.profile = R.Profile(conn); self.reg = registry or capreg.CapabilityRegistry(self.profile)
        t = time.perf_counter()
        self.semantic_retrieval = semantic_retrieval
        self.emb = Embedder(embedder_dir or os.path.join(HERE, "models", "bge-small-en-v1.5"), int8=int8_embedder, threads=threads) if semantic_retrieval else None
        self.retriever = Retriever(self.reg, self.emb, self.cache_dir); self.support = SupportCheck(self.emb, self.reg) if self.emb is not None else None; self.retrieval_load_s = time.perf_counter() - t
        self.model_path = model_path or os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"); self.lm = None; self.prefix_s = 0.0; self.model_load_s = 0.0; self.dialect = dialects.by_name(dialect) if isinstance(dialect, str) else dialect; self.decode = decode; self.schema_level = schema_level; self.core_shots = core_shots; self.extractor = extractor; self.evidence_ops = evidence_ops; self.last_extraction = None; self.demo_k = demo_k; self.demo_per_fn = demo_per_fn; self.demo_cards = demo_cards; self.demo_mask = demo_mask; self.demo_bank = D.DemoBank(self.reg, self.emb, cache_dir=self.cache_dir, mask=demo_mask) if demo_k and self.emb is not None else None
        if load_model:
            if n_ctx == "auto":                                                                               # size the context to the prompt: measure the static prefix with a vocabulary-only probe, add a margin for the message, the output and the conversation turn
                probe = Probe(self.model_path); self.dialect = self.dialect or dialects.detect(probe.template)
                sy, sh, tl = prompt.system_text(self.reg, self.dialect, self.extractor is not None), prompt.shots(self.reg, self.dialect, self.core_shots), prompt.tools(self.reg, self.dialect, self.schema_level)
                n_ctx = (probe.prefix_tokens(sy, sh, tl) + 1280 + 320 * self.demo_k + 255) // 256 * 256; probe.close()
            t = time.perf_counter(); self.lm = PrefixLlama(self.model_path, n_ctx=n_ctx, threads=threads, n_batch=64, mmap=mmap, cache_dir=self.cache_dir, max_resident=max_resident); self.model_load_s = time.perf_counter() - t; self.n_ctx = n_ctx
            self.dialect = self.dialect or dialects.detect(self.lm.template)                               # each model is used in its own documented tool-call form
        self.dialect = self.dialect or dialects.Pythonic()
        if hasattr(self.dialect, "registry"): self.dialect.registry = self.reg
        self.system = prompt.system_text(self.reg, self.dialect, self.extractor is not None); self.shots = prompt.shots(self.reg, self.dialect, self.core_shots); self.tools = prompt.tools(self.reg, self.dialect, self.schema_level); self._grammars()
        if self.lm: t = time.perf_counter(); self.prefix_s = self.lm.set_prefix(self.system, self.shots, name="plan", tools=self.tools); self.model_load_s += time.perf_counter() - t

    def _grammars(self):
        self.grammar_state = plangrammar.build(self.reg, with_state=True, style=self.dialect.style, grounded=self.dialect.grounded); self.grammar_new = plangrammar.build(self.reg, with_state=False, style=self.dialect.style, grounded=self.dialect.grounded)

    def demo_turns(self, question):
        """The nearest generated demonstrations (see demos.py) as earlier chat turns, nearest last: [(user text, assistant text)] in the prompt form of a live message."""
        if not self.demo_bank: return []
        if hasattr(self.dialect, "registry"): self.dialect.registry = self.reg
        out = []
        for e, _ in self.demo_bank.retrieve(question, self.demo_k, self.demo_per_fn):
            cards = self.reg.card_text(e["ids"]) if self.demo_cards else ""
            out.append((prompt.message_text(e["q"], cards, prompt.EXAMPLE_NOW, 0, None, None), self.dialect.assistant_text(e["fn"], e["args"])))
        return out

    def grammar_for(self, result, with_state):
        """The grammar of one already-decided result (intent surface only): the fields that belong to it, typed and bound to registered identifiers."""
        assert self.dialect.style == "intent", "a decided result needs the intent surface"
        key = (result, with_state, id(self.reg)); cache = self.__dict__.setdefault("_rg", {})
        if key not in cache: cache[key] = plangrammar.build_intent(self.reg, with_state=with_state, only=result)
        return cache[key]

    def function_grammar(self, fn, with_state, exclude=()):
        """Grammar restricted to one operation family selected from schema
        retrieval (never from a phrase table)."""
        key = (fn, with_state, tuple(sorted(exclude or ())), id(self.reg)); cache = self.__dict__.setdefault("_fg", {})
        if key not in cache: cache[key] = plangrammar.build(self.reg, with_state=with_state, style=self.dialect.style, grounded=self.dialect.grounded, only=fn, exclude=exclude)
        return cache[key]

    def reload_schema(self, conn, registry=None):
        """Point the planner at another vehicle (another schema overlay or history) without reloading the model. Everything that depends on the vehicle's schema is REGENERATED - the registry, the retrieval index, the grammars, the tool declarations, the instruction (named conditions) and the format examples (an example whose capabilities are not in the new registry is left out) - so nothing from the previous vehicle survives. The static prefix is evaluated again only if its text changed."""
        self.profile = R.Profile(conn); self.reg = registry or capreg.CapabilityRegistry(self.profile)
        self.retriever = Retriever(self.reg, self.emb, self.cache_dir); self.support = SupportCheck(self.emb, self.reg) if self.emb is not None else None; self._grammars(); self.demo_bank = D.DemoBank(self.reg, self.emb, cache_dir=self.cache_dir, mask=self.demo_mask) if self.demo_k and self.emb is not None else None
        system, shots, tools = prompt.system_text(self.reg, self.dialect, self.extractor is not None), prompt.shots(self.reg, self.dialect, self.core_shots), prompt.tools(self.reg, self.dialect, self.schema_level)
        if (system, shots, tools) != (self.system, self.shots, self.tools):
            self.system, self.shots, self.tools = system, shots, tools
            if self.lm: self.prefix_s = self.lm.set_prefix(self.system, self.shots, name="plan", tools=self.tools)

    # ---- what the model is shown ------------------------------------------------------------------
    def build_message(self, question, state, now, tz=0, k=8, feedback=None, extra_ids=None):
        """-> (message text, hits, card ids, retrieve_ms, render_ms)."""
        state = state or State(); pend = state.pending
        rq = (pend["original"] + " " + question) if pend else question                                   # a clarification reply is retrieved together with the question it answers
        hits, rms = self.retriever.search(rq, k=k)
        t = time.perf_counter()
        state_ids = prompt.plan_ids(state.prev_plan) + (prompt.plan_ids(pend.get("plan")) if pend else [])
        ids = prompt.card_ids(hits, self.reg, list(state_ids) + [i for i in (extra_ids or []) if i in self.reg.by_id])
        ev = None
        if self.extractor is not None:                                                                    # optional evidence stage (a frozen pretrained extractor): spans / absences, never a gate
            from .extractor import evidence_lines
            ex = self.extractor.extract(question); self.last_extraction = ex; ev = evidence_lines(ex, self.evidence_ops); rms += ex["ms"]
        msg = prompt.message_text(question, self.reg.card_text(ids), now, tz, state.prev_plan, pend, feedback, ev)
        return msg, hits, ids, rms, (time.perf_counter() - t) * 1000

    EXPAND_FLOOR = 0.45                                                  # the relaxed similarity floor of the one bounded expansion

    def expand_ids(self, what, question, shown, k=4):
        """ONE bounded retrieval expansion for when the model reports that no registered concept matches what was asked: search again with the quantity the model named plus the question, with a relaxed floor, and return the capabilities (signals, events, tools) that were not shown yet. Nothing is added to ordinary prompts."""
        hits, _ = self.retriever.search((what + " " + question).strip(), k=k, windows=False, floor=self.EXPAND_FLOOR)
        return [h.id for h in hits if h.id not in set(shown) and self.reg.by_id[h.id].kind in ("signal", "event", "tool")][:k]

    def rendered_prompt(self, question, state, now, tz=0):
        state = state or State()
        """The full input the model receives (system turn with the tool declarations, then the message): what the prompt-inspection check reads."""
        msg = self.build_message(question, state, now, tz)[0]
        return self.lm._render(self.system, msg, self.shots + list(state.history or []), self.tools)

    def plan(self, question, state=None, now=None, tz=0, max_tokens=MAX_TOKENS, feedback=None, extra_ids=None, result=None, exclude_fields=None):
        state = state or State(); out = PlanResult(); t0 = time.perf_counter()
        out.message, out.hits, out.card_ids, rms, rnd = self.build_message(question, state, now, tz, feedback=feedback, extra_ids=extra_ids)
        result = result if result in T.INTENT_ORDER else None; withs = bool(state.prev_plan or state.pending)
        direct = [h for h in out.hits if h.lex_rank != -1 or h.sem_rank != -1]
        schema_fn = "use_tool" if direct and all(self.reg.by_id[h.id].kind == "tool" for h in direct) else None
        excluded = set(exclude_fields or ())
        gram = None if self.decode == "free" else (self.grammar_for(result, withs) if result else self.function_grammar(schema_fn, withs, excluded) if (schema_fn or excluded) else (self.grammar_state if withs else self.grammar_new))
        out.info["schema_fn"] = schema_fn
        info = self.lm.generate(out.message, grammar=gram, max_tokens=max_tokens, stop_text=(None if self.dialect.style != "frame" else ("\n",)), prefix="plan", assistant_prefix=self.dialect.open_text, history=self.demo_turns(question) + list(state.history or [])) if self.lm else ("", {"error": "no model"})
        raw, inf = info; out.raw = raw; out.info = inf
        out.timing = {"retrieve_ms": round(rms, 1), "render_ms": round(rnd, 2), "state_ms": round(inf.get("state_ms", 0.0), 1), "prefill_ms": round(inf.get("prefill_ms", 0.0), 1), "decode_ms": round(inf.get("decode_ms", 0.0), 1), "grammar_ms": round(inf.get("grammar_ms", 0.0), 1),
                      "generate_ms": round(inf.get("ms", 0.0), 1), "prompt_tokens": inf.get("prompt_tokens"), "out_tokens": inf.get("out_tokens")}
        if inf.get("error"): out.error = inf["error"]
        elif inf.get("out_tokens", 0) >= max_tokens and not self.dialect.complete(raw): out.error = "plan truncated at the token limit"
        else:
            try: out.fn, out.args = self.dialect.parse(raw); out.plan = T.call_to_plan(out.fn, out.args)
            except T.CallError as e: out.error = str(e)
        out.timing["plan_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
