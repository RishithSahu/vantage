"""Hybrid capability retrieval: SQLite FTS5 (BM25, exact identifiers and technical terms) + a pretrained sentence embedding (BAAI/bge-small-en-v1.5, ONNX, frozen) over the capability registry's descriptions, fused by reciprocal rank.

Retrieval does NOT depend on the rule parser recognising anything first; it works from the question text and the registry descriptions. Scores are rankings, not probabilities of correctness: a hit is reported only when the semantic similarity clears a floor or the lexical match is strong, and it may return nothing.
What retrieval is for here: (1) a short focus hint for the model, (2) related capabilities (a blockage signal AND its event stream), (3) surfacing capabilities that are KNOWN but NOT RECORDED on this vehicle, so the answer can be an explained limitation, (4) validation that a plan's target is plausible for the question. Exact historical numbers are never retrieved this way: they come from indexed queries."""
import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("a an the of on in at to for from by with is are was were be been it its my me i you your we our this that these those and or but not do does did has have had can could would should will shall what which who whom when where why how any all some than then there here as if so up down over under about into out off per many much more most less least show tell give list please lately recently last past previous ago just now".split())


def model_fingerprint(path):
    """Content fingerprint for model-dependent caches (not merely file size)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""): h.update(block)
    except OSError:
        return "unavailable"
    return h.hexdigest()[:20]


@dataclass
class Hit:
    id: str
    kind: str
    sem: float
    lex_rank: int
    sem_rank: int
    fused: float
    available: bool


class Embedder:
    """bge-small-en-v1.5 on onnxruntime (CPU). No torch, no sentence-transformers; the tokenizer is the model's own tokenizer.json."""
    def __init__(self, model_dir, int8=False, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        f = os.path.join(model_dir, "model_int8.onnx" if int8 else "model.onnx")
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"]); self.path = f
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json")); self.tok.enable_truncation(max_length=128); self.tok.enable_padding()
        self.inputs = {i.name for i in self.sess.get_inputs()}; self.dim = 384

    def encode(self, texts, query=False):
        if query: texts = [QUERY_PREFIX + t for t in texts]
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64); mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs: feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        v = out[:, 0, :]                                                  # CLS pooling (the model's own pooling config)
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float32)


class Retriever:
    SEM_FLOOR = 0.58                                                      # bge-small similarities are compressed (unrelated text still scores 0.5-0.55): below this a semantic match is not reported on its own (a ranking floor, not a calibrated probability)
    BAND = 0.12                                                           # and only matches within this distance of the best one are kept

    def __init__(self, registry, embedder, cache_dir=None):
        self.reg = registry; self.emb = embedder; self.items = registry.items; self.ids = [c.id for c in self.items]
        # Some ordinary interrogatives are also meaningful words in the schema
        # (for example ``where`` is part of the places capability).  A blanket
        # stop-word filter used to erase those requests completely.  Preserve
        # stop words that occur distinctively in a tool description; this is
        # derived from registry metadata, rather than a question-phrase rule.
        tool_docs = [set(re.findall(r"[a-z0-9]+", c.text.lower())) for c in self.items if c.kind == "tool"]
        tool_df = {w: sum(w in doc for doc in tool_docs) for doc in tool_docs for w in doc}
        self.schema_query_words = {w for w, count in tool_df.items() if w in STOP and count <= 1 and len(w) > 1}
        self.db = sqlite3.connect(":memory:"); self.db.execute("CREATE VIRTUAL TABLE caps USING fts5(id UNINDEXED, body, tokenize='porter unicode61')")
        self.db.executemany("INSERT INTO caps(id, body) VALUES (?, ?)", [(c.id, c.text) for c in self.items])
        # The production 0.5B route deliberately has no second learned model.
        # In that configuration retrieval remains deterministic FTS capability
        # lookup; embeddings are reserved for explicit research experiments.
        self.vecs = self._vectors(cache_dir) if self.emb is not None else None

    def _vectors(self, cache_dir):
        key = f"{self.reg.signature()}_{model_fingerprint(self.emb.path)}"; f = os.path.join(cache_dir, f"capvec_{key}.npy") if cache_dir else None
        if f and os.path.exists(f):
            try:
                v = np.load(f)
                if v.shape[0] == len(self.items): return v
            except Exception: pass
        v = np.vstack([self.emb.encode([c.text for c in self.items[i:i + 32]]) for i in range(0, len(self.items), 32)])
        if f:
            try: os.makedirs(cache_dir, exist_ok=True); np.save(f, v)
            except OSError: pass
        return v

    def _lex(self, question, n=12):
        words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if (w not in STOP or w in self.schema_query_words) and len(w) > 1]
        if not words: return []
        q = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        try: rows = self.db.execute("SELECT id FROM caps WHERE caps MATCH ? ORDER BY bm25(caps) LIMIT ?", (q, n)).fetchall()
        except sqlite3.Error: return []
        return [r[0] for r in rows]

    COND_SEM = 0.72                                                       # a condition is offered on similarity alone only when it is clearly the best match
    GENERIC = set("condition conditions while during when at in on the of a an to for with and or driving drive car vehicle time".split())
    MARKERS = {"while", "when", "whenever", "during", "in", "on", "at", "only", "just", "under", "with", "if", "once", "as", "across", "within", "inside", "outside"}   # closed-class words that introduce a restriction

    @staticmethod
    def _stem(w):
        w = w.lower()
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3: return w[: -len(suf)]
        return w

    def conditions(self, question, k=3):
        """Named conditions (while charging, in the rain, on the highway ...) the question may restrict to. Offered when a distinctive word of the condition's description appears in the question right after a restriction-introducing word (while, when, during, in, on, at, only ...), or when the condition is clearly the closest semantic match. Empty when none: no restriction is ever invented."""
        if self.emb is None:
            return []
        qv = self.emb.encode([question], query=True)[0]; toks = re.findall(r"[a-z]+", question.lower()); stems = [self._stem(w) for w in toks]
        generic = {self._stem(g) for g in self.GENERIC} | {self._stem(w) for w in STOP}; out = []
        for i, c in enumerate(self.items):
            if c.kind != "condition": continue
            sim = float(self.vecs[i] @ qv); doc = {self._stem(w) for w in re.findall(r"[a-z]+", c.text.lower())} - generic; evidence = False
            for j, st in enumerate(stems):
                if st in doc and any(t in self.MARKERS for t in toks[max(0, j - 3): j]): evidence = True; break
            if (evidence and sim >= 0.5) or sim >= self.COND_SEM: out.append((c.id, round(sim, 3)))
        out.sort(key=lambda x: -x[1]); return out[:k]

    WIN_FLOOR = 0.72                                                      # a short phrase of the question must be this similar to a capability's description to add it
    WIN_EXTRA = 3                                                         # and at most this many capabilities are added that way

    def window_hits(self, question, have, floor=None, extra=None):
        """Sub-phrase retrieval: a question that names TWO capabilities ("highest speed when it was colder than 41 fahrenheit") has one embedding that favours one of them, so every 1-3 word window of the question is embedded too and each capability takes its best window similarity. Windows made only of function words are skipped. Adds capabilities (signals, events, tools) not already retrieved; no list of words is involved."""
        if self.emb is None:
            return []
        words = re.findall(r"[a-z0-9']+", question.lower()); wins = []
        for n in (1, 2, 3):
            for i in range(0, len(words) - n + 1):
                w = words[i:i + n]
                if all(x in STOP or len(x) < 2 for x in w): continue
                wins.append(" ".join(w))
        if not wins: return []
        W = self.emb.encode(list(dict.fromkeys(wins)), query=True); best = (self.vecs @ W.T).max(axis=1); order = np.argsort(-best); out = []
        for i in order:
            c = self.items[i]
            if float(best[i]) < (floor or self.WIN_FLOOR) or len(out) >= (extra or self.WIN_EXTRA): break
            if c.kind in ("signal", "event", "tool") and c.id not in have: out.append(Hit(c.id, c.kind, round(float(best[i]), 3), -1, 100 + len(out), 0.0, c.available))
        return out

    CARD_KINDS = ("signal", "event", "tool")                                  # what a call can name as an argument: retrieval budgets are spent on these only (named conditions, trip classes and ranking metrics are listed once in the static prefix)

    def search(self, question, k=6, windows=True, kinds=None, floor=None):
        """-> (hits, ms). Hits are ordered by fused rank; each carries its own evidence (semantic similarity, lexical and semantic ranks). The k hits are chosen among the capabilities a call can NAME (signals, events, tools): a ranking metric or a condition can no longer use up retrieval budget and push a required signal out."""
        t0 = time.perf_counter(); kinds = kinds or self.CARD_KINDS
        if self.emb is None:
            lex = [cid for cid in self._lex(question, max(12, k * 2)) if self.reg.by_id[cid].kind in kinds]
            hits = [Hit(cid, self.reg.by_id[cid].kind, 0.0, rank, -1, round(1.0 / (60 + rank), 5), self.reg.by_id[cid].available)
                    for rank, cid in enumerate(lex[:k])]
            ids = {h.id for h in hits}
            for h in list(hits):
                for related in self.reg.by_id[h.id].related:
                    if related not in ids and related in self.reg.by_id:
                        c = self.reg.by_id[related]; hits.append(Hit(related, c.kind, 0.0, -1, -1, 0.0, c.available)); ids.add(related)
            return hits, (time.perf_counter() - t0) * 1000
        qv = self.emb.encode([question], query=True)[0]; sims = self.vecs @ qv
        elig = np.array([i for i, c in enumerate(self.items) if c.kind in kinds]); order = elig[np.argsort(-sims[elig])]
        sem_rank = {self.ids[i]: r for r, i in enumerate(order[:30])}; lex = [c for c in self._lex(question, 30) if self.reg.by_id[c].kind in kinds][:12]; lex_rank = {i: r for r, i in enumerate(lex)}
        fused = {}
        for cid in set(sem_rank) | set(lex_rank):
            s_ = 0.0
            if cid in sem_rank: s_ += 1.0 / (60 + sem_rank[cid])
            if cid in lex_rank: s_ += 1.0 / (60 + lex_rank[cid])
            fused[cid] = s_
        hits = []; top = float(sims[order[0]]); cut = max(self.SEM_FLOOR if floor is None else floor, top - self.BAND)
        for cid, s_ in sorted(fused.items(), key=lambda x: -x[1]):
            i = self.ids.index(cid); sim = float(sims[i]); strong_lex = cid in lex_rank and lex_rank[cid] < 3 and sim >= 0.5
            if sim < cut and not strong_lex: continue
            c = self.items[i]; hits.append(Hit(cid, c.kind, round(sim, 3), lex_rank.get(cid, -1), sem_rank.get(cid, -1), round(s_, 5), c.available))
            if len(hits) >= k: break
        if windows: hits += self.window_hits(question, {h.id for h in hits})
        ids = {h.id for h in hits}                                       # related capabilities travel together (a signal and the event stream derived from it)
        for h in list(hits):
            for r in self.reg.by_id[h.id].related:
                if r not in ids and r in self.reg.by_id:
                    c = self.reg.by_id[r]; hits.append(Hit(r, c.kind, 0.0, -1, -1, 0.0, c.available)); ids.add(r)
        return hits, (time.perf_counter() - t0) * 1000
