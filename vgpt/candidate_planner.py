"""Candidate selection with a small frozen model.

The model never writes a query. A deterministic generator (vgpt.candidates) lists the readings the question supports; the model reads the question, the previous accepted reading and the numbered readings (as plain English from vgpt.paraphrase)
and outputs ONE index (grammar-constrained to a number), or 0 for 'none of these'. Constrained generation here only guarantees a well-formed index; whether the chosen reading is right is checked afterwards by the support + coverage checks, and an uncertain
choice becomes a clarification, never a silent answer. Consumes: the original question, the actual conversation state (the previous accepted query), and the runtime schema through the readings' labels."""
import os

from .paraphrase import describe
from .slm_backend import PrefixLlama

SYSTEM = ("You help a vehicle data assistant pick the right reading of a driver's question. The assistant can only answer from the vehicle's own recorded data. "
          "You get the question, optionally the previous reading, and numbered readings. Reply with the number of the reading that matches what the person asked, or 0 if none of them matches. Output only the number.")
GRAMMAR = "root ::= [0-9] [0-9]?"


class CandidateSelector:
    def __init__(self, model_path, n_ctx=4608, threads=2, cache_dir=None, backend=None):
        self.be = backend if backend is not None else PrefixLlama(model_path, n_ctx=n_ctx, threads=threads, cache_dir=cache_dir)
        self.be.set_prefix(SYSTEM); self.name = f"select[{os.path.basename(str(model_path))}]"

    def prompt(self, text, prev, cands):
        lines = [f"Previous reading: {describe(prev)}" if prev is not None else "Previous reading: none", f"Question: {text}", "Readings:"]
        lines += [f"{i}. {describe(q)}" for i, q in enumerate(cands, 1)]
        return "\n".join(lines) + "\nNumber:"

    def select(self, text, prev, cands):
        """-> (index into cands or None, info). None = the model said none fit or produced nothing usable."""
        if not cands: return None, {"ms": 0.0, "ptok": 0}
        out, info = self.be.generate(self.prompt(text, prev, cands), grammar=GRAMMAR, max_tokens=3, stop_text=("\n",))
        try: k = int(out.strip())
        except ValueError: k = 0
        info = {"ms": info.get("ms", 0.0), "ptok": info.get("prompt_tokens"), "raw": out, "error": info.get("error")}
        return (k - 1 if 1 <= k <= len(cands) else None), info
