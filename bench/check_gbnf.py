"""Check the offline GBNF matcher (vgpt/gbnf.py, a SECOND implementation used by tests and the representability check) against the runtime's own grammar behaviour (llama.cpp's grammar sampler) on valid, invalid and truncated outputs.
For each sample the runtime verdict is obtained by feeding the text's tokens through llama.cpp's grammar sampler: a token is allowed when its logit survives the grammar, and the text is COMPLETE when, after its last token, an end-of-generation token is allowed. Verdicts: complete (valid), prefix (valid so far, not finished), rejected. The matcher says accept (= complete) or reject; it can never say "prefix", so a runtime `prefix` must be a matcher reject.
Only vocabulary metadata of the model file is read (no weights).
    python bench/check_gbnf.py [--model models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf]"""
import argparse, ctypes, os, sqlite3, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import llama_cpp
from llama_cpp import Llama, LlamaGrammar, _internals as I
from vgpt import capreg, dialects, plangrammar, prompt, signal_registry as R
from vgpt.gbnf import Grammar

TD = np.dtype([("id", np.int32), ("logit", np.float32), ("p", np.float32)])


class Runtime:
    def __init__(self, model_path, grammar_text):
        self.l = Llama(model_path=model_path, vocab_only=True, verbose=False); self.n = self.l.n_vocab(); self.text = grammar_text; self.eog = [t for t in range(self.n) if llama_cpp.llama_vocab_is_eog(self.l._model.vocab, t)]

    def verdict(self, s):
        g = LlamaGrammar.from_string(self.text, verbose=False); smpl = I.LlamaSampler(); smpl.add_grammar(self.l._model, g)
        def allowed():
            arr = np.zeros(self.n, dtype=TD); arr["id"] = np.arange(self.n); arr["logit"] = 0.0
            t = llama_cpp.llama_token_data_array(data=arr.ctypes.data_as(ctypes.POINTER(llama_cpp.llama_token_data)), size=self.n, selected=-1, sorted=False)
            llama_cpp.llama_sampler_apply(smpl.sampler, ctypes.byref(t)); return arr["logit"] > -1e30
        ids = self.l.tokenize(s.encode(), add_bos=False, special=False)
        for tok in ids:
            if not allowed()[tok]: return "rejected"
            llama_cpp.llama_sampler_accept(smpl.sampler, tok)
        a = allowed(); return "complete" if any(a[e] for e in self.eog) else "prefix"


def samples(reg, d):
    ok = [x for x in [prompt.shots(reg, d)]][0]; out = []
    strip = lambda t: (t[len(d.open_text):-len(__import__("vgpt.toolspec", fromlist=["x"]).CALL_CLOSE)] if d.name == "lfm" else t[len(d.open_text):-len("\n</tool_call>")] if d.name == "granite" else t)
    valid = [strip(a) for u, a in ok]
    out += [(v, "valid") for v in valid]
    out += [(v[: len(v) // 2], "truncated") for v in valid[:6]]
    bad = []
    for v in valid[:10]:
        bad += [(v.replace('turn="new"', 'turn="newish"').replace('"turn": "new"', '"turn": "newish"'), "enum"), (v.replace("wh_per_km", "warp_core").replace("motor_temperature", "warp_core"), "unregistered id"), (v + " extra", "trailing text"), (v.replace('period="', 'period="in the last fortnight ').replace('"period": "', '"period": "in the last fortnight '), "bad period")]
    out += [(b, "invalid") for b, _ in bad if b not in valid]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.join(HERE, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")); a = ap.parse_args()
    reg = capreg.CapabilityRegistry(R.Profile(sqlite3.connect(f"file:{os.path.join(HERE, 'data', 'gen_courier_ext.db')}?mode=ro", uri=True))); bad = 0; n = 0
    for name, style in (("lfm", "py"), ("granite", "json"), ("frame", "frame")):
        d = dialects.by_name(name); text = plangrammar.build(reg, style=style); rt = Runtime(a.model, text); mt = Grammar(text); stats = {}
        for s, kind in samples(reg, d):
            rv = rt.verdict(s); mv = mt.accepts(s); expect_match = rv == "complete"; n += 1; stats[(kind, rv, mv)] = stats.get((kind, rv, mv), 0) + 1
            if mv != expect_match: bad += 1; print("DISAGREE", name, kind, "runtime:", rv, "matcher:", mv, "|", s[:100])
        print(name, {f"{k[0]}/{k[1]}/matcher={k[2]}": v for k, v in stats.items()})
    print(f"samples {n}, disagreements {bad}")


if __name__ == "__main__": main()
