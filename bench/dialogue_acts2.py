"""Dialogue-act decision by LETTER-SCORED multiple choice with contextual calibration and order averaging (the decision procedure that was measured to help small frozen models in this project), restricted to the acts the typed state supports. Compared against the plain grammar-constrained choice on the same cases."""
import math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.dialogue_acts import CASES
from vgpt import dialogue_state as D
from vgpt.brain import LazySlm
from vgpt.query_ir import Query

LET = "ABCDEFGHI"


def letter_scores(llm, system, body, n):
    prompt = "<|im_start|>system" + chr(10) + system + "<|im_end|>" + chr(10) + "<|im_start|>user" + chr(10) + body + "<|im_end|>" + chr(10) + "<|im_start|>assistant" + chr(10) + "Answer:"
    ids = llm.tokenize(prompt.encode(), add_bos=False, special=True); llm.reset(); llm.eval(ids)
    import llama_cpp; logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits(llm.ctx), shape=(llm.n_vocab(),)).astype(np.float64); logits = logits - logits.max(); lp = logits - math.log(np.exp(logits).sum())
    res = []
    for i in range(n):
        toks = [llm.tokenize((" " + LET[i]).encode(), add_bos=False)[-1], llm.tokenize(LET[i].encode(), add_bos=False)[-1]]
        res.append(max(lp[t] for t in toks))
    return np.array(res)


def decide(llm, summary, msg, options, args=""):
    system = "You decide how a vehicle-data assistant should read a message in a conversation. Answer with the letter of the best option."
    def body_for(order, m):
        lines = chr(10).join(f"{LET[i]}. {D.ACTS[o]}" for i, o in enumerate(order))
        return f"Previous answer was about: {summary}{chr(10)}New message: {m}{chr(10)}The message names: {args or 'no sensor, no event and no period of its own'}{chr(10)}Options:{chr(10)}{lines}{chr(10)}Answer with one letter."
    orders = [list(options), list(options)[::-1]]; tot = {o: 0.0 for o in options}
    for order in orders:
        s = letter_scores(llm, system, body_for(order, msg), len(order)); prior = letter_scores(llm, system, body_for(order, "N/A"), len(order))
        z = s - prior; z = z - z.max(); p = np.exp(z) / np.exp(z).sum()
        for i, o in enumerate(order): tot[o] += math.log(p[i] + 1e-9)
    best = max(tot, key=tot.get); return best, tot[best] - tot.get('new_question', tot[best])


STAGE1 = {"continues": "the message depends on the previous answer: it leaves out something that the previous answer supplies, or refers back to it",
          "new_question": "the message stands on its own: it says everything needed and asks something the previous answer did not cover"}


def decide2(llm, summary, msg, options):
    if len(options) < 2: return options[0]
    cont = [o for o in options if o != "new_question"]
    if not cont: return "new_question"
    system = "You decide how a vehicle-data assistant should read a message in a conversation. Answer with the letter of the best option."
    def body(order, m, table):
        lines = chr(10).join(f"{LET[i]}. {table[o]}" for i, o in enumerate(order))
        return f"Previous answer was about: {summary}{chr(10)}New message: {m}{chr(10)}Options:{chr(10)}{lines}{chr(10)}Answer with one letter."
    def pick(order_list, table, msgs=None):
        tot = {o: 0.0 for o in order_list}
        for order in (list(order_list), list(order_list)[::-1]):
            sc = letter_scores(llm, system, body(order, msg, table), len(order)); pr = letter_scores(llm, system, body(order, "N/A", table), len(order))
            z = sc - pr; z = z - z.max(); p = np.exp(z) / np.exp(z).sum()
            for i, o in enumerate(order): tot[o] += math.log(p[i] + 1e-9)
        return max(tot, key=tot.get)
    if pick(list(STAGE1), STAGE1) == "new_question": return "new_question"
    return cont[0] if len(cont) == 1 else pick(cont, D.ACTS)


def main():
    """Evaluates the PRODUCTION decision (vgpt.dialogue_state.decide_act) on the paraphrase cases of bench/dialogue_acts.py."""
    slm = LazySlm(os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), idle_s=3600, threads=2, mmap=True, fmt="words")
    class B: now = 1787109669; tz = 0
    b = B(); b.slm = slm; ok = 0; t0 = time.time()
    for topic, msg, gold in CASES:
        q = Query.from_text(topic); concepts = D._concept_set(D._mentions(msg)); opts = D.feasible_acts(b, q, msg.lower(), concepts)
        act = D.decide_act(b, msg, q, opts) if len(opts) > 1 else opts[0]; ok += act == gold
        print(("ok  " if act == gold else "MISS"), gold, "<-", act, "|", msg)
    print(f"{ok}/{len(CASES)} correct, {(time.time() - t0) / len(CASES):.1f} s per decision (orders={D.ORDERS})")


if __name__ == "__main__": main()
