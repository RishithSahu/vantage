"""Train the meaning-frame LoRA with the project's LoRA loop (training/train_lora.py), changing only what the examples look like.

    python training/meaning/train_meaning.py --data training/meaning/data/full --out training/out/meaning_full --epochs 3 --bs 16 --lr 2e-4 --rank 16 --alpha 32 --seed 1

Each row: kind (frame | term | rel), text, answer, optional hint (glossary line).  The prompt is vgpt.meaning ontology.system_prompt() (the same text the runtime sends); the loss is on the answer
tokens only.  Everything else (LoRA layers, optimiser, schedule, overlap checks, logging) is train_lora.main unchanged, except that activation checkpointing is switched on: the ~200-token system
prompt makes sequences ~4x longer than the earlier plan-parser runs, which does not fit a 6 GB card otherwise (measured: 9.8 GB peak without it, i.e. spilling into system RAM)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from training import train_lora as T                      # noqa: E402
from vgpt.meaning import interface                         # noqa: E402


def encode(tok, rows, max_len=int(os.environ.get("VGPT_MAXLEN", 420 if os.environ.get("VGPT_FRAME") != "2" else 560)), semantic=False):
    out = []
    for r in rows:
        p = tok(interface.prompt_text(r["kind"], r["text"], r.get("hint", ""), r.get("prev"), vehicle=r.get("vehicle")), add_special_tokens=False)["input_ids"]          # `prev`: frame version 2 (VGPT_FRAME=2) shows the previous frame
        a = tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > max_len: raise ValueError(f"example exceeds {max_len} tokens: {r.get('id')} {r['text'][:60]}")
        out.append((p + a, [-100] * len(p) + a))
    return out


def _checkpointed(orig):
    def load(*a, **k):
        m = orig(*a, **k)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return m
    return load


T.encode = encode
T.HERE = ROOT

if __name__ == "__main__":
    import transformers
    transformers.AutoModelForCausalLM.from_pretrained = _checkpointed(transformers.AutoModelForCausalLM.from_pretrained)
    T.main()
