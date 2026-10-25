"""A minimal LoRA (low-rank adaptation) for transformers models - no peft dependency.

Each targeted nn.Linear W is wrapped:  y = W x + (alpha / r) * B (A x)   with A: r x in (Kaiming init) and B: out x r (zeros), so training starts exactly at the base model. Only A and B are trained; the base
weights stay frozen (and can be the de-quantised weights of a Q4 GGUF - the adapter then applies to that same Q4 model at run time through llama.cpp's LoRA support, i.e. QLoRA-style: the adapter learns to
correct the quantised model it will actually be served on).

Names follow peft's convention (base_model.model.<module path>.lora_A.weight / lora_B.weight) so the saved adapter is readable by peft tooling as well as by convert_lora_gguf.py."""
import math

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base; self.r = r; self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32)); nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32)); self.drop = nn.Dropout(dropout)
        for p in self.base.parameters(): p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        z = self.drop(x).to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
        return y + (z * self.scale).to(y.dtype)


def apply_lora(model, r=16, alpha=32, dropout=0.05, targets=TARGETS):
    """Wrap every targeted linear layer of the decoder blocks. Returns the list of (name, module)."""
    wrapped = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets and ".layers." in f".{name}." + "":
                lin = LoRALinear(child, r, alpha, dropout); setattr(mod, cname, lin); wrapped.append((f"{name}.{cname}" if name else cname, lin))
    for p in model.parameters(): p.requires_grad = False
    for _, m in wrapped: m.lora_A.requires_grad = True; m.lora_B.requires_grad = True
    return wrapped


def lora_state(wrapped):
    out = {}
    for name, m in wrapped:
        out[f"base_model.model.{name}.lora_A.weight"] = m.lora_A.detach().float().cpu().clone().contiguous(); out[f"base_model.model.{name}.lora_B.weight"] = m.lora_B.detach().float().cpu().clone().contiguous()
    return out


def load_lora_state(wrapped, state):
    for name, m in wrapped:
        m.lora_A.data.copy_(state[f"base_model.model.{name}.lora_A.weight"]); m.lora_B.data.copy_(state[f"base_model.model.{name}.lora_B.weight"])


def trainable_params(wrapped): return sum(m.lora_A.numel() + m.lora_B.numel() for _, m in wrapped)
