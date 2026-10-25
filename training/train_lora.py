"""Train the query-compiler LoRA.

    python -m training.train_lora --base models/qwen2.5-0.5b-instruct-q4_k_m.gguf --data training/data --out training/out/qwen25_05b --epochs 2

* base model: a GGUF file (de-quantised by transformers on load - the adapter is trained against the very weights llama.cpp will run)
* data: training/data/train.jsonl (+ val.jsonl for monitoring), produced by training.build_dataset
* loss: ASSISTANT TOKENS ONLY - the query line and the end-of-turn token; the system prompt and the question are context, never targets
* output: adapter.safetensors (peft-style names), adapter_config.json, train_log.json; convert with training.convert_lora_gguf to ship next to the base model

Deterministic (seeded). Peak GPU memory is a few GB (bf16 base + fp32 adapter); on a laptop RTX 3050 6 GB an epoch over ~17k examples takes a few minutes."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def chatml(question, semantic=False, context=None):
    if context: question = "Previous: " + context + chr(10) + "Q: " + question
    else: question = "Q: " + question
    if semantic:
        from training.semantic_data import SYSTEM
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    from vgpt import query_grammar as G
    return f"<|im_start|>system\n{G.SYSTEM_TUNED}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def encode(tok, rows, max_len=224, semantic=False):
    """-> list of (input_ids, labels) with labels = -100 on everything but the assistant's answer."""
    out = []
    for r in rows:
        p = tok(chatml(r["text"], semantic, r.get("context")), add_special_tokens=False)["input_ids"]; a = tok(r["frame" if semantic else "ir"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > max_len:
            raise ValueError(f"training example exceeds {max_len} tokens; refusing to truncate its target: {r.get('id', r.get('text', '')[:60])}")
        ids = p + a; lab = [-100] * len(p) + a
        if not any(x != -100 for x in lab): raise ValueError("training example has no assistant target tokens")
        out.append((ids, lab))
    return out


def batches(data, bs, rnd, pad_id):
    import torch
    idx = list(range(len(data))); rnd.shuffle(idx)
    chunks = [idx[i:i + bs * 16] for i in range(0, len(idx), bs * 16)]                         # sort inside coarse chunks: less padding, still random
    order = [j for c in chunks for j in sorted(c, key=lambda k: len(data[k][0]))]
    groups = [order[i:i + bs] for i in range(0, len(order), bs)]; rnd.shuffle(groups)
    for g in groups:
        L = max(len(data[i][0]) for i in g)
        ids = torch.full((len(g), L), pad_id, dtype=torch.long); lab = torch.full((len(g), L), -100, dtype=torch.long); att = torch.zeros((len(g), L), dtype=torch.long)
        for k, i in enumerate(g):
            a, b = data[i]; ids[k, :len(a)] = torch.tensor(a); lab[k, :len(b)] = torch.tensor(b); att[k, :len(a)] = 1
        yield ids, att, lab


def answer_logits(model, ids, att, lab, dev):
    """Logits only where the loss is taken (the answer tokens): the vocabulary is 152k wide, a full (batch x length x vocab) tensor would not fit a 6 GB card.
    Returns (logits[n_answer_tokens, vocab], targets[n_answer_tokens])."""
    import torch
    with torch.autocast(dev, dtype=torch.bfloat16):
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        y = lab[:, 1:]; m = y != -100
        lg = model.lm_head(h[m]).float()
    owners = __import__("torch").arange(y.shape[0], device=y.device)[:, None].expand_as(y)[m]
    return lg, y[m], owners


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")); ap.add_argument("--data", default=os.path.join(HERE, "training", "data")); ap.add_argument("--out", default=os.path.join(HERE, "training", "out", "qwen25_05b"))
    ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--init-adapter", default=None, help="start from an existing adapter.safetensors (same rank/alpha) instead of the zero-initialised LoRA; the file is only read")
    ap.add_argument("--semantic", action="store_true", help="train the four-field semantic front door instead of Query IR")
    a = ap.parse_args()
    if a.bs <= 0: ap.error("--bs must be > 0")
    if not math.isfinite(a.epochs) or a.epochs <= 0: ap.error("--epochs must be finite and > 0")
    if not math.isfinite(a.lr) or a.lr <= 0: ap.error("--lr must be finite and > 0")
    if a.warmup < 0: ap.error("--warmup must be >= 0")
    train_path = os.path.join(a.data, "train.jsonl"); val_path = os.path.join(a.data, "val.jsonl"); comp_path = os.path.join(a.data, "test_comp.jsonl")
    train = load_rows(train_path); val = load_rows(val_path); comp = load_rows(comp_path)
    if not train: ap.error("training split is empty")
    norm = lambda x: " ".join(x.lower().split())
    key = lambda r: (norm(r["text"]), json.dumps(r.get("context"), sort_keys=True))
    sets = [{key(r) for r in rows} for rows in (train, val, comp)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]: ap.error("question/context overlaps between training/evaluation splits")
    rnd = random.Random(a.seed)
    if a.max_train: rnd.shuffle(train); train = train[:a.max_train]
    os.makedirs(a.out, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    from training import lora as L
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, f = os.path.split(os.path.abspath(a.base)); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f); model = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.bfloat16).to(dev); model.config.use_cache = False
    wrapped = L.apply_lora(model, a.rank, a.alpha); model.to(dev); n_tr = L.trainable_params(wrapped)
    if a.init_adapter:
        from safetensors.torch import load_file
        L.load_lora_state(wrapped, load_file(a.init_adapter)); print(f"initialised LoRA from {a.init_adapter}", flush=True)
    print(f"loaded {f} in {time.time() - t0:.0f}s; LoRA r={a.rank} alpha={a.alpha} on {len(wrapped)} layers = {n_tr / 1e6:.2f}M trainable parameters ({100 * n_tr / sum(p.numel() for p in model.parameters()):.2f}%)", flush=True)
    tr = encode(tok, train, semantic=a.semantic); va = encode(tok, val[:300], semantic=a.semantic); co = encode(tok, comp[:300], semantic=a.semantic); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if not tr: ap.error("training split produced no usable tokenized examples")
    steps_per_epoch = math.ceil(len(tr) / a.bs); total = math.ceil(steps_per_epoch * a.epochs)
    params = [p for _, m in wrapped for p in (m.lora_A, m.lora_B)]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(s, total) / total)) * 0.95 + 0.05))
    def sha(p):
        if not os.path.exists(p): return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""): h.update(chunk)
        return h.hexdigest()
    log = {"base": f, "base_sha256": sha(a.base), "data_sha256": {os.path.basename(p): sha(p) for p in (train_path, val_path, comp_path)},
           "trainer_sha256": sha(__file__), "rank": a.rank, "alpha": a.alpha, "lr": a.lr, "epochs": a.epochs, "bs": a.bs, "train_examples": len(tr), "seed": a.seed, "steps": [], "val": []}
    def evaluate(dataset):
        model.eval(); cor = n = rows_ok = rows_n = 0; loss_sum = 0.0
        with torch.no_grad():
            for ids, att, lab in batches(dataset, 32, random.Random(0), pad):
                ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
                lg, y, owners = answer_logits(model, ids, att, lab, dev); good = lg.argmax(-1) == y
                loss_sum += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item(); n += int(y.numel()); cor += int(good.sum())
                for owner in range(ids.shape[0]): rows_ok += int(bool(good[owners == owner].all())); rows_n += 1
        model.train(); return loss_sum / max(n, 1), cor / max(n, 1), rows_ok / max(rows_n, 1)
    step = 0; t0 = time.time(); model.train(); ep = 0; done = False; ema = None
    while not done:
        for ids, att, lab in batches(tr, a.bs, rnd, pad):
            ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
            lg, y, _owners = answer_logits(model, ids, att, lab, dev)
            loss = torch.nn.functional.cross_entropy(lg, y); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if step % 25 == 0: print(f"step {step}/{total} loss {ema:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True); log["steps"].append([step, round(ema, 5)])
            if step % max(100, steps_per_epoch // 2) == 0 or step == total:
                vl, vacc, vexact = evaluate(va); cl, cacc, cexact = evaluate(co) if co else (None, None, None)
                print(f"  val loss {vl:.4f} token {vacc:.4f} teacher-forced frame {vexact:.4f}" + (f"; composition token {cacc:.4f} teacher-forced frame {cexact:.4f}" if co else ""), flush=True)
                log["val"].append({"step": step, "loss": round(vl, 5), "token_accuracy": round(vacc, 5), "teacher_forced_frame_exact": round(vexact, 5),
                                   "composition_loss": None if cl is None else round(cl, 5), "composition_token_accuracy": None if cacc is None else round(cacc, 5),
                                   "composition_teacher_forced_frame_exact": None if cexact is None else round(cexact, 5)})
            if step >= total: done = True; break
        ep += 1
    log["train_seconds"] = round(time.time() - t0, 1); log["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None
    save_file(L.lora_state(wrapped), os.path.join(a.out, "adapter.safetensors"))
    json.dump({"base_model": f, "r": a.rank, "lora_alpha": a.alpha, "target_modules": list(L.TARGETS), "architecture": model.config.model_type, "prompt": "training.semantic_data.SYSTEM" if a.semantic else "training.train_lora.chatml (vgpt.query_grammar.SYSTEM_TUNED)"}, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1); print(f"saved {a.out} ({log['train_seconds']}s, peak GPU {log['peak_gpu_gb']} GB)")


if __name__ == "__main__": main()
