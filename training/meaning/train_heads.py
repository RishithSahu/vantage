"""Experiment: the frame read in ONE forward pass, with the language model fine-tuned together with one classifier head per slot (no decode loop).

    python training/meaning/train_heads.py --out training/out/heads_r12 --epochs 3

Starts from the shipped adapter (frame2r12), trains LoRA (lr 1e-4) and the eight heads (lr 1e-3) on the same 27,099 frame rows the shipped adapter was trained on, with the loss on the eight slots (sum of cross entropies) taken at the position where the answer would
start.  After every epoch the 3,842 audit questions are read and compared with the language model's own 3,457 strict-correct (results/meaning/audit1025_r10).  Rule written before the run: reject if 15 or more fewer; the product is not changed."""
import argparse, io, json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
import heads_probe as HP  # noqa: E402
E = HP.E


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=float, default=3.0); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--init-adapter", default="training/out/meaning_frame2r12/adapter.safetensors"); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    import torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file
    from training import lora as L
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); rnd = random.Random(a.seed)
    P_tr, Y_tr, P_ev, meta = HP.prompts(); vocab = [sorted({y[i] for y in Y_tr}) for i in range(8)]; idx = [{w: k for k, w in enumerate(v)} for v in vocab]
    base = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"); d, f = os.path.split(base)
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f); model = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.bfloat16).cuda(); model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    w = L.apply_lora(model, 32, 64); model.cuda(); L.load_lora_state(w, load_file(os.path.join(ROOT, a.init_adapter)))
    heads = nn.ModuleList([nn.Linear(model.config.hidden_size, len(v)) for v in vocab]).cuda()
    enc = lambda ps: [tok(p, add_special_tokens=False)["input_ids"] for p in ps]
    ids_tr = enc(P_tr); ids_ev = enc(P_ev); keep = [i for i, x in enumerate(ids_tr) if len(x) <= 560]; print("train rows", len(keep), "of", len(ids_tr), flush=True)
    yt = [[idx[k][Y_tr[i][k]] for k in range(8)] for i in range(len(ids_tr))]
    pad = tok.pad_token_id or 0
    def batch(ids, rows):
        L_ = max(len(ids[j]) for j in rows); x = torch.full((len(rows), L_), pad, dtype=torch.long); att = torch.zeros((len(rows), L_), dtype=torch.long)
        for k, j in enumerate(rows): x[k, :len(ids[j])] = torch.tensor(ids[j]); att[k, :len(ids[j])] = 1
        return x.cuda(), att.cuda()
    def forward(x, att):
        with torch.autocast("cuda", dtype=torch.bfloat16): h = model.model(input_ids=x, attention_mask=att).last_hidden_state
        last = att.sum(1) - 1; z = h[torch.arange(x.shape[0], device=x.device), last].float(); return [hd(z) for hd in heads]
    lora_params = [p for _, m in w for p in (m.lora_A, m.lora_B)]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": heads.parameters(), "lr": a.head_lr}], weight_decay=0.0, betas=(0.9, 0.99))
    steps_per_epoch = math.ceil(len(keep) / a.bs); total = math.ceil(steps_per_epoch * a.epochs); base_lrs = [a.lr, a.head_lr]
    base_pred = json.load(open(os.path.join(ROOT, "results/meaning/audit1025_r10/predictions.json"), encoding="utf-8"))["predictions"]; bm = {(p["set"], str(p["id"]), p["text"]): str(p["strict"]) == "True" for p in base_pred}; bc = sum(bm.values())
    def evaluate(tag):
        model.eval(); order = np.argsort([len(x) for x in ids_ev]); pred = [None] * len(ids_ev)
        with torch.no_grad():
            for i in range(0, len(order), 32):
                r = [int(j) for j in order[i:i + 32]]; x, att = batch(ids_ev, r); outs = forward(x, att); am = [o.argmax(-1).cpu().numpy() for o in outs]
                for k, j in enumerate(r): pred[j] = " ".join(vocab[s][am[s][k]] for s in range(8))
        ok = lost = won = 0
        for (s, i, t, g), fr in zip(meta, pred):
            c = E.ok_prefix(fr, g); b = bm.get((s, i, t)); ok += int(c); lost += int(bool(b) and not c); won += int((b is False) and c)
        verdict = "REJECT (15 or more fewer)" if ok <= bc - 15 else ("PASS, no loss" if ok >= bc else "PASS with a small loss (flagged)")
        print(f"[{tag}] strict-correct {ok} of {len(meta)} (model baseline {bc}); lost {lost}, won {won}, net {ok - bc:+d} -> {verdict}", flush=True)
        json.dump({"pred": pred, "meta": meta}, open(os.path.join(a.out, f"eval_{tag}.json"), "w")); model.train(); return ok
    step = 0; t0 = time.time(); model.train(); heads.train(); ema = None; done = False; ep = 0
    while not done:
        order = list(keep); rnd.shuffle(order); chunks = [order[i:i + a.bs * 16] for i in range(0, len(order), a.bs * 16)]
        groups = [g for c in chunks for g in [sorted(c, key=lambda k: len(ids_tr[k]))[i:i + a.bs] for i in range(0, len(c), a.bs)]]; rnd.shuffle(groups)
        for g in groups:
            x, att = batch(ids_tr, g); outs = forward(x, att); y = torch.tensor([yt[j] for j in g], device="cuda")
            loss = sum(nn.functional.cross_entropy(o, y[:, k]) for k, o in enumerate(outs)); loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(heads.parameters()), 1.0)
            f_ = min(1.0, (step + 1) / 40) * (0.5 * (1 + math.cos(math.pi * min(step, total) / total)) * 0.95 + 0.05)
            for gi, grp in enumerate(opt.param_groups): grp["lr"] = base_lrs[gi] * f_
            opt.step(); opt.zero_grad(set_to_none=True); step += 1; ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if step % 50 == 0: print(f"step {step}/{total} loss {ema:.4f} {time.time() - t0:.0f}s", flush=True)
            if step >= total: done = True; break
        ep += 1; evaluate(f"epoch{ep}")
        save_file(L.lora_state(w), os.path.join(a.out, "adapter.safetensors")); torch.save({"heads": heads.state_dict(), "vocab": vocab}, os.path.join(a.out, "heads.pt"))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__": main()
