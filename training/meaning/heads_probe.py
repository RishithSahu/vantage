"""Experiment: can the frame be read in ONE forward pass (no decode loop) without losing accuracy?

Step 1 (this file): freeze the shipped model (base + adapter frame2r12), take its final hidden state at the position where the answer would start (the end of the prompt), and train small classifier HEADS - one per frame slot - on the
training questions.  If heads on that single vector reach the language model's own accuracy on the 3,842 audit questions, the decode loop (2 to 3 passes of about 150 ms each on the Pi) is not needed at all.

    python training/meaning/heads_probe.py [--extract] [--epochs 40]

Acceptance rule written before the run (2026-10-26): reject if the strict-correct count on the 3,842 audit questions is 15 or more below the baseline of 3,457 (results/meaning/audit1025_r10); no change to the product."""
import argparse, io, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench", "meaning_eval"))
os.environ["VGPT_FRAME"] = "2"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
import eval_r10 as E  # noqa: E402

OUT = os.path.join(ROOT, "results", "heads"); os.makedirs(OUT, exist_ok=True)


def prompts():
    from vgpt.meaning import interface, glossary
    G = glossary.load(); tr = [json.loads(l) for l in io.open(os.path.join(ROOT, "training/meaning/data/frame2_r12/train.jsonl"), encoding="utf-8")]; tr = [t for t in tr if t.get("kind") == "frame"]
    P_tr = [interface.prompt_text("frame", t["text"], t.get("hint", ""), t.get("prev"), vehicle=t.get("vehicle")) for t in tr]; Y_tr = [E.pad(t["answer"]).split() for t in tr]
    rows, P_ev, meta = [], [], []
    for s in E.FROZEN + E.NEW:
        for r in E.load(s):
            tag = r["tag"].split(":")[0]; veh = tag if tag in E.VEHICLES else ("petrol" if s == "icevocab_petrol_v1" else "diesel" if s == "icevocab_diesel_v1" else "ev")
            P_ev.append(interface.prompt_text("frame", r["text"], G.hint_line(r["text"], vehicle=veh), r["prev"], vehicle=veh)); meta.append((s, str(r["id"]), r["text"], r["gold"]))
    return P_tr, Y_tr, P_ev, meta


def extract():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from training import lora as L
    P_tr, Y_tr, P_ev, meta = prompts()
    base = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"); d, f = os.path.split(base)
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f); model = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.bfloat16).cuda().eval()
    w = L.apply_lora(model, 32, 64); model.cuda(); L.load_lora_state(w, load_file(os.path.join(ROOT, "training/out/meaning_frame2r12/adapter.safetensors")))
    def feats(ps, name):
        ids = [tok(p, add_special_tokens=False)["input_ids"] for p in ps]; order = np.argsort([len(x) for x in ids]); out = np.zeros((len(ps), model.config.hidden_size), np.float32); t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(ps), 24):
                b = order[i:i + 24]; L_ = max(len(ids[j]) for j in b); x = torch.full((len(b), L_), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros((len(b), L_), dtype=torch.long)
                for k, j in enumerate(b): x[k, :len(ids[j])] = torch.tensor(ids[j]); att[k, :len(ids[j])] = 1
                h = model.model(input_ids=x.cuda(), attention_mask=att.cuda()).last_hidden_state
                last = att.sum(1) - 1; out[b] = h[torch.arange(len(b)), last.cuda()].float().cpu().numpy()
                if (i // 24) % 100 == 0: print(name, i, len(ps), f"{time.time() - t0:.0f}s", flush=True)
        return out
    np.save(os.path.join(OUT, "train_feats.npy"), feats(P_tr, "train")); np.save(os.path.join(OUT, "eval_feats.npy"), feats(P_ev, "eval"))
    json.dump({"train_y": Y_tr, "meta": meta}, open(os.path.join(OUT, "labels.json"), "w"))


def train_heads(epochs, hidden):
    import torch, torch.nn as nn
    Xt = np.load(os.path.join(OUT, "train_feats.npy")); Xe = np.load(os.path.join(OUT, "eval_feats.npy")); lab = json.load(open(os.path.join(OUT, "labels.json")))
    Y = lab["train_y"]; meta = lab["meta"]; vocab = [sorted({y[i] for y in Y}) for i in range(8)]; idx = [{w: k for k, w in enumerate(v)} for v in vocab]
    yt = np.array([[idx[i][y[i]] for i in range(8)] for y in Y]); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt = (Xt - mu) / sd; Xe = (Xe - mu) / sd
    dev = "cuda"; xt = torch.tensor(Xt, device=dev); xe = torch.tensor(Xe, device=dev); ytt = torch.tensor(yt, device=dev)
    class H(nn.Module):
        def __init__(s):
            super().__init__(); d = Xt.shape[1]
            s.body = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1)) if hidden else nn.Identity(); dd = hidden or d
            s.heads = nn.ModuleList([nn.Linear(dd, len(v)) for v in vocab])
        def forward(s, x): z = s.body(x); return [h(z) for h in s.heads]
    m = H().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4); n = len(xt); torch.manual_seed(0)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]; outs = m(xt[b]); loss = sum(nn.functional.cross_entropy(o, ytt[b, k]) for k, o in enumerate(outs)); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1: print("epoch", ep + 1, "loss", float(loss), flush=True)
    m.eval()
    with torch.no_grad(): pred = [o.argmax(-1).cpu().numpy() for o in m(xe)]
    frames = [" ".join(vocab[k][pred[k][i]] for k in range(8)) for i in range(len(meta))]
    base = json.load(open(os.path.join(ROOT, "results/meaning/audit1025_r10/predictions.json"), encoding="utf-8"))["predictions"]; bm = {(p["set"], str(p["id"]), p["text"]): str(p["strict"]) == "True" for p in base}
    ok = lost = won = 0; slot = np.zeros(8); nslot = 0
    for (s, i, t, g), fr in zip(meta, frames):
        c = E.ok_prefix(fr, g); b = bm.get((s, i, t)); ok += int(c); lost += int(bool(b) and not c); won += int((b is False) and c)
    bc = sum(bm.values()); verdict = "REJECT" if ok <= bc - 15 else ("PASS" if ok >= bc else "PASS with a small loss (flagged)")
    print(f"heads (hidden={hidden}): strict-correct {ok} of {len(meta)} (model baseline {bc}); lost {lost}, won {won}, net {ok - bc:+d} -> {verdict}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--epochs", type=int, default=40); a = ap.parse_args()
    if a.extract: extract()
    for hidden in (0, 1024): train_heads(a.epochs, hidden)
