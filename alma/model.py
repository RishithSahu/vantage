"""Compact shared encoder used as an algebra-conditioned semantic-card matcher."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .cards import Card


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class AlmaEncoder(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))

    def tokenize(self, texts, max_length=96):
        return self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )

    def encode_batch(self, batch):
        out = self.encoder(**batch)
        return nn.functional.normalize(mean_pool(out.last_hidden_state, batch["attention_mask"]), dim=-1)

    def encode_texts(self, texts, device=None, batch_size=64):
        device = device or next(self.parameters()).device
        chunks = []
        self.eval()
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                b = {k: v.to(device) for k, v in self.tokenize(texts[i:i + batch_size]).items()}
                chunks.append(self.encode_batch(b).cpu())
        return torch.cat(chunks, dim=0)

    def card_logits(self, texts, cards: list[Card], device=None):
        device = device or next(self.parameters()).device
        te = self.encode_texts(texts, device=device).to(device)
        ce = self.encode_texts([c.description for c in cards], device=device).to(device)
        return (te @ ce.T * self.log_temperature.exp().clamp(1.0, 100.0)).detach().cpu()

    def save_alma(self, output_dir: str, metadata=None):
        enc = os.path.join(output_dir, "encoder")
        os.makedirs(enc, exist_ok=True)
        self.encoder.save_pretrained(enc)
        self.tokenizer.save_pretrained(enc)
        torch.save({"log_temperature": self.log_temperature.detach().cpu()}, os.path.join(output_dir, "alma_head.pt"))
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata or {}, f, indent=2)

    @classmethod
    def load_alma(cls, output_dir: str):
        m = cls(os.path.join(output_dir, "encoder"))
        state = torch.load(os.path.join(output_dir, "alma_head.pt"), map_location="cpu", weights_only=True)
        m.log_temperature.data.copy_(state["log_temperature"])
        return m


class _Rows(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class TrainStats:
    examples: int
    epochs: int
    seconds: float
    peak_vram_mb: float
    losses: list[float]


def train_matcher(model: AlmaEncoder, rows, cards: list[Card], output_dir: str,
                  epochs=5, batch_size=32, lr=2e-5, seed=7, device=None) -> TrainStats:
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    label_to_i = {c.key: i for i, c in enumerate(cards)}
    card_texts = [c.description for c in cards]
    loader = DataLoader(_Rows(rows), batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    started = time.perf_counter()

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        seen = 0
        for items in loader:
            texts = [x["text"] for x in items]
            labels = torch.tensor([label_to_i[x["label"]] for x in items], dtype=torch.long, device=device)
            tb = {k: v.to(device) for k, v in model.tokenize(texts).items()}
            cb = {k: v.to(device) for k, v in model.tokenize(card_texts).items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                te = model.encode_batch(tb)
                ce = model.encode_batch(cb)
                logits = te @ ce.T * model.log_temperature.exp().clamp(1.0, 100.0)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(items)
            seen += len(items)
        losses.append(epoch_loss / max(seen, 1))

    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    stats = TrainStats(len(rows), epochs, seconds, peak, losses)
    model.save_alma(output_dir, metadata=stats.__dict__)
    return stats


def rank_cards(model: AlmaEncoder, texts: list[str], cards: list[Card], device=None):
    logits = model.card_logits(texts, cards, device=device)
    order = torch.argsort(logits, dim=1, descending=True)
    return [[cards[j].key for j in row.tolist()] for row in order], logits


class AlmaPairMatcher(nn.Module):
    """Jointly encode text and one dynamic card for attachment-sensitive scoring."""

    def __init__(self, model_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def tokenize_pairs(self, texts, card_texts, max_length=128):
        return self.tokenizer(
            list(texts), list(card_texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )

    def forward(self, batch):
        out = self.encoder(**batch)
        pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])
        return self.classifier(pooled).squeeze(-1)

    def score(self, texts: list[str], cards: list[Card], device=None, batch_size=64):
        device = device or next(self.parameters()).device
        pairs = [(i, c) for i in range(len(texts)) for c in cards]
        vals = []
        self.eval()
        with torch.inference_mode():
            for p in range(0, len(pairs), batch_size):
                chunk = pairs[p:p + batch_size]
                tb = [texts[i] for i, _ in chunk]
                cb = [c.description for _, c in chunk]
                batch = {k: v.to(device) for k, v in self.tokenize_pairs(tb, cb).items()}
                vals.append(self.forward(batch).cpu())
        return torch.cat(vals).view(len(texts), len(cards))

    def save_alma(self, output_dir: str, metadata=None):
        enc = os.path.join(output_dir, "encoder")
        os.makedirs(enc, exist_ok=True)
        self.encoder.save_pretrained(enc)
        self.tokenizer.save_pretrained(enc)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "pair_head.pt"))
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata or {}, f, indent=2)


def _pair_rows(rows, cards, seed=23, negatives=3):
    rng = random.Random(seed)
    by_kind = {}
    for c in cards:
        by_kind.setdefault(c.kind, []).append(c)
    pairs = []
    for row in rows:
        cards_k = by_kind[row["kind"]]
        gold = next(c for c in cards_k if c.key == row["label"])
        pairs.append({"text": row["text"], "card": gold.description, "y": 1.0})
        neg = [c for c in cards_k if c.key != row["label"]]
        rng.shuffle(neg)
        for c in neg[:negatives]:
            pairs.append({"text": row["text"], "card": c.description, "y": 0.0})
    rng.shuffle(pairs)
    return pairs


def train_pair_matcher(model: AlmaPairMatcher, rows, cards: list[Card], output_dir: str,
                       epochs=4, batch_size=32, lr=2e-5, seed=23, device=None) -> TrainStats:
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    pairs = _pair_rows(rows, cards, seed=seed)
    loader = DataLoader(_Rows(pairs), batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        seen = 0
        for items in loader:
            tb = [x["text"] for x in items]
            cb = [x["card"] for x in items]
            labels = torch.tensor([x["y"] for x in items], dtype=torch.float32, device=device)
            batch = {k: v.to(device) for k, v in model.tokenize_pairs(tb, cb).items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(items)
            seen += len(items)
        losses.append(epoch_loss / max(seen, 1))
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    stats = TrainStats(len(pairs), epochs, seconds, peak, losses)
    model.save_alma(output_dir, metadata=stats.__dict__)
    return stats


def rank_pairs(model: AlmaPairMatcher, texts: list[str], cards: list[Card], device=None):
    logits = model.score(texts, cards, device=device)
    order = torch.argsort(logits, dim=1, descending=True)
    return [[cards[j].key for j in row.tolist()] for row in order], logits


class AlmaClassifier(nn.Module):
    """A small closed-head classifier for a fixed algebraic decision only."""

    def __init__(self, model_path: str, labels: list[str]):
        super().__init__()
        self.labels = list(labels)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(labels))

    def tokenize(self, texts, max_length=128):
        return self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")

    def forward(self, batch):
        out = self.encoder(**batch)
        pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])
        return self.classifier(pooled)

    def predict(self, texts, device=None, batch_size=64):
        device = device or next(self.parameters()).device
        logits = []
        self.eval()
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = {k: v.to(device) for k, v in self.tokenize(texts[i:i + batch_size]).items()}
                logits.append(self.forward(batch).cpu())
        scores = torch.cat(logits)
        order = torch.argsort(scores, dim=1, descending=True)
        return [[self.labels[j] for j in row.tolist()] for row in order], scores


def train_classifier(model: AlmaClassifier, rows, output_dir: str, epochs=6,
                     batch_size=48, lr=2e-5, seed=31, device=None) -> TrainStats:
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    label_to_i = {label: i for i, label in enumerate(model.labels)}
    loader = DataLoader(_Rows(rows), batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for items in loader:
            labels = torch.tensor([label_to_i[x["label"]] for x in items], device=device)
            batch = {k: v.to(device) for k, v in model.tokenize([x["text"] for x in items]).items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.cross_entropy(model(batch), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(items)
            seen += len(items)
        losses.append(total / seen)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    os.makedirs(output_dir, exist_ok=True)
    model.encoder.save_pretrained(os.path.join(output_dir, "encoder"))
    model.tokenizer.save_pretrained(os.path.join(output_dir, "encoder"))
    torch.save({"labels": model.labels, "head": model.classifier.state_dict()},
               os.path.join(output_dir, "classifier.pt"))
    return TrainStats(len(rows), epochs, seconds, peak, losses)
