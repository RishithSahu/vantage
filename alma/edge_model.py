"""Frozen contextual encoder with token-role and typed span-edge heads."""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .edge_data import EDGE_LABELS

ROLE_NAMES = ("other", "left", "right", "modifier")


def _masks(offsets, rows, device):
    masks = {role: torch.zeros(offsets.shape[:2], dtype=torch.bool, device=device)
             for role in ("left", "right", "modifier")}
    labels = torch.zeros(offsets.shape[:2], dtype=torch.long, device=device)
    for b, row in enumerate(rows):
        for role_id, role in enumerate(("left", "right", "modifier"), 1):
            lo, hi = row[role]
            hit = (offsets[b, :, 1] > lo) & (offsets[b, :, 0] < hi) & (offsets[b, :, 1] > offsets[b, :, 0])
            masks[role][b] = hit
            labels[b, hit] = role_id
    return masks, labels


def _pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)


class FrozenEdgeParser(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.encoder.requires_grad_(False)
        dim = self.encoder.config.hidden_size
        self.role_head = nn.Linear(dim, len(ROLE_NAMES))
        self.edge_head = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(.1), nn.Linear(dim, 1))

    def checksum(self):
        h = hashlib.sha256()
        for p in self.encoder.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def contextualize(self, texts, device):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=128,
                               return_offsets_mapping=True, return_tensors="pt")
        offsets = batch.pop("offset_mapping").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        self.encoder.eval()
        with torch.no_grad():
            hidden = self.encoder(**batch).last_hidden_state
        return hidden, batch["attention_mask"].bool(), offsets

    def edge_scores(self, hidden, attention, masks):
        mod = _pool(hidden, masks["modifier"])
        left = _pool(hidden, masks["left"])
        right = _pool(hidden, masks["right"])
        global_node = _pool(hidden, attention)
        nodes = torch.stack((global_node, left, right), dim=1)
        m = mod.unsqueeze(1).expand_as(nodes)
        features = torch.cat((m, nodes, torch.abs(m - nodes), m * nodes), dim=-1)
        return self.edge_head(features).squeeze(-1)

    def predicted_masks(self, role_logits, attention):
        pred = role_logits.argmax(-1)
        masks = {}
        for role_id, role in enumerate(("left", "right", "modifier"), 1):
            mask = (pred == role_id) & attention
            missing = ~mask.any(1)
            if missing.any():
                best = role_logits[:, :, role_id].masked_fill(~attention, -1e9).argmax(1)
                mask[torch.arange(mask.shape[0], device=mask.device), best] |= missing
            masks[role] = mask
        return masks


@dataclass
class EdgeStats:
    examples: int
    epochs: int
    seconds: float
    peak_vram_mb: float
    losses: list[float]
    encoder_unchanged: bool


def train_heads(model, rows, epochs=15, batch_size=48, lr=8e-4, device="cuda", seed=317):
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)
    model.to(device)
    before = model.checksum()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    params = list(model.role_head.parameters()) + list(model.edge_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
    losses = []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for items in loader:
            hidden, attention, offsets = model.contextualize([r["text"] for r in items], device)
            masks, role_labels = _masks(offsets, items, device)
            edge_labels = torch.tensor([EDGE_LABELS.index(r["label"]) for r in items], device=device)
            role_logits = model.role_head(hidden)
            edge_logits = model.edge_scores(hidden, attention, masks)
            active = attention & (offsets[:, :, 1] > offsets[:, :, 0])
            role_loss = nn.functional.cross_entropy(role_logits[active], role_labels[active])
            edge_loss = nn.functional.cross_entropy(edge_logits, edge_labels)
            loss = edge_loss + .5 * role_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(items)
            seen += len(items)
        losses.append(total / seen)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return EdgeStats(len(rows), epochs, seconds, peak, losses, before == model.checksum())


def evaluate(model, rows, device="cuda", batch_size=64):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    gold_edge_ok = pred_edge_ok = span_exact = token_tp = token_fp = token_fn = 0
    predictions = []
    confusion = {gold: {pred: 0 for pred in EDGE_LABELS} for gold in EDGE_LABELS}
    model.eval()
    with torch.inference_mode():
        for items in loader:
            hidden, attention, offsets = model.contextualize([r["text"] for r in items], device)
            gold_masks, role_labels = _masks(offsets, items, device)
            role_logits = model.role_head(hidden)
            pred_masks = model.predicted_masks(role_logits, attention & (offsets[:, :, 1] > offsets[:, :, 0]))
            gold_pred = model.edge_scores(hidden, attention, gold_masks).argmax(-1)
            full_pred = model.edge_scores(hidden, attention, pred_masks).argmax(-1)
            gold = torch.tensor([EDGE_LABELS.index(r["label"]) for r in items], device=device)
            gold_edge_ok += int((gold_pred == gold).sum())
            pred_edge_ok += int((full_pred == gold).sum())
            predicted_roles = role_logits.argmax(-1)
            active = attention & (offsets[:, :, 1] > offsets[:, :, 0])
            span_ok = ((predicted_roles == role_labels) | ~active).all(1)
            span_exact += int(span_ok.sum())
            gold_positive = (role_labels > 0) & active
            pred_positive = (predicted_roles > 0) & active
            token_tp += int((gold_positive & pred_positive & (predicted_roles == role_labels)).sum())
            token_fp += int((pred_positive & ((~gold_positive) | (predicted_roles != role_labels))).sum())
            token_fn += int((gold_positive & ((~pred_positive) | (predicted_roles != role_labels))).sum())
            for row, gp, fp, span_is_exact in zip(items, gold_pred.tolist(), full_pred.tolist(), span_ok.tolist()):
                confusion[row["label"]][EDGE_LABELS[fp]] += 1
                predictions.append({"text": row["text"], "gold": row["label"],
                                    "gold_span_pred": EDGE_LABELS[gp], "full_pred": EDGE_LABELS[fp],
                                    "span_exact": span_is_exact})
    n = len(rows)
    f1 = 2 * token_tp / max(2 * token_tp + token_fp + token_fn, 1)
    return {"n": n, "oracle_decoder": 1.0, "gold_spans_pred_edges": gold_edge_ok / n,
            "predicted_span_token_f1": f1, "predicted_spans_exact": span_exact / n,
            "predicted_spans_pred_edges": pred_edge_ok / n,
            "complete_structure_exact": sum(p["span_exact"] and p["full_pred"] == p["gold"] for p in predictions) / n,
            "confusion": confusion,
            "errors": [p for p in predictions if (not p["span_exact"]) or p["full_pred"] != p["gold"]][:25]}
