"""Frozen B13 features and the 64-parameter SpiderWeb shared-word ranker."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import runpy
from types import ModuleType

import torch
import torch.nn.functional as F


@dataclass
class Bundle:
    trainer: ModuleType
    evaluator: ModuleType
    serial: torch.nn.Module
    vocab: dict
    method: str
    pooled: torch.nn.Module
    pool_vocab: dict
    pool_method: str
    device: torch.device


@lru_cache(maxsize=1)
def _load_sources(archive: Path):
    entry = archive / "code/predict_sealed.py"
    seal = json.loads((archive / "predictions/prediction-seal.json").read_bytes())
    if sha256(entry.read_bytes()).hexdigest() != seal["inference_code_sha256"][entry.name]:
        raise ValueError("B13 inference entry hash mismatch")
    frozen = runpy.run_path(str(entry))
    # PyTorch permits setting interop threads only once in a process.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    evaluator = frozen["load_implementation"](archive / "inputs/b12-code")
    return evaluator, evaluator.frozen_trainer(archive / "inputs/server-run")


def load_bundle(archive: Path, seed: int, device="cpu") -> Bundle:
    archive = Path(archive).resolve()
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("The frozen run uses deterministic, single-threaded CPU inference")
    evaluator, trainer = _load_sources(archive)
    model_root = archive / "inputs/server-run"
    serial, vocab, method = evaluator.load_model(trainer, model_root, device, "serial", seed)
    pooled, pool_vocab, pool_method = evaluator.load_model(trainer, model_root, device, "pooled", seed)
    adapter = serial.layer1
    if trainer.RERANKERS[adapter.reranker_index] != "factorized" or trainer.MODEL_DIM != 64:
        raise ValueError("Expected factorized B13 layer with 64-dimensional question encoding")
    serial.requires_grad_(False)
    pooled.requires_grad_(False)
    return Bundle(trainer, evaluator, serial, vocab, method, pooled, pool_vocab, pool_method, device)


@dataclass
class Features:
    base: torch.Tensor        # [N, 2], frozen directional logits
    old: torch.Tensor         # [N, 2], original B13 layer one
    pooled: torch.Tensor      # [N, 2], separately trained frozen control
    active: torch.Tensor      # [N], structural gate G
    amplitude: torch.Tensor   # [N], G times the frozen sigmoid
    hidden: torch.Tensor      # [N, T, 64], frozen question encoding for every row
    relation: torch.Tensor    # [N, 2, T], zero outside the gate
    mask: torch.Tensor        # [N, T], true at question tokens


@torch.no_grad()
def extract(bundle: Bundle, docs, batch_size=128) -> Features:
    """Extract in caller-supplied candidate order. Gold never enters a model."""
    if not docs or batch_size < 1:
        raise ValueError("Expected documents and a positive batch size")
    trainer = bundle.trainer
    enc = trainer.encode_examples(docs, bundle.vocab, bundle.method, with_alignment=True)
    same_vocabulary = bundle.vocab == bundle.pool_vocab
    pool_enc = enc if same_vocabulary else trainer.encode_examples(docs, bundle.pool_vocab, bundle.pool_method)
    n = len(enc)
    width = max(len(row.question) for row in enc)
    features = Features(
        torch.empty(n, 2), torch.empty(n, 2), torch.empty(n, 2),
        torch.zeros(n, dtype=torch.bool), torch.zeros(n),
        torch.zeros(n, width, trainer.MODEL_DIM), torch.zeros(n, 2, width),
        torch.zeros(n, width, dtype=torch.bool),
    )
    adapter = bundle.serial.layer1
    for start in range(0, n, batch_size):
        batch = enc[start:start + batch_size]
        rows = slice(start, start + len(batch))
        q, nodes = trainer.collate(batch, bundle.device)
        length = q.shape[1]
        ind, distances, mentions = trainer.collate_alignment(batch, length, bundle.device)
        # layer_outputs returns only open-gate contexts. Capture the full H1
        # question encoding from that same base call, before the gate selection.
        def save_question_hidden(_module, _args, output):
            features.hidden[rows, :length].copy_(output[1])

        with adapter.base.register_forward_hook(save_question_hidden):
            z0, old, (h, triples, valid, _, _) = adapter.layer_outputs(q, nodes, ind, distances, mentions)
        features.base[rows] = z0
        features.old[rows] = old
        features.mask[rows, :length] = q.ne(trainer.PAD_ID)
        pool_ids = (q, nodes) if same_vocabulary else trainer.collate(pool_enc[rows], bundle.device)
        features.pooled[rows] = bundle.pooled(*pool_ids)
        if ind.numel():
            active_rows = start + ind
            token_vectors = F.normalize(adapter.question_projection(h), dim=-1).transpose(1, 2)
            predicates = F.normalize(adapter.relation_projection(triples[:, :, 1]), dim=-1)
            match = (predicates @ token_vectors) / trainer.ALIGNMENT_TEMPERATURE
            match += trainer.PREDICATE_MATCH_BIAS * (mentions.transpose(1, 2) & valid.unsqueeze(1))
            gap = (z0[ind, 0] - z0[ind, 1]).abs()
            features.amplitude[active_rows] = (adapter.alpha + adapter.beta * gap).sigmoid()
            features.active[active_rows] = True
            features.relation[active_rows, :, :length] = match
    return features


def _logits(base, amplitude, hidden, relation, mask, w):
    # Mask before both normalizers, so no padded position contributes to B.
    t = (hidden @ w).masked_fill(~mask, -torch.inf)
    B = (t.unsqueeze(1) + relation).logsumexp(-1) - t.logsumexp(-1).unsqueeze(1)
    return base + amplitude.unsqueeze(1) * B


def score(features: Features, w: torch.Tensor) -> torch.Tensor:
    return _logits(features.base, features.amplitude, features.hidden,
                   features.relation, features.mask, w)


def correct_margin_bounds(features: Features, targets):
    """Per-row attainable interval from Eq. 10 for correct-minus-other *total* logit.

    Missing candidates receive NaN. These are necessary bounds, not predictions.
    At zero, the frozen ordering selects candidate 0.
    """
    y = torch.as_tensor(targets, dtype=torch.long)
    sign = torch.where(y == 0, 1., -1.)
    base_gap = sign * (features.base[:, 0] - features.base[:, 1])
    delta = sign.unsqueeze(1) * (features.relation[:, 0] - features.relation[:, 1])
    lo = delta.masked_fill(~features.mask, torch.inf).amin(1)
    hi = delta.masked_fill(~features.mask, -torch.inf).amax(1)
    bounds = torch.stack((base_gap + features.amplitude * lo,
                          base_gap + features.amplitude * hi), dim=1)
    return bounds.masked_fill((y < 0).unsqueeze(1), torch.nan)


def fit(features: Features, targets, *, epochs, lr, batch_size, seed, log=None):
    """Learn only w. Adam minimizes candidate-set NLL on eligible rows.

    The callback receives an epoch record immediately after each pass. Loss
    and hits aggregate each example's forward pass before its minibatch update.
    """
    y = torch.as_tensor(targets, dtype=torch.long)
    if y.shape != features.active.shape or not torch.all((y >= -1) & (y <= 1)):
        raise ValueError("Expected one target from {-1, 0, 1} per document")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Expected positive epochs, learning rate and batch size")
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    if eligible.numel() == 0:
        raise ValueError("No open-gate training rows with a gold candidate")
    generator = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss_sum, hits = 0.0, 0
        indices = eligible[torch.randperm(len(eligible), generator=generator)]
        for batch in indices.split(batch_size):
            logits = _logits(features.base[batch], features.amplitude[batch],
                            features.hidden[batch], features.relation[batch], features.mask[batch], w)
            loss = F.cross_entropy(logits, y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += int((logits.detach().argmax(-1) == y[batch]).sum())
        record = {"epoch": epoch, "examples": len(eligible), "nll": loss_sum / len(eligible),
                  "hits": hits, "w_norm": float(w.detach().norm())}
        history.append(record)
        if log is not None:
            log(record)
    return w.detach(), history
