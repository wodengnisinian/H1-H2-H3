"""Shared scoring and training primitives for the R21 position-score round."""
import hashlib

import torch


RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))


def pos_score_vector(s, width):
    length = s.numel()
    if width <= length:
        return s[:width]
    return torch.cat([s, torch.zeros(width - length, dtype=s.dtype)])


def make_t(f, w, s):
    t = f.hidden @ w
    t = t + pos_score_vector(s, t.shape[1])
    return t.masked_fill(~f.mask, -torch.inf)


def logits_from_logp(f, logp):
    return f.base + f.amplitude[:, None] * (logp[:, None, :] + f.relation).logsumexp(-1)


def logits_from_t(f, t):
    support = (t[:, None, :] + f.relation).logsumexp(-1) - t.logsumexp(-1)[:, None]
    return f.base + f.amplitude[:, None] * support


def fit_position(features, y, seed, cfg, mode):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    width = features.hidden.shape[1]
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    if mode == "pos_linear":
        a = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        params = [w, a, b]
    else:
        s = torch.zeros(width, requires_grad=True)
        params = [s] if mode == "pos_only" else [w, s]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            hb, rb, mb, bb, ab = (features.hidden[batch], features.relation[batch], features.mask[batch],
                                  features.base[batch], features.amplitude[batch])
            wv = torch.zeros(features.hidden.shape[-1]) if mode == "pos_only" else w
            sv = (a * torch.arange(width, dtype=torch.float32) + b) if mode == "pos_linear" else s
            t = (hb @ wv + sv).masked_fill(~mb, -torch.inf)
            logits = logits_from_t_sliced(bb, ab, t, rb)
            loss = torch.nn.functional.cross_entropy(logits, y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += int((logits.detach().argmax(-1) == y[batch]).sum())
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm())))
    if mode == "pos_linear":
        return w.detach(), (a.detach(), b.detach()), history
    return w.detach(), s.detach(), history


def logits_from_t_sliced(base, amplitude, t, relation):
    support = (t[:, None, :] + relation).logsumexp(-1) - t.logsumexp(-1)[:, None]
    return base + amplitude[:, None] * support


def expand_table(table, length):
    out = torch.zeros(length)
    out[:table.numel()] = table
    return out


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
