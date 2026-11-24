"""R3 question and candidate scorers. Inference accepts feature-only examples."""
from __future__ import annotations

import hashlib
import math
import torch
import torch.nn.functional as F


def weights(f, w, cue=None, strength=0.):
    logits = f.hidden @ w
    if cue is not None:
        logits = logits + strength * cue
    return logits.masked_fill(~f.mask, -torch.inf).log_softmax(-1)


def apply_logp(f, logp, base=None, amplitude=None):
    support = (logp[:, None, :] + f.relation).logsumexp(-1)
    return (f.base if base is None else base) + (f.amplitude if amplitude is None else amplitude)[:, None] * support


def shuffled_logp(logp, mask, docs, seed):
    out = logp.clone()
    for i, d in enumerate(docs):
        n = int(mask[i].sum())
        key = f"{seed}|{d.doc_id}|{' '.join(d.question)}".encode()
        rng = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        out[i,:n] = logp[i,:n][torch.randperm(n, generator=rng)]
    return out


# These fixed dictionaries are temporal/function words. No template or relation
# name, entity ID, candidate, gold or question-family identifier is consulted.
FIRST = set("先 首先 第一步 开头 此前".split())
SECOND = set("再 随后 之后 第二 最后 结束".split())
BOUNDARIES = set("， 。 ？ ：".split())


def clause_cues(docs, width):
    """Signed nearest preceding temporal marker in the current punctuation span."""
    cues = torch.zeros(len(docs), width)
    for i, d in enumerate(docs):
        last, strength = -100, 0
        for j, token in enumerate(d.question):
            if token in BOUNDARIES:
                strength = 0
            elif token in FIRST or token in SECOND:
                last, strength = j, 1 if token in FIRST else -1
            if strength:
                cues[i,j] = strength / (1 + 0.10 * (j - last))
    return cues


def question_groups(bundle, f):
    queries = bundle.serial.layer1.base.component_queries
    alpha = (f.hidden @ queries.T / math.sqrt(f.hidden.shape[-1])).masked_fill(
        ~f.mask.unsqueeze(-1), -torch.inf).softmax(1)
    return alpha.transpose(1, 2) @ f.hidden


@torch.no_grad()
def single_group(bundle, f):
    """One frozen query, one attention normalization and one 64-D readout."""
    query = bundle.serial.layer1.base.component_queries.mean(0)
    logits = (f.hidden @ query / math.sqrt(f.hidden.shape[-1])).masked_fill(~f.mask,-torch.inf)
    return (logits.softmax(1).unsqueeze(1) @ f.hidden).squeeze(1)


@torch.no_grad()
def one_direction(bundle, docs, f):
    """A single query representation scored against the same directional facts."""
    t = bundle.trainer
    queries = single_group(bundle, f)
    out = []
    for start in range(0, len(docs), 128):
        batch = docs[start:start+128]
        enc = t.encode_examples(batch,bundle.vocab,bundle.method)
        _, nodes = t.collate(enc,bundle.device)
        facts,_ = bundle.serial.layer1.base.encode(nodes)
        triples = facts.view(len(batch),2,3,t.MODEL_DIM)
        vectors = triples.mean(2) + t.DIRECTION_WEIGHT * (triples[:,:,0] - triples[:,:,2])
        query = queries[start:start+len(batch)]
        out.append(F.cosine_similarity(vectors,query[:,None],dim=-1) / t.LOGIT_TEMPERATURE)
    base = torch.cat(out)
    adapter = bundle.serial.layer1
    amp = (adapter.alpha + adapter.beta*(base[:,0]-base[:,1]).abs()).sigmoid() * f.active
    return base, amp


def correct_index(docs, gold):
    return torch.tensor([d.candidates.index(y) if y in d.candidates else -1
                         for d,y in zip(docs,gold,strict=True)])


def fit_joint(f, docs, y, seed, cfg, *, with_cues):
    """One context scalar replaces one hidden dimension: both compared heads have 64 parameters."""
    ids = (f.active & (y>=0)).nonzero().flatten()
    w = torch.zeros(64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=cfg['lr'])
    rng = torch.Generator().manual_seed(seed)
    cues = clause_cues(docs, f.mask.shape[1]) if with_cues else None
    history = []
    for ep in range(cfg['epochs']):
        loss_sum = 0.
        for batch in ids[torch.randperm(len(ids),generator=rng)].split(cfg['batch']):
            h = f.hidden[batch]
            t = h[:,:,:63] @ w[:63] + w[63]*cues[batch] if with_cues else h @ w
            logp = t.masked_fill(~f.mask[batch],-torch.inf).log_softmax(-1)
            support = (logp[:,None,:] + f.relation[batch]).logsumexp(-1)
            z = f.base[batch] + f.amplitude[batch,None]*support
            loss = F.cross_entropy(z,y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())*len(batch)
        history.append({'epoch':ep+1,'nll':loss_sum/len(ids),'w_norm':float(w.detach().norm())})
    return w.detach(),history


def from_state(f, docs, state):
    mode = state['model']
    w = torch.tensor(state['w'],dtype=f.hidden.dtype)
    if mode == 'cue64':
        h = f.hidden[:,:,:63] @ w[:63] + w[63]*clause_cues(docs, f.mask.shape[1])
        return apply_logp(f,h.masked_fill(~f.mask,-torch.inf).log_softmax(-1))
    if mode != 'learned':
        raise ValueError(f"Unknown sealed scorer {mode}")
    return apply_logp(f, weights(f,w))


@torch.no_grad()
def predict(bundle, docs, state):
    import model
    if any(d.gold is not None for d in docs):
        raise ValueError("Prediction API expects feature-only examples")
    return from_state(model.extract(bundle,docs), docs, state)
