"""Frozen inference module for the R24 H3 candidate (cue variant)."""
import torch

FIRST = set("先 首先 第一步 开头 此前".split())
SECOND = set("再 随后 之后 第二 最后 结束".split())
BOUNDARIES = set("， 。 ？ ：".split())
RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))


def clause_cues(docs, width):
    cues = torch.zeros(len(docs), width)
    for i, d in enumerate(docs):
        last, strength = -100, 0
        for j, token in enumerate(d.question):
            if token in BOUNDARIES:
                strength = 0
            elif token in FIRST or token in SECOND:
                last, strength = j, 1 if token in FIRST else -1
            if strength:
                cues[i, j] = strength / (1 + 0.10 * (j - last))
    return cues


def relation_position_mask(docs, width):
    mask = torch.zeros(len(docs), width, dtype=torch.bool)
    for i, d in enumerate(docs):
        for j, token in enumerate(d.question):
            if token in RELATIONS:
                mask[i, j] = True
    return mask


def predict(bundle, docs, state):
    import model

    f = model.extract(bundle, docs)
    w = torch.tensor(state["w"], dtype=torch.float32)
    s = torch.tensor(state["s"], dtype=torch.float32)
    rel_mask = relation_position_mask(docs, f.mask.shape[1])
    cues = clause_cues(docs, f.mask.shape[1])
    t = (f.hidden @ w + s * cues).masked_fill(~rel_mask, -torch.inf)
    logp = t.log_softmax(-1)
    return f.base + f.amplitude[:, None] * (logp[:, None, :] + f.relation).logsumexp(-1)
