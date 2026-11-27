"""Frozen inference module for the R44 H3 retest (mix2r_aug)."""
import torch

RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))


class Entmax2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, alpha):
        alpha = float(alpha)
        inv = 1.0 / (alpha - 1.0)
        zd = z.double()
        zmax = zd.max(dim=-1, keepdim=True).values

        def surplus(tau):
            v = torch.relu((alpha - 1.0) * (zd - tau))
            return v.pow(inv).sum(-1, keepdim=True)

        lo = zmax - inv - 1e-6
        hi = zmax - 1e-8
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            need_smaller = surplus(mid) < 1.0
            lo = torch.where(need_smaller, lo, mid)
            hi = torch.where(need_smaller, mid, hi)
        tau = 0.5 * (lo + hi)
        p = torch.relu((alpha - 1.0) * (zd - tau)).pow(inv).float()
        ctx.save_for_backward(p)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        alpha = ctx.alpha
        p2 = p.pow(2.0 - alpha).double()
        total = p2.sum(-1, keepdim=True)
        gd = g.double()
        weighted = (gd * p2).sum(-1, keepdim=True) / total
        gz = (1.0 / (alpha - 1.0)) * p2 * (gd - weighted)
        return gz.float(), None


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
    w_s = torch.tensor(state["w_s"], dtype=torch.float32)
    w_r = torch.tensor(state["w_r"], dtype=torch.float32)
    lam = float(state["lam"])
    rel_mask = relation_position_mask(docs, f.mask.shape[1])
    with torch.no_grad():
        t_s = (f.hidden @ w_s).masked_fill(~rel_mask, -torch.inf)
        t_r = (f.hidden @ w_r).masked_fill(~rel_mask, -torch.inf)
        p = lam * Entmax2.apply(t_s, 2.0) + (1 - lam) * Entmax2.apply(t_r, 2.0)
        logp = p.clamp_min(1e-12).log()
    return f.base + f.amplitude[:, None] * (logp[:, None, :] + f.relation).logsumexp(-1)
