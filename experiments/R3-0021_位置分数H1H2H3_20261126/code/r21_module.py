"""Frozen inference module for the R21 H3 candidate (position-score variant)."""
import torch


def predict(bundle, docs, state):
    import model
    from r21_core import make_t, logits_from_logp

    f = model.extract(bundle, docs)
    w = torch.tensor(state["w"], dtype=torch.float32)
    s = torch.tensor(state["position_table"], dtype=torch.float32)
    t = make_t(f, w, s)
    return logits_from_logp(f, t.log_softmax(-1))
