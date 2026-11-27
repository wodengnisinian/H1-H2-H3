"""R26: sharpened relation-token readouts without any hand-written temporal lexicon.

Hard constraint (user requirement, this round): no pre-inserted temporal/marker
word list (先/再/把/被 etc.) may appear as an input feature. Order information must
be learned from the frozen Transformer hidden states or come from lexicon-free
structural features (unsigned distance between the two relation mentions).

Variants target the three documented failure mechanisms:
  1. soft allocation dilutes the relation-matching signal -> learned temperature,
     alpha-entmax (alpha=1.5) and epoch-annealed hard selection;
  2. the one-dimensional hand-written temporal cue generalizes poorly -> replaced by
     a learned antisymmetric pairwise readout and an unsigned distance diagnostic;
  3. absolute position tables and raw attention weights are ineffective -> not used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback

import torch
import torch.nn.functional as F

SEEDS = (1701, 2701, 3701)
RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))
STRATA = ("role_reversal", "relation_order", "competing_chain")
VIEWS = ("original", "paraphrase", "unseen_template", "q0", "q1", "q2", "q3")
TOLERANCE = 2e-5
MODEL_PY_SHA = "f85636ffa19443880e48d953c7d552af3d6bda5238c60ad424f445202c2c2b0f"
WEIGHTS_SHA = ("a3cf4167344c14eb0d0e281e9e43ce379289fe40157e39de9a230074c211fa59",
               "1d3a0e23cdb417f229a7231cec039b8d9b7ec803c08b539a4f38ce4fc6f183a9",
               "9217b8edda35fad73650fbe8bdf354e26df0a4f0f8530b6549d62773f93f670e")
ENTMAX_ALPHA = 1.5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_file(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def event(out, name, **details):
    row = dict(utc=now(), event=name, **details)
    with (out / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def check_file(path, expected, receipt):
    digest = sha(path)
    require(digest == expected, f"Frozen file digest mismatch: {path}")
    receipt[str(path)] = digest
    return path


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def uniform_log(mask):
    return torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf).log_softmax(-1)


def masks_for(docs, width):
    pairs = [sorted(set(d.question).intersection(RELATIONS)) for d in docs]
    masks = torch.zeros(len(docs), 2, width, dtype=torch.bool)
    for i, (d, pair) in enumerate(zip(docs, pairs, strict=True)):
        for j, token in enumerate(d.question):
            if token in pair:
                masks[i, pair.index(token), j] = True
    return pairs, masks


def relation_position_mask(docs, width):
    mask = torch.zeros(len(docs), width, dtype=torch.bool)
    for i, d in enumerate(docs):
        for j, token in enumerate(d.question):
            if token in RELATIONS:
                mask[i, j] = True
    return mask


def relation_means(features, docs):
    """Mean hidden vector over every mention of each of the two present relations.
    Slots follow the sorted pair order (A < B by RELATIONS order); no surface order
    and no hand lexicon are used."""
    ha = torch.zeros(len(docs), features.hidden.shape[-1])
    hb = torch.zeros(len(docs), features.hidden.shape[-1])
    for i, d in enumerate(docs):
        pair = sorted(set(d.question).intersection(RELATIONS))
        require(len(pair) == 2, f"Expected two relations in {d.doc_id}")
        for slot, rel in enumerate(pair):
            positions = [j for j, token in enumerate(d.question) if token == rel]
            mean = features.hidden[i, positions].mean(0)
            if slot == 0:
                ha[i] = mean
            else:
                hb[i] = mean
    return ha, hb


def gap_features(docs, width):
    """Lexicon-free unsigned feature: for each relation token, the minimum distance
    to any mention of the other relation, one-hot in 8 bins (distances 1..7, 8+)."""
    x = torch.zeros(len(docs), width, 8)
    for i, d in enumerate(docs):
        pair = sorted(set(d.question).intersection(RELATIONS))
        pos_a = [j for j, t in enumerate(d.question) if t == pair[0]]
        pos_b = [j for j, t in enumerate(d.question) if t == pair[1]]
        for j in pos_a:
            gap = min(abs(j - k) for k in pos_b)
            x[i, j, min(gap - 1, 7)] = 1.0
        for j in pos_b:
            gap = min(abs(j - k) for k in pos_a)
            x[i, j, min(gap - 1, 7)] = 1.0
    return x


class EntmaxFunction(torch.autograd.Function):
    """alpha-entmax with bisection solve. Gradient of p wrt z:
    gz_i = 1/(alpha-1) * p_i^(2-alpha) * (g_i - sum_j g_j p_j^(2-alpha) / sum_k p_k^(2-alpha))."""

    @staticmethod
    def forward(ctx, z, alpha):
        alpha = float(alpha)
        inv = 1.0 / (alpha - 1.0)
        zd = z.double()
        zmax = zd.max(dim=-1, keepdim=True).values

        def surplus(tau):
            v = torch.relu((alpha - 1.0) * (zd - tau))
            return v.pow(inv).sum(-1, keepdim=True)

        lo = zmax - inv
        hi = zmax - 1e-8
        require(bool((surplus(lo) >= 1.0).all()), "entmax bracket low end")
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


def masses(logp, masks):
    return (logp.exp().double()[:, None, :] * masks).sum(-1)


def norm_matched_random(w, seed):
    v = torch.randn(w.shape, generator=torch.Generator().manual_seed(seed + 442))
    return v * (w.norm() / v.norm())


def shuffled_logp(logp, mask, docs, seed):
    out = logp.clone()
    for i, d in enumerate(docs):
        positions = mask[i].nonzero().flatten()
        key = f"{seed}|{d.doc_id}|{' '.join(d.question)}".encode("utf8")
        rng = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        out[i, positions] = logp[i, positions[torch.randperm(len(positions), generator=rng)]]
    return out


def logits_from_logp(f, logp):
    return f.base + f.amplitude[:, None] * (logp[:, None, :] + f.relation).logsumexp(-1)


def relation_support_loss(f, batch, logp, y):
    support = (logp[:, None, :] + f.relation[batch]).logsumexp(-1)
    logits = f.base[batch] + f.amplitude[batch, None] * support
    loss = F.cross_entropy(logits, y[batch])
    hits = int((logits.detach().argmax(-1) == y[batch]).sum())
    return loss, hits


def hard_relation_logp(t, rel_mask, docs):
    """Relation-level hard selection: pick the relation with the larger total quality
    (logsumexp over its token positions); ties follow RELATIONS lexicographic order
    (R12 rule). Mass is uniform over the chosen relation's positions."""
    logp = torch.full_like(t, -torch.inf)
    for i, d in enumerate(docs):
        present = sorted(set(d.question).intersection(RELATIONS))
        require(len(present) == 2, f"Expected two relations in {d.doc_id}")
        totals = []
        for rel in present:
            positions = [j for j, token in enumerate(d.question) if token == rel]
            totals.append(t[i, positions].logsumexp(0))
        chosen = present[1] if totals[1] > totals[0] else present[0]
        positions = [j for j, token in enumerate(d.question) if token == chosen]
        value = -float(torch.log(torch.tensor(len(positions), dtype=t.dtype)))
        logp[i, positions] = torch.full_like(t[i, positions], value)
    return logp


def pair_logp_from_p(p_a, docs, width, pairs):
    """Split predicted A-first probability over the two relations' mention sets."""
    logp = torch.full((len(docs), width), -torch.inf, dtype=torch.float32)
    for i, (d, pair) in enumerate(zip(docs, pairs, strict=True)):
        pos_a = [j for j, t in enumerate(d.question) if t == pair[0]]
        pos_b = [j for j, t in enumerate(d.question) if t == pair[1]]
        va = torch.log(p_a[i].clamp_min(1e-12)) - float(torch.log(torch.tensor(len(pos_a), dtype=p_a.dtype)))
        vb = torch.log((1 - p_a[i]).clamp_min(1e-12)) - float(torch.log(torch.tensor(len(pos_b), dtype=p_a.dtype)))
        logp[i, pos_a] = va
        logp[i, pos_b] = vb
    return logp


def fit_relmask(features, y, rel_mask, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = t.log_softmax(-1)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm())))
    return w.detach(), history


def fit_relmask_tau(features, y, rel_mask, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    log_tau = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, log_tau], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (log_tau.exp() * (features.hidden[batch] @ w)).masked_fill(~rel_mask[batch], -torch.inf)
            logp = t.log_softmax(-1)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm()), tau=float(log_tau.detach().exp())))
    return w.detach(), log_tau.detach().exp(), history


def fit_relmask_entmax(features, y, rel_mask, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            p = EntmaxFunction.apply(t, ENTMAX_ALPHA)
            logp = p.clamp_min(1e-12).log()
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm())))
    return w.detach(), history


def fit_relmask_hard(features, y, rel_mask, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        tau = float(2.0 ** epoch)
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (tau * (features.hidden[batch] @ w)).masked_fill(~rel_mask[batch], -torch.inf)
            logp = t.log_softmax(-1)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm()), tau=tau))
    return w.detach(), history


def fit_relpair_role_tau(features, y, rel_mask, ha, hb, pair_names, gold_rels, docs, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    log_tau = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, log_tau], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            logit = log_tau.exp() * ((ha[batch] - hb[batch]) @ w)
            p_a = torch.sigmoid(logit)
            labels = torch.tensor([float(pair_names[i][0] == gold_rels[i]) for i in batch.tolist()])
            loss = F.binary_cross_entropy(p_a, labels)
            batch_pairs = [pair_names[i] for i in batch.tolist()]
            logp = pair_logp_from_p(p_a, [docs[i] for i in batch.tolist()],
                                    features.mask.shape[1], batch_pairs)
            support = (logp[:, None, :] + features.relation[batch]).logsumexp(-1)
            logits = features.base[batch] + features.amplitude[batch, None] * support
            batch_hits = int((logits.detach().argmax(-1) == y[batch]).sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), loss=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm()), tau=float(log_tau.detach().exp())))
    return w.detach(), log_tau.detach().exp(), history


def fit_relmask_gap(features, y, rel_mask, gap, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    u = torch.zeros(8, requires_grad=True)
    optimizer = torch.optim.Adam([w, u], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w + gap[batch] @ u).masked_fill(~rel_mask[batch], -torch.inf)
            logp = t.log_softmax(-1)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm()), u_norm=float(u.detach().norm())))
    return w.detach(), u.detach(), history


def load_cohort(path):
    read = gzip.open if path.suffix == ".gz" else open
    with read(path, "rt", encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream]
    keys = {(r["graph_id"], r["view"]) for r in rows}
    require(rows and len(rows) == len(keys), "Repeated graph/view key")
    for r in rows:
        require(r["view"] in VIEWS and r["stratum"] in STRATA and len(r["nodes"]) == 12
                and len(set(map(tuple, r["nodes"]))) == 12
                and len(r["candidates"]) == 2 and r["gold"] in r["candidates"], "Malformed cohort row")
    return rows


def save_rows(stream, rows, docs, seed, arm_data):
    arms, pairs, gates, mass_by_arm = arm_data
    hits = Counter()
    for i, (d, r) in enumerate(zip(docs, rows, strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]
        common = dict(version="r26", graph_id=r["graph_id"], graph_index=r["graph_index"],
                      stratum=r["stratum"], view=r["view"], seed=seed, gold=r["gold"],
                      candidates=list(d.candidates), covered=r["gold"] in d.candidates,
                      gate=gates[i], first_relation=r["first_relation"], first_mentioned=r["first_mentioned"],
                      relation_names=pairs[i], uniform_predicted=selected("uniform"),
                      first_predicted=d.candidates[0], pool_predicted=selected("pool"))
        for name, (logits, logp, _) in arms.items():
            if logp is not None:
                values = mass_by_arm[name][i].tolist()
                read = None
                if values[0] != values[1]:
                    read = pairs[i][int(values[1] > values[0])]
                h1_hit = .5 if read is None else float(read == r["first_relation"])
            else:
                values, read, h1_hit = None, None, None
            if name == "first":
                predicted = d.candidates[0]
            elif logits is not None:
                predicted = selected(name)
            else:
                predicted = None
            record = dict(**common, variant=name, relation_masses=values, selected_relation=read,
                          h1_hit=h1_hit, logits=logits[i].tolist() if logits is not None else None,
                          predicted=predicted)
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            hits["prediction_rows"] += 1
    return hits


def main(a):
    event(a.out, "start")
    files = {}
    check_file(a.prior / "code/model.py", MODEL_PY_SHA, files)
    sys.path.insert(0, str(a.prior / "code"))
    import model
    from data_views import build_views
    cohort_rows = load_cohort(a.cohort)
    data_hash = sha(a.cohort)
    code_hash = sha(Path(__file__))
    event(a.out, "inputs_verified", cohort_sha256=data_hash, rows=len(cohort_rows),
          graphs=len({r["graph_id"] for r in cohort_rows}))
    first_bundle = model.load_bundle(a.b13, 1701)
    views, _ = build_views(a.b13, a.b16, first_bundle.trainer, first_bundle.evaluator)
    train_docs, train_labels = views["train"]
    require(len(train_docs) == 12000, "Unexpected training size")
    y_train = torch.tensor([d.candidates.index(lb) if lb in d.candidates else -1
                            for d, lb in zip(train_docs, train_labels, strict=True)])
    gold_rels_train = [d.nodes[int(g)][1] for d, g in zip(train_docs, train_labels, strict=True)]

    def make_docs(bundle, rows):
        docs = [bundle.trainer.Example(r["graph_id"], r["stratum"], tuple(r["question"]),
                                       tuple(map(tuple, r["nodes"])), tuple(r["candidates"]), None)
                for r in rows]
        reranked = bundle.evaluator.shortlist(docs, "direction")
        require(all(d.candidates == e.candidates for d, e in zip(docs, reranked, strict=True)),
                "Supplied candidates differ from historical direction shortlist")
        return docs

    docs = make_docs(first_bundle, cohort_rows)
    receipts = []
    cfg = dict(lr=0.003, epochs=12, batch_size=128)
    for seed in SEEDS:
        event(a.out, "seed_started", seed=seed)
        seed_files = {}
        bundle = first_bundle if seed == SEEDS[0] else model.load_bundle(a.b13, seed)
        for kind in ("serial", "pooled"):
            folder = bundle.evaluator.checkpoint_dir(a.b13 / "inputs/server-run", kind, seed)
            check_file(folder / "best.pt", bundle.evaluator.CHECKPOINTS[kind][seed], seed_files)
        weight_file = check_file(a.prior / "runs/本地开发-01" / f"w-{seed}.json",
                                 WEIGHTS_SHA[SEEDS.index(seed)], seed_files)
        learned_w = torch.tensor(json.loads(weight_file.read_bytes())["w"], dtype=torch.float32)
        train_features = model.extract(bundle, train_docs, batch_size=len(train_docs))
        eligible = int((train_features.active & (y_train >= 0)).sum())
        require(eligible == 8000, f"Unexpected eligible training rows {eligible}")
        rel_mask = relation_position_mask(train_docs, train_features.mask.shape[1])
        train_pairs = [sorted(set(d.question).intersection(RELATIONS)) for d in train_docs]
        ha_train, hb_train = relation_means(train_features, train_docs)
        gap_train = gap_features(train_docs, train_features.mask.shape[1])
        trained = {}
        trained["relmask_cand"] = fit_relmask(train_features, y_train, rel_mask, seed, cfg)[0]
        w_tau, tau, hist_tau = fit_relmask_tau(train_features, y_train, rel_mask, seed, cfg)
        trained["relmask_tau"] = dict(w=w_tau, tau=tau)
        event(a.out, "trained", seed=seed, variant="relmask_tau", final_nll=hist_tau[-1]["nll"], tau=float(tau))
        trained["relmask_entmax"] = fit_relmask_entmax(train_features, y_train, rel_mask, seed, cfg)[0]
        trained["relmask_hard"] = fit_relmask_hard(train_features, y_train, rel_mask, seed, cfg)[0]
        w_pair, tau_pair, hist_pair = fit_relpair_role_tau(train_features, y_train, rel_mask,
                                                           ha_train, hb_train, train_pairs,
                                                           gold_rels_train, train_docs, seed, cfg)
        trained["relpair_role_tau"] = dict(w=w_pair, tau=tau_pair)
        event(a.out, "trained", seed=seed, variant="relpair_role_tau", final_loss=hist_pair[-1]["loss"],
              tau=float(tau_pair))
        w_gap, u_gap, hist_gap = fit_relmask_gap(train_features, y_train, rel_mask, gap_train, seed, cfg)
        trained["relmask_gap"] = dict(w=w_gap, u=u_gap)
        event(a.out, "trained", seed=seed, variant="relmask_gap", final_nll=hist_gap[-1]["nll"])
        (a.out / f"train-{seed}.json").write_text(json.dumps(
            {k: to_jsonable(v) for k, v in trained.items()},
            ensure_ascii=False, indent=2) + "\n", encoding="utf8")

        f = model.extract(bundle, docs, batch_size=len(docs))
        pairs, masks = masks_for(docs, f.mask.shape[1])
        mentioned = masks.any(1)
        rel_mask_dev = relation_position_mask(docs, f.mask.shape[1])
        gap_dev = gap_features(docs, f.mask.shape[1])
        ha_dev, hb_dev = relation_means(f, docs)
        first_positions = torch.tensor([pair.index(next(t for t in d.question if t in pair))
                                        for pair, d in zip(pairs, docs, strict=True)])
        first_p = uniform_log(masks[torch.arange(len(docs)), first_positions])
        uniform = uniform_log(f.mask)
        mention_uniform = uniform_log(mentioned)
        learned_logp = (f.hidden @ learned_w).masked_fill(~f.mask, -torch.inf).log_softmax(-1)
        arms = {}

        def weighted(name, logp):
            arms[name] = (logits_from_logp(f, logp), logp, None)

        weighted("uniform", uniform)
        weighted("uniform_mentions", mention_uniform)
        weighted("first_mentioned", first_p)
        weighted("learned", learned_logp)
        t_relmask = (f.hidden @ trained["relmask_cand"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("relmask_cand", t_relmask.log_softmax(-1))
        state_tau = trained["relmask_tau"]
        t_tau = (state_tau["tau"] * (f.hidden @ state_tau["w"])).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("relmask_tau", t_tau.log_softmax(-1))
        t_ent = (f.hidden @ trained["relmask_entmax"]).masked_fill(~rel_mask_dev, -torch.inf)
        p_ent = EntmaxFunction.apply(t_ent, ENTMAX_ALPHA)
        weighted("relmask_entmax", p_ent.clamp_min(1e-12).log())
        t_hard = (f.hidden @ trained["relmask_hard"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("relmask_hard", hard_relation_logp(t_hard, rel_mask_dev, docs))
        state_pair = trained["relpair_role_tau"]
        logit_pair = state_pair["tau"] * ((ha_dev - hb_dev) @ state_pair["w"])
        p_a = torch.sigmoid(logit_pair)
        weighted("relpair_role_tau", pair_logp_from_p(p_a, docs, f.mask.shape[1], pairs))
        state_gap = trained["relmask_gap"]
        t_gap = (f.hidden @ state_gap["w"] + gap_dev @ state_gap["u"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("relmask_gap", t_gap.log_softmax(-1))
        weighted("random", (f.hidden @ norm_matched_random(learned_w, seed)).masked_fill(~f.mask, -torch.inf).log_softmax(-1))
        weighted("shuffled", shuffled_logp(learned_logp, f.mask, docs, seed))
        arms["pool"] = (f.pooled, None, None)
        arms["first"] = (None, None, None)
        require((model.score(f, learned_w) - arms["learned"][0]).abs().max() <= TOLERANCE,
                "Original formula mismatch")
        require((model.score(f, torch.zeros(64)) - arms["uniform"][0]).abs().max() <= TOLERANCE,
                "Uniform formula mismatch")
        closed = ~f.active
        for name in ("learned", "relmask_cand", "relmask_tau", "relmask_entmax", "relmask_hard",
                     "relpair_role_tau", "relmask_gap", "uniform", "uniform_mentions", "random", "shuffled"):
            require(torch.equal(arms[name][0][closed], f.base[closed]), f"Closed-gate discrepancy in {name}")
        mass_by_arm = {}
        for name, (_, logp, _) in arms.items():
            if logp is not None:
                mass_by_arm[name] = masses(logp, masks)
        arm_data = (arms, pairs, f.active.tolist(), mass_by_arm)
        output = a.out / f"predictions-{seed}.jsonl.gz"
        temp = a.out / f"predictions-{seed}.partial.gz"
        collected = Counter()
        with gzip.open(temp, "wt", encoding="utf8") as stream:
            collected.update(save_rows(stream, cohort_rows, docs, seed, arm_data))
        temp.replace(output)
        receipt = dict(seed=seed, cohort_sha256=data_hash, script_sha256=code_hash,
                       frozen_files_sha256=seed_files, counts=collected, predictions_sha256=sha(output))
        json_file(a.out / f"receipt-{seed}.json", receipt)
        receipts.append(receipt)
        event(a.out, "seed_saved", seed=seed, **collected)
    combined = a.out / "predictions.jsonl.gz"
    pending = a.out / "combined.partial.gz"
    with pending.open("wb") as dst:
        for seed in SEEDS:
            with (a.out / f"predictions-{seed}.jsonl.gz").open("rb") as src:
                shutil.copyfileobj(src, dst)
    pending.replace(combined)
    metadata = dict(id="R26",
                    variants=["relmask_tau", "relmask_entmax", "relmask_hard", "relpair_role_tau", "relmask_gap"],
                    seeds=SEEDS, cohort_sha256=data_hash, source_sha256=files, script_sha256=code_hash,
                    receipts=receipts, predictions_sha256=sha(combined),
                    hard_constraint="no hand-written temporal/marker lexicon features (先/再/把/被 etc.)",
                    mechanism="relation-token readout with sharpening (temperature / entmax / annealed hard) plus learned antisymmetric pairwise order and unsigned distance diagnostic",
                    entmax_alpha=ENTMAX_ALPHA,
                    views=dict(Counter(r["view"] for r in cohort_rows)),
                    python=platform.python_version(), torch=str(torch.__version__))
    json_file(a.out / "inference-receipt.json", metadata)
    event(a.out, "complete", predictions_sha256=metadata["predictions_sha256"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("b13", "b16", "prior", "cohort", "out"):
        parser.add_argument("--" + label, type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--out requires a new or empty directory")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        main(args)
    except Exception:
        json_file(args.out / "failure.json", {"status": "incomplete", "traceback": traceback.format_exc()})
        event(args.out, "failed")
        raise
