"""R28: harden the sparsemax readout's unseen-template generalization with role
supervision, multi-task loss, contrastive normalization and alpha=1.8. No hand lexicon."""
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

from r26_sharp_cue import (
    SEEDS, RELATIONS, VIEWS, TOLERANCE, MODEL_PY_SHA, WEIGHTS_SHA,
    require, now, sha, json_file, event, check_file, to_jsonable, uniform_log,
    masks_for, relation_position_mask, EntmaxFunction, masses, norm_matched_random,
    shuffled_logp, logits_from_logp, load_cohort,
)
from r27_entmax_combo import (
    relation_support_loss, relation_hits, role_targets, question_mean, entmax_logp,
    fit_entmax_cand, fit_entmax_role, fit_entmax_contrast, fit_relmask,
)


class SparsemaxLoss(torch.autograd.Function):
    """Proper sparsemax loss with multi-label target y (row-normalized). Gradient = p - y."""

    @staticmethod
    def forward(ctx, t, rel_mask, y):
        with torch.no_grad():
            p = EntmaxFunction.apply(t.masked_fill(~rel_mask, -torch.inf), 2.0)
        zc = t.masked_fill(~rel_mask, 0.0)
        support = (p > 0).float()
        count = support.sum(-1, keepdim=True).clamp_min(1)
        tau = (support * zc).sum(-1, keepdim=True) - 1.0
        tau = tau / count
        loss = (-(zc * y).sum(-1)
                + 0.5 * ((support * zc * zc).sum(-1) - (tau.squeeze(-1) * tau.squeeze(-1)) * count.squeeze(-1))
                + 0.5).mean()
        ctx.save_for_backward(p, y)
        return loss

    @staticmethod
    def backward(ctx, g):
        p, y = ctx.saved_tensors
        return g * (p - y), None, None


def fit_sparsemax_loss(features, y, rel_mask, role_target, seed, cfg, contrast=False, multi=False):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    q_mean = question_mean(features)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            centered = features.hidden[batch] - q_mean[batch, None, :] if contrast else features.hidden[batch]
            t = (centered @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = entmax_logp(t, 2.0)
            y_target = role_target[batch].float()
            y_target = y_target / y_target.sum(-1, keepdim=True).clamp_min(1)
            role_loss = SparsemaxLoss.apply(t, rel_mask[batch], y_target)
            if multi:
                cand_loss, batch_hits = relation_support_loss(features, batch, logp, y)
                loss = 0.5 * cand_loss + 0.5 * role_loss
            else:
                loss = role_loss
                batch_hits = relation_hits(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, loss=loss_sum / len(eligible), hits=hits, w_norm=float(w.detach().norm())))
    return w.detach(), history


def save_rows(stream, rows, docs, seed, arm_data):
    arms, pairs, gates, mass_by_arm = arm_data
    hits = Counter()
    for i, (d, r) in enumerate(zip(docs, rows, strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]
        common = dict(version="r28", graph_id=r["graph_id"], graph_index=r["graph_index"],
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
    check_file(a.r27_code / "r26_sharp_cue.py",
               "c52ce21a00e33c4105407b49741ecf96418ebe163e605fc1bc6725bb04a5b376", files)
    check_file(a.r27_code / "r27_entmax_combo.py",
               "ed44b5a5d79cf81f391f129f1fe418c608a0c87a87558257dd94458938b0ebc9", files)
    sys.path.insert(0, str(a.prior / "code"))
    sys.path.insert(0, str(a.r27_code))
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
        role_target = role_targets(train_docs, gold_rels_train, train_features.mask.shape[1])
        q_mean_train = question_mean(train_features)
        trained = {}
        trained["relmask_cand"] = fit_relmask(train_features, y_train, rel_mask, seed, cfg)
        trained["sparsemax"], hist_cand = fit_entmax_cand(train_features, y_train, rel_mask, seed, cfg, 2.0)
        trained["sparsemax_role"], hist_role = fit_sparsemax_loss(train_features, y_train, rel_mask,
                                                                  role_target, seed, cfg)
        trained["sparsemax_multi"], hist_multi = fit_sparsemax_loss(train_features, y_train, rel_mask,
                                                                    role_target, seed, cfg, multi=True)
        trained["sparsemax_contrast"], hist_ctr = fit_entmax_contrast(train_features, y_train, rel_mask, q_mean_train, seed, cfg, 2.0)
        trained["sparsemax_role_contrast"], hist_rctr = fit_sparsemax_loss(train_features, y_train, rel_mask,
                                                                           role_target, seed, cfg, contrast=True)
        trained["sparsemax_alpha18"], hist_a18 = fit_entmax_cand(train_features, y_train, rel_mask, seed, cfg, 1.8)
        trained["entmax15_role"], hist_e15r = fit_entmax_role(train_features, y_train, rel_mask,
                                                              role_target, seed, cfg, 1.5)
        event(a.out, "trained", seed=seed, sparsemax_role_loss=hist_role[-1]["loss"],
              sparsemax_multi_loss=hist_multi[-1]["loss"],
              sparsemax_role_contrast_loss=hist_rctr[-1]["loss"])
        (a.out / f"train-{seed}.json").write_text(json.dumps(
            {k: to_jsonable(v) for k, v in trained.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

        f = model.extract(bundle, docs, batch_size=len(docs))
        pairs, masks = masks_for(docs, f.mask.shape[1])
        mentioned = masks.any(1)
        rel_mask_dev = relation_position_mask(docs, f.mask.shape[1])
        q_mean_dev = question_mean(f)
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
        t_sp = (f.hidden @ trained["sparsemax"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax", entmax_logp(t_sp, 2.0))
        t_role = (f.hidden @ trained["sparsemax_role"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_role", entmax_logp(t_role, 2.0))
        t_multi = (f.hidden @ trained["sparsemax_multi"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_multi", entmax_logp(t_multi, 2.0))
        t_ctr = ((f.hidden - q_mean_dev[:, None, :]) @ trained["sparsemax_contrast"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_contrast", entmax_logp(t_ctr, 2.0))
        t_rctr = ((f.hidden - q_mean_dev[:, None, :]) @ trained["sparsemax_role_contrast"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_role_contrast", entmax_logp(t_rctr, 2.0))
        t_a18 = (f.hidden @ trained["sparsemax_alpha18"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_alpha18", entmax_logp(t_a18, 1.8))
        t_e15r = (f.hidden @ trained["entmax15_role"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("rolew_sparse", entmax_logp(t_e15r, 2.0))
        weighted("random", (f.hidden @ norm_matched_random(learned_w, seed)).masked_fill(~f.mask, -torch.inf).log_softmax(-1))
        weighted("shuffled", shuffled_logp(learned_logp, f.mask, docs, seed))
        arms["pool"] = (f.pooled, None, None)
        arms["first"] = (None, None, None)
        require((model.score(f, learned_w) - arms["learned"][0]).abs().max() <= TOLERANCE,
                "Original formula mismatch")
        require((model.score(f, torch.zeros(64)) - arms["uniform"][0]).abs().max() <= TOLERANCE,
                "Uniform formula mismatch")
        closed = ~f.active
        for name in ("learned", "relmask_cand", "sparsemax", "sparsemax_role", "sparsemax_multi",
                     "sparsemax_contrast", "sparsemax_role_contrast", "sparsemax_alpha18", "rolew_sparse",
                     "uniform", "uniform_mentions", "random", "shuffled"):
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
    metadata = dict(id="R28",
                    variants=["sparsemax_role", "sparsemax_multi", "sparsemax_contrast",
                              "sparsemax_role_contrast", "sparsemax_alpha18", "rolew_sparse"],
                    seeds=SEEDS, cohort_sha256=data_hash, source_sha256=files, script_sha256=code_hash,
                    receipts=receipts, predictions_sha256=sha(combined),
                    hard_constraint="no hand-written temporal/marker lexicon features (先/再/把/被 etc.)",
                    mechanism="sparsemax (alpha=2.0) readouts with role/multi-task/contrast hardening",
                    views=dict(Counter(r["view"] for r in cohort_rows)),
                    python=platform.python_version(), torch=str(torch.__version__))
    json_file(a.out / "inference-receipt.json", metadata)
    event(a.out, "complete", predictions_sha256=metadata["predictions_sha256"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("b13", "b16", "prior", "cohort", "r27_code", "out"):
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
