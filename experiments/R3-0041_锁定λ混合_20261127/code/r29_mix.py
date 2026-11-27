"""R29: mixture readout of sparsemax and entmax(1.5), alpha annealing, learnable
temperature, learned-w initialization. No hand-written temporal lexicon."""
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
    relation_support_loss, relation_hits, question_mean, entmax_logp,
    fit_entmax_cand, fit_relmask,
)


class EntmaxRobust(EntmaxFunction):
    """alpha-entmax with a slightly widened bisection bracket so arbitrary
    fractional alphas (annealing) survive floating-point rounding at the
    lower bound. Gradient is identical to the base implementation."""

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


def fit_mix(features, y, rel_mask, seed, cfg, anneal=False):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w_s = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    w_e = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    lam_param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w_s, w_e, lam_param], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        alpha = 1.5 + 0.5 * (epoch - 1) / (cfg["epochs"] - 1) if anneal else 2.0
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            lam = torch.sigmoid(lam_param)
            t_s = (features.hidden[batch] @ w_s).masked_fill(~rel_mask[batch], -torch.inf)
            t_e = (features.hidden[batch] @ w_e).masked_fill(~rel_mask[batch], -torch.inf)
            p_s = EntmaxRobust.apply(t_s, alpha)
            p_e = EntmaxRobust.apply(t_e, 1.5)
            p = lam * p_s + (1 - lam) * p_e
            logp = p.clamp_min(1e-12).log()
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, nll=loss_sum / len(eligible), hits=hits,
                            lam=float(lam.detach()), alpha=alpha))
    return w_s.detach(), w_e.detach(), lam_param.detach(), history


def fit_anneal(features, y, rel_mask, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        alpha = 1.5 + 0.5 * (epoch - 1) / (cfg["epochs"] - 1)
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = EntmaxRobust.apply(t, alpha).clamp_min(1e-12).log()
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, nll=loss_sum / len(eligible), hits=hits, alpha=alpha,
                            w_norm=float(w.detach().norm())))
    return w.detach(), history


def fit_sparsemax_tau(features, y, rel_mask, seed, cfg):
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
            logp = entmax_logp(t, 2.0)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, nll=loss_sum / len(eligible), hits=hits,
                            w_norm=float(w.detach().norm()), tau=float(log_tau.detach().exp())))
    return w.detach(), log_tau.detach().exp(), history


def fit_sparsemax_winit(features, y, rel_mask, learned_w, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = learned_w.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = entmax_logp(t, 2.0)
            loss, batch_hits = relation_support_loss(features, batch, logp, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            hits += batch_hits
        history.append(dict(epoch=epoch, nll=loss_sum / len(eligible), hits=hits,
                            w_norm=float(w.detach().norm())))
    return w.detach(), history


def save_rows(stream, rows, docs, seed, arm_data):
    arms, pairs, gates, mass_by_arm = arm_data
    hits = Counter()
    for i, (d, r) in enumerate(zip(docs, rows, strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]
        common = dict(version="r29", graph_id=r["graph_id"], graph_index=r["graph_index"],
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
        trained = {}
        trained["relmask_cand"] = fit_relmask(train_features, y_train, rel_mask, seed, cfg)
        trained["entmax15"], _ = fit_entmax_cand(train_features, y_train, rel_mask, seed, cfg, 1.5)
        trained["sparsemax"], _ = fit_entmax_cand(train_features, y_train, rel_mask, seed, cfg, 2.0)
        w_s, w_e, lam, hist_mix = fit_mix(train_features, y_train, rel_mask, seed, cfg, anneal=False)
        trained["mix"] = dict(w_s=w_s, w_e=w_e, lam=lam)
        w_as, w_ae, lam_a, hist_mixa = fit_mix(train_features, y_train, rel_mask, seed, cfg, anneal=True)
        trained["mix_anneal"] = dict(w_s=w_as, w_e=w_ae, lam=lam_a)
        trained["sparsemax_anneal"], hist_ann = fit_anneal(train_features, y_train, rel_mask, seed, cfg)
        trained["sparsemax_tau"], tau, hist_tau = fit_sparsemax_tau(train_features, y_train, rel_mask, seed, cfg)
        trained["sparsemax_tau_tau"] = tau
        trained["sparsemax_winit"], hist_winit = fit_sparsemax_winit(train_features, y_train, rel_mask, learned_w, seed, cfg)
        event(a.out, "trained", seed=seed, mix_lam=float(torch.sigmoid(lam)),
              mix_anneal_lam=float(torch.sigmoid(lam_a)), sparsemax_tau=float(tau))
        (a.out / f"train-{seed}.json").write_text(json.dumps(
            {k: to_jsonable(v) for k, v in trained.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

        f = model.extract(bundle, docs, batch_size=len(docs))
        pairs, masks = masks_for(docs, f.mask.shape[1])
        mentioned = masks.any(1)
        rel_mask_dev = relation_position_mask(docs, f.mask.shape[1])
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
        t_e15 = (f.hidden @ trained["entmax15"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("entmax15", entmax_logp(t_e15, 1.5))
        t_sp = (f.hidden @ trained["sparsemax"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax", entmax_logp(t_sp, 2.0))
        t_ann = (f.hidden @ trained["sparsemax_anneal"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_anneal", entmax_logp(t_ann, 2.0))
        t_tau = (trained["sparsemax_tau_tau"] * (f.hidden @ trained["sparsemax_tau"])).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_tau", entmax_logp(t_tau, 2.0))
        t_winit = (f.hidden @ trained["sparsemax_winit"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_winit", entmax_logp(t_winit, 2.0))

        def mix_arm(name):
            state = trained[name]
            t_s = (f.hidden @ state["w_s"]).masked_fill(~rel_mask_dev, -torch.inf)
            t_e = (f.hidden @ state["w_e"]).masked_fill(~rel_mask_dev, -torch.inf)
            lam = torch.sigmoid(state["lam"])
            p = lam * EntmaxRobust.apply(t_s, 2.0) + (1 - lam) * EntmaxRobust.apply(t_e, 1.5)
            weighted(name, p.clamp_min(1e-12).log())

        mix_arm("mix")
        mix_arm("mix_anneal")
        weighted("random", (f.hidden @ norm_matched_random(learned_w, seed)).masked_fill(~f.mask, -torch.inf).log_softmax(-1))
        weighted("shuffled", shuffled_logp(learned_logp, f.mask, docs, seed))
        arms["pool"] = (f.pooled, None, None)
        arms["first"] = (None, None, None)
        require((model.score(f, learned_w) - arms["learned"][0]).abs().max() <= TOLERANCE,
                "Original formula mismatch")
        require((model.score(f, torch.zeros(64)) - arms["uniform"][0]).abs().max() <= TOLERANCE,
                "Uniform formula mismatch")
        closed = ~f.active
        for name in ("learned", "relmask_cand", "entmax15", "sparsemax", "sparsemax_anneal",
                     "sparsemax_tau", "sparsemax_winit", "mix", "mix_anneal",
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
    metadata = dict(id="R29",
                    variants=["mix", "mix_anneal", "sparsemax_anneal", "sparsemax_tau", "sparsemax_winit"],
                    seeds=SEEDS, cohort_sha256=data_hash, source_sha256=files, script_sha256=code_hash,
                    receipts=receipts, predictions_sha256=sha(combined),
                    hard_constraint="no hand-written temporal/marker lexicon features (先/再/把/被 etc.)",
                    mechanism="distribution-level mixture of sparsemax and entmax(1.5) readouts plus alpha annealing",
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
