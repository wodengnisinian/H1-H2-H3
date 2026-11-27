"""R39: mild PARAPHRASE augmentation (1 rewrite per training graph, 1:1)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import runpy
import shutil
import sys
import traceback

import torch
import torch.nn.functional as F

from r26_sharp_cue import (
    SEEDS, RELATIONS, VIEWS, TOLERANCE, MODEL_PY_SHA, WEIGHTS_SHA,
    require, now, sha, json_file, event, check_file, to_jsonable, uniform_log,
    masks_for, relation_position_mask, masses, norm_matched_random,
    shuffled_logp, logits_from_logp, load_cohort,
)
from r27_entmax_combo import (
    relation_support_loss, entmax_logp, fit_entmax_cand, fit_entmax_role, fit_relmask, role_targets,
)
from r29_mix import EntmaxRobust


def paraphrase_augmented(b13, train_docs, train_labels, trainer, evaluator):
    import data_views as dv
    families = runpy.run_path(str(b13 / "inputs/source/generate.py"))["QUESTION_FAMILIES"]
    token_templates = [s.split() for s in families["train"]]
    strata_seen = Counter()
    extra_docs, extra_labels = [], []
    for doc, gold in zip(train_docs, train_labels, strict=True):
        roles = dv.roles_for(doc, token_templates)
        index = strata_seen[doc.stratum] % len(dv.PARAPHRASE)
        strata_seen[doc.stratum] += 1
        para = dv.make_question(doc, roles, dv.PARAPHRASE[index], evaluator)
        require(para.question != doc.question, f"Duplicate rewrite: {doc.doc_id}")
        extra_docs.append(para)
        extra_labels.append(gold)
    return extra_docs, extra_labels


def fit_role_multitask(features, y, rel_mask, role_target, seed, cfg, alpha, gamma):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = entmax_logp(t, alpha)
            role_loss = -logp.masked_fill(~role_target[batch], -torch.inf).logsumexp(-1).mean()
            cand_loss, _ = relation_support_loss(features, batch, logp, y)
            loss = (1 - gamma) * role_loss + gamma * cand_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return w.detach()


def fit_lin2(features, y, rel_mask, w_r, seed, cfg):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w_s = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    lam_param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w_s, lam_param], lr=cfg["lr"])
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            lam = torch.sigmoid(lam_param)
            t_s = (features.hidden[batch] @ w_s).masked_fill(~rel_mask[batch], -torch.inf)
            t_r = (features.hidden[batch] @ w_r).masked_fill(~rel_mask[batch], -torch.inf)
            p = lam * EntmaxRobust.apply(t_s, 2.0) + (1 - lam) * EntmaxRobust.apply(t_r, 2.0)
            loss, _ = relation_support_loss(features, batch, p.clamp_min(1e-12).log(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return w_s.detach(), lam_param.detach()


def fit_lin3(features, y, rel_mask, role_target, w_r, learned_w, seed, cfg, lam3, role_beta):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    rng = torch.Generator().manual_seed(seed)
    w_s = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    lam_params = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([w_s, lam_params], lr=cfg["lr"])
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            lam12 = torch.softmax(lam_params, -1)
            t_s = (features.hidden[batch] @ w_s).masked_fill(~rel_mask[batch], -torch.inf)
            t_r = (features.hidden[batch] @ w_r).masked_fill(~rel_mask[batch], -torch.inf)
            t_l = (features.hidden[batch] @ learned_w).masked_fill(~rel_mask[batch], -torch.inf)
            p_s = EntmaxRobust.apply(t_s, 2.0)
            p_r = EntmaxRobust.apply(t_r, 2.0)
            p_l = t_l.log_softmax(-1).exp()
            p = (1 - lam3) * (lam12[0] * p_s + lam12[1] * p_r) + lam3 * p_l
            logp = p.clamp_min(1e-12).log()
            cand_loss, _ = relation_support_loss(features, batch, logp, y)
            role_loss = -logp.masked_fill(~role_target[batch], -torch.inf).logsumexp(-1).mean()
            loss = cand_loss + role_beta * role_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return w_s.detach(), lam_params.detach()


def save_rows(stream, rows, docs, seed, arm_data):
    arms, pairs, gates, mass_by_arm = arm_data
    hits = Counter()
    for i, (d, r) in enumerate(zip(docs, rows, strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]
        common = dict(version="r39", graph_id=r["graph_id"], graph_index=r["graph_index"],
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
    check_file(a.r29_code / "r26_sharp_cue.py",
               "c52ce21a00e33c4105407b49741ecf96418ebe163e605fc1bc6725bb04a5b376", files)
    check_file(a.r29_code / "r27_entmax_combo.py",
               "ed44b5a5d79cf81f391f129f1fe418c608a0c87a87558257dd94458938b0ebc9", files)
    check_file(a.r29_code / "r29_mix.py",
               "de98a1b79d32f8d15d363c8607be9d220f90f4c07584dcfe475a983215be2f86", files)
    sys.path.insert(0, str(a.prior / "code"))
    sys.path.insert(0, str(a.r29_code))
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
    extra_docs, extra_labels = paraphrase_augmented(a.b13, train_docs, train_labels,
                                                    first_bundle.trainer, first_bundle.evaluator)
    combined_docs = train_docs + extra_docs
    combined_labels = list(train_labels) + extra_labels

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
        base_features = model.extract(bundle, train_docs, batch_size=len(train_docs))
        y_base = torch.tensor([d.candidates.index(lb) if lb in d.candidates else -1
                               for d, lb in zip(train_docs, train_labels, strict=True)])
        rel_mask_base = relation_position_mask(train_docs, base_features.mask.shape[1])
        role_target_base = role_targets(train_docs,
                                        [d.nodes[int(g)][1] for d, g in zip(train_docs, train_labels, strict=True)],
                                        base_features.mask.shape[1])
        aug_features = model.extract(bundle, combined_docs, batch_size=len(combined_docs))
        y_aug = torch.tensor([d.candidates.index(lb) if lb in d.candidates else -1
                              for d, lb in zip(combined_docs, combined_labels, strict=True)])
        rel_mask_aug = relation_position_mask(combined_docs, aug_features.mask.shape[1])
        role_target_aug = role_targets(combined_docs,
                                       [d.nodes[int(g)][1] for d, g in zip(combined_docs, combined_labels, strict=True)],
                                       aug_features.mask.shape[1])
        trained = {}
        trained["relmask_cand"] = fit_relmask(base_features, y_base, rel_mask_base, seed, cfg)
        trained["entmax15"], _ = fit_entmax_cand(base_features, y_base, rel_mask_base, seed, cfg, 1.5)
        trained["sparsemax"], _ = fit_entmax_cand(base_features, y_base, rel_mask_base, seed, cfg, 2.0)
        w_r0, _ = fit_entmax_role(base_features, y_base, rel_mask_base, role_target_base, seed, cfg, 1.5)
        trained["rolew"] = w_r0
        trained["mix2r"] = fit_lin2(base_features, y_base, rel_mask_base, w_r0, seed, cfg)
        trained["lin3_g03_lam10"] = fit_lin3(base_features, y_base, rel_mask_base, role_target_base,
                                             w_r0, learned_w, seed, cfg, lam3=0.10, role_beta=0.3)
        trained["sparsemax_aug"], _ = fit_entmax_cand(aug_features, y_aug, rel_mask_aug, seed, cfg, 2.0)
        w_r_aug = fit_role_multitask(aug_features, y_aug, rel_mask_aug, role_target_aug, seed, cfg, 1.5, 0.5)
        trained["rolew_g05_aug"] = w_r_aug
        trained["mix2r_aug"] = fit_lin2(aug_features, y_aug, rel_mask_aug, w_r_aug, seed, cfg)
        trained["lin3_aug_lam10"] = fit_lin3(aug_features, y_aug, rel_mask_aug, role_target_aug,
                                             w_r_aug, learned_w, seed, cfg, lam3=0.10, role_beta=0.3)
        trained["lin3_aug_lam15"] = fit_lin3(aug_features, y_aug, rel_mask_aug, role_target_aug,
                                             w_r_aug, learned_w, seed, cfg, lam3=0.15, role_beta=0.3)
        event(a.out, "trained", seed=seed, aug_eligible=int((aug_features.active & (y_aug >= 0)).sum()))
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
        t_rw = (f.hidden @ trained["rolew"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("rolew_sparse", entmax_logp(t_rw, 2.0))
        t_spa = (f.hidden @ trained["sparsemax_aug"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("sparsemax_aug", entmax_logp(t_spa, 2.0))
        t_rwa = (f.hidden @ trained["rolew_g05_aug"]).masked_fill(~rel_mask_dev, -torch.inf)
        weighted("rolew_g05_aug", entmax_logp(t_rwa, 2.0))

        def lin2_arm(name):
            w_s, lam_param = trained[name]
            lam = torch.sigmoid(lam_param)
            t_s = (f.hidden @ w_s).masked_fill(~rel_mask_dev, -torch.inf)
            w_r = trained["rolew_g05_aug"] if name == "mix2r_aug" else trained["rolew"]
            t_r = (f.hidden @ w_r).masked_fill(~rel_mask_dev, -torch.inf)
            p = lam * EntmaxRobust.apply(t_s, 2.0) + (1 - lam) * EntmaxRobust.apply(t_r, 2.0)
            weighted(name, p.clamp_min(1e-12).log())

        lin2_arm("mix2r")
        lin2_arm("mix2r_aug")

        def lin3_arm(name, lam3):
            w_s, lam_params = trained[name]
            lam12 = torch.softmax(lam_params, -1)
            w_r = trained["rolew_g05_aug"] if name.startswith("lin3_aug") else trained["rolew"]
            t_s = (f.hidden @ w_s).masked_fill(~rel_mask_dev, -torch.inf)
            t_r = (f.hidden @ w_r).masked_fill(~rel_mask_dev, -torch.inf)
            t_l = (f.hidden @ learned_w).masked_fill(~rel_mask_dev, -torch.inf)
            p = (1 - lam3) * (lam12[0] * EntmaxRobust.apply(t_s, 2.0)
                              + lam12[1] * EntmaxRobust.apply(t_r, 2.0)) + lam3 * t_l.log_softmax(-1).exp()
            weighted(name, p.clamp_min(1e-12).log())

        lin3_arm("lin3_g03_lam10", 0.10)
        lin3_arm("lin3_aug_lam10", 0.10)
        lin3_arm("lin3_aug_lam15", 0.15)
        weighted("random", (f.hidden @ norm_matched_random(learned_w, seed)).masked_fill(~f.mask, -torch.inf).log_softmax(-1))
        weighted("shuffled", shuffled_logp(learned_logp, f.mask, docs, seed))
        arms["pool"] = (f.pooled, None, None)
        arms["first"] = (None, None, None)
        require((model.score(f, learned_w) - arms["learned"][0]).abs().max() <= TOLERANCE,
                "Original formula mismatch")
        require((model.score(f, torch.zeros(64)) - arms["uniform"][0]).abs().max() <= TOLERANCE,
                "Uniform formula mismatch")
        closed = ~f.active
        for name in ("learned", "relmask_cand", "entmax15", "sparsemax", "rolew_sparse",
                     "sparsemax_aug", "rolew_g05_aug", "mix2r", "mix2r_aug",
                     "lin3_g03_lam10", "lin3_aug_lam10", "lin3_aug_lam15",
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
    metadata = dict(id="R39",
                    variants=["mix2r_aug", "lin3_aug_lam10", "lin3_aug_lam15"],
                    seeds=SEEDS, cohort_sha256=data_hash, source_sha256=files, script_sha256=code_hash,
                    receipts=receipts, predictions_sha256=sha(combined),
                    hard_constraint="no hand-written temporal/marker lexicon features (先/再/把/被 etc.)",
                    mechanism="mild PARAPHRASE augmentation (1:1) for sparsemax/rolew and their mixtures",
                    views=dict(Counter(r["view"] for r in cohort_rows)),
                    python=platform.python_version(), torch=str(torch.__version__))
    json_file(a.out / "inference-receipt.json", metadata)
    event(a.out, "complete", predictions_sha256=metadata["predictions_sha256"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("b13", "b16", "prior", "cohort", "r29_code", "out"):
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
