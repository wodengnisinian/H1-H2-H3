"""R22: relation-word-masked soft readout without any explicit order feature.

t_j = w . h_j is kept, but the softmax runs only over the positions where one of
the two mentioned relations appears. Training mirrors the paper budget.
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


def fit_relmask(features, y, rel_mask, role_target, seed, cfg, mode):
    eligible = (features.active & (y >= 0)).nonzero().flatten()
    width = features.hidden.shape[1]
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(features.hidden.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=cfg["lr"])
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, hits = 0.0, 0
        for batch in eligible[torch.randperm(len(eligible), generator=rng)].split(cfg["batch_size"]):
            t = (features.hidden[batch] @ w).masked_fill(~rel_mask[batch], -torch.inf)
            logp = t.log_softmax(-1)
            if mode == "cand":
                support = (logp[:, None, :] + features.relation[batch]).logsumexp(-1)
                logits = features.base[batch] + features.amplitude[batch, None] * support
                loss = F.cross_entropy(logits, y[batch])
                hits += int((logits.detach().argmax(-1) == y[batch]).sum())
            else:
                loss = -logp.masked_fill(~role_target[batch], -torch.inf).logsumexp(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        history.append(dict(epoch=epoch, examples=len(eligible), nll=loss_sum / len(eligible),
                            hits=hits, w_norm=float(w.detach().norm())))
    return w.detach(), history


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
        mentioned = [t for t in r["question"] if t in RELATIONS]
        require(len(set(mentioned)) == 2 and r["first_mentioned"] == mentioned[0]
                and r["first_relation"] in mentioned and r["nodes"][r["gold"]][1] == r["first_relation"],
                "Invalid relation labels")
    return rows


def save_rows(stream, rows, docs, seed, arm_data):
    arms, pairs, gates, mass_by_arm = arm_data
    hits = Counter()
    for i, (d, r) in enumerate(zip(docs, rows, strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]
        common = dict(version="r22", graph_id=r["graph_id"], graph_index=r["graph_index"],
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
        role_target = torch.zeros_like(train_features.mask)
        for i, (d, g) in enumerate(zip(train_docs, train_labels, strict=True)):
            rel = d.nodes[g][1]
            for j, token in enumerate(d.question):
                if token == rel:
                    role_target[i, j] = True
        trained = {}
        for mode in ("cand", "role"):
            w_mode, history = fit_relmask(train_features, y_train, rel_mask, role_target, seed, cfg, mode)
            trained[f"relmask_{mode}"] = w_mode
            (a.out / f"train-relmask_{mode}-{seed}.json").write_text(
                json.dumps(dict(seed=seed, mode=mode, w=w_mode.tolist(), history=history),
                           ensure_ascii=False, indent=2) + "\n", encoding="utf8")
            event(a.out, "trained", seed=seed, mode=mode, final_nll=history[-1]["nll"],
                  final_w_norm=history[-1]["w_norm"])
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
        for mode in ("cand", "role"):
            w_mode = trained[f"relmask_{mode}"]
            t = (f.hidden @ w_mode).masked_fill(~rel_mask_dev, -torch.inf)
            weighted(f"relmask_{mode}", t.log_softmax(-1))
        weighted("random", (f.hidden @ norm_matched_random(learned_w, seed)).masked_fill(~f.mask, -torch.inf).log_softmax(-1))
        weighted("shuffled", shuffled_logp(learned_logp, f.mask, docs, seed))
        arms["pool"] = (f.pooled, None, None)
        arms["first"] = (None, None, None)
        require((model.score(f, learned_w) - arms["learned"][0]).abs().max() <= TOLERANCE,
                "Original formula mismatch")
        require((model.score(f, torch.zeros(64)) - arms["uniform"][0]).abs().max() <= TOLERANCE,
                "Uniform formula mismatch")
        closed = ~f.active
        for name in ("learned", "relmask_cand", "relmask_role",
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
    metadata = dict(id="R22", primary_variant="relmask_cand", seeds=SEEDS, cohort_sha256=data_hash,
                    source_sha256=files, script_sha256=code_hash, receipts=receipts,
                    predictions_sha256=sha(combined),
                    mechanism="t_j = w.h_j, softmax restricted to the two relation mentions; no order features",
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
