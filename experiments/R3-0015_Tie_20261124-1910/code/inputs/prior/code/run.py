"""训练论文的首步词权重，并把每个种子、每一问的输出写入独立运行目录。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch

from data_views import build_views, build_fresh_views
import model

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "实验协议.json"


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def targets(docs, labels):
    return torch.tensor([d.candidates.index(y) if y in d.candidates else -1
                         for d, y in zip(docs, labels, strict=True)])


def save_inputs(path, views):
    with path.open("x", encoding="utf-8") as stream:
        for name, (docs, labels) in views.items():
            for ex, gold in zip(docs, labels, strict=True):
                if ex.gold is not None:
                    raise ValueError("A model input contains a gold label")
                row = dict(graph_id=ex.doc_id, kind=ex.stratum, variant=name,
                           question=ex.question, candidates=ex.candidates, gold=gold)
                if name == "original":
                    row["nodes"] = ex.nodes
                line(stream, row)


def generate_predictions(stream, bundle, w, docs, labels, variant, seed, batch_size):
    feats = model.extract(bundle, docs, batch_size=batch_size)
    y = targets(docs, labels)
    learned = model.score(feats, w)
    zero = torch.zeros_like(w)
    scores = {"pool": feats.pooled, "direction": feats.base, "old": feats.old,
              "uniform": model.score(feats, zero), "learned": learned}
    if any(v.shape != (len(docs), 2) or not torch.isfinite(v).all() for v in scores.values()):
        raise ValueError("Invalid candidate scores")
    if not torch.equal(learned[~feats.active], feats.base[~feats.active]):
        raise ValueError("Closed-gate identity failed")
    weights = (feats.hidden @ w).masked_fill(~feats.mask, -torch.inf).softmax(-1)
    normalization_error = (weights.sum(1) - 1).abs().max().item()
    if normalization_error > 1e-5 or weights.masked_select(~feats.mask).count_nonzero():
        raise ValueError("Question weights violate mask or normalization")
    bounds = model.correct_margin_bounds(feats, y)
    covered = y >= 0
    margin = (learned[:, 0] - learned[:, 1]) * torch.where(y == 0, 1., -1.)
    if ((margin[covered] < bounds[covered, 0] - 2e-5).any() or
            (margin[covered] > bounds[covered, 1] + 2e-5).any()):
        raise ValueError("Learned correct-minus-other logit gap is outside Eq. 10")
    direction_errors = feats.base.argmax(1).ne(y) & covered
    impossible = (bounds[:, 1] < 0) | ((bounds[:, 1] == 0) & (y == 1))
    # H1 diagnostics only have a word-mass interpretation for the two competing relations.
    rel_masses = []
    for i, (ex, gold) in enumerate(zip(docs, labels, strict=True)):
        cand = ex.candidates
        if feats.active[i] and gold in cand:
            other = cand[1 - cand.index(gold)]
            relations = (ex.nodes[gold][1], ex.nodes[other][1])
            rel_masses.append([sum(float(weights[i, j]) for j, token in enumerate(ex.question) if token == rel)
                               for rel in relations])
        row = {"graph_id": ex.doc_id, "kind": ex.stratum, "seed": seed, "variant": variant,
               "gold": str(gold), "candidates": list(map(str, cand)), "gate": bool(feats.active[i]),
               "predictions": {"first": str(cand[0])}, "logits": {}}
        for method, values in scores.items():
            row["predictions"][method] = str(cand[int(values[i].argmax())])
            row["logits"][method] = values[i].tolist()
        line(stream, row)
    return {"examples": len(docs), "covered": int(covered.sum()), "gate_open": int(feats.active.sum()),
            "direction_errors_covered": int(direction_errors.sum()),
            "direction_errors_unflippable_by_eq10": int((direction_errors & impossible).sum()),
            "gate_open_direction_errors": int((feats.active & direction_errors).sum()),
            "gate_open_unflippable_errors": int((direction_errors & impossible & feats.active).sum()),
            "p_sum_max_abs_error": normalization_error, "p_relation_observations": len(rel_masses),
            "mean_p_gold_relation": float(np.mean(rel_masses, axis=0)[0]) if rel_masses else None,
            "mean_p_other_relation": float(np.mean(rel_masses, axis=0)[1]) if rel_masses else None}


def run(out, archive, b16, note):
    cfg = json.loads(PROTOCOL.read_bytes())
    fit_cfg = cfg["optimization"]
    batches = cfg["feature_batch_size"]
    dump(out / "运行环境.json", {"started": timestamp(), "args": sys.argv, "python": sys.version,
             "platform": platform.platform(), "torch": str(torch.__version__), "numpy": np.__version__,
             "cpu": platform.processor(), "cpu_threads_per_model": 1, "archive": str(archive),
             "b16": str(b16), "protocol": cfg, "code_sha256":
             {p.name: digest(p) for p in sorted((ROOT / "code").glob("*.py"))},
             "protocol_sha256": digest(PROTOCOL)})
    note("started", out=str(out), protocol_sha256=digest(PROTOCOL))
    first_bundle = model.load_bundle(archive, cfg["seeds"][0])
    views, meta = build_views(archive, b16, first_bundle.trainer, first_bundle.evaluator)
    train_docs, train_labels = views.pop("train")
    fresh, fresh_meta = build_fresh_views(archive, first_bundle.trainer, first_bundle.evaluator)
    datasets = {"archive_dev": views, "fresh_dev": fresh}
    manifests = {"archive_dev_and_train": meta, "fresh_dev": fresh_meta}
    train_people = {frozenset(x for h, _, t in d.nodes for x in (h, t)) for d in train_docs}
    dev_people = {key: {frozenset(x for h, _, t in d.nodes for x in (h, t))
                        for d in val["original"][0]} for key, val in datasets.items()}
    if (train_people & dev_people["archive_dev"] or train_people & dev_people["fresh_dev"]
            or not dev_people["archive_dev"].isdisjoint(dev_people["fresh_dev"])):
        raise ValueError("Cross-split six-person sets overlap")
    for key, view_map in datasets.items():
        save_inputs(out / f"{key}-输入.jsonl", view_map)
    dump(out / "数据清单.json", manifests)
    note("data_verified", train=len(train_docs), **{k: len(v["original"][0]) for k,v in datasets.items()})
    y_train = targets(train_docs, train_labels)
    diagnostics = []
    for seed in cfg["seeds"]:
        bundle = first_bundle if seed == cfg["seeds"][0] else model.load_bundle(archive, seed)
        note("train_feature_start", seed=seed)
        train_features = model.extract(bundle, train_docs, batch_size=batches)
        note("train_features", seed=seed, eligible=int((train_features.active & (y_train >= 0)).sum()),
             gate_open=int(train_features.active.sum()), covered=int((y_train >= 0).sum()))
        w, history = model.fit(train_features, y_train, epochs=fit_cfg["epochs"], lr=fit_cfg["lr"],
                              batch_size=fit_cfg["batch_size"], seed=seed, log=lambda r: note("epoch", seed=seed, **r))
        if w.numel() != 64 or not torch.isfinite(w).all() or w.norm() == 0:
            raise ValueError("Invalid learned weight")
        dump(out / f"w-{seed}.json", dict(seed=seed, dtype="float32", w=w.tolist(), training=history))
        del train_features
        gc.collect()
        for dataset, view_map in datasets.items():
            path = out / f"{dataset}-预测.jsonl"
            with path.open("x" if seed == cfg["seeds"][0] else "a", encoding="utf-8") as stream:
                for variant, (docs, labels) in view_map.items():
                    stats = generate_predictions(stream, bundle, w, docs, labels, variant, seed, batches)
                    diagnostics.append(dict(dataset=dataset, seed=seed, variant=variant, **stats))
                    stream.flush()
                    note("view_scored", dataset=dataset, seed=seed, variant=variant, n=len(docs))
        del bundle
        gc.collect()
    dump(out / "分差边界与词权重.json", diagnostics)
    for key in datasets:
        cmd = [sys.executable, str(ROOT / "code/evaluate.py"), str(out / f"{key}-预测.jsonl"),
               str(out / f"{key}-评分.json")]
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONIOENCODING":"utf-8"})
        note("h3_scored", dataset=key, sha256=digest(out / f"{key}-评分.json"))
    dump(out / "结果校验.json", {p.name:digest(p) for p in sorted(out.iterdir())
                              if p.is_file() and p.name != "逐步记录.jsonl"})
    note("finished", seed_count=len(cfg["seeds"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="新建运行目录")
    p.add_argument("--r2", type=Path, default=ROOT.parent / "R2实验")
    args = p.parse_args()
    out = (args.out or ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(exist_ok=False, parents=True)
    archive, = args.r2.glob("13-*")
    b16, = args.r2.glob("16-*")
    def note(event, **fields):
        row = dict(time=timestamp(), event=event, **fields)
        with (out / "逐步记录.jsonl").open("a", encoding="utf-8") as stream:
            line(stream, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    try:
        run(out, archive.resolve(), b16.resolve(), note)
    except BaseException:
        note("failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
