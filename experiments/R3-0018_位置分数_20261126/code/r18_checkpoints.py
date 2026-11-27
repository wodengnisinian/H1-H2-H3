"""Re-run R18 training deterministically and save per-seed checkpoints (w, s, history)."""
import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    for label in ("b13", "b16", "prior", "out"):
        parser.add_argument("--" + label, type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r18_pos_score import (SEEDS, MODEL_PY_SHA, WEIGHTS_SHA, check_file, event,
                               fit_position, json_file, sha)
    sys.path.insert(0, str(a.prior / "code"))
    import model
    from data_views import build_views
    event(a.out, "start")
    first_bundle = model.load_bundle(a.b13, SEEDS[0])
    views, _ = build_views(a.b13, a.b16, first_bundle.trainer, first_bundle.evaluator)
    train_docs, train_labels = views["train"]
    y_train = torch.tensor([d.candidates.index(lb) if lb in d.candidates else -1
                            for d, lb in zip(train_docs, train_labels, strict=True)])
    max_len = max(len(d.question) for d in train_docs)
    cfg = dict(lr=0.003, epochs=12, batch_size=128)
    manifest = {}
    for seed in SEEDS:
        bundle = first_bundle if seed == SEEDS[0] else model.load_bundle(a.b13, seed)
        train_features = model.extract(bundle, train_docs, batch_size=len(train_docs))
        eligible = int((train_features.active & (y_train >= 0)).sum())
        assert eligible == 8000
        for mode in ("pos_table", "pos_linear", "pos_only"):
            w_mode, s_param, history = fit_position(train_features, y_train, seed, cfg, mode)
            if mode == "pos_linear":
                aa, bb = s_param
                s_table = (aa * torch.arange(max_len, dtype=torch.float32) + bb).tolist()
                extra = {"a": float(aa), "b": float(bb)}
            else:
                s_table = [0.0] * max_len
                table = s_param.tolist()
                s_table[:len(table)] = table
                extra = {}
            checkpoint = dict(seed=seed, mode=mode, w=w_mode.tolist(), position_table=s_table,
                              history=history, eligible=eligible, max_len=max_len, **extra)
            path = a.out / f"{mode}-{seed}.json"
            json_file(path, checkpoint)
            manifest[path.name] = sha(path)
            event(a.out, "saved", seed=seed, mode=mode)
    json_file(a.out / "manifest.json", manifest)
    event(a.out, "complete")


if __name__ == "__main__":
    main()
