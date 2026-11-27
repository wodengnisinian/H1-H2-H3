"""R21 stage A: train position-score heads and score the paper's development sets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r21_core import RELATIONS, fit_position, logits_from_logp, make_t, expand_table, sha


def main():
    parser = argparse.ArgumentParser()
    for label in ("b13", "b16", "prior", "out"):
        parser.add_argument("--" + label, type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.prior / "code"))
    import model
    from data_views import build_views, build_fresh_views

    first_bundle = model.load_bundle(a.b13, 1701)
    views, _ = build_views(a.b13, a.b16, first_bundle.trainer, first_bundle.evaluator)
    fresh, _ = build_fresh_views(a.b13, first_bundle.trainer, first_bundle.evaluator)
    train_docs, train_labels = views["train"]
    datasets = {"archive_dev": {k: v for k, v in views.items() if k != "train"}, "fresh_dev": fresh}
    y_train = torch.tensor([d.candidates.index(lb) if lb in d.candidates else -1
                            for d, lb in zip(train_docs, train_labels, strict=True)])
    max_len = max(len(d.question) for d in train_docs)
    for view_map in datasets.values():
        for docs, _ in view_map.values():
            max_len = max(max_len, max(len(d.question) for d in docs))
    cfg = dict(lr=0.003, epochs=12, batch_size=128)
    zero_s = torch.zeros(1)

    for seed in (1701, 2701, 3701):
        bundle = first_bundle if seed == 1701 else model.load_bundle(a.b13, seed)
        train_features = model.extract(bundle, train_docs, batch_size=len(train_docs))
        trained = {}
        for mode in ("pos_table", "pos_linear", "pos_only"):
            w_mode, s_param, history = fit_position(train_features, y_train, seed, cfg, mode)
            if mode == "pos_linear":
                aa, bb = s_param
                s_table = aa * torch.arange(max_len, dtype=torch.float32) + bb
            else:
                s_table = expand_table(s_param, max_len)
            trained[mode] = (w_mode, s_table)
            (a.out / f"train-{mode}-{seed}.json").write_text(
                json.dumps(dict(seed=seed, mode=mode, w=w_mode.tolist(), position_table=s_table.tolist(),
                                history=history), ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        learned_w = torch.tensor(json.loads((a.prior / "runs/本地开发-01" / f"w-{seed}.json").read_bytes())["w"],
                                 dtype=torch.float32)
        for dataset, view_map in datasets.items():
            path = a.out / f"{dataset}-预测.jsonl"
            with path.open("x" if seed == 1701 else "a", encoding="utf8") as stream:
                for variant, (docs, gold) in view_map.items():
                    f = model.extract(bundle, docs, batch_size=len(docs))
                    arms = {}
                    arms["pool"] = f.pooled
                    arms["direction"] = f.base
                    arms["old"] = f.old
                    arms["uniform"] = model.score(f, torch.zeros(64))
                    arms["learned"] = model.score(f, learned_w)
                    for mode, (w_mode, s_table) in trained.items():
                        t = make_t(f, w_mode, s_table)
                        arms[mode] = logits_from_logp(f, t.log_softmax(-1))
                    for i, (d, g) in enumerate(zip(docs, gold, strict=True)):
                        predictions = {name: str(d.candidates[int(v[i].argmax())])
                                       for name, v in arms.items()}
                        predictions["first"] = str(d.candidates[0])
                        row = dict(graph_id=d.doc_id, kind=d.stratum, seed=seed, variant=variant,
                                   gold=str(g), candidates=list(map(str, d.candidates)),
                                   gate=bool(f.active[i]), predictions=predictions)
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print("seed done", seed, flush=True)
    manifest = {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
