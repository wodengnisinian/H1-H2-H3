"""R21 H1 evaluation on the paper's development sets.

For each weighted arm compute the softmax token masses of the two mentioned
relations, the selected first relation, and the paired gain over the
first-mentioned baseline with a 10,000-draw stratified graph bootstrap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import torch

SEEDS = (1701, 2701, 3701)
STRATA = ("relation_order", "role_reversal", "competing_chain")
RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))
DRAWS = 10_000
BOOT_SEED = 619482
RANK = 500


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def relation_pair(d):
    return sorted(set(d.question).intersection(RELATIONS))


def main():
    parser = argparse.ArgumentParser()
    for label in ("b13", "b16", "prior", "checkpoints", "out"):
        parser.add_argument("--" + label, type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.prior / "code"))
    sys.path.insert(0, str(Path(__file__).parent))
    import model
    from data_views import build_views, build_fresh_views
    from r21_core import make_t

    first_bundle = model.load_bundle(a.b13, 1701)
    views, _ = build_views(a.b13, a.b16, first_bundle.trainer, first_bundle.evaluator)
    fresh, _ = build_fresh_views(a.b13, first_bundle.trainer, first_bundle.evaluator)
    datasets = {"archive_dev": {k: v for k, v in views.items() if k != "train"}, "fresh_dev": fresh}
    zero_s = torch.zeros(1)
    records = defaultdict(list)

    for seed in SEEDS:
        bundle = first_bundle if seed == 1701 else model.load_bundle(a.b13, seed)
        learned_w = torch.tensor(read_json(a.prior / "runs/本地开发-01" / f"w-{seed}.json")["w"],
                                 dtype=torch.float32)
        ck = {mode: read_json(a.checkpoints / f"{mode}-{seed}.json")
              for mode in ("pos_table", "pos_linear", "pos_only")}
        for dataset, view_map in datasets.items():
            for variant in ("original", "paraphrase"):
                docs, gold = view_map[variant]
                f = model.extract(bundle, docs, batch_size=len(docs))
                logps = {}
                logps["uniform"] = torch.zeros_like(f.mask, dtype=torch.float32).masked_fill(
                    ~f.mask, -torch.inf).log_softmax(-1)
                logps["learned"] = make_t(f, learned_w, zero_s).log_softmax(-1)
                for mode, state in ck.items():
                    w_mode = torch.tensor(state["w"], dtype=torch.float32)
                    s_mode = torch.tensor(state["position_table"], dtype=torch.float32)
                    logps[mode] = make_t(f, w_mode, s_mode).log_softmax(-1)
                for i, (d, g) in enumerate(zip(docs, gold, strict=True)):
                    pair = relation_pair(d)
                    gold_rel = d.nodes[g][1]
                    first_mentioned = pair[0]
                    row = dict(graph_id=d.doc_id, stratum=d.stratum, dataset=dataset, variant=variant,
                               seed=seed, relations=pair, gold_relation=gold_rel,
                               first_mentioned=first_mentioned, arms={})
                    for name, logp in logps.items():
                        m = [0.0, 0.0]
                        for j, token in enumerate(d.question):
                            if token in pair:
                                m[pair.index(token)] += float(logp.exp()[i, j])
                        if abs(m[0] - m[1]) < 1e-12:
                            selected, hit = None, 0.5
                        else:
                            selected = pair[int(m[1] > m[0])]
                            hit = float(selected == gold_rel)
                        row["arms"][name] = dict(masses=m, selected=selected, hit=hit)
                    row["arms"]["first_mentioned"] = dict(
                        masses=None, selected=first_mentioned,
                        hit=float(first_mentioned == gold_rel))
                    records[(dataset, variant)].append(row)
        print("seed done", seed, flush=True)

    result = {}
    for (dataset, variant), rows_list in records.items():
        graphs = sorted({r["graph_id"] for r in rows_list})
        graph_index = {g: i for i, g in enumerate(graphs)}
        strata_groups = {s: [g for g in graphs
                             if next(r for r in rows_list if r["graph_id"] == g)["stratum"] == s]
                         for s in STRATA}
        rng = np.random.default_rng(BOOT_SEED)
        strata_weights = {}
        for s, group in strata_groups.items():
            n = len(group)
            indices = rng.integers(n, size=(DRAWS, n))
            counts = np.zeros((DRAWS, n), dtype=np.uint16)
            np.add.at(counts, (np.arange(DRAWS)[:, None], indices), 1)
            strata_weights[s] = counts.astype(np.float64) / (len(STRATA) * n)
        out_view = {}
        for name in ("uniform", "learned", "pos_table", "pos_linear", "pos_only", "first_mentioned"):
            model_col = np.zeros(len(graphs))
            base_col = np.zeros(len(graphs))
            for g in graphs:
                panel = [r for r in rows_list if r["graph_id"] == g]
                model_col[graph_index[g]] = np.mean([r["arms"][name]["hit"] for r in panel])
                base_col[graph_index[g]] = np.mean([r["arms"]["first_mentioned"]["hit"] for r in panel])
            h1 = float(np.mean([model_col[[graph_index[g] for g in strata_groups[s]]].mean() for s in STRATA]))
            base = float(np.mean([base_col[[graph_index[g] for g in strata_groups[s]]].mean() for s in STRATA]))
            diff = model_col - base_col
            samples = np.zeros(DRAWS)
            for s, group in strata_groups.items():
                local = np.array([diff[graph_index[g]] for g in group])
                samples += strata_weights[s] @ local
            lcb = float(np.partition(samples, RANK - 1)[RANK - 1])
            out_view[name] = dict(h1_accuracy=h1, first_mentioned_accuracy=base, delta=h1 - base,
                                  lcb95=lcb, h1_pass=bool(h1 >= .80 - 1e-12 and lcb > 1e-12))
        result[f"{dataset}.{variant}"] = out_view
    (a.out / "h1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                   encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
