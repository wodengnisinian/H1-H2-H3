"""在已公开的第 13 项样本上复核交换关系名称后的首跳选择。

用法：python -B 第13项关系换序复核.py <第13项实验目录>
依赖与实验目录中的 environment.json 一致。输出 JSON 到标准输出。
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch


def read_index(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    index = {row["doc_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"重复的文档编号：{path}")
    return index


def directed_paths(nodes, anchor, first, second):
    """枚举两条有向边，要求起点、中间点、终点各不相同。"""
    return [
        (i, j)
        for i, (head, relation, bridge) in enumerate(nodes)
        if head == anchor and relation == first
        for j, (next_head, next_relation, tail) in enumerate(nodes)
        if next_head == bridge and next_relation == second and len({anchor, bridge, tail}) == 3
    ]


def matched_roles(question, templates):
    matches = [
        t for t in templates if len(t) == len(question)
        and all(word.startswith("{") or word == token for word, token in zip(t, question))
    ]
    if len(matches) != 1:
        raise ValueError("题目无法唯一还原为归档中的问法模板")
    return {word: token for word, token in zip(matches[0], question) if word.startswith("{")}


def construct_pairs(examples, gold, templates, evaluator):
    pairs = []
    eligible, no_unique_reverse = Counter(), Counter()
    if {ex.doc_id for ex in examples} != set(gold):
        raise ValueError("特征和金标的文档编号不一致")
    for ex in examples:
        question = ex.question
        roles = matched_roles(question, templates)
        anchor, first, second = (roles[k] for k in ("{anchor}", "{first}", "{second}"))
        original = directed_paths(ex.nodes, anchor, first, second)
        reverse = directed_paths(ex.nodes, anchor, second, first)
        if len(original) != 1 or original[0][0] != gold[ex.doc_id]["gold"]:
            raise ValueError(f"从图重建的原问金标与归档不一致：{ex.doc_id}")
        if len(reverse) != 1:
            no_unique_reverse[ex.stratum] += 1
            continue
        swap = {first: second, second: first}
        new_question = tuple(swap.get(token, token) for token in question)
        if sum(a != b for a, b in zip(question, new_question)) != 2:
            raise ValueError("交换操作未精确改变两个关系词元")
        other = evaluator.shortlist([replace(ex, question=new_question)], "direction")[0]
        g1, g2 = original[0][0], reverse[0][0]
        if ex.candidates != other.candidates or {g1, g2} != set(ex.candidates):
            raise ValueError(f"配对未保持候选或未交换正确首跳：{ex.doc_id}")
        pairs.append((ex, other, g1, g2))
        eligible[ex.stratum] += 1
    return pairs, eligible, no_unique_reverse


def predict(rows, model, vocab, method, serial, trainer, device):
    encoded = trainer.encode_examples(rows, vocab, method, with_alignment=serial)
    with torch.inference_mode():
        choices = trainer.predict_indices(model, encoded, device)
    return trainer.prediction_rows(rows, choices)


def run(root):
    b11 = root / "inputs/server-run"
    frozen = runpy.run_path(str(root / "code/predict_sealed.py"))
    templates = runpy.run_path(str(root / "code/generate_cohort.py"))["TEMPLATES"]
    frozen["configure_cpu"]()
    evaluator = frozen["load_implementation"](root / "inputs/b12-code")
    trainer = evaluator.frozen_trainer(b11)
    features = frozen["parse_input"](trainer, (root / "cohort/features.jsonl").read_bytes())
    examples = evaluator.shortlist(features, "direction")
    pairs, eligible, excluded = construct_pairs(
        examples, read_index(root / "cohort/gold.jsonl"), [t.split() for t in templates], evaluator
    )
    if not pairs:
        raise ValueError("没有可用的换序配对")
    report = {
        "original_gold_checked": len(examples),
        "eligible_graphs": eligible,
        "excluded_without_unique_reverse_path": excluded,
        "models": {},
    }
    device = torch.device("cpu")
    by_graph = {}
    for kind in ("serial", "pooled"):
        seeds, total = {}, Counter()
        by_graph[kind] = np.zeros(len(pairs), dtype=np.int64)
        for seed in frozen["SEEDS"]:
            model, vocab, method = evaluator.load_model(trainer, b11, device, kind, seed)
            archived = read_index(root / f"predictions/{kind}-{seed}.jsonl")
            left = predict([p[0] for p in pairs], model, vocab, method, kind == "serial", trainer, device)
            right = predict([p[1] for p in pairs], model, vocab, method, kind == "serial", trainer, device)
            counts = Counter(pairs=len(pairs))
            for i, ((ex, _, g1, g2), original, reverse) in enumerate(zip(pairs, left, right, strict=True)):
                if original != archived[ex.doc_id]:
                    raise ValueError(f"原问预测未复现归档：{kind}, {seed}, {ex.doc_id}")
                a, b = original["prediction"], reverse["prediction"]
                counts["original_correct"] += a == g1
                counts["reverse_correct"] += b == g2
                counts["prediction_flipped"] += a != b
                counts["both_correct"] += a == g1 and b == g2
                by_graph[kind][i] += a == g1 and b == g2
            seeds[seed] = counts
            total.update(counts)
        report["models"][kind] = {"by_seed": seeds, "all_seeds": total}
    # 每个图的三个固定种子先求平均。换序后保留的图只来自关系顺序层。
    delta = (by_graph["serial"] - by_graph["pooled"]) / len(frozen["SEEDS"])
    rng = np.random.default_rng(619482)
    resampled = np.empty(10_000)
    for start in range(0, len(resampled), 100):
        indices = rng.integers(len(pairs), size=(100, len(pairs)))
        resampled[start:start + 100] = delta[indices].mean(axis=1)
    ordered = np.sort(resampled)
    report["paired_both_correct_difference"] = {
        "contrast": "serial_minus_pooled",
        "difference_pp": round(100 * delta.mean(), 4),
        "one_sided_95_lower_pp": round(100 * ordered[499], 4),
        "two_sided_95_interval_pp": [round(100 * ordered[249], 4), round(100 * ordered[9749], 4)],
        "bootstrap_unit": "graph",
        "bootstrap_draws": len(resampled),
        "bootstrap_seed": 619482,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path, help="已解压的完整第 13 项实验目录")
    args = parser.parse_args()
    print(json.dumps(run(args.experiment_root.resolve()), ensure_ascii=False, indent=2))
