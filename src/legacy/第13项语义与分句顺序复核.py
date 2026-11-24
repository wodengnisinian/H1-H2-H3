"""用第13项的一组措辞交叉改变语义顺序和分句次序。在已解压的归档上只读运行。"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True

import torch

from 第13项关系换序复核 import construct_pairs, matched_roles, predict, read_index


def make_views(pairs, templates, evaluator):
    # 第七种归档措辞包含两个角色标记明确、可单独成句的分句。
    last, first, ask, *extra = templates[6].split(" 。 ")
    if extra or "{second}" not in last or "{first}" not in first or "{" in ask:
        raise ValueError("归档问法与预期不同")
    styles = (" 。 ".join((first, last, ask)), templates[6])
    template_tokens = [template.split() for template in templates]
    views = [[], [], [], []]
    gold = [[], [], [], []]
    archived_view = []
    for ex, _, original_gold, reversed_gold in pairs:
        roles = matched_roles(ex.question, template_tokens)
        anchor, a, b = (roles[k] for k in ("{anchor}", "{first}", "{second}"))
        bag = None
        for j, style in enumerate(styles):
            for k, (r1, r2, target) in enumerate(((a, b, original_gold), (b, a, reversed_gold))):
                q = tuple(style.format(anchor=anchor, first=r1, second=r2).split())
                row = evaluator.shortlist([replace(ex, question=q)], "direction")[0]
                if bag is None:
                    bag = Counter(q)
                if Counter(q) != bag or row.candidates != ex.candidates:
                    raise ValueError(f"词元或有序候选未保持：{ex.doc_id}")
                views[2 * j + k].append(row)
                gold[2 * j + k].append(target)
                if j == 1 and k == 0:
                    archived_view.append(q == ex.question)
    return views, gold, archived_view


def run(root):
    frozen = runpy.run_path(str(root / "code/predict_sealed.py"))
    templates = runpy.run_path(str(root / "code/generate_cohort.py"))["TEMPLATES"]
    frozen["configure_cpu"]()
    evaluator = frozen["load_implementation"](root / "inputs/b12-code")
    model_root = root / "inputs/server-run"
    trainer = evaluator.frozen_trainer(model_root)
    examples = frozen["parse_input"](trainer, (root / "cohort/features.jsonl").read_bytes())
    pairs, eligible, _ = construct_pairs(
        evaluator.shortlist(examples, "direction"), read_index(root / "cohort/gold.jsonl"),
        [t.split() for t in templates], evaluator,
    )
    if eligible != {"relation_order": 1000}:
        raise ValueError(f"归档可换序文档发生变化：{eligible}")
    views, gold, archived_view = make_views(pairs, templates, evaluator)
    if sum(archived_view) != 125:
        raise ValueError("归档中的第七种问法应覆盖 125 道关系顺序题")
    device = torch.device("cpu")
    result = {}
    for kind in ("serial", "pooled"):
        counts = Counter()
        for seed in frozen["SEEDS"]:
            model, vocab, method = evaluator.load_model(trainer, model_root, device, kind, seed)
            predictions = [predict(v, model, vocab, method, kind == "serial", trainer, device) for v in views]
            archived = read_index(root / f"predictions/{kind}-{seed}.jsonl")
            for i, quartet in enumerate(zip(*predictions, strict=True)):
                if archived_view[i] and quartet[2] != archived[quartet[2]["doc_id"]]:
                    raise ValueError("相同题面的预测与归档不同")
                choices = [row["prediction"] for row in quartet]
                hits = [p == targets[i] for p, targets in zip(choices, gold, strict=True)]
                counts["graph_seed"] += 1
                counts["both_meanings_correct_first_step_described_first"] += hits[0] and hits[1]
                counts["both_meanings_correct_first_step_described_last"] += hits[2] and hits[3]
                counts["all_four_correct"] += all(hits)
                counts["matches_first_relation_mentioned_in_all_four"] += hits == [True, True, False, False]
                counts["matches_last_relation_mentioned_in_all_four"] += hits == [False, False, True, True]
                counts["constant_candidate_in_all_four"] += len(set(choices)) == 1
                counts["selection_changed_on_clause_reordering"] += (choices[0] != choices[2]) + (choices[1] != choices[3])
        result[kind] = counts
    return {"template": templates[6], "graphs": len(pairs), "seeds": frozen["SEEDS"], "counts": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="第13项实验归档目录")
    print(json.dumps(run(parser.parse_args().archive.resolve()), indent=2, ensure_ascii=False))
