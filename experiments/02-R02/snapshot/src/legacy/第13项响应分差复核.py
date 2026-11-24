"""对第 13 项的合法关系换序题复核方向底座、两层残差与池化的响应分差。

用法：python -B 第13项响应分差复核.py <第13项已解压目录>
以原题金标边为 A，反序题金标边为 B。两次都计算 A 分减 B 分。
D 为原题分差减去反序题分差。输出只写入标准输出。
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

from 第13项关系换序复核 import construct_pairs, read_index
from 第13项逐层复核 import checked_bytes


@torch.inference_mode()
def logits_for_view(examples, model, vocab, method, kind, trainer, device):
    serial = kind == "serial"
    encoded = trainer.encode_examples(examples, vocab, method, with_alignment=serial)
    stages = defaultdict(list)
    for start in range(0, len(encoded), trainer.BATCH_SIZE):
        batch = encoded[start:start + trainer.BATCH_SIZE]
        question, facts = trainer.collate(batch, device)
        if serial:
            alignment = trainer.collate_alignment(batch, question.shape[1], device)
            base = model.layer1.base(question, facts)
            final, active = model(question, facts, *alignment, return_layer_scores=True)
            first = base.index_copy(0, alignment[0], active.layer1_active_logits)
            stages["z0"].append(base)
            stages["z1"].append(first)
            stages["z2"].append(final)
        else:
            stages["pooled"].append(model(question, facts))
    return {name: torch.cat(batches).numpy().astype(np.float64) for name, batches in stages.items()}


def describe(original, reverse, a_index):
    indices = np.arange(len(a_index))
    b_index = 1 - a_index
    margin = original[indices, a_index] - original[indices, b_index]
    reverse_margin = reverse[indices, a_index] - reverse[indices, b_index]
    delta = margin - reverse_margin
    left = original.argmax(axis=1) == a_index
    right = reverse.argmax(axis=1) == b_index
    return {
        "pairs": len(delta),
        "D_positive": int((delta > 0).sum()),
        "D_zero": int((delta == 0).sum()),
        "D_negative": int((delta < 0).sum()),
        "D_mean_logit": float(delta.mean()),
        "D_median_logit": float(np.median(delta)),
        "both_correct": int((left & right).sum()),
        "both_wrong": int((~left & ~right).sum()),
        "only_original_correct": int((left & ~right).sum()),
        "only_reverse_correct": int((~left & right).sum()),
        "tied_original": int((margin == 0).sum()),
        "tied_reverse": int((reverse_margin == 0).sum()),
        "mean_accuracy": float((left.mean() + right.mean()) / 2),
    }


def run(root):
    pred_dir = root / "predictions"
    receipt = json.loads((pred_dir / "prediction-seal.json").read_bytes())
    cohort = root / "cohort"
    seal = json.loads(checked_bytes(cohort / "seal.json", receipt["cohort_seal_sha256"]))
    feature_bytes = checked_bytes(cohort / "features.jsonl", seal["features_sha256"])
    checked_bytes(cohort / "gold.jsonl", seal["gold_sha256"])
    inference = root / "code/predict_sealed.py"
    checked_bytes(inference, receipt["inference_code_sha256"]["predict_sealed.py"])
    frozen = runpy.run_path(str(inference))
    frozen["configure_cpu"]()
    evaluator = frozen["load_implementation"](root / "inputs/b12-code")
    model_root = root / "inputs/server-run"
    trainer = evaluator.frozen_trainer(model_root)
    examples = evaluator.shortlist(frozen["parse_input"](trainer, feature_bytes), "direction")
    templates = runpy.run_path(str(root / "code/generate_cohort.py"))["TEMPLATES"]
    pairs, eligible, excluded = construct_pairs(
        examples, read_index(cohort / "gold.jsonl"), [t.split() for t in templates], evaluator
    )
    a_index = np.array([ex.candidates.index(gold) for ex, _, gold, _ in pairs])
    original = [p[0] for p in pairs]
    reverse = [p[1] for p in pairs]
    report = {"eligible_graphs": eligible, "excluded_graphs": excluded, "by_seed": {}, "aggregate": {}}
    pooled_seeds = defaultdict(list)
    device = torch.device("cpu")
    for seed in frozen["SEEDS"]:
        report["by_seed"][seed] = {}
        for kind in ("serial", "pooled"):
            model, vocab, method = evaluator.load_model(trainer, model_root, device, kind, seed)
            left = logits_for_view(original, model, vocab, method, kind, trainer, device)
            right = logits_for_view(reverse, model, vocab, method, kind, trainer, device)
            filename = f"{kind}-{seed}.jsonl"
            checked_bytes(pred_dir / filename, receipt["prediction_sha256"][filename])
            archived = read_index(pred_dir / filename)
            final = left["z2" if kind == "serial" else "pooled"]
            predictions = trainer.prediction_rows(original, final.argmax(axis=1).tolist())
            if predictions != [archived[ex.doc_id] for ex in original]:
                raise ValueError(f"原问逐题预测不同于归档：{filename}")
            for stage in left:
                pooled_seeds[stage].append((left[stage], right[stage]))
                report["by_seed"][seed][stage] = describe(left[stage], right[stage], a_index)
    for stage, batches in pooled_seeds.items():
        left, right = zip(*batches)
        report["aggregate"][stage] = describe(np.concatenate(left), np.concatenate(right), np.tile(a_index, len(left)))
    for stage, expected in (("z2", 444), ("pooled", 743)):
        if report["aggregate"][stage]["both_correct"] != expected:
            raise ValueError(f"{stage} 双正确数与前次按选择复核结果不一致")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="第13项已解压归档目录")
    print(json.dumps(run(parser.parse_args().archive.resolve()), ensure_ascii=False, indent=2))
