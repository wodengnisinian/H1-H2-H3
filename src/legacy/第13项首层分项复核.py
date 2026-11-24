"""复算第13项第一层显著度、关系匹配对合法换序的响应。用法见 --help。"""

import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import torch

from 第13项关系换序复核 import construct_pairs, read_index
from 第13项逐层复核 import checked_bytes


NAMES = ("底座", "门控显著度", "门控关系匹配", "第一层", "仅加显著度", "仅加关系匹配")


@torch.inference_mode()
def decompose(examples, signs, model, vocab, method, trainer, device):
    adapter = model.layer1
    if trainer.RERANKERS[adapter.reranker_index] != "factorized":
        raise ValueError("本分解只适用于冻结的分解式对齐器")
    encoded = trainer.encode_examples(examples, vocab, method, with_alignment=True)
    margins = []
    max_error = 0.0
    for offset in range(0, len(encoded), trainer.BATCH_SIZE):
        batch = encoded[offset:offset + trainer.BATCH_SIZE]
        questions, nodes = trainer.collate(batch, device)
        alignment = trainer.collate_alignment(batch, questions.shape[1], device)
        if len(alignment[0]) != len(batch):
            raise ValueError("合法换序配对中出现了未触发门控的题目")

        # 直接捕获冻结实现计算的两个 logmeanexp；其余两次调用属于其他备选评分器。
        reductions = []
        logmeanexp = adapter.logmeanexp

        def capture(values, mask):
            result = logmeanexp(values, mask)
            reductions.append(result)
            return result

        with patch.object(adapter, "logmeanexp", side_effect=capture):
            base, first, _ = adapter.layer_outputs(questions, nodes, *alignment)
        salience, relation, _, _ = reductions
        gate = (adapter.alpha + adapter.beta * (base[:, 0] - base[:, 1]).abs()).sigmoid()
        reconstructed = base + gate[:, None] * (salience + relation)
        torch.testing.assert_close(first, reconstructed, rtol=1e-6, atol=1e-6)
        max_error = max(max_error, (first - reconstructed).abs().max().item())

        direction = signs[offset:offset + len(batch)]

        def margin(scores):
            return direction * (scores[:, 0] - scores[:, 1])

        margins.append(torch.stack(
            (margin(base), gate * margin(salience), gate * margin(relation), margin(first),
             margin(base + gate[:, None] * salience), margin(base + gate[:, None] * relation)), dim=1
        ).double().numpy())
    return np.concatenate(margins), max_error


def summarize(left, right, signs):
    delta = left - right
    # torch.argmax 在完全同分时取第一个候选。sign=+1 表示原问金标是第一个候选。
    models = {}
    for index in (0, 3, 4, 5):
        correct_left = (left[:, index] > 0) | ((left[:, index] == 0) & (signs > 0))
        correct_right = (right[:, index] < 0) | ((right[:, index] == 0) & (signs < 0))
        models[NAMES[index]] = {
            "D大于零": int((delta[:, index] > 0).sum()),
            "双正确": int((correct_left & correct_right).sum()),
        }
    return {
        "配对数": len(delta),
        "原问减反序的平均分差变化_logit": dict(zip(NAMES, delta.mean(axis=0).tolist())),
        "冻结首层分项消融_两问分别沿用各自底座门值": models,
        "显著度候选分差的平均绝对值_含门值_两问合计": float(
            (np.abs(left[:, 1]).mean() + np.abs(right[:, 1]).mean()) / 2
        ),
    }


def run(root):
    cohort = root / "cohort"
    receipt = json.loads((root / "predictions/prediction-seal.json").read_bytes())
    seal = json.loads(checked_bytes(cohort / "seal.json", receipt["cohort_seal_sha256"]))
    features = checked_bytes(cohort / "features.jsonl", seal["features_sha256"])
    checked_bytes(cohort / "gold.jsonl", seal["gold_sha256"])
    inference = root / "code/predict_sealed.py"
    checked_bytes(inference, receipt["inference_code_sha256"]["predict_sealed.py"])
    frozen = runpy.run_path(str(inference))
    frozen["configure_cpu"]()
    evaluator = frozen["load_implementation"](root / "inputs/b12-code")
    model_dir = root / "inputs/server-run"
    trainer = evaluator.frozen_trainer(model_dir)
    docs = evaluator.shortlist(frozen["parse_input"](trainer, features), "direction")
    templates = runpy.run_path(str(root / "code/generate_cohort.py"))["TEMPLATES"]
    pairs, eligible, _ = construct_pairs(docs, read_index(cohort / "gold.jsonl"), [
        template.split() for template in templates
    ], evaluator)
    if dict(eligible) != {"relation_order": 1000}:
        raise ValueError(f"可合法反序的图与预期不一致：{eligible}")
    # A 始终为原问的正确首跳。即使在反序问法上，候选分差也是 A 减 B。
    signs = torch.tensor([1.0 if ex.candidates[0] == gold else -1.0 for ex, _, gold, _ in pairs])
    left_rows = [p[0] for p in pairs]
    right_rows = [p[1] for p in pairs]
    device = torch.device("cpu")
    left_parts, right_parts = [], []
    result = {"文档数": len(pairs), "各种子": {}, "逐候选第一层重构最大绝对误差": 0.0}
    for seed in frozen["SEEDS"]:
        model, vocab, method = evaluator.load_model(trainer, model_dir, device, "serial", seed)
        args = (signs, model, vocab, method, trainer, device)
        left, error_left = decompose(left_rows, *args)
        right, error_right = decompose(right_rows, *args)
        result["各种子"][seed] = summarize(left, right, signs.numpy())
        result["逐候选第一层重构最大绝对误差"] = max(
            result["逐候选第一层重构最大绝对误差"], error_left, error_right
        )
        left_parts.append(left)
        right_parts.append(right)
    result["汇总"] = summarize(np.concatenate(left_parts), np.concatenate(right_parts),
                               np.tile(signs.numpy(), len(left_parts)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="已解压的第13项实验归档")
    print(json.dumps(run(parser.parse_args().archive.resolve()), ensure_ascii=False, indent=2))
