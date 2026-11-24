"""在已开封的第13项归档上复算逐层首跳命中。位置参数为该项归档的根目录。"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch


STAGES = ("直接首位", "方向底座 z0", "第一层 z1", "串联 z2")


def checked_bytes(path, digest):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError(f"SHA-256 校验失败：{path}")
    return payload


def record(counts, hits):
    counts["判断次数"] += 1
    for stage, hit in enumerate(hits):
        counts[stage, "命中"] += hit
        if stage:
            counts[stage, "纠正"] += hit and not hits[stage - 1]
            counts[stage, "改错"] += not hit and hits[stage - 1]


def print_counts(title, counts):
    total = counts["判断次数"]
    print(f"\n{title}（{total:,} 次判断）")
    print("阶段\t命中\t准确率\t相邻层纠正\t相邻层改错\t净变化")
    for stage, name in enumerate(STAGES):
        hits = counts[stage, "命中"]
        fixed = counts[stage, "纠正"]
        broken = counts[stage, "改错"]
        change = f"{fixed}\t{broken}\t{fixed - broken:+d}" if stage else "—\t—\t—"
        print(f"{name}\t{hits:,}\t{hits / total:.2%}\t{change}")


@torch.inference_mode()
def replay(root):
    # 归档的推理入口同时校验冻结的评估代码、底座代码和模型检查点。
    sys.dont_write_bytecode = True
    predictions = root / "predictions"
    seal = json.loads((predictions / "prediction-seal.json").read_bytes())
    cohort = root / "cohort"
    data_seal = json.loads(checked_bytes(cohort / "seal.json", seal["cohort_seal_sha256"]))
    checked_bytes(root / "code/predict_sealed.py", seal["inference_code_sha256"]["predict_sealed.py"])
    sys.path.insert(0, str(root / "code"))
    import predict_sealed as original

    original.configure_cpu()
    evaluator = original.load_implementation(root / "inputs/b12-code")
    model_root = root / "inputs/server-run"
    trainer = evaluator.frozen_trainer(model_root)
    features = checked_bytes(cohort / "features.jsonl", data_seal["features_sha256"])
    docs = evaluator.shortlist(original.parse_input(trainer, features), "direction")
    gold_data = checked_bytes(cohort / "gold.jsonl", data_seal["gold_sha256"])
    gold = {row["doc_id"]: row["gold"] for row in map(json.loads, gold_data.splitlines())}
    if gold.keys() != {doc.doc_id for doc in docs}:
        raise ValueError("标签与特征的文档 ID 不一致")

    def check_predictions(filename, selected):
        rows = trainer.prediction_rows(docs, selected)
        saved = checked_bytes(predictions / filename, seal["prediction_sha256"][filename])
        if trainer.jsonl_payload(rows) != saved:
            raise ValueError(f"逐题预测与归档不同：{filename}")

    check_predictions("direct.jsonl", [0] * len(docs))
    counts = defaultdict(Counter)
    device = torch.device("cpu")
    for seed in original.SEEDS:
        model, vocab, method = evaluator.load_model(trainer, model_root, device, "serial", seed)
        encoded = trainer.encode_examples(docs, vocab, method, with_alignment=True)
        selected = []
        for offset in range(0, len(encoded), trainer.BATCH_SIZE):
            batch = encoded[offset : offset + trainer.BATCH_SIZE]
            questions, nodes = trainer.collate(batch, device)
            alignments = trainer.collate_alignment(batch, questions.shape[1], device)
            z0 = model.layer1.base(questions, nodes)
            z2, layers = model(questions, nodes, *alignments, return_layer_scores=True)
            # 未触发门控的题沿用底座分数。
            z1 = z0.index_copy(0, alignments[0], layers.layer1_active_logits)
            layer_choices = torch.stack((z0, z1, z2)).argmax(dim=-1).T.tolist()
            for index, (base, first, final) in enumerate(layer_choices):
                selected.append(final)
                doc = docs[offset + index]
                choices = (0, base, first, final)
                hits = [doc.candidates[choice] == gold[doc.doc_id] for choice in choices]
                gate = "门内" if batch[index].alignment is not None else "门外"
                for group in ("全部", gate, f"种子 {seed}"):
                    record(counts[group], hits)
        check_predictions(f"serial-{seed}.jsonl", selected)

    print(f"已逐字节匹配直接首位和三个种子的最终预测，共 {len(docs):,} 篇文档。")
    for group in ("全部", "门内", "门外", *(f"种子 {seed}" for seed in original.SEEDS)):
        print_counts(group, counts[group])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="第13项已解压归档目录")
    replay(parser.parse_args().archive.resolve())
