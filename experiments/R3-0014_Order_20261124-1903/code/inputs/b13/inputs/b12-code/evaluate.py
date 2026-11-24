"""Replay the frozen B11 scorers on each B12 feature-only shortlist."""
import argparse
from dataclasses import replace
from hashlib import file_digest
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from tie_rank import rank_candidates


MODES = ("original", "direction", "first_relation", "combined")
VIEWS = ("original", "rewritten")
TRAINER_SHA = "21d9dff3d252c8499319b05ad5aed0c05a46d9d2f37c7e811ca4890d648a8661"
CHECKPOINTS = {
    "serial": {
        1701: "703b3edfbf72b2213431c054e38dcad3300dd880f8443707b7119b9859a7728c",
        2701: "521911f7ec4b7679a4ba733754d3f87b890f0723596f075fd79b1a9b21cfa4bf",
        3701: "d5585978da8310b9e5452f0a3cd3a6a6be07559a91943816076d52d42895939b",
    },
    "pooled": {
        1701: "e9115ffc1f49266ff49efa12f3a7470fee5047bea32621f0a3d481a066e52bc7",
        2701: "146b7c702f326cc9fb050fddc076850eacd7295abf12a6a0a7bbaddaee91bae9",
        3701: "462f8ef29c84f13036353588308d17ef2858c060839bdf198e64dcb46ba69f2b",
    },
}


def file_sha(path):
    with path.open("rb") as fp:
        return file_digest(fp, "sha256").hexdigest()


def frozen_trainer(b11):
    path = b11 / "code" / "train.py"
    if file_sha(path) != TRAINER_SHA:
        raise ValueError(f"Unexpected B11 inference code: {path}")
    spec = importlib.util.spec_from_file_location("b11_frozen_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_features(trainer, path):
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if set(json.loads(line)) != {"doc_id", "stratum", "question", "nodes"}:
                raise ValueError(f"Unexpected feature fields: {path}:{line_number}")
    examples = trainer.load_examples(path, require_gold=False)
    if len(examples) != 1800 or any(example.gold is not None for example in examples):
        raise ValueError(f"Expected 1800 unlabeled development examples: {path}")
    return examples


def shortlist(examples, mode):
    ordered = []
    for example in examples:
        candidates = tuple(rank_candidates(example.question, example.nodes, mode))
        if len(candidates) != 2 or len(set(candidates)) != 2:
            raise ValueError(f"Expected two distinct candidates for {example.doc_id}")
        if any(type(i) is not int or not 0 <= i < len(example.nodes) for i in candidates):
            raise ValueError(f"Invalid node index for {example.doc_id}")
        if mode == "original" and candidates != example.candidates:
            raise ValueError(f"B11 lexical replay failed for {example.doc_id}")
        ordered.append(replace(example, candidates=candidates))
    return ordered


def checkpoint_dir(b11, kind, seed):
    return b11 / ("development" if kind == "serial" else "controls") / kind / str(seed)


def load_model(trainer, b11, device, kind, seed):
    path = checkpoint_dir(b11, kind, seed) / "best.pt"
    if file_sha(path) != CHECKPOINTS[kind][seed]:
        raise ValueError(f"Unexpected B11 checkpoint: {path}")
    state = trainer.read_checkpoint(path)
    config = state["config"]
    method = config["method"]
    if config["seed"] != seed:
        raise ValueError(f"Wrong checkpoint seed: {path}")
    vocab = state["vocabulary"]
    model = trainer.Scorer(len(vocab) + 2, method)
    if kind == "serial":
        if (method, config["reranker"], config["stack_mode"], config["source_sha256"]) != (
            trainer.BASE_METHOD, trainer.TRAIN_RERANKER, "serial", TRAINER_SHA,
        ):
            raise ValueError(f"Unexpected serial model configuration: {path}")
        model = trainer.StackedReranker(trainer.GatedReranker(model, config["reranker"]), "serial")
    elif method != "pooled" or config["source_sha256"] != trainer.LEGACY_POOLED_TRAIN_SHA256:
        raise ValueError(f"Unexpected pooled model configuration: {path}")
    model.to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, vocab, method


def save_prediction(trainer, root, b11, mode, view, kind, seed, rows):
    payload = trainer.jsonl_payload(rows)
    if mode == "original":
        reference = (checkpoint_dir(b11, kind, seed) / "dev_predictions.jsonl" if view == "original"
                     else b11 / "development" / "paired" / f"{kind}-{seed}.jsonl")
        original = reference.read_bytes()
        if payload != original:
            archived_rows = [json.loads(line) for line in original.splitlines()]
            different = sum(a != b for a, b in zip(rows, archived_rows, strict=True))
            raise ValueError(f"Unchanged-shortlist replay differs from {reference}: {different} rows")
    directory = root / "predictions" / mode
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "pooled-" if kind == "pooled" else ""
    path = directory / f"{prefix}{view}-{seed}.jsonl"
    try:
        trainer.atomic_write(path, lambda stream: stream.write(payload), new_only=True)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Existing prediction file differs from this run: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="B12 experiment root")
    parser.add_argument("--b11", type=Path, required=True, help="B11 server-run root")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root, b11 = args.root.resolve(), args.b11.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    trainer = frozen_trainer(b11)
    before = monotonic()
    print(json.dumps(dict(event="started", platform=platform.platform(), python=platform.python_version(),
                          torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                          threads=torch.get_num_threads(), trainer_sha256=TRAINER_SHA)), flush=True)
    features = {v: load_features(trainer, root / "features" / f"{v}.jsonl") for v in VIEWS}
    models = {
        (kind, seed): load_model(trainer, b11, device, kind, seed)
        for kind, seeds in CHECKPOINTS.items() for seed in seeds
    }
    # All twelve frozen replay files must agree before evaluating a changed shortlist.
    for mode in MODES:
        for view in VIEWS:
            examples = shortlist(features[view], mode)
            for (kind, seed), (model, vocab, method) in models.items():
                encoded = trainer.encode_examples(examples, vocab, method, with_alignment=kind == "serial")
                indices = trainer.predict_indices(model, encoded, device)
                rows = trainer.prediction_rows(examples, indices)
                path = save_prediction(trainer, root, b11, mode, view, kind, seed, rows)
                gate_rows = sum(x.alignment is not None for x in encoded) if kind == "serial" else None
                print(json.dumps(dict(event="prediction", mode=mode, view=view, model=kind, seed=seed,
                                      rows=len(rows), gate_rows=gate_rows, sha256=file_sha(path),
                                      path=path.relative_to(root).as_posix(), elapsed_s=round(monotonic() - before, 3))),
                      flush=True)
    print(json.dumps(dict(event="finished", streams=len(MODES) * len(VIEWS) * len(models),
                          elapsed_s=round(monotonic() - before, 3))), flush=True)


if __name__ == "__main__":
    main()
