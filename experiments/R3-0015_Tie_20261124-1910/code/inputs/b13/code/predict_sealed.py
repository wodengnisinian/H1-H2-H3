"""Seal one B13 direction-only, feature-only CPU prediction bundle."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
from pathlib import Path
import platform
import re
import sys

import torch


CODE_HASHES = {
    "evaluate.py": "68c64cf602d16f6477011fe4092677775a178a3a7c5b32b0da057443b38b0501",
    "tie_rank.py": "41c5b4b19b67be9ffe6a5735a15275ba152921466244771585d2bc324e4c64f9",
}
SEEDS = (1701, 2701, 3701)
STRATA = ("role_reversal", "relation_order", "competing_chain")
FIELDS = {"doc_id", "stratum", "question", "nodes"}
SCHEMA = "spiderweb-r2-v13-prediction-seal-v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return sha256(raw).hexdigest()


def sha_argument(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def load_implementation(code):
    for name, expected in CODE_HASHES.items():
        if digest((code / name).read_bytes()) != expected:
            raise ValueError(f"Frozen inference source differs: {name}")
    sys.path.insert(0, str(code))
    evaluator = importlib.import_module("evaluate")
    for mod_name in ("evaluate", "tie_rank"):
        actual = Path(sys.modules[mod_name].__file__).resolve()
        if actual != (code / f"{mod_name}.py").resolve():
            raise ValueError(f"Incorrect module for {mod_name}: {actual}")
    return evaluator


def parse_input(trainer, raw):
    features = []
    ids = set()
    for line in raw.splitlines():
        doc = json.loads(line)
        if not isinstance(doc, dict) or set(doc) != FIELDS:
            raise ValueError("Unexpected feature record fields")
        key = doc["doc_id"]
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("Expected unique nonempty feature document IDs")
        ids.add(key)
        if not isinstance(doc["nodes"], list) or len(doc["nodes"]) != trainer.NODE_COUNT:
            raise ValueError("Invalid node inventory")
        nodes = tuple(trainer.tokens(node) for node in doc["nodes"])
        if any(len(node) != 3 for node in nodes):
            raise ValueError("Expected triples")
        q = trainer.tokens(doc["question"])
        features.append(trainer.Example(key, doc["stratum"], q, nodes, trainer.lexical_candidates(q, nodes), None))
    if Counter(v.stratum for v in features) != {v: 1000 for v in STRATA}:
        raise ValueError("Expected 3,000 feature documents balanced across the three frozen strata")
    return features


def publish(trainer, path, payload):
    try:
        trainer.atomic_write(path, lambda output: output.write(payload), new_only=True)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Partial prediction differs from computed frozen result: {path.name}")
    return digest(payload)


def validate_seal(path, commitments, names):
    receipt = json.loads(path.read_bytes())
    if any(receipt.get(k) != v for k, v in commitments.items()):
        raise ValueError("Existing prediction seal binds different inputs or configuration")
    files = receipt["prediction_sha256"]
    if set(files) != names:
        raise ValueError("Existing prediction seal has the wrong output inventory")
    if any(digest((path.parent / n).read_bytes()) != h for n, h in files.items()):
        raise ValueError("A sealed prediction file differs")
    return receipt


def configure_cpu():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--features-sha256", required=True, type=sha_argument)
    parser.add_argument("--cohort-seal-sha256", required=True, type=sha_argument)
    parser.add_argument("--protocol-sha256", required=True, type=sha_argument)
    parser.add_argument("--freeze-sha256", required=True, type=sha_argument)
    parser.add_argument("--b11", required=True, type=Path, help="frozen B11 server-run directory")
    parser.add_argument("--b12-code", required=True, type=Path, help="frozen B12 evaluate.py and tie_rank.py directory")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = utc()
    configure_cpu()
    evaluator = load_implementation(args.b12_code.resolve())
    b11 = args.b11.resolve()
    trainer = evaluator.frozen_trainer(b11)
    raw = args.features.read_bytes()
    if digest(raw) != args.features_sha256:
        raise ValueError("Feature stream does not match opening commitment")
    ordered = evaluator.shortlist(parse_input(trainer, raw), "direction")
    outputs = {"direct.jsonl"} | {f"{kind}-{seed}.jsonl" for kind in ("serial", "pooled") for seed in SEEDS}
    commitments = {
        "schema": SCHEMA, "mode": "direction", "documents": len(ordered), "seeds": list(SEEDS),
        "features_sha256": args.features_sha256, "cohort_seal_sha256": args.cohort_seal_sha256,
        "protocol_sha256": args.protocol_sha256, "freeze_sha256": args.freeze_sha256,
        "inference_code_sha256": {**CODE_HASHES, "train.py": evaluator.TRAINER_SHA,
                                  "predict_sealed.py": digest(Path(__file__).read_bytes())},
        "checkpoints_sha256": {kind: {str(s): h for s, h in table.items()}
                               for kind, table in evaluator.CHECKPOINTS.items()},
    }
    out = args.out.resolve()
    out.mkdir(exist_ok=True, parents=True)
    seal = out / "prediction-seal.json"
    if seal.exists():
        validate_seal(seal, commitments, outputs)
        print(json.dumps({"event": "already_sealed", "sha256": digest(seal.read_bytes())}))
        return
    hashes = {}
    direct = trainer.prediction_rows(ordered, [0] * len(ordered))
    hashes["direct.jsonl"] = publish(trainer, out / "direct.jsonl", trainer.jsonl_payload(direct))
    device = torch.device("cpu")
    for kind in ("serial", "pooled"):
        for seed in SEEDS:
            model, vocab, method = evaluator.load_model(trainer, b11, device, kind, seed)
            encoded = trainer.encode_examples(ordered, vocab, method, with_alignment=kind == "serial")
            indices = trainer.predict_indices(model, encoded, device)
            rows = trainer.prediction_rows(ordered, indices)
            name = f"{kind}-{seed}.jsonl"
            hashes[name] = publish(trainer, out / name, trainer.jsonl_payload(rows))
            print(json.dumps({"event": "prediction", "file": name, "sha256": hashes[name]}), flush=True)
    receipt = {
        **commitments, "started_utc": started, "sealed_utc": utc(), "prediction_sha256": hashes,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": str(torch.__version__), "device": "cpu", "threads": torch.get_num_threads()},
    }
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        trainer.atomic_write(seal, lambda output: output.write(payload), new_only=True)
    except FileExistsError:
        pass
    validate_seal(seal, commitments, outputs)
    print(json.dumps({"event": "sealed", "sha256": digest(seal.read_bytes())}), flush=True)


if __name__ == "__main__":
    main()
