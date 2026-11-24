"""Run the H1/H2 implementation checks against public B13 fixtures on CPU."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

sys.dont_write_bytecode = True
import torch

from model import correct_margin_bounds, extract, fit, load_bundle, score


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 2701, 3701)


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_file(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Archive hash mismatch: {path}")
    return raw


def publish(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", prefix=path.name, suffix=".tmp", dir=path.parent,
                           encoding="utf-8", delete=False) as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(path)


def word_weights(features, w):
    return (features.hidden @ w).masked_fill(~features.mask, -torch.inf).softmax(-1)


def max_error(observed, expected, *, atol=4e-6):
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=atol)
    return (observed - expected).abs().max().item()


def check_seed(archive, seed, count, gold, receipt):
    bundle = load_bundle(archive, seed)
    trainer = bundle.trainer
    docs = trainer.load_examples(archive / "cohort/features.jsonl", require_gold=False)[:count]
    docs = bundle.evaluator.shortlist(docs, "direction")
    features = extract(bundle, docs, batch_size=11)
    y = torch.tensor([row.candidates.index(gold[row.doc_id]) if gold[row.doc_id] in row.candidates
                      else -1 for row in docs])
    name = f"pooled-{seed}.jsonl"
    raw = checked_file(archive / "predictions" / name, receipt["prediction_sha256"][name])
    official = {row["doc_id"]: row["prediction"] for row in map(json.loads, raw.splitlines())}
    pool_ok = sum(row.candidates[p] == official[row.doc_id]
                  for row, p in zip(docs, features.pooled.argmax(1).tolist(), strict=True))
    if pool_ok != len(docs):
        raise AssertionError("Pooled predictions differ from the B13 seal")

    # Compare with an independent, single-batch call to the frozen first layer.
    enc = trainer.encode_examples(docs, bundle.vocab, bundle.method, with_alignment=True)
    questions, nodes = trainer.collate(enc, bundle.device)
    align = trainer.collate_alignment(enc, questions.shape[1], bundle.device)
    adapter = bundle.serial.layer1
    original_reduce = adapter.logmeanexp
    reductions = []

    def capture(values, mask):
        result = original_reduce(values, mask)
        reductions.append(result)
        return result

    with torch.no_grad(), patch.object(adapter, "logmeanexp", side_effect=capture):
        z0, old, _ = adapter.layer_outputs(questions, nodes, *align)
    ind = align[0]
    gated = z0[ind]
    archived_amp = (adapter.alpha + adapter.beta * (gated[:, 0] - gated[:, 1]).abs()).sigmoid()
    uniform_expected = z0.clone()
    uniform_expected[ind] += archived_amp[:, None] * reductions[1]
    uniform = score(features, torch.zeros(64))
    errors = {
        "direction_vs_archive": max_error(features.base, z0),
        "first_layer_vs_archive": max_error(features.old, old),
        "zero_w_vs_archive_relation_only": max_error(uniform, uniform_expected),
    }

    # Short fits exercise gradients, eligibility and deterministic data order.
    # The vectors are discarded after this diagnostic.
    if not features.active.any() or features.active.all():
        raise ValueError("The selected fixture must contain both gate states")
    options = dict(epochs=2, lr=0.01, batch_size=8, seed=seed)
    missing = y.clone()
    missing[ind[0]] = -1
    discarded, missing_history = fit(features, missing, **options)
    eligible = int((features.active & (missing >= 0)).sum())
    if missing_history[-1]["examples"] != eligible:
        raise AssertionError("Missing target was included in training")
    w, history = fit(features, y, **options)
    replay, _ = fit(features, y, **options)
    if not torch.equal(w, replay) or w.norm().item() == 0:
        raise AssertionError("Tiny fit not deterministic or the vector did not update")
    if any(p.requires_grad for m in (bundle.serial, bundle.pooled) for p in m.parameters()):
        raise AssertionError("Archived weights are trainable")
    decisions = score(features, w)
    off = (~features.active).nonzero().flatten()
    if not torch.equal(decisions[off], features.base[off]):
        raise AssertionError("Closed gate changed logits")
    only_off = extract(bundle, [docs[i] for i in off.tolist()], batch_size=8)
    if not torch.equal(score(only_off, w), only_off.base):
        raise AssertionError("Closed-gate-only batch changed logits")

    covered = y >= 0
    idx = covered.nonzero().flatten()
    margins = decisions[idx, y[idx]] - decisions[idx, 1 - y[idx]]
    bounds = correct_margin_bounds(features, y)[idx]
    violation = torch.maximum(bounds[:, 0] - margins, margins - bounds[:, 1]).clamp_min(0.)
    if violation.max().item() > 1e-5:
        raise AssertionError("Total logit margin outside Eq. 10")
    if not torch.isnan(correct_margin_bounds(features, missing)[ind[0]]).all():
        raise AssertionError("Bounds reported for an absent gold candidate")

    swapped = [replace(row, candidates=tuple(reversed(row.candidates))) for row in docs]
    flipped = extract(bundle, swapped, batch_size=11)
    if not torch.equal(features.active, flipped.active):
        raise AssertionError("Candidate permutation changed gate status")
    p = word_weights(features, w)
    q = word_weights(flipped, w)
    errors["swap_word_weights_gate_open"] = max_error(p[features.active], q[features.active], atol=0.)
    errors["swap_word_weights_gate_closed"] = max_error(p[off], q[off], atol=0.)
    with torch.no_grad():
        reference_h, reference_mask = adapter.base.encode(questions)
        reference_p = (reference_h @ w).masked_fill(~reference_mask, -torch.inf).softmax(-1)
    errors["word_weights_gate_closed_vs_archive_encoder"] = max_error(p[off], reference_p[off])
    errors["word_weight_row_sum"] = max_error(p.sum(-1), torch.ones(p.shape[0]))
    if torch.count_nonzero(p[~features.mask]):
        raise AssertionError("Padding received a nonzero H1 word weight")
    errors["swap_final_logits"] = max_error(score(flipped, w), decisions.flip(1))
    errors["swap_pooled_logits"] = max_error(flipped.pooled, features.pooled.flip(1))
    errors["margin_bound_violation"] = violation.max().item()
    return {
        "seed": seed, "finished_utc": now(), "graphs": len(docs),
        "first_id": docs[0].doc_id, "last_id": docs[-1].doc_id,
        "gate_open": int(features.active.sum()), "gate_closed": len(off),
        "covered": len(idx), "pooled_sealed_agreement": pool_ok,
        "repeat_training_w_bitwise_equal": True, "frozen_parameters_verified": True,
        "closed_gate_scores_bitwise_equal": True,
        "all_closed_batch_scores_bitwise_equal": True,
        "eligible_after_removing_one_gate_open_target": eligible,
        "uncovered_example_bounds_are_nan": True,
        "word_weights_padding_zero": True, "max_absolute_error": errors,
        "learned_parameters": w.numel(), "disposable_fit_history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        help="解压的第 13 项归档，默认查找同级 R2实验")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/模型自检.json")
    parser.add_argument("--count", type=int, default=48)
    args = parser.parse_args()
    if not (2 <= args.count <= 3000):
        parser.error("--count must be in [2, 3000]")
    if args.archive is None:
        archives = list((ROOT.parent / "R2实验").glob("13-*"))
        if len(archives) != 1:
            parser.error("provide --archive")
        args.archive = archives[0]
    archive = args.archive.resolve()
    receipt = json.loads((archive / "predictions/prediction-seal.json").read_bytes())
    seal = json.loads(checked_file(archive / "cohort/seal.json", receipt["cohort_seal_sha256"]))
    checked_file(archive / "cohort/features.jsonl", seal["features_sha256"])
    raw = checked_file(archive / "cohort/gold.jsonl", seal["gold_sha256"])
    gold = {r["doc_id"]: r["gold"] for r in map(json.loads, raw.splitlines())}
    record = {
        "purpose": "公开第13项历史图上的实现核对；短训练权重只在内存中使用",
        "started_utc": now(), "status": "running", "per_seed": [], "archive": str(archive),
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(__file__).with_name("model.py"))},
        "features_sha256": seal["features_sha256"], "gold_sha256": seal["gold_sha256"],
        "python": sys.version, "torch": str(torch.__version__),
    }
    publish(args.out, record)
    try:
        for seed in SEEDS:
            measurement = check_seed(archive, seed, args.count, gold, receipt)
            record["per_seed"].append(measurement)
            publish(args.out, record)
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_utc"] = now()
        publish(args.out, record)
    print(f"Verified {args.count} graphs across {len(SEEDS)} seeds. {args.out}")


if __name__ == "__main__":
    main()
