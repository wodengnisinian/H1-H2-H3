"""Seal the independent split and audit fixed-budget first-hop predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import tempfile


SEEDS = (1701, 2701, 3701)
STRATA = ("role_reversal", "relation_order", "competing_chain")
SPLIT_COUNTS = {"train": 12_000, "dev": 1_800, "test": 3_000}
NODE_COUNT = 12
BUDGET = 2
BOOTSTRAP_SEED = 619_482
BOOTSTRAP_REPLICATES = 10_000
LOWER_RANK = 500
MIN_EFFECT_PERCENT = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return sha256(payload).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_rows(path):
    payload = path.read_bytes()
    rows = {}
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(line)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        require(isinstance(row, dict), f"{path}:{line_number}: expected an object")
        doc_id = row.get("doc_id")
        require(isinstance(doc_id, str) and bool(doc_id), f"{path}:{line_number}: missing document ID")
        require(doc_id not in rows, f"{path}:{line_number}: duplicate document ID")
        require(row.get("stratum") in STRATA, f"{path}:{line_number}: invalid stratum")
        rows[doc_id] = row
    return rows, digest(payload)


def validate_balance(rows, split):
    counts = Counter(row["stratum"] for row in rows.values())
    expected = SPLIT_COUNTS[split] // len(STRATA)
    require(counts == {stratum: expected for stratum in STRATA}, f"{split}: unexpected stratum counts")
    return dict(counts)


def validate_node_index(value, context):
    require(type(value) is int and 0 <= value < NODE_COUNT, f"{context}: invalid node index")


def validate_tokens(value, context, length=None):
    require(isinstance(value, list) and bool(value), f"{context}: expected a nonempty token list")
    if length is not None:
        require(len(value) == length, f"{context}: unexpected token count")
    require(all(isinstance(token, str) and bool(token) for token in value), f"{context}: invalid token")


def expected_candidates(row, context):
    question = row.get("question")
    nodes = row.get("nodes")
    validate_tokens(question, context)
    require(isinstance(nodes, list) and len(nodes) == NODE_COUNT, f"{context}: expected {NODE_COUNT} nodes")
    for node in nodes:
        validate_tokens(node, context, length=3)
    require(len({tuple(node) for node in nodes}) == NODE_COUNT, f"{context}: duplicate facts")
    query = set(question)
    return sorted(range(NODE_COUNT), key=lambda index: (-len(query.intersection(nodes[index])), index))[:BUDGET]


def validate_pair(value, context):
    require(isinstance(value, list) and len(value) == BUDGET, f"{context}: invalid candidate budget")
    for item in value:
        validate_node_index(item, context)
    require(len(set(value)) == BUDGET, f"{context}: repeated candidate")


def check_new_paths(paths):
    resolved = {path.resolve() for path in paths}
    require(len(resolved) == len(paths), "output paths must be distinct")
    for path in paths:
        require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
        require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")


def write_exclusive(path, payload):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(path, report):
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    write_exclusive(path, payload)


def seal(args):
    check_new_paths((args.labels, args.features, args.out))
    document_ids = set()
    graph_hashes = set()
    other_split_questions = set()
    report = {"schema_version": 1, "status": "sealed", "inputs": {}}
    for split in SPLIT_COUNTS:
        examples, checksum = read_rows(getattr(args, split))
        counts = validate_balance(examples, split)
        require(not examples.keys() & document_ids, f"{split}: document overlap with previous split")
        document_ids.update(examples)
        split_graphs = set()
        split_questions = set()
        for row_number, row in enumerate(examples.values(), 1):
            context = f"{split}:{row_number}"
            validate_node_index(row.get("gold"), context)
            shortlist = expected_candidates(row, context)
            graph = digest(json_bytes(sorted(row["nodes"])))
            require(graph not in split_graphs and graph not in graph_hashes, f"{context}: repeated graph")
            split_graphs.add(graph)
            split_questions.add(digest(json_bytes(row["question"])))
            if split == "test":
                row["expected_candidates"] = shortlist
        require(not split_questions & other_split_questions, f"{split}: exact question overlap with previous split")
        graph_hashes.update(split_graphs)
        other_split_questions.update(split_questions)
        report["inputs"][split] = {
            "sha256": checksum,
            "documents": len(examples),
            "strata": counts,
            "distinct_graphs": len(split_graphs),
            "distinct_questions": len(split_questions),
        }

    labels = bytearray()
    features = bytearray()
    for row in examples.values():
        common = {"doc_id": row["doc_id"], "stratum": row["stratum"]}
        labels.extend(json_bytes(dict(common, gold=row["gold"], expected_candidates=row["expected_candidates"])) + b"\n")
        features.extend(json_bytes(dict(common, nodes=row["nodes"], question=row["question"])) + b"\n")
    write_exclusive(args.labels, labels)
    write_exclusive(args.features, features)
    report["outputs"] = {"labels_sha256": digest(labels), "features_sha256": digest(features)}
    report["overlap_audit"] = {
        "cross_split_document_ids": 0,
        "cross_split_canonical_fact_sets": 0,
        "cross_split_exact_question_arrays": 0,
    }
    write_report(args.out, report)


def read_gold(path):
    rows, checksum = read_rows(path)
    if len(rows) == SPLIT_COUNTS["dev"]:
        split = "dev"
    elif len(rows) == SPLIT_COUNTS["test"]:
        split = "test"
    else:
        raise ValueError("gold: expected the exact development or independent document count")
    validate_balance(rows, split)
    for row_number, row in enumerate(rows.values(), 1):
        context = f"gold:{row_number}"
        validate_node_index(row.get("gold"), context)
        if split == "dev":
            row["expected_candidates"] = expected_candidates(row, context)
        else:
            require("nodes" not in row and "question" not in row, f"{context}: expected sealed labels")
            validate_pair(row.get("expected_candidates"), context)
    return split, rows, checksum


def read_arm(path, gold, label, lexical=False):
    rows, checksum = read_rows(path)
    require(rows.keys() == gold.keys(), f"{label}: document set differs from gold inventory")
    for row_number, (doc_id, row) in enumerate(rows.items(), 1):
        context = f"{label}:{row_number}"
        truth = gold[doc_id]
        require(row["stratum"] == truth["stratum"], f"{context}: stratum mismatch")
        validate_pair(row.get("candidates"), context)
        validate_node_index(row.get("prediction"), context)
        require(row["candidates"] == truth["expected_candidates"], f"{context}: ordered lexical candidates mismatch")
        require(row["prediction"] in row["candidates"], f"{context}: prediction outside the shared shortlist")
        if lexical:
            require(row["prediction"] == row["candidates"][0], f"{context}: lexical prediction is not rank one")
    return rows, checksum


def paired_bootstrap_lower(histograms):
    """Sample both contrasts on the same stratified document bootstrap draws."""
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[], []]
    groups = []
    for stratum in STRATA:
        histogram = histograms[stratum]
        categories = sorted(histogram.items())
        groups.append((sum(histogram.values()), categories))
    for _ in range(BOOTSTRAP_REPLICATES):
        pooled_total = lexical_total = 0
        for size, categories in groups:
            remaining_draws = remaining_weight = size
            # Conditional binomial draws create exactly one multinomial
            # resample of the empirical document categories for this stratum.
            for (pooled_delta, lexical_delta), weight in categories[:-1]:
                count = rng.binomialvariate(remaining_draws, weight / remaining_weight)
                pooled_total += count * pooled_delta
                lexical_total += count * lexical_delta
                remaining_draws -= count
                remaining_weight -= weight
            (pooled_delta, lexical_delta), _ = categories[-1]
            pooled_total += remaining_draws * pooled_delta
            lexical_total += remaining_draws * lexical_delta
        draws[0].append(pooled_total)
        draws[1].append(lexical_total)
    return tuple(sorted(values)[LOWER_RANK - 1] for values in draws)


def summarize(records):
    documents = len(records)
    coverage = sum(record[0] for record in records)
    lexical = sum(record[1] for record in records)
    pooled = [sum(record[2][index] for record in records) for index in range(len(SEEDS))]
    candidate = [sum(record[3][index] for record in records) for index in range(len(SEEDS))]
    model_denominator = len(SEEDS) * documents
    return {
        "documents": documents,
        "candidate_coverage": coverage / documents,
        "accuracy": {
            "lexical": lexical / documents,
            "pooled": sum(pooled) / model_denominator,
            "candidate": sum(candidate) / model_denominator,
        },
        "accuracy_by_seed": {
            str(seed): {"pooled": p / documents, "candidate": c / documents}
            for seed, p, c in zip(SEEDS, pooled, candidate, strict=True)
        },
    }


def score(args):
    check_new_paths((args.out,))
    split, gold, gold_hash = read_gold(args.gold)
    lexical, lexical_hash = read_arm(args.lexical, gold, "lexical", lexical=True)
    pooled = [read_arm(path, gold, f"pooled:{seed}") for seed, path in zip(SEEDS, args.pooled, strict=True)]
    candidate = [read_arm(path, gold, f"candidate:{seed}") for seed, path in zip(SEEDS, args.candidate, strict=True)]
    require(len({path.resolve() for path in (args.lexical, *args.pooled, *args.candidate)}) == 7,
            "prediction arms must be separate files")

    records = {stratum: [] for stratum in STRATA}
    histograms = {stratum: Counter() for stratum in STRATA}
    for doc_id, truth in gold.items():
        label = truth["gold"]
        lexical_correct = lexical[doc_id]["prediction"] == label
        pooled_correct = tuple(rows[doc_id]["prediction"] == label for rows, _ in pooled)
        candidate_correct = tuple(rows[doc_id]["prediction"] == label for rows, _ in candidate)
        coverage = label in truth["expected_candidates"]
        records[truth["stratum"]].append((coverage, lexical_correct, pooled_correct, candidate_correct))
        movement = (sum(candidate_correct) - sum(pooled_correct), sum(candidate_correct) - len(SEEDS) * lexical_correct)
        histograms[truth["stratum"]][movement] += 1

    denominator = len(gold) * len(SEEDS)  # Equal stratum sizes are enforced for these splits.
    lower_bounds = paired_bootstrap_lower(histograms)
    contrasts = {}
    for index, baseline in enumerate(("pooled", "lexical")):
        delta = sum(movement[index] * count for histogram in histograms.values() for movement, count in histogram.items())
        lower = lower_bounds[index]
        contrasts[f"candidate_minus_{baseline}"] = {
            "effect": delta / denominator,
            "lower_95": lower / denominator,
            "point_passed": delta * 100 >= MIN_EFFECT_PERCENT * denominator,
            "interval_passed": lower > 0,
        }
    write_report(args.out, {
        "schema_version": 1,
        "split": split,
        "primary_metric": "Equally weighted stratum macro first-hop Top-1, averaged within document over the fixed model seeds",
        "budget": BUDGET,
        "seeds": list(SEEDS),
        "bootstrap": {
            "method": "paired stratified document percentile, simultaneous draws across contrasts",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "lower_rank": LOWER_RANK,
            "confidence_one_sided": 0.95,
            "python_version": sys.version.split()[0],
        },
        "minimum_point_gain": MIN_EFFECT_PERCENT / 100,
        "passed": all(result["point_passed"] and result["interval_passed"] for result in contrasts.values()),
        "contrasts": contrasts,
        **summarize([record for stratum in STRATA for record in records[stratum]]),
        "by_stratum": {stratum: summarize(records[stratum]) for stratum in STRATA},
        "input_sha256": {
            "gold": gold_hash,
            "lexical": lexical_hash,
            "pooled": {str(seed): item[1] for seed, item in zip(SEEDS, pooled, strict=True)},
            "candidate": {str(seed): item[1] for seed, item in zip(SEEDS, candidate, strict=True)},
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    for name in ("train", "dev", "test", "labels", "features", "out"):
        seal_parser.add_argument(f"--{name}", type=Path, required=True)
    score_parser = commands.add_parser("score")
    for name in ("gold", "lexical", "out"):
        score_parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("pooled", "candidate"):
        score_parser.add_argument(f"--{name}", type=Path, nargs=len(SEEDS), required=True)
    args = parser.parse_args()
    try:
        if args.command == "seal":
            seal(args)
        else:
            score(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"verification failed: {exc}\n")


if __name__ == "__main__":
    main()

