"""Open the prospectively sealed B13 labels once and score the fixed direction+serial model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


SEEDS = (1701, 2701, 3701)
STRATA = ("role_reversal", "relation_order", "competing_chain")
STREAMS = ("direct.jsonl", *(f"{arm}-{seed}.jsonl" for arm in ("pooled", "serial") for seed in SEEDS))
VERIFY_SHA = "8a77ddb5b8173b21e0d54943077516563c2c7ee42167bd3a4b978f2750f598a7"
RANK_SHA = "41c5b4b19b67be9ffe6a5735a15275ba152921466244771585d2bc324e4c64f9"


def now():
    return datetime.now(timezone.utc).isoformat()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def load_record(path):
    data = path.read_bytes()
    return json.loads(data), digest(data)


def committed(path, expected):
    data = path.read_bytes()
    check(digest(data) == expected, f"Sealed bytes differ for {path.name}")
    return data


def utc_stamp(value):
    stamp = datetime.fromisoformat(value)
    check(stamp.utcoffset() is not None, "Timestamps must include UTC offsets")
    return stamp.astimezone(timezone.utc)


def load_module(path, expected, name):
    # Execute the verified bytes, avoiding cached bytecode or a second source read.
    raw = committed(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def inventory(raw, fields):
    rows = {}
    for row in map(json.loads, raw.splitlines()):
        check(set(row) == fields, "Unexpected fields in a sealed input")
        key = row["doc_id"]
        check(type(key) is str and key and key not in rows, "Invalid or duplicate document ID")
        check(row["stratum"] in STRATA, "Unknown stratum")
        rows[key] = row
    check(len(rows) == 3000 and Counter(row["stratum"] for row in rows.values()) ==
          {group: 1000 for group in STRATA}, "Expected 3,000 documents balanced over three strata")
    return rows


def public_inputs(root):
    protocol_hash = digest((root / "实验协议.json").read_bytes())
    manifest, freeze_hash = load_record(root / "冻结清单.json")
    for name, expected in (("实验协议.json", protocol_hash), ("code/score_once.py", digest(Path(__file__).read_bytes())),
                           ("inputs/source/verify.py", VERIFY_SHA), ("inputs/b12-code/tie_rank.py", RANK_SHA)):
        check(manifest["sha256"][name] == expected, f"Freeze mismatch for {name}")
    cohort, cohort_hash = load_record(root / "cohort/seal.json")
    sealed, prediction_hash = load_record(root / "predictions/prediction-seal.json")
    for record in (cohort, sealed):
        check(record["protocol_sha256"] == protocol_hash and record["freeze_sha256"] == freeze_hash,
              "Protocol or freeze commitment differs between phases")
    check(utc_stamp(cohort["sealed_utc"]) <= utc_stamp(sealed["started_utc"]) <=
          utc_stamp(sealed["sealed_utc"]) <= datetime.now(timezone.utc), "Phases are out of order")
    check(sealed["cohort_seal_sha256"] == cohort_hash and sealed["features_sha256"] == cohort["features_sha256"],
          "Predictions do not commit to this cohort")
    check(sealed["mode"] == "direction" and sealed["seeds"] == list(SEEDS) and sealed["documents"] == 3000,
          "Unexpected model configuration or cohort size in prediction seal")
    check(set(sealed["prediction_sha256"]) == set(STREAMS), "Expected exactly seven prespecified streams")
    verifier = load_module(root / "inputs/source/verify.py", VERIFY_SHA, "v13_frozen_verifier")
    ranker = load_module(root / "inputs/b12-code/tie_rank.py", RANK_SHA, "v13_frozen_tie_rank")
    features = inventory(committed(root / "cohort/features.jsonl", cohort["features_sha256"]),
                         {"doc_id", "stratum", "question", "nodes"})
    expected_pairs = {}
    for key, row in features.items():
        # The old helper checks graph/token shape. Its old ordering is discarded.
        verifier.expected_candidates(row, key)
        top = ranker.rank_candidates(row["question"], row["nodes"], "direction")
        verifier.validate_pair(top, key)
        expected_pairs[key] = top
    streams = {}
    for name in STREAMS:
        data = committed(root / "predictions" / name, sealed["prediction_sha256"][name])
        rows = inventory(data, {"doc_id", "stratum", "candidates", "prediction"})
        check(rows.keys() == features.keys(), f"{name}: document ID mismatch")
        for key, row in rows.items():
            check(row["stratum"] == features[key]["stratum"], f"{name}: stratum mismatch")
            verifier.validate_pair(row["candidates"], key)
            verifier.validate_node_index(row["prediction"], key)
            top = expected_pairs[key]
            check(row["candidates"] == top and row["prediction"] in top, f"{name}: invalid direction shortlist")
            if name == "direct.jsonl":
                check(row["prediction"] == top[0], "Direct control must choose the first ranked fact")
        streams[name] = data
    commitments = {
        "study_scope": "prospective separate B13 study; one fixed direction+serial candidate",
        "attempt": 1, "maximum_attempts": 1, "protocol_sha256": protocol_hash, "freeze_sha256": freeze_hash,
        "cohort_seal_sha256": cohort_hash, "features_sha256": cohort["features_sha256"],
        "gold_sha256": cohort["gold_sha256"], "prediction_seal_sha256": prediction_hash,
        "prediction_sha256": sealed["prediction_sha256"], "verifier_sha256": VERIFY_SHA,
        "ranker_sha256": RANK_SHA,
    }
    return verifier, features, expected_pairs, streams, commitments


def score_once(root):
    assessment = root / "assessment"
    assessment.mkdir(exist_ok=True)
    opening = assessment / "opening.json"
    score_path = assessment / "score.json"
    receipt_path = assessment / "assessment-receipt.json"
    check(not any(f.exists() or f.is_symlink() for f in (opening, score_path, receipt_path)),
          "The assessment was already opened or its output exists")
    verifier, features, top, streams, commitments = public_inputs(root)
    opened = now()
    # Atomic exclusive publication. No label bytes are read before this latch.
    verifier.write_report(opening, {**commitments, "opened_utc": opened})
    receipt = {**commitments, "opened_utc": opened, "opening_sha256": digest(opening.read_bytes()),
               "label_mapping": {"candidate": "direction+frozen serial", "lexical": "direction rank one",
                                 "pooled": "pooled on the same direction shortlist"}}
    try:
        labels = inventory(committed(root / "cohort/gold.jsonl", commitments["gold_sha256"]),
                           {"doc_id", "stratum", "gold"})
        check(labels.keys() == features.keys(), "Label IDs differ from the prevalidated features")
        truth = []
        for key, row in labels.items():
            check(row["stratum"] == features[key]["stratum"], "Label stratum mismatch")
            verifier.validate_node_index(row["gold"], key)
            truth.append(verifier.json_bytes({**row, "expected_candidates": top[key]}) + b"\n")
        joined = b"".join(truth)
        # Pass a private immutable snapshot to the completely unchanged official scorer.
        # Its 3,000-document branch consumes expected_candidates from the sealed-input adapter.
        with TemporaryDirectory(prefix=".scoring-", dir=assessment) as folder:
            scratch = Path(folder).resolve()
            check(scratch.parent == assessment and scratch.name.startswith(".scoring-"), "Invalid temporary directory")
            gold = scratch / "labels.jsonl"
            gold.write_bytes(joined)
            for name, data in streams.items():
                (scratch / name).write_bytes(data)
            output = scratch / "score.json"
            verifier.score(SimpleNamespace(gold=gold, lexical=scratch / "direct.jsonl",
                           pooled=[scratch / f"pooled-{seed}.jsonl" for seed in SEEDS],
                           candidate=[scratch / f"serial-{seed}.jsonl" for seed in SEEDS], out=output))
            score, score_hash = load_record(output)
            expected = {"gold": digest(joined), "lexical": commitments["prediction_sha256"]["direct.jsonl"],
                        "pooled": {str(s): commitments["prediction_sha256"][f"pooled-{s}.jsonl"] for s in SEEDS},
                        "candidate": {str(s): commitments["prediction_sha256"][f"serial-{s}.jsonl"] for s in SEEDS}}
            check(score["split"] == "test" and score["input_sha256"] == expected, "Official score input mismatch")
            verifier.write_exclusive(score_path, output.read_bytes())
        receipt.update(status="passed" if score["passed"] else "failed", returncode=0, score_sha256=score_hash,
                       canonical_scoring_labels_sha256=digest(joined))
    except BaseException as exc:
        receipt.update(status="error_after_opening", returncode=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        verifier.write_report(receipt_path, {**receipt, "finished_utc": now()})
    print(json.dumps({"status": receipt["status"], "score_sha256": receipt["score_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    score_once(parser.parse_args().root.resolve())
