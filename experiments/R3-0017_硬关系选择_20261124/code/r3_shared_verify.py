"""Verify a closed R16/R17 development archive without running its models or scoring program.

Usage: python -B r3_shared_verify.py <absolute-round-directory>
Requires NumPy. All results are written to standard output.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


SEEDS = {1701, 2701, 3701}
STRATA = {"role_reversal", "relation_order", "competing_chain"}
BASE_VIEWS = {"original", "paraphrase", "unseen_template"}
QUARTET = {"q0", "q1", "q2", "q3"}
PRIMARY = {"R16": "learned", "R17": "sharp_role"}
COHORT = "data/cohort/cohort.jsonl.gz"
PREDICTIONS = "results/prediction/predictions.jsonl.gz"
METRICS = "results/stats/metrics.json"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field {key}")
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line, object_pairs_hook=unique_keys)
            except ValueError as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error


def verify_inventory(root):
    manifest, artifacts, remote = (read(root / name) for name in
                                   ("manifest.json", "artifacts.json", "logs/remote.json"))
    launch = read(root / "configs/launch.json")
    rid = launch["id"]
    require(rid in PRIMARY and root.parent.name == "experiments", "Expected R16 or R17 under experiments/")
    require(manifest["status"] in {"complete", "completed", "verified", "success", "succeeded"},
            "Manifest does not describe a completed round")
    hashes = {}

    def check(name, expected):
        require(isinstance(name, str) and not re.search(r"[\\:]", name)
                and all(s not in {"", ".", ".."} for s in name.split("/")), f"Invalid relative path {name!r}")
        if name not in hashes:
            path = root / name
            require(path.resolve().is_relative_to(root) and path.is_file(), f"Missing or external file {name}")
            with path.open("rb") as source:
                hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
        require(hashes[name] == expected, f"SHA-256 mismatch for {name}")

    files = manifest["files"]
    require(files and artifacts and remote["files"], "One or more inventories are empty")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(on_disk - {"manifest.json", "artifacts.json"} <= files.keys(), "Local files missing from manifest")
    for name, item in files.items():
        check(name, item["sha256"])
        require(type(item["bytes"]) is int and (root / name).stat().st_size == item["bytes"], f"Size mismatch {name}")
    for name, digest in artifacts.items():
        check(name, digest)
    for name, digest in remote["files"].items():
        check(name, digest)
    for name in (COHORT, PREDICTIONS, METRICS):
        require(name in remote["files"], f"Remote inventory omits {name}")
    if rid == "R17":
        require("cohort.jsonl.gz" in launch["cohort_anchors"], "Missing shared development cohort commitment")
        for name, digest in launch["cohort_anchors"].items():
            check(f"data/cohort/{name}", digest)

    rel = root.relative_to(root.parents[1]).as_posix()
    events = [e for e in records(root.parents[1] / "experiment_registry.jsonl")
              if e.get("id") == rid or e.get("path") == rel]
    require([e["event"] for e in events] == ["start", "end"], f"Invalid registry event sequence for {rid}")
    start, end = events
    require(all(e["path"] == rel and e["id"] == rid for e in events), "Registry path or id mismatch")
    require(datetime.fromisoformat(start["utc"]) <= datetime.fromisoformat(end["utc"]), "Registry time order")
    check("manifest.json", end["manifest_sha256"])
    check("artifacts.json", end["artifact_manifest_sha256"])
    require(end["code"] == 0 and read(root / "logs/exit.json")["exit_code"] == 0, "Unsuccessful exit")
    return rid, {"local_files": len(files), "artifacts": len(artifacts), "remote_files": len(remote["files"])}


def read_cohort(path):
    grouped = defaultdict(dict)
    indices = {}
    for row in records(path):
        graph, view = row["graph_id"], row["view"]
        require(view not in grouped[graph], f"Repeated cohort key {(graph, view)}")
        index = row["graph_index"]
        require(indices.setdefault(index, graph) == graph, f"Graph index collision at {index}")
        nodes = tuple(tuple(edge) for edge in row["nodes"])
        require(len(nodes) == len(set(nodes)) == 12 and all(len(edge) == 3 for edge in nodes), f"Invalid graph {graph}")
        q = row["question"]
        require(isinstance(q, list) and row["stratum"] in STRATA, f"Invalid input fields on {graph}")
        words = set(q)
        anchor, = words & {t for head, _, tail in nodes for t in (head, tail)}
        require(row["anchor"] == anchor, f"Anchor mismatch on {graph}")
        ordered = sorted(range(12), key=lambda i: (-len(words & set(nodes[i])), -(nodes[i][0] == anchor), i))[:2]
        require(ordered == row["candidates"], f"Ordered shortlist mismatch on {graph}/{view}")
        gold = row["gold"]
        require(type(gold) is int and 0 <= gold < 12 and nodes[gold][:2] == (anchor, row["first_relation"]),
                f"Gold/role mismatch on {graph}/{view}")
        relations = {edge[1] for edge in nodes} & words
        require(len(relations) == 2 and next(t for t in q if t in relations) == row["first_mentioned"],
                f"Lexical-order baseline mismatch on {graph}/{view}")
        second, = relations - {row["first_relation"]}
        valid = [i for i, (start, relation, bridge) in enumerate(nodes)
                 if (start, relation) == (anchor, row["first_relation"])
                 for h, r, tail in nodes if h == bridge and r == second and len({start, bridge, tail}) == 3]
        require(valid == [gold], f"Nonunique or erroneous path label on {graph}/{view}")
        grouped[graph][view] = row
    strata = Counter()
    position_counts = Counter()
    people_sets = set()
    for graph, views in grouped.items():
        original = views["original"]
        strata[original["stratum"]] += 1
        position_counts[original["gold"]] += 1
        people = frozenset(t for h, _, v in original["nodes"] for t in (h, v))
        require(people not in people_sets and len(people) == 6, f"Invalid or repeated personnel set on {graph}")
        people_sets.add(people)
        expected = BASE_VIEWS | (QUARTET if original["stratum"] == "relation_order" else set())
        require(set(views) == expected, f"Incomplete or unexpected view panel on {graph}")
        for view, row in views.items():
            require(all(row[k] == original[k] for k in ("nodes", "stratum", "candidates", "graph_index")),
                    f"Graph or ordered candidates changed on {graph}/{view}")
        require(original["gold"] == views["paraphrase"]["gold"] == views["unseen_template"]["gold"],
                f"Semantic drift on {graph}")
        if QUARTET <= views.keys():
            require(views["q0"]["gold"] == views["q2"]["gold"] and views["q1"]["gold"] == views["q3"]["gold"]
                    and views["q0"]["gold"] != views["q1"]["gold"], f"Invalid reversal quartet on {graph}")
    require(strata == dict.fromkeys(STRATA, 200) and len(indices) == len(grouped), "Invalid graph or stratum counts")
    require(position_counts == dict.fromkeys(range(12), 50), "Unbalanced original gold positions")
    return grouped


def check_predictions(path, cohort, variant):
    seen = set()
    per_graph = defaultdict(list)
    aliases = {"learned", "sharp_role", "uniform"}
    baseline_rows = {}
    primary_controls = {}
    for r in records(path):
        graph, view, seed = r["graph_id"], r["view"], r["seed"]
        key = (graph, view, seed)
        arm = r["variant"]
        if arm not in aliases:
            continue
        row = cohort[graph][view]
        require(seed in SEEDS and all(r[k] == row[k] for k in
                ("graph_index", "stratum", "first_relation", "first_mentioned", "gold", "candidates")),
                f"Prediction graph alignment {key}")
        if arm == "uniform":
            require(key not in baseline_rows, f"Repeated uniform control {key}")
            scores = r["logits"]
            require(len(scores) == 2 and all(math.isfinite(x) for x in scores)
                    and row["candidates"][int(scores[1] > scores[0])] == r["predicted"],
                    f"Uniform logits do not reconstruct the recorded choice {key}")
            baseline_rows[key] = r["predicted"]
        if arm != variant:
            continue
        require(key not in seen, f"Repeated primary prediction {key}")
        seen.add(key)
        candidates, gold = row["candidates"], row["gold"]
        require(r["covered"] == (gold in candidates) and r["predicted"] in candidates and r["uniform_predicted"] in candidates,
                f"Invalid candidate choices {key}")
        logits = r["logits"]
        require(len(logits) == 2 and all(math.isfinite(x) for x in logits)
                and candidates[int(logits[1] > logits[0])] == r["predicted"], f"Score/decision mismatch {key}")
        labels, masses = r["relation_names"], r["relation_masses"]
        require(len(masses) == len(labels) == 2 and all(math.isfinite(m) and 0 <= m <= 1.000001 for m in masses)
                and len(set(labels)) == 2 and row["first_relation"] in labels, f"Invalid H1 masses {key}")
        role = None if masses[0] == masses[1] else labels[int(masses[1] > masses[0])]
        hit = 0.5 if role is None else float(role == row["first_relation"])
        require(r["selected_relation"] == role and r["h1_hit"] == hit, f"H1 role/readout mismatch {key}")
        primary_controls[key] = r["uniform_predicted"]
        if view != "unseen_template":
            per_graph[graph].append([hit, float(row["first_mentioned"] == row["first_relation"]),
                                     float(r["predicted"] == gold), float(r["uniform_predicted"] == gold)])
    require(len(seen) == sum(map(len, cohort.values())) * len(SEEDS), "Primary graph/view/seed panel is incomplete")
    require(baseline_rows == primary_controls, "Uniform arm and primary's recorded baseline differ")
    return per_graph


def statistics(per_graph, cohort):
    rng = np.random.default_rng(2026112417)
    samples = np.zeros((10000, 2), dtype=float)
    point = np.zeros(4, dtype=float)
    for stratum in sorted(STRATA):
        matrix = np.array([np.mean(per_graph[g], axis=0) for g in sorted(cohort)
                           if cohort[g]["original"]["stratum"] == stratum])
        point += matrix.mean(axis=0) / len(STRATA)
        differences = matrix[:, (0, 2)] - matrix[:, (1, 3)]
        for start in range(0, len(samples), 100):
            draw = rng.integers(len(matrix), size=(100, len(matrix)))
            samples[start:start+100] += differences[draw].mean(axis=1) / len(STRATA)
    lower = np.sort(samples, axis=0)[499]
    return {
        "h1": {"accuracy": point[0], "baseline": point[1], "delta": point[0]-point[1],
               "lcb": lower[0], "pass": bool(point[0] >= .8 and lower[0] > 0)},
        "h2": {"accuracy": point[2], "baseline": point[3], "delta": point[2]-point[3],
               "lcb": lower[1], "pass": bool(point[2]-point[3] >= .02-1e-12 and lower[1] > 0)},
    }


def compare_metrics(result, primary, variant):
    require(primary["variant"] == variant, "Wrong primary variant in reported metrics")
    fields = {"h1": ("h1_accuracy", "first_mentioned_accuracy", "h1_delta", "h1_lcb95", "h1_pass"),
              "h2": ("h2_accuracy", "uniform_accuracy", "h2_delta", "h2_lcb95", "h2_pass")}
    for hypothesis, names in fields.items():
        expected = result[hypothesis]
        for key, name in zip(("accuracy", "baseline", "delta", "lcb", "pass"), names, strict=True):
            actual = primary[name]
            if key == "pass":
                require(type(actual) is bool and actual == expected[key], f"Gate mismatch for {name}")
            else:
                # The locked PCG64 stream and graph order should reproduce the specified quantile.
                # Allow for numerical summation order across NumPy/BLAS builds only.
                require(math.isclose(actual, expected[key], abs_tol=1e-8, rel_tol=0),
                        f"Metric mismatch for {name}: archived={actual}, recomputed={expected[key]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path, help="Absolute path to a closed development round")
    args = parser.parse_args()
    require(args.round_dir.is_absolute(), "Supply the absolute path to the closed round")
    root = args.round_dir.resolve(strict=True)
    rid, inventory = verify_inventory(root)
    variant = PRIMARY[rid]
    cohort = read_cohort(root / COHORT)
    outcomes = check_predictions(root / PREDICTIONS, cohort, variant)
    result = statistics(outcomes, cohort)
    compare_metrics(result, read(root / METRICS)["primary"], variant)
    print(json.dumps({"verified": True, "round": rid, "primary_variant": variant, "graphs": len(cohort),
                      "eligible_quartets": sum("q0" in rows for rows in cohort.values()),
                      "seed_count": len(SEEDS), "inventory": inventory, "primary": result},
                     ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
