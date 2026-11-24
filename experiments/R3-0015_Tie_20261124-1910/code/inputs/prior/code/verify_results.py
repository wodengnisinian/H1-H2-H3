"""Recount the two finished development panels and directly resample graphs from their predictions."""
import argparse
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ["first", "pool", "direction", "old", "uniform", "learned"]
SEEDS = [1701, 2701, 3701]
RESAMPLES = 10000
AUDIT_SEED = 20261124


def check(condition, context):
    if not condition:
        raise ValueError(context)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_logits(row):
    ids = list(map(str, row["candidates"]))
    check(row["predictions"]["first"] == ids[0], "invalid first position prediction")
    for model, (left, right) in row["logits"].items():
        check(np.isfinite(left) and np.isfinite(right), "non-finite prediction score")
        check(ids[int(right > left)] == row["predictions"][model], "saved prediction disagrees with logits")


def read_panel(run, dataset):
    inputs = {}
    for ex in read_jsonl(run / f"{dataset}-输入.jsonl"):
        key = (ex["graph_id"], ex["variant"])
        check(key not in inputs, "duplicate input")
        inputs[key] = (ex["kind"], str(ex["gold"]), tuple(str(v) for v in ex["candidates"]))
    panel = defaultdict(dict)
    for row in read_jsonl(run / f"{dataset}-预测.jsonl"):
        index = (row["graph_id"], row["variant"])
        check(inputs[index] == (row["kind"], row["gold"], tuple(row["candidates"])), "input mismatch")
        check(set(row["predictions"]) == set(MODELS) and set(row["logits"]) == set(MODELS[1:]),
              "missing or extra model")
        check_logits(row)
        seed = row["seed"]
        check(seed not in panel[index], "duplicate prediction")
        panel[index][seed] = tuple(int(row["predictions"][m] == row["gold"]) for m in MODELS)
    check(set(panel) == set(inputs), "incomplete prediction views")
    check(all(sorted(p) == SEEDS for p in panel.values()), "incomplete seed panel")
    return inputs, panel


def recount_primary(view, inputs, panel, saved):
    by_kind = defaultdict(list)
    covered = 0
    for (graph, variant), pred in panel.items():
        if variant == view:
            kind, gold, candidates = inputs[(graph, variant)]
            by_kind[kind].append([pred[s] for s in SEEDS])
            covered += gold in candidates
    matrices = [np.asarray(rows, dtype=np.int64) for _, rows in sorted(by_kind.items())]
    graphs = sum(len(arr) for arr in matrices)
    hits = sum(arr.sum(axis=(0, 1)) for arr in matrices)
    macro = [sum(Fraction(int(a[:, :, m].sum()), a.shape[0]*len(SEEDS)) for a in matrices) /
             len(matrices) for m in range(len(MODELS))]
    published = saved["question_views"][view]
    check(graphs == published["overall"]["graphs"], "graph denominator disagrees")
    check(covered == published["overall"]["coverage_by_graph"]["hits"], "coverage disagrees")
    for m, n, rate in zip(MODELS, hits, macro):
        counted = published["overall"]["accuracy"][m]
        check(int(n) == counted["hits"] and counted["denominator"] == graphs * len(SEEDS),
              f"hit count disagrees: {m}")
        check(abs(float(100*rate) - published["macro_accuracy_percent"][m]) < 1e-11, "macro disagrees")
    deltas = {m: float(100 * (macro[-1]-value)) for m, value in zip(MODELS[:-1], macro[:-1])}
    for control, diff in deltas.items():
        if control in ("first", "pool"):
            expected = saved["paired_comparisons"][view]["contrasts_learned_minus"][control]["delta_pp"]
        else:
            expected = saved["development_stability"]["learned_minus_macro_pp"][view][control]
        check(abs(diff - expected) < 1e-11, f"delta disagrees: {control}")

    # Resample individual graph indices explicitly. Seed predictions remain bundled.
    rng = np.random.default_rng(AUDIT_SEED)
    samples = np.zeros((RESAMPLES, 2))
    for arr in matrices:
        graph_diffs = arr[:, :, -1, None] - arr[:, :, :2]
        graph_means = graph_diffs.mean(axis=1)
        for j in range(0, RESAMPLES, 100):
            indices = rng.integers(len(arr), size=(100, len(arr)))
            samples[j:j+100] += graph_means[indices].mean(axis=1) * (100 / len(matrices))
    ordered = np.sort(samples, axis=0)
    intervals = {}
    for c, m in enumerate(MODELS[:2]):
        bound = saved["paired_comparisons"][view]["contrasts_learned_minus"][m]["lower_95_one_sided_pp"]
        lo, est, hi = (float(ordered[rank-1,c]) for rank in (400,500,600))
        intervals[m] = {"published_lower_pp": bound, "direct_draw_lower_pp": est, "delta_pp": est-bound,
                        "direct_draw_ranks_400_600_pp": [lo, hi],
                        "published_lower_in_rank_band": lo - 1e-10 <= bound <= hi + 1e-10}
    return {"graphs": graphs, "judgments": graphs*len(SEEDS), "covered_graphs": covered,
            "graphs_by_kind": {k: len(rows) for k,rows in sorted(by_kind.items())},
            "hits": {m: int(n) for m,n in zip(MODELS, hits)}, "learned_minus_pp": deltas,
            "bootstrap_crosscheck": intervals}


def recount_quartets(panel, saved):
    graphs = [g for g,v in panel if v == "q0"]
    counts = np.zeros(len(MODELS), dtype=np.int64)
    for graph in graphs:
        for s in SEEDS:
            answers = [panel[graph, f"q{j}"][s] for j in range(4)]
            counts += np.asarray(answers).all(axis=0)
    for m, n in zip(MODELS, counts):
        published = saved["four_question"]["overall"]["all_four"][m]
        check(int(n) == published["hits"] and published["denominator"] == len(graphs)*len(SEEDS),
              f"four-question discrepancy: {m}")
    return {"graphs": len(graphs), "graph_seed_quartets": len(graphs)*len(SEEDS),
            "hits": {m: int(n) for m,n in zip(MODELS, counts)}}


def verify(run):
    events = list(read_jsonl(run / "逐步记录.jsonl"))
    check(events[-1]["event"] == "finished", "the experiment has no finished marker")
    manifest = json.loads((run / "结果校验.json").read_bytes())
    for name, value in manifest.items():
        check(digest(run / name) == value, f"artifact digest mismatch: {name}")
    report = {"verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "run": run.name, "run_artifacts_checked": len(manifest),
              "resamples": RESAMPLES, "lower_rank": 500, "independent_sampling_seed": AUDIT_SEED,
              "sampling": "direct graph index sampling within kind, jointly over both contrasts",
              "datasets": {}}
    for dataset in ["archive_dev", "fresh_dev"]:
        scores_path = run / f"{dataset}-评分.json"
        predictions_path = run / f"{dataset}-预测.jsonl"
        saved = json.loads(scores_path.read_bytes())
        inputs, panel = read_panel(run, dataset)
        result = {v: recount_primary(v, inputs, panel, saved) for v in ("original","paraphrase")}
        result["quartets"] = recount_quartets(panel, saved)
        result["prediction_rows"] = len(panel)*len(SEEDS)
        result["predictions_sha256"] = digest(predictions_path)
        result["scores_sha256"] = digest(scores_path)
        # Byte-identical replay is separate from the independent recount and bootstrap.
        from evaluate import evaluate
        replay = evaluate(predictions_path)
        replay_text = json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if scores_path.read_bytes().endswith(b"\r\n"):
            replay_text = replay_text.replace("\n", "\r\n")
        replay_hash = hashlib.sha256(replay_text.encode("utf-8")).hexdigest()
        check(result["scores_sha256"] == replay_hash, f"score replay disagrees: {dataset}")
        report["datasets"][dataset] = dict(result, score_replay_matches=True, replay_sha256=replay_hash)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = verify(args.run)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

