"""Scorer for the R42 H3 acceptance (lin3fix primary, frozen baselines)."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

MODELS = ("first", "pool", "direction", "old", "uniform", "learned", "lin3fix")
BASELINES = ("first", "pool")
MAIN = ("original", "paraphrase")
QUARTET = ("q0", "q1", "q2", "q3")
PROTOCOL_SEEDS = (1701, 2701, 3701)
PROTOCOL_KINDS = {"role_reversal", "relation_order", "competing_chain"}
BOOTSTRAP_SEED = 619482
REPLICATES = 10000
LOWER_RANK = 500
PRIMARY = "lin3fix"


@dataclass(frozen=True)
class Result:
    graph: str
    kind: str
    seed: int
    variant: str
    gold: str
    candidates: tuple
    gate: bool
    hits: tuple

    @property
    def covered(self):
        return self.gold in self.candidates


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_rows(source):
    with Path(source).open(encoding="utf-8-sig") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def collect(source):
    views = defaultdict(dict)
    signatures = {}
    graph_kinds = {}
    for line, raw in enumerate(read_rows(source), 1):
        context = f"row {line}"
        graph, kind, seed, variant = (raw[k] for k in ("graph_id", "kind", "seed", "variant"))
        gold, candidates, gate, preds = (raw[k] for k in ("gold", "candidates", "gate", "predictions"))
        check(all(isinstance(s, str) for s in (graph, kind, gold)), f"{context}: invalid graph, kind or gold")
        check(type(seed) is int and type(gate) is bool, f"{context}: invalid seed or gate")
        check(variant in MAIN + QUARTET, f"{context}: unrecognized question variant {variant}")
        check(isinstance(candidates, list) and len(candidates) == 2
              and all(isinstance(s, str) for s in candidates) and candidates[0] != candidates[1],
              f"{context}: expected two distinct candidate IDs")
        check(all(model in preds and preds[model] in candidates for model in MODELS),
              f"{context}: missing model or prediction outside the shared shortlist")
        check(preds["first"] == candidates[0], f"{context}: invalid direct-first prediction")
        check(graph_kinds.setdefault(graph, kind) == kind, f"{context}: graph changed stratum")
        key = (graph, seed)
        check(key not in views[variant], f"{context}: duplicate graph-seed-variant")
        signature = (gold, tuple(candidates), gate)
        check(signatures.setdefault((graph, variant), signature) == signature,
              f"{context}: gold, ordered candidates or gate changed across seeds")
        views[variant][key] = Result(graph, kind, seed, variant, gold, tuple(candidates), gate,
                                     tuple(preds[m] == gold for m in MODELS))
    originals = views["original"]
    check(originals, "original view is empty")
    seeds = sorted({seed for graph, seed in originals})
    graphs = {graph for graph, seed in originals}
    check(len(originals) == len(seeds) * len(graphs), "incomplete seed panel in original")
    check(views["paraphrase"].keys() == originals.keys(), "paraphrase must cover the original panel")
    for key, original in originals.items():
        p = views["paraphrase"][key]
        check(original.gold == p.gold and original.candidates == p.candidates and original.gate == p.gate,
              f"{key}: paraphrase changed gold, candidates or gate")
    eligible = views[QUARTET[0]].keys()
    check(all(views[q].keys() == eligible for q in QUARTET), "incomplete quartet")
    check(eligible <= originals.keys() and len(eligible) == len({graph for graph, seed in eligible}) * len(seeds),
          "quartet has no corresponding original or is missing a seed")
    return views, seeds


def count_rate(hits, n):
    return {"hits": int(hits), "denominator": n, "percent": 100 * int(hits) / n if n else None}


def summary(rows):
    n = len(rows)
    graph_coverage = {r.graph: r.covered for r in rows}
    return {
        "graphs": len(graph_coverage),
        "graph_seed_judgments": n,
        "coverage": count_rate(sum(r.covered for r in rows), n),
        "coverage_by_graph": count_rate(sum(graph_coverage.values()), len(graph_coverage)),
        "accuracy": {m: count_rate(sum(r.hits[i] for r in rows), n) for i, m in enumerate(MODELS)},
    }


def strata(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.kind].append(row)
    return groups


def main_scores(view, seeds):
    rows = list(view.values())
    grouped = strata(rows)
    by_kind = {kind: summary(grouped[kind]) for kind in sorted(grouped)}
    macro = {m: sum(Fraction(by_kind[k]["accuracy"][m]["hits"], by_kind[k]["graph_seed_judgments"])
                    for k in by_kind) / len(by_kind) for m in MODELS}
    return {
        "overall": summary(rows),
        "by_kind": by_kind,
        "by_seed": {str(s): summary([r for r in rows if r.seed == s]) for s in seeds},
        "by_gate": {label: summary([r for r in rows if r.gate == state])
                    for state, label in [(True, "on"), (False, "off")]},
        "macro_accuracy_percent": {m: float(100 * value) for m, value in macro.items()},
    }, macro


def quartet_scores(views, seeds):
    bundles = [[views[q][key] for q in QUARTET] for key in sorted(views[QUARTET[0]])]

    def calculate(group):
        return {
            "graphs": len({qs[0].graph for qs in group}),
            "graph_seed_quartets": len(group),
            "all_four": {m: count_rate(sum(all(r.hits[i] for r in qs) for qs in group), len(group))
                         for i, m in enumerate(MODELS)},
        }

    return {
        "overall": calculate(bundles),
        "by_kind": {k: calculate([q for q in bundles if q[0].kind == k])
                    for k in sorted({q[0].kind for q in bundles})},
        "by_seed": {str(s): calculate([q for q in bundles if q[0].seed == s]) for s in seeds},
    }


def bootstrap(view, seeds, macro):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = strata(view.values())
    replicates = np.zeros((REPLICATES, len(BASELINES)), dtype=np.float64)
    for kind, group in sorted(groups.items()):
        by_graph = defaultdict(list)
        for row in group:
            by_graph[row.graph].append(row)
        joint = []
        for panel in by_graph.values():
            successes = sum(row.hits[MODELS.index(PRIMARY)] for row in panel)
            joint.append([successes - sum(row.hits[MODELS.index(b)] for row in panel) for b in BASELINES])
        categories, counts = np.unique(np.array(joint, dtype=np.int64), axis=0, return_counts=True)
        n = len(by_graph)
        weights = rng.multinomial(n, counts / n, size=REPLICATES)
        replicates += (weights @ categories) / (n * len(seeds) * len(groups))
    lowers = np.sort(replicates, axis=0)[LOWER_RANK - 1]
    contrasts = {}
    for b, lower in zip(BASELINES, lowers):
        effect = macro[PRIMARY] - macro[b]
        lower = 0.0 if abs(lower) < 1e-14 else float(lower)
        contrasts[b] = {
            "delta_pp": float(100 * effect),
            "lower_95_one_sided_pp": 100 * lower,
            "point_pass": effect >= Fraction(5, 100),
            "interval_pass": lower > 0,
        }
    return {
        "graphs": len({row.graph for row in view.values()}),
        "graph_seed_judgments": len(view),
        "graphs_by_kind": {k: len({r.graph for r in v}) for k, v in sorted(groups.items())},
        "replicates": REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lower_rank_one_based": LOWER_RANK,
        "resampling": "paired graph within stratum, fixed seeds averaged per graph, equal stratum weight",
        "rng": "numpy.default_rng PCG64; joint multinomial",
        f"contrasts_{PRIMARY}_minus": contrasts,
        "both_thresholds_met": all(v["point_pass"] and v["interval_pass"] for v in contrasts.values()),
    }


def evaluate(source, output=None):
    views, seeds = collect(source)
    report = {"seeds": seeds, "primary": PRIMARY, "question_views": {}, "paired_comparisons": {}}
    for variant in MAIN:
        scores, macro = main_scores(views[variant], seeds)
        report["question_views"][variant] = scores
        report["paired_comparisons"][variant] = bootstrap(views[variant], seeds, macro)
    protocol_panel = tuple(seeds) == PROTOCOL_SEEDS and set(scores["by_kind"]) == PROTOCOL_KINDS
    report["h3"] = {
        "primary_view": "original",
        "reference_panel_three_fixed_seeds_and_three_kinds": protocol_panel,
        "both_thresholds_met": protocol_panel and report["paired_comparisons"]["original"]["both_thresholds_met"],
    }
    report["four_question"] = quartet_scores(views, seeds)
    if output is not None:
        out = Path(output)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.input, args.output)
