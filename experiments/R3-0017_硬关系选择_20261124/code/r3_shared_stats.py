"""Score graph-paired R16/R17 predictions on the frozen shared development cohort."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


SEEDS = (1701, 2701, 3701)
STRATA = ("competing_chain", "relation_order", "role_reversal")
BASIC = ("original", "paraphrase")
QUARTET = ("q0", "q1", "q2", "q3")
ANCILLARY = "unseen_template"
DRAWS = 10_000
BOOT_SEED = 2026112417
RANK = 500
NAN = float("nan")


def require(test, message):
    if not test:
        raise ValueError(message)


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        yield from map(json.loads, stream)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Truth(NamedTuple):
    gold: int
    first: str
    mentioned: str
    candidates: tuple
    template: str


class Decision(NamedTuple):
    h1: float | None
    relation: str | None
    prediction: int | None
    uniform: int


def load_inputs(cohort, predictions):
    truth, strata = {}, {}
    for r in rows(cohort):
        g, view = r["graph_id"], r["view"]
        require((g, view) not in truth, f"Duplicate cohort row {(g, view)}")
        require(strata.setdefault(g, r["stratum"]) == r["stratum"], f"Stratum changed {g}")
        require(r["first_relation"] == r["nodes"][r["gold"]][1], f"Incorrect first-relation label {g}")
        truth[g, view] = Truth(r["gold"], r["first_relation"], r["first_mentioned"],
                               tuple(r["candidates"]), r["template_id"])
    require(Counter(strata.values()) == {s: 200 for s in STRATA}, "Expected 200 graphs in each stratum")
    required = {(g, v) for g, s in strata.items() for v in
                BASIC + (ANCILLARY,) + (QUARTET if s == "relation_order" else ())}
    require(set(truth) == required, "Cohort view mismatch")
    data = defaultdict(dict)
    for r in rows(predictions):
        g, v, arm, seed = r["graph_id"], r["view"], r["variant"], r["seed"]
        t = truth[g, v]
        p, u, h, rel = r["predicted"], r["uniform_predicted"], r["h1_hit"], r["selected_relation"]
        require(seed in SEEDS and r.get("gold", t.gold) == t.gold, f"Gold/seed mismatch {(g, v, arm)}")
        require((p is None or p in t.candidates) and u in t.candidates, f"Off-shortlist prediction {(g, v, arm)}")
        require(h in (None, 0, 0.5, 1), f"Invalid H1 score {h}")
        if h is not None:
            require((h == .5 and rel is None) or (rel is not None and h == float(rel == t.first)),
                    f"Relation/tie score mismatch {(g, v, arm)}")
        key = (g, v, seed)
        require(key not in data[arm], f"Duplicate prediction {arm}, {key}")
        data[arm][key] = Decision(h, rel, p, u)
    for arm, mapping in data.items():
        require(len(mapping) == len(truth) * len(SEEDS), f"Incomplete arm {arm}")
    require({"first", "pool", "uniform"} <= data.keys(), "Missing prediction controls")
    return truth, strata, data


def avg(values):
    return sum(values) / len(values)


def view_metrics(arm, selected, seed, g, truth, data):
    ts = [truth[g, v] for v in selected]
    ds = [data[arm][g, v, seed] for v in selected]
    direct = [data["first"][g, v, seed].prediction for v in selected]
    pool = [data["pool"][g, v, seed].prediction for v in selected]
    baseline = [data["uniform"][g, v, seed].prediction for v in selected]
    require(all(d.uniform == u and f == t.candidates[0] and pp is not None
                for d, u, f, pp, t in zip(ds, baseline, direct, pool, ts, strict=True)), "Control mismatch")
    h1_na = [d.h1 is None for d in ds]
    h2_na = [d.prediction is None for d in ds]
    require(len(set(h1_na)) == len(set(h2_na)) == 1, "Endpoint availability varies across selected views")
    h1 = NAN if h1_na[0] else avg([d.h1 for d in ds])
    h2 = NAN if h2_na[0] else avg([d.prediction == t.gold for d, t in zip(ds, ts)])
    lex = avg([t.mentioned == t.first for t in ts])
    uniform = avg([u == t.gold for u, t in zip(baseline, ts)])
    first = avg([f == t.gold for f, t in zip(direct, ts)])
    pooled = avg([p == t.gold for p, t in zip(pool, ts)])
    return dict(h1_accuracy=h1, first_mentioned_accuracy=lex, h1_delta=h1-lex, h2_accuracy=h2,
                uniform_accuracy=uniform, h2_delta=h2-uniform, direct_accuracy=first, pool_accuracy=pooled,
                h2_minus_direct=h2-first, h2_minus_pool=h2-pooled, coverage=avg([t.gold in t.candidates for t in ts]))


def finite(value):
    return float(value) if np.isfinite(value) else None


def mean_or_na(values):
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def make_draws(groups):
    rng = np.random.default_rng(BOOT_SEED)
    weights = []
    for group in groups:
        n = len(group)
        indices = rng.integers(n, size=(DRAWS, n))
        counts = np.zeros((DRAWS, n), dtype=np.uint16)
        np.add.at(counts, (np.arange(DRAWS)[:, None], indices), 1)
        weights.append(counts.astype(np.float64) / (len(groups) * n))
    return weights


def paired_bounds(matrix, groups, weights):
    # One fixed, shared resampling table for all models, endpoints, seeds and views.
    samples = sum(w @ matrix[group, :] for group, w in zip(groups, weights))
    indexes = (249, RANK-1, 9749)
    ranked = np.partition(samples, indexes, axis=0)
    return ranked[list(indexes)]


def pack_estimates(names, values):
    return {key: finite(val) for key, val in zip(names, values, strict=True)}


def thresholds(summary):
    summary["h1_pass"] = bool(summary["h1_accuracy"] >= .80 - 1e-12 and summary["h1_lcb95"] > 1e-12)
    summary["h2_pass"] = bool(summary["h2_delta"] >= .02 - 1e-12 and summary["h2_lcb95"] > 1e-12)


def summarize(truth, strata, data, primary, graph_file):
    graphs = sorted(strata)
    groups = [np.array([i for i, g in enumerate(graphs) if strata[g] == s]) for s in STRATA]
    weights = make_draws(groups)
    summaries, primary_by_seed, primary_by_stratum, primary_sensitivity = {}, {}, {}, {}
    mode_list = ("primary", "original", ANCILLARY, "paraphrase")
    h1_diagnostics = {}
    values = {}
    for arm in sorted(data):
        arrays = {}
        for mode in mode_list:
            graph_seed_rows = []
            for g in graphs:
                selected = BASIC + (QUARTET if strata[g] == "relation_order" else ()) if mode == "primary" else (mode,)
                graph_seed_rows.append([view_metrics(arm, selected, s, g, truth, data) for s in SEEDS])
            names = list(graph_seed_rows[0][0])
            arrays[mode] = np.array([[list(r.values()) for r in rr] for rr in graph_seed_rows], dtype=float)
        values[arm] = arrays["primary"].mean(axis=1)
        summaries[arm] = {}
        for mode, a in arrays.items():
            all_seeds = a.mean(axis=1)
            summary = pack_estimates(names, all_seeds.mean(axis=0))
            summaries[arm][mode] = summary
            if arm != primary:
                continue
            blocks = [all_seeds, *(a[:, si, :] for si in range(len(SEEDS)))]
            stats = paired_bounds(np.concatenate(blocks, axis=1), groups, weights)
            for bi, (label, block) in enumerate(zip(("pooled", *map(str, SEEDS)), blocks, strict=True)):
                estimates = pack_estimates(names, block.mean(axis=0))
                for end, key in (("h1", "h1_delta"), ("h2", "h2_delta"),
                                 ("h2_minus_direct", "h2_minus_direct"), ("h2_minus_pool", "h2_minus_pool")):
                    j = bi * len(names) + names.index(key)
                    lo, lcb, hi = stats[:, j]
                    estimates[end + "_lcb95"] = finite(lcb)
                    estimates[end + "_ci95"] = [finite(lo), finite(hi)]
                thresholds(estimates)
                if label == "pooled":
                    primary_sensitivity[mode] = estimates
                elif mode == "primary":
                    primary_by_seed[label] = estimates
            if mode == "primary":
                for s, group in zip(STRATA, groups):
                    primary_by_stratum[s] = pack_estimates(names, all_seeds[group].mean(axis=0))
        # Orientation and invariance metrics have graph-level denominators.
        beh = []
        for i, g in enumerate(graphs):
            reversible = strata[g] == "relation_order"
            for si, seed in enumerate(SEEDS):
                def chosen(v):
                    return data[arm][g, v, seed]
                original, para = chosen("original"), chosen("paraphrase")
                base = dict(paraphrase_fact_stability=NAN if original.prediction is None else
                            float(original.prediction == para.prediction),
                            paraphrase_both_correct=NAN if original.prediction is None else
                            float(original.prediction == para.prediction == truth[g, "original"].gold),
                            paraphrase_relation_stability=NAN if original.h1 is None else
                            float(original.relation is not None and original.relation == para.relation),
                            quartet_all_correct=NAN, clause_order_prediction_stability=NAN,
                            semantic_reversal_flip=NAN, semantic_reversal_both_correct=NAN)
                if reversible and original.prediction is not None:
                    preds = [chosen(v).prediction for v in QUARTET]
                    hits = [p == truth[g, v].gold for p, v in zip(preds, QUARTET)]
                    base.update(quartet_all_correct=float(all(hits)),
                                clause_order_prediction_stability=avg([preds[0] == preds[2], preds[1] == preds[3]]),
                                semantic_reversal_flip=avg([preds[0] != preds[1], preds[2] != preds[3]]),
                                semantic_reversal_both_correct=avg([hits[0] and hits[1], hits[2] and hits[3]]))
                beh.append(list(base.values()))
                if arm == primary:
                    graph_file.writerow([arm, g, strata[g], seed, *arrays["primary"][i, si],
                                         *arrays["original"][i, si], *arrays[ANCILLARY][i, si],
                                         *(finite(x) for x in base.values())])
        summaries[arm]["behavior"] = {k: mean_or_na(col) for k, col in zip(base, np.array(beh).T)}
    for measure, i in (("h1", names.index("h1_accuracy")), ("h2", names.index("h2_accuracy"))):
        pairs = [(primary, name) for name in ("first_mentioned", "uniform", "shuffled", "random", "one", "four", "soft_role")]
        pairs += [("four", "one"), ("probe_four", "probe_one"), ("probe_one", "probe_one_shuffled_labels"),
                  ("probe_four", "probe_four_shuffled_labels")]
        matched = [(x, y) for x, y in pairs if x in values and y in values and
                   np.isfinite(values[x][:, i]).all() and np.isfinite(values[y][:, i]).all()]
        if matched:
            differences = np.column_stack([values[x][:, i] - values[y][:, i] for x, y in matched])
            cis = paired_bounds(differences, groups, weights)
            for j, (x, y) in enumerate(matched):
                h1_diagnostics[f"{measure}.{x}_minus_{y}"] = dict(delta=float(differences[:, j].mean()),
                    lcb95=float(cis[1, j]), ci95=[float(cis[0, j]), float(cis[2, j])])
    return dict(primary={"variant": primary, **primary_sensitivity["primary"]},
                primary_per_seed=primary_by_seed, primary_per_stratum=primary_by_stratum,
                primary_sensitivity=primary_sensitivity, variants=summaries, paired_controls=h1_diagnostics)


def main(args):
    truth, strata, data = load_inputs(args.cohort, args.predictions)
    candidates = set(data) & {"learned", "sharp_role"}
    if args.primary:
        require(args.primary in candidates, "Requested primary arm missing")
        primary = args.primary
    else:
        require(len(candidates) == 1, "Specify --primary for a mixed prediction file")
        primary = candidates.pop()
    require(all(d.h1 is not None and d.prediction is not None for d in data[primary].values()), "Incomplete primary")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "graph_metrics.csv").open("x", encoding="utf8", newline="") as f:
        csv_out = csv.writer(f)
        example_graph = next(iter(strata))
        columns = list(view_metrics(primary, ("original",), SEEDS[0], example_graph, truth, data))
        csv_out.writerow(["variant", "graph_id", "stratum", "seed",
                          *(f"{v}.{c}" for v in ("primary", "original", ANCILLARY) for c in columns),
                          "paraphrase_fact_stability", "paraphrase_both_correct", "paraphrase_relation_stability",
                          "quartet_all_correct", "clause_order_prediction_stability", "semantic_reversal_flip",
                          "semantic_reversal_both_correct"])
        result = summarize(truth, strata, data, primary, csv_out)
    result.update(scope="Development, same cohort across model versions", cohort_sha256=digest(args.cohort),
                  predictions_sha256=digest(args.predictions), total_graphs=len(strata), total_views=len(truth),
                  seed_ids=SEEDS, strata=dict(Counter(strata.values())), primary_views=BASIC + QUARTET,
                  ancillary_view=ANCILLARY, bootstrap=dict(generator="NumPy PCG64", seed=BOOT_SEED, draws=DRAWS,
                    unit="graph, stratified; all associated views and three model seeds kept together",
                    one_sided_95_order_statistic=RANK, two_sided_order_statistics=[250, 9750]),
                  thresholds=dict(h1_accuracy_min=.80, h1_gain_lcb_strict=0, h2_gain_min=.02, h2_gain_lcb_strict=0),
                  ancillary_denominators=dict(ordinary_graphs=600, reversible_graphs=200))
    with (args.out / "metrics.json").open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    print(json.dumps(result["primary"], allow_nan=False))


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "predictions", "out"):
        argp.add_argument("--" + name, required=True, type=Path)
    argp.add_argument("--primary", choices=("learned", "sharp_role"))
    main(argp.parse_args())
