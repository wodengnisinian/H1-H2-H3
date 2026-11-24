"""Measure candidate and fact presentation-order effects on frozen R12 development graphs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch


SEEDS = (1701, 2701, 3701)
STRATA = ("relation_order", "role_reversal", "competing_chain")
VIEWS = ("identity", "candidate_reverse", "fact_reverse")
METHODS = ("first", "pool", "sharp_role")
TOL = 2e-5
DRAWS = 10_000
BOOT_SEED = 619482

# Digests from the recovered development archive, fixed before the perturbation run.
R12_HASHES = {
    "results/development-inputs.jsonl.gz": "2d5c74c095109fb1819dc5c3f14e7397de89058af8364c2434736d611fae7cc3",
    "results/predictions-1701.jsonl.gz": "e605acf796e46ab58fb20b02c5e40bf2917a7ad30b5f2c9f5a00a86364184dfe",
    "results/predictions-2701.jsonl.gz": "da2480ac1ef50020c574ac5b3651fb981c2f9516f47ea0d149d839d1ee48b514",
    "results/predictions-3701.jsonl.gz": "cc74876737b2c20a2dfdb718f51e6208051e172d4b31cdb0fd531e19d8627d66",
    "results/checkpoint-1701.json": "98f1c920d07cb74a4409c4aaab32c8f21069e6e8cfcbefc8a5344dcd25a2e33d",
    "results/checkpoint-2701.json": "3e1653c209aee3161b1431fb7029d1b9464c25195c2136cafd0e9ab107012dd7",
    "results/checkpoint-3701.json": "2bd1a3da698ed71c9d816843c75cfd3d3a9299a11e5d743faed421f039f792e8",
    "results/configuration.json": "16f61d90c5acd601108cf4c0bb5d26e992ae66428d24fb12cf74c7f6e2a5736e",
    "snapshot/src/sharp_core.py": "f56c15184956476784b4922b453d4898178412e4c709f0f5dc632ab5dba86c54",
    "snapshot/src/role_core.py": "713909ab7c417c2b7f950938e291498c51651adfef3922a169067749a509ab8f",
}
INPUT_HASHES = {
    "prior/code/model.py": "f85636ffa19443880e48d953c7d552af3d6bda5238c60ad424f445202c2c2b0f",
    "b13/inputs/b12-code/tie_rank.py": "41c5b4b19b67be9ffe6a5735a15275ba152921466244771585d2bc324e4c64f9",
    "b13/inputs/b12-code/evaluate.py": "68c64cf602d16f6477011fe4092677775a178a3a7c5b32b0da057443b38b0501",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_rows(path):
    records = {}
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["split"], row["variant"]) != ("fresh", "original"):
                continue
            key = row["graph_id"]
            require(key not in records, f"Duplicate development graph {key}")
            records[key] = row
    require(Counter(r["kind"] for r in records.values()) == {s: 200 for s in STRATA},
            f"Development graph membership mismatch: {path}")
    return records


def tie_groups(doc, parse_question):
    """Count ties before the original direction rule uses source index to break them."""
    anchor, _ = parse_question(doc.question, doc.nodes)
    words = set(doc.question)
    keys = [(len(words.intersection(fact)), int(anchor is not None and fact[0] == anchor))
            for fact in doc.nodes]
    ordered = sorted(range(len(keys)), key=lambda i: (-keys[i][0], -keys[i][1], i))
    first, second, third = (keys[i] for i in ordered[:3])
    return tuple(ordered[:2]), first == second, second == third


def make_views(bundle, raw):
    from tie_rank import parse_question

    clean, labels = [], []
    for r in raw.values():
        nodes = tuple(map(tuple, r["nodes"]))
        require(len(nodes) == len(set(nodes)) == 12 and all(len(n) == 3 for n in nodes),
                f"Invalid graph structure: {r['graph_id']}")
        require(type(r["gold"]) is int and 0 <= r["gold"] < 12, "Gold index outside graph")
        clean.append(bundle.trainer.Example(r["graph_id"], r["kind"], tuple(r["question"]),
                                            nodes, tuple(r["candidates"]), None))
        labels.append(r["gold"])
    identity = bundle.evaluator.shortlist(clean, "direction")
    require(all(a.candidates == b.candidates for a, b in zip(clean, identity, strict=True)),
            "Cannot reproduce archived shortlist")
    flipped = [replace(d, candidates=d.candidates[::-1]) for d in identity]
    reversed_facts = [replace(d, nodes=d.nodes[::-1], candidates=()) for d in identity]
    reranked = bundle.evaluator.shortlist(reversed_facts, "direction")
    views = dict(zip(VIEWS, (identity, flipped, reranked), strict=True))
    graphs = []
    for i, d in enumerate(identity):
        entry = {"graph_id": d.doc_id, "kind": d.stratum, "gold": labels[i], "views": {}}
        for name, docs in views.items():
            current = docs[i]
            ordered, top_tie, cutoff_tie = tie_groups(current, parse_question)
            if name != "candidate_reverse":
                require(ordered == current.candidates, "Direction shortlist tie-key reconstruction failed")
            canonical = [11 - c if name == "fact_reverse" else c for c in current.candidates]
            for j, c in enumerate(canonical):
                require(d.nodes[c] == current.nodes[current.candidates[j]], "Canonical index mismatch")
            entry["views"][name] = {
                "ordered_candidates": canonical,
                "covered": labels[i] in canonical,
                "set_changed": set(canonical) != set(d.candidates),
                "order_changed": canonical != list(d.candidates),
                "retrieval_first_tie": top_tie,
                "retrieval_cutoff_tie": cutoff_tie,
            }
        graphs.append(entry)
    return views, graphs


@torch.inference_mode()
def infer(model, sharp, bundle, docs, ck, batch_size):
    f = model.extract(bundle, docs, batch_size)
    mask, x, classes, _ = sharp.features(docs, f.mask.shape[1], ck)
    _, hard, _, masses = sharp.decode(x, mask, classes, torch.tensor(ck["w"], dtype=torch.float32))
    logits = {"pool": f.pooled, "sharp_role": sharp.rank(f, hard)}
    require(all(s.shape == (len(docs), 2) and torch.isfinite(s).all() for s in logits.values()),
            "Invalid logits")
    return {k: v.numpy() for k, v in logits.items()}, f.active.tolist(), (masses[:, 0] == masses[:, 1]).tolist()


def score_seed(results, model, sharp, bundle, views, graphs, seed, batch_size):
    ck = json.loads((results / f"checkpoint-{seed}.json").read_bytes())
    require((ck["model"], ck["decoder"], ck["seed"]) ==
            ("sharp_role", "max-relation-mass-uniform-v1", seed), "Frozen R12 checkpoint mismatch")
    refs = selected_rows(results / f"predictions-{seed}.jsonl.gz")
    require(set(refs) == {g["graph_id"] for g in graphs}, "Prediction and input IDs differ")
    scores = {name: infer(model, sharp, bundle, docs, ck, batch_size) for name, docs in views.items()}
    diagnostics = {"archive_max_logit_error": {}, "candidate_swap_max_logit_error": {},
                   "candidate_swap_rows_over_tolerance": {}}
    for method in METHODS[1:]:
        baseline = scores["identity"][0][method]
        archived = np.asarray([refs[g["graph_id"]]["logits"][method] for g in graphs])
        diagnostics["archive_max_logit_error"][method] = float(np.max(np.abs(baseline - archived)))
        swapped = scores["candidate_reverse"][0][method]
        swap_error = np.max(np.abs(swapped - baseline[:, ::-1]), axis=1)
        diagnostics["candidate_swap_max_logit_error"][method] = float(swap_error.max())
        diagnostics["candidate_swap_rows_over_tolerance"][method] = int((swap_error > TOL).sum())
    require(max(diagnostics["archive_max_logit_error"].values()) <= TOL, "Archived identity logit parity failed")
    rows = []
    for i, g in enumerate(graphs):
        ref = refs[g["graph_id"]]
        require(type(ref["seed"]) is int and type(ref["gold"]) is int
                and type(ref["covered"]) is bool and type(ref["gate"]) is bool and type(ref["tie"]) is bool
                and all(type(c) is int for c in ref["candidates"])
                and all(type(ref["predictions"][m]) is int for m in METHODS),
                "Invalid archived identity metadata types")
        require((ref["seed"], ref["kind"], ref["gold"], ref["candidates"], ref["covered"]) ==
                (seed, g["kind"], g["gold"], g["views"]["identity"]["ordered_candidates"],
                 g["views"]["identity"]["covered"]), "Archived identity metadata mismatch")
        row = {"graph_id": g["graph_id"], "seed": seed, "kind": g["kind"], "gold": g["gold"], "views": {}}
        for name in VIEWS:
            logits, gates, role_ties = scores[name]
            candidates = g["views"][name]["ordered_candidates"]
            predictions = {"first": candidates[0]}
            score_ties, near_ties, values = {}, {}, {}
            for method, all_scores in logits.items():
                pair = all_scores[i]
                predictions[method] = candidates[int(pair.argmax())]
                score_ties[method] = bool(pair[0] == pair[1])
                near_ties[method] = bool(abs(float(pair[0]) - float(pair[1])) <= 2*TOL)
                values[method] = pair.tolist()
            if name == "identity":
                require(all(ref["predictions"][m] == predictions[m] for m in METHODS)
                        and ref["gate"] == gates[i] and ref["tie"] == role_ties[i],
                        f"Identity prediction parity failed for {seed}/{g['graph_id']}")
            row["views"][name] = dict(**g["views"][name], predictions=predictions, logits=values,
                                      score_ties=score_ties, near_ties=near_ties,
                                      gate=gates[i], role_mass_tie=role_ties[i])
        rows.append(row)
    return rows, diagnostics


def seed_table(rows):
    table = []
    for s in SEEDS:
        rs = [r for r in rows if r["seed"] == s]
        for view in VIEWS:
            for m in METHODS:
                hits = changed = nontie_changes = coverage = 0
                for r in rs:
                    before, after = r["views"]["identity"], r["views"][view]
                    different = before["predictions"][m] != after["predictions"][m]
                    changed += different
                    hits += after["predictions"][m] == r["gold"]
                    coverage += after["covered"]
                    if m != "first" and different and not (before["near_ties"][m] or after["near_ties"][m]):
                        nontie_changes += 1
                table.append(dict(seed=s, view=view, method=m, graphs=len(rs), correct=hits,
                                  top1=hits/len(rs), covered=coverage, changed_predictions=changed,
                                  changed_without_near_tie=nontie_changes if m != "first" else None))
    return table


def tables(graphs, rows):
    """One paired observation per graph for inference. Descriptive counts retain seeds."""
    by_graph = defaultdict(list)
    for r in rows:
        by_graph[r["graph_id"]].append(r)
    counts, contrasts, retrieval = [], [], []
    differences = {}
    for view in VIEWS:
        differences[(view, "coverage", "identity_coverage")] = [
            int(g["views"][view]["covered"]) - int(g["views"]["identity"]["covered"]) for g in graphs]
        for method in METHODS:
            for ref_view, ref_method in [("identity", method)] + (
                    [(view, "first"), (view, "pool")] if method == "sharp_role" else []):
                label = "identity_" + method if ref_view == "identity" and ref_method == method else view + "_" + ref_method
                differences[(view, method, label)] = [
                    np.mean([int(r["views"][view]["predictions"][method] == g["gold"]) -
                             int(r["views"][ref_view]["predictions"][ref_method] == g["gold"])
                             for r in by_graph[g["graph_id"]]]) for g in graphs]
    for kind in ("all", *STRATA):
        gs = [g for g in graphs if kind == "all" or g["kind"] == kind]
        rs = [r for g in gs for r in by_graph[g["graph_id"]]]
        for view in VIEWS:
            vs = [g["views"][view] for g in gs]
            retrieval.append(dict(stratum=kind, view=view, graphs=len(gs),
                                  covered=sum(v["covered"] for v in vs),
                                  changed_sets=sum(v["set_changed"] for v in vs),
                                  changed_ordered_pairs=sum(v["order_changed"] for v in vs),
                                  first_rank_ties=sum(v["retrieval_first_tie"] for v in vs),
                                  cutoff_ties=sum(v["retrieval_cutoff_tie"] for v in vs),
                                  coverage_gained=sum(v["covered"] and not g["views"]["identity"]["covered"]
                                                      for v, g in zip(vs, gs, strict=True)),
                                  coverage_lost=sum(not v["covered"] and g["views"]["identity"]["covered"]
                                                    for v, g in zip(vs, gs, strict=True))))
            for method in METHODS:
                current = [r["views"][view] for r in rs]
                hits = [v["predictions"][method] == r["gold"] for v, r in zip(current, rs, strict=True)]
                old = [r["views"]["identity"]["predictions"][method] for r in rs]
                counts.append(dict(stratum=kind, view=view, method=method, graphs=len(gs), decisions=len(rs),
                                   correct=sum(hits), top1=sum(hits)/len(hits),
                                   changed_predictions=sum(v["predictions"][method] != o for v,o in zip(current,old,strict=True)),
                                   corrected=sum(h and o != r["gold"] for h,o,r in zip(hits,old,rs,strict=True)),
                                   broken=sum(not h and o == r["gold"] for h,o,r in zip(hits,old,rs,strict=True)),
                                   score_ties=sum(v["score_ties"][method] for v in current) if method != "first" else None,
                                   near_ties=sum(v["near_ties"][method] for v in current) if method != "first" else None,
                                   role_mass_ties=sum(v["role_mass_tie"] for v in current) if method == "sharp_role" else None))
    # The same indices serve every contrast and its stratum-specific interval.
    names = list(differences)
    matrix = np.array([differences[n] for n in names], dtype=np.float64).T
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_means, point_means = {}, {}
    for kind in STRATA:
        block = matrix[[g["kind"] == kind for g in graphs]]
        draws = rng.integers(len(block), size=(DRAWS, len(block)))
        # Count resampled graphs before multiplying by contrast vectors, bounding peak memory.
        multiplicities = np.zeros(draws.shape, dtype=np.int32)
        np.add.at(multiplicities, (np.arange(DRAWS)[:, None], draws), 1)
        bootstrap_means[kind] = (multiplicities @ block) / len(block)
        point_means[kind] = block.mean(0)
    bootstrap_means["all"] = np.mean(list(bootstrap_means.values()), axis=0)
    point_means["all"] = np.mean(list(point_means.values()), axis=0)
    for kind in ("all", *STRATA):
        lo, hi = np.quantile(bootstrap_means[kind], [.025, .975], axis=0, method="inverted_cdf")
        for j, (view, method, reference) in enumerate(names):
            contrasts.append(dict(stratum=kind, view=view, method=method, reference=reference,
                                  delta=float(point_means[kind][j]), low95=float(lo[j]), high95=float(hi[j])))
    return counts, retrieval, contrasts


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf8") as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main(args):
    sys.dont_write_bytecode = True
    r12 = args.r3.resolve() / "experiments/10-R12"
    inputs = args.inputs.resolve()
    results = r12 / "results"
    for root, frozen in [(r12, R12_HASHES), (inputs, INPUT_HASHES)]:
        for name, expected in frozen.items():
            require(sha(root / name) == expected, f"Frozen file changed: {root / name}")
    require(not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())),
            "Output path must be a new or empty directory")
    config = json.loads((results / "configuration.json").read_bytes())["protocol"]
    require(tuple(config["seeds"]) == SEEDS and config["id"] == "R12", "Unexpected source protocol")
    sys.path[:0] = [str(r12 / "snapshot/src"), str(inputs / "prior/code")]
    import model
    import sharp_core

    raw = selected_rows(results / "development-inputs.jsonl.gz")
    reports, rows = {}, []
    for i, seed in enumerate(SEEDS):
        bundle = model.load_bundle(inputs / "b13", seed)
        if i == 0:
            views, graphs = make_views(bundle, raw)
        seed_rows, reports[seed] = score_seed(results, model, sharp_core, bundle, views, graphs, seed,
                                              config["feature_batch"])
        rows.extend(seed_rows)
        print(f"Verified identity and scored perturbations for seed {seed}", flush=True)
    counts, retrieval, contrasts = tables(graphs, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in [("graphs", graphs), ("predictions", rows)]:
        with gzip.open(args.out / f"{name}.jsonl.gz", "wt", encoding="utf8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    csv_file(args.out / "accuracy.csv", counts)
    csv_file(args.out / "per_seed.csv", seed_table(rows))
    csv_file(args.out / "retrieval.csv", retrieval)
    csv_file(args.out / "paired_contrasts.csv", contrasts)
    files = ("accuracy.csv", "per_seed.csv", "retrieval.csv", "paired_contrasts.csv",
             "graphs.jsonl.gz", "predictions.jsonl.gz")
    summary = dict(experiment="Frozen R12 development order perturbation", split="fresh", variant="original",
                   graphs=len(graphs), graphs_per_stratum=200, graph_seeds=len(rows), seeds=SEEDS, views=VIEWS,
                   formal_or_training_runs=0, identity_archive_rows_verified=len(rows),
                   accuracies=[r for r in counts if r["stratum"] == "all"],
                   retrieval=[r for r in retrieval if r["stratum"] == "all"],
                   paired_contrasts=[r for r in contrasts if r["stratum"] == "all"],
                   tie_policy="Argmax selects first candidate on exact equality; near-tie is abs(logit gap) <= 4e-5.",
                   fact_ids="Original zero-based positions. fact_reverse maps each new index j to 11-j.",
                   perturbations={"candidate_reverse": "Reverse the fixed two candidates after retrieval.",
                                  "fact_reverse": "Reverse all twelve fact rows, then rerun direction retrieval."},
                   bootstrap=dict(draws=DRAWS, seed=BOOT_SEED, interval="two-sided 95% percentile; inverse empirical CDF",
                                  unit="graph; seeds averaged within graph and strata equally weighted",
                                  unit_of_delta="accuracy proportion difference; multiply by 100 for percentage points"),
                   parity=reports, logit_tolerance=TOL, input_sha256=INPUT_HASHES, r12_sha256=R12_HASHES,
                   script_sha256=sha(Path(__file__)), outputs_sha256={n: sha(args.out / n) for n in files},
                   python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
                                            + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3", type=Path, required=True, help="R3 project root containing experiments/10-R12")
    parser.add_argument("--inputs", type=Path, required=True, help="Original unpacked inputs root containing prior and b13")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    arguments = parser.parse_args()
    require(not arguments.out.exists() or (arguments.out.is_dir() and not any(arguments.out.iterdir())),
            "Output path must be a new or empty directory")
    try:
        main(arguments)
    except Exception:
        arguments.out.mkdir(parents=True, exist_ok=True)
        with (arguments.out / "failure.json").open("x", encoding="utf8") as stream:
            json.dump({"status": "incomplete", "traceback": traceback.format_exc(),
                       "script_sha256": sha(Path(__file__))}, stream, ensure_ascii=False, indent=2)
        raise
