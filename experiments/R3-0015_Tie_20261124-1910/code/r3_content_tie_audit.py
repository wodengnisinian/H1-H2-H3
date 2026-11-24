"""R15 development diagnostic: replace the direction shortlist's final position tie-break."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import traceback

import numpy as np
import torch


R14_SHA = "affd20a5d4d1b1404d4e630e7eba8d8e03528825bd9dc7b3c049beee027e76f4"
RULES = ("source_index", "content_sha256")
ORDERS = ("original", "reversed", "shuffled")
SHUFFLE_DOMAIN = "R15-fact-order-v1\0"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_r14():
    path = Path(__file__).with_name("r3_dev_order_audit.py")
    if sha(path) != R14_SHA:
        raise ValueError("Adjacent R14 helper hash mismatch")
    spec = importlib.util.spec_from_file_location("r15_pinned_r14", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permutations(doc_id, size):
    seed_bytes = hashlib.sha256((SHUFFLE_DOMAIN + doc_id).encode("utf-8")).digest()
    shuffled = list(range(size))
    random.Random(int.from_bytes(seed_bytes, "big")).shuffle(shuffled)
    return seed_bytes.hex(), dict(original=list(range(size)), reversed=list(range(size-1, -1, -1)), shuffled=shuffled)


def content_shortlist(doc, parse_question):
    """Only the final tie-break differs from tie_rank.rank_candidates(..., 'direction')."""
    anchor, _ = parse_question(doc.question, doc.nodes)
    words = set(doc.question)
    keys = [(-len(words.intersection(fact)), -int(anchor is not None and fact[0] == anchor),
             digest(fact), fact) for fact in doc.nodes]
    ordered = sorted(range(len(keys)), key=keys.__getitem__)
    return tuple(ordered[:2])


def views_and_graphs(base, bundle, raw):
    from tie_rank import parse_question

    docs = {(rule, order): [] for rule in RULES for order in ORDERS}
    graphs = []
    for row in raw.values():
        nodes = tuple(map(tuple, row["nodes"]))
        base.require(len(nodes) == len(set(nodes)) == 12 and all(len(n) == 3 for n in nodes), "Bad original facts")
        base.require(type(row["gold"]) is int and 0 <= row["gold"] < 12, "Invalid gold index")
        template = bundle.trainer.Example(row["graph_id"], row["kind"], tuple(row["question"]), nodes, (), None)
        seed_hex, permutations_by_order = permutations(template.doc_id, len(nodes))
        fact_hashes = [digest(fact) for fact in nodes]
        graph = dict(graph_id=template.doc_id, kind=template.stratum, gold=row["gold"], shuffle_seed_hex=seed_hex,
                     canonical_fact_sha256=fact_hashes, distinct_fact_digests=len(set(fact_hashes)), presentations={})
        for order, local_to_canonical in permutations_by_order.items():
            local_nodes = tuple(nodes[j] for j in local_to_canonical)
            presented = replace(template, nodes=local_nodes)
            source = bundle.evaluator.shortlist([presented], "direction")[0]
            stable = replace(presented, candidates=content_shortlist(presented, parse_question))
            frozen_pair, first_tie, cutoff_tie = base.tie_groups(presented, parse_question)
            base.require(source.candidates == frozen_pair, "Frozen retrieval key reproduction failed")
            serialised = dict(local_to_canonical=local_to_canonical,
                              presented_fact_sha256=[fact_hashes[j] for j in local_to_canonical],
                              ordered_facts_sha256=digest(local_nodes),
                              first_tie_before_final_key=first_tie, cutoff_tie_before_final_key=cutoff_tie, rules={})
            for rule, document in zip(RULES, (source, stable), strict=True):
                docs[rule, order].append(document)
                local_pair = document.candidates
                pair = [local_to_canonical[j] for j in local_pair]
                base.require(len(set(pair)) == 2 and all(document.nodes[j] == nodes[k]
                             for j, k in zip(local_pair, pair, strict=True)), "Canonical map mismatch")
                serialised["rules"][rule] = dict(local_candidates=list(local_pair), candidates=pair,
                                                  covered=row["gold"] in pair)
            if order == "original":
                base.require(list(source.candidates) == row["candidates"], "Original archived shortlist mismatch")
            graph["presentations"][order] = serialised
        graphs.append(graph)
    return docs, graphs


def read_r14(base, directory):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    base.require(summary["script_sha256"] == R14_SHA and summary["identity_archive_rows_verified"] == 1800,
                 "Unexpected R14 result provenance")
    path = directory / "predictions.jsonl.gz"
    base.require(sha(path) == summary["outputs_sha256"][path.name], "Invalid R14 prediction file hash")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream]
    index = {(r["graph_id"], r["seed"]): r for r in rows}
    base.require(len(index) == len(rows) == 1800, "Invalid R14 row keys")
    return index, {"summary_sha256": sha(summary_path), "predictions_sha256": sha(path)}


def score(base, model, sharp, bundle, docs, graphs, ck, results, batch_size, previous):
    seed = ck["seed"]
    refs = base.selected_rows(results / f"predictions-{seed}.jsonl.gz")
    base.require(set(refs) == {g["graph_id"] for g in graphs}, "Original graph membership mismatch")
    matrices = {condition: base.infer(model, sharp, bundle, ds, ck, batch_size) for condition, ds in docs.items()}
    rows = []
    max_ref_error = {"r12_identity": 0., "r14_source_presentations": 0.}
    for i, graph in enumerate(graphs):
        ref = refs[graph["graph_id"]]
        base.require((ref["seed"], ref["kind"], ref["gold"]) == (seed, graph["kind"], graph["gold"]),
                     "Original reference metadata mismatch")
        record = dict(graph_id=graph["graph_id"], kind=graph["kind"], gold=graph["gold"], seed=seed, rules={})
        for rule in RULES:
            record["rules"][rule] = {}
            for order in ORDERS:
                meta = graph["presentations"][order]["rules"][rule]
                matrices_by_method, gates, role_ties = matrices[rule, order]
                pairs = {m: v[i].tolist() for m, v in matrices_by_method.items()}
                predictions = {"first": meta["candidates"][0]}
                local_predictions = {"first": meta["local_candidates"][0]}
                for m, pair in pairs.items():
                    position = int(np.argmax(pair))
                    predictions[m] = meta["candidates"][position]
                    local_predictions[m] = meta["local_candidates"][position]
                row = dict(**meta, predictions=predictions, local_predictions=local_predictions, logits=pairs,
                           gate=gates[i], role_mass_tie=role_ties[i], score_ties={m:v[0] == v[1] for m,v in pairs.items()})
                record["rules"][rule][order] = row
                references = []
                if rule == "source_index" and order == "original":
                    references.append(("r12_identity", ref, ref["candidates"], ref["tie"]))
                if previous is not None and rule == "source_index" and order in ("original", "reversed"):
                    old = previous[graph["graph_id"], seed]
                    base.require((old["kind"], old["gold"]) == (graph["kind"], graph["gold"]), "R14 gold mismatch")
                    view = old["views"]["identity" if order == "original" else "fact_reverse"]
                    references.append(("r14_source_presentations", view, view["ordered_candidates"], view["role_mass_tie"]))
                for name, expected, candidates, expected_role_tie in references:
                    base.require(candidates == meta["candidates"] and expected["covered"] == meta["covered"]
                                 and expected["gate"] == gates[i] and expected_role_tie == role_ties[i]
                                 and all(expected["predictions"][m] == predictions[m] for m in base.METHODS),
                                 f"{name} decision parity failed for {graph['graph_id']}/{seed}")
                    for m, pair in pairs.items():
                        e = float(np.abs(np.asarray(pair) - expected["logits"][m]).max())
                        max_ref_error[name] = max(e, max_ref_error[name])
        rows.append(record)
    base.require(max(max_ref_error.values()) <= base.TOL, "Reference logit parity failed")
    if previous is None:
        max_ref_error["r14_source_presentations"] = None
    return rows, max_ref_error


def graph_values(base, graph, rows):
    """Return graph-level endpoints; all bootstrap draws keep the three seeds together."""
    values = {}
    strict_examples = {}
    for rule in RULES:
        views = [graph["presentations"][o]["rules"][rule] for o in ORDERS]
        original_pair = views[0]["candidates"]
        ordered_equal = all(v["candidates"] == original_pair for v in views[1:])
        sets_equal = all(set(v["candidates"]) == set(original_pair) for v in views[1:])
        values[rule, "all_presentations", "ordered_pair_invariant"] = int(ordered_equal)
        values[rule, "all_presentations", "candidate_set_invariant"] = int(sets_equal)
        predictions_equal = {}
        for method in base.METHODS:
            unchanged = [len({r["rules"][rule][o]["predictions"][method] for o in ORDERS}) == 1 for r in rows]
            predictions_equal[method] = unchanged
            values[rule, "all_presentations", f"{method}_seed_invariance"] = sum(unchanged)/len(unchanged)
            values[rule, "all_presentations", f"{method}_all_seeds_invariant"] = int(all(unchanged))
        strict = ordered_equal and all(predictions_equal["sharp_role"])
        strict_examples[rule] = strict
        values[rule, "all_presentations", "strict_primary"] = int(strict)
        for order, view in zip(ORDERS, views, strict=True):
            values[rule, order, "coverage"] = int(view["covered"])
            values[rule, order, "ordered_pair_changed_from_original"] = int(view["candidates"] != original_pair)
            values[rule, order, "set_changed_from_original"] = int(set(view["candidates"]) != set(original_pair))
            for method in base.METHODS:
                p = [r["rules"][rule][order]["predictions"][method] for r in rows]
                original = [r["rules"][rule]["original"]["predictions"][method] for r in rows]
                values[rule, order, f"{method}_top1"] = sum(t == graph["gold"] for t in p)/len(p)
                values[rule, order, f"{method}_changed_from_original"] = sum(a != b for a,b in zip(p,original,strict=True))/len(p)
    return values, strict_examples


def summaries(base, graphs, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["graph_id"]].append(row)
    index = {}
    graph_records = []
    for g in graphs:
        per_graph = grouped[g["graph_id"]]
        base.require(sorted(r["seed"] for r in per_graph) == list(base.SEEDS), "Seed membership mismatch")
        values, strict = graph_values(base, g, per_graph)
        index[g["graph_id"]] = values
        graph_records.append({**g, "strict_primary_pass": strict})
    endpoints = sorted({(order, metric) for rule, order, metric in next(iter(index.values()))})
    # Retain absolute levels for both rules and signed paired content-minus-source deltas.
    names = [(cond, order, metric) for order, metric in endpoints for cond in (*RULES, "content_minus_source")]
    blocks = {}
    rng = np.random.default_rng(base.BOOT_SEED)
    resamples = {}
    for kind in base.STRATA:
        gs = [g for g in graphs if g["kind"] == kind]
        block = []
        for g in gs:
            v = index[g["graph_id"]]
            block.append([v[cond, o, m] if cond in RULES else v["content_sha256", o, m] - v["source_index", o, m]
                          for cond, o, m in names])
        block = np.asarray(block)
        blocks[kind] = block.mean(0)
        picks = rng.integers(len(gs), size=(base.DRAWS, len(gs)))
        freq = np.zeros(picks.shape, dtype=np.int32)
        np.add.at(freq, (np.arange(base.DRAWS)[:, None], picks), 1)
        resamples[kind] = freq @ block / len(gs)
    blocks["all"] = np.mean(list(blocks.values()), axis=0)
    resamples["all"] = np.mean(list(resamples.values()), axis=0)
    table = []
    for kind in ("all", *base.STRATA):
        low, high = np.quantile(resamples[kind], [.025, .975], axis=0, method="inverted_cdf")
        for i, (cond, order, metric) in enumerate(names):
            table.append(dict(stratum=kind, condition=cond, presentation=order, metric=metric,
                              estimate=float(blocks[kind][i]), low95=float(low[i]), high95=float(high[i])))
    per_seed = []
    for kind in ("all", *base.STRATA):
        for seed in base.SEEDS:
            rs = [r for r in rows if r["seed"] == seed and (kind == "all" or r["kind"] == kind)]
            for rule in RULES:
                for order in ORDERS:
                    for method in base.METHODS:
                        hits = changed = ties = coverage = 0
                        for row in rs:
                            original = row["rules"][rule]["original"]
                            current = row["rules"][rule][order]
                            p = current["predictions"][method]
                            hits += p == row["gold"]
                            coverage += current["covered"]
                            changed += p != original["predictions"][method]
                            if method != "first":
                                ties += current["score_ties"][method]
                        per_seed.append(dict(stratum=kind, seed=seed, rule=rule, presentation=order, method=method,
                                             graphs=len(rs), correct=hits, top1=hits/len(rs), covered=coverage,
                                             changed_predictions=changed, exact_score_ties=ties if method!="first" else None))
    return graph_records, table, per_seed


def run(args):
    base = load_r14()
    r12 = args.r3.resolve() / "experiments/10-R12"
    inputs = args.inputs.resolve()
    results = r12 / "results"
    for root, frozen in [(r12, base.R12_HASHES), (inputs, base.INPUT_HASHES)]:
        for name, expected in frozen.items():
            base.require(sha(root / name) == expected, f"Frozen file changed: {root / name}")
    protocol = json.loads((results / "configuration.json").read_bytes())["protocol"]
    base.require(tuple(protocol["seeds"]) == base.SEEDS and protocol["id"] == "R12", "Wrong source protocol")
    sys.path[:0] = [str(r12 / "snapshot/src"), str(inputs / "prior/code")]
    import model
    import sharp_core

    raw = base.selected_rows(results / "development-inputs.jsonl.gz")
    previous, prior_hashes = read_r14(base, args.r14) if args.r14 else (None, None)
    rows = []
    parity = {}
    for i, seed in enumerate(base.SEEDS):
        bundle = model.load_bundle(inputs / "b13", seed)
        if i == 0:
            docs, graphs = views_and_graphs(base, bundle, raw)
        ck = json.loads((results / f"checkpoint-{seed}.json").read_bytes())
        base.require((ck["seed"], ck["model"], ck["decoder"]) == (seed, "sharp_role", "max-relation-mass-uniform-v1"),
                     "Frozen checkpoint mismatch")
        per_seed, parity[seed] = score(base, model, sharp_core, bundle, docs, graphs, ck, results,
                                       protocol["feature_batch"], previous)
        rows.extend(per_seed)
        print(f"Verified reference and scored six presentations/rules for seed {seed}", flush=True)
    graph_records, metrics, per_seed = summaries(base, graphs, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in [("predictions", rows), ("graphs", graph_records)]:
        with gzip.open(args.out / f"{name}.jsonl.gz", "wt", encoding="utf8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    base.csv_file(args.out / "metrics.csv", metrics)
    base.csv_file(args.out / "per_seed.csv", per_seed)
    outputs = ["metrics.csv", "per_seed.csv", "graphs.jsonl.gz", "predictions.jsonl.gz"]
    primary_counts = {rule:sum(g["strict_primary_pass"][rule] for g in graph_records) for rule in RULES}
    summary = dict(experiment="R15 content tie-break on frozen R12 development", seeds=base.SEEDS, graphs=len(graphs),
                   graph_seeds=len(rows), presentations=ORDERS, new_training_updates=0, new_formal_assessments=0,
                   split="fresh", variant="original", methods=base.METHODS, graphs_per_stratum=200,
                   strict_primary="Same ordered canonical candidate pair and same canonical sharp_role prediction within each seed across all three presentations for every graph.",
                   primary_passing_graphs=primary_counts,
                   primary_pass=primary_counts["content_sha256"] == len(graphs),
                   strict_source_comparison="Empirical pass count under identical criteria for the historical source-index rule.",
                   content_key="Retain direction rule's lexical overlap and anchor preference; sort ties ascending by SHA-256 of compact UTF-8 JSON fact triple; on digest ties sort by literal Unicode tuple.",
                   entity_scope="Fact hashes depend on literal entity and relation strings; this experiment preserves these strings and evaluates three presentations of each graph.",
                   digest_collision_graphs=sum(g["distinct_fact_digests"] < 12 for g in graphs),
                   original_r12_parity_rows=len(rows), r14_parity_rows_eligible=len(rows)*2 if previous is not None else 0,
                   r14_reference=prior_hashes, parity_max_logit_abs_error=parity, tolerance=base.TOL,
                   shuffle="Python random.Random(int.from_bytes(SHA256(('R15-fact-order-v1' + NUL + graph_id).encode('utf-8')).digest(),'big')).shuffle(original_indices)",
                   bootstrap=dict(unit="graph, preserving presentations, rules, methods and three fixed seeds; equal stratum weights",
                                  draws=base.DRAWS, seed=base.BOOT_SEED, interval="two-sided 95%, inverse empirical CDF",
                                  contrasts="content_minus_source = content rule minus source-index rule within each graph",
                                  estimates="empirical proportions or proportion differences, conditional on the observed development graphs"),
                   all_graph_metrics=[r for r in metrics if r["stratum"] == "all"],
                   pinned_r14_helper_sha256=R14_SHA, script_sha256=sha(Path(__file__)), r12_sha256=base.R12_HASHES,
                   inputs_sha256=base.INPUT_HASHES, outputs_sha256={n:sha(args.out/n) for n in outputs},
                   python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                                            + "\n", encoding="utf8")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("r3", "inputs", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--r14", type=Path, help="Optional verified R14 output directory for reference parity checks")
    args = parser.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--out must be a new or empty directory")
    try:
        run(args)
    except Exception:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out/"failure.json").open("x", encoding="utf8") as stream:
            json.dump({"status":"incomplete", "traceback":traceback.format_exc(), "script_sha256":sha(Path(__file__))},
                      stream, indent=2, ensure_ascii=False)
        raise
