"""Describe gate-conditioned H2 decisions and concrete errors in an R16 or R17 round."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


SEEDS = (1701, 2701, 3701)
STRATA = ("competing_chain", "relation_order", "role_reversal")
VIEWS = ("original", "paraphrase", "q0", "q1", "q2", "q3")
MODELS = ("actual", "forced_off", "uniform")


def require(test, message):
    if not test:
        raise ValueError(message)


def rows(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        yield from map(json.loads, stream)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def panel(group):
    graph_seeds = defaultdict(lambda: defaultdict(list))
    hits = Counter()
    strata = {}
    for row in group:
        g, seed, hit = row["graph_id"], row["seed"], row["hit"]
        graph_seeds[g][seed].append(hit)
        strata[g] = row["stratum"]
        hits.update({arm: int(hit[arm]) for arm in MODELS})
    changes = {}
    for control in MODELS[1:]:
        counts = Counter()
        for r in group:
            if r["choice"]["actual"] != r["choice"][control]:
                key = "rescued" if r["hit"]["actual"] else "lost" if r["hit"][control] else "different_errors"
            else:
                key = "unchanged_correct" if r["hit"]["actual"] else "unchanged_wrong"
            counts[key] += 1
        changes["actual_vs_" + control] = {k: counts[k] for k in
            ("rescued", "lost", "different_errors", "unchanged_correct", "unchanged_wrong")}
    group_graphs = Counter(strata.values())
    by_stratum = {}
    for kind in STRATA:
        ids = [g for g in graph_seeds if strata[g] == kind]
        if ids:
            by_stratum[kind] = {
                arm: mean(mean(mean(h[arm] for h in vv) for vv in graph_seeds[g].values()) for g in ids)
                for arm in MODELS}
    average = None
    if len(by_stratum) == len(STRATA):
        average = {arm: mean(r[arm] for r in by_stratum.values()) for arm in MODELS}
    return dict(decisions=len(group), graphs=len(graph_seeds), graphs_per_stratum=dict(group_graphs),
                hits={arm: hits[arm] for arm in MODELS}, raw_accuracy={
                    arm: hits[arm]/len(group) if group else None for arm in MODELS},
                first_relation_supervision=dict(fully_correct=sum(r["h1_hit"] == 1 for r in group),
                    wrong=sum(r["h1_hit"] == 0 for r in group), ties=sum(r["h1_hit"] == .5 for r in group)),
                gold_in_ordered_shortlist=sum(r["covered"] for r in group), matched_choice_changes=changes,
                graph_mean_by_stratum=by_stratum, equal_three_stratum_graph_mean=average)


def analyze(root):
    cohort_path = root / "data/cohort/cohort.jsonl.gz"
    pred_path = root / "results/prediction/predictions.jsonl.gz"
    primary = json.loads((root / "results/stats/metrics.json").read_bytes())["primary"]["variant"]
    require(primary in {"learned", "sharp_role"}, "Unknown primary variant")
    cohort = {(r["graph_id"], r["view"]): r for r in rows(cohort_path)}
    require(len(cohort) == 2600, "Unexpected cohort")
    predictions = defaultdict(dict)
    for r in rows(pred_path):
        arm = r["variant"]
        if arm in (primary, "direction", "uniform"):
            g, v, s = r["graph_id"], r["view"], r["seed"]
            require(arm not in predictions[g, v, s], "Duplicate prediction arm")
            predictions[g, v, s][arm] = r
    require(set(predictions) == {(g, v, s) for g, v in cohort for s in SEEDS}, "Missing model decisions")
    decisions = []
    closed_max = 0.
    for (g, v, s), arms in predictions.items():
        t = cohort[g, v]
        actual, direction, uniform = (arms[n] for n in (primary, "direction", "uniform"))
        gold, gate = t["gold"], actual["gate"]
        for obs in (actual, direction, uniform):
            require(obs["gold"] == gold and obs["gate"] == gate and obs["candidates"] == t["candidates"]
                    and obs["predicted"] in t["candidates"], "Unpaired predictions")
        require(actual["uniform_predicted"] == uniform["predicted"], "Uniform arm mismatch")
        if not gate:
            diff = max(abs(x-y) for x, y in zip(actual["logits"], direction["logits"], strict=True))
            closed_max = max(diff, closed_max)
            require(diff == 0 and actual["predicted"] == direction["predicted"], "Closed gate differs from direction")
        choice = dict(zip(MODELS, (actual["predicted"], direction["predicted"], uniform["predicted"])))
        decisions.append(dict(graph_id=g, view=v, seed=s, stratum=t["stratum"], gate=gate,
                              covered=gold in t["candidates"], h1_hit=actual["h1_hit"], choice=choice,
                              hit={arm: pred == gold for arm, pred in choice.items()}))
    selectors = dict(primary=VIEWS, original=("original",), unseen_template=("unseen_template",))
    panels = {}
    for label, allowed in selectors.items():
        subset = [r for r in decisions if r["view"] in allowed]
        panels[label] = {}
        for state, flag in (("open", True), ("closed", False)):
            group = [r for r in subset if r["gate"] is flag]
            panels[label][state] = {"all_seeds": panel(group),
                                    "per_seed": {str(s): panel([r for r in group if r["seed"] == s]) for s in SEEDS}}
    examples = {}
    natural = [r for r in decisions if r["view"] in VIEWS]
    natural.sort(key=lambda r: (VIEWS.index(r["view"]), r["graph_id"], r["seed"]))
    definitions = {
        "h1_wrong": lambda r: r["h1_hit"] == 0,
        "h1_exact_tie": lambda r: r["h1_hit"] == .5,
        "h2_wrong": lambda r: not r["hit"]["actual"],
        "h2_changed_from_direction": lambda r: r["choice"]["actual"] != r["choice"]["forced_off"],
        "h2_recovered_from_direction": lambda r: r["hit"]["actual"] and not r["hit"]["forced_off"],
        "h2_lost_from_direction": lambda r: not r["hit"]["actual"] and r["hit"]["forced_off"],
        "h2_changed_from_uniform": lambda r: r["choice"]["actual"] != r["choice"]["uniform"],
        "h2_wrong_closed_gate": lambda r: not r["gate"] and not r["hit"]["actual"],
    }
    for category, predicate in definitions.items():
        eligible = [r for r in natural if predicate(r)]
        ex = None
        if eligible:
            row = eligible[0]
            g, v, seed = row["graph_id"], row["view"], row["seed"]
            t, pred = cohort[g, v], predictions[g, v, seed]
            p = pred[primary]
            ex = dict(row, question_tokens=t["question"], question_text="".join(t["question"]),
                      gold_fact=t["gold"], gold_first_relation=t["first_relation"], first_mentioned=t["first_mentioned"],
                      selected_relation=p["selected_relation"], relation_names=p["relation_names"],
                      relation_masses=p["relation_masses"], indexed_candidates=[
                          {"fact_index": j, "triple": t["nodes"][j]} for j in t["candidates"]],
                      gold_triple=t["nodes"][t["gold"]], logits={
                          arm: pred[n]["logits"] for arm, n in zip(MODELS, (primary, "direction", "uniform"))})
        examples[category] = {"eligible_decisions": len(eligible), "example": ex}
    return dict(primary_variant=primary, cohort_sha256=digest(cohort_path), predictions_sha256=digest(pred_path),
                scope="Post hoc descriptive inspection of the fresh development predictions",
                units={"H1": "First relation in the ordered gold two-step path; exact mass ties receive half credit.",
                       "H2": "Exact agreement with the indexed gold first fact; uncovered examples count as misses.",
                       "actual": "Primary model's archived scores, using its natural gate.",
                       "forced_off": "Direction-base scores on the identical feature rows and ordered shortlist; setting G=0 in H2 removes its token correction exactly.",
                       "uniform": "Same graph, ordered shortlist, natural gate and frozen H2; uniform real-token weights.",
                       "counts": "Raw graph-view-seed decision totals. A reversible graph supplies six primary views, other graphs two.",
                       "graph_means": "Within each gate subgroup, average its available views within seed, then available seeds within graph, then graphs within stratum. Equal-three-stratum value requires a nonempty subgroup in all three strata.",
                       "comparisons": "Changes compare actual to each control on the same decision. Rescued = control wrong, actual correct; lost = control correct, actual wrong.",
                       "examples": "First qualifying primary decision by view order original/paraphrase/q0/q1/q2/q3, graph ID, seed. Missing examples have zero qualifying decisions."},
                closed_gate_max_abs_logit_difference=closed_max, panels=panels, examples=examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(analyze(args.round_dir), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf8") as stream:
            stream.write(output)
    else:
        print(output, end="")
