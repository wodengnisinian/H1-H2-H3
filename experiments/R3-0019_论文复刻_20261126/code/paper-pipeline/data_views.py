"""Build frozen-direction Top-2 training and development views for first-hop ranking."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import random
import runpy
import sys


ARCHIVE_SHA = {
    "train.jsonl": "e3eada4ea77d212ac29daf227ea842fe62e9e542c3d0b30874824bf3c20c0327",
    "dev.jsonl": "c6e569b0f1bea757c9d447d5df4bdccfc8e74358f85761d9f77d1416b5377071",
    "train-clause-views.jsonl": "a8f08e22d33df00caf99d39e83e881b6dfdc2dcbeb2d67eef5f39c6f1ede1fa5",
    "generate.py": "c6d8eaa5081868946bc24969ddeed4450d6685ae9bd782a3cdb70dd43717b46f",
    "generate_cohort.py": "20cf4f05788fb9a438e2b385a0a190dcc0151752910444621d3a792a1fe014bb",
}
DATA_RELATIVE = Path("inputs/b15/inputs/b14/inputs/server-run/data")
PARAPHRASE = (
    "请 给出 第一步 的 记录 。 从 {anchor} 开始 ， 开头 一段 追踪 {first} 关系 。 第二 段 按 {second} 关系 找 一人 。",
    "第二 段 按 {second} 关系 找 一人 。 从 {anchor} 开始 ， 开头 一段 追踪 {first} 关系 。 请 给出 第一步 的 记录 。",
)
FIRST_CLAUSE = "以 {anchor} 为 起点 ， 第一步 按 {first} 关系 找 一人"
SECOND_CLAUSE = "第二 段 按 {second} 关系 追踪"
ASK = "请 确定 开头 一段 对应 哪条 记录 ？"
QUARTET = tuple(" 。 ".join((a, b, ASK)) for a, b in (
    (FIRST_CLAUSE, SECOND_CLAUSE), (SECOND_CLAUSE, FIRST_CLAUSE)))
FRESH_TEMPLATES = (
    "从 {anchor} 出发 ， 首先 按 {first} 关系 找 一人 。 第二 段 沿 {second} 关系 追踪 。 第一步 对应 哪条 记录 ？",
    "请 确定 第一步 的 记录 。 以 {anchor} 为 起点 ， 开头 一段 追踪 {first} 关系 。 随后 按 {second} 关系 找 一人 。",
    "第二 段 按 {second} 关系 找 一人 。 此前 从 {anchor} 开始 ， 沿 {first} 关系 追踪 。 应 先 使用 哪条 记录 ？",
    "请 给出 开头 一段 对应 的 记录 。 最后 一段 追踪 {second} 关系 。 第一步 以 {anchor} 为 起点 ， 按 {first} 关系 找 一人 。",
)
STRATA = ("relation_order", "competing_chain", "role_reversal")
FIELDS = {"doc_id", "stratum", "question", "nodes"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def graph_hash(example):
    """Stable under fact permutation and every question paraphrase."""
    return canonical_hash(sorted(example.nodes))


def checked(path):
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    require(digest == ARCHIVE_SHA[path.name], f"Archive SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def roles_for(example, templates):
    question = example.question
    matches = [template for template in templates if len(template) == len(question)
               and all(word.startswith("{") or word == token for word, token in zip(template, question))]
    require(len(matches) == 1, f"Expected one source template for {example.doc_id}")
    roles = {word: token for word, token in zip(matches[0], question) if word.startswith("{")}
    return tuple(roles[k] for k in ("{anchor}", "{first}", "{second}"))


def paths(nodes, anchor, first, second):
    return [(i, j) for i, (h, r, bridge) in enumerate(nodes) if (h, r) == (anchor, first)
            for j, (head, rel, tail) in enumerate(nodes)
            if (head, rel) == (bridge, second) and len({anchor, bridge, tail}) == 3]


def load_view(raw_rows, trainer, evaluator):
    features, gold = [], []
    for raw in raw_rows:
        require(set(raw) == FIELDS | {"gold"}, "Unexpected archive fields")
        nodes = tuple(trainer.tokens(list(node)) for node in raw["nodes"])
        require(len(nodes) == len(set(nodes)) == 12 and all(len(node) == 3 for node in nodes),
                f"Invalid directed facts for {raw['doc_id']}")
        label = raw["gold"]
        require(type(label) is int and 0 <= label < 12, f"Invalid gold for {raw['doc_id']}")
        question = trainer.tokens(raw["question"])
        features.append(trainer.Example(raw["doc_id"], raw["stratum"], question, nodes,
                                        trainer.lexical_candidates(question, nodes), None))
        gold.append(label)
    require(len({ex.doc_id for ex in features}) == len(features), "Duplicate doc ID")
    return evaluator.shortlist(features, "direction"), gold


def make_question(ex, roles, style, evaluator):
    anchor, first, second = roles
    question = tuple(style.format(anchor=anchor, first=first, second=second).split())
    new = evaluator.shortlist([replace(ex, question=question)], "direction")[0]
    require(new.candidates == ex.candidates and new.gold is None, f"Shortlist drift: {ex.doc_id}")
    return new


def make_eval_views(original, templates, evaluator, *, paraphrases=PARAPHRASE, quartet=QUARTET):
    """Rebuild unique paths and verify fixed ordered shortlists for every view.

    Can also process fresh graphs generated after a successful development gate.
    Pass the new graph generator's question templates in that case.
    """
    token_templates = [style.split() for style in templates]
    views = {name: ([], []) for name in ("paraphrase", "q0", "q1", "q2", "q3")}
    views["original"] = original
    # Alternate surface order within each stratum independently.
    strata_seen = Counter()
    for ex, gold in zip(*original, strict=True):
        roles = roles_for(ex, token_templates)
        anchor, first, second = roles
        require(first != second and ex.gold is None, f"Invalid roles/input: {ex.doc_id}")
        forward = paths(ex.nodes, anchor, first, second)
        require(len(forward) == 1 and forward[0][0] == gold, f"Original path disagrees: {ex.doc_id}")
        index = strata_seen[ex.stratum] % len(paraphrases)
        strata_seen[ex.stratum] += 1
        para = make_question(ex, roles, paraphrases[index], evaluator)
        require(para.question != ex.question, f"Duplicate rewrite: {ex.doc_id}")
        views["paraphrase"][0].append(para)
        views["paraphrase"][1].append(gold)

        reverse = paths(ex.nodes, anchor, second, first)
        if len(reverse) != 1 or reverse[0][0] == gold:
            continue
        reverse_gold = reverse[0][0]
        require({gold, reverse_gold} == set(ex.candidates), f"Quartet misses legal targets: {ex.doc_id}")
        swap = (anchor, second, first)
        bag = None
        for number, (style, query_roles, target) in enumerate(
            ( (style, r, y) for style in quartet for r, y in ((roles, gold), (swap, reverse_gold)) )
        ):
            new = make_question(ex, query_roles, style, evaluator)
            word_counts = Counter(new.question)
            require(bag is None or word_counts == bag, f"Question token multiset drift: {ex.doc_id}")
            bag = word_counts
            examples, labels = views[f"q{number}"]
            examples.append(new)
            labels.append(target)
    return views


def stats(view):
    examples, labels = view
    return {"graphs": len(examples), "strata": dict(Counter(ex.stratum for ex in examples)),
            "candidate_hits": sum(label in ex.candidates for ex, label in zip(*view, strict=True)),
            "graph_ids_sha256": canonical_hash([ex.doc_id for ex in examples]),
            "ordered_candidates_sha256": canonical_hash([ex.candidates for ex in examples]),
            "labels_sha256": canonical_hash(labels),
            "questions_sha256": canonical_hash([ex.question for ex in examples])}


def build_views(b13, b16, trainer, evaluator):
    b13, b16 = Path(b13), Path(b16)
    source_path = b13 / "inputs/source/generate.py"
    cohort_path = b13 / "code/generate_cohort.py"
    data = b16 / DATA_RELATIVE
    archives = [source_path, cohort_path, *(data / name for name in
                ("train.jsonl", "dev.jsonl", "train-clause-views.jsonl"))]
    provenance = [checked(path) for path in archives]
    families = runpy.run_path(str(source_path))["QUESTION_FAMILIES"]
    historical = set().union(*families.values(), runpy.run_path(str(cohort_path))["TEMPLATES"])
    train = load_view(rows(data / "train.jsonl"), trainer, evaluator)
    original = load_view(rows(data / "dev.jsonl"), trainer, evaluator)
    require(Counter(ex.stratum for ex in train[0]) == dict.fromkeys(STRATA, 4000), "Unexpected training strata")
    require(Counter(ex.stratum for ex in original[0]) == dict.fromkeys(STRATA, 600), "Unexpected dev strata")
    train_digests = {graph_hash(ex) for ex in train[0]}
    dev_digests = {graph_hash(ex) for ex in original[0]}
    require(len(train_digests) == len(train[0]) and len(dev_digests) == len(original[0])
            and train_digests.isdisjoint(dev_digests), "Duplicate canonical graphs across or within splits")
    require({ex.doc_id for ex in train[0]}.isdisjoint(ex.doc_id for ex in original[0]), "Train/dev doc ID overlap")
    train_families = [s.split() for s in families["train"]]
    training_roles = {}
    for ex, gold in zip(*train, strict=True):
        roles = roles_for(ex, train_families)
        valid = paths(ex.nodes, *roles)
        require(len(valid) == 1 and valid[0][0] == gold, f"Training label disagrees: {ex.doc_id}")
        training_roles[ex.doc_id] = roles
    augmented_count = 0
    for row in rows(data / "train-clause-views.jsonl"):
        anchor, first, second = training_roles[row["doc_id"]]
        for key, a, b in (("original", first, second), ("swapped", second, first)):
            slots = {anchor: "{anchor}", a: "{first}", b: "{second}"}
            for pair in row[key] or []:
                for wording in pair:
                    historical.add(" ".join(slots.get(token, token) for token in wording["question"]))
        augmented_count += 1
    require(augmented_count == 1840, "Unexpected training clause-view graph count")
    novel = {*PARAPHRASE, *QUARTET, *FRESH_TEMPLATES}
    require(len(novel) == 8 and novel.isdisjoint(historical), "New template appeared in an archive")
    views = {"train": train, **make_eval_views(original, families["dev"], evaluator)}
    require(all(len(views[k][0]) == 600 for k in ("q0", "q1", "q2", "q3"))
            and {ex.stratum for ex in views["q0"][0]} == {"relation_order"}, "Unexpected quartet cohort")
    vocabulary = trainer.build_vocabulary(train[0])
    all_tokens = {word for examples, _ in views.values() for ex in examples for word in ex.question}
    unknown = all_tokens.difference(vocabulary)
    require(not unknown, f"Unseen question words: {unknown}")
    hashes = {}
    for kind, expected in evaluator.CHECKPOINTS.items():
        for seed, digest in expected.items():
            checkpoint = evaluator.checkpoint_dir(b13 / "inputs/server-run", kind, seed) / "best.pt"
            require(evaluator.file_sha(checkpoint) == digest, f"Frozen weights changed: {checkpoint}")
            require(trainer.read_checkpoint(checkpoint)["vocabulary"] == vocabulary, "Vocabulary changed")
            hashes[f"{kind}-{seed}"] = digest
    return views, {"sources": provenance, "checkpoints_sha256": hashes,
                   "views": {key: stats(view) for key, view in views.items()},
                   "vocabulary_tokens": len(vocabulary), "oov_question_tokens": len(unknown),
                   "canonical_graph_hash": "SHA256(compact UTF8 JSON, sorted directed triples)",
                   "train_graph_set_sha256": canonical_hash(sorted(train_digests)),
                   "dev_graph_set_sha256": canonical_hash(sorted(dev_digests)),
                   "overlap_graphs": len(train_digests & dev_digests), "overlap_doc_ids": 0,
                   "augmented_training_graphs_checked": augmented_count, "new_template_overlap": 0,
                   "historical_templates_checked": len(historical), "paraphrase_templates": PARAPHRASE,
                   "quartet_templates": QUARTET, "quartet_order": ["AB-forward", "BA-forward", "AB-reversed-clauses", "BA-reversed-clauses"],
                   "candidate_rule": "B13 direction Top-2", "gold_format": "zero-based fact index"}


def build_fresh_views(b13, trainer, evaluator):
    """Deterministic, development-only draw outside historical graph populations."""
    b13 = Path(b13)
    generator = b13 / "inputs/source/generate.py"
    provenance = checked(generator)
    old = runpy.run_path(str(generator))
    b13_features = b13 / "cohort/features.jsonl"
    features_sha = evaluator.file_sha(b13_features)
    require(features_sha == "b49a3323c58e702d2562588e3ca823f604b90803b071614260611e84f399f44c",
            "Unexpected historical feature SHA-256")
    def people(nodes):
        return frozenset(name for head, _, tail in nodes for name in (head, tail))

    excluded = {people(row["nodes"]) for row in rows(b13_features)}
    require(len(excluded) == 3000, "Historical B13 graph identities are not unique")
    lower = max(start + count for _, start, count in old["SPLITS"].values())
    space = old["PERSON_SET_SPACE"]
    require(lower == 16_800 and space == 74_974_368
            and math.gcd(old["RANK_MULTIPLIER"], space) == 1, "Invalid historical graph domain")
    seed = 20261124
    rng = random.Random(seed)
    per_stratum = 200
    strata = list(old["STRATA"])
    schedule = [s for s in strata for _ in range(per_stratum)]
    rng.shuffle(schedule)
    # Counterbalance first-hop fact position and wording independently within each stratum.
    allocations = {}
    for stratum_i, stratum in enumerate(strata):
        positions = [(per_stratum * stratum_i + i) % old["FACT_COUNT"] for i in range(per_stratum)]
        styles = [i % len(FRESH_TEMPLATES) for i in range(per_stratum)]
        rng.shuffle(positions)
        rng.shuffle(styles)
        allocations[stratum] = list(zip(positions, styles, strict=True))
    selected, raw_rows, indices, rejected = set(), [], [], 0
    style_counts = Counter()
    for stratum in schedule:
        while True:
            index = rng.randrange(lower, space)
            persons = frozenset(old["ranked_people"](index))
            if persons not in excluded and persons not in selected:
                break
            rejected += 1
        selected.add(persons)
        indices.append(index)
        gold_pos, style = allocations[stratum].pop()
        row = old["document"](index, stratum, gold_pos, FRESH_TEMPLATES[style], rng)
        require(people(row["nodes"]) == persons and row["gold"] == gold_pos, "Invalid source graph")
        raw_rows.append(row)
        style_counts[(stratum, style)] += 1
    require(len(selected) == 600 and selected.isdisjoint(excluded)
            and set(style_counts.values()) == {50}, "Fresh-cohort balance/overlap failed")
    original = load_view(raw_rows, trainer, evaluator)
    views = make_eval_views(original, FRESH_TEMPLATES, evaluator)
    require(Counter(original[1]) == dict.fromkeys(range(12), 50), "Gold positions unbalanced")
    require(len(views["q0"][0]) == 200 and {ex.stratum for ex in views["q0"][0]} == {"relation_order"},
            "Unexpected quartet size/strata")
    checkpoint = evaluator.checkpoint_dir(b13 / "inputs/server-run", "serial", 1701) / "best.pt"
    require(evaluator.file_sha(checkpoint) == evaluator.CHECKPOINTS["serial"][1701], "Frozen weights changed")
    vocabulary = trainer.read_checkpoint(checkpoint)["vocabulary"]
    unknown = {w for examples, _ in views.values() for ex in examples for w in ex.question
               if w not in vocabulary}
    require(not unknown, f"Fresh cohort contains out-of-vocabulary words: {unknown}")
    digests = {graph_hash(ex) for ex in original[0]}
    require(len(digests) == len(raw_rows), "Repeated fresh graph")
    return views, {"scope": "fresh development", "seed": seed, "per_stratum": per_stratum,
                   "generator": provenance, "original_templates": FRESH_TEMPLATES,
                   "paraphrase_templates": PARAPHRASE, "quartet_templates": QUARTET,
                   "historical_unique_graphs_excluded_by_index": lower,
                   "historical_b13_feature_sha256": features_sha, "historical_b13_person_sets": len(excluded),
                   "person_indices": indices, "person_indices_sha256": canonical_hash(indices),
                   "redraws_for_duplicates_or_exclusions": rejected, "b13_person_set_overlap": 0,
                   "graph_set_sha256": canonical_hash(sorted(digests)), "oov_question_tokens": len(unknown),
                   "original_template_counts_per_stratum": {
                       stratum: [style_counts[(stratum, i)] for i in range(len(FRESH_TEMPLATES))]
                       for stratum in strata},
                   "views": {key: stats(view) for key, view in views.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b13", type=Path)
    parser.add_argument("b16", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.b13 / "inputs/b12-code"))
    import evaluate
    train_module = evaluate.frozen_trainer(args.b13 / "inputs/server-run")
    _, manifest = build_views(args.b13, args.b16, train_module, evaluate)
    _, fresh_manifest = build_fresh_views(args.b13, train_module, evaluate)
    print(json.dumps({"archived": manifest, "fresh": fresh_manifest}, ensure_ascii=False, indent=2))
