"""Build a deterministic, graph-disjoint R16/R17 development cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import sys


DRAW_SEED = 2026112416
LOW, HIGH = 60_000_000, 65_000_000
PER_STRATUM = 200
PARAPHRASE = (
    "从 {anchor} 开始 ， 第一步 追踪 {first} 关系 。 最后 一段 追踪 {second} 关系 。 哪条 记录 对应 开头 一段 ？",
    "最后 一段 追踪 {second} 关系 。 从 {anchor} 开始 ， 第一步 追踪 {first} 关系 。 哪条 记录 对应 开头 一段 ？",
)
UNSEEN = (
    "请 确定 对应 第一步 的 记录 。 以 {anchor} 为 起点 ， 首先 按 {first} 关系 追踪 。 最后 一段 按 {second} 关系 找 一人 。",
    "请 确定 对应 第一步 的 记录 。 最后 一段 按 {second} 关系 找 一人 。 以 {anchor} 为 起点 ， 首先 按 {first} 关系 追踪 。",
)
CLAUSE_A = "从 {anchor} 出发 ， 开头 一段 沿 {first} 关系 找 一人"
CLAUSE_B = "第二 段 按 {second} 关系 找 一人"
QUARTET = tuple(" 。 ".join((a, b, "哪条 记录 对应 第一步 ？")) for a, b in
                ((CLAUSE_A, CLAUSE_B), (CLAUSE_B, CLAUSE_A)))
DEV_ROUNDS = ("03-R03", "04-R04", "05-R05", "06-R06", "07-R07", "08-R08", "09-R13", "10-R12")


def require(test, message):
    if not test:
        raise ValueError(message)


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_value(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        yield from map(json.loads, stream)


def personnel(nodes):
    return tuple(sorted({name for head, _, tail in nodes for name in (head, tail)}))


def style_key(question, people, relations):
    # Ignore the particular bound names and even which relation occurs first.
    return " ".join("{P}" if word in people else "{R}" if word in relations else word for word in question)


def history(r3, inputs, generator, dv):
    people = set(generator["PEOPLE"]) | {"{anchor}"}
    rels = set(generator["RELATIONS"]) | {"{first}", "{second}"}
    signatures, blocked, sources = set(), set(), {}

    def add_style(question):
        signatures.add(style_key(question, people, rels))

    b13 = inputs / "b13"
    b16data = inputs / "b16" / dv.DATA_RELATIVE
    declared = [
        *generator["QUESTION_FAMILIES"].values(), dv.PARAPHRASE, dv.QUARTET, dv.FRESH_TEMPLATES,
        runpy.run_path(str(b13 / "code/generate_cohort.py"))["TEMPLATES"],
        runpy.run_path(str(r3 / "src/mech_data.py"))["NEW_TEMPLATES"],
        *json.loads((r3 / "configs/formal_templates.json").read_bytes()).values(),
    ]
    for family in declared:
        for template in family:
            add_style(template.split())
    graph_paths = [b16data / "train.jsonl", b16data / "dev.jsonl", b13 / "cohort/features.jsonl"]
    graph_paths += [r3 / "experiments" / name / "results" /
                    ("training/development-inputs.jsonl.gz" if name == "09-R13" else "development-inputs.jsonl.gz")
                    for name in DEV_ROUNDS]
    for path in graph_paths:
        count, original_sets = 0, set()
        for row in rows(path):
            count += 1
            add_style(row["question"])
            if "nodes" in row:
                original_sets.add(personnel(row["nodes"]))
        require(original_sets and count, f"Archive graph source is empty: {path}")
        blocked.update(original_sets)
        sources[str(path)] = {"sha256": hash_file(path), "rows": count, "personnel_sets": len(original_sets)}
    path = b16data / "train-clause-views.jsonl"
    count = 0
    for row in rows(path):
        count += 1
        for orientation in ("original", "swapped"):
            for pair in row[orientation] or ():
                for item in pair:
                    add_style(item["question"])
    sources[str(path)] = {"sha256": hash_file(path), "rows": count}
    # R11 reuses R07 graphs. Include its changed state language without using labels.
    path = r3 / "experiments/13-R11/results/inputs.jsonl"
    for row in rows(path):
        add_style(row["question"])
        require(personnel(row["nodes"]) in blocked, "Missing historical R11 graph")
    sources[str(path)] = {"sha256": hash_file(path)}
    new_keys = [style_key(t.split(), people, rels) for t in PARAPHRASE + UNSEEN + QUARTET]
    require(len(set(new_keys)) == len(new_keys) and signatures.isdisjoint(new_keys),
            "A proposed held-out wording is already represented")
    return blocked, sources, len(signatures)


def build(r3, inputs, out):
    sys.path.insert(0, str(inputs / "prior/code"))
    import data_views as dv
    import model

    b13 = inputs / "b13"
    gen_path = b13 / "inputs/source/generate.py"
    dv.checked(gen_path)
    generator = runpy.run_path(str(gen_path))
    require(HIGH < 70_000_000 < generator["PERSON_SET_SPACE"] and
            math.gcd(generator["RANK_MULTIPLIER"], generator["PERSON_SET_SPACE"]) == 1,
            "Generator does not provide the expected disjoint reserved personnel indices")
    blocked, history_sources, template_count = history(r3, inputs, generator, dv)
    bundle = model.load_bundle(b13, 1701)
    rng = random.Random(DRAW_SEED)
    originals = dv.FRESH_TEMPLATES
    raw, metadata, selected = [], {}, set()
    rejected = 0
    for k, kind in enumerate(generator["STRATA"]):
        positions = [(i + k * PER_STRATUM) % 12 for i in range(PER_STRATUM)]
        styles = [i % len(originals) for i in range(PER_STRATUM)]
        challenges = [i % len(UNSEEN) for i in range(PER_STRATUM)]
        for allocation in (positions, styles, challenges):
            rng.shuffle(allocation)
        for pos, style, challenge in zip(positions, styles, challenges, strict=True):
            while True:
                index = rng.randrange(LOW, HIGH)
                group = tuple(sorted(generator["ranked_people"](index)))
                if group not in selected and group not in blocked:
                    break
                rejected += 1
            selected.add(group)
            doc = generator["document"](index, kind, pos, originals[style], rng)
            require(doc["gold"] == pos and personnel(doc["nodes"]) == group, "Source graph/index mismatch")
            raw.append(doc)
            metadata[doc["doc_id"]] = (index, style, challenge)
    require(len(metadata) == len(selected) == 600 and selected.isdisjoint(blocked), "Graph identity collision")
    rng.shuffle(raw)
    base = dv.load_view(raw, bundle.trainer, bundle.evaluator)
    views = dv.make_eval_views(base, originals, bundle.evaluator, paraphrases=PARAPHRASE, quartet=QUARTET)
    unseen = []
    for doc in base[0]:
        roles = dv.roles_for(doc, [s.split() for s in originals])
        unseen.append(dv.make_question(doc, roles, UNSEEN[metadata[doc.doc_id][2]], bundle.evaluator))
    views["unseen_template"] = (unseen, base[1])
    styles_by_view = {"original": originals, "paraphrase": PARAPHRASE, "unseen_template": UNSEEN,
                      **{f"q{i}": (QUARTET[i // 2],) for i in range(4)}}
    records = []
    for view in ("original", "paraphrase", "q0", "q1", "q2", "q3", "unseen_template"):
        for doc, gold in zip(*views[view], strict=True):
            family = styles_by_view[view]
            anchor, first, second = dv.roles_for(doc, [x.split() for x in family])
            paths = dv.paths(doc.nodes, anchor, first, second)
            require(len(paths) == 1 and paths[0][0] == gold and doc.nodes[gold][1] == first, "Label/path mismatch")
            require(set(doc.question) <= set(bundle.vocab) and len(doc.question) <= bundle.trainer.MAX_SEQUENCE_LENGTH,
                    "Question would be truncated or contains unknown tokens")
            mentioned = [word for word in doc.question if word in generator["RELATIONS"]]
            require(len(mentioned) == len(set(mentioned)) == 2, "Expected exactly two distinct relation mentions")
            style = next(i for i, t in enumerate(family)
                         if tuple(t.format(anchor=anchor, first=first, second=second).split()) == doc.question)
            records.append(dict(graph_id=doc.doc_id, graph_index=metadata[doc.doc_id][0], stratum=doc.stratum,
                                view=view, question=doc.question, nodes=doc.nodes, candidates=doc.candidates,
                                gold=gold, first_relation=first, first_mentioned=mentioned[0], anchor=anchor,
                                second_relation=second, template_id=f"{view}-{style}"))
    require(len(records) == 2600 and set(d.stratum for d in views["q0"][0]) == {"relation_order"}
            and len(views["q0"][0]) == 200, "Expected 600 graphs with 200 reversible quartets")
    require(Counter(base[1]) == dict.fromkeys(range(12), 50), "Original gold positions unbalanced")
    for view in ("original", "paraphrase", "unseen_template"):
        for kind in generator["STRATA"]:
            group = [r for r in records if r["view"] == view and r["stratum"] == kind]
            require(len(group) == 200 and sum(r["first_relation"] == r["first_mentioned"] for r in group) == 100,
                    f"Unbalanced first mention for {kind}, {view}")
    out.mkdir(parents=True, exist_ok=False)
    dest = out / "cohort.jsonl.gz"
    with dest.open("xb") as raw_file, gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    manifest = dict(scope="New development cohort for R16 and R17", generator_sha256=hash_file(gen_path),
                    graph_rng_seed=DRAW_SEED, index_range_half_open=[LOW, HIGH], formal_index_lower=70_000_000,
                    model_seeds=[1701, 2701, 3701], archived_personnel_sets=len(blocked),
                    archived_template_signatures=template_count, historical_sources=history_sources,
                    archive_personnel_overlap=0, rejected_draws=rejected, graphs=600, questions=len(records),
                    quartet_graphs=200, graph_id_sha256=hash_value(sorted(metadata)),
                    personnel_set_sha256=hash_value(sorted(selected)), cohort_sha256=hash_file(dest),
                    per_stratum=PER_STRATUM, templates=dict(original=originals, paraphrase=PARAPHRASE,
                                                            unseen_template=UNSEEN, quartet=QUARTET),
                    view_counts=dict(Counter(r["view"] for r in records)),
                    candidate_hits_per_view=dict(Counter(r["view"] for r in records if r["gold"] in r["candidates"])),
                    question_oov=0, instruction_max_length=bundle.trainer.MAX_SEQUENCE_LENGTH,
                    graph_duplicate_check="six-person sets; entities and relation vocabulary intentionally shared",
                    primary_views=["original", "paraphrase", "q0", "q1", "q2", "q3"],
                    quartet_order=["AB/forward", "BA/forward", "AB/reordered", "BA/reordered"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"graphs": 600, "rows": len(records), "cohort_sha256": manifest["cohort_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r3", "inputs", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    build(args.r3.resolve(), args.inputs.resolve(), args.out.resolve())
