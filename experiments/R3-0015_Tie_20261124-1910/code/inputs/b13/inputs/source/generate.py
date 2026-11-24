"""Generate a controlled Chinese directed-fact benchmark for R2 first-hop evidence.

Train (seed 6111901), dev (6111902), and test (6111903) contain 12,000,
1,800 and 3,000 documents, respectively. Each document gives 12 different
directed facts and exactly one question. The question describes an ordered
two-edge path and asks for its first record. Exactly one outgoing edge from
the stated anchor matches the first relation, and its destination has exactly
one outgoing edge matching the second relation. The path has three distinct
people. There are no route or role markers attached to the facts themselves.

The three equally frequent strata emphasize reversing the first edge's roles,
following the relations in the opposite order, or following the right order
from a different person. Remaining facts are sampled from the same vocabulary
without changing an answer. Fact order is shuffled, with gold positions
balanced exactly per split and to within one count per stratum.

The splits share entity and relation vocabulary but have disjoint question
families. Each split balances which relation is mentioned first on the surface.
The dev and test families recombine train vocabulary, including the temporal
words indicating order. These are controlled paraphrases and directed graphs,
not natural QA or a claim about general Chinese understanding.

One global injective rank assignment draws a distinct six-person set for each
graph across splits. Every selected person occurs in the facts, so canonical
directed graphs cannot overlap even after shuffling fact order. Document IDs
are UUID5 hashes of canonical facts. JSONL bytes are UTF-8 with LF newlines;
an output path is created exclusively and never overwritten.
"""

import argparse
import json
import math
import random
import uuid
from pathlib import Path


SPLITS = {
    "train": (6111901, 0, 12_000),
    "dev": (6111902, 12_000, 1_800),
    "test": (6111903, 13_800, 3_000),
}
STRATA = ("role_reversal", "relation_order", "competing_chain")
PEOPLE = tuple(
    surname + given
    for surname in "陈林周许沈罗唐顾"
    for given in "宁然舟言晴南秋平"
)
RELATIONS = ("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随")
PERSON_SET_SIZE = 6
FACT_COUNT = 12
PERSON_SET_SPACE = math.comb(len(PEOPLE), PERSON_SET_SIZE)
RANK_MULTIPLIER = 50_271_647
RANK_OFFSET = 40_630_237

# In each split the first half mentions relation one before relation two.
QUESTION_FAMILIES = {
    "train": (
        "从 {anchor} 出发 ， 先 找 其 {first} 的 人 ， 再 找 此人 {second} 的 人 。 哪条 记录 是 第一步 ？",
        "以 {anchor} 为 起点 ， 首先 确定 其 {first} 的 人 ， 之后 追踪 此人 {second} 的 人 。 第一步 对应 哪条 记录 ？",
        "请 给出 第一步 的 记录 ： {anchor} 先 {first} 一人 ， 此人 随后 {second} 一人 。",
        "第二 段 沿 {second} 关系 追踪 。 第一步 从 {anchor} 出发 ， 沿 {first} 关系 。 对应 哪条 记录 ？",
        "结束 于 {second} 关系 。 以 {anchor} 为 起点 ， 此前 先 沿 {first} 关系 。 请 给出 第一步 的 记录 。",
        "要 找到 最后 一人 ， 最后 一段 按 {second} 关系 。 从 {anchor} 开始 ， 开头 一段 按 {first} 关系 。 应 先 使用 哪条 记录 ？",
    ),
    "dev": (
        "哪条 记录 对应 第一步 ？ 从 {anchor} 开始 ， 首先 沿 {first} 关系 找 一人 ， 随后 沿 {second} 关系 找 一人 。",
        "以 {anchor} 为 起点 ， 开头 一段 按 {first} 关系 找 一人 ， 之后 按 {second} 关系 追踪 。 应 先 使用 哪条 记录 ？",
        "请 给出 开头 一段 的 记录 。 第二 段 按 {second} 关系 ， 第一步 以 {anchor} 为 起点 沿 {first} 关系 。",
        "最后 一段 沿 {second} 关系 。 此前 ， 从 {anchor} 出发 按 {first} 关系 找 一人 。 哪条 记录 是 第一步 ？",
    ),
    "test": (
        "要 找到 最后 一人 ， 以 {anchor} 为 起点 ， 首先 按 {first} 关系 找 一人 ， 再 按 {second} 关系 追踪 。 哪条 记录 对应 开头 一段 ？",
        "开头 一段 对应 哪条 记录 ？ {anchor} 先 {first} 一人 ， 此人 之后 {second} 一人 。",
        "最后 一段 是 {second} 关系 ， 开头 一段 从 {anchor} 出发 ， 追踪 {first} 关系 。 请 给出 第一步 的 记录 。",
        "第二 段 按 {second} 关系 找 人 。 此前 应 从 {anchor} 开始 ， 沿 {first} 关系 找 人 。 应 先 使用 哪条 记录 ？",
    ),
}


def ranked_people(index):
    rank = (RANK_MULTIPLIER * index + RANK_OFFSET) % PERSON_SET_SPACE
    group = []
    start = 0
    for remaining in range(PERSON_SET_SIZE - 1, -1, -1):
        for position in range(start, len(PEOPLE) - remaining):
            block = math.comb(len(PEOPLE) - position - 1, remaining)
            if rank < block:
                group.append(PEOPLE[position])
                start = position + 1
                break
            rank -= block
    return group


def family_schedule(count, family_count, stratum_index):
    midpoint = family_count // 2
    return [
        start + (index + stratum_index) % midpoint
        for start in (0, midpoint)
        for index in range(count // 2)
    ]


def document(index, stratum, gold_position, template, rng):
    people = ranked_people(index)
    rng.shuffle(people)
    anchor, middle, end, other, fifth, sixth = people
    first, second, unrelated = rng.sample(RELATIONS, 3)
    gold = (anchor, first, middle)
    continuation = (middle, second, end)
    distractors = {
        "role_reversal": (
            (anchor, second, other),
            (middle, first, anchor),
            (sixth, first, anchor),
            (anchor, unrelated, sixth),
            (end, second, middle),
            (other, unrelated, fifth),
        ),
        "relation_order": (
            (anchor, second, other),
            (other, first, fifth),
            (anchor, unrelated, sixth),
            (sixth, second, anchor),
            (end, first, middle),
        ),
        "competing_chain": (
            (anchor, unrelated, other),
            (other, first, fifth),
            (fifth, second, sixth),
            (other, second, middle),
            (end, first, sixth),
        ),
    }[stratum]
    facts = (gold, continuation, *distractors)

    # Preserve the constructed path and the stratum's directed contrast.
    reserved = {(anchor, first), (anchor, second), (middle, second), (other, first)}
    candidates = [
        (subject, relation, target)
        for subject in people
        for relation in RELATIONS
        if (subject, relation) not in reserved
        for target in people
        if target != subject
        and (subject, relation, target) not in facts
        and (subject, relation, target) != (middle, first, anchor)
    ]
    fillers = rng.sample(candidates, FACT_COUNT - len(facts))
    non_gold = [continuation, *distractors, *fillers]
    rng.shuffle(non_gold)
    non_gold.insert(gold_position, gold)
    node_id_material = json.dumps(sorted(non_gold), ensure_ascii=False, separators=(",", ":"))
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, f"spiderweb/r2/directed-facts/v1/{node_id_material}")
    return {
        "doc_id": str(doc_id),
        "stratum": stratum,
        "question": template.format(anchor=anchor, first=first, second=second).split(),
        "nodes": non_gold,
        "gold": gold_position,
    }


def rows(split):
    seed, start_index, size = SPLITS[split]
    rng = random.Random(seed)
    templates = QUESTION_FAMILIES[split]
    per_stratum = size // len(STRATA)
    schedules = {}
    for stratum_index, stratum in enumerate(STRATA):
        positions = [(stratum_index * per_stratum + index) % FACT_COUNT for index in range(per_stratum)]
        families = family_schedule(per_stratum, len(templates), stratum_index)
        rng.shuffle(positions)
        rng.shuffle(families)
        schedules[stratum] = list(zip(positions, families))

    strata = [stratum for stratum in STRATA for _ in range(per_stratum)]
    rng.shuffle(strata)
    for index, stratum in enumerate(strata, start=start_index):
        gold_position, family_index = schedules[stratum].pop()
        yield document(index, stratum, gold_position, templates[family_index], rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows(args.split):
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
