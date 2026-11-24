"""Prospectively generate and seal the B13 cohort. No predictor is imported."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import runpy
import secrets
import tempfile

SOURCE = "inputs/source/generate.py"
GENERATOR = "code/generate_cohort.py"
PROTOCOL = "实验协议.json"
FREEZE = "冻结清单.json"
SOURCE_SHA = "c6d8eaa5081868946bc24969ddeed4450d6685ae9bd782a3cdb70dd43717b46f"
OLD_END = 16800
SIZE_PER_STRATUM = 1000

# Four training-only clause arrangements derived by B8 training_pairs.derive_styles
# from the six B1 training families. Hashes refer to raw UTF-8 template text.
PRIOR_AUGMENTATIONS = {
    "3029e720ca1d0e86f39896463ed929290440419caaef41a1263d1a95c9761fa7",
    "82f6fda5f7d295997975c0c704a0d5076727ef1877e9cefe0e8f10437b41b7c3",
    "8fc798a29b32ab59a49f14f70b76b12d4546d8f6e21297d1a31f70f553a67a9b",
    "04450a5225496855f5883106d7ad7851a4a12439c3354d6cdacd65c8dbdaeaf2",
}
TEMPLATES = (
    "请 确定 第一步 对应 哪条 记录 。 从 {anchor} 出发 ， 首先 找 其 {first} 的 人 ， 随后 找 此人 {second} 的 人 。",
    "哪条 记录 应 先 使用 ？ 以 {anchor} 为 起点 ， 开头 一段 沿 {first} 关系 ， 第二 段 沿 {second} 关系 。",
    "从 {anchor} 开始 ， 先 沿 {first} 关系 找 一人 ， 之后 沿 {second} 关系 找 最后 一人 。 请 给出 开头 一段 的 记录 。",
    "要 找到 最后 一人 ， {anchor} 首先 {first} 一人 ， 此人 随后 {second} 一人 。 哪条 记录 是 第一步 ？",
    "请 确定 开头 一段 的 记录 。 最后 一段 按 {second} 关系 ， 第一步 从 {anchor} 开始 按 {first} 关系 。",
    "第二 段 沿 {second} 关系 找 一人 。 此前 应 从 {anchor} 出发 ， 沿 {first} 关系 找 一人 。 哪条 记录 对应 第一步 ？",
    "最后 一段 追踪 {second} 关系 。 从 {anchor} 开始 ， 首先 沿 {first} 关系 找 一人 。 应 先 使用 哪条 记录 ？",
    "要 找到 最后 一人 ， 结束 于 {second} 关系 。 开头 一段 从 {anchor} 出发 ， 按 {first} 关系 追踪 。 请 给出 第一步 的 记录 。",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prototype(root):
    path = root / SOURCE
    require(sha(path.read_bytes()) == SOURCE_SHA, "historical generator SHA mismatch")
    old = runpy.run_path(str(path))
    expected = [(0, 12000), (12000, 1800), (13800, 3000)]
    require([(old["SPLITS"][split][1:]) for split in ("train","dev","test")] == expected, "old index domains")
    require(math.gcd(old["RANK_MULTIPLIER"], old["PERSON_SET_SPACE"]) == 1, "person-set mapping is not injective")
    require(old["PERSON_SET_SPACE"] == 74974368 and old["FACT_COUNT"] == 12, "graph population")
    families = old["QUESTION_FAMILIES"]
    all_old = {template for values in families.values() for template in values}
    vocabulary = {word for template in families["train"] for word in template.split()}
    require(len(set(TEMPLATES)) == 8 and not set(TEMPLATES).intersection(all_old), "duplicate or prior template")
    for i, template in enumerate(TEMPLATES):
        tokens = template.split()
        require(set(tokens) <= vocabulary and all(tokens.count(k) == 1 for k in ("{anchor}", "{first}", "{second}")),
                f"template {i}: invalid vocabulary/slots")
        require((tokens.index("{first}") < tokens.index("{second}")) == (i < 4), "surface-order balance")
        require(sha(template.encode()) not in PRIOR_AUGMENTATIONS, "existing B8 question template")
    return old


def bootstrap_entropy(folder, freeze_sha):
    token = folder / "entropy.json"
    # Exclusive create BEFORE asking the OS for any random bytes. An unreadable
    # partial entropy record is a stop, never permission to redraw.
    try:
        stream = token.open("xb")
    except FileExistsError:
        record = json.loads(token.read_bytes())
    else:
        with stream:
            record = {"freeze_sha256":freeze_sha, "seed_hex":secrets.token_hex(32),
                      "created_utc":utc(), "python":platform.python_version()}
            stream.write(encoded(record))
            stream.flush()
            os.fsync(stream.fileno())
    require(record["freeze_sha256"] == freeze_sha, "attempt belongs to a different freeze")
    require(record["python"] == platform.python_version() and len(bytes.fromhex(record["seed_hex"])) == 32,
            "unrecoverable entropy/Python mismatch")
    return record


def publish(path, raw):
    if path.exists():
        require(path.read_bytes() == raw, f"pre-existing cohort material differs: {path.name}")
        return
    fd, staging = tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(staging,path)
        except FileExistsError:
            require(path.read_bytes() == raw, f"concurrent publication differs: {path.name}")
    finally:
        Path(staging).unlink(missing_ok=True)


def make_rows(old, entropy):
    rng = random.Random(int(entropy["seed_hex"],16))
    strata = tuple(old["STRATA"])
    size = SIZE_PER_STRATUM * len(strata)
    indices = rng.sample(range(OLD_END,old["PERSON_SET_SPACE"]),size)
    order = [s for s in strata for _ in range(SIZE_PER_STRATUM)]
    rng.shuffle(order)
    schedules = {}
    for i,s in enumerate(strata):
        positions = [(SIZE_PER_STRATUM*i+j)%12 for j in range(SIZE_PER_STRATUM)]
        families = [t for t in range(8) for _ in range(SIZE_PER_STRATUM//8)]
        rng.shuffle(positions)
        rng.shuffle(families)
        schedules[s] = list(zip(positions,families,strict=True))

    feature_lines, label_lines = [], []
    counts = {s:Counter() for s in strata}
    places = Counter()
    person_sets, ids = set(), set()
    for i,s in zip(indices,order,strict=True):
        position, family = schedules[s].pop()
        template = TEMPLATES[family]
        data = old["document"](i,s,position,template,rng)
        tokens, edges = data["question"], data["nodes"]
        roles = {p:v for p,v in zip(template.split(),tokens,strict=True) if p in ("{anchor}","{first}","{second}")}
        anchor, first, second = [roles[k] for k in ("{anchor}", "{first}", "{second}")]
        eligible = [j for j,(a,r,_) in enumerate(edges) if (a,r)==(anchor,first)]
        require(eligible == [position] and data["gold"] == position, "unique first edge")
        bridge = edges[position][2]
        onward = [t for a,r,t in edges if (a,r)==(bridge,second)]
        require(first!=second and len(onward)==1 and len({anchor,bridge,*onward})==3, "unique legal continuation")
        people = frozenset(x for a,_,t in edges for x in (a,t))
        require(len(edges)==len(set(edges))==12 and all(a!=t for a,_,t in edges), "distinct edges")
        require(len(people)==6 and people==frozenset(old["ranked_people"](i)) and people not in person_sets,
                "invalid/repeated six-person set")
        require(data["doc_id"] not in ids, "duplicate canonical ID")
        person_sets.add(people)
        ids.add(data["doc_id"])
        counts[s][family] += 1
        places[position] += 1
        feature_lines.append(encoded({key:data[key] for key in ("doc_id","stratum","question","nodes")}))
        label_lines.append(encoded({key:data[key] for key in ("doc_id","stratum","gold")}))
    require(all(c==Counter(dict.fromkeys(range(8),125)) for c in counts.values()), "unbalanced templates")
    require(places==Counter(dict.fromkeys(range(12),250)) and len(ids)==size==3000, "gold-position balance/size")
    metadata = {
        "documents":size, "strata_counts":{s:sum(counts[s].values()) for s in strata},
        "template_counts_by_stratum":counts, "gold_position_counts":places,
        "person_preimage_range":[OLD_END,old["PERSON_SET_SPACE"]], "person_indices_sha256":sha(encoded(indices)),
        "distinct_person_sets":len(person_sets), "templates_sha256":sha(encoded(TEMPLATES)),
    }
    return b"".join(feature_lines), b"".join(label_lines), metadata


def run(root, check_only):
    old = prototype(root)
    if check_only:
        print(json.dumps({"source":SOURCE_SHA,"templates":len(TEMPLATES),"templates_sha256":sha(encoded(TEMPLATES)),
                          "new_person_index_domain":[OLD_END,old["PERSON_SET_SPACE"]]}))
        return
    manifest_raw = (root/FREEZE).read_bytes()
    freeze_sha = sha(manifest_raw)
    frozen = json.loads(manifest_raw)["sha256"]
    for name in (GENERATOR,SOURCE,PROTOCOL):
        require(sha((root/name).read_bytes()) == frozen[name], f"unfrozen {name}")
    require(Path(__file__).resolve() == (root/GENERATOR).resolve(), "wrong executable path")
    protocol = json.loads((root/PROTOCOL).read_bytes())
    folder = root/"cohort"
    folder.mkdir(exist_ok=True)
    seal = folder/"seal.json"
    require(not seal.exists(), "cohort already sealed; this protocol cannot draw another")
    if not (folder/"entropy.json").exists():
        require(not list(folder.iterdir()), "uncommitted cohort artifacts exist")
    entropy = bootstrap_entropy(folder,freeze_sha)
    features, gold, evidence = make_rows(old,entropy)
    publish(folder/"features.jsonl",features)
    publish(folder/"gold.jsonl",gold)
    evidence.update({
        "schema":"spiderweb-b13-cohort-seal-v1", "study_id": protocol.get("study_id",protocol.get("protocol_id")),
        "features_sha256":sha(features), "gold_sha256":sha(gold), "freeze_sha256":freeze_sha,
        "protocol_sha256":frozen[PROTOCOL], "source_sha256":SOURCE_SHA, "generator_sha256":frozen[GENERATOR],
        "entropy_sha256":sha((folder/"entropy.json").read_bytes()), "sealed_utc":utc(),
    })
    publish(seal,encoded(evidence))
    print(json.dumps({"status":"cohort_sealed","documents":evidence["documents"],"seal_sha256":sha(seal.read_bytes())}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-design", action="store_true",help="validate frozen public source and template design without sampling")
    args=parser.parse_args()
    run(args.root.resolve(),args.check_design)
