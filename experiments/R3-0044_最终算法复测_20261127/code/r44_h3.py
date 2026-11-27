"""R44: second independent H3 test for the frozen mix2r_aug candidate.

The cohort is drawn from the reserved index range [45,000,000, 50,000,000) with
all known historical person sets excluded, including the R21, R24, R42 and R43 queues.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import runpy
import secrets
import subprocess
import sys

import torch

SEEDS = (1701, 2701, 3701)
KINDS = ("relation_order", "role_reversal", "competing_chain")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8") as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def line(f, x):
    f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as f:
        for s in f:
            yield json.loads(s)


def now():
    return datetime.now(timezone.utc).isoformat()


def persons(r):
    return frozenset(w for h, _, t in r["nodes"] for w in (h, t))


def load_local(a):
    sys.path.insert(0, str(a.prior / "code"))
    import model
    import data_views
    return model, data_views


def freeze(a):
    entry = {"utc": now(), "out": str(a.out), "module": "r44_module",
             "checkpoints": str(a.checkpoints), "sha256": {}}
    artifacts = [a.checkpoints / f"mix2raug-{s}.json" for s in SEEDS]
    artifacts += [a.prior / "code/model.py", a.prior / "code/evaluate.py",
                  a.b13 / "冻结清单.json", a.b13 / "inputs/source/generate.py", a.templates]
    artifacts += sorted(Path(__file__).parent.glob("r44_*.py"))
    entry["sha256"] = {str(p): sha(p) for p in artifacts}
    write(a.out / "freeze.json", entry)
    write(a.out / "entropy.json", {"hex": secrets.token_hex(32), "freeze_sha256": sha(a.out / "freeze.json")})


def confirm(a):
    entry = read(a.out / "freeze.json")
    if not all(sha(Path(p)) == h for p, h in entry["sha256"].items()):
        raise ValueError("Freeze mismatch")


def draw(a):
    confirm(a)
    model, dv = load_local(a)
    b = model.load_bundle(a.b13, 1701)
    gen = runpy.run_path(str(a.b13 / "inputs/source/generate.py"))
    styles = read(a.templates)
    all_styles = {s for lst in styles.values() for s in lst}
    historical = set().union(*gen["QUESTION_FAMILIES"].values(), dv.FRESH_TEMPLATES, dv.PARAPHRASE, dv.QUARTET,
                             runpy.run_path(str(a.b13 / "code/generate_cohort.py"))["TEMPLATES"])
    for style in all_styles:
        if style in historical or any(t not in b.vocab for t in style.split() if not t.startswith("{")):
            raise ValueError("Invalid formal template or vocabulary")
    blocked = {persons(r) for r in rows(a.b13 / "cohort/features.jsonl")}
    blocked.update(persons(r) for r in rows(a.prior / "runs/本地开发-01/fresh_dev-输入.jsonl")
                   if r["variant"] == "original")
    blocked.update(persons(r) for r in rows(a.r16_cohort))
    blocked.update(persons(r) for r in rows(a.r09_features))
    blocked.update(persons(r) for r in rows(a.r21_features))
    blocked.update(persons(r) for r in rows(a.r24_features))
    blocked.update(persons(r) for r in rows(a.r42_features))
    blocked.update(persons(r) for r in rows(a.r43_features))
    rng = random.Random(int(read(a.out / "entropy.json")["hex"], 16))
    n = 1000
    raw = []
    for ki, k in enumerate(KINDS):
        pos = [(ki * n + i) % 12 for i in range(n)]
        fam = [i % len(styles["original"]) for i in range(n)]
        rng.shuffle(pos)
        rng.shuffle(fam)
        for g, j in zip(pos, fam, strict=True):
            while True:
                idx = rng.randrange(45_000_000, 50_000_000)
                people = frozenset(gen["ranked_people"](idx))
                if people not in blocked:
                    break
            blocked.add(people)
            doc = gen["document"](idx, k, g, styles["original"][j], rng)
            if persons(doc) != people:
                raise ValueError("Source index mismatch")
            raw.append(doc)
    rng.shuffle(raw)
    root_view = dv.load_view(raw, b.trainer, b.evaluator)
    views = dv.make_eval_views(root_view, styles["original"], b.evaluator,
                               paraphrases=styles["para"], quartet=styles["quartet"])
    with (a.out / "features.jsonl").open("x", encoding="utf8") as ff, \
         (a.out / "labels.jsonl").open("x", encoding="utf8") as lf:
        for var, (docs, gold) in views.items():
            for d, g in zip(docs, gold, strict=True):
                line(ff, dict(variant=var, doc_id=d.doc_id, stratum=d.stratum,
                              question=d.question, nodes=d.nodes))
                line(lf, dict(variant=var, doc_id=d.doc_id, gold=g))
    write(a.out / "cohort.json", dict(features_sha256=sha(a.out / "features.jsonl"),
                                      labels_sha256=sha(a.out / "labels.jsonl"),
                                      graphs=len(raw), per_stratum=n, utc=now(),
                                      quartet_graphs=len(views["q0"][0]),
                                      index_range=[45_000_000, 50_000_000],
                                      excluded_populations=["b13-cohort", "fresh-dev", "r16-dev-cohort",
                                                            "r09-formal-cohort", "r21-formal-cohort",
                                                            "r24-formal-cohort", "r42-formal-cohort",
                                                            "r43-formal-cohort"]))


def predict(a):
    confirm(a)
    model, _ = load_local(a)
    sys.path.insert(0, str(Path(__file__).parent))
    import r44_module
    if sha(a.out / "features.jsonl") != read(a.out / "cohort.json")["features_sha256"]:
        raise ValueError("Features changed")
    manifest = {}
    for seed in SEEDS:
        b = model.load_bundle(a.b13, seed)
        st = read(a.checkpoints / f"mix2raug-{seed}.json")
        learned_w = torch.tensor(read(a.prior / "runs/本地开发-01" / f"w-{seed}.json")["w"],
                                 dtype=torch.float32)
        path = a.out / f"prediction-{seed}.jsonl"
        with path.open("x", encoding="utf8") as f:
            from itertools import batched
            for batch in batched(rows(a.out / "features.jsonl"), 128):
                docs = []
                for r in batch:
                    q = tuple(r["question"])
                    nodes = tuple(map(tuple, r["nodes"]))
                    docs.append(b.trainer.Example(r["doc_id"], r["stratum"], q, nodes,
                                                  b.trainer.lexical_candidates(q, nodes), None))
                docs = b.evaluator.shortlist(docs, "direction")
                feat = model.extract(b, docs)
                with torch.no_grad():
                    candidate = r44_module.predict(b, docs, st)
                scores = {"mix2raug": candidate, "pool": feat.pooled, "direction": feat.base,
                          "old": feat.old, "uniform": model.score(feat, torch.zeros(64)),
                          "learned": model.score(feat, learned_w)}
                for j, (d, row) in enumerate(zip(docs, batch)):
                    choices = {k: str(d.candidates[int(v[j].argmax())]) for k, v in scores.items()}
                    choices["first"] = str(d.candidates[0])
                    line(f, dict(graph_id=d.doc_id, seed=seed, kind=d.stratum, variant=row["variant"],
                                 candidates=list(map(str, d.candidates)), gate=bool(feat.active[j]),
                                 predictions=choices,
                                 logits={k: v[j].tolist() for k, v in scores.items()}))
        manifest[path.name] = sha(path)
    write(a.out / "prediction-seal.json",
          {"sha256": manifest, "utc": now(), "cohort": sha(a.out / "cohort.json")})


def score(a):
    confirm(a)
    cohort = read(a.out / "cohort.json")
    seal = read(a.out / "prediction-seal.json")
    if sha(a.out / "cohort.json") != seal["cohort"] or \
       not all(sha(a.out / k) == h for k, h in seal["sha256"].items()):
        raise ValueError("Prediction seal mismatch")
    write(a.out / "opening.json", {"utc": now(), "seal": sha(a.out / "prediction-seal.json")})
    if sha(a.out / "labels.jsonl") != cohort["labels_sha256"]:
        raise ValueError("Labels changed")
    gold = {(r["doc_id"], r["variant"]): r["gold"] for r in rows(a.out / "labels.jsonl")}
    with (a.out / "scored-predictions.jsonl").open("x", encoding="utf8") as f:
        for s in SEEDS:
            for r in rows(a.out / f"prediction-{s}.jsonl"):
                r["gold"] = str(gold[(r["graph_id"], r["variant"])])
                line(f, r)
    subprocess.run([sys.executable, "-B", str(Path(__file__).parent / "r44_evaluate.py"),
                    str(a.out / "scored-predictions.jsonl"),
                    str(a.out / "metrics.json")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "draw", "predict", "score"])
    for key in ("b13", "b16", "prior", "checkpoints", "templates",
                "r16_cohort", "r09_features", "r21_features", "r24_features",
                "r42_features", "r43_features", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    a = parser.parse_args()
    if a.stage == "all":
        a.out.mkdir(parents=True, exist_ok=False)
        freeze(a)
        for stage in ("draw", "predict", "score"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), stage,
                            "--b13", str(a.b13), "--b16", str(a.b16), "--prior", str(a.prior),
                            "--checkpoints", str(a.checkpoints), "--templates", str(a.templates),
                            "--r16_cohort", str(a.r16_cohort), "--r09_features", str(a.r09_features),
                            "--r21_features", str(a.r21_features), "--r24_features", str(a.r24_features),
                            "--r42_features", str(a.r42_features),
                            "--r43_features", str(a.r43_features),
                            "--out", str(a.out)], check=True)
    else:
        {"draw": draw, "predict": predict, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
