"""Replay a public B13 archive and check the prior H1/H2/H3 development run.

python history_audit.py --archive /data/B13 --prior /data/H1_H2_H3 --out /results/history-01
Inputs are read-only. Frozen model replay uses the original CPU forward implementation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import subprocess
import sys
import time

SEEDS = (1701, 2701, 3701)
KINDS = ("role_reversal", "relation_order", "competing_chain")
LEGACY = Path(__file__).resolve().parent / "legacy"


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(value, detail):
    if not value:
        raise ValueError(detail)


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def index(path):
    items = {}
    for row in rows(path):
        key = row["doc_id"]
        require(key not in items, f"duplicate doc ID in {path.name}")
        items[key] = row
    return items


def cpu_replay(script, args):
    # B12 imports hashlib.file_digest, added in 3.11. Its use is a SHA256 of a file.
    if not hasattr(hashlib, "file_digest"):
        def file_digest(stream, name):
            digest = hashlib.new(name)
            for block in iter(lambda: stream.read(262144), b""):
                digest.update(block)
            return digest
        hashlib.file_digest = file_digest
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


def original_counts(archive, pred, output):
    import numpy as np
    gold = index(archive / "cohort/gold.jsonl")
    features = index(archive / "cohort/features.jsonl")
    direct = index(pred / "direct.jsonl")
    streams = {(arm,s): index(pred / f"{arm}-{s}.jsonl") for arm in ("serial","pooled") for s in SEEDS}
    require(all(x.keys() == gold.keys() for x in (features,direct,*streams.values())), "B13 IDs differ")
    parse = runpy.run_path(str(archive/"inputs/b12-code/tie_rank.py"))["parse_question"]
    histogram = {k: Counter() for k in KINDS}
    strata = {k:Counter() for k in KINDS}
    positions = Counter()
    people_sets = set()
    with (output / "B13-original-by-graph.jsonl").open("x", encoding="utf-8") as raw:
        for key, row in gold.items():
            kind = row["stratum"]
            y = row["gold"]
            feat = features[key]
            require(set(feat) == {"doc_id","stratum","question","nodes"} and feat["stratum"] == kind,
                    "feature schema or stratum differs")
            nodes = feat["nodes"]
            pair = direct[key]["candidates"]
            require(len(pair) == len(set(pair)) == 2 and direct[key]["prediction"] == pair[0],
                    "invalid direct predictions")
            anchor, first_relation = parse(feat["question"], nodes)
            matching = [i for i,(h,r,t) in enumerate(nodes) if (h,r)==(anchor,first_relation)]
            c = strata[kind]
            c["graphs"] += 1
            c["gold_covered"] += y in pair
            c["gold_rank_one"] += y == pair[0]
            c["first_edge_rule_hits"] += matching == [y]
            positions[y] += 1
            people_sets.add(frozenset(v for a,_,b in nodes for v in (a,b)))
            per_seed = {}
            for s in SEEDS:
                answers = {}
                for arm in ("serial","pooled"):
                    p = streams[arm,s][key]
                    require(p["candidates"] == pair and p["stratum"] == kind and p["prediction"] in pair,
                            "method changed the shared candidate list")
                    answers[arm] = p["prediction"]
                    c[f"{arm}_{s}"] += answers[arm] == y
                per_seed[str(s)] = answers
            serial = sum(a["serial"] == y for a in per_seed.values())
            pooled = sum(a["pooled"] == y for a in per_seed.values())
            histogram[kind][serial-pooled, serial-3*(y==pair[0])] += 1
            raw.write(json.dumps(dict(doc_id=key,kind=kind,gold=y,candidates=pair,
                                      anchor_relation_matches=matching,per_seed=per_seed), ensure_ascii=False)+"\n")
    require(set(positions) == set(range(12)) and set(positions.values()) == {250}, "gold positions changed")
    scored = json.loads((archive / "assessment/score.json").read_bytes())
    for k,c in strata.items():
        ref = scored["by_stratum"][k]
        require(c["graphs"]==ref["documents"]==1000 and
                c["gold_covered"]==round(ref["candidate_coverage"]*1000) and
                c["gold_rank_one"]==round(ref["accuracy"]["lexical"]*1000), "B13 stratum mismatch")
        for s in SEEDS:
            require(all(c[f"{arm}_{s}"]==round(ref["accuracy_by_seed"][str(s)][name]*1000)
                        for arm,name in (("serial","candidate"),("pooled","pooled"))), "seed totals differ")

    keys = ("pooled", "lexical")
    all_counts = sum(strata.values(),Counter())
    effects = [sum(v[i]*n for h in histogram.values() for v,n in h.items()) / 9000 for i in range(2)]
    if hasattr(random.Random, "binomialvariate"):
        original = runpy.run_path(str(archive/"inputs/source/verify.py"))
        bounds = [v/9000 for v in original["paired_bootstrap_lower"](histogram)]
        implementation = "archive CPython conditional-binomial, seed 619482"
    else:
        rng = np.random.default_rng(619482)
        draws = np.zeros((10000,2), dtype=np.int64)
        for k in KINDS:
            items = sorted(histogram[k].items())
            vectors = np.asarray([v for v,n in items])
            prob = np.asarray([n/1000 for v,n in items])
            draws += rng.multinomial(1000,prob,size=10000) @ vectors
        bounds = (np.sort(draws,axis=0)[499]/9000).tolist()
        implementation = "numpy paired joint multinomial, seed 619482; independent finite Monte Carlo"
    comparisons = {}
    for name,point,lower in zip(keys,effects,bounds):
        ref = scored["contrasts"]["candidate_minus_"+name]
        require(abs(point-ref["effect"])<1e-13, f"effect differs for {name}")
        if implementation.startswith("archive"):
            require(abs(lower-ref["lower_95"])<1e-13, f"frozen lower differs for {name}")
        comparisons[name] = dict(effect_pp=100*point, lower95_pp=100*lower,
                                 archived_lower95_pp=100*ref["lower_95"],
                                 threshold_met=point>=0.05 and lower>0)
    return dict(per_stratum=strata,pooled_counts=all_counts,seed_order=SEEDS,document_count=len(gold),
                model_denominator=3*len(gold),distinct_people_sets=len(people_sets),raw_edge_position=positions,
                bootstrap=implementation,contrasts=comparisons)


def audit(archive, prior, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    steps = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")

    def note(event, **fields):
        with (out/"timeline.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=utc(),event=event,**fields),ensure_ascii=False)+"\n")

    def stage(name, script, args, json_output=True):
        args = list(map(str, args))
        filename = name+(".json" if json_output else ".txt")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--replay", str(script), *args]
        note("start", stage=name, source_sha256=sha(script))
        then=time.perf_counter()
        with (out/filename).open("wb") as stdout, (out/(name+".stderr.txt")).open("wb") as stderr:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, cwd=out)
        info = dict(stage=name, duration_seconds=round(time.perf_counter()-then,3),
                    returncode=proc.returncode, stdout_sha256=sha(out/filename))
        steps.append(info)
        note("end", **info)
        require(proc.returncode==0, f"{name} failed; see its stderr in {out}")
        if json_output:
            return json.loads((out/filename).read_bytes())

    note("started")
    inputs = {}
    for p in sorted(archive.rglob("*")):
        if p.is_file() and p.suffix not in {".zip",".pyc"}:
            inputs["B13/"+p.relative_to(archive).as_posix()] = sha(p)
    frozen = json.loads((archive/"冻结清单.json").read_bytes())
    require(all(inputs.get("B13/"+k)==v for k,v in frozen["sha256"].items()),"freeze SHA256 mismatch")
    for p in [Path(__file__).resolve(), *sorted(LEGACY.glob("*.py"))]:
        inputs["audit_src/"+p.name] = sha(p)
    prior_run = prior/"runs/本地开发-01"
    prior_script = prior/"code/verify_results.py"
    require(prior_run.is_dir() and prior_script.is_file(), "--prior must contain code/ and runs/本地开发-01/")
    for p in [prior_script,prior/"code/evaluate.py",prior/"实验协议.json",*prior_run.glob("*")]:
        if p.is_file(): inputs["prior/"+p.relative_to(prior).as_posix()] = sha(p)
    write(out/"input-manifest.json",inputs)
    stage("B13-recorded-seals",archive/"复核归档.py",[])
    real_preds = out/"B13-predictions"
    stage("B13-frozen-inference",archive/"code/predict_sealed.py",
          ["--features",archive/"cohort/features.jsonl","--features-sha256",sha(archive/"cohort/features.jsonl"),
           "--cohort-seal-sha256",sha(archive/"cohort/seal.json"),"--protocol-sha256",sha(archive/"实验协议.json"),
           "--freeze-sha256",sha(archive/"冻结清单.json"),"--b11",archive/"inputs/server-run",
           "--b12-code",archive/"inputs/b12-code","--out",real_preds],json_output=False)
    seals = json.loads((archive/"predictions/prediction-seal.json").read_bytes())["prediction_sha256"]
    require(len(seals)==7 and all(sha(real_preds/k)==v for k,v in seals.items()),"7 replays differ from B13")
    b13=original_counts(archive,real_preds,out)
    write(out/"B13-primary.json",b13)
    for name, script in [("B13-layers","第13项逐层复核"),("B13-reverse","第13项关系换序复核"),
                         ("B13-margins","第13项响应分差复核"),("B13-components","第13项首层分项复核"),
                         ("B13-quartets","第13项语义与分句顺序复核")]:
        stage(name, LEGACY/(script+".py"), [archive],json_output=(name != "B13-layers"))
    quartet=json.loads((out/"B13-quartets.json").read_bytes())
    require([quartet["counts"][k]["all_four_correct"] for k in ("serial","pooled")]==[1509,2089],
            "historical quartet counts differ")
    stage("prior-verifier",prior_script,[prior_run,out/"prior-development.json"],json_output=False)
    output_hashes = {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()
                     and p.name!="timeline.jsonl"}
    import torch
    import numpy as np
    write(out/"receipt.json",dict(status="completed",finished_utc=utc(),elapsed_s=round(time.perf_counter()-start,3),
          environment=dict(platform=platform.platform(),python=sys.version,torch=str(torch.__version__),
                           numpy=np.__version__,historical_device="cpu"),steps=steps,outputs_sha256=output_hashes))
    note("finished",receipt_sha256=sha(out/"receipt.json"))


if __name__=="__main__":
    if sys.argv[1:2] == ["--replay"]:
        cpu_replay(Path(sys.argv[2]),sys.argv[3:])
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--archive",type=Path,required=True)
        parser.add_argument("--prior",type=Path,required=True)
        parser.add_argument("--out",type=Path,required=True)
        args=parser.parse_args()
        audit(args.archive.resolve(),args.prior.resolve(),args.out.resolve())

