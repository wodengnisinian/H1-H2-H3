"""Hash transferred archives and check graph, question, label and position associations."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,platform,time

def readlines(path):
    with path.open(encoding="utf-8-sig") as f:
        yield from (json.loads(l) for l in f if l.strip())

def audit(inputs, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    source=json.loads(manifest.read_bytes())["files"]
    inventory={}
    for name, expected in source.items():
        f=inputs.parent/name
        with f.open("rb") as stream:
            h=hashlib.file_digest(stream,"sha256").hexdigest()
        if h!=expected["sha256"] or f.stat().st_size!=expected["bytes"]:
            raise ValueError(name)
        inventory[name]=h
    sets={}
    tables={}
    for split in ("train","dev"):
        directory=inputs/"b16/inputs/b15/inputs/b14/inputs/server-run/data"
        positions=defaultdict(Counter)
        persons=set()
        unique=set()
        samples=0
        for row in readlines(directory/(split+".jsonl")):
            nodes=row["nodes"]; gold=row["gold"]
            assert len(nodes)==12 and 0<=gold<12
            key=tuple(sorted({word for h,_,t in nodes for word in (h,t)}))
            assert key not in persons and row["doc_id"] not in unique
            persons.add(key);unique.add(row["doc_id"]);samples+=1
            positions[row["stratum"]][gold]+=1
        tables[split]={"graphs":samples,"gold_positions":positions,"unique_person_sets":len(persons)}
        sets[split]=persons
    exposed=set()
    for row in readlines(inputs/"b13/cohort/features.jsonl"):
        assert set(row)=={"doc_id","stratum","question","nodes"}
        exposed.add(tuple(sorted({w for h,_,t in row["nodes"] for w in (h,t)})))
    assert not sets["train"]&sets["dev"] and not exposed&(sets["train"]|sets["dev"])
    shortcut={}
    for name in ("archive_dev","fresh_dev"):
        path=inputs/"prior/runs/本地开发-01"/(name+"-输入.jsonl")
        group=defaultdict(lambda:[0,0])
        for row in readlines(path):
            if row["variant"]=="original":
                n,correct=group[row["kind"]]
                group[row["kind"]]=[n+1,correct+int(row["gold"]==row["candidates"][0])]
        shortcut[name]=group
    result={"round":"audit","verified_file_count":len(inventory),"splits":tables,
            "old_b13_unique_person_sets":len(exposed),"train_dev_or_b13_overlap":0,
            "first_candidate_by_stratum_n_and_hits":shortcut,"platform":platform.platform()}
    (out/"input_digests.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding="utf8")
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("inputs","manifest","out"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();audit(a.inputs,a.manifest,a.out)
