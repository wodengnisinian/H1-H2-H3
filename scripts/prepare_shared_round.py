"""Prepare a fresh local archive before starting a new, named development run."""
import argparse,hashlib,json,shutil
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
REFERENCE="/data/spiderweb-r3-20261124-1740/R3"; CLOUD="/data/spiderweb-shared-protocol-20261124-"
PYTHON="/data/miniconda/envs/torch/bin/python"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def j(p):return json.loads(p.read_bytes())
def newj(p,obj):
    with p.open("x",encoding="utf8") as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write("\n")
def prep(where,scripts,version,first):
    protocol=j(ROOT/"configs/共享权重续研冻结协议_20261124.json")
    reg=[json.loads(b) for b in (ROOT/"experiment_registry.jsonl").read_bytes().splitlines()]
    n=1+sum(x['event']=='start' for x in reg)
    if n>20 or n!=int(where.name[3:7]):raise ValueError("Budget or attempt numbering changed")
    rid="R"+str(n)
    if (where/"configs/launch.json").exists():raise ValueError("Round already prepared")
    for n_ in ("code","configs","results","data","checkpoints","logs"): (where/n_).mkdir(parents=True,exist_ok=True)
    proto={**protocol,"id":rid,"version":version,"utc":datetime.now(timezone.utc).isoformat(),
           "hypothesis":("论文原始共享词权重在一组全新开发图上同时达到 H1 和 H2 的预置门槛。"
                        if version=="original" else "R08 的已训练关系读头配合 R12 冻结硬选择策略，在同一批开发图上同时达到两个门槛。"),
           "controls": ["first-mentioned relation, uniform/random/learned/shuffled token mass; matched archived 1/4 groups and random-label probe as separate diagnostics",
                        "same shortlist: direction, original first layer, uniform all non-padding tokens, learned, shuffled, random, independent pool, direct first; open/closed gate subgroups"],
           "checkpoint_policy":"Reuse original trained checkpoint per seed matched to historical inputs, with stored training configuration and loss records where available. No model selection or fitting to this new cohort."}
    newj(where/"configs/protocol.json",proto)
    frozen={}
    for name in ["r3_shared_steps.py","r3_shared_cohort.py","r3_shared_weight_validation.py","r3_shared_stats.py","r3_shared_verify.py"]:
        shutil.copyfile(scripts/name,where/"code"/name)
    for name in ["prepare_shared_round.py","shared_cloud_run.py","seal_shared_round.py"]:
        shutil.copyfile(ROOT/"scripts"/name,where/"code"/name)
    assets=ROOT/"datasets/归档输入.zip"
    index=j(ROOT/"datasets/inputs.json")
    assert sha(assets)==index["zip_sha256"]
    for name,v in index["files"].items():frozen[name]=v["sha256"]
    inv=j(ROOT/"logs/恢复校验/remote_files.json")["files"]
    # Reproduction dependencies use archived versions. Formal cohort files and labels are excluded.
    for name,v in inv.items():
        use=(name.startswith("src/") and name.endswith(".py") or name.startswith("configs/") and name.endswith(".json") or
             name.startswith("experiments/") and ("development-inputs.jsonl.gz" in name or
             name.split("/")[1] in ("03-R03","06-R06","08-R08","10-R12") and
             ("/snapshot/src/" in name and name.endswith(".py") or "/results/" in name and
              name.rsplit("/",1)[1].startswith(("parameters-","checkpoint-","configuration","dataset.json","events.jsonl")))))
        if name=="experiments/13-R11/results/inputs.jsonl":use=True
        if not use or "/11-R09/" in name:continue
        p=ROOT/name
        if sha(p)!=v["sha256"]:raise ValueError("Historical source mismatch "+name)
        frozen["R3/"+name]=v["sha256"]
    newj(where/"configs/frozen-dependencies.json",dict(files=frozen,shared_zip={"path":"datasets/归档输入.zip","sha256":sha(assets)},instructions="R3/ names resolve at the delivered local archive root. inputs/ files are present in the checksum-pinned shared ZIP. To reproduce on Linux restore R3 and its sibling inputs."))
    # Make the exact trained primary checkpoint immediately accessible in the round.
    with ZipFile(assets) as z:
        for seed in protocol["seeds"]:
            target=where/"checkpoints"/f"primary-{seed}.json"
            if version=="original":
                member=f"inputs/prior/runs/本地开发-01/w-{seed}.json";target.write_bytes(z.read(member))
            else:shutil.copyfile(ROOT/f"experiments/10-R12/results/checkpoint-{seed}.json",target)
    launch=dict(id=rid,version=version,reference=REFERENCE,py=PYTHON,remote=CLOUD+rid.lower(),
                first_run=CLOUD+"r16",steps=[])
    if version=="original" and rid=="R16":
        launch["steps"].append(["{py}","-B","-u","{code}/r3_shared_cohort.py","--r3","{r3}","--inputs","{inputs}","--out","{cohortdir}"])
    else:
        launch["cohort_anchors"]={p.name:sha(p) for p in (first/"data/cohort").iterdir() if p.is_file()}
        assert "cohort.jsonl.gz" in launch["cohort_anchors"]
    launch["steps"].extend([
      ["{py}","-B","-u","{code}/r3_shared_weight_validation.py","--version",version,"--r3","{r3}","--inputs","{inputs}","--cohort","{cohort}","--out","{scoredir}"],
      ["{py}","-B","-u","{code}/r3_shared_stats.py","--cohort","{cohort}","--predictions","{pred}","--out","{stats}"]])
    newj(where/"configs/launch.json",launch)
    print("Prepared",rid,len(frozen),"dependency hashes")
if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("folder",type=Path);a.add_argument("scripts",type=Path);a.add_argument("version",choices=["original","r12"]);a.add_argument("first",type=Path)
    v=a.parse_args();prep(v.folder,v.scripts,v.version,v.first)
