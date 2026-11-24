"""Compare completed development runs by graph and verify linkage to the preexisting H3 opening.
No model calls or access to formal question/label rows.
"""
import csv,hashlib,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile
import numpy as np
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def j(p):return json.loads(p.read_bytes())
def primary_by_graph(folder):
    groups={}
    with (folder/"results/stats/graph_metrics.csv").open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            key=row["stratum"],row["graph_id"]; group=groups.setdefault(key,{})
            seed=int(row["seed"]);assert seed not in group
            group[seed]=np.array([float(row["primary.h1_accuracy"]),float(row["primary.h2_accuracy"])])
    assert all(set(v)=={1701,2701,3701} for v in groups.values()) and len(groups)==600
    return {k:np.mean(list(v.values()),axis=0) for k,v in groups.items()}
def compare(root,out):
    reg=[json.loads(b) for b in (root/"experiment_registry.jsonl").read_bytes().splitlines()]
    newest={v["id"]:v for v in reg if v["event"]=="end"};dev=[root/newest[n]["path"] for n in ("R16","R17")]
    assert all(j(p/"manifest.json")["exit_code"]==0 for p in dev)
    assert sha(dev[0]/"data/cohort/cohort.jsonl.gz")==sha(dev[1]/"data/cohort/cohort.jsonl.gz")
    a,b=map(primary_by_graph,dev);assert a.keys()==b.keys()
    rng=np.random.default_rng(2026112417);bt=np.zeros((10000,2));delta=np.zeros(2)
    for k in sorted({k for k,_ in a}):
        dif=np.array([b[idx]-a[idx] for idx in sorted(a) if idx[0]==k]);assert len(dif)==200
        delta+=dif.mean(0)/3
        for i in range(0,10000,100):
            bt[i:i+100]+=dif[rng.integers(0,200,(100,200))].mean(1)/3
    data={}
    for j_,name in enumerate(("h1_accuracy","h2_accuracy")):
        sorted_b=np.sort(bt[:,j_]);data[name]=dict(delta=float(delta[j_]),lower_one_sided=float(sorted_b[499]),ci95=[float(sorted_b[249]),float(sorted_b[9749])])
    rd=root/"experiments/11-R09/results";decision=j(root/"analysis/H3验收判定.json");freeze=j(rd/"freeze.json");opening=j(rd/"opening.json")
    assert decision["used_formal_openings"]==decision["max_formal_openings"]==1 and decision["h3_passed"]
    assert sha(rd/"freeze.json")==decision["freeze_sha256"] and sha(rd/"prediction-seal.json")==opening["seal"]==decision["prediction_seal_sha256"]
    assert sha(rd/"metrics.json")==decision["scoring_sha256"] and sha(root/"datasets/formal-attempt.json")==sha(rd/"freeze.json")
    prefix="/data/spiderweb-r3-20261124-1740/"
    h=freeze["sha256"];links={}
    for seed in (1701,2701,3701):
        k=f"R3/experiments/10-R12/results/checkpoint-{seed}.json";flag=sha(dev[1]/f"checkpoints/primary-{seed}.json");assert flag==h[prefix+k];links[k]=flag
    for name in ("sharp_core.py","role_core.py"):
        k="R3/src/"+name;flag=sha(root/"experiments/10-R12/snapshot/src"/name);assert flag==h[prefix+k];links[k]=flag
    with ZipFile(root/"datasets/归档输入.zip") as z:
        for k in ("inputs/prior/code/model.py","inputs/b13/冻结清单.json"):
            flag=hashlib.sha256(z.read(k)).hexdigest();assert flag==h[prefix+k];links[k]=flag
    assert all(newest[id_]["utc"]>opening["utc"] for id_ in ("R16","R17"))
    verification={}
    for p in dev:
        chk=subprocess.run([sys.executable,"-B",str(p/"code/r3_shared_verify.py"),str(p)],capture_output=True,check=True,encoding="utf8")
        result=json.loads(chk.stdout);assert result["verified"];verification[result["round"]]=result
    record=dict(utc=datetime.now(timezone.utc).isoformat(),development_protocol_sha256=sha(root/"configs/共享权重续研冻结协议_20261124.json"),same_dev_cohort=sha(dev[0]/"data/cohort/cohort.jsonl.gz"),development_graph_paired_R12_minus_original=data,formal_scope="existing R09 historical H3, no second opening; R16 and R17 development checks postdate formal R09",formal_version_freeze_links=links,formal_model=freeze["module"],formal_openings=1,formal_opened_utc=opening["utc"],historical_metrics_sha256=decision["scoring_sha256"],historic_h3_passed=decision["h3_passed"],verification=verification)
    with out.open("x",encoding="utf8") as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps(data,ensure_ascii=False))
if __name__=="__main__":compare(Path(sys.argv[1]).resolve(),Path(sys.argv[2]))
