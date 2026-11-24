"""Run preregistered R3 experiments on the allocated Linux host and archive each attempt."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
from hashlib import file_digest
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent/"inputs"
REG = ROOT/"experiment_registry.jsonl"
STATE = ROOT/"state.json"
CORE = ROOT/"configs/research_protocol.json"


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f: return file_digest(f,"sha256").hexdigest()
def save(p,d):
    tmp=p.with_name(p.name+".pending")
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,p)
def append(p,d):
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
def events():return [json.loads(s) for s in REG.read_text(encoding="utf8").splitlines()] if REG.exists() else []
def budget():
    protocol=read(CORE)
    elapsed=time.time()-datetime.fromisoformat(protocol["start_local"]).timestamp()
    return dict(attempts=sum(r["event"]=="start" for r in events()),max_attempts=protocol["max_attempts"],
                elapsed_seconds=max(0,round(elapsed)),max_seconds=3600*protocol["max_elapsed_hours"])
def status(row,phase,next_task):
    save(STATE,dict(stage=phase,latest=row,budget=budget(),next_task=next_task,updated=now()))
def logmsg(text):
    with (ROOT/"research_log.md").open("a",encoding="utf8") as f:f.write("\n- "+now()+"  "+text+"\n")

def verify_finished(run):
    hashes=read(run/"artifacts.json")
    for name,expected in hashes.items():
        if sha(run/name)!=expected:raise ValueError("Archived artifact changed: "+str(run/name))

def summarize(expid, rel, code, elapsed, metrics):
    def values(obj, prefix=""):
        if isinstance(obj,dict):
            for k,v in obj.items():
                yield from values(v,prefix+"."+str(k) if prefix else str(k))
        elif type(obj) in (int,float,bool):
            yield prefix,obj
    table=ROOT/"experiment_summary.csv"
    new=not table.exists()
    with table.open("a",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        if new:writer.writerow(["experiment","archive","return_code","seconds","metric","value"])
        writer.writerow([expid,rel,code,elapsed,"completed",code==0])
        for name,value in values(metrics):writer.writerow([expid,rel,code,elapsed,name,value])

def execute(expid):
    specpath=ROOT/"configs"/(expid+".json")
    cfg=read(specpath)
    limits=budget()
    tracked={p.relative_to(ROOT).as_posix():sha(p) for pat in cfg["sources"] for p in ROOT.glob(pat) if p.is_file()}
    for p in [specpath,CORE,ROOT/"run_loop.py"]:tracked[p.relative_to(ROOT).as_posix()]=sha(p)
    previous=[e for e in events() if e["id"]==expid and e["event"]=="end" and e["code"]==0]
    if previous:
        last=previous[-1];verify_finished(ROOT/last["path"])
        if last["source_sha256"]!=tracked:raise ValueError("Completed hypothesis has different sources; assign new ID")
        print("Verified completed "+expid,flush=True);return
    if limits["attempts"]>=limits["max_attempts"] or limits["elapsed_seconds"]>=limits["max_seconds"]:
        raise RuntimeError("Study budget exhausted")
    seq=limits["attempts"]+1; rel="experiments/%02d-%s"%(seq,expid)
    run=ROOT/rel;run.mkdir(parents=True,exist_ok=False)
    out=run/"results"
    for name in tracked:
        to=run/"snapshot"/name;to.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,to)
    argv=[sys.executable,"-B",*[s.format(root=ROOT,inputs=INPUT,out=out,py=sys.executable) for s in cfg["argv"]]]
    datahash=sha(ROOT/"datasets/inputs.json")
    common=dict(id=expid,path=rel,command=argv,protocol_sha256=sha(specpath),source_sha256=tracked,input_manifest_sha256=datahash)
    started=dict(event="start",utc=now(),**common);append(REG,started)
    status(started,"running",cfg["after"]);save(run/"preregistration.json",cfg)
    with (ROOT/"hypotheses.md").open("a",encoding="utf8") as f:
        f.write("\n## "+expid+"  "+now()+"\n\n"+cfg["hypothesis"]+"\n\n"+cfg["criteria"]+"\n")
    logmsg(expid+" 开始。"+cfg["hypothesis"])
    env=dict(os.environ, PYTHONIOENCODING="utf8",PYTHONDONTWRITEBYTECODE="1",
             OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONUNBUFFERED="1")
    # Reserve time to persist state and checkpoint receipts.
    timeout=max(1,min(cfg.get("timeout_seconds",2400),limits["max_seconds"]-limits["elapsed_seconds"]))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
    with (run/"stdout.log").open("w",encoding="utf-8") as fo:
        proc=subprocess.Popen(argv,stdout=fo,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
        try: code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    seconds=round(time.monotonic()-t,3)
    if code==0 and not (out/cfg["receipt"]).is_file():code=66
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    measured=dict(wall_seconds=seconds,cpu_seconds=round(after.ru_stime+after.ru_utime-before.ru_stime-before.ru_utime,3),
                  child_peak_rss_process_high_water_kib=after.ru_maxrss,platform=platform.platform(),python=sys.version)
    save(run/"resources.json",measured)
    inventory={p.relative_to(run).as_posix():sha(p) for p in run.rglob("*") if p.is_file() and p.name!="artifacts.json"}
    save(run/"artifacts.json",inventory)
    end=dict(event="end",utc=now(),code=code,result_file=cfg["receipt"],**common,**measured,
             artifact_manifest_sha256=sha(run/"artifacts.json"),checkpoints=[k for k in inventory if "checkpoint" in k.lower() or k.endswith(".pt")])
    append(REG,end);status(end,"review" if code==0 else "failed",cfg["after"] if code==0 else "检查失败尝试的 stdout.log 后修复并重试")
    summarize(expid,rel,code,seconds,read(out/cfg["receipt"]) if code==0 else {})
    logmsg(expid+(" 完成。" if code==0 else " 执行失败。")+"归档 "+rel)
    print(expid,code,rel,seconds,flush=True)
    if code: raise RuntimeError("Research step failed: "+expid)
    return end

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids",nargs="*");p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    with (ROOT/".loop.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.verify:
            for e in events():
                if e["event"]=="end":verify_finished(ROOT/e["path"])
            print("Verified all archived hashes");return
        for expid in a.ids:execute(expid)
        if not a.ids:print(json.dumps(read(STATE) if STATE.exists() else budget(),ensure_ascii=False))
if __name__=="__main__":main()
