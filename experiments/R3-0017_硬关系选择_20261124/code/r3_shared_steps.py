"""Run preregistered development stages with bounded resources on the assigned CPU."""
import argparse,hashlib,json,os,platform,resource,shutil,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(path,datum):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(datum,ensure_ascii=False,indent=2)+"\n",encoding="utf8");os.replace(tmp,path)
def main():
    here=Path.cwd();cfg=json.loads((here/"configs/launch.json").read_bytes())
    ref=Path(cfg["reference"]);inputs=ref.parent/"inputs"; py=sys.executable
    r16=Path(cfg["first_run"])
    setup=json.loads((here/"configs/frozen-dependencies.json").read_bytes())
    for name,h in setup["files"].items():
        p=ref.parent/name
        if sha(p)!=h:raise RuntimeError("Frozen study dependency mismatch "+name)
    import torch,numpy
    save(here/"configs/environment.json",dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,host=platform.node(),platform=platform.platform(),device="cpu",cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONDONTWRITEBYTECODE")}))
    print("Verified frozen dependency files",len(setup["files"]),flush=True)
    output=here/"data/cohort";score=here/"results/prediction";stats=here/"results/stats"
    if cfg["id"]!="R16":
        for n,h in cfg["cohort_anchors"].items():
            if sha(r16/"data/cohort"/n)!=h:raise ValueError("Development cohort hash mismatch")
        shutil.copytree(r16/"data/cohort",output)
    argv=dict(py=py,code=str(here/"code"),r3=str(ref),inputs=str(inputs),cohortdir=str(output),cohort=str(output/"cohort.jsonl.gz"),pred=str(score/"predictions.jsonl.gz"),scoredir=str(score),stats=str(stats))
    for i,parts in enumerate(cfg["steps"]):
        command=[v.format(**argv) for v in parts]
        start=time.monotonic()
        with (here/"logs/stages.jsonl").open("a",encoding="utf8") as f:
            f.write(json.dumps(dict(utc=now(),event="start",stage=i,argv=command),ensure_ascii=False)+"\n")
        print("Stage",i+1,Path(command[3] if command[2]=="-u" else command[2]).name,flush=True)
        proc=subprocess.run(command,check=False)
        with (here/"logs/stages.jsonl").open("a",encoding="utf8") as f:
            f.write(json.dumps(dict(utc=now(),event="end",stage=i,returncode=proc.returncode,seconds=time.monotonic()-start))+"\n")
        if proc.returncode:return proc.returncode
    return 0
if __name__=="__main__":
    t=time.monotonic();rc=1
    try:rc=main()
    except Exception:traceback.print_exc()
    finally:
        res=resource.getrusage(resource.RUSAGE_CHILDREN); own=resource.getrusage(resource.RUSAGE_SELF)
        save(Path("logs/resources.json"),dict(exit_code=rc,elapsed_seconds=time.monotonic()-t,user_cpu_seconds=res.ru_utime+own.ru_utime,system_cpu_seconds=res.ru_stime+own.ru_stime,max_resident_kib=max(res.ru_maxrss,own.ru_maxrss)))
    sys.exit(rc)
