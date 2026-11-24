"""Run or collect one preregistered development experiment. SSH password is supplied on stdin."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,posixpath,shlex,shutil,sys,time,stat
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
import paramiko
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def j(p):return json.loads(p.read_bytes())
def save(p,obj):
    tmp=p.with_suffix(p.suffix+".partial");tmp.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode());os.replace(tmp,p)
def ledger_add(path,e):
    old=path.read_bytes()
    if old and not old.endswith(b'\n'):raise ValueError(path)
    tmp=path.with_suffix(".append.tmp")
    with tmp.open("xb") as f:f.write(old+(json.dumps(e,ensure_ascii=False)+"\n").encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
class Pinned(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        if base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode()!="kEh67cJw4PZxlHiwdITQptZmjuYHjQO6n3+K7YxKZd8=":raise ValueError("Unexpected SSH host key")
def query(client,cmd):
    _,o,e=client.exec_command(cmd,timeout=180);data=o.read();err=e.read()
    if o.channel.recv_exit_status()!=0:raise RuntimeError(err.decode() or data.decode())
    return data.decode()
def exists(s,path):
    try:s.stat(path);return True
    except FileNotFoundError:return False
def run(round_,collect=False):
    root=round_.parents[1];l=round_/"logs";cfg=j(round_/"configs/launch.json");id_=cfg["id"];remote=cfg["remote"]
    assert id_.startswith("R") and id_[1:].isdigit() and int(id_[1:])>=16
    assert remote=="/data/spiderweb-shared-protocol-20261124-"+id_.lower()
    limit=j(root/"configs/共享权重续研冻结协议_20261124.json")["budget"]
    reg=root/"experiment_registry.jsonl";events=[json.loads(b) for b in reg.read_bytes().splitlines()]
    starts=[e for e in events if e["event"]=="start"]; prior=j(root/"state.json")
    if not collect:
        assert len(starts)<limit["max_attempts"] and id_ not in [e["id"] for e in starts]
        if len(starts)!=sum(e["event"]=="end" for e in events):raise ValueError("An earlier attempt lacks a completed archive")
        prev=[e for e in events if e["event"]=="end"][-1]
        assert sha(root/prev["path"]/"artifacts.json")==prev["artifact_manifest_sha256"]
        if "manifest_sha256" in prev:assert sha(root/prev["path"]/"manifest.json")==prev["manifest_sha256"]
        assert prior["formal_openings"]==limit["formal_openings_max"]==1
    elif not (l/"start.json").is_file():raise ValueError("No attempt to recover")
    budget_remaining=int(limit["max_elapsed_hours"]*3600-(datetime.now(timezone.utc)-datetime.fromisoformat(limit["since"])).total_seconds())
    seconds=min(900,budget_remaining-30)
    if seconds<=0 and not collect:raise ValueError("Elapsed budget exhausted")
    c=paramiko.SSHClient();c.set_missing_host_key_policy(Pinned())
    c.connect("0eceidqyu1kyeqqecho.funhpc.com",port=21442,username="root",password=sys.stdin.readline().strip(),allow_agent=False,look_for_keys=False,timeout=12)
    try:
        s=c.open_sftp()
        cmd=shlex.join([cfg["py"],"-B","-u","code/r3_shared_steps.py"])
        if not collect:
            s.mkdir(remote,mode=0o700)
            for name in ("code","configs","data","results","checkpoints","logs"):s.mkdir(remote+"/"+name,mode=0o700)
            staged=[*(round_/"code").glob("*.py"),*(round_/"configs").glob("*.json")]
            for p in staged:s.put(str(p),remote+"/"+p.relative_to(round_).as_posix())
            start=dict(event="start",id=id_,path=round_.relative_to(root).as_posix(),utc=now(),command=[cfg["py"],"-B","-u",remote+"/code/r3_shared_steps.py"],source_sha256={p.relative_to(round_).as_posix():sha(p) for p in staged},protocol_sha256=sha(round_/"configs/protocol.json"),input_manifest_sha256=sha(round_/"configs/frozen-dependencies.json"),phase="development",max_seconds=seconds)
            save(l/"start.json",start);ledger_add(reg,start)
            prior.update(stage="running_development",latest=start,updated=now(),next_task="同步输出、核查本轮，再决定后续。")
            prior["budget"]["attempts"]=len(starts)+1;save(root/"state.json",prior)
            run_cmd="cd "+shlex.quote(remote)+" && (timeout --kill-after=8s "+str(seconds)+"s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "+cmd+" >logs/remote.log 2>&1; echo $? >logs/exit.part; mv logs/exit.part logs/exit.txt)"
            remote_pid=query(c,"nohup bash -c "+shlex.quote(run_cmd)+" </dev/null >/dev/null 2>&1 & echo $!").strip()
            save(l/"process.json",{"pid":remote_pid,"utc":now(),"remote":remote})
        # Mirror stdout and completed checkpoint/seed files. Final copy validates complete bytes.
        offset=(l/"live.log").stat().st_size if (l/"live.log").exists() else 0
        deadline=time.monotonic()+max(seconds,120)+60
        mirrored={}
        while True:
            if exists(s,remote+"/logs/remote.log"):
                with s.open(remote+"/logs/remote.log","rb") as rf:
                    rf.seek(offset);b=rf.read()
                if b:
                    with (l/"live.log").open("ab") as lf:lf.write(b)
                    offset+=len(b);print(b.decode(errors="replace"),end="",flush=True)
            for dirname in ("data/cohort","results/prediction","results/stats","logs"):
                full=remote+"/"+dirname
                if not exists(s,full):continue
                for ent in s.listdir_attr(full):
                    if ".partial" in ent.filename or ".tmp" in ent.filename or not stat.S_ISREG(ent.st_mode) or not ent.filename.endswith((".gz",".jsonl",".json",".csv")):continue
                    name=dirname+"/"+ent.filename;sig=(ent.st_size,ent.st_mtime)
                    dst=round_/name;dst.parent.mkdir(parents=True,exist_ok=True)
                    if mirrored.get(name)==sig:continue
                    tmp=dst.with_suffix(dst.suffix+".sync");s.get(remote+"/"+name,str(tmp));os.replace(tmp,dst);mirrored[name]=sig
            if exists(s,remote+"/logs/exit.txt"):break
            if time.monotonic()>deadline:raise TimeoutError("Collection timeout. Use --collect with the recorded remote run.")
            time.sleep(2)
        # A single remote snapshot backs every output, including partial files from failed runs.
        export="""from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
p=Path.cwd()
def digest(f):
    with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
files=sorted(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ('archive.zip','archive.part'))
with ZipFile('archive.part','w',ZIP_DEFLATED,compresslevel=2) as z:
    for f in files:z.write(f,f.relative_to(p).as_posix())
Path('archive.part').replace('archive.zip')
print(json.dumps(dict(files={f.relative_to(p).as_posix():digest(f) for f in files},archive_sha256=digest(Path('archive.zip')))))"""
        checks=json.loads(query(c,"cd "+shlex.quote(remote)+" && "+shlex.join([cfg["py"],"-B","-c",export])))
        out=l/"cloud-snapshot.zip";tmp=out.with_suffix(".download");s.get(remote+"/archive.zip",str(tmp),prefetch=True,max_concurrent_prefetch_requests=64)
        if sha(tmp)!=checks["archive_sha256"]:raise ValueError("Download digest mismatch")
        os.replace(tmp,out)
        with ZipFile(out) as z:
            assert set(z.namelist())==set(checks["files"])
            for n,expected in checks["files"].items():
                dest=(round_/n).resolve()
                assert dest.is_relative_to(round_) and hashlib.sha256(z.read(n)).hexdigest()==expected
                if n in j(l/"start.json")["source_sha256"]:assert sha(dest)==expected
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
        save(l/"remote.json",checks);rc=int((l/"exit.txt").read_text().strip());save(l/"exit.json",dict(exit_code=rc,utc=now(),remote=remote,archive_sha256=sha(out)))
        print("collected",id_,rc,len(checks["files"]),flush=True);return rc
    finally:c.close()
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("round_dir",type=Path);p.add_argument("--collect",action="store_true");a=p.parse_args();sys.exit(run(a.round_dir.resolve(),a.collect))
