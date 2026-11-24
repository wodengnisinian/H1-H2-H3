"""Stage one supplemental run, with an exclusive remote path, pin the SSH host key,
stream logs to the Windows archive and verify downloaded outputs. Password via stdin.
Archives and the formal experiment are read-only on the compute host.
"""
import base64,hashlib,json,os,shlex,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
import paramiko

here=Path(__file__).resolve().parents[1]
root=here.parents[1]
protocol=json.loads((here/"configs/protocol.json").read_bytes())
remote=protocol["remote_run"]
original=protocol["source_study"]
inputs=original.rsplit("/",1)[0]+"/inputs"
python=protocol["py"]
rcmd=[python,"-B",remote+"/r3_content_tie_audit.py","--r3",original,"--inputs",inputs,"--r14","/data/spiderweb-r3-supplement-R3-0014_Order_20261124-1903/results","--out",remote+"/results"]
def now():return datetime.now(timezone.utc).isoformat()
def filehash(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".pending")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def append_registry(entry):
    path=root/"experiment_registry.jsonl"
    tmp=path.with_suffix(".new.tmp");old=path.read_bytes()
    if old and not old.endswith(b"\n"):raise ValueError("Unexpected registry format")
    with tmp.open("xb") as dest:
        dest.write(old+(json.dumps(entry,ensure_ascii=False)+"\n").encode());dest.flush();os.fsync(dest.fileno())
    os.replace(tmp,path)
def prepare():
    if (here/"logs/start.json").exists():raise ValueError("An attempt has already been started. Preserve and recover it.")
    for d in ("code","data","checkpoints","configs","logs","results"): (here/d).mkdir(exist_ok=True)
    manifest=json.loads((root/"logs/恢复校验/remote_files.json").read_bytes())
    # Every original local and remote reference needed for model/shortlist/identity scoring.
    wanted=["experiments/10-R12/results/"+name for name in ["development-inputs.jsonl.gz","configuration.json",*(f"predictions-{s}.jsonl.gz" for s in protocol["seeds"]),*(f"checkpoint-{s}.json" for s in protocol["seeds"])]]
    wanted += [name for name in manifest["files"] if name.startswith(("src/","experiments/10-R12/snapshot/")) and name.endswith(".py")]
    wanted.append("datasets/inputs.json")
    wanted=sorted(set(wanted))
    hashes={}
    for name in wanted:
        digest=filehash(root/name)
        if digest!=manifest["files"][name]["sha256"]:raise ValueError(name)
        hashes["R3/"+name]=digest
    historical=root/"datasets/归档输入.zip"
    shared=json.loads((root/"datasets/inputs.json").read_bytes())
    if filehash(historical)!=shared["zip_sha256"]:raise ValueError("Historical inputs archive checksum changed")
    for name,stamp in shared["files"].items():
        if not name.startswith("inputs/"):raise ValueError(name)
        hashes[name]=stamp["sha256"]
    for s in protocol["seeds"]:
        src=root/"experiments/10-R12/results"/f"checkpoint-{s}.json"
        shutil.copyfile(src,here/"checkpoints"/src.name)
    for n in wanted:
        if n.endswith(".py") and n.startswith("experiments/10-R12/snapshot/"):
            rel=n.split("/snapshot/",1)[1];dst=here/"code/R12"/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/n,dst)
    with ZipFile(historical) as z:
        for name in shared["files"]:
            if name.endswith(".py"):
                dest=here/"code"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    save(here/"data/frozen-inputs.json",{"note":"R3/ denotes the local archive root. inputs/ members are retained in datasets/归档输入.zip. Use only the 600 original fresh-development questions; archived model decisions serve as a parity check. No formal cohort used.","historical_archive_sha256":filehash(historical),"sha256":hashes})
    save(here/"configs/command.json",{"argv":rcmd,"env":{"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1"},"timeout_seconds":protocol["max_seconds"],"cpus_used":1})
    return hashes
class Pinned(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        if base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode()!="kEh67cJw4PZxlHiwdITQptZmjuYHjQO6n3+K7YxKZd8=":raise ValueError("SSH host key mismatch")
def shell(c,cmd):
    _,o,e=c.exec_command(cmd,timeout=180);out=o.read();err=e.read()
    if o.channel.recv_exit_status()!=0:raise RuntimeError(err.decode() or out.decode())
    return out.decode()
def transfer_outputs(c,s):
    code="import hashlib,json;from pathlib import Path;p=Path("+repr(remote)+");print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}))"
    checks=json.loads(shell(c,shlex.join([python,"-B","-c",code])))
    for n,h in checks.items():
        if n.startswith("results/") or n in ("timing.txt","remote.log","exit.txt"):
            p=(here / (n if n.startswith("results/") else "logs/"+n)).resolve()
            if not p.is_relative_to(here):raise ValueError(n)
            p.parent.mkdir(parents=True,exist_ok=True)
            tmp=p.with_name(p.name+".part");s.get(remote+"/"+n,str(tmp))
            if filehash(tmp)!=h:raise ValueError(n)
            os.replace(tmp,p)
    save(here/"logs/remote-files.json",checks)
def main():
    hashes=prepare();client=paramiko.SSHClient();client.set_missing_host_key_policy(Pinned())
    client.connect("0eceidqyu1kyeqqecho.funhpc.com",port=21442,username="root",password=sys.stdin.readline().strip(),allow_agent=False,look_for_keys=False,timeout=12)
    try:
        s=client.open_sftp();s.mkdir(remote,mode=0o700)
        for p in [here/"code/r3_dev_order_audit.py",here/"code/r3_content_tie_audit.py",here/"code/record_resources.py",here/"configs/protocol.json",here/"data/frozen-inputs.json"]:
            s.put(str(p),remote+"/"+p.name)
        pre="import hashlib,json;from pathlib import Path;base=Path("+repr(original)+").parent;spec=json.loads(Path("+repr(remote+"/frozen-inputs.json")+").read_bytes())['sha256'];bad=[n for n,h in spec.items() if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h];print(len(spec),bad);raise SystemExit(int(bool(bad)))"
        check=shell(client,shlex.join([python,"-B","-c",pre]))
        env=shell(client,shlex.join([python,"-B","-c","import json,platform,sys,torch,numpy,os;print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,platform=platform.platform(),cpu_quota=open('/sys/fs/cgroup/cpu.max').read().strip(),device='cpu')))"]))
        save(here/"configs/environment.json",json.loads(env))
        start=dict(event="start",id="R15",path=here.relative_to(root).as_posix(),utc=now(),command=rcmd,protocol_sha256=filehash(here/"configs/protocol.json"),source_sha256={n:filehash(here/n) for n in ("code/r3_dev_order_audit.py","code/r3_content_tie_audit.py","code/launch_from_windows.py","code/record_resources.py")},input_manifest_sha256=filehash(here/"data/frozen-inputs.json"),remote_preflight=check.strip())
        save(here/"logs/start.json",start);append_registry(start)
        state=json.loads((root/"state.json").read_bytes())
        state.update(stage="running_supplement",latest=start,next_task="Collect the development-only intervention and verify its archive",updated=now())
        state["budget"]["attempts"]=15;save(root/"state.json",state)
        run=shlex.join([python,"-B",remote+"/record_resources.py"])+" timeout --kill-after=10s "+str(protocol["max_seconds"])+"s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "+shlex.join(rcmd)
        body="cd "+shlex.quote(remote)+" && { "+run+"; ret=$?; echo \"$ret\" >exit.txt; exit \"$ret\"; } 2>&1 | tee -a remote.log; exit ${PIPESTATUS[0]}"
        timed=time.monotonic();ch=client.get_transport().open_session();ch.set_combine_stderr(True);ch.settimeout(protocol["max_seconds"]+45);ch.exec_command(shlex.join(["bash","-c",body]))
        with (here/"logs/live.log").open("xb") as log:
            while True:
                piece=ch.recv(32768)
                if not piece:break
                log.write(piece);log.flush()
            rc=ch.recv_exit_status()
        transfer_outputs(client,s)
        save(here/"logs/exit.json",dict(exit_code=rc,utc=now(),wall_seconds=round(time.monotonic()-timed,3),command=rcmd))
        print("cloud exit:",rc,flush=True);return rc
    finally:client.close()
if __name__=="__main__":sys.exit(main())
