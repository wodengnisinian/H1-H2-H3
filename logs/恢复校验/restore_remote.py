"""Copy an existing sealed study and verify the exact remote byte manifest. Password via stdin."""
import base64, hashlib, json, os, posixpath, shlex, sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import paramiko
root=Path(__file__).resolve().parents[2]
receipts=Path(__file__).resolve().parent
remote="/data/spiderweb-r3-20261124-1740"
def h(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
class Pin(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        if base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode()!="kEh67cJw4PZxlHiwdITQptZmjuYHjQO6n3+K7YxKZd8=":
            raise ValueError("SSH host key mismatch")
cl=paramiko.SSHClient();cl.set_missing_host_key_policy(Pin())
cl.connect("0eceidqyu1kyeqqecho.funhpc.com",port=21442,username="root",password=sys.stdin.readline().strip(),allow_agent=False,look_for_keys=False,timeout=15)
def query(command):
    _,out,err=cl.exec_command(command)
    data=out.read()
    if out.channel.recv_exit_status(): raise ValueError(err.read())
    return data
inventory_code="""import hashlib,json
from pathlib import Path
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
root=Path('R3')
files={p.relative_to(root).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='.loop.lock'}
print(json.dumps(dict(files=files, archives={n:sha(Path(n)) for n in ['inputs.zip','r3-closed-20261124.zip']})))
"""
inv=json.loads(query("cd "+remote+" && /data/miniconda/envs/torch/bin/python -B -c "+shlex.quote(inventory_code)))
def save(path,data):
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f"Different local file exists: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_bytes(data);temp.rename(path)
save(receipts/"remote_files.json",json.dumps(inv,ensure_ascii=False,indent=2).encode())
def copy_file(item):
    name,dest,expected=item
    if dest.exists() and h(dest)==expected: return
    if dest.exists(): raise ValueError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+".partial")
    s=cl.open_sftp()
    try:s.get(posixpath.join(remote,name),str(tmp),prefetch=True,max_concurrent_prefetch_requests=64)
    finally:s.close()
    if h(tmp)!=expected: raise ValueError("transfer checksum: "+name)
    tmp.rename(dest); print("received",name,flush=True)
try:
    bundle=receipts/"r3-closed-20261124.zip"
    inputzip=root/"datasets/归档输入.zip"
    tasks=[("r3-closed-20261124.zip",bundle,inv["archives"]["r3-closed-20261124.zip"]),("inputs.zip",inputzip,inv["archives"]["inputs.zip"])]
    with ThreadPoolExecutor(2) as pool: list(pool.map(copy_file,tasks))
    with ZipFile(bundle) as z:
        for member in z.infolist():
            name=member.filename
            if name not in inv["files"] or not (root/name).resolve().is_relative_to(root):raise ValueError(name)
            data=z.read(member); expect=inv["files"][name]
            if len(data)!=expect["size"] or hashlib.sha256(data).hexdigest()!=expect["sha256"]:raise ValueError(name)
            save(root/name,data)
        extras=set(inv["files"])-set(z.namelist())
    for name in sorted(extras):
        if not (root/name).resolve().is_relative_to(root):raise ValueError(name)
        copy_file(("R3/"+name,root/name,inv["files"][name]["sha256"]))
    for name,spec in inv["files"].items():
        if h(root/name)!=spec["sha256"] or (root/name).stat().st_size!=spec["size"]:raise ValueError(name)
    print("verified exact cloud files",len(inv["files"]),flush=True)
finally:cl.close()
