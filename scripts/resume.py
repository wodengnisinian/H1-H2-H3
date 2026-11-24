"""Inspect interrupted attempts, archive their partial work and resume specified IDs."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_loop as loop

def live_cmd(argv):
    wanted=[os.fsencode(x) for x in argv]
    for pid in Path("/proc").glob("[0-9]*"):
        try:
            actual=(pid/"cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if actual==wanted:return pid.name
        except (OSError, PermissionError):pass
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("ids",nargs="*");a=ap.parse_args()
    with (ROOT/".loop.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ev=loop.events();finished={x["path"] for x in ev if x["event"]=="end"}
        for entry in (x for x in ev if x["event"]=="start" and x["path"] not in finished):
            pid=live_cmd(entry["command"])
            if pid:raise RuntimeError("An orphaned worker is still active. PID "+pid)
            run=ROOT/entry["path"];spec=loop.read(run/"preregistration.json")
            inventory={p.relative_to(run).as_posix():loop.sha(p) for p in run.rglob("*") if p.is_file() and p.name!="artifacts.json"}
            loop.save(run/"artifacts.json",inventory)
            row=dict(entry,event="end",utc=loop.now(),code=125,status="interrupted",
                     artifact_manifest_sha256=loop.sha(run/"artifacts.json"),result_file=spec["receipt"])
            loop.append(loop.REG,row)
            loop.status(row,"interrupted","保留部分结果，核对日志并复跑原假设。正式验收只可恢复已封存样本。")
        for x in loop.events():
            if x["event"]=="end":loop.verify_finished(ROOT/x["path"])
    if a.ids:
        os.execv(sys.executable,[sys.executable,"-B",str(ROOT/"run_loop.py"),*a.ids])
    else:print(json.dumps(loop.read(loop.STATE),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
