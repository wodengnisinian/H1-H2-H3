"""Verify each recovered byte and register the scientific development outcome separately from process success."""
from pathlib import Path
import argparse,csv,hashlib,io,json,os,sys,shutil,difflib
from datetime import datetime,timezone
from shared_cloud_run import sha,j,save,ledger_add,now

def record(root,filename,entry):
    p=root/filename;old=p.read_bytes();add=entry.encode()
    if add not in old:
        tmp=p.with_suffix(".update.tmp");tmp.write_bytes(old+add);os.replace(tmp,p)

def seal(folder):
    root=folder.parents[1]
    rc=j(folder/"logs/exit.json")["exit_code"]; cfg=j(folder/"configs/launch.json")
    reg=root/"experiment_registry.jsonl"; lease=j(folder/"logs/start.json");rid=cfg["id"]
    events=[json.loads(x) for x in reg.read_bytes().splitlines()]
    old=[r for r in events if r["id"]==rid]
    if len(old)==2:
        assert sha(folder/"manifest.json")==old[-1]["manifest_sha256"];print("archived",rid);return
    assert len(old)==1 and old[0]==lease and (folder/"report.md").is_file()
    remote=j(folder/"logs/remote.json")
    assert sha(folder/"logs/cloud-snapshot.zip")==remote["archive_sha256"]
    for n,h in {**remote["files"],**lease["source_sha256"]}.items():
        assert sha(folder/n)==h,n
    if not rc:
        co=j(folder/"data/cohort/manifest.json"); assert co["graphs"]==600 and sha(folder/"data/cohort/cohort.jsonl.gz")==co["cohort_sha256"]
        primary=j(folder/"results/stats/metrics.json")["primary"]
        passed=bool(primary["h1_pass"] and primary["h2_pass"])
        assert bool(primary["h1_pass"])==(primary["h1_accuracy"]>=.8 and primary["h1_lcb95"]>0)
        assert bool(primary["h2_pass"])==(primary["h2_delta"]>=.02 and primary["h2_lcb95"]>0)
    else:primary={};passed=False
    # Diffs of the new experimental sources and the prior scorer are retained beside exact frozen study refs.
    patches=[]
    for p in sorted((folder/"code").glob("r3_shared_*.py")):
        patches.extend(difflib.unified_diff([],p.read_text(encoding="utf8").splitlines(keepends=True),fromfile="/dev/null",tofile="b/"+p.name))
    (folder/"code/新增实验脚本.patch").write_text("".join(patches),encoding="utf8")
    files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob("*")) if p.is_file() and p.relative_to(folder).as_posix() not in ("manifest.json","artifacts.json")}
    save(folder/"artifacts.json",{n:v["sha256"] for n,v in files.items()})
    save(folder/"manifest.json",dict(status="verified",id=rid,utc=now(),files=files,exit_code=rc,development_pass=passed,primary=primary,cohort_sha256=co["cohort_sha256"] if not rc else None))
    for n,v in files.items():assert (folder/n).stat().st_size==v["bytes"] and sha(folder/n)==v["sha256"],n
    elapsed=j(folder/"logs/resources.json") if (folder/"logs/resources.json").exists() else {}
    end={**lease,"event":"end","utc":now(),"code":rc,"status":"verified","artifact_manifest_sha256":sha(folder/"artifacts.json"),"manifest_sha256":sha(folder/"manifest.json"),"result_file":"stats/metrics.json" if not rc else None,"checkpoints":[f"checkpoints/primary-{s}.json" for s in [1701,2701,3701]],"wall_seconds":elapsed.get("elapsed_seconds"),"development":primary}
    p=(folder/"report.md"); (root/"reports"/(rid+".md")).write_text("# "+rid+" 开发复核\n\n[完整报告与实验材料](../"+folder.relative_to(root).as_posix()+"/report.md)\n",encoding="utf8")
    record(root,"hypotheses.md","\n## "+rid+" 新开发队列\n\n"+j(folder/"configs/protocol.json")["hypothesis"]+" 事前主判据见本轮 configs/protocol.json。\n")
    record(root,"research_log.md",f"\n- {lease['utc']} {rid} 开始，固定新开发评测和代码版本。\n- {end['utc']} {rid} 退出 {rc}，{len(files)} 个文件本地 SHA-256 复核完成。\n")
    buf=io.StringIO(newline="");w=csv.writer(buf)
    for k,v in primary.items():
        if isinstance(v,(int,float,bool)):w.writerow([rid,end["path"],rc,end["wall_seconds"],"primary."+k,v])
    if not primary:w.writerow([rid,end["path"],rc,end["wall_seconds"],"completed",False])
    record(root,"experiment_summary.csv",buf.getvalue())
    ledger_add(reg,end)
    state=j(root/"state.json")
    state.update(stage="development_review",latest=end,updated=end["utc"],latest_development_report=lease["path"]+"/report.md",next_task="按开发结果核对原公式与改进版本",formal_openings=1,formal_attempts_cap=1)
    state["budget"]["attempts"]=len([r for r in events if r["event"]=="start"])
    state["budget"]["elapsed_seconds"]=int((datetime.now(timezone.utc)-datetime.fromisoformat("2026-11-24T17:38:00+08:00")).total_seconds())
    save(root/"state.json",state)
    print("sealed",rid,len(files),"process",rc,"scientific_pass",passed,flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("folder",type=Path);seal(a.parse_args().folder.resolve())
