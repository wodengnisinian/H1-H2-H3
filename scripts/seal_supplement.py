"""Seal a successful supplemental development-only run and atomically update the local study ledger."""
import argparse,csv,difflib,hashlib,io,json,os
from datetime import datetime,timezone
from pathlib import Path
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def j(p):return json.loads(p.read_bytes())
def utc():return datetime.now(timezone.utc).isoformat()
def replace(p,b):
    tmp=p.with_suffix(p.suffix+".pending");tmp.write_bytes(b);os.replace(tmp,p)
def writej(p,v):
    replace(p,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
def add(p,txt):
    if txt.encode() not in p.read_bytes():replace(p,p.read_bytes()+txt.encode())
def main(folder):
    root=folder.parents[1];rel=folder.relative_to(root).as_posix()
    start=j(folder/"logs/start.json");rid=start["id"];proto=j(folder/"configs/protocol.json")
    endlog=j(folder/"logs/exit.json");timing=j(folder/"logs/timing.txt")
    receipt=j(folder/"logs/remote-files.json");summary=j(folder/"results/summary.json")
    assert endlog["exit_code"]==timing["exit_code"]==0 and (folder/"logs/exit.txt").read_text().strip()=="0"
    ledger=root/"experiment_registry.jsonl";registered=[json.loads(b) for b in ledger.read_bytes().splitlines() if b]
    checked=[r for r in registered if r["id"]==rid];assert len(checked) in (1,2) and checked[0]==start
    if len(checked)==2:
        assert sha(folder/"manifest.json")==checked[1]["manifest_sha256"]
        fresh=j(folder/"manifest.json")["files"]
        assert all(sha(folder/n)==v["sha256"] for n,v in fresh.items())
        print("Already sealed",rid);return
    assert start["protocol_sha256"]==sha(folder/"configs/protocol.json")
    assert all(sha(folder/n)==v for n,v in start["source_sha256"].items())
    assert start["input_manifest_sha256"]==sha(folder/"data/frozen-inputs.json")
    for n,h in receipt.items():
        if n.startswith("results/"):p=folder/n
        elif n.endswith(".py"):p=folder/"code"/n
        elif n=="protocol.json":p=folder/"configs"/n
        elif n=="frozen-inputs.json":p=folder/"data"/n
        else:p=folder/"logs"/n
        assert sha(p)==h,n
    assert (folder/"report.md").stat().st_size>1000
    delta_dir=folder/"code"; primary=proto.get("entry","r3_dev_order_audit.py")
    text=(delta_dir/primary).read_text(encoding="utf8").splitlines(keepends=True)
    (delta_dir/"experiment.patch").write_text("".join(difflib.unified_diff([],text,fromfile="/dev/null",tofile="b/"+primary)),encoding="utf8")
    files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob("*"))
           if p.is_file() and p.relative_to(folder).as_posix() not in ("artifacts.json","manifest.json")}
    code=folder/"artifacts.json";writej(code,{n:v["sha256"] for n,v in files.items()})
    m=folder/"manifest.json";writej(m,dict(id=rid,status="verified",time=utc(),exit_code=0,files=files,
        artifact_manifest_sha256=sha(code),inputs="data/frozen-inputs.json",command=start["command"],
        checkpoints="Frozen R12 weights (no training), plus original encoders in datasets/归档输入.zip",
        code_diff="code/experiment.patch",result="results/summary.json",research_outcome=proto.get("decision_file","report.md")))
    assert all((folder/n).stat().st_size==v["bytes"] and sha(folder/n)==v["sha256"] for n,v in files.items())
    end={**start,"event":"end","utc":utc(),"code":0,"status":"verified","result_file":"summary.json",
         "artifact_manifest_sha256":sha(code),"manifest_sha256":sha(m),"wall_seconds":timing["elapsed_seconds"],
         "cpu_seconds":timing["user_cpu_seconds"]+timing["system_cpu_seconds"],"checkpoints":[f"checkpoints/checkpoint-{s}.json" for s in proto["seeds"]]}
    table=root/"experiment_summary.csv";buf=io.StringIO(newline="");writer=csv.writer(buf)
    counts=list(csv.DictReader((folder/"results"/proto.get("summary_csv","accuracy.csv")).open(encoding="utf8")))
    for row in counts:
        for k,v in row.items():
            if k not in ("stratum","method","view","rule","order","presentation","policy","seed","condition","metric") and v and v not in ("False","True"):
                try:num=float(v)
                except ValueError:continue
                key=".".join(str(row[x]) for x in ("stratum","method","view","rule","order","presentation","policy","seed","condition","metric") if x in row)
                writer.writerow([rid,rel,0,end["wall_seconds"],key+"."+k,v])
    add(table,buf.getvalue())
    add(root/"hypotheses.md",f"\n## {rid} 开发对照\n\n{proto['hypothesis']}\n\n{proto['criteria']}\n")
    add(root/"research_log.md",f"\n- {start['utc']} {rid} 开发干预开始，原有正式验收开封次数为 1。\n- {end['utc']} {rid} 云端退出 0，本地复核 {len(files)} 个归档文件。详见 [{rid} 报告]({rel}/report.md)。\n")
    info=f"# {rid} 开发诊断\n\n{proto['hypothesis']}\n\n[完整实验报告、结果与复现材料](../{rel}/report.md)\n"
    report=root/"reports"/(rid+".md");replace(report,info.encode())
    add(ledger,json.dumps(end,ensure_ascii=False)+"\n")
    state=j(root/"state.json");stage=proto.get("after_stage","development_review");next_=proto.get("after_task","R15 同分裁决对照")
    clock_start=datetime.fromisoformat(j(root/"configs/research_protocol.json")["start_local"])
    state.update(stage=stage,updated=end["utc"],latest=end,next_task=next_,historical_formal_model="R12 sharp_role",formal_openings=1,
                 formal_attempts_cap=1,latest_development_report=rel+"/report.md",
                 stop_reason=proto.get("stop_reason","既有正式研究已封存。正在完成针对候选同分的开发对照。"))
    state["budget"]["attempts"]=sum(r["event"]=="start" for r in registered)
    state["budget"]["elapsed_seconds"]=round((datetime.now(timezone.utc)-clock_start).total_seconds())
    writej(root/"state.json",state)
    print(rid,len(files),end["manifest_sha256"])
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("round",type=Path);main(p.parse_args().round.resolve())
