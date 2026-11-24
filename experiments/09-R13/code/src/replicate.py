"""Repeat R08 from zero in a fresh process and compare persisted scientific outputs."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys
def dig(p):
    opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
p=argparse.ArgumentParser()
for k in ("inputs","candidate","out"):p.add_argument("--"+k,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
repeat=a.out/"training"
subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("role_run.py")),"--b13",str(a.inputs/"b13"),
                "--b16",str(a.inputs/"b16"),"--prior",str(a.inputs/"prior"),"--diagnostic-only","--out",str(repeat)],check=True)
files=[f.name for f in a.candidate.iterdir() if f.name.startswith(("parameters-","checkpoint-","predictions-","errors-"))]
receipt={}
for name in sorted(files):
    before=dig(a.candidate/name);after=dig(repeat/name);receipt[name]={"sha256":after,"identical":before==after}
left=json.loads((a.candidate/"metrics.json").read_bytes());right=json.loads((repeat/"metrics.json").read_bytes())
metrics={k:left[k]==right[k] for k in ("aggregate","h1","per_seed","objective_intervals","paired_intervals","criterion_met")}
ok=all(x["identical"] for x in receipt.values()) and len(files)==12 and all(metrics.values())
(a.out/"metrics.json").write_text(json.dumps(dict(hypothesis="R08 deterministic repeat",matched_files=receipt,
    matched_metrics=metrics,reproduced=ok),ensure_ascii=False,indent=2)+"\n",encoding="utf8")
if not ok:raise ValueError("Repeat mismatch; see metrics.json")
