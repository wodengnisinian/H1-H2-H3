"""Compare each registered source snapshot with its last version in this study."""
from pathlib import Path
import difflib,json
root=Path(__file__).resolve().parents[1];out=root/"analysis/diffs";out.mkdir(exist_ok=True)
seen={}
for e in map(json.loads,(root/"experiment_registry.jsonl").read_text().splitlines()):
    if e["event"]!="end":continue
    snap=root/e["path"]/"snapshot";changes=[]
    for f in sorted(snap.rglob("*")):
        if not f.is_file():continue
        k=f.relative_to(snap).as_posix();new=f.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        old_id,old=seen.get(k,("new",[]))
        changes.extend(difflib.unified_diff(old,new,fromfile=f"{old_id}/{k}" if old else "/dev/null",tofile=e["id"]+"/"+k))
        seen[k]=(e["id"],new)
    (out/(Path(e["path"]).name+".patch")).write_text("".join(changes),encoding="utf8")
