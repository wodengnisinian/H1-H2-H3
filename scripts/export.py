"""Create a checksummed study export on the server, preserving stored source and failure artifacts."""
import argparse,json,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--out",required=True,type=Path);p.add_argument("--completed-only",action="store_true")
a=p.parse_args();root=Path(__file__).resolve().parents[1]
if a.out.is_relative_to(root) or a.out.exists():raise ValueError("Use a new path outside the study")
if a.completed_only:
    reg=[json.loads(s) for s in (root/"experiment_registry.jsonl").read_text().splitlines()]
    files=sorted({f for r in reg if r["event"]=="end" for f in (root/r["path"]).rglob("*") if f.is_file()})
else:files=sorted(f for f in root.rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.name!=".loop.lock")
with ZipFile(a.out,"x",ZIP_DEFLATED,compresslevel=2) as z:
    for f in files:z.write(f,f.relative_to(root).as_posix())
with a.out.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
a.out.with_suffix(".sha256").write_text(h+"\n")
print(len(files),h,flush=True)
