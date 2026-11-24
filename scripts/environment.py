"""Record only the host and numerical-runtime facts used in these experiments."""
import json,platform,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
gpu=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,text=True)
d=dict(time=datetime.now(timezone.utc).isoformat(),system=platform.platform(),python=sys.version,python_path=sys.executable,
       torch=str(torch.__version__),numpy=np.__version__,gpu_available=torch.cuda.is_available(),gpu=gpu.stdout.strip(),
       cgroup_cpu=(Path("/sys/fs/cgroup/cpu.max").read_text().strip()),cgroup_memory=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
       disk_free_bytes=shutil.disk_usage(root).free,workdir=str(root),used_device="cpu",compute_note="Frozen archive and small fitted heads used CPU; no additional instances requested.")
with (root/"logs/environment.json").open("x",encoding="utf8") as file:json.dump(d,file,ensure_ascii=False,indent=2)
