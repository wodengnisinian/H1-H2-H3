"""Run the exact bounded argument vector and record Linux child resource use."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

if __name__=="__main__":
    start=time.monotonic()
    result=subprocess.run(sys.argv[1:],check=False)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    with Path("timing.txt").open("x",encoding="utf8") as dst:
        json.dump(dict(exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,
                       user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime,
                       max_resident_kib=usage.ru_maxrss),dst,indent=2)
    raise SystemExit(result.returncode)
