"""Execute one frozen B13 study. No data or model selection takes place here."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(rel):
    return json.loads((ROOT / rel).read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def check_freeze(frozen):
    for name, expected in frozen["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Changed frozen input {name}")


def phase(name, args, deadline):
    stage = LEDGER / name
    stage.mkdir(exist_ok=True)
    earlier = list(stage.glob("*/finish.json"))
    if any(json.loads(f.read_bytes())["returncode"] == 0 for f in earlier):
        return
    if name == "assessment" and (ROOT / "assessment/opening.json").exists():
        raise RuntimeError("The single opening has already been consumed")
    now = time.time()
    if now >= deadline:
        raise TimeoutError("Study budget expired")
    trial = stage / f"{len(list(stage.iterdir()))+1:03}"
    trial.mkdir()
    command = [sys.executable, "-B", "-u", "-X", "utf8", *map(str, args)]
    emit(trial / "start.json", {"started_utc": utc(), "argv": command})
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (trial / "stdout.log").open("xb") as out, (trial / "stderr.log").open("xb") as err:
        try:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=deadline-now).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    emit(trial / "finish.json", {"finished_utc": utc(), "returncode": rc})
    if rc:
        raise RuntimeError(f"{name} failed with return code {rc}; see {trial}")


def main():
    freeze = ROOT / "冻结清单.json"
    protocol = ROOT / "实验协议.json"
    frozen = read("冻结清单.json")
    check_freeze(frozen)
    LEDGER.mkdir(exist_ok=True)
    start_path = LEDGER / "started.json"
    if not start_path.exists():
        emit(start_path, {"started_utc": utc(), "deadline_unix": time.time()+90*60, "freeze_sha256": digest(freeze)})
    started = read("ledger/started.json")
    if started["freeze_sha256"] != digest(freeze):
        raise ValueError("Freeze differs from original start")
    deadline = started["deadline_unix"]
    phase("generation", ["code/generate_cohort.py", "--root", ROOT], deadline)
    cohortseal = ROOT / "cohort/seal.json"
    cohort = read("cohort/seal.json")
    if cohort["features_sha256"] != digest(ROOT / "cohort/features.jsonl"):
        raise ValueError("Changed feature data")
    if cohort["freeze_sha256"] != digest(freeze) or cohort["protocol_sha256"] != digest(protocol):
        raise ValueError("Cohort does not match declared freeze")
    phase("prediction", ["code/predict_sealed.py", "--features", "cohort/features.jsonl",
                         "--features-sha256", cohort["features_sha256"], "--cohort-seal-sha256", digest(cohortseal),
                         "--protocol-sha256", digest(protocol), "--freeze-sha256", digest(freeze),
                         "--b11", "inputs/server-run", "--b12-code", "inputs/b12-code", "--out", "predictions"], deadline)
    check_freeze(frozen)
    prediction_seal = read("predictions/prediction-seal.json")
    if len(prediction_seal["prediction_sha256"]) != 7:
        raise ValueError("Expected seven predictions")
    for name, expected in prediction_seal["prediction_sha256"].items():
        if digest(ROOT / "predictions" / name) != expected:
            raise ValueError(f"Unsealed prediction {name}")
    phase("assessment", ["code/score_once.py", "--root", ROOT], deadline)
    if not (LEDGER / "terminal.json").exists():
        emit(LEDGER / "terminal.json", {"study_completed": True, "finished_utc": utc(), "formal_openings": 1,
                                      "score_sha256": digest(ROOT / "assessment/score.json")})
    print((LEDGER / "terminal.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
