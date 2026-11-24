"""Read-only integrity check for the R14 cloud run and its local archive."""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


STAGED = {
    "r3_dev_order_audit.py": "code/r3_dev_order_audit.py",
    "record_resources.py": "code/record_resources.py",
    "protocol.json": "configs/protocol.json",
    "frozen-inputs.json": "data/frozen-inputs.json",
}
REMOTE_RECEIPTS = {"timing.txt", "remote.log", "exit.txt"}
REMOTE_RESULTS = {
    "results/accuracy.csv", "results/retrieval.csv", "results/paired_contrasts.csv",
    "results/graphs.jsonl.gz", "results/predictions.jsonl.gz", "results/summary.json",
}


def require(test, message):
    if not test:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique_object)


class Round:
    def __init__(self, folder):
        self.folder = folder
        self.stamps = {}

    def file(self, name):
        require(isinstance(name, str) and all(s not in {"", ".", ".."} for s in name.split("/"))
                and "\\" not in name and ":" not in name, f"Invalid archive path: {name!r}")
        path = self.folder.joinpath(*name.split("/"))
        require(path.resolve().is_relative_to(self.folder) and path.is_file(), f"Missing or external file: {name}")
        return path

    def document(self, name):
        return parse(self.file(name).read_bytes())

    def check(self, name, expected, require_size=False):
        if name not in self.stamps:
            path = self.file(name)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.stamps[name] = {"sha256": digest, "bytes": path.stat().st_size}
        actual = self.stamps[name]
        if isinstance(expected, dict):
            if require_size or "bytes" in expected:
                require(type(expected.get("bytes")) is int and expected["bytes"] == actual["bytes"],
                        f"Byte count mismatch: {name}")
            expected = expected["sha256"]
        else:
            require(not require_size, f"Missing byte count: {name}")
        require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected) and
                actual["sha256"] == expected, f"SHA-256 mismatch: {name}")
        return actual


def registry(round_):
    root = round_.folder.parents[1]
    relative = round_.folder.relative_to(root).as_posix()
    require(round_.folder.parent.name == "experiments", "Expected a round inside experiments/")
    records = [parse(line) for line in (root / "experiment_registry.jsonl").read_bytes().splitlines() if line.strip()]
    events = [row for row in records if row.get("path") == relative or row.get("id") == "R14"]
    require([row.get("event") for row in events] == ["start", "end"] and
            all(row.get("id") == "R14" and row.get("path") == relative for row in events),
            "Expected exactly one matching R14 start and end pair in order")
    start, end = events
    require(round_.document("logs/start.json") == start, "Start receipt and registry row differ")
    require(type(end.get("code")) is int and end["code"] == 0, "R14 end event lacks successful exit")
    if "status" in end:
        require(end["status"] in {"verified", "complete", "completed", "success", "succeeded"},
                "R14 registered status is unsuccessful")
    require(datetime.fromisoformat(start["utc"]) <= datetime.fromisoformat(end["utc"]),
            "Registry end precedes start")
    for field in ("command", "protocol_sha256", "source_sha256", "input_manifest_sha256"):
        if field in end:
            require(end[field] == start[field], f"Registered {field} changed between start and end")
    require(start["command"] and start["source_sha256"], "Missing registered command or sources")
    round_.check("configs/protocol.json", start["protocol_sha256"])
    round_.check("data/frozen-inputs.json", start["input_manifest_sha256"])
    for source, digest in start["source_sha256"].items():
        round_.check(source, digest)
    round_.check("artifacts.json", end["artifact_manifest_sha256"])
    return start, end


def local_inventory(round_, end):
    manifest = round_.document("manifest.json")
    require(manifest.get("status") == "verified", "Local manifest does not declare verified status")
    entries = manifest["files"]
    require(isinstance(entries, dict) and entries, "Local manifest contains no inventory")
    actual_files = {p.relative_to(round_.folder).as_posix() for p in round_.folder.rglob("*") if p.is_file()}
    exempt = {"manifest.json", "artifacts.json"}
    require(actual_files - set(entries) <= exempt,
            f"Files missing from local manifest: {sorted(actual_files - set(entries) - exempt)}")
    for name, entry in entries.items():
        round_.check(name, entry, require_size=True)
    artifact_document = round_.document("artifacts.json")
    artifacts = artifact_document.get("files", artifact_document)
    require(isinstance(artifacts, dict) and artifacts, "Empty registered artifact inventory")
    for name, entry in artifacts.items():
        round_.check(name, entry)
    if "manifest_sha256" in end:
        round_.check("manifest.json", end["manifest_sha256"])
    require("logs/remote-files.json" in artifacts or "manifest.json" in artifacts or "manifest_sha256" in end,
            "Remote result checksums lack a binding to the registered end event")
    return len(entries), len(artifacts)


def remote_outputs(round_, start):
    remote = round_.document("logs/remote-files.json")
    require(isinstance(remote, dict), "Invalid remote file ledger")
    required = REMOTE_RESULTS | REMOTE_RECEIPTS | set(STAGED)
    require(required <= set(remote), f"Missing remote entries: {sorted(required - set(remote))}")
    results = []
    for name, digest in remote.items():
        if name.startswith("results/"):
            round_.check(name, digest)
            results.append(name)
        elif name in REMOTE_RECEIPTS:
            round_.check("logs/" + name, digest)
        elif name in STAGED:
            round_.check(STAGED[name], digest)
    timing = round_.document("logs/timing.txt")
    exit_receipt = round_.document("logs/exit.json")
    require(round_.file("logs/exit.txt").read_text(encoding="ascii").strip() == "0",
            "Remote process did not record exit code 0")
    require(all(type(r.get("exit_code")) is int and r["exit_code"] == 0 for r in (timing, exit_receipt)),
            "Timing record or SSH exit receipt lacks exit code 0")
    require(exit_receipt["command"] == start["command"], "SSH exit command differs from frozen command")
    for fields, zero_ok in ((("elapsed_seconds", "max_resident_kib"), False),
                            (("user_cpu_seconds", "system_cpu_seconds"), True)):
        for key in fields:
            value = timing[key]
            require(type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero_ok else value > 0),
                    f"Invalid time/resource field: {key}")
    return {"downloaded_results": len(results), "remote_sealed_entries": len(remote),
            "remote_elapsed_seconds": timing["elapsed_seconds"], "max_resident_kib": timing["max_resident_kib"],
            "remote_exit_code": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_directory", type=Path)
    folder = parser.parse_args().round_directory.resolve(strict=True)
    round_ = Round(folder)
    start, end = registry(round_)
    local_files, registered_files = local_inventory(round_, end)
    result = remote_outputs(round_, start)
    print(json.dumps({"id": "R14", "status": "verified", "local_manifest_files": local_files,
                      "registered_artifacts": registered_files,
                      "artifact_manifest_sha256": end["artifact_manifest_sha256"], **result}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        sys.exit(f"R14 verification failed: {error}")
