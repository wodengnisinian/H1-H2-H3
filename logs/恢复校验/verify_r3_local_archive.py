"""Verify frozen R3 bytes and optionally add local navigation without replacing originals."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


STATUS = "logs/恢复校验/本地逐轮归档核验.json"
INPUTS = "datasets/inputs.json"
INPUT_ZIP = "datasets/归档输入.zip"
REGISTRY = "experiment_registry.jsonl"
ENVIRONMENT = "logs/environment.json"
REMOTE_INDEX = "logs/恢复校验/remote_files.json"
TRANSFER = "logs/恢复校验/r3-closed-20261124.zip"
FROZEN_REGISTRY = "logs/恢复校验/云端原始轮次登记.jsonl"
LOCAL_VERIFIER = "logs/恢复校验/verify_r3_local_archive.py"
ROUND_IDS = {f"R{i:02d}" for i in range(1, 14)}
LINK = re.compile(r"(?<=\]\()(<[^>\n]+>|[^)\s]+)(?=\))")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def joined(base, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            f"Invalid archive path {name!r}")
    parts = name.split("/")
    require(all(part not in {"", ".", ".."} for part in parts), f"Invalid archive path {name!r}")
    path = base.joinpath(*parts)
    require(path.resolve().is_relative_to(base.resolve()), f"Path outside archive: {path}")
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(path.read_bytes())


def check_hash(actual, expected, label):
    require(re.fullmatch(r"[0-9a-f]{64}", str(expected)) and actual == expected,
            f"SHA-256 mismatch for {label}: expected {expected}, found {actual}")


class Archive:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.additions = {}
        self.cloud_files = read_json(root / REMOTE_INDEX)["files"]

    def fingerprint(self, path, expected=None):
        if path not in self.cache:
            require(path.is_file(), f"Required file missing: {path}")
            with path.open("rb") as stream:
                self.cache[path] = {
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "bytes": os.fstat(stream.fileno()).st_size,
                }
        result = self.cache[path]
        cloud = self.cloud_files.get(path.relative_to(self.root).as_posix())
        if cloud is not None:
            check_hash(result["sha256"], cloud["sha256"], path)
            require(result["bytes"] == cloud["size"], f"Cloud file size mismatch: {path}")
        if expected is not None:
            check_hash(result["sha256"], expected, path)
        return result

    def ref(self, path, expected=None):
        return {"path": path.relative_to(self.root).as_posix(), **self.fingerprint(path, expected)}

    def add(self, path, content):
        require(path.resolve().is_relative_to(self.root), f"Outside archive: {path}")
        require(path not in self.additions, f"Duplicate planned file: {path}")
        if path.exists():
            require(path.is_file() and path.read_bytes() == content,
                    f"Existing file differs; retained without changes: {path}")
        self.additions[path] = content

    def finish(self, apply, verify):
        missing = [p for p in self.additions if not p.is_file()]
        if verify:
            require(not missing, f"Missing local overlay: {missing[:4]}")
        if apply:
            for path in missing:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Publish complete bytes atomically with create-only semantics.
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temp = Path(stream.name)
                    try:
                        stream.write(self.additions[path])
                    except BaseException:
                        stream.close()
                        temp.unlink(missing_ok=True)
                        raise
                try:
                    try:
                        os.link(temp, path)
                    except FileExistsError:
                        require(path.read_bytes() == self.additions[path], f"Concurrent conflict: {path}")
                finally:
                    temp.unlink(missing_ok=True)
            for path, content in self.additions.items():
                require(path.read_bytes() == content, f"Overlay verification failed: {path}")
        return len(missing)


def verify_inputs(archive):
    root = archive.root
    manifest = read_json(root / INPUTS)
    files = manifest["files"]
    zip_path = root / INPUT_ZIP
    zip_reference = archive.ref(zip_path, manifest["zip_sha256"])
    checked = {}
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if not name.startswith("inputs/"):
                name = "inputs/" + name
            joined(root, name)
            require(name in files and name not in checked, f"Unexpected or duplicate ZIP member {name}")
            expected = files[name]
            require(member.file_size == expected["bytes"], f"Size mismatch for ZIP member {name}")
            with z.open(member) as stream:
                check_hash(hashlib.file_digest(stream, "sha256").hexdigest(), expected["sha256"], name)
            checked[name] = {"member": member.filename, "bytes": member.file_size, "sha256": expected["sha256"]}
    require(set(checked) == set(files), f"Missing ZIP members: {set(files) - set(checked)}")
    return zip_reference, dict(sorted(checked.items()))


def verify_registry(archive):
    transfer = archive.root / TRANSFER
    transfers = read_json(archive.root / REMOTE_INDEX)["archives"]
    archive.fingerprint(transfer, transfers[transfer.name])
    with zipfile.ZipFile(transfer) as z:
        frozen = z.read(REGISTRY)
    digest = hashlib.sha256(frozen).hexdigest()
    check_hash(digest, archive.cloud_files[REGISTRY]["sha256"], "original event registry")
    archive.add(archive.root / FROZEN_REGISTRY, frozen)
    archive.registry_ref = {"path": FROZEN_REGISTRY, "bytes": len(frozen), "sha256": digest}
    original_rows = list(map(json.loads, frozen.splitlines()))
    live_rows = map(json.loads, (archive.root / REGISTRY).read_bytes().splitlines())
    require(original_rows == [r for r in live_rows if r.get("id") in ROUND_IDS],
            "Original R01–R13 events in the live registry differ from the frozen transfer")
    starts, ends = {}, {}
    for row in original_rows:
        rid = row["id"]
        require(rid in ROUND_IDS, f"Unexpected round {rid}")
        if row["event"] == "start":
            require(rid not in starts, f"Duplicate start: {rid}")
            starts[rid] = row
        else:
            require(row["event"] == "end" and rid in starts and rid not in ends and row["code"] == 0,
                    f"Invalid completion event for {rid}")
            for key in ("path", "command", "protocol_sha256", "source_sha256", "input_manifest_sha256"):
                require(row[key] == starts[rid][key], f"Start/end differ for {rid}: {key}")
            require(row["utc"] >= starts[rid]["utc"], f"Run ends before start: {rid}")
            require(re.fullmatch(r"experiments/\d{2}-" + rid, row["path"]), f"Invalid round path: {rid}")
            ends[rid] = row
    require(set(ends) == ROUND_IDS, f"Missing completed rounds: {ROUND_IDS - set(ends)}")
    require(all(joined(archive.root, row["path"]).is_dir() for row in ends.values()),
            "A registered original round directory is missing")
    return starts, ends


def relocated_report(archive, source, destination):
    def relocate(match):
        target = match.group().strip("<>")
        href = urlsplit(target)
        if href.scheme or href.netloc or not href.path:
            return match.group()
        path = (source.parent / unquote(href.path)).resolve()
        require(path.is_relative_to(archive.root) and path.is_file(), f"Broken report link: {source} -> {target}")
        remapped = Path(os.path.relpath(path, destination.parent)).as_posix()
        return "<" + remapped + ("?" + href.query if href.query else "") + ("#" + href.fragment if href.fragment else "") + ">"

    return LINK.sub(relocate, source.read_text(encoding="utf-8")).encode("utf-8")


def verify_round(archive, begin, end, inputs, zip_reference, ends):
    root = archive.root
    rid = end["id"]
    folder = joined(root, end["path"])
    seal = folder / "artifacts.json"
    seal_ref = archive.ref(seal, end["artifact_manifest_sha256"])
    input_ref = archive.ref(root / INPUTS, end["input_manifest_sha256"])
    original = {name: archive.ref(joined(folder, name), digest) for name, digest in read_json(seal).items()}
    protocol = folder / "preregistration.json"
    protocol_ref = archive.ref(protocol, end["protocol_sha256"])
    require(read_json(protocol)["id"] == rid, f"Mismatched preregistration for {rid}")
    require("results/" + end["result_file"] in original, f"Missing sealed result for {rid}")
    resources = read_json(folder / "resources.json")
    require(all(end[k] == v for k, v in resources.items()), f"Registered resources differ for {rid}")
    source_refs, config_refs = [], [protocol_ref]
    for name, digest in end["source_sha256"].items():
        snapshot = joined(folder / "snapshot", name)
        require("snapshot/" + name in original, f"Snapshot outside seal for {rid}: {name}")
        reference = archive.ref(snapshot, digest)
        if name.startswith("configs/"):
            config_refs.append(reference)
            output = joined(folder, name)
        else:
            source_refs.append(reference)
            output = joined(folder / "code", name)
        archive.add(output, snapshot.read_bytes())
    env_ref = archive.ref(root / ENVIRONMENT)
    patch = archive.ref(root / "analysis/diffs" / (folder.name + ".patch"))
    archive.add(folder / "code/inventory.json", json_bytes(
        {"path_base": "R3", "registered_sources": source_refs, "changes": patch, "runtime_dependencies": env_ref}))
    archive.add(folder / "configs/inventory.json", json_bytes(
        {"path_base": "R3", "registered_configurations": config_refs}))

    require(all(cp in original for cp in end["checkpoints"]), f"Unsealed checkpoint for {rid}")
    argv = read_json(protocol)["argv"]
    upstream_paths = [a[len("{root}/"):] for a in argv if a.startswith("{root}/experiments/")]
    upstream = []
    for result_path in dict.fromkeys(upstream_paths):
        dependency = next(r for r in ends.values() if r["path"] + "/results" == result_path)
        require(dependency["utc"] <= begin["utc"], f"Future dependency for {rid}: {result_path}")
        upstream.extend(archive.ref(joined(root, dependency["path"] + "/" + name)) for name in dependency["checkpoints"])
    input_prefixes = [a.replace("{inputs}", "inputs") + "/" for a in argv if a.startswith("{inputs}")]
    input_members = [v for k, v in inputs.items() if any(k.startswith(p) for p in input_prefixes)]
    data = [ref for name, ref in original.items() if name.startswith("results/") and
            re.search(r"input|gold|label|feature|dataset|cohort|provenance|entropy", name, re.I)]
    archive.add(folder / "data/inventory.json", json_bytes({
        "path_base": "R3", "shared_input_manifest": input_ref, "shared_archive": zip_reference,
        "declared_archive_prefixes": input_prefixes, "available_members_within_prefixes": input_members,
        "round_data_files": data, "upstream_result_directories": upstream_paths}))
    checkpoints = {
        "path_base": "R3", "produced": [original[cp] for cp in end["checkpoints"]],
        "upstream_result_checkpoints": upstream,
        "weights_available_within_declared_archive_prefixes": [v for v in input_members if v["member"].endswith(".pt")],
        "shared_archive_for_weights": zip_reference, "fitted_parameter_records":
            [ref for name, ref in original.items() if PurePosixPath(name).name.startswith("parameters-")]}
    archive.add(folder / "checkpoints/inventory.json", json_bytes(checkpoints))
    run_logs = [ref for name, ref in original.items() if re.search(r"stdout|stderr|events\.jsonl|losses\.jsonl|timeline\.jsonl|resources\.json", name)]
    archive.add(folder / "logs/inventory.json", json_bytes({
        "path_base": "R3", "execution_logs": run_logs, "archived_environment_and_resource_limits": env_ref,
        "registry": archive.registry_ref, "start_utc": begin["utc"], "end_utc": end["utc"],
        "command": end["command"], "exit_code": end["code"]}))

    report = root / "reports" / (rid + ".md")
    archive.add(folder / "report.md", relocated_report(archive, report, folder / "report.md"))
    indices = {p.relative_to(folder).as_posix(): {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
               for p, content in archive.additions.items() if p.is_relative_to(folder)}
    archive.add(folder / "manifest.json", json_bytes({
        "id": rid, "path_base": "R3", "registry": archive.registry_ref, "source_report": archive.ref(report),
        "original_artifact_seal": seal_ref, "registered_artifacts": original,
        "shared_input_manifest": input_ref, "shared_input_archive": zip_reference,
        "local_navigation_files_relative_to_round": indices}))
    return {"id": rid, "path": end["path"], "registered_files": len(original), "artifact_seal": seal_ref,
            "verified_artifact_bytes": sum(ref["bytes"] for ref in original.values()),
            "checks": ["start/end", "artifacts", "source", "protocol", "inputs", "checkpoints", "report_links"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="add verified local navigation with create-only writes")
    modes.add_argument("--verify", action="store_true", help="require and verify all local navigation files")
    args = ap.parse_args()
    archive = Archive(args.root.resolve(strict=True))
    archive.add(archive.root / LOCAL_VERIFIER, Path(__file__).read_bytes())
    zip_ref, members = verify_inputs(archive)
    starts, ends = verify_registry(archive)
    rounds = [verify_round(archive, starts[rid], end, members, zip_ref, ends)
              for rid, end in sorted(ends.items(), key=lambda item: item[1]["path"])]
    for row in rounds:
        path = joined(archive.root, row["path"]) / "manifest.json"
        content = archive.additions[path]
        row["local_manifest"] = {"path": path.relative_to(archive.root).as_posix(), "bytes": len(content),
                                 "sha256": hashlib.sha256(content).hexdigest()}
    status = {"schema": 1, "rounds": rounds, "input_archive": zip_ref, "verified_input_members": len(members),
              "verified_input_uncompressed_bytes": sum(v["bytes"] for v in members.values()),
              "registry": archive.registry_ref, "input_manifest": archive.ref(archive.root / INPUTS),
              "original_transfer": archive.ref(archive.root / TRANSFER),
              "cloud_file_index": archive.ref(archive.root / REMOTE_INDEX),
              "environment": archive.ref(archive.root / ENVIRONMENT), "local_navigation_files": len(archive.additions)}
    archive.add(archive.root / STATUS, json_bytes(status))
    missing = archive.finish(args.apply, args.verify)
    print(json.dumps({"rounds": len(rounds), "verified_round_artifacts": sum(r["registered_files"] for r in rounds),
                      "verified_input_members": len(members), "navigation_files": len(archive.additions),
                      "new_files_added": missing if args.apply else 0, "missing_after": 0 if args.apply else missing,
                      "status_path": STATUS}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, StopIteration) as exc:
        sys.exit(f"Archive verification failed: {exc}")
