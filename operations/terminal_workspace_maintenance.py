#!/usr/bin/env python3
"""Back up a validated natural terminal's two workspaces; delete only after local verification.

The recovery solver, its budgets and all compact execution artifacts are untouched.
This tool is an independent administrative lifecycle. File bodies are hashed and
copied opaquely, never interpreted. Non-natural or historical reused outcomes fail
the admission gate. An infrastructure-interrupted active question is never admitted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMESPACE = "strict_recovery_v1"
VERSION = "terminal-workspace-maintenance-v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def checked(value, field):
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), field + " differs")


def load(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def safe(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative, "Invalid relative path")
    p = PurePosixPath(relative)
    require(p.parts and not p.is_absolute() and ".." not in p.parts and p.as_posix() == relative,
            "Unsafe relative path")
    root = Path(root).resolve()
    target = root.joinpath(*p.parts)
    require(target.resolve().is_relative_to(root), "Path escapes its root")
    current = target
    while current != root:
        require(not current.is_symlink(), "Symlink in administrative path")
        current = current.parent
    return target


def fingerprint(s):
    require(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode), "Only regular files/directories are admitted")
    return {"type": "file" if stat.S_ISREG(s.st_mode) else "dir", "dev": s.st_dev,
            "inode": s.st_ino, "bytes": s.st_size, "mode": stat.S_IMODE(s.st_mode),
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns, "nlink": s.st_nlink}


def stable_stat(value):
    # Other active questions legitimately add/remove links to the same corpus
    # inode. These fields are recorded for provenance but are not content drift.
    return {key: item for key, item in value.items() if key not in {"ctime_ns", "nlink"}}


def stable_entries(rows):
    return [{**{key: value for key, value in row.items() if key not in {"sha256", "stat"}},
             "stat": stable_stat(row["stat"])} for row in rows]


def content_marker(value):
    # ctime is only a conservative hash-cache invalidator, never an admission
    # condition. A link-count change causes a fresh content hash, not rejection.
    return tuple(value[key] for key in ("dev", "inode", "bytes", "mtime_ns", "ctime_ns"))


def verify_open_file(descriptor, row, cache):
    before = fingerprint(os.fstat(descriptor))
    require(stable_stat(before) == stable_stat(row["stat"]) and before["type"] == "file",
            "Workspace file identity changed")
    marker = content_marker(before)
    if cache.get(marker) != row["sha256"]:
        h = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while data := os.read(descriptor, 8 << 20):
            h.update(data)
        after = fingerprint(os.fstat(descriptor))
        require(stable_stat(after) == stable_stat(before) and h.hexdigest() == row["sha256"],
                "Workspace content no longer matches the verified local backup")
        cache[content_marker(after)] = row["sha256"]
        return after
    return before


def audit_terminal(root, arm, qid):
    """Immutable natural-terminal publication is the per-question completion barrier."""
    from operations.recovery_suite import RecoverySuite
    from strict_ablations import runner
    require(arm in {"auto_rerank", "search_paging"} and type(qid) is int and qid > 0,
            "An explicit authorized arm and numeric question are required")
    suite = RecoverySuite(root)
    require(suite.directory == NAMESPACE, "Only the current recovery namespace is admitted")
    manifest = suite.verify_manifest(resources=False)
    query = next((q for q in runner.shared.validate_queries() if q["query_id"] == qid), None)
    require(query is not None, "Question is outside the frozen cohort")
    outcome_path = suite.reports / "outcomes" / arm / f"qid_{qid}.json"
    outcome = load(outcome_path)
    checked(outcome, "outcome_sha256")
    require(outcome.get("kind") == "natural_terminal" and outcome.get("arm") == arm
            and outcome.get("query_id") == qid and outcome.get("manifest_sha256") == manifest["manifest_sha256"],
            "Only a verified natural terminal may release a workspace")
    path = suite.terminal_path(arm, qid)
    require(path == Path(root).resolve() / "runs" / NAMESPACE / arm / "_per_query" / f"qid_{qid}.json",
            "Historical/reused terminals cannot authorize current workspace deletion")
    terminal = load(path)
    effective = suite.effective_manifest(manifest, arm, qid)
    suite.validate_terminal(terminal, effective, arm, query)
    suite.verify_execution_artifacts(effective, arm, query, terminal)
    require(suite.validate_outcome(manifest, arm, query) == outcome,
            "Selected outcome/accounting changed")
    require(outcome.get("terminal_sha256") == file_sha(path)
            and outcome.get("final_sha256") == terminal["final_sha256"], "Outcome does not bind the natural terminal")
    return {"source_root": str(Path(root).resolve()), "namespace": NAMESPACE, "arm": arm, "query_id": qid,
            "manifest_sha256": manifest["manifest_sha256"], "effective_config_sha256": effective["config_sha256"],
            "terminal_sha256": file_sha(path), "row_sha256": terminal["row_sha256"],
            "outcome_sha256": outcome["outcome_sha256"], "outcome_file_sha256": file_sha(outcome_path),
            "execution_artifacts": terminal["execution_artifacts"]}


@contextmanager
def operation_lock(root, arm, qid):
    require(arm in {"auto_rerank", "search_paging"} and type(qid) is int and qid > 0,
            "An explicit authorized arm and numeric question are required")
    path = safe(root, f"reports/{NAMESPACE}/workspace_maintenance_locks/{arm}_{qid}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "Aliased maintenance lock")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def targets(root, arm, qid):
    return {name: safe(root, f"runs/{NAMESPACE}/{arm}/{directory}/qid_{qid}")
            for name, directory in (("working", "_working"), ("runtime", "_rerank_runtime"))}


def scan(root, arm, qid):
    roots, entries = {}, []
    for key, target in targets(root, arm, qid).items():
        roots[key] = {"path": target.relative_to(Path(root).resolve()).as_posix(), "exists": target.exists()}
        if not target.exists():
            continue
        root_stat = fingerprint(target.lstat())
        require(root_stat["type"] == "dir", "Workspace root is not a directory")
        roots[key]["parent_identity"] = {k: fingerprint(target.parent.stat())[k] for k in ("dev", "inode")}
        entries.append({"member": key, "root_key": key, "relative": "", "stat": root_stat})
        for parent, dirs, files in os.walk(target, followlinks=False):
            dirs.sort(); files.sort()
            for name in dirs + files:
                path = Path(parent) / name
                relative = path.relative_to(target).as_posix()
                safe(target, relative)
                info = fingerprint(path.lstat())
                require(info["dev"] == root_stat["dev"], "Cross-device workspace entry")
                entries.append({"member": key + "/" + relative, "root_key": key, "relative": relative, "stat": info})
    return roots, entries


class HashingReader:
    def __init__(self, stream):
        self.stream, self.sha, self.bytes = stream, hashlib.sha256(), 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.sha.update(data)
        self.bytes += len(data)
        return data


def export_workspace(root, arm, qid, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed = root / "reports" / NAMESPACE / "workspace_backups"
    require(output.is_relative_to(allowed) and output != allowed and not output.exists(),
            "Use a fresh directory inside reports/strict_recovery_v1/workspace_backups")
    with operation_lock(root, arm, qid):
        binding = audit_terminal(root, arm, qid)
        source_roots, entries = scan(root, arm, qid)
        require(entries, "Both terminal workspaces are already absent")
        output.mkdir(parents=True, exist_ok=False)
        archive = output / "workspace.tar.gz"
        with archive.open("xb") as stream:
            os.chmod(archive, 0o600)
            with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as tar:
                for row in entries:
                    target = targets(root, arm, qid)[row["root_key"]]
                    path = target if not row["relative"] else safe(target, row["relative"])
                    require(stable_stat(fingerprint(path.lstat())) == stable_stat(row["stat"]), "Workspace changed before backup")
                    info = tarfile.TarInfo(row["member"])
                    info.mode, info.mtime = row["stat"]["mode"], row["stat"]["mtime_ns"] / 1e9
                    if row["stat"]["type"] == "dir":
                        info.type = tarfile.DIRTYPE
                        tar.addfile(info)
                    else:
                        info.size = row["stat"]["bytes"]
                        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(descriptor, "rb") as original:
                            require(stable_stat(fingerprint(os.fstat(original.fileno()))) == stable_stat(row["stat"]), "Opened source identity changed")
                            reader = HashingReader(original)
                            tar.addfile(info, reader)
                            require(reader.bytes == info.size and stable_stat(fingerprint(os.fstat(original.fileno()))) == stable_stat(row["stat"]),
                                    "Source changed while being backed up")
                            row["sha256"] = reader.sha.hexdigest()
                now_roots, now_entries = scan(root, arm, qid)
                require(now_roots == source_roots and stable_entries(now_entries) == stable_entries(entries),
                        "Workspace tree changed while being backed up")
                require(audit_terminal(root, arm, qid) == binding, "Natural terminal changed during backup")
                manifest = {"version": VERSION, "binding": binding, "roots": source_roots, "entries": entries,
                    "files": sum(row["stat"]["type"] == "file" for row in entries),
                    "directories": sum(row["stat"]["type"] == "dir" for row in entries),
                    "logical_bytes": sum(row["stat"]["bytes"] for row in entries if row["stat"]["type"] == "file")}
                manifest["manifest_sha256"] = digest(manifest)
                data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
                info = tarfile.TarInfo("workspace_manifest.json")
                info.size, info.mode = len(data), 0o600
                tar.addfile(info, io.BytesIO(data))
        manifest_path = output / "workspace_manifest.json"
        with manifest_path.open("xb") as stream:
            os.chmod(manifest_path, 0o600)
            stream.write(data)
        receipt = {"version": VERSION, "binding": binding, "created_unix": time.time(),
            "archive": {"sha256": file_sha(archive), "bytes": archive.stat().st_size},
            "manifest": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
            "files": manifest["files"], "directories": manifest["directories"],
            "logical_bytes": manifest["logical_bytes"], "local_download_required_before_delete": True}
        receipt["receipt_sha256"] = digest(receipt)
        receipt_path = output / "receipt.json"
        write_new(receipt_path, receipt)
        # Read the completed archive back, checking all members independently.
        verify_archive(archive, receipt_path)
    return {"exported": True, "archive": str(archive), "receipt": str(receipt_path),
            "receipt_file_sha256": file_sha(receipt_path), "receipt_bytes": receipt_path.stat().st_size,
            **receipt, "deleted": False}


def verify_archive(archive, receipt_path, *, expected_receipt_sha256=None, expected_receipt_bytes=None):
    archive, receipt_path = Path(archive), Path(receipt_path)
    if expected_receipt_sha256 is not None:
        require(file_sha(receipt_path) == expected_receipt_sha256, "Independent receipt SHA differs")
    if expected_receipt_bytes is not None:
        require(receipt_path.stat().st_size == expected_receipt_bytes, "Independent receipt byte count differs")
    receipt = load(receipt_path)
    checked(receipt, "receipt_sha256")
    require(receipt.get("version") == VERSION and receipt.get("local_download_required_before_delete") is True,
            "Unexpected workspace receipt")
    require(archive.stat().st_size == receipt["archive"]["bytes"] and file_sha(archive) == receipt["archive"]["sha256"],
            "Downloaded archive SHA/bytes differ")
    observed, manifest_bytes = {}, None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            safe(Path("/unused_workspace_root"), member.name)
            require(member.name not in observed and (member.isfile() or member.isdir())
                    and not member.issym() and not member.islnk(), "Duplicate or unsafe archive member")
            if member.name == "workspace_manifest.json":
                require(member.isfile() and manifest_bytes is None, "Duplicate/invalid workspace manifest")
                manifest_bytes = tar.extractfile(member).read()
                observed[member.name] = None
            elif member.isfile():
                h, size = hashlib.sha256(), 0
                with tar.extractfile(member) as stream:
                    for data in iter(lambda: stream.read(8 << 20), b""):
                        h.update(data); size += len(data)
                require(size == member.size, "Archive file length differs")
                observed[member.name] = {"type": "file", "bytes": size, "sha256": h.hexdigest(), "mode": member.mode}
            else:
                require(member.size == 0, "Directory member contains a payload")
                observed[member.name] = {"type": "dir", "mode": member.mode}
    require(manifest_bytes is not None and len(manifest_bytes) == receipt["manifest"]["bytes"]
            and hashlib.sha256(manifest_bytes).hexdigest() == receipt["manifest"]["sha256"], "Manifest bytes differ")
    manifest = json.loads(manifest_bytes)
    checked(manifest, "manifest_sha256")
    require(manifest.get("version") == VERSION and manifest.get("binding") == receipt["binding"], "Manifest binding differs")
    entries = manifest["entries"]
    require(len({row["member"] for row in entries}) == len(entries)
            and set(observed) == {row["member"] for row in entries} | {"workspace_manifest.json"}, "Archive membership differs")
    for row in entries:
        require(row["root_key"] in {"working", "runtime"} and row["member"] ==
                row["root_key"] + ("/" + row["relative"] if row["relative"] else ""), "Workspace member mapping differs")
        expected = {"type": row["stat"]["type"], "mode": row["stat"]["mode"]}
        if expected["type"] == "file":
            expected.update(bytes=row["stat"]["bytes"], sha256=row["sha256"])
        require(observed[row["member"]] == expected, "Archive file bytes/metadata differ")
    require(manifest["files"] == receipt["files"] == sum(row["stat"]["type"] == "file" for row in entries)
            and manifest["directories"] == receipt["directories"] == sum(row["stat"]["type"] == "dir" for row in entries)
            and manifest["logical_bytes"] == receipt["logical_bytes"] == sum(row["stat"]["bytes"] for row in entries
                 if row["stat"]["type"] == "file"), "Workspace totals differ")
    acknowledgement = {"version": VERSION, "kind": "local_workspace_download_verified", "verified": True,
        "binding": receipt["binding"], "archive": receipt["archive"], "manifest": receipt["manifest"],
        "receipt_file_sha256": file_sha(receipt_path), "receipt_bytes": receipt_path.stat().st_size,
        "files": receipt["files"], "directories": receipt["directories"], "verified_unix": time.time()}
    acknowledgement["acknowledgement_sha256"] = digest(acknowledgement)
    return acknowledgement


def delete_workspace(root, arm, qid, receipt_path, acknowledgement):
    root, receipt_path = Path(root).resolve(), Path(receipt_path).resolve()
    require(receipt_path.is_relative_to(root / "reports" / NAMESPACE / "workspace_backups"), "Receipt outside current backups")
    safe(root, receipt_path.relative_to(root).as_posix())
    checked(acknowledgement, "acknowledgement_sha256")
    require(acknowledgement.get("kind") == "local_workspace_download_verified" and acknowledgement.get("verified") is True,
            "A successful local download verification is mandatory")
    receipt = load(receipt_path)
    checked(receipt, "receipt_sha256")
    require(file_sha(receipt_path) == acknowledgement.get("receipt_file_sha256")
            and receipt_path.stat().st_size == acknowledgement.get("receipt_bytes")
            and all(receipt[key] == acknowledgement.get(key) for key in ("binding", "archive", "manifest")),
            "Local verification does not authorize this backup")
    manifest_path = receipt_path.parent / "workspace_manifest.json"
    require(file_sha(manifest_path) == receipt["manifest"]["sha256"]
            and manifest_path.stat().st_size == receipt["manifest"]["bytes"], "Deletion manifest changed")
    manifest = load(manifest_path)
    checked(manifest, "manifest_sha256")
    with operation_lock(root, arm, qid):
        binding = audit_terminal(root, arm, qid)
        require(binding == receipt["binding"] == manifest["binding"], "Natural terminal changed or wrong question")
        now_roots, now_entries = scan(root, arm, qid)
        require(now_roots == manifest["roots"] and stable_entries(now_entries) == stable_entries(manifest["entries"]),
                "Full workspace tree/stat differs; deletion refused before any unlink")
        lookup = {row["member"]: row for row in manifest["entries"]}
        content_cache, removed_files, removed_dirs = {}, 0, 0
        # Hash all current file bytes before the first deletion. A late drift
        # discovered here cannot leave a partly removed workspace.
        for row in manifest["entries"]:
            if row["stat"]["type"] != "file":
                continue
            path = safe(targets(root, arm, qid)[row["root_key"]], row["relative"])
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                verify_open_file(descriptor, row, content_cache)
            finally:
                os.close(descriptor)
        for key, target in targets(root, arm, qid).items():
            if not manifest["roots"][key]["exists"]:
                continue
            root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                anchor = lookup[key]["stat"]
                require(stable_stat(fingerprint(os.fstat(root_fd))) == stable_stat(anchor), "Opened workspace root differs")

                def parent_fd(parts):
                    fd, built = os.dup(root_fd), []
                    try:
                        for part in parts:
                            built.append(part)
                            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                            info, expected = fingerprint(os.fstat(nxt)), lookup[key + "/" + "/".join(built)]["stat"]
                            require(info["type"] == "dir" and (info["dev"], info["inode"]) ==
                                    (expected["dev"], expected["inode"]), "Parent directory identity differs")
                            os.close(fd); fd = nxt
                        return fd
                    except BaseException:
                        os.close(fd); raise

                for row in manifest["entries"]:
                    if row["root_key"] != key or row["stat"]["type"] != "file":
                        continue
                    path = PurePosixPath(row["relative"])
                    fd = parent_fd(path.parts[:-1])
                    try:
                        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            info = verify_open_file(descriptor, row, content_cache)
                            identity = (info["dev"], info["inode"])
                            linked = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                            require((linked.st_dev, linked.st_ino) == identity, "Unlink target was replaced")
                            os.unlink(path.name, dir_fd=fd)
                            after = fingerprint(os.fstat(descriptor))
                            content_cache[content_marker(after)] = row["sha256"]
                            removed_files += 1
                        finally:
                            os.close(descriptor)
                    finally:
                        os.close(fd)
                dirs = sorted((row for row in manifest["entries"] if row["root_key"] == key and
                    row["relative"] and row["stat"]["type"] == "dir"),
                    key=lambda row: len(PurePosixPath(row["relative"]).parts), reverse=True)
                for row in dirs:
                    path, expected = PurePosixPath(row["relative"]), row["stat"]
                    fd = parent_fd(path.parts[:-1])
                    try:
                        info = fingerprint(os.stat(path.name, dir_fd=fd, follow_symlinks=False))
                        require(info["type"] == "dir" and (info["dev"], info["inode"]) ==
                                (expected["dev"], expected["inode"]), "Directory identity differs")
                        os.rmdir(path.name, dir_fd=fd); removed_dirs += 1
                    finally:
                        os.close(fd)
            finally:
                os.close(root_fd)
            parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = fingerprint(os.fstat(parent))
                require({name: info[name] for name in ("dev", "inode")} == manifest["roots"][key]["parent_identity"],
                        "Workspace parent was replaced")
                info = fingerprint(os.stat(target.name, dir_fd=parent, follow_symlinks=False))
                require((info["dev"], info["inode"]) == (anchor["dev"], anchor["inode"]), "Workspace root was replaced")
                os.rmdir(target.name, dir_fd=parent); removed_dirs += 1
            finally:
                os.close(parent)
        require(audit_terminal(root, arm, qid) == binding, "Compact artifacts changed during maintenance")
        result = {"version": VERSION, "deleted": True, "binding": binding,
            "acknowledgement_sha256": acknowledgement["acknowledgement_sha256"],
            "files_removed": removed_files, "directories_removed": removed_dirs, "finished_unix": time.time(),
            "compact_artifacts_preserved": True, "working_parent_preserved": True, "runtime_parent_preserved": True}
        result["deletion_sha256"] = digest(result)
        write_new(receipt_path.parent / "deletion_receipt.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--arm", required=True); export.add_argument("--qid", type=int, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True); verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--expected-receipt-sha256", required=True); verify.add_argument("--expected-receipt-bytes", type=int, required=True)
    verify.add_argument("--acknowledgement-file", type=Path, required=True)
    delete = sub.add_parser("delete")
    delete.add_argument("--arm", required=True); delete.add_argument("--qid", type=int, required=True)
    delete.add_argument("--receipt", type=Path, required=True)
    delete.add_argument("--acknowledgement-stdin", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_workspace(args.root, args.arm, args.qid, args.output_dir)
    elif args.command == "verify":
        result = verify_archive(args.archive, args.receipt, expected_receipt_sha256=args.expected_receipt_sha256,
                                expected_receipt_bytes=args.expected_receipt_bytes)
        write_new(args.acknowledgement_file, result)
    else:
        result = delete_workspace(args.root, args.arm, args.qid, args.receipt, json.load(sys.stdin))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
