#!/usr/bin/env python3
"""Export a verified auto20 snapshot, then verify its extracted bytes offline.

Only administrative files are created. No model, network, judge, service or
machine operation is invoked. The frozen experiment source is never edited.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from strict_ablations import auto20_analysis as analysis
from strict_ablations.auto20 import Auto20Suite, ARM, COUNT, DIRECTORY
from strict_ablations import runner
from strict_ablations.control import verify_control_bundle

VERSION = "strict-auto20-snapshot-v1"
SUFFIXES = {".json", ".jsonl", ".log", ".py", ".sh", ".txt"}
ARTIFACT_DIRS = ("_per_query", "request_contracts", "api_ledgers", "_traces", "_checkpoints")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe(root, relative):
    path = PurePosixPath(relative)
    require(isinstance(relative, str) and not path.is_absolute() and ".." not in path.parts,
            "Snapshot paths must be relative and contained")
    target = root.joinpath(*path.parts)
    require(target.resolve().is_relative_to(root.resolve()) and not target.is_symlink(),
            "Snapshot path escapes its root or aliases a file")
    return target


def _checked(value, field):
    require(isinstance(value, dict) and value.get(field) == runner.digest(
        {key: item for key, item in value.items() if key != field}), field + " digest changed")


def _records(rows, cohort):
    return [rows[(arm, qid)] for arm in analysis.GROUPS for qid in cohort]


def snapshot_files(root, output, *, operation_files=()):
    """Select only reports, compact execution artifacts and auto20 operations.

    Archives/previous snapshot directories and the current output are excluded.
    Current output's three analysis JSON files are added explicitly by export.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    reports, runs = root / "reports" / DIRECTORY, root / "runs" / DIRECTORY / ARM
    selected = set()
    def add(path):
        if not path.is_file() or path.is_symlink():
            return
        relative = path.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        if (path.resolve().is_relative_to(output) or "archives" in parts or "snapshots" in parts
                or any(part.startswith("snapshot_") for part in parts)
                or path.name in {"snapshot_receipt.json", "analysis_package.json", "blind40.json", "private40.json"}
                or path.name.endswith((".tar", ".tar.gz", ".tgz", ".zip"))):
            return
        selected.add(path)
    for path in reports.rglob("*"):
        add(path)
    for directory in ARTIFACT_DIRS:
        for path in (runs / directory).rglob("*"):
            add(path)
    operations = root / "operations"
    for path in operations.rglob("auto20*"):
        if path.suffix in SUFFIXES:
            add(path)
    for name in operation_files:
        path = Path(name)
        path = path if path.is_absolute() else root / path
        require(path.resolve().is_relative_to(operations.resolve()) and path.suffix in SUFFIXES,
                "Extra operation file must be an administrative text file under operations/")
        require(path.is_file() and not path.is_symlink(), "Requested operation file is missing")
        add(path)
    require(all("_working" not in p.parts and "_rerank_runtime" not in p.parts for p in selected),
            "Large original-document pools must not enter the snapshot")
    return sorted(selected)


def _archive(root, files, path):
    """Hash the exact captured bytes added to tar, even if a log later grows."""
    root = Path(root).resolve()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    checksums = {}
    with os.fdopen(descriptor, "wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for source in files:
            require(not source.is_symlink(), "Aliased archive member")
            source = source.resolve()
            relative = source.relative_to(root).as_posix()
            require(relative not in checksums and not source.is_symlink(), "Duplicate or aliased archive member")
            data = source.read_bytes()
            checksums[relative] = _sha(data)
            member = tarfile.TarInfo(relative)
            member.size, member.mode, member.mtime = len(data), 0o600, int(source.stat().st_mtime)
            archive.addfile(member, io.BytesIO(data))
    return checksums


def export_snapshot(root, output, *, operation_files=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and output != root, "Output must be a new directory under the artifact root")
    require(not output.exists(), "Snapshot output must not already exist")
    suite = Auto20Suite(root)
    with (suite.reports / "worker.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = suite.status()
        require(status.get("completed_new_runs") == COUNT
                and not status["arms"][ARM]["pending"] and not status["arms"][ARM]["interrupted"],
                "All twenty natural terminals must be complete before snapshot")
        for path in suite.runs.iterdir():
            require(path.name == ARM or not any(p.is_file() for p in path.rglob("*")),
                    "Another solver arm exists in the isolated pilot output")
        rows, cohort, manifest, provenance = analysis.collect(root, suite=suite)
        completion = runner.load(suite.reports / "pilot20_completion.json")
        require(completion.get("passed") is True and completion.get("manifest_sha256") == manifest["manifest_sha256"]
                and completion.get("question_runs") == COUNT and completion.get("control_model_requests") == 0
                and completion.get("other_arm_model_requests") == 0, "The auto20 completion receipt is missing/mismatched")
        require(runner.source_hashes() == manifest["source_sha256"], "Frozen source changed")
        manifest_bytes = (suite.reports / "run_manifest.json").read_bytes()
        packet, private = analysis.export_blind(rows, cohort, manifest)
        output.mkdir(parents=True, exist_ok=False)
        analysis.write_new(output / "blind40.json", packet, private=True)
        analysis.write_new(output / "private40.json", private, private=True)
        package = {"version": VERSION, "source_root": str(root),
            "manifest_original_bytes_base64": base64.b64encode(manifest_bytes).decode("ascii"),
            "manifest_file_sha256": _sha(manifest_bytes), "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": manifest["source_sha256"], "provenance": provenance,
            "rows": _records(rows, cohort), "cohort": list(cohort.values()),
            "blind_packet_sha256": packet["packet_sha256"], "private_mapping_sha256": private["mapping_sha256"],
            "new_question_runs": COUNT, "reused_control_questions": COUNT,
            "solver_control_model_requests": 0, "solver_other_arm_model_requests": 0,
            "analysis_policy": "Public finals only; all forty labels require fresh blinded review"}
        package["analysis_package_sha256"] = runner.digest(package)
        analysis.write_new(output / "analysis_package.json", package, private=True)
        exports = [output / name for name in ("blind40.json", "private40.json", "analysis_package.json")]
        files = sorted({*snapshot_files(root, output, operation_files=operation_files), *exports})
        archive = output / "auto20_snapshot.tar.gz"
        hashes = _archive(root, files, archive)
        require(runner.source_hashes() == manifest["source_sha256"], "Frozen source changed during snapshot")
        package_name = (output / "analysis_package.json").relative_to(root).as_posix()
        require(hashes[(suite.reports / "run_manifest.json").relative_to(root).as_posix()] == _sha(manifest_bytes),
                "Manifest changed while snapshotting")
        receipt = {"version": VERSION, "ready_for_shutdown": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest["manifest_sha256"], "manifest_file_sha256": _sha(manifest_bytes),
            "source_sha256": manifest["source_sha256"], "selected_query_ids": list(cohort),
            "terminal_rows": COUNT, "public_finals_for_fresh_review": 40,
            "solver_control_model_requests": 0, "solver_other_arm_model_requests": 0,
            "compatibility_probes_separate_from_solver_counts": True,
            "new_solver_api_attempts": sum(rows[(ARM, qid)]["metrics"]["api_attempts"] for qid in cohort),
            "files": hashes, "analysis_package": package_name,
            "blind_packet": (output / "blind40.json").relative_to(root).as_posix(),
            "private_mapping": (output / "private40.json").relative_to(root).as_posix(),
            "archive": {"path": archive.relative_to(root).as_posix(), "sha256": runner.file_sha(archive),
                        "bytes": archive.stat().st_size},
            "checker_sha256": runner.file_sha(Path(__file__)),
            "exclusions": ["_working", "_rerank_runtime", "archives", "prior snapshots", "archive itself", "receipt itself"]}
        receipt["receipt_sha256"] = runner.digest(receipt)
        analysis.write_new(output / "snapshot_receipt.json", receipt, private=True)
    return {"ready_for_shutdown": True, "terminal_rows": COUNT,
            "archive": str(archive), "archive_sha256": receipt["archive"]["sha256"],
            "receipt": str(output / "snapshot_receipt.json"), "files": len(hashes),
            "external_model_requests": 0}


def load_verified_snapshot(root, *, receipt=None, control_root=None, archive=None):
    """Verify extracted compact artifacts without consulting server live paths.

    root is the extracted archive's artifact root. Its code need not live there:
    all frozen Python hashes are checked against this script's local checkout.
    The independent receipt is downloaded separately and passed explicitly, or
    placed at root/snapshot_receipt.json. Optionally verify the downloaded tar.
    Returns rows, cohort, manifest, provenance for offline import/summarize.
    """
    root = Path(root).resolve()
    receipt = Path(receipt) if receipt else root / "snapshot_receipt.json"
    proof = runner.load(receipt)
    _checked(proof, "receipt_sha256")
    require(proof.get("version") == VERSION and proof.get("ready_for_shutdown") is True
            and proof.get("terminal_rows") == COUNT and proof.get("solver_control_model_requests") == 0
            and proof.get("solver_other_arm_model_requests") == 0, "Snapshot scope differs")
    require(proof.get("source_sha256") == runner.source_hashes(), "Local frozen Python source does not match the run")
    files = proof.get("files")
    require(isinstance(files, dict) and files, "Snapshot has no file checksum map")
    for relative, checksum in files.items():
        path = _safe(root, relative)
        require(path.is_file() and runner.file_sha(path) == checksum, "Snapshot file SHA differs: " + relative)
    if archive is not None:
        require(runner.file_sha(archive) == proof["archive"]["sha256"], "Downloaded archive SHA differs")
    package_path = _safe(root, proof["analysis_package"])
    package = runner.load(package_path)
    _checked(package, "analysis_package_sha256")
    require(package.get("version") == VERSION and package.get("source_sha256") == proof["source_sha256"],
            "Analysis package/source differs")
    manifest_bytes = base64.b64decode(package["manifest_original_bytes_base64"], validate=True)
    manifest = json.loads(manifest_bytes)
    require(_sha(manifest_bytes) == package.get("manifest_file_sha256") == proof.get("manifest_file_sha256"),
            "Original manifest bytes changed")
    require((root / "reports" / DIRECTORY / "run_manifest.json").read_bytes() == manifest_bytes,
            "Extracted manifest differs from the original manifest bytes")
    analysis.check_manifest(manifest)
    require(manifest["manifest_sha256"] == package.get("manifest_sha256") == proof.get("manifest_sha256")
            and manifest["source_sha256"] == runner.source_hashes(), "Manifest digest/source binding differs")
    cohort = analysis.fixed_cohort()
    require(package.get("cohort") == list(cohort.values()) and proof.get("selected_query_ids") == list(cohort),
            "Frozen first20 question bytes/order differ")
    suite = Auto20Suite(root)
    terminals = []
    for qid, query in cohort.items():
        path = suite.terminal_path(ARM, qid)
        relative = path.relative_to(root).as_posix()
        require(relative in files, "Terminal was not included in receipt")
        row = runner.load(path)
        suite.validate_terminal(row, manifest, ARM, query)
        suite.verify_execution_artifacts(manifest, ARM, query, row)
        terminals.append(row)
    # This path is local and explicit. Never call verify_manifest(), whose
    # frozen control_root is intentionally the original server absolute path.
    control = verify_control_bundle(control_root or SOURCE_ROOT, common_config=manifest["config"])
    require(runner.control_reference(control) == manifest["control_reference"], "Local public B100 control differs")
    selected = [row for row in control["records"] if row["query_id"] in cohort]
    rows = analysis.normalize_records(terminals, selected, cohort, manifest)
    require(package.get("rows") == _records(rows, cohort), "Packaged public finals/metrics differ from verified artifacts")
    source_root = Path(package["source_root"])
    for path, checksum in package["provenance"].items():
        if path == "control_reference_sha256":
            require(checksum == runner.digest(manifest["control_reference"]), "Control provenance changed")
        else:
            relative = Path(path).relative_to(source_root).as_posix()
            require(files.get(relative) == checksum, "Terminal provenance is outside the receipt")
    packet = runner.load(_safe(root, proof["blind_packet"]))
    private = runner.load(_safe(root, proof["private_mapping"]))
    analysis._mapping(private, rows, manifest)
    require(analysis._packet(private, rows, manifest) == packet
            and packet["packet_sha256"] == package["blind_packet_sha256"]
            and private["mapping_sha256"] == package["private_mapping_sha256"], "Blind review package changed")
    provenance = {"snapshot_receipt_sha256": proof["receipt_sha256"],
        "snapshot_archive_sha256": proof["archive"]["sha256"],
        "archive_bytes_verified": archive is not None,
        "analysis_package_sha256": package["analysis_package_sha256"],
        "manifest_file_sha256": _sha(manifest_bytes), **package["provenance"]}
    return rows, cohort, manifest, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--operation-file", action="append", default=[])
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", type=Path)
    verify.add_argument("--control-root", type=Path, default=SOURCE_ROOT)
    verify.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_snapshot(args.root, args.output_dir, operation_files=args.operation_file)
    else:
        rows, cohort, manifest, provenance = load_verified_snapshot(args.root, receipt=args.receipt,
            control_root=args.control_root, archive=args.archive)
        result = {"verified": True, "terminal_rows": len(cohort), "fresh_review_finals": len(rows),
            "manifest_sha256": manifest["manifest_sha256"],
            "archive_bytes_verified": provenance["archive_bytes_verified"], "external_model_requests": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
