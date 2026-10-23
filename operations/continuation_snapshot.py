#!/usr/bin/env python3
"""Opaque trajectory backup and offline verification for auto80 then paging20.

This administrative tool never solves, scores, calls a model, starts a service,
or alters experiment sources. Raw traces/checkpoints are only copied, hashed,
and passed to the existing artifact integrity verifier; private reasoning is
never emitted or included in the public-final projection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from strict_ablations import runner
from strict_ablations import control
from strict_ablations.auto20 import Auto20Suite
from strict_ablations.prompt import ARMS, CONTROL, build_contract
from operations.continuation_suite import ContinuationSuite, verify_reused_auto20

VERSION = "strict-continuation-snapshot-v1"
DIRECTORY = "strict_continue_v1"
COUNT = 100
AUTO20_MANIFEST = "67b4ff0f8f1a9d25583ad77ab150f919dec078aa34c824126648d88433ee1384"
ARTIFACT_DIRS = ("_per_query", "request_contracts", "api_ledgers", "_traces", "_checkpoints")
CONTROL_FILES = (control.MANIFEST_PATH, control.PUBLIC_PATH, control.METADATA_PATH,
                 control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checked(value, field):
    require(isinstance(value, dict) and value.get(field) == runner.digest(
        {k: v for k, v in value.items() if k != field}), field + " digest differs")


def _safe(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Snapshot path must be a nonempty POSIX relative path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == relative
            and relative != ".", "Snapshot path is not canonical and contained")
    root = Path(root).resolve()
    target = root.joinpath(*path.parts)
    require(target.resolve().is_relative_to(root), "Snapshot path escapes its root")
    current = target
    while current != root:
        require(not current.is_symlink(), "Snapshot path contains a symlink")
        current = current.parent
    return target


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n").encode())


def expected_jobs():
    queries = runner.shared.validate_queries()
    require(len(queries) == 100, "The original frozen cohort must contain 100 questions")
    return [("auto_rerank", query) for query in queries[20:]] + [
        ("search_paging", query) for query in queries[:20]]


class ArtifactSuite(runner.Suite):
    directory = DIRECTORY


def _manifest(root, *, control_root):
    """Validate bytes/identities without following any server paths offline."""
    root = Path(root)
    manifest = runner.load(root / "reports" / DIRECTORY / "run_manifest.json")
    _checked(manifest, "manifest_sha256")
    require(manifest.get("version") == runner.SUITE_VERSION, "Wrong suite manifest version")
    runner.shared.validate_config(manifest["config"], actual_resources=False)
    require(manifest.get("config_sha256") == runner.digest(manifest["config"])
            and manifest.get("source_sha256") == runner.source_hashes(), "Frozen config/source differs")
    require(manifest.get("contracts") == {arm: build_contract(arm) for arm in ARMS}
            and manifest.get("control_model_inputs") == build_contract(CONTROL), "Frozen contracts differ")
    queries = runner.shared.validate_queries()
    require(manifest.get("cohort_sha256") == runner.shared.DATASET_SHA
            and manifest.get("query_ids") == [q["query_id"] for q in queries], "Fixed100 identity/order differs")
    stages = [{"name": name, "arm": arm, "query_ids": [q["query_id"] for q in selected]}
              for name, arm, selected in (("auto80", "auto_rerank", queries[20:]),
                                          ("paging20", "search_paging", queries[:20]))]
    require(manifest.get("question_runs") == COUNT
            and manifest.get("execution_plan", {}).get("stages") == stages, "Continuation scope differs")
    suite = ContinuationSuite(root)
    expected_scope = suite.execution_scope(queries)
    require(all(runner.digest(manifest.get(key)) == runner.digest(value)
                for key, value in expected_scope.items()), "Frozen continuation execution scope differs")
    require(manifest.get("analysis_contract") == suite.analysis_contract(), "Frozen analysis contract differs")
    verified_control = control.verify_control_bundle(control_root, common_config=manifest["config"])
    require(runner.control_reference(verified_control) == manifest.get("control_reference"),
            "Pinned B100 control differs")
    admin = manifest.get("administrative_source_sha256")
    require(isinstance(admin, dict) and admin, "Administrative source pin is missing")
    for relative, checksum in admin.items():
        require(relative.startswith("operations/") and runner.file_sha(_safe(SOURCE_ROOT, relative)) == checksum,
                "Administrative scheduler source differs")
    epoch = runner.load(root / "reports" / DIRECTORY / "service_epochs" /
                        (manifest["initial_service_epoch_sha256"] + ".json"))
    _checked(epoch, "epoch_sha256")
    require(epoch["epoch_sha256"] == manifest["initial_service_epoch_sha256"], "Initial service epoch differs")
    return manifest, verified_control


def _verify_rows(root, manifest):
    suite = ArtifactSuite(root)
    ContinuationSuite(root).assert_run_scope()
    expected = {(arm, q["query_id"]) for arm, q in expected_jobs()}
    actual = set()
    for arm_dir in suite.runs.iterdir():
        if not arm_dir.is_dir():
            require(not arm_dir.is_file(), "Unexpected file in run root")
            continue
        for path in (arm_dir / "_per_query").glob("*.json"):
            row = runner.load(path)
            pair = (arm_dir.name, row.get("query_id"))
            require(path.name == f"qid_{row.get('query_id')}.json" and pair not in actual,
                    "Duplicate or renamed terminal")
            actual.add(pair)
        require(arm_dir.name in {"auto_rerank", "search_paging"}
                or not any(p.is_file() for p in arm_dir.rglob("*")), "Unexpected solver arm artifacts")
    require(actual == expected and len(actual) == COUNT,
            "Exactly auto80 and paging20 natural terminals are required")
    rows = []
    for arm, query in expected_jobs():
        row = runner.load(suite.terminal_path(arm, query["query_id"]))
        suite.validate_terminal(row, manifest, arm, query)
        suite.verify_execution_artifacts(manifest, arm, query, row)
        rows.append(row)
    return rows


def _verify_completion(root, manifest, rows):
    reports = Path(root) / "reports" / DIRECTORY
    state = runner.load(reports / "phase_state.json")
    require(state.get("phase") == "complete" and state.get("manifest_sha256") == manifest["manifest_sha256"],
            "Pipeline phase is not complete or belongs to another manifest")
    completion = runner.load(reports / "continuation_completion.json")
    require(completion.get("passed") is True and completion.get("question_runs") == COUNT
            and completion.get("terminal_rows_checked") == COUNT
            and completion.get("manifest_sha256") == manifest["manifest_sha256"],
            "Continuation completion receipt differs")
    require(all(completion.get(key) == 0 for key in ("natural_error_reruns", "control_model_requests", "other_arm_model_requests",
                                                     "reused_auto20_model_requests")),
            "Control, reused auto20, or other arms were scheduled")
    require(state.get("receipt_sha256") == runner.file_sha(reports / "continuation_completion.json")
            and state.get("terminal_rows_checked") == COUNT, "Final phase does not bind the completion receipt")
    stage_receipts = []
    for name, arm, count in (("auto80", "auto_rerank", 80), ("paging20", "search_paging", 20)):
        receipt = runner.load(reports / (name + "_completion.json"))
        require(receipt.get("passed") is True and receipt.get("manifest_sha256") == manifest["manifest_sha256"],
                "Stage completion identity differs")
        expected = [{"arm": arm, "query_id": row["query_id"], "row_sha256": row["row_sha256"],
                     "file_sha256": runner.file_sha(reports.parent.parent / "runs" / DIRECTORY / arm /
                                                     "_per_query" / f"qid_{row['query_id']}.json"),
                     "execution_artifacts": row["execution_artifacts"]}
                    for row in rows if row["arm"] == arm]
        require(len(expected) == count and receipt.get("terminal_rows") == expected,
                "Stage receipt does not bind exactly the immutable natural terminals")
        require(receipt.get("question_runs") == count and receipt.get("terminal_rows_checked") == count
                and all(receipt.get(key) == 0 for key in ("natural_error_reruns", "control_model_requests",
                    "other_arm_model_requests", "reused_auto20_model_requests")), "Stage counts or rerun policy differ")
        stage_receipts.append(receipt)
    require(completion.get("stages") == stage_receipts, "Final completion stage bindings differ")
    return state, completion


def verify_worker_exit(identity, *, proc_root=Path("/proc")):
    """Prove the exact worker identity has exited; PID reuse is not a match."""
    require(isinstance(identity, dict) and type(identity.get("pid")) is int and identity["pid"] > 0
            and str(identity.get("proc_start_ticks", "")).isdigit()
            and isinstance(identity.get("boot_id"), str) and identity["boot_id"],
            "A Linux worker PID/start-ticks/boot identity is required")
    proc_root = Path(proc_root)
    boot = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    stat = proc_root / str(identity["pid"]) / "stat"
    observed = {"expected": identity, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "observed_boot_id": boot, "worker_exited": True}
    if boot != identity["boot_id"]:
        return {**observed, "reason": "different_boot"}
    try:
        data = stat.read_text()
    except FileNotFoundError:
        return {**observed, "reason": "pid_absent"}
    # comm may contain spaces or parentheses; fields after the last ')' start at field 3.
    fields = data.rsplit(")", 1)[1].split()
    require(len(fields) > 19, "Malformed process identity")
    ticks, state = fields[19], fields[0]
    observed.update(observed_start_ticks=ticks, observed_process_state=state)
    if ticks != str(identity["proc_start_ticks"]):
        return {**observed, "reason": "pid_reused"}
    require(state == "Z", "The experiment worker is still running")
    return {**observed, "reason": "exited_zombie"}


def _verify_reused(root, manifest):
    root = Path(root)
    reference = manifest.get("reused_auto20_reference")
    require(isinstance(reference, dict) and reference.get("manifest_sha256") == AUTO20_MANIFEST
            and reference.get("terminal_rows") == 20, "Original auto20 reference differs")
    require(isinstance(reference.get("files"), dict) and len(reference["files"]) == 102,
            "Original auto20 must pin manifest, completion and 100 artifacts")
    observed = verify_reused_auto20(root, common_config=manifest["config"],
                                    control_reference=manifest["control_reference"])
    require({k: v for k, v in observed.items() if k != "root"} ==
            {k: v for k, v in reference.items() if k != "root"}, "Pinned original auto20 reference differs")
    suite = Auto20Suite(root)
    return [runner.load(suite.terminal_path("auto_rerank", query["query_id"]))
            for query in runner.shared.validate_queries()[:20]]


def _public_finals(new_rows, reused_rows, control_bundle):
    queries = {q["query_id"]: q for q in runner.shared.validate_queries()}
    records = []
    for row in [*reused_rows, *new_rows]:
        query = queries[row["query_id"]]
        records.append({"arm": row["arm"], "query_id": row["query_id"], "question": query["query"],
            "gold_answer": query["answer"], "final_text": row["final_text"], "final_sha256": row["final_sha256"],
            "row_sha256": row["row_sha256"], "manifest_sha256": row["manifest_sha256"],
            "terminated_by": row["terminated_by"]})
    for row in control_bundle["records"]:
        records.append({"arm": "control", **{key: row[key] for key in
            ("query_id", "question", "gold_answer", "final_text", "final_sha256", "terminated_by")},
            "row_sha256": row["terminal_row_sha256"], "manifest_sha256": control_bundle["manifest_sha256"]})
    require(len(records) == 220 and len({(row["arm"], row["query_id"]) for row in records}) == 220,
            "Finals must cover auto100, paging20, and reused control100 exactly once")
    return sorted(records, key=lambda row: (row["arm"], row["query_id"]))


def _copy_pinned(source_root, target_root, files):
    for relative in files:
        source, target = _safe(source_root, str(relative)), _safe(target_root, str(relative))
        require(source.is_file(), "Missing bundle file")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            os.chmod(target, 0o600)
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)


def snapshot_files(root, output, manifest, *, operation_files=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    selected = set()
    for path in (root / "reports" / DIRECTORY).rglob("*"):
        relative = path.relative_to(root)
        if (path.is_file() and not path.resolve().is_relative_to(output)
                and not {"snapshots", "downloaded", "archives"}.intersection(relative.parts)
                and not path.name.endswith((".tar", ".tar.gz", ".tgz", ".zip"))):
            selected.add(_safe(root, relative.as_posix()))
    for arm, query in expected_jobs():
        base, qid = root / "runs" / DIRECTORY / arm, query["query_id"]
        selected.update([base / "_per_query" / f"qid_{qid}.json", base / "request_contracts" / f"qid_{qid}.json",
            base / "api_ledgers" / f"qid_{qid}.jsonl", base / "_traces" / f"qid_{qid}" / "single.json",
            base / "_checkpoints" / f"qid_{qid}.json"])
    for relative, checksum in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        path = _safe(root, relative)
        require(path.is_file() and runner.file_sha(path) == checksum, "Source copy differs: " + relative)
        selected.add(path)
    for relative in {"operations/continuation_snapshot.py", *operation_files}:
        path = _safe(root, relative)
        require(relative.startswith("operations/") and path.is_file()
                and path.suffix in {".json", ".jsonl", ".log", ".py", ".sh", ".txt"},
                "Extra file must be an explicit administrative text file under operations/")
        require(not any(part.lower() in {".env", "credentials", "secrets"} for part in path.parts),
                "Credential paths must not enter the snapshot")
        selected.add(path)
    require(all(not {"_working", "_rerank_runtime", "venv", "models", "index"}.intersection(p.parts)
                for p in selected), "Large runtime data must not enter the snapshot")
    return sorted(selected)


def _archive(root, files, output):
    root = Path(root).resolve()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    pins = {}
    with os.fdopen(descriptor, "wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            source = _safe(root, relative)
            require(source.is_file() and relative not in pins, "Duplicate/missing archive member")
            data = source.read_bytes()
            pins[relative] = {"sha256": _sha(data), "bytes": len(data)}
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = len(data), 0o600, int(source.stat().st_mtime)
            archive.addfile(info, io.BytesIO(data))
    return pins


def export_snapshot(root, output, *, reused_auto20_root=None, operation_files=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and output != root and not output.exists(),
            "Snapshot output must be a previously nonexistent directory under the run root")
    reports = root / "reports" / DIRECTORY
    # The scheduler creates this file. Do not create or modify a missing lock.
    with (reports / "worker.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = runner.load(reports / "run_manifest.json")
        manifest, verified_control = _manifest(root, control_root=original["control_root"])
        rows = _verify_rows(root, manifest)
        state, completion = _verify_completion(root, manifest, rows)
        exit_proof = verify_worker_exit(state.get("worker_identity"))
        reused_root = Path(reused_auto20_root or manifest["reused_auto20_reference"]["root"])
        reused = _verify_reused(reused_root, manifest)
        output.mkdir(parents=True, exist_ok=False)
        control_target, reused_target = output / "bundles/control", output / "bundles/auto20"
        _copy_pinned(Path(manifest["control_root"]), control_target, CONTROL_FILES)
        _copy_pinned(reused_root, reused_target, manifest["reused_auto20_reference"]["files"])
        # Re-read copies through the same offline validators before making a receipt.
        copied_control = control.verify_control_bundle(control_target, common_config=manifest["config"])
        copied_reused = _verify_reused(reused_target, manifest)
        require(_public_finals(rows, reused, verified_control) == _public_finals(rows, copied_reused, copied_control),
                "Copied public finals differ")
        finals_path = output / "public_finals.json"
        finals = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
                  "policy": "No labels reused or generated; future scoring requires fresh blinded review",
                  "records": _public_finals(rows, reused, verified_control)}
        finals["public_finals_sha256"] = runner.digest(finals)
        _write_new(finals_path, finals)
        files = snapshot_files(root, output, manifest, operation_files=operation_files)
        files = sorted({*files, finals_path, *(p for p in (output / "bundles").rglob("*") if p.is_file())})
        archive = output / "continuation_snapshot.tar.gz"
        pins = _archive(root, files, archive)
        # A second full pass catches unexpected writes while acquiring the backup.
        require(_verify_rows(root, manifest) == rows, "Natural terminals changed during backup")
        require(runner.source_hashes() == manifest["source_sha256"], "Frozen sources changed during backup")
        receipt = {"version": VERSION, "ready_for_shutdown": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": runner.file_sha(reports / "run_manifest.json"),
            "source_sha256": manifest["source_sha256"], "terminal_rows": COUNT,
            "new_auto_rerank_rows": 80, "new_search_paging_rows": 20, "reused_auto20_rows": 20,
            "public_finals": 220, "control_model_requests": 0, "other_arm_model_requests": 0,
            "reused_auto20_model_requests": 0, "worker_exit_verification": exit_proof,
            "worker_lock_acquired": True, "pipeline_phase": state["phase"],
            "files": pins, "public_finals_path": finals_path.relative_to(root).as_posix(),
            "bundles": {"control": control_target.relative_to(root).as_posix(),
                        "reused_auto20": reused_target.relative_to(root).as_posix()},
            "archive": {"path": archive.relative_to(root).as_posix(), "sha256": runner.file_sha(archive),
                        "bytes": archive.stat().st_size}, "checker_sha256": runner.file_sha(Path(__file__)),
            "exclusions": ["original-document pools", "weights", "venv", "corpus", "index", "credentials",
                           "prior archives", "prior snapshots", "archive itself", "receipt itself"]}
        receipt["receipt_sha256"] = runner.digest(receipt)
        receipt_path = output / "snapshot_receipt.json"
        _write_new(receipt_path, receipt)
    return {"ready_for_shutdown": True, "terminal_rows": COUNT, "reused_auto20_rows": 20,
            "archive": str(archive), "archive_sha256": receipt["archive"]["sha256"],
            "archive_bytes": archive.stat().st_size, "receipt": str(receipt_path),
            "receipt_file_sha256": runner.file_sha(receipt_path), "receipt_bytes": receipt_path.stat().st_size,
            "files": len(pins), "external_model_requests": 0}


def _receipt(path, *, expected_sha256=None, expected_bytes=None):
    path = Path(path)
    if expected_sha256 is not None:
        require(runner.file_sha(path) == expected_sha256, "Downloaded independent receipt SHA differs")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, "Downloaded independent receipt byte size differs")
    proof = runner.load(path)
    _checked(proof, "receipt_sha256")
    require(proof.get("version") == VERSION and proof.get("ready_for_shutdown") is True
            and proof.get("terminal_rows") == COUNT and proof.get("reused_auto20_rows") == 20
            and proof.get("pipeline_phase") == "complete" and proof.get("worker_lock_acquired") is True
            and proof.get("worker_exit_verification", {}).get("worker_exited") is True,
            "Snapshot completion/safety scope differs")
    require(all(proof.get(key) == 0 for key in ("control_model_requests", "other_arm_model_requests",
                                               "reused_auto20_model_requests")), "Snapshot contains forbidden solver runs")
    require(isinstance(proof.get("files"), dict) and proof["files"], "Snapshot checksum map is missing")
    return proof


def _check_archive(archive, proof):
    archive = Path(archive)
    require(archive.is_file() and archive.stat().st_size == proof["archive"]["bytes"]
            and runner.file_sha(archive) == proof["archive"]["sha256"], "Downloaded archive SHA/byte size differs")


def safe_extract(archive, receipt, output, *, expected_receipt_sha256=None, expected_receipt_bytes=None):
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    _check_archive(archive, proof)
    output = Path(output).resolve()
    require(not output.exists(), "Extract into a new, previously nonexistent directory")
    # Validate all member types/paths before creating any output file.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)) and set(names) == set(proof["files"]),
                "Archive membership differs from receipt or contains duplicate members")
        for member in members:
            _safe(output, member.name)
            require(member.isfile() and not member.issym() and not member.islnk()
                    and member.size == proof["files"][member.name]["bytes"],
                    "Archive contains an unsafe member type or changed size")
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = _safe(output, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as original, target.open("xb") as stream:
                os.chmod(target, 0o600)
                shutil.copyfileobj(original, stream)
            require(runner.file_sha(target) == proof["files"][member.name]["sha256"],
                    "Extracted member checksum differs")
    return {"safely_extracted": True, "root": str(output), "files": len(members), "archive_bytes_verified": True}


def load_verified_snapshot(root, *, receipt, archive, expected_receipt_sha256=None, expected_receipt_bytes=None):
    root = Path(root).resolve()
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    _check_archive(archive, proof)
    for relative, pin in proof["files"].items():
        path = _safe(root, relative)
        require(path.is_file() and path.stat().st_size == pin["bytes"]
                and runner.file_sha(path) == pin["sha256"], "Snapshot file bytes differ: " + relative)
    manifest, verified_control = _manifest(root, control_root=_safe(root, proof["bundles"]["control"]))
    require(manifest["manifest_sha256"] == proof["manifest_sha256"]
            and manifest["source_sha256"] == proof["source_sha256"]
            and runner.file_sha(root / "reports" / DIRECTORY / "run_manifest.json") == proof["manifest_file_sha256"],
            "Original manifest bytes/identity differ")
    required = {f"reports/{DIRECTORY}/{name}" for name in (
        "run_manifest.json", "phase_state.json", "continuation_completion.json",
        "auto80_completion.json", "paging20_completion.json")}
    required.add(f"reports/{DIRECTORY}/service_epochs/{manifest['initial_service_epoch_sha256']}.json")
    required.add(proof["public_finals_path"])
    required.update(f"{proof['bundles']['control']}/{name}" for name in CONTROL_FILES)
    required.update(f"{proof['bundles']['reused_auto20']}/{name}"
                    for name in manifest["reused_auto20_reference"]["files"])
    for arm, query in expected_jobs():
        qid, base = query["query_id"], f"runs/{DIRECTORY}/{arm}"
        required.update((f"{base}/_per_query/qid_{qid}.json", f"{base}/request_contracts/qid_{qid}.json",
            f"{base}/api_ledgers/qid_{qid}.jsonl", f"{base}/_traces/qid_{qid}/single.json",
            f"{base}/_checkpoints/qid_{qid}.json"))
    require(required <= set(proof["files"]), "A required execution artifact is outside the independent checksum receipt")
    for relative, checksum in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        require(proof["files"].get(relative, {}).get("sha256") == checksum, "Archived frozen source differs")
    rows = _verify_rows(root, manifest)
    state, completion = _verify_completion(root, manifest, rows)
    require(state.get("worker_identity") == proof["worker_exit_verification"]["expected"],
            "Worker exit proof belongs to another pipeline")
    reused = _verify_reused(_safe(root, proof["bundles"]["reused_auto20"]), manifest)
    finals = runner.load(_safe(root, proof["public_finals_path"]))
    _checked(finals, "public_finals_sha256")
    require(finals.get("manifest_sha256") == manifest["manifest_sha256"]
            and finals.get("records") == _public_finals(rows, reused, verified_control),
            "Public finals changed or were incompletely exported")
    return {"verified": True, "terminal_rows": COUNT, "new_auto_rerank_rows": 80,
            "new_search_paging_rows": 20, "reused_auto20_rows": 20, "merged_auto_rerank_rows": 100,
            "public_finals": 220, "archive_bytes_verified": True,
            "manifest_sha256": manifest["manifest_sha256"], "external_model_requests": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--reused-auto20-root", type=Path)
    export.add_argument("--operation-file", action="append", default=[])
    for name in ("extract", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--expected-receipt-sha256")
        command.add_argument("--expected-receipt-bytes", type=int)
        if name == "extract":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_snapshot(args.root, args.output_dir, reused_auto20_root=args.reused_auto20_root,
                                 operation_files=args.operation_file)
    elif args.command == "extract":
        result = safe_extract(args.archive, args.receipt, args.output_dir,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    else:
        result = load_verified_snapshot(args.root, receipt=args.receipt, archive=args.archive,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
