#!/usr/bin/env python3
"""Compact, opaque backup and independent restore gate for strict no-bash20.

No solver, service, judge, network, or model resources are started or opened.
Private traces/checkpoints are copied and hashed without projecting their text;
the frozen artifact verifier checks their public execution metadata. Original
server paths in manifests stay unchanged and are never followed by verification.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
import tempfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from strict_ablations import control, runner
from strict_ablations.prompt import ARMS, CONTROL, build_contract
from operations import no_bash20_suite as job

VERSION = "strict-no-bash20-snapshot-v1"
DIRECTORY, ARM, COUNT = job.DIRECTORY, job.ARM, job.COUNT
COUNTS = {"target_coverage": 20, "validated_natural_terminals": 20,
          "new_no_bash_rows": 20, "reused_control_rows": 20,
          "control_bundle_rows": 100, "public_finals": 40}
CONTROL_FILES = tuple(str(p) for p in (control.MANIFEST_PATH, control.PUBLIC_PATH,
    control.METADATA_PATH, control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH))
LOCK_FILES = (f"reports/{DIRECTORY}/worker.lock", str(job.OLD_AUTO20_ROOT / "reports/strict_auto20_v1/worker.lock"),
              "reports/strict_ablations_v1/worker.lock")
RUN_OPERATION_FILES = ("service_launch.json", "worker_launch.json", "code_deployment.json",
                       "no_bash20_config.json", "pipeline.log", "vllm_no_bash20.log")
CHECKER_FILES = ("operations/no_bash20_snapshot.py", "operations/test_no_bash20_snapshot.py")
OPERATION_FILES = frozenset({
    "operations/service_launch.json", "operations/worker_launch.json",
    "operations/code_deployment.json", "operations/docker_image_restored.json",
    "operations/no_bash20_service_launch.json", "operations/no_bash20_worker_launch.json",
    "operations/no_bash20_code_deployment.json", "operations/no_bash20_snapshot_deployment.json",
    "operations/no_bash20_pipeline.log", "operations/vllm_no_bash20.log",
    "operations/no_bash20_service.log", "operations/no_bash20_cleanup_receipt.json",
})
EXCLUDED_PARTS = {"_working", "_rerank_runtime", "venv", ".venv", "models", "weights",
                  "corpus", "index", "credentials", "secrets", ".env", ".ssh"}


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
    require(not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == relative and relative != ".", "Snapshot path is not canonical and contained")
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


def _copy_pinned(source_root, target_root, names):
    for name in names:
        source, target = _safe(source_root, str(name)), _safe(target_root, str(name))
        require(source.is_file(), "Missing bundle file: " + str(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination, source.open("rb") as original:
            os.chmod(target, 0o600)
            shutil.copyfileobj(original, destination)


def _pins(root, pins):
    require(isinstance(pins, dict) and pins, "Missing checksum map")
    for name, pin in pins.items():
        path = _safe(root, name)
        require(path.is_file() and path.stat().st_size == pin["bytes"] and
                runner.file_sha(path) == pin["sha256"], "Snapshot file bytes differ: " + name)


def compact_paths(qid):
    base = f"runs/{DIRECTORY}/{ARM}"
    return [f"{base}/_per_query/qid_{qid}.json", f"{base}/request_contracts/qid_{qid}.json",
            f"{base}/api_ledgers/qid_{qid}.jsonl", f"{base}/_traces/qid_{qid}/single.json",
            f"{base}/_checkpoints/qid_{qid}.json", f"{base}/_traces/qid_{qid}/single.api.jsonl"]


def _manifest(root, *, control_root):
    """Validate frozen admission with explicit physical roots, never server paths."""
    root = Path(root)
    manifest = runner.load(_safe(root, f"reports/{DIRECTORY}/run_manifest.json"))
    _checked(manifest, "manifest_sha256")
    require(manifest.get("version") == runner.SUITE_VERSION, "Wrong strict suite manifest version")
    runner.shared.validate_config(manifest["config"], actual_resources=False)
    require(manifest.get("config_sha256") == runner.digest(manifest["config"]) and
            manifest.get("source_sha256") == job.verify_frozen_sources() and
            len(manifest["source_sha256"]) == 135, "Frozen config/source differs")
    require(manifest["config"].get("concurrency") == 4, "The frozen C4 configuration is required")
    require(manifest.get("contracts") == {arm: build_contract(arm) for arm in ARMS} and
            manifest.get("control_model_inputs") == build_contract(CONTROL), "Frozen contracts differ")
    queries = runner.shared.validate_queries()
    require(manifest.get("cohort_sha256") == runner.shared.DATASET_SHA and
            manifest.get("query_ids") == [q["query_id"] for q in queries], "Fixed100 identity/order differs")
    suite = job.NoBash20Suite(root)
    expected = suite.execution_scope(queries)
    require(all(manifest.get(key) == value for key, value in expected.items()), "no_bash20 execution scope differs")
    require(manifest.get("analysis_contract") == suite.analysis_contract(), "Frozen scoring contract differs")
    require(manifest.get("administrative_source_sha256") == job.admin_hashes(), "Administrative source differs")
    verified_control = control.verify_control_bundle(control_root, common_config=manifest["config"])
    require(runner.control_reference(verified_control) == manifest.get("control_reference"), "Pinned B100 control differs")
    for name, sha in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        require(runner.file_sha(_safe(root, name)) == sha, "Archived source differs: " + name)
    epoch = runner.load(_safe(root, f"reports/{DIRECTORY}/service_epochs/{manifest['initial_service_epoch_sha256']}.json"))
    _checked(epoch, "epoch_sha256")
    require(epoch["epoch_sha256"] == manifest["initial_service_epoch_sha256"], "Initial service epoch differs")
    preflight = runner.load(_safe(root, f"reports/{DIRECTORY}/preflight_live.json"))
    require(preflight.get("passed") is True and preflight.get("manifest_sha256") == manifest["manifest_sha256"],
            "Passed live preflight does not bind this manifest")
    require(set(preflight.get("tools_and_native_calls", {})) == {ARM} and
            preflight["tools_and_native_calls"][ARM].get("passed") is True and
            preflight.get("cohort_questions_used") == 0 and
            preflight.get("service", {}).get("epoch_sha256") == epoch["epoch_sha256"],
            "Live preflight tool scope or service epoch differs")
    return manifest, verified_control


def _verify_rows(root, manifest):
    suite = job.NoBash20Suite(root)
    suite.assert_run_scope()
    queries = job.selected_queries()
    expected = {q["query_id"] for q in queries}
    paths = list((suite.runs / ARM / "_per_query").glob("*.json"))
    actual = {p.name for p in paths}
    require(len(paths) == COUNT and actual == {f"qid_{qid}.json" for qid in expected},
            "Exactly fixed first20 no_bash natural terminals are required")
    require({p.name for p in (suite.reports / "dispatches").iterdir()} == {f"qid_{qid}.json" for qid in expected},
            "Exactly twenty unique first-attempt dispatches are required")
    rows = []
    for query in queries:
        qid = query["query_id"]
        for name in compact_paths(qid):
            require(_safe(root, name).is_file(), "Missing compact/raw API artifact: " + name)
        row = runner.load(suite.terminal_path(ARM, qid))
        suite.validate_terminal(row, manifest, ARM, query)
        suite.verify_execution_artifacts(manifest, ARM, query, row)
        # Dispatch is an immutable single-attempt admission, not a completion counter.
        dispatch = runner.load(_safe(root, f"reports/{DIRECTORY}/dispatches/qid_{qid}.json"))
        _checked(dispatch, "dispatch_sha256")
        require(all(dispatch.get(k) == v for k, v in {
            "arm": ARM, "query_id": qid, "manifest_sha256": manifest["manifest_sha256"],
            "question_sha256": _sha(query["query"].encode()), "config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][ARM]["contract_sha256"],
            "version": job.VERSION, "attempt_id": 1}.items()), "Dispatch identity differs")
        suite.validate_dispatch(manifest, query)
        rows.append(row)
    return rows


def _completion(root, manifest, rows):
    reports = Path(root) / "reports" / DIRECTORY
    state = runner.load(reports / "phase_state.json")
    require(state.get("phase") == "complete" and state.get("manifest_sha256") == manifest["manifest_sha256"],
            "Pipeline is not complete")
    complete = runner.load(reports / "no_bash20_completion.json")
    expected = {"version": job.VERSION, "arm": ARM, "passed": True, "manifest_sha256": manifest["manifest_sha256"],
        "selected_query_ids": [q["query_id"] for q in job.selected_queries()],
        "terminal_rows_checked": COUNT, "question_runs": COUNT, "validated_natural_terminals": COUNT,
        "control_model_requests": 0, "other_arm_model_requests": 0, "natural_error_reruns": 0,
        "target_coverage": COUNT, "full_three_arm_suite_complete": False}
    require(all(complete.get(k) == v for k, v in expected.items()), "no_bash20 completion scope/count differs")
    bindings = [{"query_id": row["query_id"], "row_sha256": row["row_sha256"],
        "terminal_sha256": runner.file_sha(Path(root) / compact_paths(row["query_id"])[0]),
        "execution_artifacts": row["execution_artifacts"]} for row in rows]
    require(complete.get("terminal_rows") == bindings, "Completion does not bind all twenty natural terminals")
    require(state.get("receipt_sha256") == runner.file_sha(reports / "no_bash20_completion.json"),
            "Complete phase does not bind its receipt")
    require(state.get("active_query_ids") == [] and state.get("pending_query_ids") == [],
            "Complete phase still contains active or pending questions")
    return state, complete


def verify_worker_exit(identity, *, proc_root=Path("/proc")):
    require(isinstance(identity, dict) and type(identity.get("pid")) is int and identity["pid"] > 0 and
            str(identity.get("proc_start_ticks", "")).isdigit() and
            isinstance(identity.get("boot_id"), str) and identity["boot_id"], "Linux worker PID/start-ticks/boot identity required")
    proc_root = Path(proc_root)
    boot = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    observed = {"expected": identity, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "observed_boot_id": boot, "worker_exited": True}
    if boot != identity["boot_id"]:
        return {**observed, "reason": "different_boot"}
    try:
        stat = (proc_root / str(identity["pid"]) / "stat").read_text()
    except FileNotFoundError:
        return {**observed, "reason": "pid_absent"}
    fields = stat.rsplit(")", 1)[1].split()
    require(len(fields) > 19, "Malformed worker identity")
    observed.update(observed_start_ticks=fields[19], observed_process_state=fields[0])
    if fields[19] != str(identity["proc_start_ticks"]):
        return {**observed, "reason": "pid_reused"}
    require(fields[0] == "Z", "The experiment worker is still running")
    return {**observed, "reason": "exited_zombie"}


def _public_finals(rows, bundle):
    queries = {q["query_id"]: q for q in job.selected_queries()}
    records = [{"arm": ARM, "query_id": row["query_id"], "question": queries[row["query_id"]]["query"],
        "gold_answer": queries[row["query_id"]]["answer"], "final_text": row["final_text"],
        "final_sha256": row["final_sha256"], "row_sha256": row["row_sha256"],
        "manifest_sha256": row["manifest_sha256"], "terminated_by": row["terminated_by"],
        "archived_evidence": "complete_trace_checkpoint", "natural_terminal": True} for row in rows]
    records.extend({"arm": CONTROL, **{k: row[k] for k in ("query_id", "question", "gold_answer", "final_text",
        "final_sha256", "terminated_by")}, "row_sha256": row["terminal_row_sha256"],
        "manifest_sha256": bundle["manifest_sha256"], "archived_evidence": "public_control_bundle_only",
        "natural_terminal": True} for row in bundle["records"] if row["query_id"] in queries)
    require(len(records) == 40 and {(r["arm"], r["query_id"]) for r in records} ==
            {(arm, qid) for arm in (ARM, CONTROL) for qid in queries}, "Exactly twenty paired public finals are required")
    return sorted(records, key=lambda row: (row["arm"], row["query_id"]))


def _required(manifest):
    required = {f"reports/{DIRECTORY}/{name}" for name in ("run_manifest.json", "phase_state.json",
                "no_bash20_completion.json", "preflight_live.json", "worker.lock")}
    required.add(f"reports/{DIRECTORY}/service_epochs/{manifest['initial_service_epoch_sha256']}.json")
    required.update(manifest["source_sha256"])
    required.update(manifest["administrative_source_sha256"])
    required.update(CHECKER_FILES)
    required.update(f"reports/{DIRECTORY}/operations/{name}" for name in RUN_OPERATION_FILES)
    for query in job.selected_queries():
        required.update(compact_paths(query["query_id"]))
        required.add(f"reports/{DIRECTORY}/dispatches/qid_{query['query_id']}.json")
    return required


def _report_files(root, output):
    selected = set()
    for path in (Path(root) / "reports" / DIRECTORY).rglob("*"):
        relative = path.relative_to(root)
        if path.resolve().is_relative_to(output) or {"snapshots", "downloaded", "archives", "workspace_backups"}.intersection(relative.parts):
            continue
        allowed_log = (path.parent == Path(root) / "reports" / DIRECTORY / "operations" and
                       path.name in {"pipeline.log", "vllm_no_bash20.log"})
        if path.is_file() and (path.suffix in {".json", ".jsonl", ".lock"} or allowed_log):
            selected.add(_safe(root, relative.as_posix()))
    return selected


def _files(root, output, manifest, operation_files):
    selected = {_safe(root, name) for name in _required(manifest)} | _report_files(root, output)
    for name in operation_files:
        require(name in OPERATION_FILES or name in {f"reports/{DIRECTORY}/operations/{p}" for p in RUN_OPERATION_FILES},
                "Operation file is outside the explicit administrative allowlist")
        selected.add(_safe(root, name))
    for path in selected:
        require(path.is_file() and not EXCLUDED_PARTS.intersection(part.lower() for part in path.relative_to(root).parts),
                "Missing compact file or excluded runtime/credential path")
    return sorted(selected)


def _archive(root, files, output):
    pins = {}
    with Path(output).open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        os.chmod(output, 0o600)
        for path in files:
            relative = path.relative_to(root).as_posix()
            source = _safe(root, relative)
            require(source.is_file() and relative not in pins, "Duplicate/missing archive member")
            data = source.read_bytes()  # Opaque copy; do not decode raw traces/checkpoints.
            pins[relative] = {"sha256": _sha(data), "bytes": len(data)}
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = len(data), 0o600, int(source.stat().st_mtime)
            archive.addfile(info, io.BytesIO(data))
    return pins


def _validate_receipt(proof):
    _checked(proof, "receipt_sha256")
    require(proof.get("version") == VERSION and proof.get("counts") == COUNTS and
            proof.get("ready_for_shutdown") is True and proof.get("pipeline_phase") == "complete" and
            proof.get("worker_locks_acquired") == 3 and proof.get("active_questions") == 0 and
            proof.get("worker_lock_paths") == list(LOCK_FILES) and
            proof.get("worker_exit_verification", {}).get("worker_exited") is True,
            "Snapshot completion/safety/count scope differs")
    require(all(proof.get(k) == 0 for k in ("control_model_requests", "other_arm_model_requests", "natural_error_reruns")),
            "Forbidden solver runs or natural-terminal retries recorded")
    require(isinstance(proof.get("files"), dict) and proof["files"], "Missing snapshot checksum map")
    for name, pin in proof["files"].items():
        require(not EXCLUDED_PARTS.intersection(part.lower() for part in PurePosixPath(name).parts) and
                not name.endswith((".tar", ".tar.gz", ".tgz", ".zip")), "Excluded runtime/credential/archive member")
        require(isinstance(pin, dict) and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and
                pin["bytes"] >= 0 and isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in pin["sha256"]), "Invalid file checksum pin")


def _receipt(path, *, expected_sha256, expected_bytes):
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64 and
            type(expected_bytes) is int and expected_bytes > 0, "Independent receipt SHA and byte size are required")
    path = Path(path)
    require(runner.file_sha(path) == expected_sha256, "Downloaded independent receipt SHA differs")
    require(path.stat().st_size == expected_bytes, "Downloaded independent receipt byte size differs")
    proof = runner.load(path)
    _validate_receipt(proof)
    return proof


def _check_archive(archive, proof):
    archive = Path(archive)
    require(archive.is_file() and archive.stat().st_size == proof["archive"]["bytes"] and
            runner.file_sha(archive) == proof["archive"]["sha256"], "Downloaded archive SHA/byte size differs")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)) and set(names) == set(proof["files"]), "Unsafe or changed archive membership")
        for member in members:
            _safe(archive.parent / "_path_validation_only", member.name)
            pin = proof["files"][member.name]
            require(member.isfile() and not member.issym() and not member.islnk() and member.size == pin["bytes"],
                    "Archive contains an unsafe member type or size")
            with tar.extractfile(member) as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
                require(hasher.hexdigest() == pin["sha256"], "Archive member checksum differs")


def safe_extract(archive, receipt, output, *, expected_receipt_sha256=None, expected_receipt_bytes=None):
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    return _extract_with_proof(archive, proof, output)


def _extract_with_proof(archive, proof, output):
    _validate_receipt(proof)
    _check_archive(archive, proof)
    output = Path(output).resolve()
    require(not output.exists(), "Extract into a new, previously nonexistent directory")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            _safe(output, member.name)
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = _safe(output, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as original, target.open("xb") as destination:
                os.chmod(target, 0o600)
                shutil.copyfileobj(original, destination)
            require(runner.file_sha(target) == proof["files"][member.name]["sha256"], "Extracted member checksum differs")
    return {"safely_extracted": True, "root": str(output), "files": len(members), "archive_bytes_verified": True}


def _verify_snapshot(root, proof, archive):
    root = Path(root).resolve()
    _validate_receipt(proof)
    _check_archive(archive, proof)
    _pins(root, proof["files"])
    require(set(proof.get("bundles", {})) == {"control"}, "Physical bundle mapping differs")
    manifest, bundle = _manifest(root, control_root=_safe(root, proof["bundles"]["control"]))
    require(manifest["manifest_sha256"] == proof["manifest_sha256"] and
            manifest["source_sha256"] == proof["source_sha256"] and
            runner.file_sha(root / f"reports/{DIRECTORY}/run_manifest.json") == proof["manifest_file_sha256"],
            "Original manifest bytes/identity differ")
    required = _required(manifest) | {proof["public_finals_path"]}
    required.update(f"{proof['bundles']['control']}/{name}" for name in CONTROL_FILES)
    require(required <= set(proof["files"]), "A required artifact is outside the independent receipt")
    for name, sha in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        require(proof["files"].get(name, {}).get("sha256") == sha, "Archived source pin differs")
    require(proof["files"][CHECKER_FILES[0]]["sha256"] == proof["checker_sha256"] == runner.file_sha(Path(__file__)),
            "Archived checker differs from verifier")
    rows = _verify_rows(root, manifest)
    state, _ = _completion(root, manifest, rows)
    require(state.get("worker_identity") == proof["worker_exit_verification"]["expected"], "Worker exit proof identity differs")
    finals = runner.load(_safe(root, proof["public_finals_path"]))
    _checked(finals, "public_finals_sha256")
    require(finals.get("version") == VERSION and finals.get("manifest_sha256") == manifest["manifest_sha256"] and
            finals.get("counts") == COUNTS and finals.get("records") == _public_finals(rows, bundle), "Public final coverage differs")
    return {"verified": True, **COUNTS, "archive_bytes_verified": True,
            "source_files_verified": len(manifest["source_sha256"]), "manifest_sha256": manifest["manifest_sha256"],
            "external_model_requests": 0, "control_model_requests": 0, "other_arm_model_requests": 0}


def _publish_verified_receipt(root, receipt_path, proof, archive):
    # Service logs may continue to grow while vLLM remains available. Verify the
    # exact archived bytes in a separate physical tree, never mutable live files.
    with tempfile.TemporaryDirectory(prefix=".no_bash20_verify_", dir=Path(archive).parent) as temporary:
        extracted = Path(temporary) / "extracted"
        _extract_with_proof(archive, proof, extracted)
        verified = _verify_snapshot(extracted, proof, archive)
        require(verified.get("verified") is True, "Backup validation did not finish")
    _write_new(receipt_path, proof)
    return verified


def export_snapshot(root, output, *, control_root=None, operation_files=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root / "reports" / DIRECTORY / "snapshots") and not output.exists(),
            "Snapshot output must be a previously nonexistent directory under this run's snapshots/")
    with ExitStack() as stack:
        for path in job.NoBash20Suite(root).lock_paths():
            path = _safe(Path("/"), str(path).lstrip("/"))
            handle = stack.enter_context(path.open("rb"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = runner.load(root / f"reports/{DIRECTORY}/run_manifest.json")
        control_root = Path(control_root or header["control_root"]).resolve()
        manifest, bundle = _manifest(root, control_root=control_root)
        rows = _verify_rows(root, manifest)
        state, complete = _completion(root, manifest, rows)
        exit_proof = verify_worker_exit(state.get("worker_identity"))
        # Validate all explicitly requested paths before creating any archive output.
        selected = _files(root, output, manifest, operation_files)
        output.mkdir(parents=True, exist_ok=False)
        target = output / "bundles/control"
        _copy_pinned(control_root, target, CONTROL_FILES)
        copied = control.verify_control_bundle(target, common_config=manifest["config"])
        require(runner.control_reference(copied) == manifest["control_reference"], "Copied control differs")
        finals_path = output / "public_finals.json"
        finals = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"], "counts": COUNTS,
                  "records": _public_finals(rows, copied),
                  "policy": "Fresh blinded review required; control is reused historical public evidence only"}
        finals["public_finals_sha256"] = runner.digest(finals)
        _write_new(finals_path, finals)
        files = sorted({*selected, finals_path, *(_safe(target, name) for name in CONTROL_FILES)})
        archive = output / "no_bash20_snapshot.tar.gz"
        pins = _archive(root, files, archive)
        require(_verify_rows(root, manifest) == rows and _completion(root, manifest, rows) == (state, complete),
                "Terminals or completion changed during backup")
        proof = {"version": VERSION, "ready_for_shutdown": True, "counts": COUNTS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": runner.file_sha(root / f"reports/{DIRECTORY}/run_manifest.json"),
            "source_sha256": manifest["source_sha256"], "files": pins,
            "bundles": {"control": target.relative_to(root).as_posix()},
            "public_finals_path": finals_path.relative_to(root).as_posix(),
            "archive": {"path": archive.relative_to(root).as_posix(), "sha256": runner.file_sha(archive),
                        "bytes": archive.stat().st_size},
            "worker_exit_verification": exit_proof, "worker_locks_acquired": 3,
            "worker_lock_paths": list(LOCK_FILES), "pipeline_phase": "complete", "active_questions": 0,
            "control_model_requests": 0, "other_arm_model_requests": 0, "natural_error_reruns": 0,
            "checker_sha256": runner.file_sha(Path(__file__)),
            "exclusions": ["working/runtime document pools", "weights", "venv", "corpus", "index", "credentials",
                           "prior archives", "archive itself", "receipt itself"]}
        proof["receipt_sha256"] = runner.digest(proof)
        receipt_path = output / "snapshot_receipt.json"
        _publish_verified_receipt(root, receipt_path, proof, archive)
    return {"ready_for_shutdown": True, **COUNTS, "archive": str(archive),
        "archive_sha256": proof["archive"]["sha256"], "archive_bytes": proof["archive"]["bytes"],
        "receipt": str(receipt_path), "receipt_file_sha256": runner.file_sha(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size, "files": len(pins), "external_model_requests": 0}


def load_verified_snapshot(root, *, receipt, archive, expected_receipt_sha256=None, expected_receipt_bytes=None):
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    return _verify_snapshot(root, proof, archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--control-root", type=Path)
    export.add_argument("--operation-file", action="append", default=[])
    for name in ("extract", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--expected-receipt-sha256", required=True)
        command.add_argument("--expected-receipt-bytes", type=int, required=True)
        if name == "extract":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_snapshot(args.root, args.output_dir, control_root=args.control_root, operation_files=args.operation_file)
    elif args.command == "extract":
        result = safe_extract(args.archive, args.receipt, args.output_dir,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    else:
        result = load_verified_snapshot(args.root, receipt=args.receipt, archive=args.archive,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
