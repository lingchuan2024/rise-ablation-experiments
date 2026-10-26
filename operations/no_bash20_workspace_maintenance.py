#!/usr/bin/env python3
"""Administrative, terminal-only workspace backup for this frozen no-bash20 run.

The frozen recovery maintenance algorithm is loaded into a private module; its
archive, member verification, acknowledgement, inode and content checks are
unchanged. No solver/configuration/source file is modified. Bodies remain opaque.
This does not authorize active/partial questions or any other namespace.

Live phase/pending checks precede every unlink/rmdir. They fail closed when the
tail batch begins, but are not a cooperative runner lock: operators must finish
maintenance while undispatched questions remain, before final resource scanning.
Guard failures and Python interruptions after an unlink publish a partial receipt;
an abrupt process kill can prevent receipt publication. Never automatically retry
a partial deletion: the original tree no longer matches its deletion manifest.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMESPACE = "strict_no_bash20_v1"
ARM = "no_bash"
VERSION = "no-bash20-workspace-maintenance-v1"
CORE_SHA256 = "f50a9fd4d53fdf34d1b4bd752a902cabd6d292bdbf28a2af5155de94868fbf81"
MANIFEST_SHA256 = "cc56f24667dbe2100f2b58c3d52ab032db15c27f5e041c40b67bca404d53ee3c"
CONFIG_SHA256 = "4066e6ae83c5427924a3eb3577eb5c5621a4372b5be673852667b64fe8433c1a"
QIDS = frozenset((70, 81, 96, 116, 122, 127, 132, 153, 156, 161, 193,
                  209, 228, 229, 239, 245, 246, 251, 256, 270))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def core():
    path = ROOT / "operations/terminal_workspace_maintenance.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA256,
            "Frozen maintenance algorithm changed")
    spec = importlib.util.spec_from_file_location("_private_no_bash20_workspace_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.NAMESPACE, module.VERSION = NAMESPACE, VERSION
    module.audit_terminal = audit_terminal
    module.operation_lock = operation_lock
    return module


def authorized(arm, qid):
    require(arm == ARM and type(qid) is int and qid in QIDS,
            "Only explicit no_bash fixed-first20 questions are admitted")


def suite_for(root):
    from operations.no_bash20_suite import NoBash20Suite
    suite = NoBash20Suite(root)
    require(suite.directory == NAMESPACE and suite.reports == Path(root).resolve() / "reports" / NAMESPACE,
            "Wrong no-bash namespace")
    return suite


def audit_terminal(root, arm, qid):
    """Validate actual published terminal, dispatch, contract, ledger and artifacts."""
    from operations.no_bash20_suite import selected_queries
    authorized(arm, qid)
    c, suite = core_helpers(), suite_for(root)
    manifest = suite.verify_manifest(resources=False)
    require(manifest["manifest_sha256"] == MANIFEST_SHA256 and manifest["config_sha256"] == CONFIG_SHA256,
            "Wrong frozen no-bash manifest/configuration")
    query = next((q for q in selected_queries() if q["query_id"] == qid), None)
    require(query is not None, "Question is outside frozen first20")
    expected = Path(root).resolve() / "runs" / NAMESPACE / ARM / "_per_query" / f"qid_{qid}.json"
    path = suite.terminal_path(arm, qid)
    require(path == expected, "Historical or other namespace terminal is forbidden")
    c.safe(root, path.relative_to(Path(root).resolve()).as_posix())
    require(path.is_file(), "No natural terminal: active/partial workspaces are forbidden")
    infra = path.parent.parent / "_infrastructure" / f"qid_{qid}.json"
    require(not infra.exists() and not list(infra.parent.glob(infra.name + ".*")),
            "Infrastructure interruption evidence forbids maintenance")
    require(not any(list((path.parent.parent / name).glob(f"qid_{qid}.json.*"))
                    for name in ("_per_query", "_checkpoints", "request_contracts")),
            "Temporary publication evidence forbids maintenance")
    dispatch_path = suite.dispatch_path(qid)
    c.safe(root, dispatch_path.relative_to(Path(root).resolve()).as_posix())
    dispatch = suite.validate_dispatch(manifest, query)
    terminal = c.load(path)
    suite.validate_terminal(terminal, manifest, arm, query)
    suite.verify_execution_artifacts(manifest, arm, query, terminal)
    return {"source_root": str(Path(root).resolve()), "namespace": NAMESPACE, "arm": arm,
            "query_id": qid, "manifest_sha256": manifest["manifest_sha256"],
            "effective_config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][arm]["contract_sha256"],
            "dispatch_sha256": dispatch["dispatch_sha256"], "dispatch_file_sha256": c.file_sha(dispatch_path),
            "worker_identity": dispatch["worker_identity"], "terminal_sha256": c.file_sha(path),
            "row_sha256": terminal["row_sha256"], "execution_artifacts": terminal["execution_artifacts"]}


@lru_cache(maxsize=1)
def core_helpers():
    # Private loading keeps the old namespace/module globals unchanged.
    return core()


def live_gate(root, binding):
    """Cheap repeated gate; importantly, no corpus resource/stat scan is started."""
    from operations.no_bash20_suite import process_matches
    c, suite = core_helpers(), suite_for(root)
    state = c.load(suite.reports / "phase_state.json")
    age = time.time() - state.get("checked_unix", 0)
    require(0 <= age <= 90, "Stale live phase cannot authorize maintenance")
    require(state.get("phase") in {"running_no_bash20", "waiting_storage"},
            "Maintenance requires running/waiting-storage phase")
    require(state.get("manifest_sha256") == binding["manifest_sha256"] and
            state.get("worker_identity") == binding["worker_identity"] and
            process_matches(state.get("worker_identity")), "Original worker is not live or identity changed")
    require(not state.get("errors") and not (suite.reports / "no_bash20_completion.json").exists(),
            "Error/completion barrier forbids maintenance")
    active, pending = state.get("active_query_ids", []), state.get("pending_query_ids", [])
    require(isinstance(active, list) and len(active) <= 4 and len(set(active)) == len(active)
            and all(type(q) is int and q in QIDS for q in active), "Invalid active question state")
    require(binding["query_id"] not in active and binding["query_id"] not in pending,
            "Selected question is still active/pending")
    require(isinstance(pending, list) and pending and len(set(pending)) == len(pending)
            and all(type(q) is int and q in QIDS for q in pending), "No undispatched work remains; tail cleanup forbidden")
    # A state file can lag dispatch. Admit only pending entries with no actual
    # dispatch or any execution evidence, rather than trusting its count alone.
    require(any(not suite.evidence_present(q) for q in pending),
            "Pending state is stale: all remaining questions were dispatched")


@contextmanager
def operation_lock(root, arm, qid):
    authorized(arm, qid)
    c = core_helpers()
    path = c.safe(root, f"reports/{NAMESPACE}/workspace_maintenance_locks/namespace.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "Aliased maintenance lock")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_binding(binding):
    authorized(binding.get("arm"), binding.get("query_id"))
    require(binding.get("namespace") == NAMESPACE and binding.get("manifest_sha256") == MANIFEST_SHA256
            and binding.get("effective_config_sha256") == CONFIG_SHA256,
            "Receipt is not from this frozen no-bash namespace")


def admission_gate(root, binding):
    """Full immutable-artifact status once before an export/deletion operation."""
    status = suite_for(root).status()
    require(status.get("prepared") is True and status.get("worker_identity_live") is True
            and status.get("worker_identity") == binding["worker_identity"]
            and status.get("phase") in {"running_no_bash20", "waiting_storage"},
            "Full status does not admit live maintenance")
    arm = status.get("arms", {}).get(ARM, {})
    terminal, partial = arm.get("terminal", []), arm.get("partial_or_active", [])
    active, pending = arm.get("active", []), arm.get("pending", [])
    require(binding["query_id"] in terminal and binding["query_id"] not in partial
            and binding["query_id"] not in active and binding["query_id"] not in pending
            and pending and set(active) == set(partial) and len(active) <= 4,
            "Terminal/active/partial/pending status is inconsistent")
    require(len(terminal) == len(set(terminal)) and not set(terminal).intersection(partial)
            and not set(terminal).intersection(pending) and not set(partial).intersection(pending),
            "Overlapping or duplicate question status")
    live_gate(root, binding)


def export_workspace(root, arm, qid, output):
    authorized(arm, qid)
    binding = audit_terminal(root, arm, qid)
    admission_gate(root, binding)
    return core().export_workspace(root, arm, qid, output)


def verify_archive(archive, receipt, *, expected_receipt_sha256=None, expected_receipt_bytes=None):
    c = core()
    result = c.verify_archive(archive, receipt, expected_receipt_sha256=expected_receipt_sha256,
                              expected_receipt_bytes=expected_receipt_bytes)
    validate_binding(result["binding"])
    return result


class GuardedOS:
    """Only this private core gets guarded unlink/rmdir; global os is untouched."""
    def __init__(self, root, binding):
        self.root, self.binding = root, binding
        self.files_removed = self.directories_removed = 0

    def __getattr__(self, name):
        return getattr(os, name)

    def unlink(self, *args, **kwargs):
        live_gate(self.root, self.binding)
        result = os.unlink(*args, **kwargs)
        self.files_removed += 1
        return result

    def rmdir(self, *args, **kwargs):
        live_gate(self.root, self.binding)
        result = os.rmdir(*args, **kwargs)
        self.directories_removed += 1
        return result


def delete_workspace(root, arm, qid, receipt, acknowledgement):
    authorized(arm, qid)
    c = core()
    record = c.load(receipt)
    require(record.get("version") == VERSION and acknowledgement.get("version") == VERSION,
            "Other maintenance versions cannot authorize deletion")
    validate_binding(record["binding"])
    binding = audit_terminal(root, arm, qid)
    admission_gate(root, binding)
    proxy = GuardedOS(root, binding)
    c.os = proxy
    try:
        return c.delete_workspace(root, arm, qid, receipt, acknowledgement)
    except BaseException as exc:
        if proxy.files_removed or proxy.directories_removed:
            result = {"version": VERSION, "deleted": False, "partial_maintenance": True,
                      "binding": binding, "files_removed": proxy.files_removed,
                      "directories_removed": proxy.directories_removed, "error_type": type(exc).__name__,
                      "full_verified_local_backup_retained": True, "finished_unix": time.time()}
            result["interruption_sha256"] = c.digest(result)
            # Never overwrite evidence or imply a successful complete deletion.
            c.write_new(Path(receipt).parent / f"maintenance_interrupted_{time.time_ns()}.json", result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True, choices=[ARM]); p.add_argument("--qid", required=True, type=int)
        if name == "export": p.add_argument("--output-dir", type=Path, required=True)
        else:
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--acknowledgement-stdin", action="store_true", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--archive", type=Path, required=True); p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--expected-receipt-sha256", required=True); p.add_argument("--expected-receipt-bytes", required=True, type=int)
    p.add_argument("--acknowledgement-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export": result = export_workspace(args.root, args.arm, args.qid, args.output_dir)
    elif args.command == "delete": result = delete_workspace(args.root, args.arm, args.qid, args.receipt, json.load(sys.stdin))
    else:
        result = verify_archive(args.archive, args.receipt, expected_receipt_sha256=args.expected_receipt_sha256,
                                expected_receipt_bytes=args.expected_receipt_bytes)
        core().write_new(args.acknowledgement_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
