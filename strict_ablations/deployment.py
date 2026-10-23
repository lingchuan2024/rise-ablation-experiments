"""Read-only deployment epochs for the frozen B100 serving/capacity identity.

The old service_config and capacity report remain byte-for-byte unchanged.
Only PID/start ticks and their derived identity digest may change. No service
is launched and no generation/tokenization request is made. The existing
identity collector reads /proc, package/source metadata, and GET /models.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from strict_experiment import deployment_identity as identity
from strict_experiment import runner as shared

VERSION = "strict-ablations-deployment-epoch-v1"
LIFECYCLE_FIELDS = frozenset({"pid", "process_start_ticks", "identity_sha256"})
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PROOF_KEYS = {
    "version", "config_sha256", "baseline_resource_identity_sha256",
    "service_config_sha256", "previous_identity_sha256", "actual_identity",
    "static_identity_sha256", "allowed_lifecycle_fields", "capacity_report_sha256",
    "capacity_report_identity", "host_capacity_equivalence", "operator_source_sha256",
    "created_unix", "epoch_sha256",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(value, label):
    _require(type(value) is str and bool(_HEX.fullmatch(value)), "Invalid " + label + " SHA256")
    return value


def static_identity(receipt):
    _require(type(receipt) is dict, "Missing serving identity")
    claimed = _sha(receipt.get("identity_sha256"), "serving identity")
    _require(claimed == identity.digest({k: v for k, v in receipt.items() if k != "identity_sha256"}),
             "Invalid serving identity digest")
    _require(type(receipt.get("pid")) is int and receipt["pid"] > 0
        and type(receipt.get("process_start_ticks")) is int and receipt["process_start_ticks"] >= 0,
        "Invalid serving process lifecycle")
    return {k: v for k, v in receipt.items() if k not in LIFECYCLE_FIELDS}


def assert_lifecycle_only(previous, actual):
    """Equivalent to the B-only restart policy, without patching legacy code."""
    before, after = static_identity(previous), static_identity(actual)
    _require(identity.digest(before) == identity.digest(after),
             "Restart changed static model/argv/parser/template/package/source identity")
    return identity.digest(after)


def _frozen_inputs(config, baseline_resource_identity):
    shared.validate_config(config, actual_resources=True)
    baseline = baseline_resource_identity
    _require(type(baseline) is dict and baseline.get("model_revision") == config["model_identity"]["revision"],
             "Baseline model revision differs")
    model_files = baseline.get("model_files")
    _require(type(model_files) is dict, "Missing B100 static model file identities")
    expected_service_sha = _sha(model_files.get("service_config"), "B100 service config")
    service_path = Path(config["model_identity"]["service_config"])
    _require(service_path.is_file() and shared.file_sha(service_path) == expected_service_sha,
             "Frozen B100 service_config bytes changed; preserve the old file")
    source = shared.load(service_path)
    previous = source.get("actual_identity")
    static_identity(previous)
    _require(previous.get("version") == identity.VERSION, "Wrong original serving identity version")
    # The full pinned file authenticates the old command and actual_identity;
    # collect_service_identity independently checks the current command/files.
    _require(source.get("command") == previous.get("configured_command"),
             "Original serving command/identity disagree")

    capacity = baseline.get("capacity_report")
    _require(type(capacity) is dict and capacity.get("passed") is True,
             "Missing passed B100 capacity attestation")
    expected_capacity_sha = _sha(capacity.get("sha256"), "B100 capacity report")
    capacity_path = Path(config["capacity_report"])
    _require(capacity.get("path") == str(capacity_path.resolve()), "Capacity report path differs from B100")
    _require(capacity_path.is_file() and shared.file_sha(capacity_path) == expected_capacity_sha,
             "Frozen B100 capacity report bytes changed")
    # Read-only checks include report/config/service/source digests, the exact
    # two C4 workloads, model/usage validation and measured coexistence. This
    # helper never invokes capacity_probe.execute or sends a model request.
    measured_capacity = shared.verify_capacity_report(config)
    _require(identity.digest(measured_capacity) == identity.digest(capacity),
             "B100 capacity identity/config/source/workload differs")
    return previous, measured_capacity, expected_service_sha


def _host_snapshot():
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return {"system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "physical_memory_bytes": page_size * pages}


def host_capacity_evidence(config):
    """Reuse the old measured scope on matching GPU SKU/memory and image.

    The old capacity JSON did not hash its telemetry file or record a kernel
    driver/CPU/RAM baseline. Record those limits explicitly; bind telemetry
    bytes and the current host within this new epoch, without inventing history.
    """
    capacity = shared.load(config["capacity_report"])
    telemetry_path = Path(capacity["telemetry"]["path"])
    telemetry_bytes = telemetry_path.read_bytes()
    original_gpu = None
    for line in telemetry_bytes.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows = record.get("gpu", {}).get("rows", [])
        if rows:
            original_gpu = [{"index": int(row[0]), "name": row[1].strip(),
                "memory_total_mib": int(float(row[2]))} for row in rows]
            break
    _require(bool(original_gpu), "Original capacity GPU telemetry is missing")
    measured = subprocess.run(["nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    gpu = [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
        "memory_total_mib": int(float(row[3])), "driver_version": row[4].strip()}
        for row in csv.reader(measured.stdout.splitlines()) if row]
    _require(bool(gpu) and [{k: row[k] for k in ("index", "name", "memory_total_mib")} for row in gpu] == original_gpu,
             "Current GPU index/SKU/memory differs from the measured C4 host")
    _require(all(row["uuid"] and row["driver_version"] for row in gpu), "Missing current GPU metadata")
    _require(config["sandbox_backend"] == "docker", "B100 epoch requires the original Docker sandbox")
    environment = {**os.environ, **config["runtime_environment"]}
    image = environment["RISE_DOCKER_IMAGE"]
    actual_image = subprocess.run(["/usr/bin/docker", "image", "inspect", image, "--format", "{{.Id}}"],
        env=environment, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    _require(actual_image == image, "Current Docker image differs from B100")
    return {"passed": True, "original_capacity_sha256": shared.file_sha(config["capacity_report"]),
        "original_telemetry_sha256": hashlib.sha256(telemetry_bytes).hexdigest(),
        "original_telemetry_path": str(telemetry_path.resolve()), "original_gpu": original_gpu,
        "gpu": gpu, "docker_image_id": actual_image, "host": _host_snapshot(),
        "historical_telemetry_binding": "observed historical telemetry bytes; original capacity JSON did not record their hash",
        "host_history": "current driver/UUID/kernel/CPU/RAM recorded; absent historical fields are not claimed equal",
        "capacity_scope": capacity["certification_scope"], "long_capacity_probe_repeated": False}


def prepare_epoch(config, baseline_resource_identity, *, pid=None) -> dict:
    """Observe an existing service; CLI callers must supply --service-pid.

    There is deliberately no PID guess or implicit old-PID reuse. Persist the
    returned proof under the new suite namespace, never in old B100 files.
    """
    _require(type(pid) is int and pid > 0, "An explicit positive --service-pid is required")
    previous, capacity, service_sha = _frozen_inputs(config, baseline_resource_identity)
    actual = identity.collect_service_identity(config["model_identity"]["service_config"], pid)
    static_sha = assert_lifecycle_only(previous, actual)
    _require(actual["pid"] == pid, "Collected identity belongs to another PID")
    host = host_capacity_evidence(config)
    _require(host.get("passed") is True and host.get("original_capacity_sha256") == capacity["sha256"],
             "Current host capacity attestation is missing or mismatched")
    payload = {"version": VERSION, "config_sha256": identity.digest(config),
        "baseline_resource_identity_sha256": identity.digest(baseline_resource_identity),
        "service_config_sha256": service_sha,
        "previous_identity_sha256": previous["identity_sha256"], "actual_identity": actual,
        "static_identity_sha256": static_sha, "allowed_lifecycle_fields": sorted(LIFECYCLE_FIELDS),
        "capacity_report_sha256": capacity["sha256"], "capacity_report_identity": capacity,
        "host_capacity_equivalence": host, "operator_source_sha256": shared.file_sha(Path(__file__)),
        "created_unix": time.time()}
    return {**payload, "epoch_sha256": identity.digest(payload)}


def verify_epoch(config, baseline_resource_identity, proof) -> dict:
    """Recheck the same epoch, rejecting PID reuse, host or static drift."""
    _require(type(proof) is dict and set(proof) == _PROOF_KEYS and proof.get("version") == VERSION,
             "Invalid ablation deployment epoch proof")
    _require(_sha(proof.get("epoch_sha256"), "deployment epoch") ==
        identity.digest({k: v for k, v in proof.items() if k != "epoch_sha256"}), "Deployment epoch digest changed")
    _require(proof.get("config_sha256") == identity.digest(config)
        and proof.get("baseline_resource_identity_sha256") == identity.digest(baseline_resource_identity),
        "Deployment epoch belongs to another config/B100 baseline")
    _require(proof.get("operator_source_sha256") == shared.file_sha(Path(__file__)),
             "Deployment epoch verifier source changed")
    _require(proof.get("allowed_lifecycle_fields") == sorted(LIFECYCLE_FIELDS), "Lifecycle exception policy changed")
    previous, capacity, service_sha = _frozen_inputs(config, baseline_resource_identity)
    _require(proof.get("service_config_sha256") == service_sha
        and proof.get("previous_identity_sha256") == previous["identity_sha256"]
        and proof.get("capacity_report_sha256") == capacity["sha256"]
        and identity.digest(proof.get("capacity_report_identity")) == identity.digest(capacity),
        "Deployment epoch frozen service/capacity bindings changed")
    saved = proof.get("actual_identity")
    static_identity(saved)
    actual = identity.collect_service_identity(config["model_identity"]["service_config"], saved["pid"])
    _require(identity.digest(actual) == identity.digest(saved), "Epoch process/software identity drifted or PID was reused")
    _require(assert_lifecycle_only(previous, actual) == proof.get("static_identity_sha256"),
             "Deployment epoch static identity differs")
    host = host_capacity_evidence(config)
    _require(host.get("passed") is True and identity.digest(host) == identity.digest(proof.get("host_capacity_equivalence")),
             "Epoch GPU/Docker/host/telemetry capacity evidence changed")
    return {"verified": True, "version": identity.VERSION, "identity_sha256": actual["identity_sha256"],
        "pid": actual["pid"], "process_start_ticks": actual["process_start_ticks"],
        "model_root": actual["model_root"], "packages": actual["packages"],
        "deployment_epoch_sha256": proof["epoch_sha256"],
        "static_identity_sha256": proof["static_identity_sha256"],
        "capacity_reused": True, "long_capacity_probe_repeated": False}
