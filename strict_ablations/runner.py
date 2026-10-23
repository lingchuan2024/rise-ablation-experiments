"""Freeze/run three minimal ablations; reuse B100 without rerunning control.

Offline commands make no model requests. Live commands require an existing,
identity-verified service; this module never starts a server or obtains secrets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen

from strict_experiment import runner as shared
from strict_ablations import deployment
from strict_ablations.control import verify_control_bundle
from strict_ablations.prompt import ARMS, CONTROL, VERSION, build_contract, verify_contracts

ROOT = Path(__file__).resolve().parents[1]
SUITE_VERSION = "strict-three-ablations-suite-v1"
DIRECTORY = "strict_ablations_v1"
MODEL = shared.MODEL
digest, file_sha, load, atomic_json = shared.digest, shared.file_sha, shared.load, shared.atomic_json


def source_hashes():
    # Do not modify the old source list or any old frozen Python file.
    return {**shared.source_hashes(), **{
        str(p.relative_to(ROOT)): file_sha(p)
        for p in sorted((ROOT / "strict_ablations").rglob("*.py"))}}


def control_reference(bundle):
    return {k: bundle[k] for k in (
        "version", "manifest_sha256", "public_export_sha256", "integrity_receipt_sha256",
        "model", "cohort_sha256", "contract_sha256", "config_sha256")}


def export_control_bundle(control_root, output):
    """Copy only the six pinned public files; no private runs or credentials."""
    from strict_ablations import control
    control_root, output = Path(control_root).resolve(), Path(output).resolve()
    verified = verify_control_bundle(control_root)
    # An explicit new directory avoids overwriting an old experiment or export.
    output.mkdir(parents=True, exist_ok=False)
    for name in (control.MANIFEST_PATH, control.PUBLIC_PATH, control.METADATA_PATH,
                 control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            os.chmod(target, 0o600)
            stream.write((control_root / name).read_bytes())
    copied = verify_control_bundle(output)
    if control_reference(copied) != control_reference(verified):
        raise RuntimeError("Copied public control bundle failed verification")
    return {"output": str(output), "files": 6, "control_reference": control_reference(copied),
            "private_trajectories_included": False, "model_requests": 0}


def same_static_resources(current, baseline):
    if digest(current) != digest(baseline):
        raise ValueError("Static model/corpus/index/sandbox/capacity provenance differs from the pinned control")
    # Preserve the entire old service receipt. A separate lifecycle proof
    # permits a new PID without relaxing software/parser/argv identity.


class Suite:
    def __init__(self, root=None):
        self.root = Path(root or ROOT).resolve()
        self.reports = self.root / "reports" / DIRECTORY
        self.runs = self.root / "runs" / DIRECTORY

    def control_bundle(self, manifest):
        bundle = verify_control_bundle(manifest["control_root"], common_config=manifest["config"])
        if control_reference(bundle) != manifest["control_reference"]:
            raise RuntimeError("Reused B100 control identity changed")
        return bundle

    def template(self, control_root):
        bundle = verify_control_bundle(control_root)
        return {"version": SUITE_VERSION, "control_root": str(Path(control_root).resolve()),
                "common_config": copy.deepcopy(bundle["common_config"])}

    def offline_preflight(self, config=None, *, control_root=None):
        rows = shared.validate_queries()
        prompt = verify_contracts()
        for name in source_hashes():
            if name.endswith(".py"):
                compile((ROOT / name).read_bytes(), name, "exec")
        bundle = None
        if config is not None:
            if set(config) != {"version", "control_root", "common_config"} or config["version"] != SUITE_VERSION:
                raise ValueError("Use the independent strict ablation configuration")
            shared.validate_config(config["common_config"])
            bundle = verify_control_bundle(config["control_root"], common_config=config["common_config"])
        elif control_root is not None:
            bundle = verify_control_bundle(control_root)
        return {"version": SUITE_VERSION, "passed": True, "live_ready": False,
                "model_requests": 0, "arms": list(ARMS), "new_question_runs": 300,
                "reused_control_questions": 100, "fixed100": len(rows),
                "pilot_query_ids": [r["query_id"] for r in rows[:10]],
                "prompt_checks": prompt,
                "control_reference": control_reference(bundle) if bundle else None}

    def prepare(self, config, *, service_pid=None):
        self.offline_preflight(config)
        if (self.reports / "run_manifest.json").exists() or self.runs.exists():
            raise RuntimeError("An existing ablation output/manifest cannot be rebound")
        common = config["common_config"]
        bundle = verify_control_bundle(config["control_root"], common_config=common)
        control_manifest = load(Path(config["control_root"]) / "reports/strict_qwen35/run_manifest.json")
        if shared.source_hashes() != control_manifest["source_sha256"]:
            raise ValueError("Frozen shared runtime code differs from the B100 control")
        current = shared.resource_identity(common)
        same_static_resources(current, control_manifest["resource_identity"])
        epoch = deployment.prepare_epoch(common, current, pid=service_pid)
        from strict_ablations import analyze
        rows = shared.validate_queries()
        payload = {"version": SUITE_VERSION, "prompt_version": VERSION,
            "config": copy.deepcopy(common), "config_sha256": digest(common),
            "control_root": str(Path(config["control_root"]).resolve()),
            "control_reference": control_reference(bundle),
            "control_model_inputs": build_contract(CONTROL),
            "contracts": {arm: build_contract(arm) for arm in ARMS},
            "cohort_sha256": shared.DATASET_SHA,
            "query_ids": [r["query_id"] for r in rows],
            "pilot_query_ids": [r["query_id"] for r in rows[:10]],
            "source_sha256": source_hashes(), "resource_identity": current,
            "initial_service_epoch_sha256": epoch["epoch_sha256"],
            "question_runs": 300, "reused_control_questions": 100,
            "analysis_contract": {"version": analyze.VERSION,
                                  "rubric_sha256": digest(analyze.rubric_document())},
            "scheduling": "original question order; three fixed arms; shared global C4",
            "terminal_policy": "immutable; all natural failures included; same-checkpoint infrastructure resume only",
            "causal_scope": "historical B100 control; matched static resources/config; process epochs are separate",
            "auto_rerank_scope": "coupled retrieval/rerank policy; not an isolated neural-ranking ablation"}
        manifest = {**payload, "manifest_sha256": digest(payload)}
        self.save_epoch(epoch)
        atomic_json(self.reports / "run_manifest.json", manifest, immutable=True)
        return manifest

    def save_epoch(self, epoch):
        path = self.reports / "service_epochs" / (epoch["epoch_sha256"] + ".json")
        if path.exists():
            if load(path) != epoch:
                raise RuntimeError("An immutable service epoch changed")
        else:
            atomic_json(path, epoch, immutable=True)
        atomic_json(self.reports / "service_epoch.json", epoch)

    def verify_service(self, manifest):
        config = manifest["config"]
        epoch = load(self.reports / "service_epoch.json")
        if load(self.reports / "service_epochs" / (epoch["epoch_sha256"] + ".json")) != epoch:
            raise RuntimeError("Current service epoch is not archived")
        observed = deployment.verify_epoch(config, manifest["resource_identity"], epoch)
        contract = manifest["control_model_inputs"]
        payload = {"model": MODEL, "messages": [
            {"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": "Compatibility check only."}],
            "tools": contract["tool_specs"], "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True}}
        request = Request(config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as stream:
            tokenization = json.load(stream)
        if type(tokenization.get("count")) is not int or tokenization.get("max_model_len") != config["context_length"]:
            raise RuntimeError("The serving tokenizer/context budget differs from the control")
        return {"deployment_identity": observed, "epoch_sha256": epoch["epoch_sha256"],
                "context_length": tokenization["max_model_len"], "probe_prompt_tokens": tokenization["count"]}

    def verify_manifest(self, *, resources=True):
        manifest = load(self.reports / "run_manifest.json")
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest.get("version") != SUITE_VERSION or digest(payload) != manifest.get("manifest_sha256"):
            raise RuntimeError("Invalid independent ablation manifest")
        shared.validate_config(manifest["config"], actual_resources=resources)
        if digest(manifest["config"]) != manifest["config_sha256"] or source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("Frozen common configuration/source changed")
        expected = {arm: build_contract(arm) for arm in ARMS}
        if manifest["contracts"] != expected or manifest["control_model_inputs"] != build_contract(CONTROL):
            raise RuntimeError("Frozen prompt/tool contracts changed")
        rows = shared.validate_queries()
        if manifest["cohort_sha256"] != shared.DATASET_SHA or manifest["query_ids"] != [r["query_id"] for r in rows]:
            raise RuntimeError("Fixed100 questions/order changed")
        self.control_bundle(manifest)
        initial = load(self.reports / "service_epochs" / (manifest["initial_service_epoch_sha256"] + ".json"))
        if (initial.get("epoch_sha256") != manifest["initial_service_epoch_sha256"] or
                digest({k: v for k, v in initial.items() if k != "epoch_sha256"}) != initial["epoch_sha256"]):
            raise RuntimeError("Frozen initial process epoch is missing")
        from strict_ablations import analyze
        if manifest["analysis_contract"] != {"version": analyze.VERSION, "rubric_sha256": digest(analyze.rubric_document())}:
            raise RuntimeError("Scoring rules changed after freeze")
        if resources and shared.resource_identity(manifest["config"]) != manifest["resource_identity"]:
            raise RuntimeError("Frozen live resources/service/capacity changed")
        return manifest

    def terminal_path(self, arm, qid):
        if arm not in ARMS:
            raise ValueError("Only the three new variants may have new terminal rows")
        return self.runs / arm / "_per_query" / f"qid_{qid}.json"

    def validate_terminal(self, row, manifest, arm, query):
        if arm not in ARMS:
            raise ValueError("Control must be reused, never scheduled")
        expected = {"version": SUITE_VERSION, "arm": arm, "model": MODEL,
            "query_id": query["query_id"], "manifest_sha256": manifest["manifest_sha256"],
            "config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][arm]["contract_sha256"],
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest()}
        if any(row.get(k) != v for k, v in expected.items()) or row.get("row_sha256") != digest({k: v for k, v in row.items() if k != "row_sha256"}):
            raise RuntimeError("Terminal identity/checksum differs from frozen arm")
        final = row.get("final_text")
        if not isinstance(final, str) or row.get("final_sha256") != hashlib.sha256(final.encode()).hexdigest():
            raise RuntimeError("Public final bytes changed")
        reason = row.get("terminated_by")
        if not isinstance(reason, str) or not reason or reason.startswith(("api_error", "infrastructure", "solver_api_failed")):
            raise RuntimeError("Infrastructure interruption is not a natural terminal")
        if any(row.get("execution_checks", {}).get(k) is not True for k in
               ("contract_verified", "model_verified", "token_ledger_verified")):
            raise RuntimeError("Actual request/model/token verification is missing")
        from strict_experiment.analyze import row_metrics
        row_metrics(row, "B")
        return row

    def verify_execution_artifacts(self, manifest, arm, query, terminal):
        output = self.runs / arm
        qid = query["query_id"]
        contract = manifest["contracts"][arm]
        path = output / "request_contracts" / f"qid_{qid}.json"
        record = load(path)
        expected = {"arm": arm, "query_id": qid, "model": MODEL,
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "user_prompt": contract["user_prompt_template"].format(question=query["query"]),
            "common_config_sha256": digest(manifest["config"]),
            **{k: contract[k] for k in ("system_prompt", "tool_specs", "system_prompt_sha256",
                                       "tool_specs_sha256", "contract_sha256")}}
        if any(record.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Actual sent prompt/tools differ from frozen contract")
        ledger_path = output / "api_ledgers" / f"qid_{qid}.jsonl"
        entries = [json.loads(s) for s in ledger_path.read_text().splitlines() if s.strip()]
        successful = [r for r in entries if r.get("status") == "ok"]
        if not successful or any(r.get("contract_sha256") != contract["contract_sha256"] or r.get("requested_model") != MODEL for r in entries):
            raise RuntimeError("API requests are missing or use a different contract/model")
        for r in successful:
            if r.get("reported_model") != MODEL or any(type((r.get("usage") or {}).get(k)) is not int or r["usage"][k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                raise RuntimeError("Successful response model/usage is invalid")
        totals = {k: sum(r["usage"][v] for r in successful) for k, v in
                  (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"))}
        if any(terminal.get(k) != v for k, v in totals.items()):
            raise RuntimeError("Actual numeric ledger differs from terminal totals")
        counts = {"attempts": len(entries), "successful_calls": len(successful),
                  "attempts_without_usage": sum(not isinstance(r.get("usage"), dict) for r in entries),
                  "agent_totals_match": True, **totals}
        if any(type(terminal.get("api_ledger", {}).get(k)) is not type(v) or
               terminal["api_ledger"][k] != v for k, v in counts.items()):
            raise RuntimeError("Terminal ledger counters differ from actual API attempts")
        from strict_ablations.artifacts import verify_trace_checkpoint
        artifacts = verify_trace_checkpoint(output, arm=arm, query=query,
            config=manifest["config"], contract=contract, terminal=terminal)
        if terminal.get("execution_artifacts") != self.artifact_hashes(arm, qid):
            raise RuntimeError("Immutable terminal artifact bytes changed")
        return {"request_contract_sha256": file_sha(path), "api_ledger_sha256": file_sha(ledger_path),
                "recorded_requests": len(entries), **totals, **artifacts}

    def artifact_hashes(self, arm, qid):
        output = self.runs / arm
        paths = {"request_contract": output / "request_contracts" / f"qid_{qid}.json",
                 "api_ledger": output / "api_ledgers" / f"qid_{qid}.jsonl",
                 "trace": output / "_traces" / f"qid_{qid}" / "single.json",
                 "checkpoint": output / "_checkpoints" / f"qid_{qid}.json"}
        return {key: file_sha(path) for key, path in paths.items()}

    def pilot_gate(self, manifest):
        checks = []
        for query in shared.validate_queries()[:10]:
            for arm in ARMS:
                row = self.validate_terminal(load(self.terminal_path(arm, query["query_id"])), manifest, arm, query)
                checks.append({"arm": arm, "query_id": query["query_id"], "row_sha256": row["row_sha256"],
                               **self.verify_execution_artifacts(manifest, arm, query, row)})
        return {"passed": True, "manifest_sha256": manifest["manifest_sha256"],
                "checked_rows": checks, "correctness_used": False, "new_pilot_runs": 30}

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        path = self.terminal_path(arm, query["query_id"])
        if path.exists():
            row = self.validate_terminal(load(path), manifest, arm, query)
            self.verify_execution_artifacts(manifest, arm, query, row)
            return {"arm": arm, "query_id": query["query_id"], "status": "reused_immutable_terminal"}
        output = self.runs / arm
        qid = query["query_id"]
        marker = output / "_infrastructure" / f"qid_{qid}.json"
        if marker.exists():
            if load(marker).get("manifest_sha256") != manifest["manifest_sha256"]:
                raise RuntimeError("Infrastructure marker belongs to another manifest")
            if not resume_infrastructure:
                raise RuntimeError("Infrastructure marker requires explicit infrastructure resume")
        partial = any((output / directory / f"qid_{qid}{extension}").exists() for directory, extension in
                      (("_checkpoints", ".json"), ("api_ledgers", ".jsonl"), ("request_contracts", ".json")))
        if partial and not resume_infrastructure:
            raise RuntimeError("Partial execution requires explicit same-checkpoint infrastructure resume")
        if partial and not (output / "_checkpoints" / f"qid_{qid}.json").exists():
            raise RuntimeError("A partial request has no checkpoint; fresh-budget retry is forbidden")
        from strict_ablations.runtime import run_query
        try:
            # No gold answers, evidence IDs or historical verdicts reach the solver.
            row = run_query(arm=arm, query={"query_id": qid, "query": query["query"]},
                config=manifest["config"], output_dir=output, expected_contract=manifest["contracts"][arm],
                report_dir=self.reports)
            row = {**row, "version": SUITE_VERSION, "arm": arm, "query_id": qid,
                "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
                "contract_sha256": manifest["contracts"][arm]["contract_sha256"],
                "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest()}
            row["final_sha256"] = hashlib.sha256(row["final_text"].encode()).hexdigest()
            row["execution_artifacts"] = self.artifact_hashes(arm, qid)
            row["row_sha256"] = digest(row)
            self.validate_terminal(row, manifest, arm, query)
            self.verify_execution_artifacts(manifest, arm, query, row)
            atomic_json(path, row, immutable=True)
            return {"arm": arm, "query_id": qid, "status": "terminal_saved", "terminated_by": row["terminated_by"]}
        except Exception as exc:
            atomic_json(output / "_infrastructure" / f"qid_{qid}.json", {
                "manifest_sha256": manifest["manifest_sha256"], "error_type": type(exc).__name__,
                "checkpoint_required_for_partial_query": True, "unix_time": time.time()})
            raise

    def run(self, stage, *, resume_infrastructure=False):
        if stage not in {"pilot", "full"}:
            raise ValueError("Unknown stage")
        self.reports.mkdir(parents=True, exist_ok=True)
        with (self.reports / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.verify_manifest()
            ready = load(self.reports / "preflight_live.json")
            if ready.get("passed") is not True or ready.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise RuntimeError("New ablation manifest must pass its own live preflight")
            os.environ.update(manifest["config"]["runtime_environment"])
            current_service = self.verify_service(manifest)
            if (ready.get("service", {}).get("epoch_sha256") != current_service["epoch_sha256"]):
                raise RuntimeError("The current process epoch needs its own live preflight")
            if stage == "full":
                atomic_json(self.reports / "pilot_gate.json", self.pilot_gate(manifest))
            queries = shared.validate_queries()[:10] if stage == "pilot" else shared.validate_queries()
            jobs = iter((arm, query) for query in queries for arm in ARMS)
            completed = []
            with ThreadPoolExecutor(max_workers=manifest["config"]["concurrency"]) as pool:
                active = {}
                for _ in range(manifest["config"]["concurrency"]):
                    job = next(jobs, None)
                    if job:
                        arm, query = job
                        active[pool.submit(self.run_one, manifest, arm, query, resume_infrastructure=resume_infrastructure)] = job
                while active:
                    future = next(as_completed(active)); active.pop(future)
                    completed.append(future.result())
                    job = next(jobs, None)
                    if job:
                        arm, query = job
                        active[pool.submit(self.run_one, manifest, arm, query, resume_infrastructure=resume_infrastructure)] = job
            self.verify_manifest()
            if stage == "pilot":
                atomic_json(self.reports / "pilot_gate.json", self.pilot_gate(manifest))
            receipt = {"passed": True, "stage": stage, "manifest_sha256": manifest["manifest_sha256"],
                       "finished_unix": time.time(), "natural_error_reruns": 0,
                       "terminal_rows_checked": len(completed), "control_model_requests": 0}
            atomic_json(self.reports / f"{stage}_completion.json", receipt)
            return receipt

    def status(self):
        if not (self.reports / "run_manifest.json").exists():
            return {"version": SUITE_VERSION, "prepared": False, "live_ready": False, "new_question_runs": 300}
        manifest = self.verify_manifest(resources=False)
        groups = {}
        for arm in ARMS:
            complete, interrupted, pending = [], [], []
            for query in shared.validate_queries():
                qid = query["query_id"]
                if self.terminal_path(arm, qid).exists():
                    row = self.validate_terminal(load(self.terminal_path(arm, qid)), manifest, arm, query)
                    self.verify_execution_artifacts(manifest, arm, query, row); complete.append(qid)
                elif any((self.runs / arm / d / f"qid_{qid}{e}").exists() for d, e in
                         (("_checkpoints", ".json"), ("request_contracts", ".json"), ("api_ledgers", ".jsonl"), ("_infrastructure", ".json"))):
                    interrupted.append(qid)
                else:
                    pending.append(qid)
            groups[arm] = {"terminal": complete, "interrupted": interrupted, "pending": pending}
        return {"version": SUITE_VERSION, "prepared": True, "arms": groups,
                "reused_control_questions": 100, "completed_new_runs": sum(len(v["terminal"]) for v in groups.values())}

    def live_preflight(self, *, resume_infrastructure=False, service_pid=None):
        self.reports.mkdir(parents=True, exist_ok=True)
        with (self.reports / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._live_preflight(resume_infrastructure=resume_infrastructure, service_pid=service_pid)

    def _live_preflight(self, *, resume_infrastructure=False, service_pid=None):
        manifest = self.verify_manifest()
        config = manifest["config"]
        os.environ.update(config["runtime_environment"])
        if service_pid is not None:
            if not resume_infrastructure:
                raise RuntimeError("A new process epoch requires explicit infrastructure recovery")
            self.save_epoch(deployment.prepare_epoch(config, manifest["resource_identity"], pid=service_pid))
        if (self.reports / "infrastructure_pause.json").exists():
            if not resume_infrastructure:
                raise RuntimeError("Explicit infrastructure recovery is required")
            pause = self.reports / "infrastructure_pause.json"
            saved = self.reports / "recovered_pauses" / (str(time.time_ns()) + ".json")
            saved.parent.mkdir(parents=True, exist_ok=True); os.replace(pause, saved)
        report = {"passed": False, "manifest_sha256": manifest["manifest_sha256"],
                  "started_unix": time.time(), "cohort_questions_used": 0}
        try:
            report["service"] = self.verify_service(manifest)
            report["tools_and_native_calls"] = self._probe_surfaces(manifest)
            self.verify_manifest()
            report["passed"] = True
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            raise
        finally:
            report["finished_unix"] = time.time()
            atomic_json(self.reports / "preflight_live.json", report)
        return report

    def _probe_surfaces(self, manifest):
        from strict_ablations import runtime
        from rise.api_retry import chat_with_retry
        from rise.decompose import make_client
        from rise.local_vllm import extract_reasoning
        config = manifest["config"]
        searcher, docids, mapping = runtime.resources(config)
        probe_root = self.reports / "live_probes"
        probe_root.mkdir(parents=True, exist_ok=True)
        health = runtime.health_state(config, self.reports)
        report, pools = {}, []
        with tempfile.TemporaryDirectory(prefix="surface-", dir=probe_root) as directory:
            for arm in ARMS:
                work = Path(directory) / arm; work.mkdir()
                service = runtime.make_service(config, work, Path(directory) / (arm + "-rerank"), mapping, health=health)
                surface = runtime.build_tools(arm, config=config, searcher=searcher, doc_ids=docids,
                    mapping=mapping, corpus=Path(config["corpus"]), working_dir=work,
                    rerank_service=service, health=health)
                output = surface["search"]({"queries": ["strict ablation compatibility probe"]})
                sources = sorted(p.relative_to(work).as_posix() for p in work.rglob("*.txt"))
                if not sources:
                    raise RuntimeError("Search probe did not import any original documents")
                pools.append(sources)
                if "bash" in surface:
                    probes = [surface["bash"]({"command": command}) for command in (
                        "printf RISE_STRICT_SANDBOX_READY",
                        "printf RISE_TMP_PROBE >/tmp/rise-strict-probe",
                        "test ! -e /tmp/rise-strict-probe && printf FRESH_TMP",
                        "rg --version | head -n 1")]
                    if probes[0] != "RISE_STRICT_SANDBOX_READY" or "FRESH_TMP" not in probes[2] or "ripgrep" not in probes[3]:
                        raise RuntimeError("Original bash execution, rg or temporary-file isolation failed")
                if arm == "search_paging":
                    page = json.loads(output.split("\nSearch pages: ", 1)[1])[0]
                    before = copy.deepcopy(surface.summary)
                    surface["search"]({"search_id": page["search_id"], "offset": 10})
                    if surface.summary["retrieval_calls"] != before["retrieval_calls"] or service.summary["scoring_calls"] != 0:
                        raise RuntimeError("Snippet paging performed retrieval/scoring")
                elif arm == "auto_rerank":
                    result = json.loads(output)
                    if result.get("status") != "ok" or "rerank" in surface or surface.summary["automatic_rerank_calls"] != 1:
                        raise RuntimeError("Automatic search surface failed")
                elif "bash" in surface:
                    raise RuntimeError("Removed bash capability was registered")
                if "rerank" in surface:
                    result = json.loads(surface["rerank"]({"query": "compatibility probe", "file_paths": [sources[0]],
                        "bm25_candidates": 2, "top_k": 2, "limit": 2}))
                    if result.get("status") != "ok" or not result.get("scored_passages"):
                        raise RuntimeError("Real neural rerank probe did not execute")
                args = {"file_path": sources[0], "offset": 0, "limit": 1}
                contract = manifest["contracts"][arm]
                query = {"query_id": 0, "question": "Compatibility probe only: call read with " + json.dumps(args) + "; then finish with RISE_ABLATIONS_READY."}
                client = runtime.RequestRecorder(make_client(MODEL), arm=arm, query=query, config=config,
                    contract=contract, output_dir=probe_root / arm, health=health)
                history = [{"role": "system", "content": contract["system_prompt"]},
                    {"role": "user", "content": contract["user_prompt_template"].format(question=query["question"])}]
                response = chat_with_retry(client, model=MODEL, messages=history, tools=contract["tool_specs"],
                    tool_choice={"type": "function", "function": {"name": "read"}}, max_tokens=2048, max_retries=1)
                message = response.choices[0].message
                calls = message.tool_calls or []
                if len(calls) != 1 or calls[0].function.name != "read":
                    raise RuntimeError("Native read call was incomplete")
                arguments = json.loads(calls[0].function.arguments)
                if arguments.get("file_path") != sources[0]:
                    raise RuntimeError("Native read probe changed its source")
                value = surface["read"](arguments)
                if value.startswith("Error:"):
                    raise RuntimeError("Native read probe failed")
                assistant = {"role": "assistant", "content": message.content or "", "tool_calls": [c.model_dump() for c in calls]}
                reasoning = extract_reasoning(message)
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                history.extend([assistant, {"role": "tool", "tool_call_id": calls[0].id, "content": value}])
                final = chat_with_retry(client, model=MODEL, messages=history, tools=contract["tool_specs"],
                    tool_choice="none", max_tokens=2048, max_retries=1)
                if final.choices[0].finish_reason != "stop" or not final.choices[0].message.content:
                    raise RuntimeError("Native follow-up response was incomplete")
                report[arm] = {"passed": True, "tool_names": list(surface),
                    "usage": surface.summary, "reranker_summary": service.summary, "model_calls": 2}
            if any(pool != pools[0] for pool in pools[1:]):
                raise RuntimeError("Ablation search changed the imported document pool")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="New suite artifact root; old outputs are never rebound")
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template"); template.add_argument("--control-root", default=str(ROOT))
    bundle = sub.add_parser("export-control-bundle")
    bundle.add_argument("--control-root", default=str(ROOT)); bundle.add_argument("--output", required=True)
    offline = sub.add_parser("offline-preflight"); offline.add_argument("--config"); offline.add_argument("--control-root")
    prep = sub.add_parser("prepare"); prep.add_argument("--config", required=True)
    prep.add_argument("--service-pid", type=int, required=True)
    live = sub.add_parser("live-preflight"); live.add_argument("--resume-infrastructure", action="store_true")
    live.add_argument("--service-pid", type=int)
    sub.add_parser("status")
    run = sub.add_parser("run"); run.add_argument("--stage", choices=("pilot", "full"), required=True)
    run.add_argument("--resume-infrastructure", action="store_true")
    all_runs = sub.add_parser("run-all"); all_runs.add_argument("--resume-infrastructure", action="store_true")
    args = parser.parse_args(argv); suite = Suite(args.root)
    if args.command == "template": result = suite.template(args.control_root)
    elif args.command == "export-control-bundle": result = export_control_bundle(args.control_root, args.output)
    elif args.command == "offline-preflight": result = suite.offline_preflight(load(args.config) if args.config else None, control_root=args.control_root)
    elif args.command == "prepare": result = suite.prepare(load(args.config), service_pid=args.service_pid)
    elif args.command == "live-preflight": result = suite.live_preflight(resume_infrastructure=args.resume_infrastructure, service_pid=args.service_pid)
    elif args.command == "status": result = suite.status()
    elif args.command == "run-all": result = {"pilot": suite.run("pilot", resume_infrastructure=args.resume_infrastructure),
        "full": suite.run("full", resume_infrastructure=args.resume_infrastructure), "new_question_runs": 300, "control_model_requests": 0}
    else: result = suite.run(args.stage, resume_infrastructure=args.resume_infrastructure)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
