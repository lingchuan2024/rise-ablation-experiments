"""Metadata-only artifact verification with actual Agent/ToolSurface schemas."""
from __future__ import annotations

import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/tests"))

from strict_ablations import artifacts, runtime
from strict_ablations.prompt import ARMS, build_contract
from rise.agent import AgentRun, TurnRecord, _dump_trace
from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.rerank_v13 import RerankService
from rise.trajectory import compute_run_config_hash
from test_indexed_accumulated_space import build_raw_index, CheapBackend
from test_reranker_tool import structured


def fixture_config():
    return {"model": runtime.MODEL, "concurrency": 4, "max_turns": 1,
        "per_call_max_tokens": 32768, "wall_clock_seconds": 3600,
        "api_max_retries": 1, "bm25_k": 1000, "bm25_top_n_preview": 10,
        "bm25_snippet_chars": 100, "read_default_limit": 2000,
        "runtime_environment": {}}


@lru_cache(maxsize=3)
def _empty_surface_state(arm):
    # Generate a genuine indexed manager/service/wrapper state once per arm.
    # No corpus text is imported and no scoring or shell process runs.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus"
        index = build_raw_index(root / "index", corpus, {"docs/fixture.txt": structured("fixture")})
        manager = IndexedAccumulatedSpaceManager(working_dir=root / "work", views_dir=root / "views",
            backend=CheapBackend(), candidates=1000, top_k=100, passage_index=index,
            runtime_to_corpus={"docs/fixture.txt": "docs/fixture.txt"})
        service = RerankService(manager)
        surface = runtime.build_tools(arm, config=fixture_config(), searcher=None,
            doc_ids=["0"], mapping={"0": "docs/fixture.txt"}, corpus=corpus,
            working_dir=root / "work", rerank_service=service)
        return surface.export_state(), surface.summary, service.summary


def write_fixture_artifacts(output, *, arm, query, config, terminal, blank_max_reason=False):
    """Write real dataclass/trace shapes; update terminal numeric/state bindings."""
    output = Path(output)
    contract = build_contract(arm)
    state, summary, physical = copy.deepcopy(_empty_surface_state(arm))
    terminal["ablation_summary"], terminal["reranker_summary"] = summary, physical
    run_hash = compute_run_config_hash(runtime.run_config_for(arm, config).to_hashable_dict())
    terminal["run_config_hash"] = run_hash
    terminal["common_config_sha256"] = runtime.object_sha(config)
    user = contract["user_prompt_template"].format(question=query["query"])
    run = AgentRun(model=runtime.MODEL, system_prompt=contract["system_prompt"], user_prompt=user,
        final_answer=terminal["final_text"], terminated_reason=terminal["terminated_by"],
        elapsed_seconds=terminal["elapsed_seconds"],
        total_prompt_tokens=terminal["raw_non_cached_input_tokens"],
        total_cached_prompt_tokens=terminal["cached_input_tokens"],
        total_completion_tokens=terminal["output_tokens"],
        total_reasoning_tokens=terminal["reasoning_tokens"],
        tool_call_count=terminal["n_tool_calls"], tool_call_breakdown=terminal["tool_call_breakdown"])
    run.turns = [TurnRecord(turn=i, content=terminal["final_text"] if i == 0 else "",
        finish_reason="stop", prompt_tokens=run.total_prompt_tokens if i == 0 else 0,
        cached_prompt_tokens=run.total_cached_prompt_tokens if i == 0 else 0,
        completion_tokens=run.total_completion_tokens if i == 0 else 0,
        reasoning_tokens=run.total_reasoning_tokens if i == 0 else 0)
        for i in range(terminal["n_turns"])]
    qid = str(query["query_id"])
    trace_path = output / "_traces" / f"qid_{qid}" / "single.json"
    _dump_trace(run, trace_path)
    saved = asdict(run)
    # Reproduce the unchanged Agent's save-before-final-elapsed ordering.
    saved["elapsed_seconds"] = 0.0
    if blank_max_reason:
        saved["terminated_reason"] = ""
    checkpoint = {"version": 1, "identity": {
        "model": runtime.MODEL, "query_id": qid, "ablation_arm": arm,
        "ablation_runtime_version": runtime.RUNTIME_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "common_config_sha256": runtime.object_sha(config),
        "question_sha256": runtime.text_sha(query["query"]), "run_config_hash": run_hash,
        "system_prompt_sha256": contract["system_prompt_sha256"],
        "tool_specs_sha256": contract["tool_specs_sha256"]},
        "elapsed_seconds": max(0.0, terminal["elapsed_seconds"] - 0.001), "run": saved,
        "messages": [{"role": "system", "content": contract["system_prompt"]},
                     {"role": "user", "content": user}],
        "workspace_docs": [], "tool_state": state}
    cp_path = output / "_checkpoints" / f"qid_{qid}.json"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text(json.dumps(checkpoint))
    terminal["execution_artifacts"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "trace": trace_path, "checkpoint": cp_path,
            "request_contract": output / "request_contracts" / f"qid_{qid}.json",
            "api_ledger": output / "api_ledgers" / f"qid_{qid}.jsonl"}.items() if path.is_file()}
    return trace_path, cp_path


class ArtifactMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.arm = ARMS[0]
        self.config = fixture_config()
        self.query = {"query_id": 7, "query": "Offline metadata question"}
        self.terminal = {"final_text": "A public answer", "terminated_by": "done",
            "n_turns": 1, "n_tool_calls": 0, "tool_call_breakdown": {},
            "input_tokens": 50, "raw_non_cached_input_tokens": 30, "cached_input_tokens": 20,
            "output_tokens": 10, "reasoning_tokens": 4, "elapsed_seconds": 3.0}
        self.paths = write_fixture_artifacts(self.output, arm=self.arm, query=self.query,
                                            config=self.config, terminal=self.terminal)

    def verify(self):
        return artifacts.verify_trace_checkpoint(self.output, arm=self.arm, query=self.query,
            config=self.config, contract=build_contract(self.arm), terminal=self.terminal)

    def mutate(self, index, transform, *, resign=True):
        path = self.paths[index]
        value = json.loads(path.read_text()); transform(value)
        path.write_text(json.dumps(value))
        if resign:
            self.terminal["execution_artifacts"]["trace" if index == 0 else "checkpoint"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_real_schemas_return_hashes_only_for_all_three_surfaces(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.arm = arm
                write_fixture_artifacts(self.output, arm=arm, query=self.query,
                                        config=self.config, terminal=self.terminal)
                result = self.verify()
                self.assertEqual(set(result), {"trace_sha256", "checkpoint_sha256"})
                self.assertTrue(all(len(value) == 64 for value in result.values()))

    def test_legal_max_turn_checkpoint_can_have_empty_saved_reason(self):
        self.terminal.update(final_text="", terminated_by="max_turns")
        write_fixture_artifacts(self.output, arm=self.arm, query=self.query,
            config=self.config, terminal=self.terminal, blank_max_reason=True)
        self.verify()
        self.config["max_turns"] += 1
        with self.assertRaises(ValueError):
            self.verify()

    def test_checkpoint_identity_and_budget_cannot_be_reset_or_moved(self):
        original = self.paths[1].read_bytes()
        mutations = (lambda v: v["identity"].update(ablation_arm="auto_rerank"),
            lambda v: v["identity"].update(run_config_hash="wrong"),
            lambda v: v["identity"].update(common_config_sha256="0" * 64),
            lambda v: v.update(elapsed_seconds=0), lambda v: v.update(elapsed_seconds=-1),
            lambda v: v.update(elapsed_seconds=4), lambda v: v.update(elapsed_seconds=True))
        for change in mutations:
            self.paths[1].write_bytes(original)
            self.mutate(1, change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify()

    def test_private_file_tamper_without_new_terminal_anchor_is_rejected(self):
        self.mutate(1, lambda v: v.update(elapsed_seconds=2), resign=False)
        with self.assertRaisesRegex(ValueError, "SHA anchors"):
            self.verify()

    def test_changed_final_model_prompt_or_initial_messages_are_rejected(self):
        original = self.paths[1].read_bytes()
        for change in (lambda v: v["run"].update(final_answer="another answer"),
            lambda v: v["run"].update(model="another model"),
            lambda v: v["run"].update(system_prompt="another prompt"),
            lambda v: v["messages"][1].update(content="another question")):
            self.paths[1].write_bytes(original); self.mutate(1, change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify()

    def test_trace_total_turn_tokens_or_elapsed_tamper_is_rejected(self):
        original = self.paths[0].read_bytes()
        for change in (lambda v: v["totals"].update(completion_tokens=11),
            lambda v: v["totals"].update(n_turns=2),
            lambda v: v["turns"][0]["tokens"].update(prompt=29),
            lambda v: v.update(elapsed_seconds=0)):
            self.paths[0].write_bytes(original); self.mutate(0, change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify()

    def test_composite_service_search_usage_and_pool_bindings_are_checked(self):
        original = self.paths[1].read_bytes()
        changes = (lambda s: s.update(version="old-version"),
            lambda s: s.update(arm="no_bash"), lambda s: s["service"].update(version="old-version"),
            lambda s: s["search"].update(variant="auto_rerank"),
            lambda s: s["usage"].update(retrieval_calls=1),
            lambda s: s["service"]["usage"].update(rerank_calls=1),
            lambda s: s["service"]["manager"]["usage"].update(input_tokens=1),
            lambda s: s["service"]["manager"]["passage_index"].update(identity={}),
            lambda s: s["service"]["manager"].update(backend_identity={}),
            lambda s: s["service"].update(cache_version=1))
        for change in changes:
            self.paths[1].write_bytes(original)
            self.mutate(1, lambda v: change(v["tool_state"]))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify()

    def test_actual_agent_outputs_and_natural_unknown_tool_or_max_turns_pass(self):
        # Exercise the checker against the real SDK/Agent/search/read/service
        # integration fixture, including a removed-capability hallucination.
        from strict_ablations import test_runtime as integration
        fixture = integration.StrictAblationRuntimeTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for arm in ARMS:
            with self.subTest(arm=arm, mode="ordinary"):
                output = fixture.root / ("checked-" + arm)
                terminal = fixture.run_query(arm, [integration.reply("search", {"queries": ["aurora"]}),
                    integration.reply("read", {"file_path": fixture.mapping["11"]}),
                    integration.reply()], output=output)
                self._verify_actual(output, arm, fixture, terminal)
        for arm, removed in (("no_bash", "bash"), ("auto_rerank", "rerank")):
            with self.subTest(arm=arm, mode="unknown_tool_max_turns"):
                config = copy.deepcopy(fixture.config); config["max_turns"] = 1
                output = fixture.root / ("unknown-" + arm)
                terminal = fixture.run_query(arm, [integration.reply(removed, {})],
                                             output=output, config=config)
                self.assertEqual(terminal["terminated_by"], "max_turns")
                cp = json.loads((output / "_checkpoints/qid_1.json").read_text())
                self.assertEqual(cp["run"]["terminated_reason"], "")
                self.assertEqual(cp["run"]["turns"][0]["tool_calls"][0]["error"], "unknown_tool")
                self._verify_actual(output, arm, fixture, terminal, config=config)
                # None was a legal pre-normalization empty final in historical
                # Agent artifacts; comparison must preserve that natural case.
                trace_path = output / "_traces/qid_1/single.json"
                trace = json.loads(trace_path.read_text()); trace["final_text"] = None
                cp["run"]["final_answer"] = None
                trace_path.write_text(json.dumps(trace))
                (output / "_checkpoints/qid_1.json").write_text(json.dumps(cp))
                self._verify_actual(output, arm, fixture, terminal, config=config)

    def _verify_actual(self, output, arm, fixture, terminal, *, config=None):
        terminal["execution_artifacts"] = {"trace": hashlib.sha256(
            (output / "_traces/qid_1/single.json").read_bytes()).hexdigest(),
            "checkpoint": hashlib.sha256((output / "_checkpoints/qid_1.json").read_bytes()).hexdigest()}
        result = artifacts.verify_trace_checkpoint(output, arm=arm, query=fixture.query,
            config=config or fixture.config, contract=build_contract(arm), terminal=terminal)
        self.assertEqual(set(result), {"trace_sha256", "checkpoint_sha256"})


if __name__ == "__main__":
    unittest.main()
