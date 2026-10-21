"""A length-limited chat response is never recorded as a final answer."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.agent import Agent
from rise.turn_checkpoint import TurnCheckpoint


def _reply(content: str, finish_reason: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=[], reasoning_content="")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=8)
    return SimpleNamespace(choices=[choice], usage=usage)


class OutputTruncationTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> TurnCheckpoint:
        corpus = root / "corpus"
        corpus.mkdir()
        workspace = root / "workspace"
        workspace.mkdir()
        return TurnCheckpoint(
            root / "checkpoint.json",
            identity={"model": "nvidia/test", "query_id": "1", "run_config_hash": "abc"},
            working_dir=workspace, corpus_root=corpus, relpath_to_docid={},
        )

    def _agent(self, *, max_turns: int, coerce: bool) -> Agent:
        return Agent(
            client=object(), model="nvidia/test", tools={}, tool_specs=[],
            max_turns=max_turns, coerce_final_on_max_turns=coerce,
        )

    def test_partial_visible_response_is_traced_but_not_scored_as_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            trace_path = root / "trace.json"
            with patch("rise.agent.chat_with_retry", return_value=_reply("Partial answer...", "length")) as api:
                run = self._agent(max_turns=3, coerce=True).run(
                    system_prompt="system", user_prompt="question",
                    trace_path=trace_path, turn_checkpoint=checkpoint,
                )
            self.assertEqual(api.call_count, 1)
            self.assertEqual(run.terminated_reason, "output_truncated")
            self.assertEqual(run.final_answer, "")
            self.assertEqual(run.turns[-1].content, "Partial answer...")
            self.assertEqual(run.turns[-1].finish_reason, "length")

            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["terminated_by"], "output_truncated")
            self.assertEqual(trace["final_text"], "")
            self.assertEqual(trace["turns"][-1]["content"], "Partial answer...")
            self.assertEqual(
                json.loads(checkpoint.path.read_text(encoding="utf-8"))["run"]["terminated_reason"],
                "output_truncated",
            )
            with patch("rise.agent.chat_with_retry", side_effect=AssertionError("unexpected API call")):
                resumed = self._agent(max_turns=3, coerce=True).run(
                    system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
                )
            self.assertEqual(resumed.terminated_reason, "output_truncated")
            self.assertEqual(len(resumed.turns), 1)

    def test_coerced_partial_response_is_also_terminal_and_checkpointed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            with patch("rise.agent.chat_with_retry", return_value=_reply("Still thinking...", "length")) as api:
                run = self._agent(max_turns=0, coerce=True).run(
                    system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
                )
            self.assertEqual(api.call_count, 1)
            self.assertEqual(run.terminated_reason, "output_truncated")
            self.assertEqual(run.final_answer, "")
            self.assertEqual(run.turns[-1].content, "Still thinking...")
            self.assertEqual(run.turns[-1].finish_reason, "coerced/length")
            with patch("rise.agent.chat_with_retry", side_effect=AssertionError("unexpected API call")):
                resumed = self._agent(max_turns=0, coerce=True).run(
                    system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
                )
            self.assertEqual(resumed.terminated_reason, "output_truncated")
            self.assertEqual(len(resumed.turns), 1)


if __name__ == "__main__":
    unittest.main()
