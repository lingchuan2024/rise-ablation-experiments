"""Opt-in, private per-turn checkpoints for chat-completions RISE runs."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any


class TurnCheckpoint:
    def __init__(
        self,
        path: Path,
        *,
        identity: dict[str, str],
        working_dir: Path,
        corpus_root: Path,
        relpath_to_docid: dict[str, str],
        tool_state_provider=None,
        tool_state_restorer=None,
    ) -> None:
        if working_dir.is_symlink():
            raise ValueError(f"Checkpoint workspace must not be a symlink: {working_dir}")
        self.path = path
        self.identity = identity
        self.working_dir = working_dir.resolve()
        self.corpus_root = corpus_root.resolve()
        self.relpath_to_docid = relpath_to_docid
        self.tool_state_provider = tool_state_provider
        self.tool_state_restorer = tool_state_restorer

    def _source(self, relpath: str, docid: str) -> Path:
        path = PurePosixPath(relpath)
        if (not relpath or path.is_absolute() or ".." in path.parts
                or path.as_posix() != relpath or path.suffix != ".txt"
                or self.relpath_to_docid.get(relpath) != docid):
            raise ValueError(f"Unsafe or unmapped checkpoint document: {relpath!r}")
        source = (self.corpus_root / relpath).resolve()
        if not source.is_relative_to(self.corpus_root) or not source.is_file():
            raise ValueError(f"Checkpoint source document is missing or unsafe: {relpath!r}")
        return source

    def _workspace_docs(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        for path in sorted(self.working_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Checkpoint workspace contains a symlink: {path}")
            if not path.is_file():
                continue
            relpath = path.relative_to(self.working_dir).as_posix()
            docid = self.relpath_to_docid.get(relpath)
            if docid is None:
                raise ValueError(f"Checkpoint workspace contains an unmapped file: {relpath!r}")
            source = self._source(relpath, docid)
            if not os.path.samefile(path, source):
                raise ValueError(f"Checkpoint workspace file is not a corpus hardlink: {relpath!r}")
            docs.append({"relpath": relpath, "docid": docid})
        return docs

    def save(self, run: Any, messages: list[dict[str, Any]], elapsed_seconds: float) -> None:
        payload = {
            "version": 1,
            "identity": self.identity,
            "elapsed_seconds": max(0.0, elapsed_seconds),
            "run": asdict(run),
            "messages": messages,
            "workspace_docs": self._workspace_docs(),
        }
        if self.tool_state_provider is not None:
            payload["tool_state"] = self.tool_state_provider()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                name = stream.name
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    def load(self) -> dict[str, Any] | None:
        if self.path.is_symlink():
            raise ValueError(f"Checkpoint file must not be a symlink: {self.path}")
        if not self.path.is_file():
            return None
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("identity") != self.identity:
            raise ValueError(f"Checkpoint identity changed: {self.path}")
        if not isinstance(state.get("run"), dict) or not isinstance(state.get("messages"), list):
            raise ValueError(f"Checkpoint state is incomplete: {self.path}")
        docs = state.get("workspace_docs")
        if not isinstance(docs, list):
            raise ValueError(f"Checkpoint workspace list is invalid: {self.path}")
        if self.tool_state_restorer is not None and not isinstance(state.get("tool_state"), dict):
            raise ValueError(f"Checkpoint reranker state is missing: {self.path}")
        return state

    def restore_workspace(self, state: dict[str, Any]) -> None:
        docs = state["workspace_docs"]
        validated: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for doc in docs:
            if not isinstance(doc, dict):
                raise ValueError(f"Checkpoint workspace entry is invalid: {self.path}")
            relpath, docid = doc.get("relpath"), doc.get("docid")
            if not isinstance(relpath, str) or not isinstance(docid, str) or relpath in seen:
                raise ValueError(f"Checkpoint workspace entry is invalid: {self.path}")
            seen.add(relpath)
            validated.append((relpath, self._source(relpath, docid)))
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self.working_dir.mkdir(parents=True)
        for relpath, source in validated:
            target = self.working_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        if self.tool_state_restorer is not None:
            self.tool_state_restorer(state["tool_state"])
