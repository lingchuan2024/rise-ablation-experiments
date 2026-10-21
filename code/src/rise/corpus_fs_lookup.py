"""Disk-on-demand `{docid: text}` lookup for the BCP+ corpus.

Replaces the in-memory dict (formerly loaded from `data.parquet` at runner
startup) with a `Mapping` whose `.get(docid)` reads `bc_plus_docs / relpath`
from disk on demand. Byte-equivalence with the parquet dict is verified by
`scripts/test_corpus_storage_equivalence.py` (all 100,195 docids, 0 diffs).

Memory impact: BCP/Pi-Serini retrieval-agent runs go from ~2.4 GB RSS
(corpus dict was the dominant resident-set contributor at 100k scale) to
~500 MB. At 1M scale the dict would have been infeasible on a 16 GB
MacBook; this swap makes 1M dev_100 runs possible.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path


class DocidFsLookup(Mapping[str, str]):
    """Dict-like `{docid: text}` interface backed by `.txt` file reads.

    Same contract as `dict[str, str]` for the read paths the BCP and
    Pi-Serini agents use (`.get(docid)`, `.get(docid, default)`, `len()`).
    Thread-safe (no shared mutable state besides the immutable map; each
    `.get(docid)` is an independent `open + read`).
    """

    def __init__(
        self,
        bc_plus_docs_root: Path,
        docid_to_relpath: dict[str, str],
    ) -> None:
        self._root = Path(bc_plus_docs_root).resolve()
        self._docid_to_relpath = docid_to_relpath

    def __len__(self) -> int:
        return len(self._docid_to_relpath)

    def __iter__(self) -> Iterator[str]:
        return iter(self._docid_to_relpath)

    def __contains__(self, docid: object) -> bool:
        return docid in self._docid_to_relpath

    def __getitem__(self, docid: str) -> str:
        text = self.get(docid)
        if text is None:
            raise KeyError(docid)
        return text

    def get(self, docid: str, default=None):  # type: ignore[override]
        relpath = self._docid_to_relpath.get(docid)
        if relpath is None:
            return default
        try:
            return (self._root / relpath).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return default


def load_docid_to_relpath_map(map_json_path: Path) -> dict[str, str]:
    """Load `outputs/corpus_filename_docid_map.json` and invert the
    `{relpath: docid}` field into `{docid: relpath}` for lookup.
    """
    payload = json.loads(Path(map_json_path).read_text(encoding="utf-8"))
    relpath_to_docid: dict[str, str] = payload["relpath_to_docid"]
    return {docid: relpath for relpath, docid in relpath_to_docid.items()}
