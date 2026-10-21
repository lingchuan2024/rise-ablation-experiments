#!/usr/bin/env python3
"""Read-only checks for the local inputs of the full 100-question RISE run.

This script never downloads data, builds an index, calls a model, or writes files.
It checks the original 100k corpus/index and the TOC-augmented document tree
used by ``run_rise.py --structured-docs``. Parquet metadata and docid ordering
are checked when the optional ``pyarrow`` dependency is available.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rise.query_ids import safe_query_id

EXPECTED_QUERIES = 100
EXPECTED_DOCUMENTS = 100_195
REQUIRED_QUERY_FIELDS = {"query_id", "query", "answer", "gold_doc_ids"}
REQUIRED_PARQUET_COLUMNS = {"docid", "text", "url"}


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def ok(self, message: str) -> None:
        self.lines.append(("OK", message))

    def warn(self, message: str) -> None:
        self.lines.append(("WARN", message))

    def error(self, message: str) -> None:
        self.lines.append(("ERROR", message))

    @property
    def ready(self) -> bool:
        return not any(level == "ERROR" for level, _ in self.lines)

    def display(self) -> None:
        for level, message in self.lines:
            print(f"[{level}] {message}")
        print("READY" if self.ready else "NOT READY: provide the missing local inputs above")


def _read_json(path: Path, report: Report, label: str) -> Any | None:
    if not path.is_file():
        report.error(f"{label} missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.error(f"{label} cannot be read as JSON: {path} ({exc})")
        return None


def check_queries(path: Path, report: Report, expected_count: int) -> list[dict[str, Any]] | None:
    if not path.is_file():
        report.error(f"100-question JSONL missing: {path}")
        return None
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.error(f"queries line {line_number} is invalid JSON: {exc}")
                    continue
                if not isinstance(row, dict):
                    report.error(f"queries line {line_number} is not a JSON object")
                    continue
                missing = REQUIRED_QUERY_FIELDS - row.keys()
                if missing:
                    report.error(f"queries line {line_number} missing fields: {', '.join(sorted(missing))}")
                    continue
                try:
                    id_key = safe_query_id(row["query_id"])
                except ValueError as exc:
                    report.error(f"queries line {line_number} has invalid query_id ({exc})")
                    continue
                if id_key in seen_ids:
                    report.error(f"queries line {line_number} repeats query_id {id_key!r}")
                seen_ids.add(id_key)
                if not isinstance(row["query"], str) or not row["query"].strip():
                    report.error(f"queries line {line_number} has an empty/non-string query")
                if not isinstance(row["answer"], str) or not row["answer"].strip():
                    report.error(f"queries line {line_number} has an empty/non-string answer")
                gold = row["gold_doc_ids"]
                if not isinstance(gold, list) or any(
                    not isinstance(docid, (str, int)) or isinstance(docid, bool) or not str(docid).strip()
                    for docid in gold
                ):
                    report.error(f"queries line {line_number} has invalid gold_doc_ids")
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        report.error(f"queries cannot be read: {path} ({exc})")
        return None
    if len(rows) != expected_count:
        report.error(f"query count is {len(rows)}, expected {expected_count}: {path}")
    else:
        report.ok(f"query JSONL: {len(rows)} rows with required fields ({path})")
    if len(seen_ids) != len(rows):
        report.error(f"query IDs are not unique: {len(seen_ids)} unique among {len(rows)} rows")
    return rows


def check_index(index_dir: Path, report: Report, expected_documents: int) -> list[str] | None:
    index = index_dir / "bm25_index"
    if not index.is_dir() or not any(index.iterdir()):
        report.error(f"BM25 index directory missing/empty: {index}")
    else:
        report.ok(f"BM25 index files present: {index}")
    raw = _read_json(index_dir / "doc_ids.json", report, "BM25 doc_ids.json")
    if raw is None:
        return None
    if not isinstance(raw, list) or any(not isinstance(x, str) or not x for x in raw):
        report.error("BM25 doc_ids.json must be a list of nonempty string docids")
        return None
    if len(raw) != len(set(raw)):
        report.error("BM25 doc_ids.json contains duplicate docids")
    if len(raw) != expected_documents:
        report.error(f"BM25 doc_ids.json has {len(raw)} ids; expected {expected_documents} for the paper's 100k corpus")
    else:
        report.ok(f"BM25 doc_ids.json: {len(raw):,} unique document IDs")
    meta_path = index_dir / "meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path, report, "BM25 meta.json")
        if isinstance(meta, dict):
            if meta.get("n_docs") != len(raw):
                report.error(f"BM25 meta.json n_docs={meta.get('n_docs')!r} differs from doc_ids.json count {len(raw)}")
            if meta.get("k1") != 1.5 or meta.get("b") != 0.75:
                report.error("BM25 meta.json parameters differ from paper defaults k1=1.5, b=0.75")
    return raw


def check_parquet(path: Path, report: Report, doc_ids: list[str] | None, expected_documents: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        report.error(f"original corpus parquet missing/empty: {path}")
        return
    try:
        import pyarrow.parquet as pq  # optional; never needed just to run this script
    except ImportError:
        report.warn("pyarrow is unavailable; parquet schema, row count and BM25 docid order were not checked. Install project dependencies to enable this check.")
        report.ok(f"original corpus parquet file present: {path}")
        return
    try:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = REQUIRED_PARQUET_COLUMNS - columns
        if missing:
            report.error(f"original parquet lacks columns: {', '.join(sorted(missing))}")
        else:
            report.ok("original parquet has docid/text/url columns")
        n_rows = parquet.metadata.num_rows
        if n_rows != expected_documents:
            report.error(f"original parquet has {n_rows} rows; expected {expected_documents}")
        else:
            report.ok(f"original parquet has {n_rows:,} rows")
        if doc_ids is not None and "docid" in columns and n_rows == len(doc_ids):
            at = 0
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(group, columns=["docid"]).column("docid").to_pylist()
                for value in ids:
                    if str(value) != doc_ids[at]:
                        report.error(f"parquet/BM25 docid order differs at row {at}: parquet={value!r}, index={doc_ids[at]!r}")
                        return
                    at += 1
            report.ok("original parquet docid order matches BM25 doc_ids.json")
    except Exception as exc:
        report.error(f"original parquet cannot be inspected: {path} ({type(exc).__name__}: {exc})")


def _safe_relpath(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value) and "\\" not in value and not path.is_absolute()
        and ".." not in path.parts and path.as_posix() == value
        and path.suffix == ".txt" and len(path.parts) >= 2
    )


def check_map(path: Path, report: Report, doc_ids: list[str] | None, expected_documents: int) -> dict[str, str] | None:
    blob = _read_json(path, report, "filename map")
    if blob is None:
        return None
    if not isinstance(blob, dict) or not isinstance(blob.get("relpath_to_docid"), dict):
        report.error("filename map must contain a relpath_to_docid object")
        return None
    mapping = blob["relpath_to_docid"]
    bad_paths = [key for key in mapping if not isinstance(key, str) or not _safe_relpath(key)]
    if bad_paths:
        report.error(f"filename map contains {len(bad_paths)} unsafe/non-.txt relative paths (first: {bad_paths[0]!r})")
    bad_ids = [value for value in mapping.values() if not isinstance(value, str) or not value]
    if bad_ids:
        report.error(f"filename map contains {len(bad_ids)} invalid docids")
        return None
    if len(mapping) != expected_documents:
        report.error(f"filename map has {len(mapping)} entries; expected {expected_documents}")
    else:
        report.ok(f"filename map has {len(mapping):,} relative paths")
    mapped_ids = set(mapping.values())
    if len(mapped_ids) != len(mapping):
        report.error("filename map has duplicate docids; mapping must be one-to-one")
    if doc_ids is not None and mapped_ids != set(doc_ids):
        missing = set(doc_ids) - mapped_ids
        extra = mapped_ids - set(doc_ids)
        report.error(f"filename map and BM25 docids differ: {len(missing)} unmapped index ids, {len(extra)} extra map ids")
    elif doc_ids is not None:
        report.ok("filename map docids match the BM25 index docids")
    return mapping


def check_gold_ids(queries: list[dict[str, Any]] | None, doc_ids: list[str] | None, report: Report) -> None:
    if queries is None or doc_ids is None:
        return
    gold = {str(docid) for row in queries if isinstance(row.get("gold_doc_ids"), list)
            for docid in row["gold_doc_ids"]}
    missing = gold - set(doc_ids)
    if missing:
        report.error(f"{len(missing)} gold docids from the 100 queries are absent from BM25 doc_ids.json (first: {sorted(missing)[0]!r})")
    else:
        report.ok(f"all {len(gold)} query gold docids appear in the BM25 index")


def check_document_trees(raw_root: Path, structured_root: Path, mapping: dict[str, str] | None,
                         report: Report, sample_size: int) -> None:
    roots = (("original document tree", raw_root), ("structured document tree", structured_root))
    for label, root in roots:
        if not root.is_dir():
            report.error(f"{label} missing: {root}")
    if mapping is None or any(not root.is_dir() for _, root in roots):
        return
    if not mapping:
        report.error("filename map is empty; cannot sample document paths")
        return
    paths = sorted(key for key in mapping if isinstance(key, str) and _safe_relpath(key))
    if not paths:
        report.error("filename map has no safe paths to sample")
        return
    sample = random.Random(260606880).sample(paths, min(sample_size, len(paths)))
    for label, root in roots:
        root_resolved = root.resolve()
        missing: list[str] = []
        escaped: list[str] = []
        for relpath in paths:
            file = root / relpath
            if not file.resolve().is_relative_to(root_resolved):
                escaped.append(relpath)
            elif not file.is_file():
                missing.append(relpath)
        missing_set = set(missing)
        escaped_set = set(escaped)
        malformed: list[str] = []
        for relpath in sample:
            file = root / relpath
            if relpath in missing_set or relpath in escaped_set:
                continue
            try:
                with file.open(encoding="utf-8") as stream:
                    head = stream.read(1024 * 1024)
            except (OSError, UnicodeError):
                malformed.append(relpath)
                continue
            if not head.strip():
                malformed.append(relpath)
            elif root == structured_root and (
                "## Table of Contents" not in head or "=== DOCUMENT BODY ===" not in head
            ):
                malformed.append(relpath)
        if missing:
            report.error(f"{label}: {len(missing)}/{len(paths)} mapped files missing (first: {missing[0]!r})")
        if escaped:
            report.error(f"{label}: {len(escaped)}/{len(paths)} mapped paths resolve outside root (first: {escaped[0]!r})")
        if malformed:
            report.error(f"{label}: {len(malformed)}/{len(sample)} sampled files empty, unreadable, or missing TOC/body markers (first: {malformed[0]!r})")
        if not missing and not malformed and not escaped:
            report.ok(f"{label}: all {len(paths)} mapped files present; {len(sample)} content samples readable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--raw-parquet", type=Path, default=ROOT / "corpus/browsecomp_plus.parquet")
    parser.add_argument("--raw-docs", type=Path, default=ROOT / "corpus/bcp_plus")
    parser.add_argument("--structured-docs", type=Path, default=ROOT / "corpus/bcp_plus_structured")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "runs/bm25_full")
    parser.add_argument("--skip-index", action="store_true", help="For the pre-index stage: check all supplied corpus inputs without requiring a built BM25 index")
    parser.add_argument("--filename-map", type=Path, default=ROOT / "runs/corpus_filename_docid_map.json")
    parser.add_argument("--sample-size", type=int, default=128, help="Deterministic mapped-path sample per document tree (default: 128)")
    parser.add_argument("--expected-queries", type=int, default=EXPECTED_QUERIES)
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    args = parser.parse_args(argv)
    if args.sample_size < 1 or args.expected_queries < 1 or args.expected_documents < 1:
        parser.error("sample-size and expected counts must be positive")
    report = Report()
    queries = check_queries(args.queries, report, args.expected_queries)
    if args.skip_index:
        report.warn("BM25 index check skipped; rerun preflight without --skip-index before agent execution")
        doc_ids = None
    else:
        doc_ids = check_index(args.index_dir, report, args.expected_documents)
    check_parquet(args.raw_parquet, report, doc_ids, args.expected_documents)
    mapping = check_map(args.filename_map, report, doc_ids, args.expected_documents)
    check_gold_ids(queries, doc_ids, report)
    check_document_trees(args.raw_docs, args.structured_docs, mapping, report, args.sample_size)
    report.display()
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
