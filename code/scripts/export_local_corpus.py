#!/usr/bin/env python3
"""Export *local* BrowseComp-Plus Arrow/Parquet files for the RISE runners.

No dataset is downloaded. The raw corpus can be an existing parquet or an
already-cached Hugging Face Arrow shard directory. Arrow input is converted
locally to the raw parquet required by BM25. An optional structured parquet is
written at the same relative paths, matched by docid rather than row order.
Both inputs need docid, text, and url columns. The map format is the one
consumed by scripts/run_rise.py: {"relpath_to_docid": {"domain/file.txt": "id"}}.

Prefer --existing-map when the official corpus exporter has supplied a map.
Without one, this script derives readable, deterministic paths from URL hosts
and final URL path segments. That naming is *not* the official BrowseComp-Plus
filename algorithm; paths may differ from the official export even when their
docids and text match. Use one map and its accompanying file trees consistently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "runs" / "corpus_filename_docid_map.json"
Row = Mapping[str, Any]
RowSource = Callable[[Sequence[str]], Iterable[Row]]


def iter_parquet_rows(path: Path, columns: Sequence[str]) -> Iterable[Row]:
    """Read only requested columns, one Arrow record batch at a time."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Parquet input; install this project's dependencies") from exc
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
    for batch in parquet.iter_batches(batch_size=1024, columns=list(columns)):
        yield from batch.to_pylist()


def _arrow_shards(directory: Path) -> list[Path]:
    """Require one complete Hugging Face split, in its published shard order."""
    shards = sorted(directory.glob("*.arrow"))
    if not shards:
        raise ValueError(f"{directory}: no local .arrow shards")
    match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.arrow", shards[0].name)
    if match is None:
        raise ValueError(f"{shards[0]}: expected a Hugging Face numbered Arrow shard")
    prefix, _, total_text = match.groups()
    total = int(total_text)
    if total <= 0 or len(shards) != total:
        raise ValueError(f"{directory}: expected {total} Arrow shards, found {len(shards)}")
    for number, shard in enumerate(shards):
        expected = f"{prefix}-{number:05d}-of-{total:05d}.arrow"
        if shard.name != expected:
            raise ValueError(f"{directory}: incomplete or mixed Arrow shards; expected {expected}")
    return shards


def convert_arrow_cache_to_parquet(directory: Path, destination: Path) -> int:
    """Stream a complete local HF Arrow cache into the raw BM25 parquet.

    Existing output is never overwritten. A source manifest allows an exact
    rerun to reuse its own output without repeating the multi-GB conversion.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Arrow input; install this project's dependencies") from exc
    shards = _arrow_shards(directory)
    provenance_path = destination.with_name(destination.name + ".arrow-source.json")
    if destination.is_symlink() or provenance_path.is_symlink():
        raise ValueError(f"refusing symlink as parquet output or provenance: {destination}")
    manifest = {
        "format": "rise-local-arrow-conversion-v1",
        "shards": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in shards
        ],
    }
    if destination.exists() or provenance_path.exists():
        if not destination.is_file() or not provenance_path.is_file():
            raise ValueError(f"incomplete existing Arrow conversion at {destination}; refusing to overwrite")
        saved = json.loads(provenance_path.read_text(encoding="utf-8"))
        output_stat = destination.stat()
        if saved.get("source") != manifest or saved.get("output") != {
            "size": output_stat.st_size, "mtime_ns": output_stat.st_mtime_ns,
        }:
            raise ValueError(f"Arrow sources changed for {destination}; refusing to reuse")
        parquet = pq.ParquetFile(destination)
        if set(("docid", "text", "url")) - set(parquet.schema_arrow.names):
            raise ValueError(f"{destination}: existing parquet has the wrong schema")
        return parquet.metadata.num_rows

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".rise-raw-parquet-", suffix=".parquet", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    manifest_temporary_path: Path | None = None
    schema = pa.schema([("docid", pa.string()), ("text", pa.string()), ("url", pa.string())])
    count = 0
    try:
        with pq.ParquetWriter(temporary_path, schema, compression="zstd") as writer:
            for shard in shards:
                with pa.memory_map(str(shard), "r") as stream:
                    reader = ipc.open_stream(stream)
                    missing = set(schema.names) - set(reader.schema.names)
                    if missing:
                        raise ValueError(f"{shard}: missing columns: {', '.join(sorted(missing))}")
                    for batch in reader:
                        table = pa.Table.from_batches([batch]).select(schema.names).cast(schema)
                        writer.write_table(table)
                        count += table.num_rows
        if count == 0:
            raise ValueError(f"{directory}: empty Arrow cache")
        output_stat = temporary_path.stat()
        manifest_fd, manifest_temporary = tempfile.mkstemp(prefix=".rise-arrow-source-", suffix=".json", dir=destination.parent)
        manifest_temporary_path = Path(manifest_temporary)
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as output:
            json.dump({"source": manifest, "output": {
                "size": output_stat.st_size, "mtime_ns": output_stat.st_mtime_ns,
            }}, output, indent=2)
            output.write("\n")
        # Hard links create files only when absent; a concurrent export cannot
        # replace output from a different source.
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise ValueError(f"parquet output appeared during conversion: {destination}") from exc
        try:
            os.link(manifest_temporary_path, provenance_path)
        except Exception:
            destination.unlink()
            raise
    finally:
        temporary_path.unlink(missing_ok=True)
        if manifest_temporary_path is not None:
            manifest_temporary_path.unlink(missing_ok=True)
    return count


def _docid(row: Row) -> str:
    value = row.get("docid")
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid docid: {value!r}")
    did = str(value)
    if not did or did != did.strip():
        raise ValueError(f"empty or whitespace-padded docid: {value!r}")
    return did


def _url(row: Row, did: str) -> str:
    value = row.get("url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"docid {did!r}: url must be a nonempty string")
    return value


def _component(value: str, *, max_bytes: int) -> str:
    """Keep readable Unicode letters/numbers, ASCII dots/dashes and underscores."""
    value = unicodedata.normalize("NFKC", value)
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-") or "index"
    # File systems usually limit components to 255 UTF-8 bytes. Leave room
    # for a collision suffix and the .txt extension.
    while len(cleaned.encode("utf-8")) > max_bytes:
        cleaned = cleaned[:-1]
    return cleaned.rstrip("._-") or "index"


def _candidate_relpath(url: str, did: str) -> str:
    try:
        parsed = urlsplit(url)
        # One BrowseComp-Plus row stores a PDF viewer URL rather than the
        # underlying page URL.  Use its embedded HTTPS target for the local
        # filename, while retaining the original URL in the corpus metadata.
        if parsed.scheme.lower() == "chrome-extension":
            embedded = parsed.path.lstrip("/")
            if embedded.startswith(("https://", "http://")):
                parsed = urlsplit(embedded)
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"docid {did!r}: invalid URL {url!r}") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        raise ValueError(f"docid {did!r}: expected an absolute HTTP(S) URL, got {url!r}")
    domain = _component(host.lower(), max_bytes=120)
    last_segment = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1]) or "index"
    title = _component(last_segment, max_bytes=180)
    return f"{domain}/{title}.txt"


def _validate_relpath(relpath: str) -> None:
    if (not isinstance(relpath, str) or not relpath or "\\" in relpath
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in relpath)):
        raise ValueError(f"unsafe corpus relative path: {relpath!r}")
    path = PurePosixPath(relpath)
    if (path.is_absolute() or len(path.parts) < 2 or path.suffix != ".txt"
            or any(part in {"", ".", ".."} for part in relpath.split("/"))):
        raise ValueError(f"unsafe corpus relative path: {relpath!r}")


def _load_existing_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("relpath_to_docid"), dict):
        raise ValueError(f"{path}: expected a relpath_to_docid object")
    mapping = payload["relpath_to_docid"]
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for relpath, did in mapping.items():
        _validate_relpath(relpath)
        if not isinstance(did, str) or not did or did in seen_ids:
            raise ValueError(f"{path}: invalid or duplicate docid {did!r}")
        folded = unicodedata.normalize("NFC", relpath).casefold()
        if folded in seen_paths:
            raise ValueError(f"{path}: paths collide on a case-insensitive filesystem: {relpath!r}")
        seen_ids.add(did)
        seen_paths.add(folded)
    return mapping


def _metadata(source: RowSource, label: str) -> dict[str, str]:
    urls: dict[str, str] = {}
    for row in source(("docid", "url")):
        did = _docid(row)
        if did in urls:
            raise ValueError(f"{label}: duplicate docid {did!r}")
        urls[did] = _url(row, did)
    if not urls:
        raise ValueError(f"{label}: empty input")
    return urls


def _generated_map(urls: Mapping[str, str]) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for did, url in urls.items():
        candidate = _candidate_relpath(url, did)
        folded = unicodedata.normalize("NFC", candidate).casefold()
        groups[folded].append(did)
    by_id: dict[str, str] = {}
    # Reserve every unsuffixed singleton before assigning suffixes, so a
    # digest-suffixed collision cannot take a later singleton's name.
    occupied: set[str] = {key for key, group in groups.items() if len(group) == 1}
    for group in sorted(groups.values(), key=lambda g: min(g)):
        for did in sorted(group):
            candidate = _candidate_relpath(urls[did], did)
            if len(group) == 1:
                path = candidate
            else:
                stem = candidate[:-4]
                digest = hashlib.sha256(did.encode("utf-8")).hexdigest()
                for digest_len in range(12, 65):
                    path = f"{stem}-{digest[:digest_len]}.txt"
                    if path.casefold() not in occupied:
                        break
                else:
                    raise ValueError("could not assign collision-free corpus filenames")
            # A suffixed name could equal an unrelated unsuffixed candidate.
            # Raise rather than silently changing paths based on row order.
            folded = unicodedata.normalize("NFC", path).casefold()
            if folded in occupied and len(group) != 1:
                raise ValueError(f"generated path collision: {path!r}; use an official map")
            occupied.add(folded)
            by_id[did] = path
    return {relpath: did for did, relpath in sorted(by_id.items(), key=lambda item: item[1])}


def _safe_destination(root: Path, relpath: str) -> Path:
    _validate_relpath(relpath)
    current = root
    for part in PurePosixPath(relpath).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symlink inside corpus tree: {current}")
        current.mkdir(exist_ok=True)
    destination = current / PurePosixPath(relpath).name
    if destination.is_symlink():
        raise ValueError(f"refusing symlink inside corpus tree: {destination}")
    return destination


def _write_if_absent_or_same(destination: Path, text: str) -> None:
    data = text.encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise ValueError(f"existing corpus file differs; refusing to overwrite: {destination}")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=".rise-export-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        try:
            os.link(tmp_name, destination)  # create without overwriting a concurrent file
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != data:
                raise ValueError(f"existing corpus file differs; refusing to overwrite: {destination}")
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _write_tree(
    source: RowSource, root: Path, by_id: Mapping[str, str],
    expected_urls: Mapping[str, str], label: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    seen: set[str] = set()
    for row in source(("docid", "text", "url")):
        did = _docid(row)
        if did in seen:
            raise ValueError(f"{label}: duplicate docid {did!r} on write pass")
        if did not in by_id:
            raise ValueError(f"{label}: unexpected docid {did!r} on write pass")
        if _url(row, did) != expected_urls[did]:
            raise ValueError(f"{label}: URL changed between metadata and write pass for {did!r}")
        seen.add(did)
        value = row.get("text")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label}: docid {did!r} has non-string text")
        destination = _safe_destination(root, by_id[did])
        _write_if_absent_or_same(destination, value or "")
    if seen != by_id.keys():
        missing = sorted(by_id.keys() - seen)
        raise ValueError(f"{label}: {len(missing)} docids disappeared on write pass; first: {missing[:3]}")


def _check_existing_tree(root: Path, expected_relpaths: set[str]) -> None:
    """Reject stale files that could silently contaminate a corpus tree."""
    if not root.exists():
        return
    if root.is_symlink():
        raise ValueError(f"refusing symlink as corpus output root: {root}")
    if not root.is_dir():
        raise ValueError(f"corpus output is not a directory: {root}")
    for file in root.rglob("*.txt"):
        if file.is_symlink():
            raise ValueError(f"refusing symlink inside corpus tree: {file}")
        relpath = file.relative_to(root).as_posix()
        if relpath not in expected_relpaths:
            raise ValueError(f"stale .txt file outside the filename map: {file}")


def export_from_sources(
    raw_source: RowSource,
    *,
    raw_out: Path,
    map_path: Path,
    structured_source: RowSource | None = None,
    structured_out: Path | None = None,
    existing_map: Path | None = None,
) -> int:
    """Export from repeatable row sources; public for local synthetic tests."""
    raw_out = Path(raw_out)
    map_path = Path(map_path)
    if structured_source is not None and structured_out is None:
        raise ValueError("structured_out is required with structured_source")
    if structured_out is not None:
        structured_out = Path(structured_out)
        a, b = raw_out.resolve(), structured_out.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("raw and structured output trees must be separate")

    raw_urls = _metadata(raw_source, "raw parquet")
    if structured_source is not None:
        structured_urls = _metadata(structured_source, "structured parquet")
        if structured_urls != raw_urls:
            missing = set(raw_urls) - set(structured_urls)
            extra = set(structured_urls) - set(raw_urls)
            changed = {d for d in raw_urls.keys() & structured_urls.keys()
                       if raw_urls[d] != structured_urls[d]}
            raise ValueError(
                "structured parquet must have exactly the same docid→url values "
                f"as raw parquet (missing={len(missing)}, extra={len(extra)}, "
                f"changed URLs={len(changed)})"
            )

    map_source = Path(existing_map) if existing_map is not None else (map_path if map_path.exists() else None)
    if map_source is not None:
        mapping = _load_existing_map(map_source)
        if set(mapping.values()) != set(raw_urls):
            raise ValueError(f"{map_source}: docids do not exactly match raw parquet")
    else:
        mapping = _generated_map(raw_urls)
    if map_path.exists() and _load_existing_map(map_path) != mapping:
        raise ValueError(f"{map_path}: existing map differs; refusing to overwrite")
    by_id = {did: relpath for relpath, did in mapping.items()}
    _check_existing_tree(raw_out, set(mapping))
    if structured_source is not None:
        assert structured_out is not None
        _check_existing_tree(structured_out, set(mapping))
    _write_tree(raw_source, raw_out, by_id, raw_urls, "raw parquet")
    if structured_source is not None:
        assert structured_out is not None
        _write_tree(structured_source, structured_out, by_id, structured_urls, "structured parquet")
    if not map_path.exists():
        map_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".rise-map-", dir=map_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump({"relpath_to_docid": mapping}, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            if map_path.exists():
                if _load_existing_map(map_path) != mapping:
                    raise ValueError(f"{map_path}: map changed during export")
            else:
                try:
                    os.link(tmp_name, map_path)
                except FileExistsError:
                    if _load_existing_map(map_path) != mapping:
                        raise ValueError(f"{map_path}: map changed during export")
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    return len(by_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-parquet", type=Path,
                        help="Existing local BrowseComp-Plus parquet; never downloaded")
    source.add_argument("--raw-arrow-dir", type=Path,
                        help="Existing local Hugging Face Arrow shard directory; never downloaded")
    parser.add_argument("--raw-parquet-out", type=Path, default=REPO_ROOT / "corpus" / "browsecomp_plus.parquet",
                        help="Local parquet created from --raw-arrow-dir for the BM25 index")
    parser.add_argument("--structured-parquet", type=Path,
                        help="Optional local TOC-structured parquet with matching docids and URLs")
    parser.add_argument("--raw-out", type=Path, default=REPO_ROOT / "corpus" / "bcp_plus")
    parser.add_argument("--structured-out", type=Path,
                        default=REPO_ROOT / "corpus" / "bcp_plus_structured")
    parser.add_argument("--filename-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--existing-map", type=Path,
                        help="Reuse the official path→docid map. If omitted, an existing --filename-map is reused.")
    args = parser.parse_args()
    for path in (args.raw_parquet, args.structured_parquet, args.existing_map):
        if path is not None and not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.raw_arrow_dir is not None and not args.raw_arrow_dir.is_dir():
        parser.error(f"Arrow directory does not exist: {args.raw_arrow_dir}")
    if args.raw_parquet is not None and args.raw_parquet_out != REPO_ROOT / "corpus" / "browsecomp_plus.parquet":
        parser.error("--raw-parquet-out is used only with --raw-arrow-dir")
    raw_parquet = args.raw_parquet if args.raw_parquet is not None else args.raw_parquet_out
    if args.structured_parquet is not None and raw_parquet.resolve() == args.structured_parquet.resolve():
        parser.error("raw and structured parquet files must be different")
    if args.raw_arrow_dir is not None:
        n_rows = convert_arrow_cache_to_parquet(args.raw_arrow_dir, raw_parquet)
        print(f"Converted/reused {n_rows:,} local Arrow rows in {raw_parquet}")
    generated_paths = args.existing_map is None and not args.filename_map.exists()
    raw_source: RowSource = lambda columns: iter_parquet_rows(raw_parquet, columns)
    structured_source = (
        (lambda columns: iter_parquet_rows(args.structured_parquet, columns))
        if args.structured_parquet is not None else None
    )
    count = export_from_sources(
        raw_source, raw_out=args.raw_out, map_path=args.filename_map,
        structured_source=structured_source,
        structured_out=args.structured_out if structured_source is not None else None,
        existing_map=args.existing_map,
    )
    print(f"Exported {count:,} documents to {args.raw_out}")
    if structured_source is not None:
        print(f"Exported {count:,} structured documents to {args.structured_out}")
    print(f"Filename map: {args.filename_map}")
    print(f"Raw BM25 parquet: {raw_parquet}")
    if generated_paths:
        print("Warning: generated paths may differ from the official exporter")


if __name__ == "__main__":
    main()
