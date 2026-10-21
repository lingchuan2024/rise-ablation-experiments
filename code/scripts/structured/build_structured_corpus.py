"""Materialize the restructured BCP corpus as a parallel `bc_plus_docs` tree.

Read each (relpath, docid) pair from corpus_filename_docid_map.json, look up the
restructured_text for that docid in our full-corpus parquet, and write it to
`corpus/bcp_plus_structured/<relpath>` (same directory structure as the original
corpus, so RISE can use it via `--bc-plus-docs` without any other code changes).

Run once before any A/B experiment. Idempotent: re-running overwrites files,
so it's safe to re-execute if the parquet is regenerated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]  # repo root (works on any host)
DEFAULT_OUT_DIR = ROOT / "corpus" / "bcp_plus_structured"
DEFAULT_MAP = ROOT / "runs" / "corpus_filename_docid_map.json"
DEFAULT_PARQUET = ROOT / "runs" / "structured_corpus" / "data.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                        help="Local TOC-augmented parquet from run_full_corpus.py")
    parser.add_argument("--filename-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    parquet = args.parquet.expanduser().resolve()
    map_file = args.filename_map.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas and a parquet engine are required to build the local structured document tree") from exc
    if not map_file.is_file():
        raise SystemExit(f"local filename map missing: {map_file}")
    if not parquet.is_file():
        raise SystemExit(f"local structured parquet missing: {parquet}")
    print(f"Loading filename map …")
    m = json.loads(map_file.read_text(encoding="utf-8"))
    relpath_to_docid: dict[str, str] = m["relpath_to_docid"]
    bad_paths = [path for path in relpath_to_docid if
                 not isinstance(path, str) or PurePosixPath(path).is_absolute()
                 or ".." in PurePosixPath(path).parts or "\\" in path]
    if bad_paths:
        raise SystemExit(f"filename map contains unsafe relative path: {bad_paths[0]!r}")
    print(f"  {len(relpath_to_docid):,} relpath → docid entries")

    print(f"Loading restructured corpus parquet ({parquet.stat().st_size/1e9:.2f} GB) …")
    df = pd.read_parquet(parquet, columns=["docid", "restructured_text"])
    df["docid"] = df["docid"].astype(str)
    docid_to_text: dict[str, str] = dict(zip(df["docid"], df["restructured_text"]))
    print(f"  {len(docid_to_text):,} docid → restructured_text entries")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir} …")

    t0 = time.time()
    n_written = 0
    n_missing_text = 0
    n_empty_text = 0
    total_bytes = 0
    last_report = 0
    for relpath, docid in relpath_to_docid.items():
        text = docid_to_text.get(str(docid))
        if text is None or pd.isna(text):
            n_missing_text += 1
            continue
        if not text:
            n_empty_text += 1
            continue
        dest = out_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Use byte-level write to be safe with weird Unicode in restructured text.
        dest.write_text(text, encoding="utf-8")
        n_written += 1
        total_bytes += len(text)
        # Periodic progress (every 5000 files)
        if n_written - last_report >= 5000:
            elapsed = time.time() - t0
            rate = n_written / max(0.1, elapsed)
            print(f"  {n_written:>6,} / {len(relpath_to_docid):,}  ({rate:.0f}/s, {total_bytes/1e9:.2f} GB)")
            last_report = n_written

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.")
    print(f"  written:        {n_written:,}")
    print(f"  missing text:   {n_missing_text:,}")
    print(f"  empty text:     {n_empty_text:,}")
    print(f"  total bytes:    {total_bytes/1e9:.2f} GB")
    print(f"  corpus tree:    {out_dir}")


if __name__ == "__main__":
    main()
