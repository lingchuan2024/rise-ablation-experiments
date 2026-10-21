#!/usr/bin/env python3
"""Build a BM25 index over the BrowseComp-Plus 100k corpus.

Output goes to outputs/bm25_full/, containing:
  - bm25_index/   (bm25s on-disk index)
  - doc_ids.json  (parallel list of corpus docids)
  - meta.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rise.retrieval import build_index

DEFAULT_CORPUS = REPO_ROOT / "corpus" / "browsecomp_plus.parquet"
DEFAULT_OUT = REPO_ROOT / "runs" / "bm25_full"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    t0 = time.time()
    meta = build_index(args.corpus, args.out, k1=args.k1, b=args.b)
    elapsed = time.time() - t0
    print(f"\nbuilt index over {meta['n_docs']:,} docs in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
