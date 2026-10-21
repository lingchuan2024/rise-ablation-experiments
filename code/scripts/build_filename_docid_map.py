"""Build the (relative_path) -> docid map for the exported BrowseComp-Plus
filesystem corpus.

The agents address corpus documents by file path, while `gold_doc_ids` (and
BM25 results) are docids, so we need a map between them. The map is keyed by
the corpus-relative path used in the exported file tree
(`corpus/bcp_plus/<domain>/<sanitized_title>.txt`) and valued by the docid:

    {"relpath_to_docid": {"<domain>/<title>.txt": "<docid>", ...}}

`run_rise.py` / `run_retrieval_agent.py` read this file via `--filename-map`
(default: runs/corpus_filename_docid_map.json).

How to produce it
-----------------
The exported file tree and this map are both produced by the *official*
BrowseComp-Plus export tooling, which sanitizes titles into filenames and
de-duplicates collisions deterministically. When you export the corpus to a
flat tree (README step 2), keep the path->docid mapping it emits and write it
here as the JSON above.

We deliberately do not vendor the upstream exporter to avoid shipping a partial
copy of the BrowseComp-Plus benchmark code; apply the same filename logic as
the exporter so these keys match the files on disk exactly, or BM25 docids will
not resolve to files.
"""

if __name__ == "__main__":
    raise SystemExit(__doc__)
