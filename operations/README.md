# Continue automatic remaining80, then snippet paging first20

`continuation_suite.py` is an administrative scheduler for the 2026-10-23
follow-up. It preserves all 135 frozen solver/tool source files and the exact
common configuration. Its isolated `strict_continue_v1` namespace admits only
`auto_rerank` on fixed100[20:] followed by `search_paging` on fixed100[:20].
The existing automatic20 is verified as an immutable external reference,
including its original manifest, completion receipt and102 artifact files;
neither those20 nor the historical B100 control are solved again.

Prepare a new output root with a new serving epoch and a configuration from
`template --control-root CONTROL_ROOT --reuse-auto20-root OLD_AUTO20_ROOT`.
Run `prepare --config CONFIG --service-pid ACTUAL_PID`, then `live-preflight`,
then `run`, all with `--root NEW_ROOT` before the subcommand. The run holds its
own worker lock and the old automatic20 lock, schedules at most four questions,
validates all80 automatic terminals before starting paging, and saves
`auto80_completion.json`, `paging20_completion.json`, `phase_state.json` and
`continuation_completion.json` under `reports/strict_continue_v1/`.

The new manifest also freezes the administrative scheduler's own SHA. Natural
terminal failures remain final; infrastructure recovery uses the unchanged
checkpoint and cumulative budget rules with explicit `--resume-infrastructure`.
Known atomic-write temporary files may coexist with committed artifacts but
are neither terminals nor a substitute for checkpoint/ledger evidence. The
snapshot exporter must exclude them. This follow-up does not mark the separate
three-arm100 suite complete or alter prompts to improve scores.

`continuation_snapshot.py` provides `export`, `extract` and `verify`. Export
requires all100 new terminals, both stage receipts, final phase `complete`,
an idle worker lock and proof that the exact worker PID/start-ticks/boot
identity has exited. It includes compact original-byte artifacts for the100
new runs and the20 reused automatic runs, the six-file public control bundle,
frozen sources, administrative receipts and220 public finals without labels.
It excludes duplicate working pools, weights, indexes, credentials and atomic
temporary files. Keep its raw output in ignored directories.

Download the tar and independent receipt, separately verify their remote/local
SHA256 and byte counts, then use `extract --archive TAR --receipt RECEIPT
--expected-receipt-sha256 SHA --expected-receipt-bytes N --output-dir FRESH_DIR`.
Use `--root FRESH_DIR verify` with the same archive/receipt arguments; require
`verified=true`, `terminal_rows=100`, `merged_auto_rerank_rows=100`,
`new_search_paging_rows=20`, `public_finals=220` and
`archive_bytes_verified=true` before normal cloud shutdown. The old server
paths remain unchanged in manifests; verification uses the archived bundles.
An integration test generates100 synthetic terminal artifacts through the
real recorder/checkpoint contracts, exports them, deletes the simulated
server tree and verifies the downloaded archive without any server paths.

# Automatic-rerank 20-question run

`auto20_pipeline.py` is an administrative driver outside the frozen runtime.
It performs prepare, live preflight and `run20` in one process, using the
existing configuration and service-launch receipt. It does not launch a
second worker, reset budgets, rerun natural terminal results or schedule the
other ablations. The frozen common configuration uses four concurrent
questions. An instantaneous vLLM request count below four is expected when
some questions are executing retrieval, reranking or file tools.

`auto20_audit.py --root ARTIFACT_ROOT` reads saved tool metadata only. It
reports checkpoints, terminal results, ordinary tool errors, automatic
reranking and actual scoring counts. It neither evaluates answers nor
inspects hidden reasoning. Checkpoints describe completed turns; a newly
started question may have a request contract but no checkpoint yet.

After the pipeline reports completion and releases its worker lock, export
a new, exclusive snapshot directory:

```bash
python operations/auto20_snapshot.py --root ARTIFACT_ROOT export \
  --output-dir ARTIFACT_ROOT/reports/strict_auto20_v1/snapshots/UNIQUE_NAME \
  --operation-file operations/service_launch.json \
  --operation-file operations/worker_launch.json \
  --operation-file operations/code_deployment.json \
  --operation-file operations/docker_image_restored.json \
  --operation-file operations/first_prepare_interruption.json \
  --operation-file operations/vllm_auto20.log
```

Download both `auto20_snapshot.tar.gz` and its separate
`snapshot_receipt.json`. Safely extract the archive to a new local directory
and verify it against the unchanged local source checkout and frozen public
control bundle:

```bash
python operations/auto20_snapshot.py --root EXTRACT_ROOT verify \
  --receipt DOWNLOADED_RECEIPT --archive DOWNLOADED_ARCHIVE \
  --control-root LOCAL_CONTROL_ROOT
```

Require the exporter's `ready_for_shutdown: true` and the local verifier's
`verified: true`, `terminal_rows: 20`, `fresh_review_finals: 40` and
`archive_bytes_verified: true`. The verifier checks the archive, every recorded file, frozen source identity,
20 terminal/request/ledger/trace/checkpoint anchors, public control identity,
and the anonymous 40-answer packet. It preserves the original server paths
and manifest bytes rather than rewriting provenance to fit the local machine.
After also confirming the worker has exited and no unrelated compute is
running, normally power off the instance and visually verify that it is off;
keep persistent storage and results.

Grade all 40 public finals freshly under the frozen rubric while keeping
`private40.json` out of reviewers' context. Use `load_verified_snapshot` to
load rows, cohort and manifest offline, followed by
`strict_ablations.auto20_analysis.import_reviews`, `load_labels` and
`summarize`. This avoids restarting the server merely because the original
manifest contains absolute server paths. Keep raw traces and the private
mapping in ignored artifact directories; publish aggregate results and
provenance separately.

# B100 completion-based shutdown

Run `python operations/strict_qwen35_check_completion.py` from the independent
experiment checkout on its server. This administrative script is outside the
frozen experiment runtime; it neither runs questions nor shuts down the machine.

The completion monitor runs every ten minutes. It must inspect the returned
JSON and require `ready_for_shutdown: true`; an exit status of zero alone is
insufficient. The gate requires both fixed100 cohorts, the completed pipeline
receipts, an idle worker lock, and verified terminal/request/token/trace/
checkpoint/maintenance artifacts for all200 runs. Empty finals and legitimate
natural budget failures remain terminal results.

Successful verification creates a hash-only integrity receipt and a fresh
blinded public-final scoring export. The private arm map remains private.
Confirm the pipeline has exited and that the instance has no other user jobs,
preserve the minimal scoring export locally, then normally power off q3535bf
in the Yingbo Cloud console. Preserve its persistent disk and all results, and
visually verify the cloud instance is off. This is completion-based shutdown,
not a fixed wall-clock cutoff.

Administrative validation used200 synthetic terminal fixtures: a legitimate
empty-final/max-turn case, idempotent checks/exports, and corrupted-ledger
rejection. No benchmark question was executed by these tests.
