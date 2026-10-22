# Completion-based shutdown

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
