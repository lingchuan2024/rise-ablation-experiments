#!/usr/bin/env python3
"""One-process administrative preparation and B-only recovery."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_experiment import runner
from operations import resume_b_only as recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--epoch-dir', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args()
    directory = recovery.epoch_directory(args.epoch_dir)
    state_path = directory / 'b_pipeline_state.json'
    def state(phase, **extra):
        runner.atomic_json(state_path, {'phase': phase, 'unix_time': time.time(),
            'scope': 'B_only', 'a_dispatches': 0, **extra})
    try:
        state('verifying_original_resources')
        # Old Lustre caches can refresh metadata during the first content read.
        # Retain the original strict byte/stat check; retry only this exact
        # pre-query metadata condition, never a question or any other error.
        for attempt in range(3):
            try:
                manifest = runner.verify_manifest()
                break
            except RuntimeError as exc:
                if str(exc) != 'Resource changed while verifying its actual bytes' or attempt == 2:
                    raise
                runner.atomic_json(directory / f'resource_metadata_retry_{attempt + 1}.json',
                    {'unix_time': time.time(), 'guard_unchanged': True,
                     'all_byte_hashes_passed_before_metadata_recheck': True,
                     'formal_queries_dispatched': 0}, immutable=True)
        if manifest['manifest_sha256'] != '6d64352b47ae0ec09ea2af6763be74d94b451754caf5e096dc98b5c920c67736':
            raise RuntimeError('Original manifest identity differs')
        state('binding_restart_epoch', manifest_sha256=manifest['manifest_sha256'])
        recovery.create_epoch(manifest, directory, args.pid)
        state('b_only_synthetic_preflight', manifest_sha256=manifest['manifest_sha256'])
        recovery.short_preflight(manifest, directory)
        state('running_B_only', manifest_sha256=manifest['manifest_sha256'], concurrency=4)
        receipt = recovery.run_b(manifest, directory, args.audit)
        state('B100_completed_integrity_export_pending', receipt=receipt)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__)
        raise


if __name__ == '__main__':
    main()
