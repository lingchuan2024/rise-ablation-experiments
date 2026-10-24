"""Synthetic filesystem tests only; never accesses the production root."""
from pathlib import Path
import base64
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from operations import prune_verified_workspace_archives as m


class ArchivePruningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root_patch = patch.object(m, 'ROOT', str(self.root))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        (self.root / m.OUTPUT_PREFIX).mkdir(parents=True)
        self.local_path = m.OUTPUT_PREFIX + '/local.json'
        self.ack_path = m.OUTPUT_PREFIX + '/acks.json'
        self.audit_path = m.OUTPUT_PREFIX + '/audit.json'
        self.journal_path = m.OUTPUT_PREFIX + '/deletion.jsonl'
        entries, allowlist, ack_rows = [], [], []
        for qid in range(1, m.COUNT + 1):
            relative = m.PREFIX + f'/auto_rerank_qid{qid}_20261024_1200/workspace.tar.gz'
            parent = (self.root / relative).parent
            parent.mkdir(parents=True)
            archive_bytes = ('opaque fixture ' + str(qid)).encode()
            (self.root / relative).write_bytes(archive_bytes)
            archive = m.raw_identity(archive_bytes)
            binding = {'source_root': str(self.root), 'namespace': 'strict_recovery_v1',
                       'arm': 'auto_rerank', 'query_id': qid, 'manifest_sha256': m.MANIFEST}
            receipt = {'version': 'terminal-workspace-maintenance-v1', 'archive': archive,
                       'binding': binding, 'manifest': {'sha256': 'fixture manifest', 'bytes': 5},
                       'files': 2, 'directories': 1, 'created_unix': 1}
            receipt['receipt_sha256'] = m.digest(receipt)
            receipt_raw = self.encode(receipt)
            (parent / 'receipt.json').write_bytes(receipt_raw)
            ack = {'version': receipt['version'], 'kind': 'local_workspace_download_verified',
                   'verified': True, 'archive': archive, 'binding': binding,
                   'manifest': receipt['manifest'], 'files': 2, 'directories': 1,
                   'receipt_file_sha256': m.raw_identity(receipt_raw)['sha256'],
                   'receipt_bytes': len(receipt_raw), 'verified_unix': 2}
            ack['acknowledgement_sha256'] = m.digest(ack)
            ack_raw = self.encode(ack)
            deletion = {'version': receipt['version'], 'binding': binding, 'deleted': True,
                        'compact_artifacts_preserved': True, 'working_parent_preserved': True,
                        'runtime_parent_preserved': True, 'files_removed': 2, 'directories_removed': 1,
                        'acknowledgement_sha256': ack['acknowledgement_sha256'], 'finished_unix': 3}
            deletion['deletion_sha256'] = m.digest(deletion)
            deletion_raw = self.encode(deletion)
            (parent / 'deletion_receipt.json').write_bytes(deletion_raw)
            e = {'arm': 'auto_rerank', 'query_id': qid, 'verified': True, 'errors': [],
                 'prior_full_member_verification': True, 'prior_workspace_deletion_closed': True,
                 'remote_archive_relative_path': relative, 'archive': archive,
                 'receipt_file': m.raw_identity(receipt_raw), 'local_verification_file': m.raw_identity(ack_raw),
                 'deletion_receipt_file': m.raw_identity(deletion_raw),
                 'acknowledgement_sha256': ack['acknowledgement_sha256'],
                 'deletion_sha256': deletion['deletion_sha256']}
            entries.append(e)
            allowlist.append({'relative_path': relative, 'arm': 'auto_rerank', 'query_id': qid, **archive})
            ack_rows.append({'archive_relative_path': relative, 'raw_base64': base64.b64encode(ack_raw).decode()})
        self.local = {'version': 'local-backup-reaudit-v1', 'verified': True,
                      'remote_root': str(self.root), 'local_backup_retained': True,
                      'required_entries': m.COUNT, 'verified_entries': m.COUNT, 'errors': [],
                      'entries': entries, 'remote_archive_cleanup_allowlist': allowlist}
        self.ack_rows = ack_rows
        self.save_inputs()
        self.others = self.root / 'other_person'
        self.others.mkdir()
        (self.others / 'workspace.tar.gz').write_bytes(b'not ours')
        (self.root / 'compact.json').write_bytes(b'compact result')

    def encode(self, value):
        return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()

    def save_inputs(self):
        raw = self.encode(self.local)
        self.expected = m.raw_identity(raw)
        (self.root / self.local_path).write_bytes(raw)
        (self.root / self.ack_path).write_bytes(self.encode({
            'version': 'original-local-workspace-ack-bundle-v1', 'local_audit_file': self.expected,
            'entries': self.ack_rows}))

    def audit(self):
        with m.open_root(str(self.root)) as fd:
            report = m.audit(fd, self.local_path, self.expected, self.ack_path, self.audit_path)
        self.audit_expected = m.raw_identity((self.root / self.audit_path).read_bytes())
        return report

    def delete(self):
        with m.open_root(str(self.root)) as fd:
            return m.delete(fd, self.local_path, self.expected, self.ack_path,
                            self.audit_path, self.audit_expected, self.journal_path)

    def tar(self, index=0):
        return self.root / self.local['remote_archive_cleanup_allowlist'][index]['relative_path']

    def assert_preserved(self):
        self.assertEqual((self.others / 'workspace.tar.gz').read_bytes(), b'not ours')
        self.assertEqual((self.root / 'compact.json').read_bytes(), b'compact result')
        for e in self.local['entries']:
            parent = (self.root / e['remote_archive_relative_path']).parent
            self.assertTrue((parent / 'receipt.json').is_file())
            self.assertTrue((parent / 'deletion_receipt.json').is_file())

    def test_complete_only_64_exact_archives_removed_without_remote_acks(self):
        self.assertTrue(self.audit()['verified'])
        result = self.delete()
        self.assertTrue(result['complete'])
        self.assertEqual(result['deleted_archives'], 64)
        self.assertTrue(all(not (self.root / x['relative_path']).exists()
                            for x in self.local['remote_archive_cleanup_allowlist']))
        self.assert_preserved()
        events = [json.loads(x) for x in (self.root / self.journal_path).read_text().splitlines()]
        self.assertEqual(sum(x['event'] == 'unlink_intent' for x in events), 64)
        self.assertEqual(sum(x['event'] == 'unlinked' for x in events), 64)

    def test_wrong_root_rejected(self):
        with self.assertRaises(RuntimeError):
            with m.open_root(str(self.root) + '/other_person'):
                pass

    def test_last_archive_drift_before_audit_deletes_nothing(self):
        self.tar(-1).write_bytes(b'corrupt archive')
        self.assertFalse(self.audit()['verified'])
        self.assertFalse(self.delete()['complete'])
        self.assertTrue(self.tar().is_file())
        self.assert_preserved()

    def test_last_archive_drift_after_audit_deletes_nothing(self):
        self.assertTrue(self.audit()['verified'])
        self.tar(-1).write_bytes(b'corrupt after audit')
        result = self.delete()
        self.assertFalse(result['complete'])
        self.assertEqual(result['deleted_archives'], 0)
        self.assertTrue(self.tar().is_file())

    def test_archive_symlink_rejected(self):
        self.tar().unlink()
        self.tar().symlink_to(self.others / 'workspace.tar.gz')
        self.assertFalse(self.audit()['verified'])
        self.assert_preserved()

    def test_parent_symlink_rejected(self):
        parent = self.tar().parent
        alternate = self.root / 'moved_backup'
        parent.rename(alternate)
        parent.symlink_to(alternate)
        self.assertFalse(self.audit()['verified'])
        self.assertTrue((alternate / 'workspace.tar.gz').exists())

    def test_hardlinked_archive_rejected(self):
        os.link(self.tar(), self.root / 'extra_hardlink')
        self.assertFalse(self.audit()['verified'])

    def test_local_raw_identity_and_ack_raw_identity_must_match(self):
        (self.root / self.local_path).write_bytes(b'{}')
        self.assertFalse(self.audit()['verified'])
        self.assertTrue(self.tar().exists())

    def test_ack_raw_bytes_mismatch_rejected(self):
        self.ack_rows[-1]['raw_base64'] = base64.b64encode(b'{}').decode()
        self.save_inputs()
        self.assertFalse(self.audit()['verified'])
        self.assertTrue(self.tar().exists())

    def test_existing_remote_ack_must_match_original(self):
        (self.tar().parent / 'local_verification.json').write_bytes(b'{}')
        self.assertFalse(self.audit()['verified'])
        self.assertTrue(self.tar().exists())

    def test_path_traversal_rejected_even_with_repinned_audit(self):
        self.local['remote_archive_cleanup_allowlist'][0]['relative_path'] = m.PREFIX + '/../outside/workspace.tar.gz'
        self.save_inputs()
        self.assertFalse(self.audit()['verified'])

    def test_changed_remote_receipt_invalidates_full_gate(self):
        self.assertTrue(self.audit()['verified'])
        (self.tar(-1).parent / 'receipt.json').write_bytes(b'{}')
        result = self.delete()
        self.assertFalse(result['complete'])
        self.assertEqual(result['deleted_archives'], 0)
        self.assertTrue(self.tar().exists())

    def test_output_cannot_overwrite_existing_receipt(self):
        self.assertTrue(self.audit()['verified'])
        original = (self.root / self.audit_path).read_bytes()
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual((self.root / self.audit_path).read_bytes(), original)
        (self.root / self.journal_path).write_text('do not replace')
        with self.assertRaises(FileExistsError):
            self.delete()
        self.assertEqual((self.root / self.journal_path).read_text(), 'do not replace')
        self.assertTrue(self.tar().exists())

    def test_mid_delete_failure_stops_and_leaves_partial_journal(self):
        self.assertTrue(self.audit()['verified'])
        original = m.unlink_archive
        calls = []
        def fail_second(fd, row):
            calls.append(row['relative_path'])
            if len(calls) == 2:
                raise RuntimeError('simulated late failure')
            return original(fd, row)
        with patch.object(m, 'unlink_archive', side_effect=fail_second):
            result = self.delete()
        self.assertFalse(result['complete'])
        self.assertEqual(result['deleted_archives'], 1)
        self.assertFalse(self.tar().exists())
        self.assertTrue(self.tar(1).exists())
        self.assertTrue(self.tar(2).exists())
        events = [json.loads(x) for x in (self.root / self.journal_path).read_text().splitlines()]
        self.assertEqual(events[-1]['event'], 'finished')
        self.assertIn('simulated late failure', events[-1]['error'])
        self.assert_preserved()


if __name__ == '__main__':
    unittest.main()
