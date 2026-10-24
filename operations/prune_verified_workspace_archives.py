"""Prune only 64 independently downloaded duplicate workspace archives.

Administrative code: never walks/deletes a workspace, compact result, or local
backup. Audit is read-only. Delete requires the current audit receipt, independently
pinned local audit and original local ACK bytes. Its exclusive JSONL journal is
fsynced before/after every unlink, including partial failure. No model calls.
"""
from contextlib import contextmanager
from pathlib import PurePosixPath
import argparse
import base64
import hashlib
import json
import os
import re
import stat
import time

ROOT = '/root/data/rise-strict-continue-20261023'
PREFIX = 'reports/strict_recovery_v1/workspace_backups'
OUTPUT_PREFIX = 'reports/strict_no_bash20_v1/operations'
COUNT = 64
VERSION = 'verified-workspace-archive-pruning-v1'
MANIFEST = '85ea1642f45d659026b4a081818c896020fbdcbd9eed1d0eef2d9f7d9740f8ca'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def raw_identity(data):
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def checked(value, field):
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            field + ' logical digest mismatch')


def fingerprint(info):
    return {'dev': info.st_dev, 'ino': info.st_ino, 'mode': info.st_mode,
            'uid': info.st_uid, 'nlink': info.st_nlink, 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


@contextmanager
def open_root(root):
    require(root == ROOT, 'Only the frozen experiment root is permitted')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in PurePosixPath(root).parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        yield fd
    finally:
        os.close(fd)


def parts(relative):
    p = PurePosixPath(relative)
    require(isinstance(relative, str) and not p.is_absolute() and
            p.as_posix() == relative and all(x not in ('', '.', '..') for x in p.parts),
            'Noncanonical or escaping relative path')
    return p.parts


@contextmanager
def parent_fd(root_fd, relative):
    components = parts(relative)
    require(len(components) >= 2, 'Root-level paths are not permitted')
    anchor = os.fstat(root_fd)
    fd = os.dup(root_fd)
    try:
        for part in components[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            info = os.fstat(nxt)
            if info.st_dev != anchor.st_dev or info.st_uid != anchor.st_uid:
                os.close(nxt)
                raise RuntimeError('Parent crosses device or owner boundary')
            os.close(fd)
            fd = nxt
        yield fd, components[-1]
    finally:
        os.close(fd)


def read_file(root_fd, relative, *, retain=False, archive=False):
    with parent_fd(root_fd, relative) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), 'Not a regular file: ' + relative)
            require(before.st_uid == os.fstat(root_fd).st_uid, 'File owner differs: ' + relative)
            require(before.st_dev == os.fstat(root_fd).st_dev, 'File device differs: ' + relative)
            if archive:
                require(before.st_nlink == 1, 'Archive must have one link: ' + relative)
            if retain:
                require(before.st_size <= 4 << 20, 'Administrative JSON exceeds size bound')
            h, count, chunks = hashlib.sha256(), 0, []
            while True:
                block = os.read(fd, 8 << 20)
                if not block:
                    break
                h.update(block)
                count += len(block)
                if retain:
                    chunks.append(block)
            after = os.fstat(fd)
            linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(fingerprint(before) == fingerprint(after) == fingerprint(linked),
                    'File changed during read: ' + relative)
            identity = {'sha256': h.hexdigest(), 'bytes': count}
            require(count == before.st_size, 'Short file read')
            return identity, fingerprint(after), b''.join(chunks) if retain else None
        finally:
            os.close(fd)


def load_pinned(root_fd, relative, expected):
    identity, _, data = read_file(root_fd, relative, retain=True)
    require(identity == expected, 'Pinned administrative file bytes differ: ' + relative)
    return json.loads(data)


def load_inputs(root_fd, local_path, expected, ack_bundle_path):
    local = load_pinned(root_fd, local_path, expected)
    require(local.get('version') == 'local-backup-reaudit-v1' and local.get('verified') is True,
            'Local audit is not successful')
    require(local.get('remote_root') == ROOT and local.get('local_backup_retained') is True,
            'Local backup root/retention differs')
    require(local.get('required_entries') == local.get('verified_entries') == COUNT and
            not local.get('errors'), 'Incomplete local verification')
    entries, allow = local['entries'], local['remote_archive_cleanup_allowlist']
    require(len(entries) == len(allow) == COUNT, 'Expected exactly 64 entries')
    index = {(e['arm'], e['query_id']): e for e in entries}
    require(len(index) == COUNT, 'Duplicate arm/qid')
    require(len({a['relative_path'] for a in allow}) == COUNT, 'Duplicate archive path')
    for a in allow:
        relative = a['relative_path']
        parts(relative)
        require(re.fullmatch(re.escape(PREFIX) + r'/(auto_rerank|search_paging)_qid[0-9]+_[0-9]{8}_[0-9]{4}/workspace\.tar\.gz', relative),
                'Archive outside exact backup scope')
        require(a['arm'] in ('auto_rerank', 'search_paging') and isinstance(a['query_id'], int),
                'Invalid arm or query ID')
        require(PurePosixPath(relative).parent.name.startswith(f"{a['arm']}_qid{a['query_id']}_"),
                'Archive name does not bind arm/qid')
        e = index[(a['arm'], a['query_id'])]
        require(e.get('verified') is True and not e.get('errors') and
                e.get('prior_full_member_verification') is True and e.get('prior_workspace_deletion_closed') is True,
                'Unverified local transaction')
        require(e.get('remote_archive_relative_path') == relative and e['archive'] ==
                {'sha256': a['sha256'], 'bytes': a['bytes']}, 'Allowlist differs from local evidence')
    identity, _, raw = read_file(root_fd, ack_bundle_path, retain=True)
    bundle = json.loads(raw)
    require(bundle.get('version') == 'original-local-workspace-ack-bundle-v1' and
            bundle.get('local_audit_file') == expected, 'ACK bundle does not bind pinned audit')
    acks = bundle['entries']
    require(len(acks) == COUNT and len({x['archive_relative_path'] for x in acks}) == COUNT,
            'ACK bundle count/uniqueness mismatch')
    ack_index = {x['archive_relative_path']: x for x in acks}
    require(set(ack_index) == {a['relative_path'] for a in allow}, 'ACK bundle path set mismatch')
    for a in allow:
        e = index[(a['arm'], a['query_id'])]
        data = base64.b64decode(ack_index[a['relative_path']]['raw_base64'], validate=True)
        require(raw_identity(data) == e['local_verification_file'], 'Original ACK raw bytes mismatch')
        ack_index[a['relative_path']] = json.loads(data)
    return local, index, allow, ack_index, identity


def audit_entry(root_fd, a, e, ack):
    relative = a['relative_path']
    base = str(PurePosixPath(relative).parent)
    receipt = load_pinned(root_fd, base + '/receipt.json', e['receipt_file'])
    deletion = load_pinned(root_fd, base + '/deletion_receipt.json', e['deletion_receipt_file'])
    for value, field in ((receipt, 'receipt_sha256'), (ack, 'acknowledgement_sha256'),
                         (deletion, 'deletion_sha256')):
        checked(value, field)
    require(receipt['version'] == ack['version'] == deletion['version'] == 'terminal-workspace-maintenance-v1',
            'Original maintenance version differs')
    require(receipt['binding'] == ack['binding'] == deletion['binding'], 'Receipt binding mismatch')
    binding = receipt['binding']
    require(binding['source_root'] == ROOT and binding['namespace'] == 'strict_recovery_v1' and
            binding['arm'] == a['arm'] and binding['query_id'] == a['query_id'] and
            binding['manifest_sha256'] == MANIFEST, 'Wrong run/arm/qid binding')
    require(receipt['archive'] == ack['archive'] == e['archive'], 'Archive evidence mismatch')
    require(receipt['manifest'] == ack['manifest'], 'Original member manifest mismatch')
    require(ack['verified'] is True and ack['kind'] == 'local_workspace_download_verified' and
            ack['receipt_file_sha256'] == e['receipt_file']['sha256'] and
            ack['receipt_bytes'] == e['receipt_file']['bytes'], 'Original full-member ACK invalid')
    require(ack['acknowledgement_sha256'] == deletion['acknowledgement_sha256'] == e['acknowledgement_sha256'],
            'ACK logical chain mismatch')
    require(deletion['deletion_sha256'] == e['deletion_sha256'] and deletion['deleted'] is True and
            deletion['compact_artifacts_preserved'] is True and deletion['working_parent_preserved'] is True and
            deletion['runtime_parent_preserved'] is True, 'Original deletion not closed')
    require(receipt['files'] == ack['files'] == deletion['files_removed'] and
            receipt['directories'] == ack['directories'] == deletion['directories_removed'], 'Original counts mismatch')
    require(receipt['created_unix'] <= ack['verified_unix'] <= deletion['finished_unix'], 'Invalid chronology')
    # ACK originally arrived on stdin; it need not have been saved remotely.
    # If any recognized copy exists, it must also match the original raw ACK.
    present_acks = []
    for name in ('acknowledgement.json', 'local_acknowledgement.json', 'local_verification.json'):
        try:
            identity, _, _ = read_file(root_fd, base + '/' + name, retain=True)
        except FileNotFoundError:
            continue
        require(identity == e['local_verification_file'], 'Existing remote ACK differs: ' + name)
        present_acks.append(name)
    identity, info, _ = read_file(root_fd, relative, archive=True)
    require(identity == e['archive'], 'Remote archive SHA/bytes mismatch: ' + relative)
    return {'relative_path': relative, 'arm': a['arm'], 'query_id': a['query_id'],
            'archive': identity, 'stat': info, 'remote_ack_copies_checked': present_acks,
            'receipt_file': e['receipt_file'], 'deletion_receipt_file': e['deletion_receipt_file'],
            'verified': True}


def output_file(root_fd, relative):
    require(str(PurePosixPath(relative).parent) == OUTPUT_PREFIX, 'Output outside new administrative directory')
    with parent_fd(root_fd, relative) as (fd, name):
        return os.fdopen(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=fd), 'w', encoding='utf-8')


def append_record(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def audit(root_fd, local_path, expected, ack_bundle_path, output):
    report = {'version': VERSION, 'kind': 'remote_archive_audit', 'root': ROOT,
              'started_unix': time.time(), 'local_audit_file': expected, 'verified': False,
              'entries': [], 'error': None}
    # Reserve an exclusive output before potentially expensive reads.
    with output_file(root_fd, output) as stream:
        try:
            _, index, allow, acks, ack_identity = load_inputs(root_fd, local_path, expected, ack_bundle_path)
            report['ack_bundle_file'] = ack_identity
            for a in allow:
                report['entries'].append(audit_entry(root_fd, a, index[(a['arm'], a['query_id'])], acks[a['relative_path']]))
            report['verified'] = len(report['entries']) == COUNT
            report['total_archive_bytes'] = sum(x['archive']['bytes'] for x in report['entries'])
        except Exception as exc:
            report['error'] = type(exc).__name__ + ': ' + str(exc)
        report['finished_unix'] = time.time()
        report['audit_sha256'] = digest(report)
        stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return report


def unlink_archive(root_fd, row):
    relative = row['relative_path']
    # Rehash immediately before unlink; use the same opened parent and file.
    with parent_fd(root_fd, relative) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            require(fingerprint(before) == row['stat'] and stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                    'Archive metadata changed before unlink: ' + relative)
            h, count = hashlib.sha256(), 0
            while True:
                data = os.read(fd, 8 << 20)
                if not data:
                    break
                h.update(data)
                count += len(data)
            require({'sha256': h.hexdigest(), 'bytes': count} == row['archive'], 'Archive bytes changed before unlink')
            require(fingerprint(os.fstat(fd)) == row['stat'] and
                    fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False)) == row['stat'],
                    'Archive replaced during final verification')
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(fd)


def delete(root_fd, local_path, expected, ack_bundle_path, audit_path, audit_expected, output):
    result = {'version': VERSION, 'kind': 'remote_archive_deletion', 'root': ROOT,
              'local_audit_file': expected, 'audit_file': audit_expected,
              'deleted_archives': 0, 'deleted_bytes': 0, 'complete': False, 'error': None}
    # Never overwrite an earlier journal; intents make interruption visible.
    with output_file(root_fd, output) as stream:
        append_record(stream, {**result, 'event': 'started', 'unix': time.time()})
        try:
            previous = load_pinned(root_fd, audit_path, audit_expected)
            checked(previous, 'audit_sha256')
            require(previous.get('version') == VERSION and previous.get('kind') == 'remote_archive_audit' and
                    previous.get('root') == ROOT and previous.get('verified') is True and
                    previous.get('local_audit_file') == expected and len(previous['entries']) == COUNT,
                    'Successful matching audit required')
            _, index, allow, acks, ack_identity = load_inputs(root_fd, local_path, expected, ack_bundle_path)
            require(previous['ack_bundle_file'] == ack_identity, 'ACK bundle changed since audit')
            prior_index = {r['relative_path']: r for r in previous['entries']}
            require(len(prior_index) == COUNT and set(prior_index) == {a['relative_path'] for a in allow},
                    'Audit path set differs')
            fresh = []
            # All 64 files and chains pass again before the first unlink.
            for a in allow:
                row = audit_entry(root_fd, a, index[(a['arm'], a['query_id'])], acks[a['relative_path']])
                require(row == prior_index[row['relative_path']], 'File/evidence changed since audit')
                fresh.append(row)
            append_record(stream, {'event': 'all_64_verified_before_first_unlink', 'unix': time.time(), 'count': COUNT})
            for row in fresh:
                append_record(stream, {'event': 'unlink_intent', 'unix': time.time(),
                                       'relative_path': row['relative_path'], 'archive': row['archive']})
                unlink_archive(root_fd, row)
                result['deleted_archives'] += 1
                result['deleted_bytes'] += row['archive']['bytes']
                append_record(stream, {'event': 'unlinked', 'unix': time.time(),
                                       'relative_path': row['relative_path'], 'archive': row['archive']})
            result['complete'] = result['deleted_archives'] == COUNT
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        append_record(stream, {**result, 'event': 'finished', 'unix': time.time(),
                               'local_archives_untouched': True, 'small_receipts_and_compact_untouched': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--local-audit', required=True, help='Root-relative uploaded local audit')
    parser.add_argument('--local-audit-sha256', required=True)
    parser.add_argument('--local-audit-bytes', required=True, type=int)
    parser.add_argument('--ack-bundle', required=True, help='Root-relative original local ACK byte bundle')
    parser.add_argument('--output', required=True, help='Fresh root-relative administrative receipt or JSONL journal')
    parser.add_argument('mode', choices=('audit', 'delete'))
    parser.add_argument('--audit-receipt')
    parser.add_argument('--audit-receipt-sha256')
    parser.add_argument('--audit-receipt-bytes', type=int)
    args = parser.parse_args()
    expected = {'sha256': args.local_audit_sha256, 'bytes': args.local_audit_bytes}
    if args.mode == 'delete':
        if not all((args.audit_receipt, args.audit_receipt_sha256, args.audit_receipt_bytes)):
            parser.error('delete requires audit receipt, SHA and bytes')
    with open_root(args.root) as root_fd:
        if args.mode == 'audit':
            report = audit(root_fd, args.local_audit, expected, args.ack_bundle, args.output)
            passed = report['verified']
        else:
            report = delete(root_fd, args.local_audit, expected, args.ack_bundle, args.audit_receipt,
                            {'sha256': args.audit_receipt_sha256, 'bytes': args.audit_receipt_bytes}, args.output)
            passed = report['complete']
    print(json.dumps({'passed': passed, 'output': args.output, 'error': report['error'],
                      'deleted_archives': report.get('deleted_archives', 0)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
