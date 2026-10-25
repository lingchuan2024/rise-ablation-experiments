#!/usr/bin/env python3
"""Back up only validated no_bash terminal pools, with cooperative scan exclusion.

Old first20 maintenance is admitted idle or behind the live80 scan barrier, excluding the
already partially deleted153. New80 maintenance requires a real live worker and
undispatched questions; dispatch and final resource validation share its lock.
The frozen archive/member/ACK/FD-deletion algorithm is reused without changes.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operations import no_bash80_suite as job
from operations import no_bash20_suite as old_job
from operations import no_bash20_workspace_maintenance as old
from strict_ablations import runner
CORE_SHA256='f50a9fd4d53fdf34d1b4bd752a902cabd6d292bdbf28a2af5155de94868fbf81'
NAMESPACES=(old_job.DIRECTORY,job.DIRECTORY)
ARM='no_bash'
VERSION='strict-no-bash80-terminal-workspace-maintenance-v1'

def require(value,message):
    if not value:raise RuntimeError(message)

def authorize(namespace,arm,qid):
    require(namespace in NAMESPACES and arm==ARM and type(qid) is int,'Wrong namespace/arm/qid')
    queries=old_job.selected_queries() if namespace==old_job.DIRECTORY else job.selected_queries()
    require(qid in {q['query_id'] for q in queries},'Question is outside this namespace')
    require(not(namespace==old_job.DIRECTORY and qid==153),'Previously partial153 must never be retried')

def suite_for(root,namespace):
    return (old_job.NoBash20Suite if namespace==old_job.DIRECTORY else job.NoBash80Suite)(root)

def audit(root,namespace,arm,qid):
    authorize(namespace,arm,qid)
    if namespace==old_job.DIRECTORY:return old.audit_terminal(root,arm,qid)
    suite=suite_for(root,namespace)
    manifest=suite.verify_manifest(resources=False)
    query=next(q for q in job.selected_queries() if q['query_id']==qid)
    path=suite.terminal_path(arm,qid)
    require(path.is_file() and not path.is_symlink(),'No immutable natural terminal')
    base=path.parent.parent
    marker=base/'_infrastructure'/f'qid_{qid}.json'
    require(not marker.exists() and not list(marker.parent.glob(marker.name+'.*')),'Infrastructure/partial evidence forbids cleanup')
    require(not any(list((base/d).glob(f'qid_{qid}.json.*')) for d in ('_per_query','_checkpoints','request_contracts')),'Temporary publication forbids cleanup')
    dispatch=suite.validate_dispatch(manifest,query)
    terminal=runner.load(path)
    suite.validate_terminal(terminal,manifest,arm,query)
    suite.verify_execution_artifacts(manifest,arm,query,terminal)
    return {'source_root':str(Path(root).resolve()),'namespace':namespace,'arm':arm,'query_id':qid,
        'manifest_sha256':manifest['manifest_sha256'],'effective_config_sha256':manifest['config_sha256'],
        'contract_sha256':manifest['contracts'][arm]['contract_sha256'],
        'dispatch_sha256':dispatch['dispatch_sha256'],'dispatch_file_sha256':runner.file_sha(suite.dispatch_path(qid)),
        'worker_identity':dispatch['worker_identity'],'terminal_sha256':runner.file_sha(path),
        'row_sha256':terminal['row_sha256'],'execution_artifacts':terminal['execution_artifacts']}

def live_admission(root,binding,*,require_selected_terminal=True):
    suite=job.NoBash80Suite(root)
    state=runner.load(suite.reports/'phase_state.json')
    require(state.get('phase') in {'running_no_bash80','waiting_storage'},'No live maintenance during scans or tail completion')
    require(0<=time.time()-state.get('checked_unix',0)<=90,'Stale phase')
    require(state.get('manifest_sha256')==binding['manifest_sha256'] and
        state.get('worker_identity')==binding['worker_identity'] and job.process_matches(state['worker_identity']),'Wrong or dead worker')
    require(not state.get('errors') and not(suite.reports/'no_bash80_completion.json').exists(),'Error/completion barrier')
    active,pending=state.get('active_query_ids',[]),state.get('pending_query_ids',[])
    require(binding['query_id'] not in active and binding['query_id'] not in pending,'Selected pool is active/pending')
    require(pending and any(not suite.evidence_present(q) for q in pending),'Tail cleanup forbidden')
    status=suite.status()['arms'][ARM]
    require((not require_selected_terminal or binding['query_id'] in status['terminal']) and status['pending'] and
        set(status['active'])==set(status['partial_or_active']),'Actual status does not admit terminal-only maintenance')

@contextmanager
def maintenance_lock(root,namespace,arm,qid):
    authorize(namespace,arm,qid)
    new_suite=job.NoBash80Suite(root)
    if namespace==old_job.DIRECTORY:
        # All solver lock families exclude preparing or running either namespace.
        state_path=new_suite.reports/'phase_state.json'
        state=runner.load(state_path) if state_path.exists() else {}
        if state.get('phase') in {'running_no_bash80','waiting_storage'}:
            with new_suite.resource_maintenance_lock():
                state=runner.load(state_path)
                live_admission(root,{'manifest_sha256':state.get('manifest_sha256'),
                    'worker_identity':state.get('worker_identity'),'query_id':qid},require_selected_terminal=False)
                require(runner.load(Path(root)/'reports'/namespace/'phase_state.json').get('phase')=='complete','Old first20 must be fully complete')
                # This namespace lock also excludes a second old-pool maintenance process.
                lock=Path(root)/'reports'/namespace/'worker.lock'
                with lock.open('rb') as stream:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    yield
        else:
            with old_job.NoBash20Suite(root).worker_locks(),new_suite.resource_maintenance_lock():
                require(runner.load(Path(root)/'reports'/namespace/'phase_state.json').get('phase')=='complete','Old first20 must be fully complete')
                yield
    else:
        # Same lock surrounds every dispatch and full resource scan in the new80 worker.
        with new_suite.resource_maintenance_lock():
            live_admission(root,audit(root,namespace,arm,qid))
            yield

def core(namespace):
    path=ROOT/'operations/terminal_workspace_maintenance.py'
    require(runner.file_sha(path)==CORE_SHA256,'Frozen core changed')
    spec=importlib.util.spec_from_file_location('_isolated_no_bash80_maintenance_core',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    c.NAMESPACE,c.VERSION=namespace,VERSION
    c.audit_terminal=lambda root,arm,qid:audit(root,namespace,arm,qid)
    c.operation_lock=lambda root,arm,qid:maintenance_lock(root,namespace,arm,qid)
    return c

def validate_binding(binding):
    authorize(binding.get('namespace'),binding.get('arm'),binding.get('query_id'))
    require(binding.get('effective_config_sha256')=='4066e6ae83c5427924a3eb3577eb5c5621a4372b5be673852667b64fe8433c1a','Common configuration changed')
    require(binding.get('contract_sha256')=='e9ed046be4d2203d4a248caecc9f2f95060612ca49deb320fee05342d2ab2cff','no_bash contract changed')

def export_workspace(root,namespace,arm,qid,output):
    authorize(namespace,arm,qid)
    return core(namespace).export_workspace(root,arm,qid,output)

def verify_archive(archive,receipt,*,expected_receipt_sha256,expected_receipt_bytes):
    record=runner.load(receipt);validate_binding(record['binding'])
    result=core(record['binding']['namespace']).verify_archive(archive,receipt,
        expected_receipt_sha256=expected_receipt_sha256,expected_receipt_bytes=expected_receipt_bytes)
    validate_binding(result['binding']);return result

class CountedOS:
    def __init__(self):self.files_removed=self.directories_removed=0
    def __getattr__(self,name):return getattr(os,name)
    def unlink(self,*args,**kwargs):
        result=os.unlink(*args,**kwargs);self.files_removed+=1;return result
    def rmdir(self,*args,**kwargs):
        result=os.rmdir(*args,**kwargs);self.directories_removed+=1;return result

def delete_workspace(root,namespace,arm,qid,receipt,acknowledgement):
    authorize(namespace,arm,qid)
    record=runner.load(receipt);validate_binding(record['binding'])
    require(record['binding']['namespace']==namespace and record['binding']['query_id']==qid,'Wrong receipt namespace/qid')
    require(record.get('version')==VERSION and acknowledgement.get('version')==VERSION,'Wrong receipt/ACK version')
    require(not list(Path(receipt).parent.glob('maintenance_interrupted_*.json')),'Partial maintenance cannot be blindly retried')
    c=core(namespace);counter=CountedOS();c.os=counter
    try:return c.delete_workspace(root,arm,qid,receipt,acknowledgement)
    except BaseException as exc:
        if counter.files_removed or counter.directories_removed:
            result={'version':VERSION,'binding':record['binding'],'deleted':False,'partial_maintenance':True,
                'files_removed':counter.files_removed,'directories_removed':counter.directories_removed,
                'error_type':type(exc).__name__,'finished_unix':time.time(),'full_verified_local_backup_retained':True}
            result['interruption_sha256']=runner.digest(result)
            c.write_new(Path(receipt).parent/f'maintenance_interrupted_{time.time_ns()}.json',result)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('export','delete'):
        a=sub.add_parser(name);a.add_argument('--namespace',required=True,choices=NAMESPACES)
        a.add_argument('--arm',required=True,choices=[ARM]);a.add_argument('--qid',required=True,type=int)
        if name=='export':a.add_argument('--output-dir',required=True,type=Path)
        else:a.add_argument('--receipt',required=True,type=Path);a.add_argument('--acknowledgement-stdin',required=True,action='store_true')
    a=sub.add_parser('verify');a.add_argument('--archive',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True)
    a.add_argument('--expected-receipt-sha256',required=True);a.add_argument('--expected-receipt-bytes',required=True,type=int)
    a.add_argument('--acknowledgement-file',required=True,type=Path)
    a=p.parse_args()
    if a.command=='export':result=export_workspace(a.root,a.namespace,a.arm,a.qid,a.output_dir)
    elif a.command=='delete':result=delete_workspace(a.root,a.namespace,a.arm,a.qid,a.receipt,json.load(sys.stdin))
    else:
        result=verify_archive(a.archive,a.receipt,expected_receipt_sha256=a.expected_receipt_sha256,expected_receipt_bytes=a.expected_receipt_bytes)
        c=core(result['binding']['namespace']);c.write_new(a.acknowledgement_file,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
