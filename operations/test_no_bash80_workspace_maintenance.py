from contextlib import nullcontext,redirect_stdout
from io import StringIO
from pathlib import Path
import copy,json,os,shutil,tempfile,threading,time,unittest
from unittest.mock import patch
from operations import no_bash80_workspace_maintenance as m

class Maintenance80Tests(unittest.TestCase):
 def test_namespace_qid_and_previously_partial153(self):
  for ns,qid in ((m.old_job.DIRECTORY,122),(m.job.DIRECTORY,278)):m.authorize(ns,'no_bash',qid)
  for ns,arm,qid in ((m.old_job.DIRECTORY,'no_bash',153),(m.job.DIRECTORY,'no_bash',70),(m.old_job.DIRECTORY,'no_bash',278),(m.job.DIRECTORY,'search_paging',278),(m.job.DIRECTORY,'no_bash',True)):
   with self.subTest(ns=ns,arm=arm,qid=qid),self.assertRaises(RuntimeError):m.authorize(ns,arm,qid)

 def lifecycle(self,ns,qid):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);base=Path(t.name);root=base/'remote';root.mkdir()
  c=m.core(ns);work,runtime=c.targets(root,'no_bash',qid).values();(work/'nested').mkdir(parents=True);runtime.mkdir(parents=True)
  corpus=root/'shared_corpus';corpus.write_bytes(b'opaque unchanged source'*100);os.link(corpus,work/'nested'/'link')
  (runtime/'body').write_bytes(b'opaque runtime');other=work.parent/'qid_999999';other.mkdir();(other/'preserve').write_bytes(b'other question')
  terminal=root/'runs'/ns/'no_bash'/'_per_query'/f'qid_{qid}.json';terminal.parent.mkdir();terminal.write_bytes(b'opaque terminal')
  binding={'source_root':str(root.resolve()),'namespace':ns,'arm':'no_bash','query_id':qid,
    'terminal_sha256':c.file_sha(terminal),'effective_config_sha256':'4066e6ae83c5427924a3eb3577eb5c5621a4372b5be673852667b64fe8433c1a',
    'contract_sha256':'e9ed046be4d2203d4a248caecc9f2f95060612ca49deb320fee05342d2ab2cff'}
  def audit(*args):
   if c.file_sha(terminal)!=binding['terminal_sha256']:raise RuntimeError('terminal changed')
   return binding
  with patch.object(m,'audit',side_effect=audit),patch.object(m,'maintenance_lock',side_effect=lambda *a:nullcontext()):
   output=root/'reports'/ns/'workspace_backups'/'fresh';r=m.export_workspace(root,ns,'no_bash',qid,output)
   local=base/'local';local.mkdir();archive=local/'workspace.tar.gz';receipt=local/'receipt.json'
   shutil.copyfile(output/'workspace.tar.gz',archive);shutil.copyfile(output/'receipt.json',receipt)
   ack=m.verify_archive(archive,receipt,expected_receipt_sha256=r['receipt_file_sha256'],expected_receipt_bytes=r['receipt_bytes'])
   wrong=copy.deepcopy(ack);wrong['archive']['sha256']='0'*64;wrong['acknowledgement_sha256']=c.digest({k:v for k,v in wrong.items() if k!='acknowledgement_sha256'})
   with self.assertRaises(RuntimeError):m.delete_workspace(root,ns,'no_bash',qid,output/'receipt.json',wrong)
   self.assertTrue(work.exists())
   deleted=m.delete_workspace(root,ns,'no_bash',qid,output/'receipt.json',ack)
   self.assertTrue(deleted['deleted']);self.assertFalse(work.exists());self.assertFalse(runtime.exists())
   self.assertEqual(corpus.read_bytes(),b'opaque unchanged source'*100);self.assertTrue((other/'preserve').is_file())
   self.assertEqual(c.file_sha(terminal),binding['terminal_sha256'])
   shutil.rmtree(root)
   self.assertTrue(m.verify_archive(archive,receipt,expected_receipt_sha256=r['receipt_file_sha256'],expected_receipt_bytes=r['receipt_bytes'])['verified'])

 def test_old20_full_backup_local_verify_exact_delete_and_server_independent_restore(self):self.lifecycle(m.old_job.DIRECTORY,122)
 def test_new80_full_backup_local_verify_exact_delete_and_server_independent_restore(self):self.lifecycle(m.job.DIRECTORY,278)

 def test_core_not_globally_modified(self):
  from operations import terminal_workspace_maintenance as frozen
  before=(frozen.NAMESPACE,frozen.VERSION,frozen.os)
  a=m.core(m.job.DIRECTORY);b=m.core(m.old_job.DIRECTORY);a.os=m.CountedOS()
  self.assertEqual(before,(frozen.NAMESPACE,frozen.VERSION,frozen.os));self.assertIs(b.os,os)

 def test_phase_and_pending_gate_fail_closed(self):
  with tempfile.TemporaryDirectory() as root:
   suite=m.job.NoBash80Suite(root);suite.reports.mkdir(parents=True)
   identity={'pid':123};binding={'worker_identity':identity,'manifest_sha256':'manifest','query_id':278}
   state={'phase':'running_no_bash80','checked_unix':time.time(),'worker_identity':identity,'manifest_sha256':'manifest','active_query_ids':[280],'pending_query_ids':[288]}
   def save():m.runner.atomic_json(suite.reports/'phase_state.json',state)
   status={'arms':{'no_bash':{'terminal':[278],'pending':[288],'active':[280],'partial_or_active':[280]}}}
   with patch.object(m.job,'process_matches',return_value=True),patch.object(m.job.NoBash80Suite,'evidence_present',return_value=False),patch.object(m.job.NoBash80Suite,'status',return_value=status):
    save();m.live_admission(root,binding)
    for phase in ('prepare','live_preflight','complete','interrupted'):
     state['phase']=phase;save()
     with self.assertRaises(RuntimeError):m.live_admission(root,binding)
    state['phase']='running_no_bash80';state['pending_query_ids']=[];save()
    with self.assertRaises(RuntimeError):m.live_admission(root,binding)

 def test_maintenance_barrier_blocks_dispatch_until_released(self):
  with tempfile.TemporaryDirectory() as root:
   suite=m.job.NoBash80Suite(root);query=m.job.selected_queries()[0];done=threading.Event()
   manifest={'manifest_sha256':'m','config_sha256':'c','contracts':{'no_bash':{'contract_sha256':'t'}}}
   with suite.resource_maintenance_lock():
    thread=threading.Thread(target=lambda:(suite._dispatch(manifest,query,{}),done.set()));thread.start()
    self.assertFalse(done.wait(.05));self.assertFalse(suite.dispatch_path(query['query_id']).exists())
   thread.join(timeout=2);self.assertTrue(done.is_set());self.assertTrue(suite.dispatch_path(query['query_id']).exists())


 def test_old_completed_pool_live80_barrier_never_touches_shared_solver_lock(self):
  import fcntl
  with tempfile.TemporaryDirectory() as root:
   new=m.job.NoBash80Suite(root);new.reports.mkdir(parents=True)
   old=m.old_job.NoBash20Suite(root);old.reports.mkdir(parents=True)
   m.runner.atomic_json(old.reports/'phase_state.json',{'phase':'complete'})
   (old.reports/'worker.lock').touch()
   identity={'pid':123}
   state={'phase':'running_no_bash80','checked_unix':time.time(),'worker_identity':identity,
     'manifest_sha256':'manifest','active_query_ids':[278],'pending_query_ids':[280]}
   m.runner.atomic_json(new.reports/'phase_state.json',state)
   status={'arms':{'no_bash':{'terminal':[],'active':[278],'partial_or_active':[278],'pending':[280]}}}
   shared=Path(root)/'reports/strict_ablations_v1/worker.lock';shared.parent.mkdir();shared.touch()
   with shared.open('rb') as handle,patch.object(m.job,'process_matches',return_value=True),patch.object(m.job.NoBash80Suite,'evidence_present',return_value=False),patch.object(m.job.NoBash80Suite,'status',return_value=status):
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with m.maintenance_lock(root,m.old_job.DIRECTORY,'no_bash',122):
     with (new.reports/'maintenance_resource.lock').open('rb') as check,self.assertRaises(BlockingIOError):
      fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state['pending_query_ids']=[];m.runner.atomic_json(new.reports/'phase_state.json',state)
    with self.assertRaisesRegex(RuntimeError,'Tail cleanup forbidden'),m.maintenance_lock(root,m.old_job.DIRECTORY,'no_bash',122):pass

if __name__=='__main__':unittest.main()
