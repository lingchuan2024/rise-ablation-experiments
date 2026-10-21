"""Docker backend boundaries and opt-in real containment test."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rise import tools
from rise.protocol import for_rise

class DockerSandboxTests(unittest.TestCase):
    def test_command_has_only_readonly_workspace_and_no_network(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'RISE_BASH_SANDBOX':'docker', 'RISE_DOCKER_IMAGE':'sha256:test'}), patch.object(tools.sys,'platform','linux'):
            cmd=tools._sandboxed_bash_command(Path(d), 'ls')
            for flag in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges:true','--pull=never']:
                self.assertIn(flag,cmd)
            self.assertEqual(cmd.count('--volume'),1)
            self.assertEqual(cmd[cmd.index('--volume')+1],f'{Path(d).resolve()}:{Path(d).resolve()}:ro')
            self.assertNotIn('/var/run/docker.sock', ' '.join(cmd))
            cfg=for_rise(agent_model='Qwen3.5-35B-A3B',enable_sandbox=True)
            self.assertEqual(cfg.extras['bash_sandbox_image'],'sha256:test')

    @unittest.skipUnless(os.environ.get('RISE_TEST_DOCKER')=='1','requires explicit Docker integration opt-in')
    def test_real_readonly_visibility_network_and_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'RISE_BASH_SANDBOX':'docker','RISE_SANDBOX_SECRET':'do-not-expose'}):
            root=Path(d); work=root/'work'; work.mkdir()
            (work/'visible.txt').write_text('VISIBLE')
            (root/'hidden.txt').write_text('HIDDEN_SECRET')
            bash=tools.make_bash_tool(work,enable_sandbox=True)
            self.assertIn('VISIBLE',bash({'command':'cat visible.txt'}))
            self.assertNotIn('HIDDEN_SECRET',bash({'command':f'cat {root}/hidden.txt'}))
            bash({'command':'echo changed > visible.txt'})
            self.assertEqual((work/'visible.txt').read_text(),'VISIBLE')
            self.assertIn('SCRATCH_OK',bash({'command':'echo SCRATCH_OK > /tmp/a; cat /tmp/a'}))
            self.assertEqual(bash({'command':'printf "%s" "${RISE_SANDBOX_SECRET-unset}"'}),'unset')
            self.assertIn('NETWORK_ISOLATED',bash({'command':'test ! -e /sys/class/net/eth0 && echo NETWORK_ISOLATED'}))
            self.assertIn('[exit=',bash({'command':'echo forbidden > /etc/rise-write-test'}))
            self.assertIn('NO_SOCKET',bash({'command':'test ! -e /var/run/docker.sock && echo NO_SOCKET'}))
            self.assertIn('timed out',bash({'command':'sleep 20','timeout':1}))
            leftover=subprocess.check_output(['/usr/bin/docker','ps','-aq','--filter','label=rise-sandbox=true'],text=True)
            self.assertEqual(leftover.strip(),'')
