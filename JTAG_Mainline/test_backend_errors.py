from pathlib import Path
import unittest
from unittest.mock import patch
import backend
ROOT=Path(__file__).parent
class FakeProcess:
    pid=0
    def __init__(self,text):self.stdout=text.encode('utf-8').splitlines(keepends=True)
    def wait(self):return 0
class InitializationErrors(unittest.TestCase):
    def invoke_text(self,text):
        folder=ROOT/'qa/init_error_fixtures';folder.mkdir(parents=True,exist_ok=True)
        script=folder/'probe.tcl'
        with patch.object(backend,'ensure_server'),patch.object(backend.subprocess,'Popen',return_value=FakeProcess(text)):
            return backend._invoke_unlocked(script)
    def test_actual_zero_chain_error_even_when_tool_returns_zero(self):
        text='no targets found with name APU. available targets:\n  1 whole scan chain (DR shift through all zeroes)\n'
        with self.assertRaisesRegex(RuntimeError,'扫描链返回全零'):self.invoke_text(text)
    def test_missing_target_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError,'没有找到需要的'):self.invoke_text('no targets found with name ARM\n')
    def test_success_marker_survives(self):
        self.assertIn('HD_INITIALIZED',self.invoke_text('PHASE: JTAG target verified; loading hardware\nHD_INITIALIZED\n'))
if __name__=='__main__':unittest.main()
