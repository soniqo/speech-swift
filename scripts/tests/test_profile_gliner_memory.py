import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

script=Path(__file__).resolve().parents[1]/'profile_gliner_memory.py'
spec=importlib.util.spec_from_file_location('profile_gliner_memory',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DATA=script.parent/'tests/fixtures/gliner'

class ReferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.result=json.loads((DATA/'swift-output-fp32.json').read_text())
        self.reference=json.loads((DATA/'reference-fixtures.json').read_text())

    def test_saved_fp32_output_passes_reference(self):
        self.assertTrue(module.check_reference(self.result,self.reference,.001)['passed'])

    def test_changed_label_and_offset_fail(self):
        changed=copy.deepcopy(self.result)
        max(changed['rows'][0]['choices'],key=lambda x:x['probability'])['label']='invalid'
        self.assertFalse(module.check_reference(changed,self.reference,.001)['passed'])
        changed=copy.deepcopy(self.result)
        changed['rows'][16]['entities']['person'][0]['start']+=1
        self.assertFalse(module.check_reference(changed,self.reference,.001)['passed'])

    def test_confidence_drift_and_missing_case_fail(self):
        changed=copy.deepcopy(self.result)
        max(changed['rows'][0]['choices'],key=lambda x:x['probability'])['probability']+=.1
        self.assertFalse(module.check_reference(changed,self.reference,.005)['passed'])
        changed['rows'].pop()
        with self.assertRaises(ValueError):module.check_reference(changed,self.reference,.005)

class ActiveJobDetectionTests(unittest.TestCase):
    def test_detects_model_gates_probes_and_compilers(self):
        table='''  PID COMM
  100 /usr/bin/voxcpm2-gate
  101 /x/stenograf-llm-probe
  102 /usr/bin/swift-frontend
  103 /x/gliner-bench
  104 /usr/bin/zsh
  105 /Applications/Code Helper (Plugin)
'''
        with mock.patch.object(module.subprocess,'check_output',return_value=table):
            self.assertEqual(module.active_models(),[(100,'voxcpm2-gate'),(101,'stenograf-llm-probe'),(102,'swift-frontend'),(103,'gliner-bench')])
            self.assertEqual([pid for pid,_ in module.active_models(ignore_pid=103)],[100,101,102])

if __name__ == '__main__':unittest.main()
