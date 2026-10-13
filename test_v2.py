import unittest
from checker import Config
from checker_v2 import evaluate_v2,validate_v2,WORKLOADS_V2
from agent_v2 import VALID

class V2Tests(unittest.TestCase):
    def test_capacity_rejection(self):
        self.assertIn('MEMORY_CAPACITY',validate_v2(Config(64,64,64)))
    def test_irregular_shapes(self):
        self.assertTrue(any(w.name=='ragged' for w in WORKLOADS_V2))
    def test_valid_correct(self):
        self.assertTrue(evaluate_v2(Config(32,32,32))['correct'])
    def test_k_affects_objective(self):
        a=evaluate_v2(Config(32,32,16),verify=False)['aggregate_cost']
        b=evaluate_v2(Config(32,32,32),verify=False)['aggregate_cost']
        self.assertNotEqual(a,b)
    def test_valid_list(self):
        self.assertGreater(len(VALID),0)
        self.assertEqual(len(VALID),len(set(VALID)))

if __name__=='__main__':unittest.main()
