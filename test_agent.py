import unittest
from unittest.mock import patch
from agent import get_model_cfg, run_once, prompt

class AgentTests(unittest.TestCase):
    def test_json_extraction(self):
        config = get_model_cfg({'choices':[{'message':{'content':'```json\n{"tile_m": 32, "tile_n": 16, "tile_k": 8}\n```'}}]})
        self.assertEqual(config.to_dict(), {'tile_m':32,'tile_n':16,'tile_k':8})

    def test_diagnostic_feedback_in_prompt(self):
        item = {'config':{'tile_m':8,'tile_n':8,'tile_k':8},
                'result': {'valid': True,'correct':True,'score':0.4,
                           'aggregate_bytes':100,'reason':'Increase reuse.'}}
        self.assertIn('Increase reuse.', prompt([item], 'diagnostic')[1]['content'])
        self.assertNotIn('Increase reuse.', prompt([item], 'score')[1]['content'])

    def test_llm_loop_with_mock_responses(self):
        from checker import Config
        responses = iter([Config(8,8,8), Config(32,32,16)])
        with patch('agent.query_model', side_effect=lambda *a, **kw: next(responses)):
            result = run_once('diagnostic', 2, 100, 'http://localhost:8000', 'fake-model', 10, 0.5)
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['best']['config'], Config(32,32,16).to_dict())

if __name__ == '__main__':
    unittest.main()
