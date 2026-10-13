import unittest
from checker import Config
from agent_v3 import BASE_SYSTEM, messages, run, summarize

class V3Tests(unittest.TestCase):
    def test_equal_initial_constraints(self):
        a = messages([], 'score')
        b = messages([], 'diagnostic')
        self.assertEqual(a, b)
        self.assertIn('6144', BASE_SYSTEM)

    def test_duplicate_is_logged_without_retry(self):
        calls=[]
        def fake(*args):
            calls.append(1)
            return Config(32, 32, 32)
        trial = run('diagnostic', 100, 2, '', '', 1, 0, fake)
        self.assertEqual(len(calls), 2)
        self.assertEqual([x['status'] for x in trial['attempts']], ['VALID', 'DUPLICATE'])
        self.assertIn('DUPLICATE', trial['attempts'][1]['checker_detail'])

    def test_invalid_candidate_visible_to_both_modes(self):
        def fake(*args): return Config(64, 64, 64)
        a=run('score', 100, 1, '', '', 1, 0, fake)
        b=run('diagnostic', 100, 1, '', '', 1, 0, fake)
        self.assertEqual(a['attempts'][0]['status'], 'INVALID')
        self.assertEqual(a['attempts'][0]['feedback'], 'Invalid proposal (score 0).')
        self.assertIn('MEMORY_CAPACITY', b['attempts'][0]['feedback'])

    def test_random_budget(self):
        r=run('random',100,3,'','',1,0)
        self.assertEqual(len(r['attempts']),3)
        self.assertEqual(r['valid_proposals'],3)
        self.assertEqual(summarize([r],['random'])['random']['total_evaluation_slots'],3)

if __name__ == '__main__': unittest.main()
