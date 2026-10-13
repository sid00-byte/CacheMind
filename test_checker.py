import unittest
from checker import Config, TILES, evaluate, traffic_bytes, validate, blocked_matmul
from workloads import WORKLOADS, matrices
import numpy as np

class CheckerTests(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(validate(Config(16, 32, 16)))
        self.assertIn('invalid', validate(Config(13, 32, 16)))
        self.assertIn('limit', validate(Config(64, 64, 64)))

    def test_every_legal_configuration(self):
        for m in TILES:
            for n in TILES:
                for k in TILES:
                    config = Config(m, n, k)
                    if validate(config) is None:
                        with self.subTest(config=config):
                            self.assertTrue(evaluate(config)['correct'])

    def test_boundary_handling(self):
        w = WORKLOADS[-1]
        a,b = matrices(w)
        self.assertTrue(np.allclose(blocked_matmul(a,b,Config(16,32,8)), a@b,
                                    rtol=1e-3, atol=1e-3))

    def test_traffic_proxy(self):
        self.assertLess(traffic_bytes(64,64,64,Config(32,32,16)),
                        traffic_bytes(64,64,64,Config(8,8,16)))

if __name__ == '__main__':
    unittest.main()
