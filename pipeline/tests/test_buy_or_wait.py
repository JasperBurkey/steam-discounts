"""Buy-or-wait model (15): solver, chain and review-lag mechanics.

python -m unittest tests.test_buy_or_wait -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
B = importlib.import_module("15_buy_or_wait")


def toy_chain(p_sale=0.8):
    """Discount only in sale weeks, with probability p_sale (deep), else full price."""
    g = np.zeros((1, B.W * 30), int)
    w = np.array([(-t) % B.W for t in range(B.W * 30)])
    rng = np.random.default_rng(0)
    g[0, w == 0] = np.where(rng.random((w == 0).sum()) < p_sale, 3, 0)
    return B.transition(g, w)


class Model(unittest.TestCase):
    def setUp(self):
        self.P = toy_chain()
        self.price = np.tile(np.array([1.0, 0.75, 0.5, 0.25]), B.W)

    def test_vectorized_solver_matches_discretedp(self):
        v = B.valuations(0.0, 0.9)
        V, buy = B.solve(self.P, self.price, v, 0.99, 0.05)
        for i in range(0, B.N_V, 7):
            Vq, bq = B.solve_discretedp(self.P, self.price, v[i], 0.99, 0.05)
            self.assertTrue(np.array_equal(buy[i], bq))
            self.assertTrue(np.allclose(V[i], np.maximum(Vq, 0), atol=1e-6))

    def test_buyers_wait_just_before_a_likely_sale(self):
        v = np.array([1.1])  # values the game a little above full price
        _, buy = B.solve(self.P, self.price, v, 0.995, 0.02)
        week_before = 1 * B.D + 0  # w = 1, full price
        self.assertFalse(buy[0, week_before])  # waits: a deep sale is a week away
        self.assertTrue(buy[0, 0 * B.D + 3])  # buys in the sale

    def test_review_lag_preserves_totals_and_delays(self):
        q = np.zeros((1, 60))
        q[0, 5] = 1.0
        r = B.lagged(q, 0.7)
        self.assertAlmostEqual(r.sum(), 1.0, delta=1e-6)
        self.assertEqual(r[0, :5].sum(), 0)
        self.assertGreater(r[0, 6:].sum(), r[0, 5])

    def test_no_waiting_means_no_pre_sale_dip(self):
        arms, T0 = B.simulate(
            self.P, self.price, (0.0, 0.9, 0.999, 0.0, 0.3), 0.99, n=50
        )
        q0, q3 = arms[0][0], arms[3][0]
        self.assertTrue(np.allclose(q0[:, T0 - 1], q3[:, T0 - 1]))


if __name__ == "__main__":
    unittest.main()
