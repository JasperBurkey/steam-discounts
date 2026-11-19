"""Smooth buy-or-wait model (31): logit solver limits, flows, welfare accounting.

python -m unittest tests.test_estimation -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
E = importlib.import_module("31_estimate_buy_or_wait")
B = E.B


def toy():
    P, d = B.schedule_chain([3, 0])
    pg = np.array([1.0, 0.8, 0.55, 0.25])
    return P, pg[d], pg


class Solver(unittest.TestCase):
    def test_small_lambda_matches_hard_choice(self):
        P, p, _ = toy()
        v = B.valuations(0.0, 0.8)
        _, buy = B.solve(P, p, v, 0.99, 0.05)
        _, pb, _ = E.solve_logit(P, p, v, 0.99, 0.05, 1e-4)
        agree = ((pb > 0.5) == buy).mean()
        self.assertGreater(agree, 0.99)

    def test_probabilities_and_monotone_in_valuation(self):
        P, p, _ = toy()
        v = B.valuations(0.0, 0.8)
        _, pb, _ = E.solve_logit(P, p, v, 0.99, 0.05, 0.05)
        self.assertTrue(((pb >= 0) & (pb <= 1)).all())
        self.assertTrue((np.diff(pb, axis=0) >= -1e-9).all())  # richer buyers buy more

    def test_flows_conserve_mass(self):
        mass = np.zeros((1, 4))
        pb = np.full((1, 4), 0.3)
        q, m = E.step_p(mass, pb, 0.0, arrivals=1.0)
        self.assertAlmostEqual(q[0] + m.sum(), 1.0)


class Welfare(unittest.TestCase):
    def test_never_discount_baseline_is_zero(self):
        _, _, pg = toy()
        out = E.counterfactuals(pg, (0.0, 0.8, 0.05, 0.3, 0.5), 0.99, 0.05, 1.0)
        for v in out["never discount"].values():
            self.assertAlmostEqual(v, 0.0, places=10)
        r = out["61%+ every sale"]
        self.assertAlmostEqual(
            r["publisher (70%)"] + r["Valve (30%)"], r["revenue"], 10
        )


if __name__ == "__main__":
    unittest.main()
