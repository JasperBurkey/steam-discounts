"""Policy learning (13) on synthetic scores with a known best rule.

python -m unittest tests.test_policy -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PL = importlib.import_module("13_policy")


def planted(n=4000, seed=0):
    """Best action: 'none' (0) for old games; for young games, shallow (1) if cheap and
    deep (3) if expensive."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, len(PL.POLICY_FEATURES)))
    young, pricey = x[:, 1] <= 0.5, x[:, 2] > 0.6
    best = np.where(~young, 0, np.where(pricey, 3, 1))
    gamma = rng.normal(0, 1, (n, 4))  # noisy scores, as doubly robust scores are
    gamma[np.arange(n), best] += 1.0
    return x, gamma, best


class Trees(unittest.TestCase):
    def test_two_level_tree_recovers_planted_rule(self):
        x, gamma, best = planted()
        tree = PL.fit_tree(x, gamma, 2)
        # age-then-price and price-then-age give the same rule, so check the rule itself
        self.assertIn(tree["feature"], (1, 2))
        self.assertGreater((PL.predict(tree, x) == best).mean(), 0.9)

    def test_one_split_and_leaf_size(self):
        x, gamma, _ = planted()
        tree = PL.fit_tree(x, gamma, 1)
        a = PL.predict(tree, x)
        self.assertEqual(tree["feature"], 1)
        for k in np.unique(a):
            self.assertGreaterEqual((a == k).mean(), PL.MIN_LEAF)

    def test_no_signal_gives_single_action(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, (3000, len(PL.POLICY_FEATURES)))
        gamma = rng.normal(0, 1, (3000, 4))
        gamma[:, 2] += 0.5  # one action best everywhere
        self.assertEqual(set(PL.predict(PL.fit_tree(x, gamma, 2), x)), {2})

    def test_crossfit_beats_status_quo_out_of_sample(self):
        x, gamma, best = planted(seed=5)
        groups = np.arange(len(x)) // 3
        sq = np.zeros(len(x), int)
        out = PL.crossfit_policies(x, gamma, groups, sq)
        n = np.arange(len(x))
        gain = (gamma[n, out["tree, 2 levels"]] - gamma[n, sq]).mean()
        self.assertGreater(gain, 0.5)
        self.assertGreater(
            gamma[n, out["tree, 2 levels"]].mean(),
            gamma[n, out["best fixed depth"]].mean(),
        )


if __name__ == "__main__":
    unittest.main()
