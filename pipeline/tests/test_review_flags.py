"""Review-flag checks (16): windows, outcomes and thinning.

python -m unittest tests.test_review_flags -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
F = importlib.import_module("16_review_flags")


class Windows(unittest.TestCase):
    def test_counts_match_window_bounds(self):
        days = np.arange(0, 400)  # one review a day
        s = F.window_sums(days, d0=100, n=7)
        self.assertEqual(s["C"], 14)  # days -21..-8
        self.assertEqual(s["E"], 7)  # the sale
        self.assertEqual(s["F"], 14)

    def test_flat_series_has_zero_effects(self):
        days = np.repeat(np.arange(0, 400), 3)
        o = F.outcomes(F.window_sums(days, 100, 7), 7)
        for v in o.values():
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_doubling_during_the_sale(self):
        days = np.concatenate(
            [np.repeat(np.arange(0, 400), 10), np.repeat(np.arange(100, 107), 10)]
        )
        o = F.outcomes(F.window_sums(np.sort(days), 100, 7), 7)
        self.assertAlmostEqual(o["during"], np.log1p(280) - np.log1p(140), places=9)
        self.assertAlmostEqual(o["after"], 0.0, places=9)


class Thinning(unittest.TestCase):
    def test_thinned_benchmark_matches_target_share(self):
        rng = np.random.default_rng(0)
        n = 200_000
        rv = pd.DataFrame(
            {
                "app_id": rng.integers(0, 4, n),
                "day": rng.integers(0, 500, n),
                "steam_purchase": rng.random(n) < 0.8,
                "received_for_free": rng.random(n) < 0.02,
                "playtime_min": rng.integers(0, 2000, n),
            }
        )
        days, shares = F.series_days(rv, np.random.default_rng(1))
        for g in range(4):
            target = len(days["placebo"][g])
            thin = len(days["paid_thin_placebo"][g])
            self.assertAlmostEqual(thin / target, 1.0, delta=0.05)
            self.assertTrue(np.all(np.diff(days["paid"][g]) >= 0))


if __name__ == "__main__":
    unittest.main()
