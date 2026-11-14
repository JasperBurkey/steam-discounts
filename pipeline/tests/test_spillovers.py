"""Spillovers (12) on synthetic data with a planted effect.

python -m unittest tests.test_spillovers -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SP = importlib.import_module("12_spillovers")


class Exposure(unittest.TestCase):
    def test_share_among_known_rivals_and_rank_band(self):
        status = pd.DataFrame({"S1": [1, 0, np.nan, 1], "S2": [0, 0, 1, 1]}, index=[10, 11, 12, 13])
        rivals = pd.DataFrame({"app_id": [1, 1, 1, 1], "rival": [10, 11, 12, 13], "rank": [1, 2, 3, 4]})
        e = SP.exposure(rivals, status).set_index("sale")["x"]
        self.assertAlmostEqual(e["S1"], 2 / 3)  # unknown rival 12 is left out
        self.assertAlmostEqual(e["S2"], 2 / 4)
        e2 = SP.exposure(rivals, status, 1, 2).set_index("sale")["x"]
        self.assertAlmostEqual(e2["S1"], 1 / 2)


class Recovery(unittest.TestCase):
    def test_plr_multi_recovers_planted_spillover(self):
        rng = np.random.default_rng(0)
        n = 3000
        df = pd.DataFrame({"app_id": np.arange(n) // 3, "sale": rng.choice(["A", "B"], n)})
        conf = rng.normal(size=n)
        df["x_real"] = np.clip(0.8 + 0.1 * conf + rng.normal(0, 0.1, n), 0, 1)
        df["x_fake"] = np.clip(0.8 + 0.1 * rng.normal(size=n), 0, 1)
        df["c"] = conf
        df["y"] = -0.5 * df["x_real"] + 0.3 * conf + rng.normal(0, 0.1, n)
        orig = SP.M.features
        SP.M.features = lambda d, extra=(), names=False: d[list(extra)].to_numpy(float)
        try:
            res = SP.plr_multi(df, "y", ["x_real", "x_fake"], extra=["c"], repeats=1)
        finally:
            SP.M.features = orig
        self.assertAlmostEqual(res["x_real"][0], -0.5, delta=0.08)
        self.assertAlmostEqual(res["x_fake"][0], 0.0, delta=0.08)


if __name__ == "__main__":
    unittest.main()
