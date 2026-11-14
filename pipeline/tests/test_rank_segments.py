"""Rank validation (10) and segmentation (11) on synthetic data with known answers.

python -m unittest tests.test_rank_segments -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RV = importlib.import_module("10_rank_validation")
SG = importlib.import_module("11_segments")


def utc(s):
    return pd.Timestamp(s, tz="UTC")


class SaleWeeks(unittest.TestCase):
    def setUp(self):
        self.weeks = list(
            pd.date_range("2025-08-05", "2026-01-20", freq="7D", tz="UTC")
        )
        self.cal = pd.DataFrame(
            {
                "sale": ["Autumn Sale 2025", "Winter Sale 2025"],
                "start": ["2025-09-29", "2025-12-18"],
                "end": ["2025-10-06", "2026-01-05"],
            }
        )
        self.cal["s"] = pd.to_datetime(self.cal["start"]).dt.tz_localize("UTC") + RV.H
        self.cal["e"] = pd.to_datetime(self.cal["end"]).dt.tz_localize("UTC") + RV.H

    def test_monday_start_and_long_winter_sale(self):
        sets = RV.week_sets(self.cal, self.weeks)
        # Autumn 2025 starts on a Monday: only the week from Tuesday Sep 30 is mostly inside
        self.assertEqual(sets["Autumn Sale 2025"]["sale"], [utc("2025-09-30")])
        # an 18-day Winter Sale starting Thursday Dec 18 covers three chart weeks
        self.assertEqual(
            sets["Winter Sale 2025"]["sale"],
            [utc("2025-12-16"), utc("2025-12-23"), utc("2025-12-30")],
        )
        # baseline: three weeks ending at least 7 days before the start, none in a sale
        base = sets["Winter Sale 2025"]["base"]
        self.assertEqual(len(base), 3)
        self.assertTrue(all(w + RV.WEEK <= self.cal["s"][1] - RV.WEEK for w in base))


class WeightingTest(unittest.TestCase):
    def test_recovers_planted_coefficients(self):
        rng = np.random.default_rng(0)
        n = 1500
        y = rng.normal(0.3, 0.4, n)
        ratio = rng.uniform(0.2, 1.0, n)
        sale = rng.choice(["A", "B", "C"], n)
        noise = rng.normal(0, 0.2, n)
        b_y, b_p = 0.6, 0.15  # price matters less than reviews
        df = pd.DataFrame(
            {
                "y_during": y,
                "price_ratio_sale": ratio,
                "sale": sale,
                "app_id": np.arange(n) // 3,
                "discounted": ratio < 1,
                "dlogrank": -(b_y * y + b_p * np.log(ratio) + noise),
            }
        )
        out = RV.weighting_test(df)
        self.assertAlmostEqual(out["b_reviews"], b_y, delta=0.05)
        self.assertAlmostEqual(out["b_log_price"], b_p, delta=0.05)
        self.assertAlmostEqual(out["diff_price_minus_reviews"], b_p - b_y, delta=0.06)


class Segments(unittest.TestCase):
    def test_factors_recover_planted_audiences(self):
        """Three audiences, each reviewing its own block of 20 games (plus some noise)."""
        rng = np.random.default_rng(1)
        rows = []
        for u in range(6000):
            block = u % 3
            n = rng.integers(2, 6)
            own = rng.choice(np.arange(20) + 20 * block, n, replace=False)
            stray = rng.choice(60, 1) if rng.random() < 0.15 else []
            for g in list(own) + list(stray):
                rows.append((u, int(g)))
        r = pd.DataFrame(rows, columns=["reviewer", "app_id"]).assign(
            received_for_free=False, steam_purchase=True
        )
        r = SG.clean(r)
        x, games = SG.matrix(r)
        e = SG.embed(SG.sppmi(SG.cooc(x)), 8)
        same = []
        for i in range(len(games)):
            nb = np.argsort(-(e @ e[i]))[1:11]
            same.append(np.mean(games[nb] // 20 == games[i] // 20))
        self.assertGreater(np.mean(same), 0.9)

    def test_leave_one_out_beats_popularity(self):
        rng = np.random.default_rng(2)
        rows = [
            (u, int(g))
            for u in range(4000)
            for g in rng.choice(np.arange(15) + 15 * (u % 4), 4, replace=False)
        ]
        r = pd.DataFrame(rows, columns=["reviewer", "app_id"]).assign(
            received_for_free=False, steam_purchase=True
        )
        r = SG.clean(r)
        train, test = SG.split_reviewers(r)
        x, games = SG.matrix(train)
        c = SG.cooc(x)
        e = SG.embed(SG.sppmi(c), 8)
        hit_f = SG.leave_one_out(e @ e.T, games, test, n_users=500)[0]
        hit_p = SG.leave_one_out(
            np.tile(c.sum(0), (len(games), 1)), games, test, n_users=500
        )[0]
        self.assertGreater(hit_f, hit_p + 0.2)


if __name__ == "__main__":
    unittest.main()
