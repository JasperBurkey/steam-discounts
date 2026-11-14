"""The ratio estimator recovers a multiplicative effect when review counts are SMALL,
where the log(1 + x) outcome is compressed toward zero (the problem found in the first
full-sample run). Synthetic game-events, true effect x1.8 (log 0.588), with confounding.

  python -m unittest tests.test_ratio_lowcount -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LEARNER", "boost")
os.environ.setdefault("CF_REPEATS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dml_core as M  # noqa: E402


def make(n_games=1500, n_events=6, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gid in range(n_games):
        lam = scale * np.exp(
            rng.normal(np.log(0.25), 0.9)
        )  # reviews per day, mostly < 1
        age = rng.integers(100, 3000)
        for k in range(n_events):
            p = 1 / (
                1 + np.exp(-(1.5 + 0.6 * np.log(lam) + 0.0003 * age))
            )  # busier and older discount more
            dsc = rng.random() < p
            base = rng.poisson(14 * lam)
            during = rng.poisson(14 * lam * 1.15 * (1.8 if dsc else 1.0))
            rows.append(
                {
                    "app_id": gid,
                    "sale": f"S{k}",
                    "discounted": dsc,
                    "age_days": age,
                    "pre_rate_28": rng.poisson(28 * lam) / 28,
                    "pre_rate_60": rng.poisson(60 * lam) / 60,
                    "regular_usd": 14.99,
                    "lvl_base": base,
                    "lvl_during": during,
                    "y_during": np.log1p(during) - np.log1p(base),
                }
            )
    return pd.DataFrame(rows)


class RatioLowCount(unittest.TestCase):
    def test_ratio_recovers_truth_where_log_is_compressed(self):
        df = make()
        truth = np.log(1.8)
        ratio, se = M.ratio_att(df, "lvl_during")
        logv = M.aipw(df, ["y_during"])[0]["y_during"]["ATT"][0]
        print(
            f"\\ntruth {truth:.3f} | ratio ATT {ratio:.3f} (se {se:.3f}) | log(1+x) ATT {logv:.3f}"
            f" | median reviews per 14 days {df['lvl_base'].median():.0f}"
        )
        self.assertLess(abs(ratio - truth), max(3 * se, 0.06))
        self.assertLess(logv, truth - 0.05)  # the log version is compressed toward zero

    def test_revenue_proxy_case(self):
        """Revenue proxy (discounted games pay 35% of the price): compression is larger."""
        df = make()
        df["lvl_r_during"] = np.where(
            df["discounted"], df["lvl_during"] * 0.35, df["lvl_during"]
        )
        df["r_during"] = np.log1p(df["lvl_r_during"]) - np.log1p(df["lvl_base"])
        truth = np.log(1.8 * 0.35)
        ratio, se = M.ratio_att(df, "lvl_r_during")
        logv = M.aipw(df, ["r_during"])[0]["r_during"]["ATT"][0]
        print(
            f"\nrevenue: truth {truth:.3f} | ratio {ratio:.3f} (se {se:.3f}) | log(1+x) {logv:.3f}"
        )
        self.assertLess(abs(ratio - truth), max(3 * se, 0.08))
        self.assertGreater(logv, truth + 0.08)  # log version understates the loss


if __name__ == "__main__":
    unittest.main()
