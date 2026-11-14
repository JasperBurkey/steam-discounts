"""End-to-end test of the study pipeline on fake data with a known effect (takes ~1 minute).

  python -m unittest tests.test_study -v
Overwrites data/ and output/; run it before downloading real data or in a scratch copy.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Study(unittest.TestCase):
    def test_recovers_known_effect(self):
        env = {
            **os.environ,
            "COLLECTOR_DATA": str(ROOT / "data" / "fake_collector"),
            "CF_REPEATS": "1",
        }
        # STUDY_TEST_REUSE=1 checks outputs already produced (to run the slow steps separately)
        scripts = (
            []
            if os.environ.get("STUDY_TEST_REUSE")
            else [
                "tests/make_fake_study.py",
                "04b_build_events.py",
                "05b_did_event_study.py",
                "07_dml.py",
            ]
        )
        for script in scripts:
            subprocess.run(
                [sys.executable, script],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
            )
        res = pd.read_csv(ROOT / "output" / "dml_results.csv")

        def get(spec_start, estimand):
            r = res[
                res["specification"].str.startswith(spec_start)
                & (res["estimand"] == estimand)
            ]
            return r.iloc[0]

        # reviews during the sale: truth x1.8 for discounted games (log 0.588, ~ +80%)
        for est in ["ATT", "ATE", "PLR", "ATT, residual balancing (L2)"]:
            self.assertAlmostEqual(
                get("Reviews, during", est)["pct_change"], 0.80, delta=0.12, msg=est
            )
        # revenue proxy during the sale: the exact value the proxy SHOULD show for discounted games,
        # from each game's true review rate and its actual average price ratio during the sale:
        #   log(1 + 14 x rate x 1.15 x 1.8 x ratio) - log(1 + 14 x rate x 1.15)
        ev_all = pd.read_parquet(ROOT / "data" / "events.parquet")
        lam = pd.read_csv(ROOT / "data" / "fake_truth.csv").set_index("app_id")["lam"]
        tr = ev_all[ev_all["analysis_sample"] & ev_all["discounted"]].dropna(
            subset=["r_during"]
        )
        rate = tr["app_id"].map(lam) * 1.15
        oracle = float(
            np.mean(
                np.log1p(14 * rate * 1.8 * tr["price_ratio_sale"]) - np.log1p(14 * rate)
            )
        )
        est = get("Revenue proxy, during", "ATT")
        self.assertLess(
            abs(est["estimate"] - oracle),
            3 * est["se"] + 0.01,
            msg=f"oracle {oracle:.3f}",
        )
        # placebo near zero; after the sale negative (truth x0.9)
        self.assertLess(abs(get("Reviews, week before", "PLR")["estimate"]), 0.06)
        self.assertLess(get("Reviews, 14 days after", "PLR")["estimate"], 0)
        # counted games combined with paged ones; band and weighted rows present
        ev = pd.read_parquet(ROOT / "data" / "events.parquet")
        self.assertEqual(set(ev["method"]), {"page", "count"})
        for col in [
            "r_net",
            "disc_starts_365",
            "prev_discounted",
            "pub_prev_share",
            "depth_bin",
        ]:
            self.assertIn(col, ev)
        self.assertTrue(res["specification"].str.startswith("WEIGHTED").any())
        self.assertTrue(res["specification"].str.startswith("Size band large").any())
        # ratio estimator recovers the review multiplier (x1.8) without the log compression
        ratio = res[res["specification"].str.startswith("RATIO: Reviews, during")].iloc[
            0
        ]
        self.assertAlmostEqual(
            ratio["estimate"], np.log(1.8), delta=max(3 * ratio["se"], 0.08)
        )
        self.assertTrue(
            res["specification"].str.startswith("Without Winter 2023").any()
        )
        diag = pd.read_csv(ROOT / "output" / "nuisance_fit.csv")
        self.assertTrue(
            ((diag.filter(like="weight_").sum(axis=1) - 1).abs() < 0.01).all()
        )  # ensemble weights


if __name__ == "__main__":
    unittest.main()
