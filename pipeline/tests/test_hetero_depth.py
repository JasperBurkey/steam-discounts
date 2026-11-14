"""Heterogeneity (honest forest + BLP/GATES/CLAN) and multi-arm depth effects on synthetic
data with known answers.

  python -m unittest tests.test_hetero_depth -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LEARNER", "boost")
os.environ.setdefault("CF_REPEATS", "1")
os.environ.setdefault("HET_SPLITS", "6")
os.environ.setdefault("HET_TREES", "60")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dml_core as M  # noqa: E402
from hetero import generic_ml  # noqa: E402


def make(heterogeneous, n_games=1200, n_events=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        age = rng.uniform(100, 3000)
        rate = np.exp(rng.normal(0, 0.7))
        for k in range(n_events):
            p = 1 / (1 + np.exp(-(0.3 + 0.5 * np.log(rate))))
            d = rng.random() < p
            tau = (0.1 + 0.5 * (age > 1550)) if heterogeneous else 0.35
            rows.append(
                {
                    "app_id": g,
                    "sale": f"S{k}",
                    "discounted": d,
                    "age_days": age,
                    "regular_usd": 19.99,
                    "pre_rate_28": rate,
                    "pre_rate_60": rate,
                    "y": 0.3 * np.log(rate) + tau * d + rng.normal(0, 0.4),
                }
            )
    return pd.DataFrame(rows)


class Heterogeneity(unittest.TestCase):
    def run_case(self, het):
        df = make(het)
        _, phi, _ = M.aipw(df, ["y"])
        x = M.features(df)
        (blp, gates, clan), _ = generic_ml(
            x, phi["y"], df["app_id"].to_numpy(), df[["age_days"]]
        )
        return blp, gates, clan

    def test_detects_real_heterogeneity(self):
        blp, gates, clan = self.run_case(True)
        print(
            f"\\nheterogeneous: loading {blp['het_loading']:.2f}, adj p {blp['het_p_adjusted']:.4f}, "
            f"top-bottom {gates.loc['top - bottom', 'effect']:.3f}, age diff {clan.loc['age_days', 'difference']:.0f}"
        )
        self.assertLess(blp["het_p_adjusted"], 0.05)
        self.assertGreater(
            gates.loc["top - bottom", "effect"], 0.2
        )  # truth 0.5 between age halves
        self.assertGreater(
            clan.loc["age_days", "difference"], 500
        )  # most affected are older
        self.assertAlmostEqual(blp["ate"], 0.35, delta=0.06)  # average 0.1 + 0.5/2

    def test_no_false_heterogeneity(self):
        blp, gates, _ = self.run_case(False)
        print(
            f"\\nhomogeneous: loading {blp['het_loading']:.2f}, adj p {blp['het_p_adjusted']:.3f}"
        )
        self.assertGreater(blp["het_p_adjusted"], 0.05)
        self.assertAlmostEqual(blp["ate"], 0.35, delta=0.06)


class Depth(unittest.TestCase):
    def test_multiarm_recovers_effects(self):
        rng = np.random.default_rng(1)
        arms, truth = (
            ["none", "1-33%", "34-60%", "61%+"],
            {"1-33%": 0.10, "34-60%": 0.25, "61%+": 0.45},
        )
        rows = []
        for g in range(1500):
            age = rng.uniform(100, 3000)
            rate = np.exp(rng.normal(0, 0.7))
            for k in range(5):
                logits = np.array(
                    [0.0, 0.8, 0.6 + 0.0004 * age, 0.2 + 0.0008 * age]
                )  # older games go deeper
                pr = np.exp(logits) / np.exp(logits).sum()
                a = arms[rng.choice(4, p=pr)]
                rows.append(
                    {
                        "app_id": g,
                        "sale": f"S{k}",
                        "age_days": age,
                        "regular_usd": 19.99,
                        "pre_rate_28": rate,
                        "pre_rate_60": rate,
                        "depth_bin": a,
                        "discounted": a != "none",
                        "y": 0.3 * np.log(rate)
                        + 0.0001 * age
                        + truth.get(a, 0.0)
                        + rng.normal(0, 0.4),
                    }
                )
        df = pd.DataFrame(rows)
        eff, _, low, _ = M.multiarm(df, "y", "depth_bin", arms, "none")
        print(
            "\\n"
            + "; ".join(
                f"{a}: {e:+.3f} ({s:.3f}) truth {truth[a]:+.2f}"
                for a, (e, s) in eff.items()
            )
        )
        for a, (e, s) in eff.items():
            self.assertLess(abs(e - truth[a]), max(3 * s, 0.05), msg=a)
        disc = df[df["depth_bin"] != "none"].reset_index(drop=True)
        eff2, _, _, _ = M.multiarm(disc, "y", "depth_bin", arms[1:], "1-33%")
        self.assertLess(
            abs(eff2["61%+"][0] - 0.35), max(3 * eff2["61%+"][1], 0.05)
        )  # 0.45 - 0.10

    def test_att_by_arm_with_selection_on_gains(self):
        """Older games choose deep discounts more often but gain LESS from them, so the effect
        for those choosing deep (ATT) is below the all-games average (ATE)."""
        rng = np.random.default_rng(2)
        arms = ["none", "1-33%", "34-60%", "61%+"]
        rows, tau_deep_chosen = [], []
        for g in range(1500):
            age = rng.uniform(100, 3000)
            old = age > 1550
            for k in range(5):
                logits = np.array([0.0, 0.8, 0.6, -0.6 + 1.6 * old])
                pr = np.exp(logits) / np.exp(logits).sum()
                a = arms[rng.choice(4, p=pr)]
                tau = {"1-33%": 0.1, "34-60%": 0.25, "61%+": 0.2 if old else 0.7}.get(
                    a, 0.0
                )
                if a == "61%+":
                    tau_deep_chosen.append(tau)
                rows.append(
                    {
                        "app_id": g,
                        "sale": f"S{k}",
                        "age_days": age,
                        "regular_usd": 19.99,
                        "pre_rate_28": 1.0,
                        "pre_rate_60": 1.0,
                        "depth_bin": a,
                        "discounted": a != "none",
                        "y": 0.0001 * age + tau + rng.normal(0, 0.4),
                    }
                )
        df = pd.DataFrame(rows)
        eff, _, _, _, att = M.multiarm(
            df, "y", "depth_bin", arms, "none", return_att=True
        )
        truth_ate, truth_att = 0.45, float(np.mean(tau_deep_chosen))
        print(
            f"\n61%+: ATE {eff['61%+'][0]:+.3f} (truth {truth_ate:+.2f}); "
            f"ATT {att['61%+'][0]:+.3f} (truth {truth_att:+.2f})"
        )
        self.assertLess(abs(eff["61%+"][0] - truth_ate), max(3 * eff["61%+"][1], 0.06))
        self.assertLess(abs(att["61%+"][0] - truth_att), max(3 * att["61%+"][1], 0.06))
        self.assertLess(att["61%+"][0], eff["61%+"][0])


if __name__ == "__main__":
    unittest.main()
