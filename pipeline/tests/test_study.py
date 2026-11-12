"""End-to-end test of the study pipeline on fake data with a known effect (takes ~1 minute).

  python -m unittest tests.test_study -v
Overwrites data/ and output/; run it before downloading real data or in a scratch copy.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Study(unittest.TestCase):
    def test_recovers_known_effect(self):
        env = {**os.environ, "COLLECTOR_DATA": str(ROOT / "data" / "fake_collector")}
        for script in [
            "tests/make_fake_study.py",
            "04b_build_events.py",
            "05b_did_event_study.py",
            "07_dml.py",
        ]:
            subprocess.run(
                [sys.executable, script],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
            )
        res = pd.read_csv(ROOT / "output" / "dml_results.csv").set_index(
            "specification"
        )
        main = res.loc["PLR: discounted -> reviews during sale (MAIN)"]
        self.assertAlmostEqual(main["pct_change"], 0.80, delta=0.08)  # truth: +80%
        placebo = res[res.index.str.startswith("Placebo")].iloc[0]
        self.assertLess(abs(placebo["estimate"]), 0.06)
        after = res[res.index.str.startswith("After sale")].iloc[0]
        self.assertLess(after["estimate"], 0)  # truth: -10%


if __name__ == "__main__":
    unittest.main()
