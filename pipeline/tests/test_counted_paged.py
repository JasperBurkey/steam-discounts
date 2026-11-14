"""Paged and counted games must give identical outcomes from the same reviews.

Builds one game's reviews, then derives its window totals two ways: from a daily panel
(as for paged games) and by counting timestamps inside each window's exact bounds (as
Steam does for counted games). Every control and outcome must match.

  python -m unittest tests.test_counted_paged -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ev_mod = importlib.import_module("04b_build_events")
pull = importlib.import_module("03_pull_reviews")


class CountedMatchesPaged(unittest.TestCase):
    def test_same_outcomes(self):
        rng = np.random.default_rng(0)
        for sale_start, n in [
            ("2024-06-27", 14),
            ("2024-03-14", 7),
            ("2024-12-19", 14),
        ]:
            d0 = pd.Timestamp(sale_start)
            lo = (d0 - pd.Timedelta(days=80)).tz_localize(config.TZ)
            hi = (d0 + pd.Timedelta(days=n + 60)).tz_localize(config.TZ)
            ts = rng.integers(int(lo.timestamp()), int(hi.timestamp()), 4000)
            voted = rng.random(len(ts)) < 0.8
            # paged: daily panel by Pacific day, as 04_build_panel.py does
            day = (
                pd.to_datetime(ts, unit="s", utc=True)
                .tz_convert(config.TZ)
                .tz_localize(None)
                .normalize()
            )
            df = pd.DataFrame({"day": day, "r": 1, "p": voted.astype(int)})
            daily = df.groupby("day")[["r", "p"]].sum()
            span = pd.date_range(
                d0 - pd.Timedelta(days=60), d0 + pd.Timedelta(days=n + 41)
            )
            daily = daily.reindex(span, fill_value=0)
            paged = ev_mod.outcomes(ev_mod.paged_sums(daily["r"], daily["p"], d0, n), n)
            # counted: exact timestamp bounds per window, as Steam is asked
            sums = {}
            for w, f in pull.WINDOWS.items():
                s_ts, e_ts = pull.day_bounds(d0, *f(n))
                m = (ts >= s_ts) & (ts <= e_ts)
                sums[w] = float((m & voted).sum() if w == "POS" else m.sum())
            counted = ev_mod.outcomes(sums, n)
            self.assertTrue(paged["coverage_long"] and counted["coverage_long"])
            for k, v in paged.items():
                self.assertAlmostEqual(v, counted[k], places=9, msg=f"{sale_start} {k}")


if __name__ == "__main__":
    unittest.main()
