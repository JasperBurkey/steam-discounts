"""Offline tests for the IsThereAnyDeal price step and the shared study sample.

python -m unittest tests.test_prices -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import prices  # noqa: E402

pull = importlib.import_module("03b_pull_prices")


def rec(ts, price, regular, cut, shop=61):
    return {
        "timestamp": ts,
        "shop": {"id": shop, "name": "Steam"},
        "deal": {
            "price": {"amountInt": price, "currency": "USD"},
            "regular": {"amountInt": regular, "currency": "USD"},
            "cut": cut,
        },
    }


class Prices(unittest.TestCase):
    def test_history_rows_keep_steam_only(self):
        rows = pull.history_rows(
            1,
            "g",
            [
                rec("2024-06-27T19:00:00+02:00", 499, 1999, 75),
                rec("2024-06-27T19:00:00+02:00", 399, 1999, 80, shop=35),
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cut"], 75)

    def test_window_state(self):
        h = pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:00:00Z",
                    "price_cents": 1999,
                    "regular_cents": 1999,
                    "cut": 0,
                },
                {
                    "ts": "2024-06-27T17:05:00Z",
                    "price_cents": 499,
                    "regular_cents": 1999,
                    "cut": 75,
                },
                {
                    "ts": "2024-07-11T17:05:00Z",
                    "price_cents": 1999,
                    "regular_cents": 1999,
                    "cut": 0,
                },
            ]
        )
        h["ts"] = pd.to_datetime(h["ts"], utc=True)
        summer = prices.sale_times({"start": "2024-06-27", "end": "2024-07-11"})
        st = prices.window_state(h, *summer)
        self.assertEqual(st["max_cut"], 75)  # discount started minutes after 10am PT
        spring = prices.sale_times({"start": "2024-03-14", "end": "2024-03-21"})
        self.assertEqual(
            prices.window_state(h, *spring)["max_cut"], 0
        )  # full price carried forward
        early = prices.sale_times({"start": "2023-06-22", "end": "2023-07-06"})
        self.assertIsNone(
            prices.window_state(h, *early)
        )  # before the first record: unknown

    def test_lookup_fallback(self):
        calls = []

        def fake_call(method, path, params=None, body=None, tries=5):
            calls.append(path)
            if method == "POST":
                return None  # batch lookup fails
            return {"found": True, "game": {"id": f"id{params['appid']}"}}

        pull.call = fake_call
        self.assertEqual(pull.lookup_ids([5, 6]), {5: "id5", 6: "id6"})
        self.assertIn("/games/lookup/v1", calls)

    def test_study_sample_is_reproducible(self):
        s = pd.DataFrame(
            {"app_id": range(2000), "mid_sized": [i % 2 == 0 for i in range(2000)]}
        )
        a, b = common.study_games(s, 50), common.study_games(s, 50)
        self.assertEqual(list(a["app_id"]), list(b["app_id"]))
        self.assertTrue(a["mid_sized"].all())
        self.assertNotEqual(
            list(a["app_id"]), sorted(a["app_id"])[:50]
        )  # not just the first 50


if __name__ == "__main__":
    unittest.main()
