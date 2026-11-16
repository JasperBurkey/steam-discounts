"""News pull (03g): paging back with enddate, cutoff date, duplicates.

python -m unittest tests.test_news -v
"""

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
N = importlib.import_module("03g_pull_news")


def fake_feed(n, start, step):
    """n items, newest first, `step` seconds apart."""
    return [
        {
            "gid": str(i),
            "date": start - i * step,
            "title": f"t{i}",
            "tags": ["patchnotes"],
        }
        for i in range(n)
    ]


class Paging(unittest.TestCase):
    def test_pages_back_to_cutoff_without_duplicates(self):
        since = pd.Timestamp("2023-01-01", tz="UTC")
        start = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp())
        feed = fake_feed(450, start, 4 * 86400)  # every 4 days, back to ~2021
        calls = []

        def get(params):
            calls.append(params)
            end = params.get("enddate", 10**12)
            page = [it for it in feed if it["date"] <= end][: params["count"]]
            return {"newsitems": page}

        df = N.page_news(1, get=get, since=since)
        expected = sum(
            pd.Timestamp(it["date"], unit="s", tz="UTC") >= since for it in feed
        )
        self.assertEqual(len(df), expected)
        self.assertTrue(df["gid"].is_unique)
        self.assertTrue((df["date"] >= since).all())
        self.assertEqual(df["tags"].iloc[0], "patchnotes")
        self.assertLessEqual(len(calls), 6)  # stops once past the cutoff

    def test_failure_and_empty_feed(self):
        self.assertIsNone(N.page_news(1, get=lambda p: None))
        self.assertEqual(len(N.page_news(1, get=lambda p: {})), 0)


if __name__ == "__main__":
    unittest.main()
