"""RD review-count pull (03h): windows, future windows skipped, resume, failures retried.

python -m unittest tests.test_rd_pull -v
"""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
H = importlib.import_module("03h_pull_rd_reviews")

ROW = {
    "app_id": 1,
    "sale": "Spring Sale 2023",
    "sale_date": "2023-03-16",
    "sale_days": 7,
    "release_date": "2023-02-10",
    "a": 4,
}


class Windows(unittest.TestCase):
    def test_five_windows_in_order(self):
        w = H.windows(ROW)
        self.assertEqual([x[0] for x in w], ["rel7", "D", "E", "F", "G"])
        for _, a, b in w:
            self.assertLess(a, b)
        # E covers exactly the 7 sale days
        e = [x for x in w if x[0] == "E"][0]
        self.assertAlmostEqual((e[2] + 1 - e[1]) / 86400, 7, places=3)

    def test_future_windows_skipped(self):
        r = dict(ROW, sale_date="2099-03-16", release_date="2099-02-10")
        self.assertEqual(H.windows(r), [])

    def test_failure_returns_none(self):
        with mock.patch.object(H.P, "count", side_effect=[3, None]):
            self.assertIsNone(H.pull_one(ROW))
        with mock.patch.object(H.P, "count", return_value=2):
            r = H.pull_one(ROW)
        self.assertEqual(
            {k: r[k] for k in ["rel7", "D", "E", "F", "G"]},
            dict.fromkeys(["rel7", "D", "E", "F", "G"], 2),
        )


class Resume(unittest.TestCase):
    def test_skips_done_and_saves(self):
        with tempfile.TemporaryDirectory() as d:
            uni = Path(d) / "universe.csv"
            out = Path(d) / "out.parquet"
            pd.DataFrame([ROW, dict(ROW, app_id=2)]).to_csv(uni, index=False)
            pd.DataFrame([{"app_id": 1, "sale": ROW["sale"], "D": 0}]).to_parquet(out)
            with (
                mock.patch.object(H, "UNIVERSE", uni),
                mock.patch.object(H, "OUT", out),
                mock.patch.object(H.P, "count", return_value=5) as c,
                mock.patch.object(sys, "argv", ["x", "--workers", "1"]),
            ):
                H.main()
            self.assertEqual(c.call_count, 5)  # only app 2
            got = pd.read_parquet(out)
            self.assertEqual(sorted(got["app_id"]), [1, 2])


class Long(unittest.TestCase):
    def test_long_windows_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            uni = Path(d) / "universe.csv"
            out = Path(d) / "long.parquet"
            pd.DataFrame([ROW]).to_csv(uni, index=False)
            with (
                mock.patch.object(H, "UNIVERSE", uni),
                mock.patch.object(H, "OUT_LONG", out),
                mock.patch.object(H, "OUT", H.OUT),
                mock.patch.object(H, "WINDOWS", H.WINDOWS),
                mock.patch.object(H, "WITH_REL7", True),
                mock.patch.object(H.P, "count", return_value=4) as c,
                mock.patch.object(sys, "argv", ["x", "--long", "--workers", "1"]),
            ):
                H.main()
                self.assertEqual(c.call_count, 2)
            got = pd.read_parquet(out)
            self.assertEqual(sorted(c for c in got.columns if len(c) == 1), ["H", "I"])


if __name__ == "__main__":
    unittest.main()
