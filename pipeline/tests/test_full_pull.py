"""Offline test of batch 4: stratified sample, verification, planning pass, page and count
methods, reuse of earlier downloads. A fake GetAppReviews answers from known reviews.

  python -m unittest tests.test_full_pull -v
Uses a temporary folder; never touches data/.
"""

import importlib
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="steamfull_"))
for name, rel in [
    ("DATA", "data"),
    ("RAW", "data/raw"),
    ("REVIEWS_DIR", "data/raw/reviews"),
    ("COUNTS_DIR", "data/raw/review_counts"),
    ("STUDY_SAMPLE_CSV", "data/study_sample.csv"),
    ("REVIEW_PLAN_CSV", "data/review_plan.csv"),
]:
    setattr(config, name, TMP / rel)
for d in (config.REVIEWS_DIR, config.COUNTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.COLLECTOR_DATA = TMP / "collector"
config.STRATA = {
    "small": (50, 1_000, 10),
    "mid": (1_000, 20_000, 10),
    "large": (20_000, 10**12, 5),
}
config.PAGE_MAX_REVIEWS = 400
config.REVIEW_SLEEP = 0.0

sample_mod = importlib.import_module("01b_build_strata_sample")
pull = importlib.import_module("03_pull_reviews")

rng = np.random.default_rng(0)
NOW = int(time.time())
REVIEWS = {}  # appid -> DataFrame(created_ts, voted_up)


def fake_reviews(appid, n):
    t = np.sort(rng.integers(pull.START_TS - 86400 * 200, NOW, n))[::-1]
    REVIEWS[appid] = pd.DataFrame(
        {
            "created_ts": t,
            "voted_up": rng.random(n) < 0.8,
            "rid": [f"{appid}-{i}" for i in range(n)],
        }
    )


def fake_call(payload, tries=6):
    df = REVIEWS[payload["appid"]]
    lo, hi = payload.get("date_range_start"), payload.get("date_range_end")
    if lo is not None and hi is not None:
        df = df[(df["created_ts"] >= lo) & (df["created_ts"] <= hi)]
    if payload.get("review_type") == 1:
        df = df[df["voted_up"]]
    k = int(payload.get("num_per_page", 20))
    start = 0 if payload.get("cursor", "*") == "*" else int(payload["cursor"])
    page = df.iloc[start : start + k]
    reviews = [
        {
            "recommendationid": r.rid,
            "timestamp_created": int(r.created_ts),
            "voted_up": bool(r.voted_up),
            "author": {"steamid": "7656"},
        }
        for r in page.itertuples()
    ]
    return {"reviews": reviews, "cursor": str(start + k), "total_matching": len(df)}


pull.call = fake_call


class FullPull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snap = config.COLLECTOR_DATA / "tables" / "store_items" / "date=2026-11-11"
        snap.mkdir(parents=True)
        counts = np.concatenate(
            [
                rng.integers(50, 1000, 60),
                rng.integers(1000, 20000, 40),
                rng.integers(20000, 300000, 15),
                [10, 20],
            ]
        )
        n = len(counts)
        # Like real Steam data: games have NO type (default omitted); DLC has type "4".
        # ...and false flags are omitted too, so they arrive as missing (NA), as in the collector.
        items = pd.DataFrame(
            {
                "appid": np.arange(n) + 100,
                "name": [f"G{i}" for i in range(n)],
                "http_status": "200",
                "type": None,
                "is_coming_soon": pd.array([pd.NA] * n, dtype="boolean"),
                "is_free": pd.array(
                    [True if i % 30 == 0 else pd.NA for i in range(n)], dtype="boolean"
                ),
                "review_count": counts,
            }
        )
        dlc = pd.DataFrame(
            {
                "appid": [9001, 9002],
                "name": ["DLC1", "DLC2"],
                "http_status": "200",
                "type": "4",
                "is_coming_soon": pd.array([pd.NA, pd.NA], dtype="boolean"),
                "is_free": pd.array([pd.NA, pd.NA], dtype="boolean"),
                "review_count": [5000, 50000],
            }
        )
        pd.concat([items, dlc]).to_parquet(snap / "r1.parquet", index=False)
        uni = config.COLLECTOR_DATA / "tables" / "universe" / "date=2026-11-11"
        uni.mkdir(parents=True)
        pd.DataFrame(
            {
                "appid": list(items["appid"]) + [9001, 9002],
                "kind": ["game"] * n + ["dlc", "dlc"],
            }
        ).to_parquet(uni / "r1.parquet", index=False)
        sample_mod.main()
        cls.sample = pd.read_csv(config.STUDY_SAMPLE_CSV)
        for appid in cls.sample["app_id"]:
            fake_reviews(int(appid), int(rng.choice([30, 150, 900])))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_1_sample(self):
        s = self.sample
        self.assertEqual(
            dict(s["band"].value_counts()), {"small": 10, "mid": 10, "large": 5}
        )
        self.assertFalse(set(s["app_id"]) & {9001, 9002})  # DLC excluded
        free = {100 + i for i in range(0, 117, 30)}
        self.assertFalse(set(s["app_id"]) & free)  # free games excluded
        self.assertTrue((s["weight"] >= 1).all())
        self.assertGreater(
            s.loc[s["band"] == "small", "weight"].iloc[0], 1
        )  # small games under-sampled

    def test_2_verify_and_plan_and_pull(self):
        # "already downloaded" files for verification and reuse
        prev = TMP / "mvp_reviews"
        prev.mkdir()
        for appid in self.sample["app_id"][:6]:
            df = REVIEWS[int(appid)]
            df[df["created_ts"] >= pull.START_TS].rename(
                columns={"rid": "review_id"}
            ).assign(app_id=int(appid)).to_parquet(
                prev / f"{appid}.parquet", index=False
            )
        pull.verify(prev, n_games=4)  # prints PASS (fake counts are exact)

        pull.plan(workers=2)
        plan = pd.read_csv(config.REVIEW_PLAN_CSV)
        self.assertEqual(set(plan["method"]), {"page", "count"})
        big = plan[plan["reviews_since_start"] > config.PAGE_MAX_REVIEWS]
        self.assertTrue((big["method"] == "count").all())

        pull.pull(workers=2, reuse_from=str(prev))
        for r in plan.itertuples():
            df = REVIEWS[int(r.app_id)]
            if r.method == "page":
                got = pd.read_parquet(config.REVIEWS_DIR / f"{r.app_id}.parquet")
                want = (df["created_ts"] >= pull.START_TS).sum()
                self.assertEqual(len(got), want)
            else:
                got = pd.read_parquet(config.COUNTS_DIR / f"{r.app_id}.parquet")
                e = got[(got["window"] == "E")].iloc[0]
                want = (
                    (df["created_ts"] >= e.start_ts) & (df["created_ts"] <= e.end_ts)
                ).sum()
                self.assertEqual(e.reviews, want)
                pos = got[got["window"] == "POS"].iloc[0]
                want_pos = (
                    (df["created_ts"] >= pos.start_ts)
                    & (df["created_ts"] <= pos.end_ts)
                    & df["voted_up"]
                ).sum()
                self.assertEqual(pos.reviews, want_pos)
        # rerun: everything finished, nothing to do
        pull.pull(workers=2, reuse_from=None)

    def test_3_windows_tile_the_period(self):
        for n in (7, 14, 18):
            spans = [pull.WINDOWS[w](n) for w in "ABCDEFGH"]
            for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
                self.assertEqual(b1 + 1, a2)  # no gaps, no overlaps


if __name__ == "__main__":
    unittest.main()
