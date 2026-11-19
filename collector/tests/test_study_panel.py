"""Thesis study panel: schedule and budget reserve.

python -m unittest tests.test_study_panel -v
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

# a scratch data folder and test keys, set before settings load (as the other tests do)
os.environ.update(
    STEAM_DATA_DIR=tempfile.mkdtemp(prefix="steamcol_"),
    STEAM_API_KEY="test",
    TWITCH_CLIENT_ID="id",
    TWITCH_CLIENT_SECRET="s",
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collect  # noqa: E402
import jobs  # noqa: E402
import settings as S  # noqa: E402


def utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


class StudyPanel(unittest.TestCase):
    def test_study_list_ships_with_collector(self):
        self.assertEqual(len(jobs._study_ids()), 3850)

    def test_every_four_hours_outside_sales(self):
        self.assertEqual(
            collect.latest_slot("ccu_study", utc("2026-11-20 13:30")),
            utc("2026-11-20 12:00"),
        )
        self.assertIsNone(
            collect.latest_slot("ccu_study_sale", utc("2026-11-20 13:30"))
        )

    def test_every_two_hours_during_winter_sale(self):
        self.assertEqual(
            collect.latest_slot("ccu_study_sale", utc("2026-12-20 15:30")),
            utc("2026-12-20 14:00"),
        )

    def test_reserve_counts_remaining_runs_today(self):
        n = len(jobs._study_ids())
        with mock.patch.object(jobs, "utcnow", return_value=utc("2026-11-20 06:10")):
            # 08, 12, 16, 20, plus the 09:00 achievement trial
            self.assertEqual(jobs._study_reserve(), n * 5)
        with mock.patch.object(jobs, "utcnow", return_value=utc("2026-12-20 06:10")):
            self.assertEqual(jobs._study_reserve(), n * 9)  # plus 10, 14, 18, 22
        with mock.patch.object(jobs, "utcnow", return_value=utc("2027-02-01 06:10")):
            self.assertEqual(jobs._study_reserve(), n * 4)  # trial over

    def test_jobs_registered_in_fast_group(self):
        for j in ("ccu_study", "ccu_study_sale", "achievements"):
            self.assertIn(j, S.GROUPS["fast"])
            self.assertIn(j, jobs.JOBS)


class FakeStats:
    """Achievements for app 10, none for 20, an error for 30."""

    def get(self, url, params=None, **kw):
        g = int(params["gameid"])
        if g == 10:
            ach = [{"name": "A1", "percent": "52.25"}, {"name": "A2", "percent": 3.1}]
            return {"achievementpercentages": {"achievements": ach}}, 200
        if g == 20:
            return {"achievementpercentages": {"achievements": []}}, 200
        return None, 403


class AchievementTrial(unittest.TestCase):
    def test_daily_only_during_trial(self):
        self.assertEqual(
            collect.latest_slot("achievements", utc("2026-12-01 10:00")),
            utc("2026-12-01 09:00"),
        )
        self.assertIsNone(collect.latest_slot("achievements", utc("2027-02-01 10:00")))

    def test_rows_per_achievement_and_placeholders(self):
        with (
            mock.patch.object(jobs, "_study_ids", return_value=[10, 20, 30]),
            mock.patch.object(jobs, "_budget_left", return_value=10),
        ):
            stats = jobs.job_achievements(FakeStats(), "t1")
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["games_with_achievements"], 1)
        df = pd.concat(
            pd.read_parquet(f)
            for f in (S.TABLES / "achievements").glob("date=*/t1.parquet")
        )
        a = df[df["appid"] == 10].sort_values("name")
        self.assertEqual(a["percent"].tolist(), [52.25, 3.1])
        self.assertEqual(int(a["n_achievements"].iloc[0]), 2)
        self.assertEqual(df.loc[df["appid"] == 30, "http_status"].iloc[0], "403")
        self.assertTrue(df.loc[df["appid"] == 20, "name"].isna().all())

    def test_budget_trims(self):
        with (
            mock.patch.object(jobs, "_study_ids", return_value=[10, 20, 30]),
            mock.patch.object(jobs, "_budget_left", return_value=1),
        ):
            stats = jobs.job_achievements(FakeStats(), "t2")
        self.assertEqual(stats["trimmed_for_budget"], 2)


if __name__ == "__main__":
    unittest.main()
