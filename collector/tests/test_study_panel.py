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
            self.assertEqual(jobs._study_reserve(), n * 4)  # 08, 12, 16, 20
        with mock.patch.object(jobs, "utcnow", return_value=utc("2026-12-20 06:10")):
            self.assertEqual(jobs._study_reserve(), n * 8)  # plus 10, 14, 18, 22

    def test_jobs_registered_in_fast_group(self):
        for j in ("ccu_study", "ccu_study_sale"):
            self.assertIn(j, S.GROUPS["fast"])
            self.assertIn(j, jobs.JOBS)


if __name__ == "__main__":
    unittest.main()
