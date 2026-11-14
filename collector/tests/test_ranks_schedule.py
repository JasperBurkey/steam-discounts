"""Sale-period scheduling for ranks_sale, and the tag-list comparison verdicts.

python -m unittest tests.test_ranks_schedule -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collect  # noqa: E402
import settings as S  # noqa: E402


def utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


class SaleSchedule(unittest.TestCase):
    def test_due_only_inside_sale_periods(self):
        self.assertIsNone(collect.latest_slot("ranks_sale", utc("2026-11-20 12:30")))
        self.assertEqual(
            collect.latest_slot("ranks_sale", utc("2026-12-20 12:30")),
            utc("2026-12-20 11:00"),
        )
        self.assertEqual(
            collect.latest_slot("ranks_sale", utc("2026-12-14 06:10")),
            utc("2026-12-14 05:00"),
        )
        self.assertIsNone(collect.latest_slot("ranks_sale", utc("2026-12-13 23:30")))
        self.assertEqual(
            collect.latest_slot("ranks", utc("2026-11-20 18:00")),
            utc("2026-11-20 17:00"),
        )

    def test_health_skips_sale_job_outside_periods(self):
        self.assertFalse(
            collect.in_periods(S.SLOTS["ranks_sale"], utc("2027-02-01 00:00"))
        )
        self.assertTrue(
            collect.in_periods(S.SLOTS["ranks_sale"], utc("2027-01-07 23:00"))
        )
        self.assertTrue(collect.in_periods(S.SLOTS["items"], utc("2027-02-01 00:00")))


class TagComparison(unittest.TestCase):
    def run_case(self, redundant):
        tmp = tempfile.mkdtemp(prefix="tagcmp_")
        try:
            env = {**os.environ, "STEAM_DATA_DIR": tmp}
            code = f"""
import sys, json; sys.path.insert(0, r"{ROOT}")
import core, parse as P
now = core.utcnow()
glob = [dict(captured_at=now, list="global", tagid=None, rank=i + 1, itemkey=f"App_{{i}}", appid=i, packageid=None,
             bundleid=None, name=str(i), price_final_cents=None, discount_pct=None, page_start=0, total_count=300)
        for i in range(300)]
tag_games = [i for i in range(300) if i % 3 == 0]
if not {redundant}:
    tag_games = tag_games[::-1]
tags = [dict(glob[i], list="tag", tagid=7, rank=k + 1) for k, i in enumerate(tag_games)]
core.write_table("search_ranks", glob + tags, P.SEARCH_RANKS, "r1", when=now)
items = [dict(captured_at=now, appid=i, tags=json.dumps([[7, 100]] if i % 3 == 0 else [[9, 100]]),
              tagids=json.dumps([7] if i % 3 == 0 else [9])) for i in range(300)]
core.write_table("store_items", items, dict(captured_at=P.T_, appid=P.I_, tags=P.S_, tagids=P.S_), "r1", when=now)
"""
            subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=tmp)
            return subprocess.run(
                [sys.executable, str(ROOT / "compare_tag_ranks.py")],
                env=env,
                cwd=tmp,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_redundant_verdict(self):
        self.assertIn("can be rebuilt from data", self.run_case(True))

    def test_not_redundant_verdict(self):
        self.assertIn("Keep them for now", self.run_case(False))


if __name__ == "__main__":
    unittest.main()
