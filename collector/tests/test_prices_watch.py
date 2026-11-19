"""Price-change watcher: first run, a quiet hour, an hour with a change.

python -m unittest tests.test_prices_watch -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.update(
    STEAM_DATA_DIR=tempfile.mkdtemp(prefix="steamcol_"),
    STEAM_API_KEY="test",
    TWITCH_CLIENT_ID="id",
    TWITCH_CLIENT_SECRET="s",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402
import jobs  # noqa: E402
import settings as S  # noqa: E402

STUDY = [10, 20, 30]


class FakeStore:
    """GetAppList (two pages) and GetItems with prices."""

    def __init__(self):
        self.pcn = {10: 5, 20: 7, 30: None, 40: 1, 50: 2}  # 30 has no counter
        self.cut = {10: 0, 20: 0, 30: 0}
        self.item_calls = []

    def get(self, url, params=None, **kw):
        p = params or {}
        if "GetAppList" in url:
            ids = sorted(self.pcn)
            last = int(p.get("last_appid", 0))
            page = [a for a in ids if a > last][:3]
            more = page[-1] != ids[-1]
            return {
                "response": {
                    "apps": [
                        {"appid": a, "price_change_number": self.pcn[a]} for a in page
                    ],
                    "have_more_results": more,
                    "last_appid": page[-1],
                }
            }, 200
        if "GetItems" in url:
            req = json.loads(p["input_json"])
            ids = [x["appid"] for x in req["ids"]]
            self.item_calls.append(ids)
            items = []
            for a in ids:
                full = 1999
                final = round(full * (100 - self.cut[a]) / 100)
                items.append(
                    {
                        "appid": a,
                        "success": 1,
                        "best_purchase_option": {
                            "final_price_in_cents": str(final),
                            "original_price_in_cents": str(full),
                            "discount_pct": self.cut[a],
                        },
                    }
                )
            return {"response": {"store_items": items}}, 200
        return None, 404


def utc(h):
    from datetime import datetime, timezone

    return datetime(2026, 12, 17, h, 5, tzinfo=timezone.utc)


class PricesWatch(unittest.TestCase):
    def test_three_hours(self):
        store = FakeStore()
        with mock.patch.object(jobs, "_study_ids", return_value=STUDY):
            with mock.patch.object(jobs, "utcnow", return_value=utc(15)):
                first = jobs.job_prices_watch(store, "r1")
            self.assertTrue(
                first["full_snapshot"]
            )  # first run: every study game priced
            self.assertEqual(first["prices_recorded"], 3)
            self.assertEqual(first["changed_games"], 0)

            store.item_calls.clear()
            with mock.patch.object(jobs, "utcnow", return_value=utc(16)):
                quiet = jobs.job_prices_watch(store, "r2")
            self.assertEqual(
                quiet["changed_games"], 0
            )  # a missing counter is not a change
            self.assertEqual(store.item_calls, [])
            self.assertIn("note", quiet)  # so the health check doesn't flag 0 rows

            store.pcn[20] = 8  # a study game goes on sale
            store.cut[20] = 75
            store.pcn[40] = 2  # a game outside the study changes too
            with mock.patch.object(jobs, "utcnow", return_value=utc(17)):
                change = jobs.job_prices_watch(store, "r3")
            self.assertEqual(change["changed_games"], 2)
            self.assertEqual(change["changed_study_games"], 1)
            self.assertEqual(store.item_calls, [[20]])  # only the study game is priced

        prices = core.read_table("study_prices")
        last = prices[prices["reason"] == "changed"].iloc[-1]
        self.assertEqual(int(last["appid"]), 20)
        self.assertEqual(int(last["discount_pct"]), 75)
        events = core.read_table("price_changes")
        self.assertEqual(sorted(events["appid"].astype(int)), [20, 40])
        self.assertEqual(
            int(events.loc[events["appid"] == 20, "old_number"].iloc[0]), 7
        )

    def test_daily_snapshot_hour(self):
        self.assertIn(S.PRICE_SNAPSHOT_HOUR, S.SLOTS["prices_watch"]["hours"])
        self.assertIn("prices_watch", S.GROUPS["fast"])
        self.assertLess(
            S.GROUPS["fast"].index("prices_watch"), S.GROUPS["fast"].index("ccu_study")
        )


if __name__ == "__main__":
    unittest.main()
