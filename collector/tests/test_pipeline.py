"""Offline end-to-end test: every job runs against fake API responses shaped like Steam's.

  python -m unittest discover -s tests -v

This checks the plumbing (batching, tiers, budget trimming, schemas, dedupe, scheduling),
not that Steam's live responses match. `python collect.py selftest` checks that part.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="steamcol_")
os.environ.update(
    STEAM_DATA_DIR=TMP,
    STEAM_API_KEY="test",
    TWITCH_CLIENT_ID="id",
    TWITCH_CLIENT_SECRET="s",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect  # noqa: E402
import core  # noqa: E402
import jobs  # noqa: E402
import pandas as pd  # noqa: E402
import settings as S  # noqa: E402

GAMES, DLC = list(range(10, 260)), list(range(5000, 5020))
MISSING = 77  # Steam "forgets" this app, to test failure rows
NOW_TS = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp())


def item(appid, req):
    it = {
        "item_type": 0,
        "id": appid,
        "appid": appid,
        "success": 1,
        "visible": True,
        "name": f"Game {appid}",
        "type": 4 if appid >= 5000 else 0,
        "is_free": False,
        "release": {
            "steam_release_date": NOW_TS - 86400 * appid * 10,
            "is_coming_soon": appid == 259,
        },
        "platforms": {
            "windows": True,
            "mac": appid % 2 == 0,
            "steam_deck_compat_category": 3,
        },
        "reviews": {
            "summary_filtered": {
                "review_count": appid * 20,
                "percent_positive": 85,
                "review_score": 8,
            }
        },
        "best_purchase_option": {
            "packageid": appid,
            "final_price_in_cents": "999",
            "original_price_in_cents": "1999",
            "lowest_recent_price_in_cents": "899",
            "discount_pct": 50,
            "formatted_final_price": "$9.99",
            "active_discounts": [
                {
                    "discount_end_date": NOW_TS + 86400,
                    "discount_description": "Autumn Sale",
                }
            ],
        },
        "tags": [{"tagid": 492, "weight": 900}, {"tagid": 19, "weight": 500}],
        "basic_info": {
            "developers": [{"name": "Dev"}],
            "publishers": [{"name": "Pub"}],
            "short_description": f"Short {appid}",
        },
        "assets": {
            "asset_url_format": f"steam/apps/{appid}/${{FILENAME}}",
            "header": "header.jpg?t=1",
        },
    }
    if appid == 150:
        it["assets"] = {
            "asset_url_format": f"steam/apps/{appid}/${{FILENAME}}"
        }  # no header listed
    if appid == 20:
        it["reviews"]["summary_unfiltered"] = {
            "review_count": 900,
            "percent_positive": 40,
        }
    if req.get("include_full_description"):
        it["full_description"] = f"<p>About game {appid}</p>"
        it["screenshots"] = {
            "all_ages_screenshots": [{"filename": f"ss_{appid}.jpg", "ordinal": 0}]
        }
        it["purchase_options"] = [{"packageid": appid, "final_price_in_cents": "999"}]
    return it


class FakeHttp:
    def __init__(self, job="test", run_id="r"):
        self.calls, self.failures, self.raw_path = {}, 0, Path(TMP) / "fake.jsonl.gz"

    def close(self):
        core.write_json(
            S.STATE / f"calls_{core.utcnow():%Y-%m-%d}_fake.json", self.calls
        )

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        data=None,
        save_raw=True,
        binary=False,
        tries=5,
    ):
        host = url.split("/")[2]
        self.calls[host] = self.calls.get(host, 0) + 1
        p = params or {}
        if binary:
            return b"\xff\xd8fakejpeg", 200
        if "GetAppList" in url:
            ids = GAMES if p.get("include_games") == "true" else DLC
            return {
                "response": {
                    "apps": [
                        {
                            "appid": a,
                            "name": f"Game {a}",
                            "last_modified": NOW_TS,
                            "price_change_number": 1,
                        }
                        for a in ids
                    ]
                }
            }, 200
        if "GetItems" in url:
            req = json.loads(p["input_json"])
            ids = [x["appid"] for x in req["ids"]]
            if len(ids) > 60:
                return None, 400  # forces the adaptive split
            return {
                "response": {
                    "store_items": [
                        item(a, req["data_request"]) for a in ids if a != MISSING
                    ]
                }
            }, 200
        if "GetNumberOfCurrentPlayers" in url:
            return {"response": {"player_count": p["appid"], "result": 1}}, 200
        if "GetMostPlayedGames" in url:
            return {
                "response": {
                    "rollup_date": NOW_TS,
                    "ranks": [{"rank": 1, "appid": 15, "peak_in_game": 5}],
                }
            }, 200
        if "GetGamesByConcurrentPlayers" in url:
            return {
                "response": {
                    "ranks": [{"rank": 1, "appid": 15, "concurrent_in_game": 3}]
                }
            }, 200
        if "GetWeeklyTopSellers" in url:
            return {
                "response": {"start_date": NOW_TS, "ranks": [{"rank": 1, "appid": 12}]}
            }, 200
        if "featured" in url:
            return {
                "specials": {
                    "id": "cat_specials",
                    "items": [
                        {
                            "id": 30,
                            "type": 0,
                            "discounted": True,
                            "discount_percent": 75,
                        }
                    ],
                },
                "large_capsules": [{"id": 31, "type": 0}],
            }, 200
        if "GetTagList" in url:
            return {"response": {"tags": [{"tagid": 492, "name": "Indie"}]}}, 200
        if "GetStoreCategories" in url:
            return {
                "response": {
                    "categories": [
                        {"categoryid": 2, "type": 1, "display_name": "Single-player"}
                    ]
                }
            }, 200
        if "oauth2/token" in url:
            return {"access_token": "t", "expires_in": 5_000_000}, 200
        if "helix/streams" in url:
            if p.get("after"):
                return {
                    "data": [
                        {
                            "id": "3",
                            "game_id": "g2",
                            "game_name": "B",
                            "viewer_count": 10,
                            "language": "de",
                        }
                    ],
                    "pagination": {},
                }, 200
            return {
                "data": [
                    {
                        "id": "1",
                        "user_login": "a",
                        "game_id": "g1",
                        "game_name": "A",
                        "viewer_count": 500,
                        "language": "en",
                    },
                    {
                        "id": "2",
                        "user_login": "b",
                        "game_id": "g1",
                        "game_name": "A",
                        "viewer_count": 100,
                        "language": "es",
                    },
                ],
                "pagination": {"cursor": "c"},
            }, 200
        return None, 404

    def get(self, url, params=None, **kw):
        return self.request("GET", url, params=params, **kw)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        S.ITEMS_BATCH = 100
        cls.stats = {}
        for job in [
            "universe",
            "charts",
            "topsellers",
            "items",
            "regional",
            "featured",
            "ccu_a",
            "ccu_b",
            "ccu_c",
            "twitch",
            "weekly",
        ]:
            cls.stats[job] = jobs.JOBS[job](FakeHttp(), f"run_{job}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_universe(self):
        u = core.latest_run("universe")
        self.assertEqual(len(u), len(GAMES) + len(DLC))
        self.assertTrue(u["in_current_list"].all())

    def test_items_schema_and_failures(self):
        it = core.latest_run("store_items")
        self.assertEqual(list(it.columns), list(jobs.P.STORE_ITEMS))
        self.assertEqual(len(it), len(GAMES) + len(DLC))
        self.assertEqual(it.loc[it["appid"] == MISSING, "http_status"].item(), "failed")
        ok = it[it["http_status"] == "200"]
        self.assertTrue(
            (ok["price_final_cents"] == 999).all()
        )  # Steam sends these as strings
        self.assertTrue((ok["lowest_recent_price_cents"] == 899).all())
        self.assertEqual(
            ok.loc[ok["appid"] == 20, "review_count_unfiltered"].item(), 900
        )
        self.assertEqual(json.loads(ok["tags"].iloc[0])[0], [492, 900])

    def test_tiers(self):
        t = core.latest_run("tiers")
        self.assertNotIn(259, set(t["appid"]))  # coming soon: no player counts
        self.assertFalse(set(t["appid"]) & set(DLC))  # DLC: no player counts
        tier = dict(zip(t["appid"], t["tier"]))
        self.assertEqual(tier[100], "A")  # 2,000 reviews
        self.assertEqual(tier[15], "A")  # charted
        self.assertEqual(tier[11], "B")  # 220 reviews, not recent

    def test_ccu_and_rotation(self):
        c = core.read_table("ccu")
        a_ids = set(core.latest_run("tiers").query("tier == 'A'")["appid"])
        self.assertTrue(a_ids <= set(c.loc[c["tier"] == "A", "appid"]))
        c_ids = set(c.loc[c["tier"] == "C", "appid"])
        if c_ids:
            self.assertEqual(len({a % S.CCU_C_ROTATION for a in c_ids}), 1)

    def test_regional(self):
        r = core.latest_run("regional_prices")
        self.assertEqual(set(r["country"]), set(S.REGIONAL_COUNTRIES))

    def test_featured_charts_twitch(self):
        self.assertEqual(len(core.latest_run("featured")), 4)  # 2 endpoints x 2 items
        self.assertEqual(core.latest_run("charts_most_played")["appid"].item(), 15)
        g = core.latest_run("twitch_games")
        allrow = g[(g["game_id"] == "g1") & (g["language"] == "ALL")]
        self.assertEqual(allrow["viewers"].item(), 600)
        self.assertEqual(len(core.latest_run("twitch_streams")), 3)

    def test_weekly_dedupe(self):
        first = self.stats["weekly"]["text_versions_new"]
        self.assertGreater(first, 0)
        again = jobs.job_weekly(FakeHttp(), "run_weekly_2")
        self.assertEqual(
            again["text_versions_new"], 0
        )  # unchanged text is not stored twice

    def test_capsules(self):
        self.assertGreater(self.stats["items"]["capsules_downloaded"], 0)
        again = jobs.job_items(FakeHttp(), "run_items_2")
        self.assertEqual(
            again["capsules_downloaded"], 0
        )  # unchanged art is not downloaded twice
        self.assertEqual(
            again["capsule_errors"], 0
        )  # a game with no header image is skipped

    def test_capsule_missing_values(self):
        """The crash seen on real data: a tier-A game whose header filename is NaN."""
        today = pd.DataFrame(
            {
                "appid": [100, 101, 102],
                "asset_header": [float("nan"), None, "h.jpg"],
                "asset_url_format": [
                    "steam/apps/100/${FILENAME}",
                    float("nan"),
                    "steam/apps/102/${FILENAME}",
                ],
            }
        )
        tiers = [{"appid": a, "tier": "A"} for a in (100, 101, 102)]
        out = jobs._download_capsules(FakeHttp(), today, pd.DataFrame(), tiers)
        self.assertEqual(out, {"downloaded": 1, "errors": 0})

    def test_budget_trims(self):
        old = S.DAILY_STEAM_API_BUDGET
        n_a = (core.latest_run("tiers")["tier"] == "A").sum()
        S.DAILY_STEAM_API_BUDGET = core.calls_today() + n_a + 2000 + 5
        stats = jobs.job_ccu_b(FakeHttp(), "run_budget")
        S.DAILY_STEAM_API_BUDGET = old
        self.assertLessEqual(stats["requested"], 5)

    def test_schedule(self):
        t = datetime(2026, 11, 4, 21, 30, tzinfo=timezone.utc)  # a Wednesday
        self.assertEqual(collect.latest_slot("items", t).hour, 20)
        self.assertEqual(collect.latest_slot("ccu_a", t).hour, 16)
        self.assertEqual(
            collect.latest_slot("weekly", t).day, 28
        )  # previous Wednesday 22:00
        is_due, slot = collect.due(
            "items", t, {"items": core.iso(collect.latest_slot("items", t))}
        )
        self.assertFalse(is_due)


if __name__ == "__main__":
    unittest.main()
