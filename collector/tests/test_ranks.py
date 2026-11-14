"""Offline tests for the top-seller ranking job: parsing Steam's search HTML (and the
json=1 format), paging, de-duplication, depth limits, tag rankings and the table schema.
Whether Steam's LIVE pages match is checked by check_ranks.py.

  python -m unittest tests.test_ranks -v
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="steamranks_")
os.environ.update(STEAM_DATA_DIR=TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core  # noqa: E402
import jobs  # noqa: E402
import parse as P  # noqa: E402
import settings as S  # noqa: E402


def row_html(itemkey, appid, name, final=None, disc=None, extra=""):
    price = ""
    if final is not None:
        price = (
            f'<div class="discount_block search_discount_block" data-price-final="{final}" '
            f'data-bundlediscount="0" data-discount="{disc or 0}">...</div>'
        )
    return (
        f'<a href="https://store.steampowered.com/app/{appid}/x/?snr=1" data-ds-appid="{appid}" '
        f'data-ds-itemkey="{itemkey}" data-ds-tagids="[1,2]" {extra}'
        f'class="search_result_row ds_collapse_flag " data-search-page="1">'
        f'<div class="responsive_search_name_combined"><div class="col search_name ellipsis">'
        f'<span class="title">{name}</span></div>{price}</div></a>\n'
    )


def page_body(items, start, total):
    html = "".join(row_html(k, a, n, f, d) for k, a, n, f, d in items)
    return {"success": 1, "results_html": html, "total_count": total, "start": start}


class Parse(unittest.TestCase):
    def test_html_rows(self):
        items = [
            ("App_730", "730", "Counter-Strike 2", None, None),
            ("App_413150", "413150", "Stardew Valley", 1049, 30),
            ("Bundle_232", "1,2,3", "Some Bundle", 2999, 50),
        ]
        rows = P.search_rank_rows(
            page_body(items, 100, 8485), "now", "global", None, 100
        )
        self.assertEqual([r["rank"] for r in rows], [101, 102, 103])
        self.assertEqual(rows[1]["appid"], 413150)
        self.assertEqual(
            (rows[1]["price_final_cents"], rows[1]["discount_pct"]), (1049, 30)
        )
        self.assertIsNone(rows[0]["price_final_cents"])  # free game: no price block
        self.assertEqual(rows[2]["appid"], 1)  # bundle: first app listed
        self.assertEqual(rows[0]["total_count"], 8485)
        self.assertEqual(rows[1]["name"], "Stardew Valley")

    def test_json_items_format(self):
        body = {
            "items": [
                {"name": "A", "logo": "https://cdn/steam/apps/570/capsule.jpg"},
                {"name": "B", "logo": "https://cdn/steam/subs/999/capsule.jpg"},
            ],
            "total_count": 2,
        }
        rows = P.search_rank_rows(body, "now", "global", None, 0)
        self.assertEqual((rows[0]["appid"], rows[0]["itemkey"]), (570, "apps_570"))
        self.assertEqual((rows[1]["appid"], rows[1]["packageid"]), (None, 999))

    def test_single_quotes_and_attribute_order(self):
        html = (
            "<a class='search_result_row ds_collapse_flag' data-ds-itemkey='App_10' data-ds-appid='10' "
            "href='https://store.steampowered.com/app/10/x/'><span class='title'>Ten</span>"
            "<div data-price-final='499' data-discount='50'></div></a>"
            "<a data-ds-appid='20' class='search_result_row' data-ds-itemkey='App_20' "
            "href='https://store.steampowered.com/app/20/y/'><span class='title'>Twenty</span></a>"
        )
        rows = P.search_rank_rows(
            {"results_html": html, "total_count": 2}, "now", "global", None, 0
        )
        self.assertEqual([r["appid"] for r in rows], [10, 20])
        self.assertEqual(
            (rows[0]["price_final_cents"], rows[0]["discount_pct"], rows[0]["name"]),
            (499, 50, "Ten"),
        )

    def test_fallback_without_row_class(self):
        """A different layout: no search_result_row class; each product linked twice (image, title)."""
        html = ""
        for appid, name in [(730, "CS2"), (570, "Dota 2")]:
            html += (
                f'<div class="row"><a href="https://store.steampowered.com/app/{appid}/n/"><img src="c.jpg"></a>'
                f'<a href="https://store.steampowered.com/app/{appid}/n/"><span class="title">{name}</span></a>'
                f'<div data-price-final="1999" data-discount="0"></div></div>'
            )
        html += '<a href="https://store.steampowered.com/bundle/232/b/"><span class="title">A Bundle</span></a>'
        rows = P.search_rank_rows(
            {"results_html": html, "total_count": 3}, "now", "global", None, 0
        )
        self.assertEqual(
            [r["itemkey"] for r in rows], ["App_730", "App_570", "Bundle_232"]
        )
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        self.assertEqual((rows[1]["name"], rows[2]["bundleid"]), ("Dota 2", 232))

    def test_empty_or_error_body(self):
        self.assertEqual(P.search_rank_rows(None, "now", "global", None, 0), [])
        self.assertEqual(
            P.search_rank_rows(
                {"success": 1, "results_html": ""}, "now", "global", None, 0
            ),
            [],
        )


class FakeSearch:
    """Pretends to be Steam search: 250 games globally, 120 per tag; one item shifts pages."""

    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, **kw):
        self.calls += 1
        start, count, tag = params["start"], params["count"], params.get("tags")
        total = 120 if tag else 250
        ids = list(range(1, total + 1))
        if start == 100 and not tag:
            ids[100] = 100  # item 100 shows up again on page 2
        page = ids[start : start + count]
        items = [
            (
                f"App_{(tag or 0) * 10000 + i}",
                str((tag or 0) * 10000 + i),
                f"Game {i}",
                999,
                0,
            )
            for i in page
        ]
        return page_body(items, start, total), 200


class Job(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_job_pages_dedupes_and_writes(self):
        S.RANKS_GLOBAL_DEPTH, S.RANKS_TAG_DEPTH, S.RANKS_PAGE = 10_000, 500, 100
        core.write_table(
            "tag_names",
            [
                {"captured_at": core.utcnow(), "tagid": 7, "name": "X"},
                {"captured_at": core.utcnow(), "tagid": 9, "name": "Y"},
            ],
            P.TAG_NAMES,
            "t1",
        )
        http = FakeSearch()
        stats = jobs.job_ranks(http, "r1")
        self.assertEqual(stats["global_ranked"], 249)  # 250 minus the duplicate
        self.assertEqual(stats["tags"], 2)
        t = core.read_table("search_ranks")
        t = t[t["captured_at"] == t["captured_at"].max()]  # this run only
        g = t[t["list"] == "global"]
        self.assertEqual(g["itemkey"].duplicated().sum(), 0)
        self.assertEqual(
            int(g.loc[g["appid"] == 100, "rank"].iloc[0]), 100
        )  # keeps its first rank
        self.assertEqual(len(t[t["list"] == "tag"]), 240)  # 2 tags x 120
        self.assertEqual(http.calls, 3 + 2 * 2)  # stops at each list's end

    def test_depth_limit(self):
        S.RANKS_GLOBAL_DEPTH, S.RANKS_TAG_DEPTH = 100, 0
        stats = jobs.job_ranks(FakeSearch(), "r2")
        self.assertEqual((stats["global_ranked"], stats["tags"]), (100, 0))


if __name__ == "__main__":
    unittest.main()
