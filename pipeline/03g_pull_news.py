"""Step 3g: news and patch notes for every study game (ISteamNews/GetNewsForApp).

Developers often time updates and announcements to land during a sale, so a review jump
in a sale can come from the update rather than the discount. This pulls each game's news
history back to NEWS_SINCE so the analysis can flag updates near each sale.

Steam returns newest first; we page back with `enddate`. No API key needed. One or a few
requests per game: about 3,850 games, roughly 1-2 hours. Resumable: games with a saved
file are skipped.

Saved to data/raw/news/<appid>.parquet, one row per news item:
  app_id, gid, date (UTC), title, url, author, feedname, feedlabel,
  feed_type (1 = the developer's own Steam announcement, 0 = an outside news site),
  tags (Steam's labels joined with ';', e.g. 'patchnotes'), is_external_url, contents
  (first NEWS_CHARS characters only)

  .venv\\Scripts\\python 03g_pull_news.py            (all study games)
  .venv\\Scripts\\python 03g_pull_news.py --limit 3  (quick test on 3 games)
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import pandas as pd
import requests

URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
OUT_DIR = config.RAW / "news"
NEWS_SINCE = pd.Timestamp("2022-12-01", tz="UTC")  # a few months before the first sale
NEWS_CHARS = 300
PAGE = 100
MAX_PAGES = 50
COLUMNS = [
    "app_id",
    "gid",
    "date",
    "title",
    "url",
    "author",
    "feedname",
    "feedlabel",
    "feed_type",
    "tags",
    "is_external_url",
    "contents",
]
_local = threading.local()
STATS = {"requests": 0, "rate_limited": 0}


def _get(params, tries=6):
    """One request; the `appnews` dict, or None after repeated failure."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = config.USER_AGENT
    for attempt in range(tries):
        STATS["requests"] += 1
        try:
            r = _local.s.get(URL, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(0.5)
            try:
                return r.json().get("appnews") or {}
            except ValueError:
                return None
        if r.status_code == 429:
            STATS["rate_limited"] += 1
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code in (403, 404):
            return {}  # app has no news feed
        return None
    return None


def page_news(appid, get=_get, since=NEWS_SINCE):
    """All news items for appid dated at or after `since`, newest first. None on failure."""
    rows, seen, enddate = [], set(), None
    for _ in range(MAX_PAGES):
        params = {"appid": int(appid), "count": PAGE, "maxlength": NEWS_CHARS}
        if enddate is not None:
            params["enddate"] = enddate
        res = get(params)
        if res is None:
            return None
        items = res.get("newsitems") or []
        new = [it for it in items if it.get("gid") not in seen]
        if not new:
            break
        for it in new:
            seen.add(it.get("gid"))
            d = pd.Timestamp(int(it.get("date") or 0), unit="s", tz="UTC")
            if d < since:
                continue
            rows.append(
                {
                    "app_id": int(appid),
                    "gid": str(it.get("gid")),
                    "date": d,
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "author": it.get("author"),
                    "feedname": it.get("feedname"),
                    "feedlabel": it.get("feedlabel"),
                    "feed_type": it.get("feed_type"),
                    "tags": ";".join(it.get("tags") or []),
                    "is_external_url": it.get("is_external_url"),
                    "contents": (it.get("contents") or "")[:NEWS_CHARS],
                }
            )
        oldest = min(int(it.get("date") or 0) for it in new)
        if oldest < since.timestamp() or len(items) < PAGE:
            break
        # items stamped in the same second as `oldest` are repeated on the next page and
        # dropped by `seen`
        enddate = oldest
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(config.STUDY_SAMPLE_CSV)["app_id"].astype(int).tolist()
    todo = [a for a in games if not (OUT_DIR / f"{a}.parquet").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(games):,} study games; {len(games) - len(todo):,} done, {len(todo):,} to go"
    )
    done = failed = items = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(page_news, a): a for a in todo}
        for f in as_completed(futures):
            a, df = futures[f], f.result()
            if df is None:
                failed += 1
            else:
                df.to_parquet(OUT_DIR / f"{a}.parquet", index=False)
                items += len(df)
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(
                    f"  {done:,}/{len(todo):,} games | {items:,} news items | "
                    f"{STATS['requests']:,} requests | rate-limited {STATS['rate_limited']} | "
                    f"failed {failed} | {(time.time() - t0) / 60:.0f} min",
                    flush=True,
                )
    print(f"\nDone. {failed} games failed and will be retried if you run this again.")


if __name__ == "__main__":
    main()
