"""Step 3e: Steam's weekly top-seller charts for every week since 2023 (revenue ranks).

Steam's weekly chart ranks games by revenue (game price plus DLC and in-game purchases;
ranks only, no amounts). The same endpoint the collector uses returns past weeks when
given a date, which 03d_test_topsellers.py confirmed. This downloads, for every week from
the first week of 2023 to the last completed week:

    lists:  global, plus US, DE, GB, BR, IN, JP (as the collector)
    depth:  up to 500 places each (5 requests of 100)

About 200 weeks x 7 lists x up to 5 pages = up to ~7,000 Web API requests (~1 hour). They
count toward Steam's 100,000-per-day limit together with the collector's (capped at 90,000),
so the total stays under it. Resumable: finished week-lists are skipped on rerun.

  .venv\\Scripts\\python 03e_pull_weekly_topsellers.py
Output: data/weekly_topsellers.parquet (week_start, list, rank, appid, ...)
"""

import json
import time

import config
import pandas as pd
import requests

URL = "https://api.steampowered.com/IStoreTopSellersService/GetWeeklyTopSellers/v1/"
LISTS = ["", "US", "DE", "GB", "BR", "IN", "JP"]  # "" = global
DEPTH, PAGE, SLEEP = 500, 100, 0.5
OUT_DIR = config.RAW / "weekly_topsellers"
OUT = config.DATA / "weekly_topsellers.parquet"
FIRST_WEEK = pd.Timestamp(
    "2023-01-03", tz="UTC"
)  # the first chart week (a Tuesday) of 2023


def steam_key():
    import os

    key = os.environ.get("STEAM_API_KEY", "")
    env = config.COLLECTOR_DATA.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("STEAM_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def fetch(session, key, country, start_ts, page_start):
    payload = {
        "country_code": country,
        "context": {
            "language": "english",
            "country_code": country or "US",
            "steam_realm": 1,
        },
        "page_start": page_start,
        "page_count": PAGE,
    }
    if start_ts is not None:
        payload["start_date"] = int(start_ts)
    for attempt in range(5):
        try:
            r = session.get(
                URL, params={"key": key, "input_json": json.dumps(payload)}, timeout=30
            )
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        time.sleep(SLEEP)
        if r.status_code == 200:
            return (r.json() or {}).get("response") or {}
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(20 * (attempt + 1))
            continue
        return None
    return None


def rows_from(resp, list_name, page_start):
    week = resp.get("start_date")
    out = []
    for i, x in enumerate(resp.get("ranks") or []):
        item = x.get("item") or {}
        out.append(
            {
                "week_start": pd.to_datetime(week, unit="s", utc=True)
                if week
                else pd.NaT,
                "list": list_name or "global",
                "rank": x.get("rank") or page_start + i + 1,
                "appid": x.get("appid") or item.get("appid") or item.get("id"),
                "last_week_rank": x.get("last_week_rank"),
                "consecutive_weeks": x.get("consecutive_weeks"),
                "first_top100": x.get("first_top100"),
            }
        )
    return out


def main():
    key = steam_key()
    if not key:
        raise SystemExit(
            "No STEAM_API_KEY found (this folder's .env or the collector's .env)."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT

    current = fetch(s, key, "", None, 0) or {}
    if not current.get("start_date"):
        raise SystemExit("Could not read the current chart week. Send this to Claude.")
    latest = pd.to_datetime(current["start_date"], unit="s", utc=True)
    weeks = pd.date_range(
        FIRST_WEEK, latest - pd.Timedelta(days=7), freq="7D"
    )  # completed weeks only
    jobs = [(w, c) for w in weeks for c in LISTS]
    todo = [
        (w, c)
        for w, c in jobs
        if not (OUT_DIR / f"{w:%Y-%m-%d}_{c or 'global'}.parquet").exists()
    ]
    print(
        f"{len(weeks)} completed weeks x {len(LISTS)} lists = {len(jobs)} week-lists; "
        f"{len(jobs) - len(todo)} already done, {len(todo)} to go (~{len(todo) * 5 * 0.6 / 3600:.1f} h at most)"
    )

    mismatched, failed = 0, 0
    for n, (w, c) in enumerate(todo, 1):
        mid = w + pd.Timedelta(days=3)  # ask for mid-week; Steam returns its week
        rows = []
        for page_start in range(0, DEPTH, PAGE):
            resp = fetch(s, key, c, mid.timestamp(), page_start)
            if resp is None:
                failed += 1
                break
            got = rows_from(resp, c, page_start)
            rows += got
            if len(got) < PAGE:
                break
        if rows:
            df = pd.DataFrame(rows)
            if df["week_start"].iloc[0] != w:
                mismatched += 1
            df["requested_week"] = w
            df.to_parquet(
                OUT_DIR / f"{w:%Y-%m-%d}_{c or 'global'}.parquet", index=False
            )
        if n % 50 == 0 or n == len(todo):
            print(
                f"  {n:,}/{len(todo):,} week-lists | failed {failed} | week label differed {mismatched}",
                flush=True,
            )

    files = sorted(OUT_DIR.glob("*.parquet"))
    allw = (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )
    if len(allw):
        allw.to_parquet(OUT, index=False)
        g = allw[allw["list"] == "global"]
        print(
            f"\nWrote {OUT}: {len(allw):,} rows, {allw['requested_week'].nunique()} weeks, "
            f"{allw['appid'].nunique():,} distinct games; global list depth per week: "
            f"median {g.groupby('requested_week').size().median():.0f}"
        )
    if mismatched:
        print(
            f"Note: for {mismatched} week-lists Steam labeled a different week than requested; "
            f"the analysis uses Steam's own label (week_start)."
        )
    if failed:
        print(f"{failed} week-lists failed; run again to retry them.")


if __name__ == "__main__":
    main()
