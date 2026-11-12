"""Step 3 (full project): review data for the stratified sample, as fast as Steam allows.

Uses Steam's IUserReviewsService/GetAppReviews (no key needed). Keyless calls share a
rate limit that Steam enforces with HTTP 429; this script waits and retries when it sees one.

Two ways to get a game's review data:
  page    download every review since REVIEW_HISTORY_START, 100 per request (keeps reviewer
          IDs, playtime and daily counts; used for games up to PAGE_MAX_REVIEWS reviews)
  count   ask Steam how many reviews fall inside each window around each sale event, one
          request per window (date_range_start/end + total_matching). About 9 requests per
          event regardless of game size, so huge games cost ~135 requests instead of thousands.

Run in this order:
  python 03_pull_reviews.py --verify-dir C:\\steam_mvp\\data\\raw\\reviews
        Checks that Steam's window counts match reviews you already downloaded. Do not use
        the count method unless this passes.
  python 03_pull_reviews.py --plan
        One request per game: exact reviews since 2023, chosen method, total requests, time.
  python 03_pull_reviews.py --reuse-from C:\\steam_mvp\\data\\raw\\reviews
        The pull. Copies games you already downloaded, then pages or counts the rest.
        Stop with Ctrl+C at any time; rerun to continue. Finished games are skipped.

Outputs:
  data/raw/reviews/<appid>.parquet          paged games (same format as the MVP)
  data/raw/review_counts/<appid>.parquet    counted games: one row per event x window
"""

import argparse
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import config
import pandas as pd
import requests
from common import hash_id, study_games

URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
_local = threading.local()
_lock = threading.Lock()
STATS = {"requests": 0, "rate_limited": 0, "failed": 0}

# Windows (days relative to sale start, inclusive) whose counts add up to every outcome and
# control the analysis uses. n = number of sale days.
WINDOWS = {
    "A": lambda n: (-60, -29),
    "B": lambda n: (-28, -22),
    "C": lambda n: (-21, -8),
    "D": lambda n: (-7, -1),
    "E": lambda n: (0, n - 1),
    "F": lambda n: (n, n + 13),
    "G": lambda n: (n + 14, n + 27),
    "H": lambda n: (n + 28, n + 41),
    "POS": lambda n: (
        -60,
        -1,
    ),  # positive reviews only, for % positive before the event
}


# ---------------------------------------------------------------- HTTP
def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = config.USER_AGENT
    return _local.s


def call(payload, tries=6):
    """One GetAppReviews request. Returns the `response` dict, or None after repeated failure."""
    params = {"input_json": json.dumps(payload, separators=(",", ":"))}
    for attempt in range(tries):
        with _lock:
            STATS["requests"] += 1
        try:
            r = _session().get(URL, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(config.REVIEW_SLEEP)
            try:
                return r.json().get("response") or {}
            except ValueError:
                return None
        if r.status_code == 429:
            with _lock:
                STATS["rate_limited"] += 1
            wait = r.headers.get("Retry-After")
            time.sleep(
                float(wait) + 1 if wait and wait.isdigit() else 30 * (attempt + 1)
            )
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        return None
    with _lock:
        STATS["failed"] += 1
    return None


def base_payload(appid):
    return {
        "appid": int(appid),
        "filter": 1,
        "review_type": 0,
        "purchase_type": 1,
        "languages": ["all"],
    }


def count(appid, start_ts, end_ts, positive=False):
    """Number of reviews created in [start_ts, end_ts] (inclusive), or None on failure."""
    p = {
        **base_payload(appid),
        "num_per_page": 1,
        "cursor": "*",
        "date_range_start": int(start_ts),
        "date_range_end": int(end_ts),
    }
    if positive:
        p["review_type"] = 1
    resp = call(p)
    return None if resp is None else int(resp.get("total_matching") or 0)


# ---------------------------------------------------------------- time windows
def ts(dt):
    return int(dt.timestamp())


START_TS = ts(
    datetime.fromisoformat(config.REVIEW_HISTORY_START).replace(tzinfo=timezone.utc)
)


def day_bounds(day0, a, b):
    """Unix bounds covering Pacific calendar days day0+a .. day0+b (inclusive)."""
    start = (pd.Timestamp(day0) + pd.Timedelta(days=a)).tz_localize(config.TZ)
    end = (pd.Timestamp(day0) + pd.Timedelta(days=b + 1)).tz_localize(config.TZ)
    return ts(start), ts(end) - 1


def sale_events():
    s = pd.read_csv(config.SALES_CSV)
    s = s[s["verified"].astype(str).str.lower() == "yes"].dropna(
        subset=["start", "end"]
    )
    return [
        (
            r["sale"],
            pd.Timestamp(r["start"]),
            (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days,
        )
        for _, r in s.iterrows()
    ]


def count_requests_per_game():
    now = time.time() - 2 * 3600
    return sum(
        1
        for _, day0, n in sale_events()
        for f in WINDOWS.values()
        if day_bounds(day0, *f(n))[1] < now
    )


# ---------------------------------------------------------------- the two methods
REVIEW_COLUMNS = [
    "app_id",
    "review_id",
    "reviewer_hash",
    "created_ts",
    "voted_up",
    "language",
    "playtime_at_review_min",
    "steam_purchase",
    "received_for_free",
    "refunded",
]


def page_app(appid):
    end_ts = int(time.time())
    rows, cursor, seen = [], "*", set()
    while True:
        resp = call(
            {
                **base_payload(appid),
                "num_per_page": 100,
                "cursor": cursor,
                "date_range_start": START_TS,
                "date_range_end": end_ts,
            }
        )
        if resp is None:
            return None
        page = resp.get("reviews") or []
        if not page:
            break
        for rv in page:
            rid = rv.get("recommendationid")
            if rid in seen:
                continue
            seen.add(rid)
            a = rv.get("author") or {}
            rows.append(
                {
                    "app_id": int(appid),
                    "review_id": str(rid),
                    "reviewer_hash": hash_id(a.get("steamid")),
                    "created_ts": int(rv.get("timestamp_created") or 0),
                    "voted_up": rv.get("voted_up"),
                    "language": rv.get("language"),
                    "playtime_at_review_min": a.get("playtime_at_review"),
                    "steam_purchase": rv.get("steam_purchase"),
                    "received_for_free": rv.get("received_for_free"),
                    "refunded": rv.get("refunded"),
                }
            )
        new_cursor = resp.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def count_app(appid):
    now = time.time() - 2 * 3600
    rows = []
    for sale, day0, n in sale_events():
        for w, f in WINDOWS.items():
            a, b = f(n)
            s_ts, e_ts = day_bounds(day0, a, b)
            if e_ts >= now:
                continue
            c = count(appid, s_ts, e_ts, positive=(w == "POS"))
            if c is None:
                return None
            rows.append(
                {
                    "app_id": int(appid),
                    "sale": sale,
                    "window": w,
                    "rel_start": a,
                    "rel_end": b,
                    "start_ts": s_ts,
                    "end_ts": e_ts,
                    "reviews": c,
                }
            )
    return pd.DataFrame(rows)


def save(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # only complete files get the final name


# ---------------------------------------------------------------- modes
def verify(src_dir, n_games=8):
    files = sorted(Path(src_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No downloaded review files in {src_dir}")
    rng = random.Random(0)
    picks = rng.sample(files, min(n_games, len(files)))
    events = [
        e for e in sale_events() if e[1] < pd.Timestamp.now() - pd.Timedelta(days=30)
    ]
    print(
        f"Comparing Steam's window counts with {len(picks)} games you already downloaded\n"
    )
    print(
        f"{'app':>9} {'event':<18} {'window':<7} {'downloaded':>10} {'Steam count':>11} {'diff':>6}"
    )
    checked, worst = 0, 0.0
    for f in picks:
        rv = pd.read_parquet(f, columns=["created_ts"])
        appid = int(f.stem)
        for sale, day0, n in rng.sample(events, 2):
            for w in ("C", "E"):
                s_ts, e_ts = day_bounds(day0, *WINDOWS[w](n))
                local = int(
                    ((rv["created_ts"] >= s_ts) & (rv["created_ts"] <= e_ts)).sum()
                )
                remote = count(appid, s_ts, e_ts)
                if remote is None:
                    print(f"{appid:>9} {sale:<18} {w:<7} request failed")
                    continue
                checked += 1
                diff = remote - local
                if abs(diff) > 3:
                    worst = max(worst, abs(diff) / max(local, 1))
                print(
                    f"{appid:>9} {sale:<18} {w:<7} {local:>10,} {remote:>11,} {diff:>+6}"
                )
    print()
    if checked == 0:
        print("FAIL: no requests succeeded. Send this output to Claude.")
    elif worst <= 0.03:
        print("PASS: Steam's counts match your downloads (within 3%, or 3 reviews).")
        print("The count method is safe to use. Next: python 03_pull_reviews.py --plan")
    else:
        print(
            f"FAIL: counts differ by up to {worst:.0%}. Do not use the count method; send this to Claude."
        )


def plan(workers):
    games = study_games()
    if games is None or len(games) == 0:
        raise SystemExit(
            "The study sample is empty. Run 01b_build_strata_sample.py first and check "
            "that it reports eligible games."
        )
    print(
        f"Planning pass: {len(games):,} games, one request each "
        f"(~{len(games) * config.SECONDS_PER_REQUEST / workers / 60:.0f} minutes)",
        flush=True,
    )
    end_ts = int(time.time())
    results = {}

    def one(appid):
        return appid, count(appid, START_TS, end_ts)

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, a) for a in games["app_id"]]
        for i, fut in enumerate(as_completed(futures), 1):
            appid, n = fut.result()
            results[appid] = n
            if i % 250 == 0:
                print(f"  {i:,}/{len(games):,}", flush=True)

    per_count = count_requests_per_game()
    g = games.copy()
    g["reviews_since_start"] = g["app_id"].map(results)
    g["method"] = [
        "count" if pd.notna(n) and n > config.PAGE_MAX_REVIEWS else "page"
        for n in g["reviews_since_start"]
    ]
    g["est_requests"] = [
        per_count if m == "count" else int((n if pd.notna(n) else 0) // 100 + 1)
        for m, n in zip(g["method"], g["reviews_since_start"])
    ]
    g.to_csv(config.REVIEW_PLAN_CSV, index=False)

    print(
        f"\n{'band':<11} {'games':>6} {'paged':>6} {'counted':>8} {'reviews since 2023':>19} {'requests':>9}"
    )
    for band, b in g.groupby("band", sort=False):
        print(
            f"{band:<11} {len(b):>6,} {(b['method'] == 'page').sum():>6,} {(b['method'] == 'count').sum():>8,} "
            f"{int(b['reviews_since_start'].fillna(0).sum()):>19,} {int(b['est_requests'].sum()):>9,}"
        )
    total = int(g["est_requests"].sum())
    print(
        f"\nTotal: about {total:,} requests, roughly "
        f"{total * config.SECONDS_PER_REQUEST / workers / 3600:.0f} hours with {workers} worker(s)."
    )
    print(
        f"Lookups that failed: {int(g['reviews_since_start'].isna().sum())} (those games will be paged)."
    )
    print(f"Wrote {config.REVIEW_PLAN_CSV}.")
    print(
        "Next: python 03_pull_reviews.py --reuse-from C:\\steam_mvp\\data\\raw\\reviews"
    )


def pull(workers, reuse_from):
    if not config.REVIEW_PLAN_CSV.exists():
        raise SystemExit(
            "Run the planning pass first: python 03_pull_reviews.py --plan"
        )
    g = pd.read_csv(config.REVIEW_PLAN_CSV)

    if reuse_from:
        src, copied = Path(reuse_from), 0
        for appid in g.loc[g["method"] == "page", "app_id"]:
            s, d = src / f"{appid}.parquet", config.REVIEWS_DIR / f"{appid}.parquet"
            if s.exists() and not d.exists():
                shutil.copy2(s, d)
                copied += 1
        print(f"Reused {copied} games already downloaded in {src}")

    def is_done(row):
        d = config.REVIEWS_DIR if row.method == "page" else config.COUNTS_DIR
        return (d / f"{row.app_id}.parquet").exists()

    todo = sorted(
        (r for r in g.itertuples() if not is_done(r)), key=lambda r: -r.est_requests
    )
    total_req = int(sum(r.est_requests for r in todo))
    print(
        f"{len(g) - len(todo):,} games already finished; {len(todo):,} to go "
        f"(~{total_req:,} requests, ~{total_req * config.SECONDS_PER_REQUEST / workers / 3600:.1f} h "
        f"with {workers} worker(s))\n",
        flush=True,
    )
    if not todo:
        print("Nothing to do.")
        return
    started, finished, failed = time.time(), 0, 0

    def work(row):
        if row.method == "page":
            df, path = (
                page_app(row.app_id),
                config.REVIEWS_DIR / f"{row.app_id}.parquet",
            )
        else:
            df, path = (
                count_app(row.app_id),
                config.COUNTS_DIR / f"{row.app_id}.parquet",
            )
        if df is None:
            return False
        save(df, path)
        return True

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for fut in as_completed(futures):
            ok = fut.result()
            finished += 1
            failed += 0 if ok else 1
            if finished % 25 == 0 or finished == len(todo):
                el = time.time() - started
                rate = STATS["requests"] / max(el, 1)
                left = (
                    f"~{max(total_req - STATS['requests'], 0) / rate / 3600:.1f} h left"
                    if rate > 0
                    else "time left: estimating"
                )
                print(
                    f"  {finished:,}/{len(todo):,} games | {STATS['requests']:,} requests "
                    f"({rate:.2f}/s) | rate-limited {STATS['rate_limited']} | failed {failed} | "
                    f"{left}",
                    flush=True,
                )
    print(f"\nDone. {failed} games failed and will be retried if you run this again.")
    if STATS["rate_limited"] > 0.02 * max(STATS["requests"], 1):
        print(
            "Steam rate-limited more than 2% of requests: set REVIEW_WORKERS = 1 in config.py."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify-dir",
        help="compare Steam's window counts with reviews already downloaded here",
    )
    ap.add_argument(
        "--plan",
        action="store_true",
        help="planning pass: exact counts, method and cost per game",
    )
    ap.add_argument(
        "--reuse-from", help="copy matching games already downloaded in this folder"
    )
    ap.add_argument("--workers", type=int, default=config.REVIEW_WORKERS)
    a = ap.parse_args()
    if a.verify_dir:
        verify(a.verify_dir)
    elif a.plan:
        plan(a.workers)
    else:
        pull(a.workers, a.reuse_from)


if __name__ == "__main__":
    main()
