"""Step 3f: a sample of recent reviews for the largest games, for segmentation (batch 6).

The 645 largest games were COUNTED by window (03_pull_reviews.py), so they have no
reviewer-level data. Segmentation places games by who reviews them, so this pages each
counted game's most recent reviews since REVIEW_HISTORY_START, up to SAMPLE_REVIEWS per game
(Steam returns newest first). Same endpoint, hashing and columns as the main pull.

Saved to data/raw/reviews_sample/<appid>.parquet, a separate folder: these are samples,
not complete daily histories, so the panel and event builders never read them.

  .venv\\Scripts\\python 03f_pull_big_sample.py            (about 4 hours; resumable)
  .venv\\Scripts\\python 03f_pull_big_sample.py --limit 3  (quick test on 3 games)
"""

import argparse
import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import pandas as pd
from common import hash_id

P = importlib.import_module("03_pull_reviews")

SAMPLE_REVIEWS = 2_000
OUT_DIR = config.RAW / "reviews_sample"


def page_recent(appid, n=SAMPLE_REVIEWS):
    """Up to n most recent reviews since REVIEW_HISTORY_START. None on failure."""
    rows, cursor, seen = [], "*", set()
    end_ts = int(time.time())
    while len(rows) < n:
        resp = P.call(
            {
                **P.base_payload(appid),
                "num_per_page": 100,
                "cursor": cursor,
                "date_range_start": P.START_TS,
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
    return pd.DataFrame(rows[:n], columns=P.REVIEW_COLUMNS)


def check_salt(max_games=5):
    """Reviewer IDs must be hashed the same way as in the main pull. Re-download the newest
    page of reviews for a few already-downloaded games and compare hashes review by review."""
    # mid-sized files: busy enough to have reviews, quiet enough that the newest page
    # still overlaps with what the main pull stored
    files = sorted(config.REVIEWS_DIR.glob("*.parquet"), key=lambda f: f.stat().st_size)
    mid = len(files) // 2
    for f in files[mid : mid + max_games]:
        stored = pd.read_parquet(f, columns=["review_id", "reviewer_hash"])
        if stored.empty:
            continue
        fresh = page_recent(int(f.stem), n=100)
        if fresh is None or fresh.empty:
            continue
        both = fresh.merge(stored, on="review_id", suffixes=("_new", "_old"))
        if len(both) == 0:
            continue
        same = (both["reviewer_hash_new"] == both["reviewer_hash_old"]).mean()
        if same > 0.99:
            print(
                f"Reviewer IDs match the main pull ({len(both)} reviews checked). Starting."
            )
            return
        raise SystemExit(
            f"Reviewer IDs do NOT match the main pull ({same:.0%} of {len(both)} reviews): "
            "the salt differs. Send this to Claude; nothing was downloaded."
        )
    raise SystemExit(
        "Could not find reviews to check the salt against. Send this to Claude."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=config.REVIEW_WORKERS)
    ap.add_argument(
        "--limit", type=int, default=0, help="only this many games (testing)"
    )
    args = ap.parse_args()
    if not config.REVIEW_PLAN_CSV.exists():
        raise SystemExit("No review_plan.csv: run 03_pull_reviews.py --plan first.")
    check_salt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(config.REVIEW_PLAN_CSV)
    games = plan.loc[plan["method"] == "count", "app_id"].astype(int).tolist()
    todo = [a for a in games if not (OUT_DIR / f"{a}.parquet").exists()]
    if args.limit:
        todo = todo[: args.limit]
    est = len(todo) * (SAMPLE_REVIEWS / 100) * config.SECONDS_PER_REQUEST / args.workers
    print(
        f"{len(games)} counted games; {len(games) - len(todo)} already sampled, "
        f"{len(todo)} to go (~{est / 3600:.1f} h with {args.workers} worker(s))",
        flush=True,
    )
    done, failed, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(page_recent, a): a for a in todo}
        for f in as_completed(futures):
            a = futures[f]
            df = f.result()
            if df is None:
                failed += 1
            else:
                P.save(df, OUT_DIR / f"{a}.parquet")
            done += 1
            if done % 25 == 0 or done == len(todo):
                s = P.STATS
                print(
                    f"  {done:,}/{len(todo):,} games | {s['requests']:,} requests "
                    f"({s['requests'] / max(time.time() - t0, 1):.2f}/s) | "
                    f"rate-limited {s['rate_limited']} | failed {failed}",
                    flush=True,
                )
    print(f"\nDone. {failed} games failed and will be retried if you run this again.")


if __name__ == "__main__":
    main()
