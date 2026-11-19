"""Step 3h: review counts for every game near the 30-day release cutoff (for 22_release_rd).

The release-rule design (22_release_rd.py) has a strong first stage but only ~1,000
game-events with review data, because the study sample is mostly older games. This pulls
window counts for every paid game released 9-50 days before a seasonal sale (the list
22_release_rd.py writes to results/release_rd/universe.csv), using the same count method
as 03_pull_reviews.py: one request per window, Steam's total_matching.

Windows per game-event (Pacific calendar days; d0 = sale start date, n = sale days):
  rel7   release day to release day + 6 (balance check)
  D      d0-7 .. d0-1         E   d0 .. d0+n-1
  F      d0+n .. d0+n+13      G   d0+n+14 .. d0+n+27
Windows that end in the future are skipped (left missing).

About 5 requests per game-event; with the default 2 workers and 1 s pause, roughly
2 requests a second. Stop with Ctrl+C any time; rerun to continue (finished game-events
are skipped).

  python 03h_pull_rd_reviews.py --plan
  python 03h_pull_rd_reviews.py
Output: data/raw/rd_review_counts.parquet (export_for_claude.py includes it)
"""

import argparse
import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import pandas as pd

P = importlib.import_module("03_pull_reviews")
# from the repo's results folder, or a copy next to this script (as on the analysis PC)
UNIVERSE = next(
    (
        f
        for f in (
            config.ROOT / "universe.csv",
            config.ROOT.parent / "results" / "release_rd" / "universe.csv",
        )
        if f.exists()
    ),
    config.ROOT.parent / "results" / "release_rd" / "universe.csv",
)
OUT = config.RAW / "rd_review_counts.parquet"
WINDOWS = {
    "D": lambda n: (-7, -1),
    "E": lambda n: (0, n - 1),
    "F": lambda n: (n, n + 13),
    "G": lambda n: (n + 14, n + 27),
}
CHECKPOINT = 200


def windows(r):
    """(name, start_ts, end_ts) for one game-event, skipping windows not yet over."""
    now = time.time()
    out = [("rel7", *P.day_bounds(pd.Timestamp(r["release_date"]), 0, 6))]
    for w, f in WINDOWS.items():
        out.append(
            (w, *P.day_bounds(pd.Timestamp(r["sale_date"]), *f(int(r["sale_days"]))))
        )
    return [(w, a, b) for w, a, b in out if b < now]


def pull_one(r):
    row = {"app_id": int(r["app_id"]), "sale": r["sale"]}
    for w, a, b in windows(r):
        c = P.count(r["app_id"], a, b)
        if c is None:
            return None  # retry on the next run
        row[w] = c
    return row


def load_done():
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=config.REVIEW_WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    u = pd.read_csv(UNIVERSE)
    done = load_done()
    if len(done):
        key = set(zip(done["app_id"], done["sale"]))
        u = u[[(a, s) not in key for a, s in zip(u["app_id"], u["sale"])]]
    if args.limit:
        u = u.head(args.limit)
    n_req = sum(len(windows(r)) for _, r in u.iterrows())
    rate = args.workers / (config.REVIEW_SLEEP + 0.3)
    print(
        f"{len(u):,} game-events to pull ({len(done):,} done), ~{n_req:,} requests, "
        f"~{n_req / rate / 3600:.1f} hours at {args.workers} workers"
    )
    if args.plan or u.empty:
        return
    rows, t0 = [], time.time()

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(OUT, index=False)
            rows = []

    ex = ThreadPoolExecutor(args.workers)
    try:
        futs = [ex.submit(pull_one, r) for _, r in u.iterrows()]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if i % CHECKPOINT == 0:
                flush()
                print(
                    f"  {i:,}/{len(u):,} game-events, {P.STATS['requests']:,} requests, "
                    f"{P.STATS['rate_limited']} rate-limited  [{(time.time() - t0) / 60:.0f} min]",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Stopped; progress saved. Rerun to continue.")
        ex.shutdown(wait=False, cancel_futures=True)
    finally:
        flush()
        ex.shutdown(wait=False)
    print(f"Saved {len(done):,} game-events to {OUT}")


if __name__ == "__main__":
    main()
