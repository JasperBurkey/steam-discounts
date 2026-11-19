"""Step 3i: Steam-purchaser review counts up to each sales-milestone date (for 24_milestones).

24_milestones.py matches developers' "X copies sold" announcements to the game's cumulative
purchaser reviews on that date. Without this pull it can only do so for games with complete
review histories (about 100 claims); the largest games were counted by window and have
none. One request per claim (Steam's total_matching for reviews from 2000-01-01 to the
claim date), so ~250 requests, a few minutes.

  python 03i_pull_milestone_counts.py
Reads results/milestones/claims.csv; writes data/raw/milestone_counts.parquet
(export_for_claude.py includes it).
"""

import importlib

import config
import pandas as pd

P = importlib.import_module("03_pull_reviews")
CLAIMS = config.ROOT.parent / "results" / "milestones" / "claims.csv"
OUT = config.RAW / "milestone_counts.parquet"
EPOCH_2000 = 946684800


def main():
    c = pd.read_csv(CLAIMS, parse_dates=["date"])
    done = (
        pd.read_parquet(OUT)
        if OUT.exists()
        else pd.DataFrame(columns=["app_id", "date"])
    )
    key = set(zip(done["app_id"], pd.to_datetime(done["date"], utc=True)))
    todo = c[[(a, d) not in key for a, d in zip(c["app_id"], c["date"])]]
    print(f"{len(todo)} claims to count ({len(done)} done)")
    rows = []
    try:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            n = P.count(r.app_id, EPOCH_2000, int(r.date.timestamp()))
            if n is not None:
                rows.append({"app_id": r.app_id, "date": r.date, "reviews_to_date": n})
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    except KeyboardInterrupt:
        print("Stopped; progress saved. Rerun to continue.")
    finally:
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(OUT, index=False)
    print(f"Saved {len(done)} counts to {OUT}")


if __name__ == "__main__":
    main()
