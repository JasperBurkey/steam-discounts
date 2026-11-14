"""Step 4: turn raw reviews into a game-by-day panel of review counts.

Days are Pacific-time calendar days, because Steam sales start at 10:00 AM Pacific.
Days with no reviews are filled with zeros, from each game's first review in the data
to the last day covered by the pull.

Output: data/review_panel.parquet with columns
  app_id, day, reviews, positive, negative
"""

import config
import pandas as pd

files = sorted(config.REVIEWS_DIR.glob("*.parquet"))
if not files:
    raise SystemExit("No review files yet. Run 03_pull_reviews.py first.")
frames = [pd.read_parquet(f) for f in files]
n_empty = sum(len(x) == 0 for x in frames)
# Games with no reviews since the start date have empty files; they contribute nothing,
# and including them would turn every column into a generic "object" type.
rv = pd.concat([x for x in frames if len(x)], ignore_index=True)
rv = rv.drop_duplicates("review_id")
rv["app_id"] = rv["app_id"].astype(int)
# Count positive and negative reviews explicitly, so the counts are always integers.
vote = rv["voted_up"].astype("boolean")
rv["is_pos"] = (vote == True).fillna(False).astype(int)  # noqa: E712
rv["is_neg"] = (vote == False).fillna(False).astype(int)  # noqa: E712
rv["day"] = (
    pd.to_datetime(rv["created_ts"], unit="s", utc=True)
    .dt.tz_convert(config.TZ)
    .dt.tz_localize(None)
    .dt.normalize()
)

daily = (
    rv.groupby(["app_id", "day"])
    .agg(
        reviews=("review_id", "size"),
        positive=("is_pos", "sum"),
        negative=("is_neg", "sum"),
    )
    .reset_index()
)

# The last day is usually only partly covered (the pull ran mid-day), so drop it.
last_day = daily["day"].max() - pd.Timedelta(days=1)

parts = []
for app_id, g in daily.groupby("app_id"):
    days = pd.date_range(g["day"].min(), last_day, freq="D")
    full = g.set_index("day").reindex(days, fill_value=0)
    full["app_id"] = app_id
    parts.append(full.rename_axis("day").reset_index())
panel = pd.concat(parts, ignore_index=True)[
    ["app_id", "day", "reviews", "positive", "negative"]
]
panel = panel.astype(
    {"app_id": "int64", "reviews": "int64", "positive": "int64", "negative": "int64"}
)
panel.to_parquet(config.PANEL_PARQUET, index=False)

print(
    f"{rv['app_id'].nunique()} games, {len(rv):,} reviews, "
    f"{len(panel):,} game-days ({panel['day'].min():%Y-%m-%d} to {panel['day'].max():%Y-%m-%d})"
)
print(
    f"Median reviews per game-day: {panel['reviews'].median():.1f}; "
    f"share of zero days: {(panel['reviews'] == 0).mean():.0%}"
)
print(
    f"{n_empty} games had no reviews since {config.REVIEW_HISTORY_START} (empty files, skipped)"
)
print(f"Wrote {config.PANEL_PARQUET}")
