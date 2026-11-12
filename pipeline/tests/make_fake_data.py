"""Write fake sample, review and tag files so steps 4-6 can be tested offline.

Fake games get a known effect: reviews double during sales and dip 15% for two weeks
afterwards. If 05_event_study.py recovers roughly that, the code works.

WARNING: overwrites data/. Run in a scratch copy, or delete data/ before the real pull.
  python tests/make_fake_data.py && python 04_build_panel.py && python 05_event_study.py
  python 06_cluster_tags.py --limit 300
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Safety: never overwrite a real download.
MARKER = config.DATA / "FAKE_DATA_MARKER"
if (
    any(config.REVIEWS_DIR.glob("*.parquet")) or config.PRICE_HISTORY_PARQUET.exists()
) and not MARKER.exists():
    raise SystemExit(
        "Refusing to run: real review or price data found in data/. "
        "Run tests in a separate copy of the project folder."
    )
MARKER.write_text("data/ holds fake test data; safe to overwrite or delete.")

rng = np.random.default_rng(1)
GENRES = {
    "roguelike": ["Roguelike", "Deckbuilding", "Indie", "Difficult", "Turn-Based"],
    "survival": ["Survival", "Crafting", "Open World", "Co-op", "Building"],
    "shooter": ["FPS", "Shooter", "Multiplayer", "Action", "PvP"],
    "cozy": ["Farming Sim", "Relaxing", "Cute", "Life Sim", "Pixel Graphics"],
    "strategy": ["Strategy", "4X", "Grand Strategy", "Historical", "Turn-Based"],
}
genre_names = list(GENRES)
n_apps = 300
sample = pd.DataFrame(
    {
        "app_id": np.arange(1000, 1000 + n_apps),
        "name": [f"Fake {genre_names[i % 5]} game {i}" for i in range(n_apps)],
        "developer": "fake",
        "publisher": "fake",
        "reviews_total": rng.integers(500, 30_000, n_apps),
        "base_price_usd": rng.choice([9.99, 19.99, 29.99], n_apps),
    }
)
sample["mid_sized"] = sample["reviews_total"].between(
    config.REVIEW_MIN, config.REVIEW_MAX
)
sample["genre"] = [genre_names[i % 5] for i in range(n_apps)]
sample.drop(columns="genre").to_csv(config.SAMPLE_CSV, index=False)

for r in sample.itertuples():
    tags = {t: int(rng.integers(50, 2000)) for t in GENRES[r.genre]}
    tags.update(
        {t: int(rng.integers(1, 80)) for t in rng.choice(sum(GENRES.values(), []), 3)}
    )
    (config.TAGS_DIR / f"{r.app_id}.json").write_text(json.dumps(tags))

sales = pd.read_csv(config.SALES_CSV, parse_dates=["start", "end"]).dropna(
    subset=["start"]
)
days = pd.date_range("2025-06-01", "2026-11-04", freq="D")
mult = np.ones(len(days))
for s in sales.itertuples():
    mult[(days >= s.start) & (days <= s.end)] = 2.0
    post = (days > s.end) & (days <= s.end + pd.Timedelta(days=14))
    mult[post] = 0.85

reviewer_pools = {
    g: np.arange(i * 20_000, (i + 1) * 20_000) for i, g in enumerate(genre_names)
}
for r in sample[sample["mid_sized"]].head(80).itertuples():
    lam = rng.uniform(1, 8) * mult
    n = rng.poisson(lam)
    day_idx = np.repeat(np.arange(len(days)), n)
    ts = days[day_idx] + pd.to_timedelta(rng.uniform(0, 86_400, len(day_idx)), unit="s")
    pool = reviewer_pools[r.genre]
    who = np.where(
        rng.random(len(ts)) < 0.8,
        rng.choice(pool, len(ts)),
        rng.integers(0, 100_000, len(ts)),
    )
    df = pd.DataFrame(
        {
            "app_id": r.app_id,
            "review_id": [f"{r.app_id}-{i}" for i in range(len(ts))],
            "reviewer_hash": [f"u{w}" for w in who],
            "created_ts": (
                ts.tz_localize(
                    config.TZ, ambiguous="NaT", nonexistent="shift_forward"
                ).astype("int64")
                // 10**9
            ),
            "voted_up": rng.random(len(ts)) < 0.85,
        }
    )
    df = df[df["created_ts"] > 0]
    df.to_parquet(config.REVIEWS_DIR / f"{r.app_id}.parquet", index=False)
print(
    "Fake data written. Expected: ~2.0x during sales, ~0.85x for 14 days after, placebo ~1.0x."
)
