"""Step 1: build the game sample.

Pulls SteamSpy's popularity-ranked app list (about 1,000 apps per page), keeps paid
games with at least MIN_REVIEWS reviews, and flags the mid-sized games whose full
review histories step 3 will pull. SteamSpy limits these pages to 1 request a minute,
so 5 pages take about 5 minutes.

Output: data/sample.csv
"""

import config
import pandas as pd
from common import get_json

rows = []
for page in range(config.STEAMSPY_PAGES):
    print(f"SteamSpy page {page + 1}/{config.STEAMSPY_PAGES} (1 per minute)")
    data = get_json(
        "https://steamspy.com/api.php",
        {"request": "all", "page": page},
        sleep=config.SLEEP_STEAMSPY_ALL,
    )
    if not data:
        print("  empty page; stopping")
        break
    rows.extend(data.values())

df = pd.DataFrame(rows)
df["app_id"] = df["appid"].astype(int)
for col in ("positive", "negative", "price", "initialprice", "ccu"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["reviews_total"] = df["positive"] + df["negative"]
df["base_price_usd"] = df["initialprice"] / 100  # SteamSpy prices are in cents

keep = df["reviews_total"] >= config.MIN_REVIEWS
if config.PAID_ONLY:
    keep &= df["base_price_usd"] > 0
sample = df.loc[
    keep,
    [
        "app_id",
        "name",
        "developer",
        "publisher",
        "reviews_total",
        "positive",
        "negative",
        "base_price_usd",
        "owners",
    ],
].copy()
sample["share_positive"] = (sample["positive"] / sample["reviews_total"]).round(3)
sample["mid_sized"] = sample["reviews_total"].between(
    config.REVIEW_MIN, config.REVIEW_MAX
)
sample = sample.drop_duplicates("app_id").sort_values("reviews_total", ascending=False)
sample.to_csv(config.SAMPLE_CSV, index=False)

print(
    f"\nPulled {len(df)} apps; kept {len(sample)} paid games with "
    f">= {config.MIN_REVIEWS} reviews"
)
print(
    f"Mid-sized ({config.REVIEW_MIN:,}-{config.REVIEW_MAX:,} reviews): "
    f"{sample['mid_sized'].sum()}  (step 3 pulls up to {config.MAX_REVIEW_APPS})"
)
print(f"Wrote {config.SAMPLE_CSV}")
