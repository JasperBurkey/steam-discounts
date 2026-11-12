"""Step 1b: draw the full project's stratified sample from EVERY game on Steam.

Reads the collector's latest daily store snapshot, keeps released, paid games with at
least MIN_REVIEWS_FULL reviews, splits them into size bands by total review count, and
draws a reproducible random sample from each band (sizes in config.STRATA).

Big games are rare but matter, so they are over-sampled relative to their share of Steam.
Each game therefore gets a sampling WEIGHT = games in its band / games sampled from it.
Use the weights for any estimate meant to describe "the typical Steam game"; leave them
out for estimates by band.

  python 01b_build_strata_sample.py
Output: data/study_sample.csv (every later script uses it automatically)
"""

import config
import pandas as pd


def latest_snapshot():
    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = (
        sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if base.exists()
        else []
    )
    if not files:
        raise SystemExit(
            f"No collector store snapshot found under {base}. "
            f"Check COLLECTOR_DATA in config.py."
        )
    print(f"Using collector snapshot {files[-1].parent.name} ({files[-1].name})")
    return pd.read_parquet(files[-1])


def collector_games(items):
    """Rows of the collector's store snapshot that are games.

    Steam's API omits fields at their default value, and "game" (type 0) is the default,
    so games usually have NO type. The collector's daily universe table (from Steam's game
    list) says directly which apps are games; use it, falling back to a missing type."""
    uni_base = config.COLLECTOR_DATA / "tables" / "universe"
    files = (
        sorted(uni_base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if uni_base.exists()
        else []
    )
    if files:
        uni = pd.read_parquet(files[-1], columns=["appid", "kind"])
        game_ids = set(uni.loc[uni["kind"] == "game", "appid"].astype(int))
        return items[items["appid"].astype(int).isin(game_ids)]
    t = items["type"].astype("string")
    return items[t.isna() | t.isin(["0", "None", "nan", ""])]


def flag(series):
    """True/False from a column where Steam omits False: missing counts as False."""
    return series.astype("boolean").fillna(False).astype(bool)


def eligible_games(items, verbose=False):
    """Released, paid games with at least MIN_REVIEWS_FULL reviews, step by step."""
    steps = [
        ("games (not DLC, software, etc.)", collector_games),
        ("store page fetched OK", lambda d: d[d["http_status"].astype(str) == "200"]),
        ("released (not coming soon)", lambda d: d[~flag(d["is_coming_soon"])]),
        ("paid (not free to play)", lambda d: d[~flag(d["is_free"])]),
        (
            f">= {config.MIN_REVIEWS_FULL} reviews",
            lambda d: d[
                pd.to_numeric(d["review_count"], errors="coerce").fillna(0)
                >= config.MIN_REVIEWS_FULL
            ],
        ),
    ]
    d = items
    for label, f in steps:
        d = f(d)
        if verbose:
            print(f"  after keeping {label:<34} {len(d):>9,}")
    return d.copy()


def main():
    items = latest_snapshot()
    print(f"Apps in snapshot: {len(items):,}")
    games = eligible_games(items, verbose=True)
    games["review_count"] = pd.to_numeric(games["review_count"]).astype(int)
    games = games.drop_duplicates("appid")
    print(f"Eligible games: {len(games):,}\n")
    if len(games) == 0:
        raise SystemExit(
            "No eligible games found in the collector snapshot. Send this output to Claude."
        )

    parts = []
    print(
        f"{'band':<11} {'reviews':>22} {'games on Steam':>15} {'sampled':>8} {'weight':>8}"
    )
    for band, (lo, hi, n) in config.STRATA.items():
        pool = games[(games["review_count"] >= lo) & (games["review_count"] < hi)]
        take = pool.sample(n=min(n, len(pool)), random_state=config.SAMPLE_SEED_FULL)
        w = len(pool) / max(len(take), 1)
        parts.append(
            pd.DataFrame(
                {
                    "app_id": take["appid"].astype(int),
                    "name": take["name"],
                    "band": band,
                    "review_count_total": take["review_count"],
                    "band_population": len(pool),
                    "weight": w,
                    "mid_sized": band == "mid",
                }
            )
        )
        hi_txt = "+" if hi >= 10**12 else f"{hi:,}"
        print(
            f"{band:<11} {lo:>10,} - {hi_txt:<10} {len(pool):>15,} {len(take):>8,} {w:>8.1f}"
        )

    sample = pd.concat(parts, ignore_index=True)
    sample.to_csv(config.STUDY_SAMPLE_CSV, index=False)
    print(f"\n{len(sample):,} games written to {config.STUDY_SAMPLE_CSV}")
    print(
        "Next: python 03_pull_reviews.py --verify-dir C:\\steam_mvp\\data\\raw\\reviews"
    )


if __name__ == "__main__":
    main()
