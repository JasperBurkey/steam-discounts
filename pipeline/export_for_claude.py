"""Pack the data Claude needs for the next analyses into one zip, so they can run without
this PC: rank validation, segmentation (batch 6) and spillovers (batch 8).

  .venv\\Scripts\\python export_for_claude.py

Writes data/for_claude.zip. Contents, all derived from public data:
  reviews_compact.parquet   one row per review: game, reviewer (an integer code: the
                            reviewer hashes are replaced by 0, 1, 2, ...), vote, purchase
                            flags, playtime, language, day. Includes the big-game sample
                            (03f) if it exists.
  game_info.parquet         from the latest collector store snapshot, for eligible and
                            sampled games: name, tags, release date, publishers, developers
  weekly_topsellers.parquet Steam's weekly revenue charts (03e)
  price_history.parquet     IsThereAnyDeal price histories (03b)
  study_sample.csv          the sampled games and their size bands
No API keys, no raw Steam IDs, nothing from .env.
"""

import json
import zipfile

import numpy as np
import pandas as pd

import config

OUT = config.DATA / "for_claude"
ZIP = config.DATA / "for_claude.zip"


def reviews():
    dirs = [config.REVIEWS_DIR, config.RAW / "reviews_sample"]
    files = [f for d in dirs if d.exists() for f in sorted(d.glob("*.parquet"))]
    cols = [
        "app_id",
        "reviewer_hash",
        "created_ts",
        "voted_up",
        "language",
        "playtime_at_review_min",
        "steam_purchase",
        "received_for_free",
    ]
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        if len(df):
            df["sampled"] = f.parent.name == "reviews_sample"
            parts.append(df)
    if not parts:
        raise SystemExit("No review files found in data/raw/reviews.")
    rv = pd.concat(parts, ignore_index=True)
    rv = rv.dropna(subset=["reviewer_hash"])
    codes, _ = pd.factorize(rv["reviewer_hash"])
    day0 = pd.Timestamp(config.REVIEW_HISTORY_START, tz="UTC")
    day = (pd.to_datetime(rv["created_ts"], unit="s", utc=True) - day0).dt.days
    out = pd.DataFrame(
        {
            "app_id": rv["app_id"].astype("int32"),
            "reviewer": codes.astype("int32"),
            "day": day.clip(lower=-1).astype("int16"),
            "voted_up": rv["voted_up"].astype("boolean"),
            "steam_purchase": rv["steam_purchase"].astype("boolean"),
            "received_for_free": rv["received_for_free"].astype("boolean"),
            "playtime_min": pd.to_numeric(rv["playtime_at_review_min"], errors="coerce")
            .clip(upper=2**31 - 1)
            .astype("Int32"),
            "language": rv["language"].astype("category"),
            "sampled": rv["sampled"],
        }
    )
    n_sample = out.loc[out["sampled"], "app_id"].nunique()
    print(
        f"reviews: {len(out):,} reviews, {out['app_id'].nunique():,} games "
        f"({n_sample} from the big-game sample), {out['reviewer'].nunique():,} reviewers"
    )
    return out


def game_info(ids):
    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
    if not files:
        print("game_info: no collector store snapshot found; skipped")
        return None
    cols = [
        "appid",
        "name",
        "type",
        "steam_release_date",
        "review_count",
        "price_original_cents",
        "developers",
        "publishers",
        "tags",
        "tagids",
    ]
    g = pd.read_parquet(files[-1])
    g = g[[c for c in cols if c in g.columns]]
    g = g[g["appid"].isin(ids)].drop_duplicates("appid")
    print(f"game_info: {len(g):,} games from {files[-1].parent.name}/{files[-1].name}")
    return g


def tag_names():
    base = config.COLLECTOR_DATA / "tables" / "tag_names"
    files = sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
    return pd.read_parquet(files[-1]) if files else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    rv = reviews()
    rv.to_parquet(OUT / "reviews_compact.parquet", index=False)
    written.append("reviews_compact.parquet")

    sample = pd.read_csv(config.STUDY_SAMPLE_CSV)
    sample.to_csv(OUT / "study_sample.csv", index=False)
    written.append("study_sample.csv")

    ids = set(sample["app_id"].astype(int))
    if config.PRICE_HISTORY_PARQUET.exists():
        ph = pd.read_parquet(config.PRICE_HISTORY_PARQUET)
        ids |= set(ph["app_id"].astype(int))
        ph.to_parquet(OUT / "price_history.parquet", index=False)
        written.append("price_history.parquet")
        print(
            f"price_history: {ph['app_id'].nunique():,} games, {len(ph):,} price changes"
        )
    else:
        print("price_history: data/price_history.parquet not found; skipped")

    g = game_info(ids)
    if g is not None:
        g.to_parquet(OUT / "game_info.parquet", index=False)
        written.append("game_info.parquet")
    t = tag_names()
    if t is not None:
        t.to_parquet(OUT / "tag_names.parquet", index=False)
        written.append("tag_names.parquet")

    wk = config.DATA / "weekly_topsellers.parquet"
    if wk.exists():
        pd.read_parquet(wk).to_parquet(OUT / "weekly_topsellers.parquet", index=False)
        written.append("weekly_topsellers.parquet")
    else:
        print("weekly_topsellers: not found; run 03e_pull_weekly_topsellers.py")

    (OUT / "manifest.json").write_text(json.dumps({"files": written}, indent=2))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for name in written + ["manifest.json"]:
            z.write(OUT / name, name)
    mb = ZIP.stat().st_size / 1e6
    print(f"\nWrote {ZIP} ({mb:,.1f} MB). Upload it to Claude.")
    if mb > 100:
        print(
            "It is over 100 MB; if the upload fails, upload the files in data/for_claude one by one."
        )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
