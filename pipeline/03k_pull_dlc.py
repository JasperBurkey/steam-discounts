"""Step 3k: DLC and in-app purchases for the study games, and DLC reviews around sales.

Revenue the review proxy cannot see: DLC and in-game purchases. Two pieces:

1. Flags (from the collector's latest store snapshot, no requests):
     n_dlc            DLC apps whose parent is the game (store `related_items.parent_appid`)
     dlc_reviews      their combined review count now
     top_dlc          the game's most-reviewed DLC (at least MIN_DLC_REVIEWS reviews)
     in_app           the game lists Steam's "In-App Purchases" feature category (id 35)
   Written to data/raw/dlc_info.parquet.

2. Review counts for each game's top DLC around every seasonal sale in the analysis
   sample since 2024 (windows as in 03h, Pacific days from the sale start date d0):
     base   d0-21 .. d0-8 (two weeks, before any pre-sale dip)
     E      the sale        F   d0+n .. d0+n+13
   Only DLC released before the base window. About 3 requests per game-event; the plan
   prints the count and time. Stop with Ctrl+C any time; rerun to continue.
   Written to data/raw/dlc_review_counts.parquet.

If the snapshot has no parent links (an older collector request), part 1 reports zero
DLC and part 2 has nothing to do.

  python 03k_pull_dlc.py --plan
  python 03k_pull_dlc.py
export_for_claude.py includes both files.
"""

import argparse
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import pandas as pd

P = importlib.import_module("03_pull_reviews")
INFO_OUT = config.RAW / "dlc_info.parquet"
OUT = config.RAW / "dlc_review_counts.parquet"
IN_APP_CATEGORY = 35  # Steam store feature category "In-App Purchases"
MIN_DLC_REVIEWS = 10
FIRST_SALE_YEAR = 2024
CHECKPOINT = 200


def store_snapshot():
    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
    if not files:
        raise SystemExit("no collector store snapshot found")
    return pd.read_parquet(files[-1])


def has_category(s, cid):
    try:
        v = json.loads(s) if isinstance(s, str) else None
    except ValueError:
        return False
    if isinstance(v, dict):
        return any(cid in (x or []) for x in v.values() if isinstance(x, list))
    if isinstance(v, list):
        return cid in v
    return False


def dlc_info(study):
    st = store_snapshot().drop_duplicates("appid")
    st["review_count"] = pd.to_numeric(st["review_count"], errors="coerce").fillna(0)
    dlc = st[st["parent_appid"].isin(study)].copy()
    dlc["rel"] = pd.to_datetime(dlc["steam_release_date"], utc=True, errors="coerce")
    top = (
        dlc[dlc["review_count"] >= MIN_DLC_REVIEWS]
        .sort_values("review_count", ascending=False)
        .drop_duplicates("parent_appid")
        .set_index("parent_appid")
    )
    g = st[st["appid"].isin(study)].set_index("appid")
    info = pd.DataFrame({"app_id": sorted(study)})
    info["n_dlc"] = info["app_id"].map(dlc.groupby("parent_appid").size()).fillna(0)
    info["dlc_reviews"] = (
        info["app_id"].map(dlc.groupby("parent_appid")["review_count"].sum()).fillna(0)
    )
    info["top_dlc"] = info["app_id"].map(top["appid"])
    info["top_dlc_reviews"] = info["app_id"].map(top["review_count"])
    info["top_dlc_release"] = info["app_id"].map(top["rel"])
    info["in_app"] = info["app_id"].map(
        g["categories"].map(lambda s: has_category(s, IN_APP_CATEGORY))
        if "categories" in g
        else pd.Series(dtype=bool)
    )
    return info


def todo_events(info):
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev = ev[ev["analysis_sample"]]
    ev = ev[pd.to_datetime(ev["event_start"]).dt.year >= FIRST_SALE_YEAR]
    ev = ev.merge(info.dropna(subset=["top_dlc"]), on="app_id")
    start = pd.to_datetime(ev["event_start"], utc=True)
    ev = ev[ev["top_dlc_release"] < start - pd.Timedelta(days=21)]
    ev["sale_date"] = start.dt.tz_convert(config.TZ).dt.date
    return ev[["app_id", "sale", "top_dlc", "sale_date", "sale_days"]].reset_index(
        drop=True
    )


def windows(r):
    n = int(r["sale_days"])
    d0 = pd.Timestamp(r["sale_date"])
    w = {"base": (-21, -8), "E": (0, n - 1), "F": (n, n + 13)}
    now = time.time()
    out = [(k, *P.day_bounds(d0, a, b)) for k, (a, b) in w.items()]
    return [(k, a, b) for k, a, b in out if b < now]


def pull_one(r):
    row = {"app_id": int(r["app_id"]), "sale": r["sale"], "dlc": int(r["top_dlc"])}
    for k, a, b in windows(r):
        c = P.count(int(r["top_dlc"]), a, b)
        if c is None:
            return None
        row[k] = c
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=config.REVIEW_WORKERS)
    args = ap.parse_args()
    study = set(pd.read_csv(config.STUDY_SAMPLE_CSV)["app_id"].astype(int))
    info = dlc_info(study)
    info.to_parquet(INFO_OUT, index=False)
    print(
        f"{len(info):,} study games: {(info['n_dlc'] > 0).sum():,} with DLC, "
        f"{info['top_dlc'].notna().sum():,} with a DLC of {MIN_DLC_REVIEWS}+ reviews, "
        f"{info['in_app'].fillna(False).sum():,} with in-app purchases -> {INFO_OUT}"
    )
    ev = todo_events(info)
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    if len(done):
        key = set(zip(done["app_id"], done["sale"]))
        ev = ev[[(a, s) not in key for a, s in zip(ev["app_id"], ev["sale"])]]
    n_req = sum(len(windows(r)) for _, r in ev.iterrows())
    rate = args.workers / (config.REVIEW_SLEEP + 0.3)
    print(
        f"{len(ev):,} game-events to pull ({len(done):,} done), ~{n_req:,} requests, "
        f"~{n_req / rate / 3600:.1f} hours at {args.workers} workers"
    )
    if args.plan or ev.empty:
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
        futs = [ex.submit(pull_one, r) for _, r in ev.iterrows()]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if i % CHECKPOINT == 0:
                flush()
                print(
                    f"  {i:,}/{len(ev):,} game-events  [{(time.time() - t0) / 60:.0f} min]",
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
