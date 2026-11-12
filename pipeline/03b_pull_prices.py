"""Step 3b: download each study game's past Steam prices from IsThereAnyDeal.

Steam itself only shows today's price. IsThereAnyDeal has recorded Steam price changes
for years, which tells us whether each game was discounted during each past sale event.

  python 03b_pull_prices.py --check     one game (Stardew Valley), printed; run this first
  python 03b_pull_prices.py             all study games (about 10 minutes for 500)

Needs ITAD_API_KEY in .env (register an app at isthereanydeal.com).
Each game's response is cached in data/raw/itad/, so reruns only fetch what is missing.

Outputs:
  data/itad_ids.csv            Steam app ID -> IsThereAnyDeal game ID (blank if not found)
  data/price_history.parquet   one row per Steam price change, US prices
"""

import argparse
import json
import time
from importlib import import_module

import config
import pandas as pd
import requests
from common import study_games

eligible_games = import_module("01b_build_strata_sample").eligible_games

API = "https://api.isthereanydeal.com"
STEAM_SHOP = 61
SESSION = requests.Session()
SESSION.headers["User-Agent"] = config.USER_AGENT


def call(method, path, params=None, body=None, tries=5):
    """One request with the API key. Waits and retries when rate-limited."""
    params = {**(params or {}), "key": config.ITAD_API_KEY}
    for attempt in range(tries):
        try:
            r = SESSION.request(
                method, API + path, params=params, json=body, timeout=30
            )
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(config.SLEEP_ITAD)
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 30))
            print(f"  rate-limited; waiting {wait}s")
            time.sleep(wait + 1)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        print(f"  HTTP {r.status_code} from {path}: {r.text[:200]}")
        return None
    return None


def lookup_ids(app_ids):
    """Map Steam app IDs to IsThereAnyDeal game IDs, 200 at a time."""
    out = {}
    for i in range(0, len(app_ids), 200):
        chunk = app_ids[i : i + 200]
        res = call(
            "POST", f"/lookup/id/shop/{STEAM_SHOP}/v1", body=[f"app/{a}" for a in chunk]
        )
        if res is None:  # fall back to one-at-a-time lookups
            for a in chunk:
                one = call("GET", "/games/lookup/v1", {"appid": a}) or {}
                out[a] = (one.get("game") or {}).get("id") if one.get("found") else None
            continue
        for a in chunk:
            out[a] = res.get(f"app/{a}")
    return out


def history(itad_id):
    """Every recorded Steam price change for one game, US prices."""
    return call(
        "GET",
        "/games/history/v2",
        {
            "id": itad_id,
            "country": "US",
            "shops": STEAM_SHOP,
            "since": config.PRICE_HISTORY_SINCE,
        },
    )


def history_rows(app_id, itad_id, records):
    rows = []
    for rec in records or []:
        deal = rec.get("deal") or {}
        price, regular = deal.get("price") or {}, deal.get("regular") or {}
        if (rec.get("shop") or {}).get("id") != STEAM_SHOP:
            continue
        rows.append(
            {
                "app_id": app_id,
                "itad_id": itad_id,
                "ts": rec.get("timestamp"),
                "price_cents": price.get("amountInt"),
                "regular_cents": regular.get("amountInt"),
                "cut": deal.get("cut"),
                "currency": price.get("currency"),
            }
        )
    return rows


def check():
    ids = lookup_ids([413150])
    gid = ids.get(413150)
    print(f"Stardew Valley (app 413150) -> IsThereAnyDeal ID: {gid}")
    if not gid:
        raise SystemExit(
            "Lookup failed. Check ITAD_API_KEY in .env, then send this output to Claude."
        )
    rows = history_rows(413150, gid, history(gid))
    if not rows:
        raise SystemExit("No price history returned. Send this output to Claude.")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    print(
        f"{len(df)} Steam price changes, from {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}"
    )
    if df["ts"].min() > pd.Timestamp("2023-01-01", tz="UTC"):
        print(
            "WARNING: history starts after 2023-01-01, so early sale events would be missing. "
            "Send this output to Claude."
        )
    else:
        print("History reaches back before 2023: good.")
    print("Most recent 10:")
    print(
        df.tail(10)[
            ["ts", "price_cents", "regular_cents", "cut", "currency"]
        ].to_string(index=False)
    )
    print("\nCompare with the price history chart on steamdb.info/app/413150/ .")


def all_eligible_games():
    """Every released, paid game with >= MIN_REVIEWS_FULL reviews, from the collector, plus the sample."""
    base = config.COLLECTOR_DATA / "tables" / "store_items"
    files = (
        sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if base.exists()
        else []
    )
    if not files:
        raise SystemExit(f"No collector snapshot under {base}")
    ids = set(eligible_games(pd.read_parquet(files[-1]))["appid"].astype(int))
    if config.STUDY_SAMPLE_CSV.exists():
        ids |= set(pd.read_csv(config.STUDY_SAMPLE_CSV)["app_id"].astype(int))
    print(
        f"Scope 'all': {len(ids):,} games (every eligible game plus the study sample)"
    )
    return pd.DataFrame({"app_id": sorted(ids)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="test with one game and stop"
    )
    parser.add_argument(
        "--scope",
        choices=["sample", "all"],
        default="sample",
        help="sample: study games only; all: every paid game with enough reviews "
        "(the rivals needed for spillovers; includes the sample)",
    )
    args = parser.parse_args()
    if not config.ITAD_API_KEY:
        raise SystemExit(
            "ITAD_API_KEY is missing. Add a line ITAD_API_KEY=yourkey to .env in this project folder"
        )
    if args.check:
        return check()

    if args.scope == "all":
        games = all_eligible_games()
    else:
        games = study_games(
            pd.read_csv(config.SAMPLE_CSV) if config.SAMPLE_CSV.exists() else None
        )
    app_ids = games["app_id"].astype(int).tolist()
    print(f"Looking up {len(app_ids)} study games on IsThereAnyDeal")
    ids = lookup_ids(app_ids)
    pd.DataFrame({"app_id": list(ids), "itad_id": list(ids.values())}).to_csv(
        config.ITAD_IDS_CSV, index=False
    )
    found = {a: g for a, g in ids.items() if g}
    print(f"Found {len(found)} of {len(app_ids)}")

    for n, (app_id, gid) in enumerate(found.items(), 1):
        path = config.ITAD_DIR / f"{app_id}.json"
        if path.exists():
            continue
        res = history(gid)
        if res is None:
            print(f"  {app_id}: request failed; will retry next run")
            continue
        path.write_text(json.dumps(res))
        if n % 50 == 0:
            print(f"  {n}/{len(found)} games", flush=True)

    rows = []
    for app_id, gid in found.items():
        path = config.ITAD_DIR / f"{app_id}.json"
        if path.exists():
            rows += history_rows(app_id, gid, json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["app_id", "ts"])
    df.to_parquet(config.PRICE_HISTORY_PARQUET, index=False)
    per_game = df.groupby("app_id").size()
    print(
        f"\nSaved {len(df):,} price changes for {df['app_id'].nunique()} games "
        f"(median {per_game.median():.0f} changes per game)"
    )
    first = df.groupby("app_id")["ts"].min()
    late = (first > pd.Timestamp("2023-01-01", tz="UTC")).sum()
    print(
        f"Earliest record: {df['ts'].min():%Y-%m-%d}. {late} games' history starts after 2023-01-01 "
        f"(expected for games released or added later)."
    )
    print(f"Wrote {config.PRICE_HISTORY_PARQUET}")


if __name__ == "__main__":
    main()
