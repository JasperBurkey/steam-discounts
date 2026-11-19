"""Step 3l: past Steam prices of each study game's top DLC, from IsThereAnyDeal.

The DLC test (does a base-game discount lift DLC sales?) needs to know whether the DLC was
discounted in the same sale: publishers usually discount both. 03k_pull_dlc.py picks each
game's most-reviewed DLC (data/raw/dlc_info.parquet); this pulls those DLCs' price
histories exactly as 03b_pull_prices.py does for games (same API, same cache folder, so
reruns only fetch what is missing). About 1,000 DLCs, a few minutes.

  python 03l_pull_dlc_prices.py
Needs ITAD_API_KEY in .env (as for 03b) and data/raw/dlc_info.parquet (from 03k).
Output: data/raw/dlc_price_history.parquet (same columns as price_history.parquet, plus
parent_app_id); export_for_claude.py includes it.
"""

import json
from importlib import import_module

import config
import pandas as pd

B = import_module("03b_pull_prices")
INFO = config.RAW / "dlc_info.parquet"
OUT = config.RAW / "dlc_price_history.parquet"


def main():
    if not config.ITAD_API_KEY:
        raise SystemExit("ITAD_API_KEY is missing from .env")
    if not INFO.exists():
        raise SystemExit(f"{INFO} not found: run 03k_pull_dlc.py first")
    info = pd.read_parquet(INFO).dropna(subset=["top_dlc"])
    parent = dict(zip(info["top_dlc"].astype(int), info["app_id"].astype(int)))
    dlcs = sorted(parent)
    print(f"Looking up {len(dlcs):,} DLCs on IsThereAnyDeal")
    ids = B.lookup_ids(dlcs)
    found = {a: g for a, g in ids.items() if g}
    print(f"Found {len(found):,} of {len(dlcs):,}")
    config.ITAD_DIR.mkdir(parents=True, exist_ok=True)
    for n, (app_id, gid) in enumerate(found.items(), 1):
        path = config.ITAD_DIR / f"{app_id}.json"
        if path.exists():
            continue
        res = B.history(gid)
        if res is None:
            print(f"  {app_id}: request failed; will retry next run")
            continue
        path.write_text(json.dumps(res))
        if n % 100 == 0:
            print(f"  {n:,}/{len(found):,} DLCs", flush=True)
    rows = []
    for app_id, gid in found.items():
        path = config.ITAD_DIR / f"{app_id}.json"
        if path.exists():
            for r in B.history_rows(app_id, gid, json.loads(path.read_text())):
                r["parent_app_id"] = parent[app_id]
                rows.append(r)
    if not rows:
        raise SystemExit("No price histories returned. Send this output to Claude.")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["app_id", "ts"])
    df.to_parquet(OUT, index=False)
    first = df.groupby("app_id")["ts"].min()
    print(
        f"Saved {len(df):,} price changes for {df['app_id'].nunique():,} DLCs to {OUT}; "
        f"{(first <= pd.Timestamp('2024-01-01', tz='UTC')).mean():.0%} have history from "
        "before 2024"
    )


if __name__ == "__main__":
    main()
