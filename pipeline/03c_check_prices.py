"""Step 3c: spot-check IsThereAnyDeal's price data before relying on it.

Picks 15 random study games and prints, for each past sale event in sale_calendar.csv,
the deepest discount IsThereAnyDeal recorded during the event. Compare a handful with
the price history chart on each game's SteamDB page (link printed). If most match, the
price data is trustworthy enough for the analysis.

Also prints a summary of discount participation by event, which should look plausible:
for mid-sized paid games, typically a large share discount in each seasonal sale.

  python 03c_check_prices.py
"""

import config
import pandas as pd
from prices import load_history, sale_times, window_state

hist = load_history()
sales = pd.read_csv(config.SALES_CSV)
sales = sales[sales["verified"].astype(str).str.lower() == "yes"].dropna(
    subset=["start", "end"]
)
now = pd.Timestamp.now(tz="UTC")
events = [(s["sale"], *sale_times(s)) for _, s in sales.iterrows()]
events = [e for e in events if e[2] < now]  # only finished sale events
if not events:
    raise SystemExit("No finished, verified sale events in sale_calendar.csv yet.")
print(
    f"{len(events)} finished sale events; {hist['app_id'].nunique()} games with price history\n"
)

# ---- 15-game spot check ----
games = pd.Series(hist["app_id"].unique()).sample(
    n=min(15, hist["app_id"].nunique()), random_state=1
)
for app_id in games:
    h = hist[hist["app_id"] == app_id]
    print(
        f"app {app_id}   compare with https://steamdb.info/app/{app_id}/   (price history chart)"
    )
    for name, start, end in events:
        st = window_state(h, start, end)
        if st is None:
            print(f"    {name:<20} no record yet (game may not have been released)")
        else:
            print(
                f"    {name:<20} deepest cut {st['max_cut']:>3}%   lowest ${st['min_price_cents'] / 100:>6.2f}"
                f"   regular ${st['regular_cents'] / 100:>6.2f}"
            )
    print()

# ---- participation summary ----
rows = []
for app_id, h in hist.groupby("app_id"):
    for name, start, end in events:
        st = window_state(h, start, end)
        if st is not None:
            rows.append(
                {"sale": name, "discounted": st["max_cut"] > 0, "cut": st["max_cut"]}
            )
summary = (
    pd.DataFrame(rows)
    .groupby("sale", sort=False)
    .agg(
        games=("discounted", "size"),
        share_discounted=("discounted", "mean"),
        median_cut_if_discounted=("cut", lambda c: c[c > 0].median()),
    )
)
summary["share_discounted"] = (summary["share_discounted"] * 100).round(0).astype(
    int
).astype(str) + "%"
print("Discount participation by sale event:")
print(summary.to_string())
