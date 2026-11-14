"""Fake reviews and prices with a KNOWN discount effect, to test 04b, 05b and 07 offline.

Truth: during a sale every game gets +15% reviews (event traffic); discounted games get
another x1.8 on top, then x0.9 for the 14 days after. Bigger and older games discount
more often (confounding). 07_dml.py should find a clearly positive effect during the sale,
a placebo near zero and a negative after-sale effect.

WARNING: overwrites data/. Run in a scratch copy.
"""

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

rng = np.random.default_rng(3)
days = pd.date_range("2022-12-31", "2026-11-10", freq="D")
days_max = days[-1]
sales = pd.read_csv(config.SALES_CSV, parse_dates=["start", "end"])
sales = sales[sales["end"] < pd.Timestamp("2026-10-20")]
n = 300
lam = np.exp(rng.normal(0.3, 0.8, n))
age0 = rng.integers(100, 3000, n)  # days old on 2022-12-31
panel_rows, price_rows = [], []
for i in range(n):
    app = 10_000 + i
    mult = np.ones(len(days))
    regular = int(rng.choice([999, 1499, 1999, 2999]))
    price_rows.append(
        {
            "app_id": app,
            "itad_id": "x",
            "ts": "2022-06-01T00:00:00Z",
            "price_cents": regular,
            "regular_cents": regular,
            "cut": 0,
            "currency": "USD",
        }
    )
    for s in sales.itertuples():
        during = (days >= s.start) & (days < s.end)
        mult[during] *= 1.15
        p = 1 / (1 + np.exp(-(0.8 * np.log(lam[i]) + 0.0004 * age0[i] - 0.2)))
        if rng.random() < p:
            cut = int(rng.choice([20, 40, 50, 75]))
            mult[during] *= 1.8
            mult[(days >= s.end) & (days < s.end + pd.Timedelta(days=14))] *= 0.9
            for ts, c in [
                (s.start + pd.Timedelta(hours=17, minutes=30), cut),
                (s.end + pd.Timedelta(hours=17, minutes=30), 0),
            ]:
                price_rows.append(
                    {
                        "app_id": app,
                        "itad_id": "x",
                        "ts": ts.tz_localize("UTC"),
                        "price_cents": regular * (100 - c) // 100,
                        "regular_cents": regular,
                        "cut": c,
                        "currency": "USD",
                    }
                )
    r = rng.poisson(lam[i] * mult)
    pos = rng.binomial(r, 0.85)
    panel_rows.append(
        pd.DataFrame(
            {
                "app_id": app,
                "day": days,
                "reviews": r,
                "positive": pos,
                "negative": r - pos,
            }
        )
    )
# The busiest fifth of games become "counted" games (window totals only, like the real
# pull's largest games); the rest stay "paged". The sample file gives bands and weights.
import importlib  # noqa: E402

WINDOWS = importlib.import_module("03_pull_reviews").WINDOWS
panel = pd.concat(panel_rows)
busy = set(10_000 + np.argsort(lam)[-n // 5 :])
config.COUNTS_DIR.mkdir(parents=True, exist_ok=True)
for f in config.COUNTS_DIR.glob("*.parquet"):
    f.unlink()
for app in busy:
    g = panel[panel["app_id"] == app].set_index("day")
    rows = []
    for s in sales.itertuples():
        nd = (s.end - s.start).days
        for w, f in WINDOWS.items():
            a, b = f(nd)
            days = pd.date_range(
                s.start + pd.Timedelta(days=a), s.start + pd.Timedelta(days=b)
            )
            if days[-1] > days_max:
                continue
            col = "positive" if w == "POS" else "reviews"
            rows.append(
                {
                    "app_id": app,
                    "sale": s.sale,
                    "window": w,
                    "rel_start": a,
                    "rel_end": b,
                    "reviews": int(g.loc[days, col].sum()),
                }
            )
    pd.DataFrame(rows).to_parquet(config.COUNTS_DIR / f"{app}.parquet", index=False)
panel[~panel["app_id"].isin(busy)].to_parquet(config.PANEL_PARQUET, index=False)
band = [
    "large" if 10_000 + i in busy else ("mid" if lam[i] > 1 else "small")
    for i in range(n)
]
pd.DataFrame(
    {
        "app_id": 10_000 + np.arange(n),
        "band": band,
        "weight": [{"small": 10.0, "mid": 4.0, "large": 1.0}[b] for b in band],
        "mid_sized": [b == "mid" for b in band],
    }
).to_csv(config.STUDY_SAMPLE_CSV, index=False)
ph = pd.DataFrame(price_rows)
ph["ts"] = pd.to_datetime(ph["ts"], utc=True)
ph.sort_values(["app_id", "ts"]).to_parquet(config.PRICE_HISTORY_PARQUET, index=False)

# a fake collector snapshot with release dates
snap = config.DATA / "fake_collector" / "tables" / "store_items" / "date=2026-11-11"
snap.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "appid": 10_000 + np.arange(n),
        "steam_release_date": pd.Timestamp("2022-12-31", tz="UTC")
        - pd.to_timedelta(age0, unit="D"),
        "publishers": [f'["Publisher {i % 40}"]' for i in range(n)],
    }
).to_parquet(snap / "run1.parquet", index=False)
pd.DataFrame({"app_id": 10_000 + np.arange(n), "lam": lam}).to_csv(
    config.DATA / "fake_truth.csv", index=False
)
print("Fake study data written. Set COLLECTOR_DATA to", config.DATA / "fake_collector")
