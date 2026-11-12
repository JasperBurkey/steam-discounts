"""Helpers for working with past Steam prices from IsThereAnyDeal.

IsThereAnyDeal stores a log of price *changes*: each row says "from this moment, the
price was X". The price at any time is therefore the most recent change before it.
"""

import config
import pandas as pd


def sale_times(row):
    """Steam sales start and end at 10:00 Pacific on the calendar dates."""
    start = pd.Timestamp(row["start"]).tz_localize(config.TZ) + pd.Timedelta(hours=10)
    end = pd.Timestamp(row["end"]).tz_localize(config.TZ) + pd.Timedelta(hours=10)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def load_history():
    h = pd.read_parquet(config.PRICE_HISTORY_PARQUET)
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    return h.sort_values(["app_id", "ts"])


def window_state(app_hist, start, end):
    """Deepest discount and lowest price in force at any time in [start, end).

    Includes the price already in force at `start` (the last change before it).
    Returns None if the history has no record at or before `end`, meaning the
    price at that time is unknown.
    """
    before = app_hist[app_hist["ts"] <= start].tail(1)
    during = app_hist[(app_hist["ts"] > start) & (app_hist["ts"] < end)]
    states = pd.concat([before, during])
    if states.empty:
        return None
    return {
        "max_cut": int(states["cut"].max()),
        "min_price_cents": int(states["price_cents"].min()),
        "regular_cents": int(states["regular_cents"].iloc[0]),
        "changes_in_window": len(during),
    }
