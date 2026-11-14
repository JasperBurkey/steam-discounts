"""Helpers for working with past Steam prices from IsThereAnyDeal.

IsThereAnyDeal stores a log of price *changes*: each row says "from this moment, the
price was X". The price at any time is therefore the most recent change before it.
"""

import config
import numpy as np
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


# ---------------------------------------------------------------- fast per-game price arrays
class PriceArrays:
    """One game's price history as sorted arrays, for fast window calculations.

    ratio = price paid / regular price (1.0 at full price, 0.3 at 70% off).
    """

    def __init__(self, h):
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        self.ts = (
            (pd.to_datetime(h["ts"], utc=True) - epoch) // pd.Timedelta(seconds=1)
        ).to_numpy(dtype="int64")
        self.cut = h["cut"].to_numpy(dtype=float)
        reg = h["regular_cents"].to_numpy(dtype=float)
        self.ratio = (
            h["price_cents"].to_numpy(dtype=float)
            / pd.Series(reg).where(reg > 0).to_numpy()
        )
        self.regular = reg

    def state_at(self, t):
        """Index of the price in force at unix time t, or -1 if unknown."""
        return int(self.ts.searchsorted(t, side="right")) - 1

    def mean_ratio(self, a, b):
        """Time-weighted average price ratio over [a, b] (unix seconds); NaN if unknown."""
        i = self.state_at(a)
        if i < 0:
            return float("nan")
        j = int(self.ts.searchsorted(b, side="right"))
        knots = [a] + [int(x) for x in self.ts[i + 1 : j]] + [b]
        vals = self.ratio[i:j]
        dur = [max(knots[k + 1] - knots[k], 0) for k in range(len(vals))]
        total = sum(dur)
        return (
            float(sum(v * d for v, d in zip(vals, dur)) / total)
            if total > 0
            else float(vals[-1])
        )

    def max_cut(self, a, b):
        """Deepest discount in force at any time in [a, b); NaN if unknown."""
        i = self.state_at(a)
        if i < 0:
            return float("nan")
        j = int(self.ts.searchsorted(b, side="left"))
        return float(self.cut[i : max(j, i + 1)].max())

    def history_features(self, t, days=365):
        """Discount history in the `days` before unix time t (all strictly before t)."""
        lo = t - days * 86400
        i0 = int(self.ts.searchsorted(lo, side="right")) - 1
        i1 = int(self.ts.searchsorted(t, side="left"))
        if i1 <= 0:
            return {
                "disc_starts_365": np.nan,
                "mean_cut_365": np.nan,
                "share_days_disc_365": np.nan,
                "min_ratio_365": np.nan,
                "max_cut_ever": np.nan,
            }
        idx = range(max(i0, 0), i1)
        starts = [
            k
            for k in idx
            if self.cut[k] > 0 and (k == 0 or self.cut[k - 1] == 0) and self.ts[k] >= lo
        ]
        knots = (
            [max(lo, int(self.ts[max(i0, 0)]))]
            + [int(self.ts[k]) for k in range(max(i0, 0) + 1, i1)]
            + [t]
        )
        cuts = self.cut[max(i0, 0) : i1]
        dur = np.diff(knots)
        covered = dur.sum()
        return {
            "disc_starts_365": len(starts),
            "mean_cut_365": float(np.mean([self.cut[k] for k in starts]))
            if starts
            else 0.0,
            "share_days_disc_365": float((dur * (cuts > 0)).sum() / covered)
            if covered > 0
            else np.nan,
            "min_ratio_365": float(np.nanmin(self.ratio[max(i0, 0) : i1])),
            "max_cut_ever": float(self.cut[:i1].max()),
        }


def price_arrays(hist):
    """{app_id: PriceArrays} for every game in a price history table."""
    return {int(a): PriceArrays(g) for a, g in hist.groupby("app_id")}
