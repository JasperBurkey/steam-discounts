"""Event study: discounted minus non-discounted games within each sale event."""

import numpy as np
import pandas as pd

from auxiliary import settings


def daily_changes(events, daily):
    """log(1 + reviews) on each day relative to the game's own pre-event baseline.

    Uses game-events in the analysis sample with the full long follow-up, so the same
    game-events appear at every horizon.
    """
    ev = events[events["analysis_sample"] & events["coverage_long"]]
    ev = ev[["app_id", "sale", "discounted"]]
    d = daily.merge(ev, on=["app_id", "sale"])
    d["lr"] = np.log1p(d["reviews"])
    b0, b1 = settings.BASELINE
    base = (
        d[(d["rel_day"] >= b0) & (d["rel_day"] <= b1)]
        .groupby(["app_id", "sale"])["lr"]
        .mean()
        .rename("base")
    )
    d = d.join(base, on=["app_id", "sale"])
    d["change"] = d["lr"] - d["base"]
    mix = ev.groupby("sale")["discounted"].agg(["min", "max"])
    both = mix[(mix["min"] == 0) & (mix["max"] == 1)].index  # events with both groups
    return d[d["sale"].isin(both)]


def did_curve(cells, days):
    """Within each event, mean change (discounted) minus mean change (not), then a
    weighted average over events (weight = game-events in the event)."""
    g = (
        cells.groupby(["sale", "rel_day", "discounted"])["change"]
        .agg(["mean", "size"])
        .unstack("discounted")
    )
    diff = (g[("mean", True)] - g[("mean", False)]).rename("diff")
    w = (g[("size", True)] + g[("size", False)]).rename("w")
    d = pd.concat([diff, w], axis=1).reset_index().dropna()
    out = d.groupby("rel_day")[["diff", "w"]].apply(
        lambda x: np.average(x["diff"], weights=x["w"])
    )
    return out.reindex(days)


def event_study(events, daily):
    """The difference by day with 95% intervals from a bootstrap that resamples games."""
    changes = daily_changes(events, daily)
    days = np.arange(settings.WINDOW[0], settings.WINDOW[1] + 1)
    cells = (
        changes.groupby(["sale", "discounted", "rel_day", "app_id"])["change"]
        .mean()
        .reset_index()
    )
    est = did_curve(cells, days)
    rng = np.random.default_rng(settings.SEED)
    apps = cells["app_id"].unique()
    by_app = {a: g for a, g in cells.groupby("app_id")}
    boots = []
    for _ in range(settings.BOOTSTRAP_REPS):
        pick = rng.choice(apps, len(apps), replace=True)
        sample = pd.concat([by_app[a] for a in pick], ignore_index=True)
        boots.append(did_curve(sample, days).to_numpy())
    lo, hi = np.nanpercentile(np.vstack(boots), [2.5, 97.5], axis=0)
    n = changes[["app_id", "sale"]].drop_duplicates().shape[0]
    return pd.DataFrame(
        {"rel_day": days, "diff": est.to_numpy(), "lo": lo, "hi": hi}
    ), n


def summarize(curve):
    """Average difference over the key windows."""
    c = curve.set_index("rel_day")["diff"]
    return pd.Series(
        {
            "days -21 to -8 (baseline)": c.loc[-21:-8].mean(),
            "days -7 to -1 (waiting for the sale?)": c.loc[-7:-1].mean(),
            "days 0 to 6 (sale)": c.loc[0:6].mean(),
            "days 35 to 56 (pulled-forward demand?)": c.loc[35:56].mean(),
        }
    ).round(3)
