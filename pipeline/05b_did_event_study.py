"""Step 5b: event study comparing discounted with non-discounted games in the same sale event.

For each game-event in the analysis sample, daily reviews are put on a log scale and
measured relative to that game's own pre-event baseline (days -21 to -8):

    change_k = log(1 + reviews on day k) - mean of log(1 + reviews) over days -21..-8

Within each sale event, the average change of discounted games minus the average change
of non-discounted games gives the difference for day k. Event-wide traffic (everyone
browsing during a sale) affects both groups, so it cancels. The figure averages these
differences across events, with 95% intervals from a bootstrap that resamples games.

Before day 0 the line should be near zero (no difference before the sale). A clear jump
from day 0 is the discount effect on review activity; values below zero after the sale
would suggest purchases were pulled forward.

Interpretation: this figure shows the SHAPE over time (when the effect starts, how long it
lasts, whether it reverses). Because single days have few reviews, the daily log(1 + x)
scale understates the size; use 07_dml.py for the size of the effect.

  python 05b_did_event_study.py
Outputs: output/did_event_study.png, output/did_event_study.csv
"""

import matplotlib

matplotlib.use("Agg")
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
ev = pd.read_parquet(config.EVENTS_PARQUET)
# Same game-events at every horizon: only those with the full long follow-up, so the line
# does not change composition partway (sales without 6 weeks of follow-up yet are left out).
ev = ev[ev["analysis_sample"] & ev["coverage_long"]][
    ["app_id", "sale", "season", "discounted", "sale_days"]
]
daily = pd.read_parquet(config.EVENTS_DAILY_PARQUET).merge(ev, on=["app_id", "sale"])
daily["lr"] = np.log1p(daily["reviews"])
b0, b1 = config.BASELINE
base = (
    daily[(daily["rel_day"] >= b0) & (daily["rel_day"] <= b1)]
    .groupby(["app_id", "sale"])["lr"]
    .mean()
    .rename("base")
)
daily = daily.join(base, on=["app_id", "sale"])
daily["change"] = daily["lr"] - daily["base"]

# keep events that have both discounted and non-discounted games
mix = ev.groupby("sale")["discounted"].agg(["min", "max"])
keep = mix[(mix["min"] == 0) & (mix["max"] == 1)].index
daily = daily[daily["sale"].isin(keep)]
print(
    f"{len(keep)} sale events with both discounted and non-discounted games; "
    f"{daily[['app_id', 'sale']].drop_duplicates().shape[0]:,} game-events"
)

days = np.arange(config.WINDOW_LONG[0], config.WINDOW_LONG[1] + 1)
cell = (
    daily.groupby(["sale", "discounted", "rel_day", "app_id"])["change"]
    .mean()
    .reset_index()
)


def did_curve(cell_df):
    """Within each event: mean change (discounted) - mean change (not); then average over events,
    weighting each event by its number of game-events."""
    g = (
        cell_df.groupby(["sale", "rel_day", "discounted"])["change"]
        .agg(["mean", "size"])
        .unstack("discounted")
    )
    diff = (g[("mean", True)] - g[("mean", False)]).rename("diff")
    w = (g[("size", True)] + g[("size", False)]).rename("w")
    d = pd.concat([diff, w], axis=1).reset_index().dropna()
    out = d.groupby("rel_day").apply(
        lambda x: np.average(x["diff"], weights=x["w"]), include_groups=False
    )
    return out.reindex(days)


est = did_curve(cell)
apps = cell["app_id"].unique()
by_app = {a: g for a, g in cell.groupby("app_id")}
boots = []
for _ in range(300):
    pick = rng.choice(apps, len(apps), replace=True)
    sample = pd.concat([by_app[a] for a in pick], ignore_index=True)
    boots.append(did_curve(sample).to_numpy())
lo_ci, hi_ci = np.nanpercentile(np.vstack(boots), [2.5, 97.5], axis=0)

pd.DataFrame(
    {"rel_day": days, "diff": est.to_numpy(), "lo": lo_ci, "hi": hi_ci}
).to_csv(config.OUTPUT / "did_event_study.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(days, est.to_numpy(), color="C0", label="Discounted minus not discounted")
ax.fill_between(
    days, lo_ci, hi_ci, color="C0", alpha=0.2, label="95% interval (games resampled)"
)
ax.axvspan(
    0,
    6,
    color="gold",
    alpha=0.15,
    label="Days 0-6 (Summer and Winter sales run longer)",
)
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8, ls=":")
ax.set_xlabel("Days relative to sale start (Pacific time)")
ax.set_ylabel("Difference in log(1 + daily reviews)\nrelative to pre-event baseline")
ax.set_title(
    "Review activity of discounted vs. non-discounted games in the same sale event"
)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(config.OUTPUT / "did_event_study.png", dpi=150)

pre = est.loc[-21:-8].mean()
pre_week = est.loc[-7:-1].mean()
first_week = est.loc[0:6].mean()
late = est.loc[35:56].mean()
print(f"Average difference, days -21 to -8 (baseline, ~0 by construction): {pre:+.3f}")
print(
    f"Average difference, days -7 to -1 (negative = buyers waiting for the sale): {pre_week:+.3f}"
)
print(
    f"Average difference, days 0 to 6:  {first_week:+.3f}  (shape only; see 07_dml.py for the size)"
)
print(
    f"Average difference, days 35 to 56 (negative = pulled-forward demand): {late:+.3f}"
)
print(f"Wrote {config.OUTPUT / 'did_event_study.png'} and did_event_study.csv")
