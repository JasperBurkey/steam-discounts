"""Step 5: event study of daily reviews around Valve's seasonal sales.

For every game x sale with full panel coverage, daily reviews are divided by that
game's average over the BASELINE days before the sale, so 1.0 means "the pre-sale
level". The figure averages across all game-sales, with 95% intervals from a
bootstrap that resamples games.

A placebo repeats everything with fake sale dates PLACEBO_SHIFT_DAYS earlier. If the
placebo line also jumps at day 0, the method is picking up something other than sales.

What this measures: the effect of a sale period on review flow, pooled across games
whether or not each game was discounted. It is not the effect of a discount.

Outputs: output/event_study.png, output/event_study.csv
"""

import matplotlib

matplotlib.use("Agg")
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
panel = pd.read_parquet(config.PANEL_PARQUET)
wide = panel.pivot(index="day", columns="app_id", values="reviews")  # NaN = no coverage

sales = pd.read_csv(config.SALES_CSV, parse_dates=["start", "end"])
missing = sales["start"].isna().sum()
if missing:
    print(
        f"Note: {missing} sales in sale_calendar.csv have no dates yet and are skipped."
    )
sales = sales.dropna(subset=["start"])

lo, hi = config.WINDOW
rel_days = np.arange(lo, hi + 1)
b0, b1 = config.BASELINE


def stack_events(shift_days=0):
    """Rows = game-sale events, columns = days relative to the (possibly shifted) start."""
    profiles, apps, seasons = [], [], []
    for s in sales.itertuples():
        start = s.start - pd.Timedelta(days=shift_days)
        days = [start + pd.Timedelta(days=int(k)) for k in rel_days]
        if days[0] < wide.index.min() or days[-1] > wide.index.max():
            continue
        block = wide.loc[days]  # window x apps
        block = block.loc[:, block.notna().all()]  # games covered for the whole window
        base = block.iloc[(b0 - lo) : (b1 - lo + 1)].mean()
        block = block.loc[:, base >= config.MIN_BASELINE_RATE]
        base = base[block.columns]
        if block.empty:
            continue
        profiles.append((block / base).T.to_numpy())
        apps += list(block.columns)
        seasons += [s.season] * block.shape[1]
    if not profiles:
        return None
    y = np.vstack(profiles)
    y = np.minimum(y, np.nanpercentile(y, 99))  # cap extreme spikes
    return y, np.array(apps), np.array(seasons)


def mean_and_ci(y, apps, reps=500):
    """Average profile with a bootstrap that resamples games (all of a game's sales together)."""
    uniq = np.unique(apps)
    sums = np.vstack([y[apps == a].sum(0) for a in uniq])
    counts = np.array([(apps == a).sum() for a in uniq])
    est = sums.sum(0) / counts.sum()
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots.append(sums[pick].sum(0) / counts[pick].sum())
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5], axis=0)
    return est, lo_ci, hi_ci, len(uniq), len(y)


real = stack_events()
if real is None:
    raise SystemExit(
        "No sale has full panel coverage. Check sale dates and the review history window."
    )
y, apps, seasons = real
results = {"All seasonal sales": mean_and_ci(y, apps)}
placebo = stack_events(config.PLACEBO_SHIFT_DAYS)
if placebo is not None:
    results[f"Placebo ({config.PLACEBO_SHIFT_DAYS} days earlier)"] = mean_and_ci(
        *placebo[:2]
    )
for label, group in [
    ("Spring and Autumn sales", ["spring", "autumn"]),
    ("Summer and Winter sales", ["summer", "winter"]),
]:
    m = np.isin(seasons, group)
    if m.sum() > 0:
        results[label] = mean_and_ci(y[m], apps[m])

# --- Table ---
table = pd.DataFrame({"rel_day": rel_days})
for label, (est, lo_ci, hi_ci, n_apps, n_ev) in results.items():
    key = label.split(" (")[0].lower().replace(" ", "_")
    table[f"{key}_est"], table[f"{key}_lo"], table[f"{key}_hi"] = est, lo_ci, hi_ci
table.to_csv(config.OUTPUT / "event_study.csv", index=False)

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
styles = {
    "All seasonal sales": ("C0", "-"),
    f"Placebo ({config.PLACEBO_SHIFT_DAYS} days earlier)": ("grey", "--"),
    "Spring and Autumn sales": ("C1", "-"),
    "Summer and Winter sales": ("C2", "-"),
}
panels = [
    ["All seasonal sales", f"Placebo ({config.PLACEBO_SHIFT_DAYS} days earlier)"],
    ["Spring and Autumn sales", "Summer and Winter sales"],
]
for ax, labels in zip(axes, panels):
    for label in labels:
        if label not in results:
            continue
        est, lo_ci, hi_ci, n_apps, n_ev = results[label]
        color, ls = styles[label]
        ax.plot(
            rel_days,
            est,
            color=color,
            ls=ls,
            label=f"{label} ({n_apps} games, {n_ev} game-sales)",
        )
        ax.fill_between(rel_days, lo_ci, hi_ci, color=color, alpha=0.15)
    ax.axvspan(
        0,
        6,
        color="gold",
        alpha=0.15,
        label="Days 0-6 (all sales; Summer/Winter run longer)",
    )
    ax.axhline(1, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Days relative to sale start (Pacific time)")
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel(f"Daily reviews / pre-sale average (days {b0} to {b1})")
axes[0].set_title("Daily reviews around Steam seasonal sales")
axes[1].set_title("By season (holiday overlap in Summer and Winter)")
fig.tight_layout()
fig.savefig(config.OUTPUT / "event_study.png", dpi=150)

# --- Summary ---
est, lo_ci, hi_ci, n_apps, n_ev = results["All seasonal sales"]
during = (rel_days >= 0) & (rel_days <= 6)
after = (rel_days >= 10) & (rel_days <= 24)
print(f"{n_apps} games, {n_ev} game-sales")
print(f"Average ratio, sale days 0-6:   {est[during].mean():.2f}x pre-sale level")
if "Spring and Autumn sales" in results:
    # Summer and Winter sales last 14+ days, so only 7-day sales have a clean post-period here.
    sa = results["Spring and Autumn sales"][0]
    print(
        f"Spring/Autumn ratio, days 10-24: {sa[after].mean():.2f}x  "
        f"(after these 7-day sales end; below 1 suggests pulled-forward demand)"
    )
if placebo is not None:
    pe = results[f"Placebo ({config.PLACEBO_SHIFT_DAYS} days earlier)"][0]
    print(
        f"Placebo ratio, days 0-6:         {pe[during].mean():.2f}x  (should be near 1)"
    )
print(f"Wrote {config.OUTPUT / 'event_study.png'} and event_study.csv")
