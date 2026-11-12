"""Figures for the notebook."""

import matplotlib.pyplot as plt


def plot_event_study(curve, n):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        curve["rel_day"],
        curve["diff"],
        color="C0",
        label="Discounted minus not discounted",
    )
    ax.fill_between(
        curve["rel_day"],
        curve["lo"],
        curve["hi"],
        color="C0",
        alpha=0.2,
        label="95% interval (games resampled)",
    )
    ax.axvspan(
        0, 6, color="gold", alpha=0.15, label="Days 0-6 (Summer and Winter run longer)"
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Days relative to sale start (Pacific time)")
    ax.set_ylabel(
        "Difference in log(1 + daily reviews)\nrelative to pre-event baseline"
    )
    ax.set_title(
        f"Discounted vs. non-discounted games in the same sale event ({n:,} game-events)"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_overlap(propensity, discounted):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(
        propensity[discounted == 1],
        bins=40,
        alpha=0.6,
        label="discounted",
        density=True,
    )
    ax.hist(
        propensity[discounted == 0],
        bins=40,
        alpha=0.6,
        label="not discounted",
        density=True,
    )
    ax.set_xlabel("Estimated probability of discounting (propensity score)")
    ax.set_title("Overlap: where both groups appear, they can be compared")
    ax.legend()
    fig.tight_layout()
    return fig
