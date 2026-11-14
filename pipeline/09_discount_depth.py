"""Step 9 (batch 7b): how deep should a discount be?

Discount depth in four groups (none, 1-33%, 34-60%, 61%+), doubly robust multi-arm effects
(generalized propensity from the classifier ensemble; one outcome model per group):
  A. each depth versus NOT discounting (all game-events)
  B. deeper versus SHALLOW discounts, among discounted game-events only (better overlap:
     most games discount, so comparing depths among discounters needs less extrapolation)
Outcomes: reviews during the sale, revenue proxy during the sale, net revenue proxy, and the
net revenue proxy LAG-ADJUSTED (the 14 days after the sale valued at the sale price; with the
standard version it brackets the effect of review lag).
Two estimands per depth: the average over all game-events (ATE) and the effect for the
game-events that actually chose that depth (ATT).
Also for busier games only (>= BUSY_MIN reviews/day), where the log scale is not compressed.

  python 09_discount_depth.py
Outputs (output/): depth_effects.csv, depth_effects.png
"""

import time

import matplotlib

matplotlib.use("Agg")
import config
import dml_core as M
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTCOMES = {
    "y_during": "Reviews, during",
    "r_during": "Revenue proxy, during",
    "r_net": "Revenue proxy, net",
    "r_net_lag": "Revenue proxy, net, lag-adjusted",
}


def main():
    t0 = time.time()
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    if "r_net_lag" not in ev:
        raise SystemExit(
            "events.parquet lacks r_net_lag. Rebuild it with the current 04b_build_events.py."
        )
    base = ev[ev["analysis_sample"]].dropna(
        subset=["y_during", "r_during", "r_net", "r_net_lag", "pre_trend", "depth_bin"]
    )
    base = base.reset_index(drop=True)
    print(
        "Game-events by depth group:",
        base["depth_bin"].value_counts().reindex(config.DEPTH_ARMS).to_dict(),
    )
    rows = []
    for sname, df in [
        ("all game-events", base),
        (
            "busier games",
            base[base["base_rate"] >= config.BUSY_MIN].reset_index(drop=True),
        ),
    ]:
        disc = df[df["depth_bin"] != "none"].reset_index(drop=True)
        for comp, sub, arms, ref in [
            ("A. vs not discounting", df, config.DEPTH_ARMS, "none"),
            ("B. vs shallow (discounted only)", disc, config.DEPTH_ARMS[1:], "1-33%"),
        ]:
            for yc, lab in OUTCOMES.items():
                eff, means, low, _, att = M.multiarm(
                    sub, yc, "depth_bin", arms, ref, return_att=True
                )
                for estimand, table in [
                    ("ATE (all game-events)", eff),
                    ("ATT (those choosing this depth)", att),
                ]:
                    for a, (est, se) in table.items():
                        rows.append(
                            {
                                "sample": sname,
                                "comparison": comp,
                                "outcome": lab,
                                "estimand": estimand,
                                "depth": a,
                                "versus": ref,
                                "estimate": est,
                                "se": se,
                                "ci_low": est - 1.96 * se,
                                "ci_high": est + 1.96 * se,
                                "pct_change": float(np.expm1(est)),
                                "n": len(sub),
                                "games": sub["app_id"].nunique(),
                                "share_propensity_below_1pct": low,
                            }
                        )
                print(
                    f"  {sname} | {comp} | {lab}: "
                    + "; ".join(
                        f"{a} ATE {eff[a][0]:+.3f} / ATT {att[a][0]:+.3f}" for a in eff
                    )
                    + f"   [{(time.time() - t0) / 60:.0f} min]",
                    flush=True,
                )
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT / "depth_effects.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    for ax, lab in zip(axes, OUTCOMES.values()):
        for sname, mk in [("all game-events", "o"), ("busier games", "s")]:
            t = res[
                (res["outcome"] == lab)
                & (res["sample"] == sname)
                & (res["comparison"] == "A. vs not discounting")
                & (res["estimand"] == "ATE (all game-events)")
            ]
            xs = [config.DEPTH_ARMS.index(d) for d in t["depth"]]
            ax.errorbar(
                [v + (0.08 if mk == "s" else -0.08) for v in xs],
                t["estimate"],
                yerr=1.96 * t["se"],
                fmt=mk,
                capsize=4,
                label=sname,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels(config.DEPTH_ARMS[1:])
        ax.set_title(f"{lab}: vs not discounting")
    axes[0].set_ylabel("Effect (log scale)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "depth_effects.png", dpi=150)

    t = res.copy()
    t["estimate (se)"] = t.apply(
        lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
    )
    t["~% change"] = t["pct_change"].map(lambda v: f"{v:+.0%}")
    pd.set_option("display.width", 220)
    print()
    print(
        t[
            [
                "sample",
                "comparison",
                "outcome",
                "estimand",
                "depth",
                "estimate (se)",
                "~% change",
                "n",
                "share_propensity_below_1pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(
        f"\nDone in {(time.time() - t0) / 60:.0f} minutes. Wrote depth_effects.csv and depth_effects.png"
    )


if __name__ == "__main__":
    main()
