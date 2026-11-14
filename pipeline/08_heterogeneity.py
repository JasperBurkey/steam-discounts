"""Step 8 (batch 7): which games gain or lose more from discounting?

Honest forest on doubly robust scores + generic ML inference (see hetero.py):
  BLP    is there real heterogeneity at all? (adjusted p-value for the loading)
  GATES  effects by quintile of the predicted effect, and top minus bottom
  CLAN   how the most- and least-affected game-events differ
Outcomes: reviews during the sale (y_during) and the net revenue proxy (r_net, headline).
Samples: MAIN = busier games (>= BUSY_MIN reviews/day before the sale), where the log scale
is not compressed; OVERLAP CHECK = busier games with discount propensity in OVERLAP_TRIM,
where discounted and non-discounted games are directly comparable; SECONDARY = all
game-events (volume-related differences there are partly the compression artifact).
Characteristics are measured before the sale; the CLAN table also shows each group's
average propensity, i.e. how thin the data are where the effect is largest.

  python 08_heterogeneity.py
Quick check: $env:LEARNER="boost"; $env:CF_REPEATS="1"; $env:HET_SPLITS="5"; $env:HET_TREES="50"
Outputs (output/): het_blp.csv, het_gates.csv, het_clan.csv, het_importance.csv, het_gates.png
"""

import time

import matplotlib

matplotlib.use("Agg")
import config
import dml_core as M
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hetero import generic_ml

CLAN_COLS = [
    "age_days",
    "regular_usd",
    "pre_rate_28",
    "pre_trend",
    "pct_pos_60",
    "days_since_discount",
    "disc_starts_365",
    "mean_cut_365",
    "prev_y_during",
    "pub_n_games",
    "cut",
    "price_ratio_sale",
]
OUTCOMES = {
    "y_during": "Reviews, during the sale",
    "r_net": "Revenue proxy, net (headline)",
}


def clan_frame(df):
    c = df[[col for col in CLAN_COLS if col in df]].copy()
    if "band" in df:
        for b in ["small", "mid", "large", "very_large"]:
            c[f"band_{b}"] = (df["band"] == b).astype(float)
    for s in ["spring", "summer", "autumn", "winter"]:
        c[f"season_{s}"] = (df["season"] == s).astype(float)
    return c


def main():
    t0 = time.time()
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    base = (
        ev[ev["analysis_sample"]]
        .dropna(subset=["y_during", "r_net", "pre_trend"])
        .reset_index(drop=True)
    )
    samples = {
        "busier games (main)": base[base["base_rate"] >= config.BUSY_MIN].reset_index(
            drop=True
        ),
        "all game-events": base,
    }
    blp_rows, gates_rows, clan_rows, imp_rows = [], [], [], []
    for sname, df in samples.items():
        print(
            f"\n=== {sname}: {len(df):,} game-events, {df['app_id'].nunique():,} games, "
            f"{df['discounted'].mean():.0%} discounted",
            flush=True,
        )
        _, phi, e = M.aipw(df, list(OUTCOMES))
        x, names = M.features(df, names=True)
        g = df["app_id"].to_numpy()
        cf = clan_frame(df).assign(
            propensity=e
        )  # where in the overlap range each group sits
        runs = [(sname, np.ones(len(df), bool))]
        if sname.startswith("busier"):
            lo, hi = config.OVERLAP_TRIM
            inside = (e >= lo) & (e <= hi)
            runs.append((f"busier games, overlap region ({lo}-{hi})", inside))
            print(
                f"  overlap region: {inside.sum():,} game-events ({inside.mean():.0%}), "
                f"{df.loc[inside, 'app_id'].nunique():,} games",
                flush=True,
            )
        for yc, lab in OUTCOMES.items():
            for rname, m in runs:
                (b, gates, clan), imp = generic_ml(
                    x[m], phi[yc][m], g[m], cf[m].reset_index(drop=True)
                )
                blp_rows.append({"sample": rname, "outcome": lab, **b})
                gates_rows.append(gates.reset_index().assign(sample=rname, outcome=lab))
                clan_rows.append(clan.reset_index().assign(sample=rname, outcome=lab))
                imp_rows.append(
                    pd.DataFrame({"feature": names, "importance": imp}).assign(
                        sample=rname, outcome=lab
                    )
                )
                verdict = (
                    "HETEROGENEITY DETECTED"
                    if b["het_p_adjusted"] < 0.05
                    else "no clear heterogeneity"
                )
                print(
                    f"  [{rname}] {lab}: average {b['ate']:+.3f} ({b['ate_se']:.3f}); heterogeneity loading "
                    f"{b['het_loading']:+.2f} ({b['het_se']:.2f}), adjusted p = {b['het_p_adjusted']:.3f} -> {verdict}"
                    f"   [{(time.time() - t0) / 60:.0f} min]",
                    flush=True,
                )

    blp = pd.DataFrame(blp_rows)
    gates = pd.concat(gates_rows, ignore_index=True)
    clan = pd.concat(clan_rows, ignore_index=True)
    imp = pd.concat(imp_rows, ignore_index=True)
    for name, t in [
        ("het_blp", blp),
        ("het_gates", gates),
        ("het_clan", clan),
        ("het_importance", imp),
    ]:
        t.to_csv(config.OUTPUT / f"{name}.csv", index=False)

    # figure: GATES for the main sample
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lab in zip(axes, OUTCOMES.values()):
        t = gates[
            (gates["sample"] == "busier games (main)")
            & (gates["outcome"] == lab)
            & (gates["quintile"].astype(str) != "top - bottom")
        ]
        q = t["quintile"].astype(int)
        ax.errorbar(q, t["effect"], yerr=1.96 * t["se"], fmt="o", capsize=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Quintile of predicted effect (1 = smallest)")
        ax.set_title(lab)
    axes[0].set_ylabel("Average effect in quintile (log scale)")
    fig.suptitle("Effects by predicted-effect group (GATES), busier games")
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "het_gates.png", dpi=150)

    pd.set_option("display.width", 220)
    print("\nBLP (is there heterogeneity?):")
    print(blp.round(3).to_string(index=False))
    for samp in [s for s in clan["sample"].unique() if s.startswith("busier")]:
        sub = clan[clan["sample"] == samp]
        for lab in OUTCOMES.values():
            t = sub[sub["outcome"] == lab]
            t = pd.concat(
                [
                    t[t["characteristic"] == "propensity"],
                    t[t["characteristic"] != "propensity"]
                    .sort_values("p_adjusted")
                    .head(8),
                ]
            )
            print(
                f"\nCLAN, {samp}, {lab}: most- vs least-affected fifth (propensity, then top 8)"
            )
            print(
                t[
                    [
                        "characteristic",
                        "most_affected",
                        "least_affected",
                        "difference",
                        "p_adjusted",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )
    top = (
        imp[imp["sample"] == "busier games (main)"]
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nForest's most-used characteristics (busier games, both outcomes):")
    print(top.head(10).round(3).to_string())
    print(
        f"\nDone in {(time.time() - t0) / 60:.0f} minutes. Wrote het_*.csv and het_gates.png in {config.OUTPUT}"
    )


if __name__ == "__main__":
    main()
