"""Step 7: the effect of discounting during Steam seasonal sales (batch 5b).

Unit: one game in one seasonal sale. Treatment: discounted during the event.
Outcomes (log scale; exp(effect) - 1 is roughly the % change):
  reviews         change in review activity versus the game's pre-sale baseline
  revenue proxy   the same, with each window's reviews valued at the price actually paid
                  (assumes reviews move with units sold; see the analysis plan for caveats)
Headline (fixed in advance): the ATT, the effect for games that discounted, on the NET
revenue proxy (7 days before to 4 weeks after the sale).

Method (Athey and Imbens 2019): doubly robust AIPW with an ensemble of LASSO, random forest
and boosting, cross-fitted by game and repeated over random splits; the partially linear
model; and approximate residual balancing for the ATT. Identification: unconfoundedness
given pre-event controls, overlap, and no interference between games.

  python 07_dml.py
Outputs (output/): dml_results.csv, nuisance_fit.csv, heterogeneity_tests.csv, dml_overlap.png
Set LEARNER = "boost" and CF_REPEATS = 1 in config.py for a quick run.
"""

import time

import matplotlib

matplotlib.use("Agg")
import config
import dml_core as M
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def group_tests(phi, labels, clusters):
    """Joint Wald test that all groups have the same effect, and the largest-smallest gap."""
    cats = list(labels.cat.categories)
    taus, cols = [], []
    for c in cats:
        m = (labels == c).to_numpy()
        tau = phi[m].mean()
        taus.append(tau)
        cols.append(np.where(m, (phi - tau) / m.sum(), 0.0))
    taus = np.array(taus)
    s = (
        pd.DataFrame(np.column_stack(cols))
        .groupby(np.asarray(clusters))
        .sum()
        .to_numpy()
    )
    v = s.T @ s
    k = len(cats)
    cm = np.hstack([np.ones((k - 1, 1)), -np.eye(k - 1)])
    diff = cm @ taus
    wald = float(diff @ np.linalg.solve(cm @ v @ cm.T, diff))
    i, j = int(np.argmax(taus)), int(np.argmin(taus))
    return {
        "largest": cats[i],
        "smallest": cats[j],
        "gap": taus[i] - taus[j],
        "gap_se": float(np.sqrt(v[i, i] + v[j, j] - 2 * v[i, j])),
        "p_value": float(stats.chi2.sf(wald, k - 1)),
    }


def holm(p):
    order, m, adj, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def groups_for(df):
    g = {
        "Game age": pd.qcut(
            df["age_days"], 3, labels=["newest third", "middle third", "oldest third"]
        ),
        "Regular price": pd.cut(
            df["regular_usd"],
            [0, 9.99, 19.99, np.inf],
            labels=["under $10", "$10-$19.99", "$20+"],
        ),
        "Pre-event review rate": pd.qcut(
            df["pre_rate_28"].rank(method="first"),
            3,
            labels=["lowest third", "middle third", "highest third"],
        ),
    }
    if "band" in df and df["band"].notna().all():
        cats = [
            b
            for b in ["small", "mid", "large", "very_large"]
            if (df["band"] == b).any()
        ]
        g["Size band"] = pd.Series(
            pd.Categorical(df["band"], categories=cats), index=df.index
        )
    return g


def show(rows):
    t = pd.DataFrame(rows)
    t["estimate (se)"] = t.apply(
        lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
    )
    t["95% CI"] = t.apply(
        lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1
    )
    t["~% change"] = t["pct_change"].map(lambda v: f"{v:+.0%}")
    print(
        t[
            [
                "specification",
                "estimand",
                "estimate (se)",
                "95% CI",
                "~% change",
                "n",
                "games",
            ]
        ].to_string(index=False)
    )


def main():
    t0 = time.time()
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    need = list(config.MAIN_OUTCOMES) + ["y_pre_week", "y_after", "pre_trend"]
    missing = [c for c in need if c not in ev]
    if missing:
        raise SystemExit(
            f"events.parquet lacks {missing}. Rebuild it with the current 04b_build_events.py."
        )
    main_df = ev[ev["analysis_sample"]].dropna(
        subset=["y_during", "r_during", "y_pre_week", "y_after", "pre_trend"]
    )
    main_df = main_df.reset_index(drop=True)
    long_df = main_df.dropna(subset=["y_net", "r_net"]).reset_index(drop=True)
    has_w = "weight" in main_df and main_df["weight"].notna().all()
    print(
        f"Analysis sample: {len(main_df):,} game-events, {main_df['app_id'].nunique():,} games, "
        f"{main_df['discounted'].mean():.0%} discounted; {len(long_df):,} with the long follow-up"
    )
    print(
        f"Learner: {config.LEARNER}; cross-fitting repetitions: {config.CF_REPEATS}\n",
        flush=True,
    )
    rows, diag = [], []

    # ---- 1. main estimates: ATT (headline), ATE, ATU, PLR, residual balancing ----
    for df, outs in [
        (main_df, ["y_during", "r_during"]),
        (long_df, ["y_net", "r_net"]),
    ]:
        res, phi, e = M.aipw(df, outs, diag=diag)
        for yc in outs:
            lab = config.MAIN_OUTCOMES[yc] + (
                "   <- HEADLINE" if yc == config.HEADLINE_OUTCOME else ""
            )
            for est in ["ATT", "ATE", "ATU"]:
                rows.append(M.result_row(lab, *res[yc][est], df, est))
            rows.append(
                M.result_row(lab, *M.plr(df, yc, "discounted", diag=diag), df, "PLR")
            )
            att_b, se_b, imb, ess = M.residual_balancing_att(df, yc, diag=diag)
            rows.append(
                M.result_row(lab, att_b, se_b, df, "ATT, residual balancing (L2)")
            )
            print(
                f"  {yc}: done ({(time.time() - t0) / 60:.0f} min); balancing max std. imbalance "
                f"{imb:.3f}, effective control sample {ess:,.0f}",
                flush=True,
            )
        if df is main_df:
            phi_main, e_main = phi, e

    # ---- 2. overlap ----
    d = main_df["discounted"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(e_main[d == 1], bins=40, alpha=0.6, label="discounted", density=True)
    ax.hist(e_main[d == 0], bins=40, alpha=0.6, label="not discounted", density=True)
    ax.set_xlabel("Estimated probability of discounting (propensity score)")
    ax.set_title("Overlap: where both groups appear, they can be compared")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "dml_overlap.png", dpi=150)
    lo, hi = config.OVERLAP_TRIM
    inside = (e_main >= lo) & (e_main <= hi)
    print(
        f"\nOverlap: {inside.mean():.0%} of game-events have propensity in [{lo}, {hi}]; "
        f"comparison group {(d == 0).sum():,} game-events from {main_df.loc[d == 0, 'app_id'].nunique():,} games\n"
    )

    # ---- 3. timing: anticipation, persistence, pulled-forward demand (PLR) ----
    for yc, lab, df in [
        ("y_pre_week", "Reviews, week before the sale (placebo / waiting)", main_df),
        ("y_after", "Reviews, 14 days after the sale", main_df),
        ("y_wk3_4", "Reviews, weeks 3-4 after", long_df),
        ("y_wk5_6", "Reviews, weeks 5-6 after", long_df),
        ("r_after", "Revenue proxy, 14 days after the sale", main_df),
    ]:
        if yc in df:
            sub = df.dropna(subset=[yc]).reset_index(drop=True)
            rows.append(
                M.result_row(lab, *M.plr(sub, yc, "discounted", repeats=1), sub, "PLR")
            )

    # ---- 4. by size band, and weighted to all eligible Steam games ----
    if "band" in main_df and main_df["band"].notna().all():
        for band in [
            b
            for b in ["small", "mid", "large", "very_large"]
            if (long_df["band"] == b).any()
        ]:
            sub = long_df[long_df["band"] == band].reset_index(drop=True)
            if sub["discounted"].sum() >= 30 and (~sub["discounted"]).sum() >= 30:
                r, _, _ = M.aipw(sub, ["r_net", "y_during"], repeats=1)
                rows.append(
                    M.result_row(
                        f"Size band {band}: revenue proxy, net",
                        *r["r_net"]["ATT"],
                        sub,
                        "ATT",
                    )
                )
                rows.append(
                    M.result_row(
                        f"Size band {band}: reviews, during",
                        *r["y_during"]["ATT"],
                        sub,
                        "ATT",
                    )
                )
    if has_w:
        r, _, _ = M.aipw(long_df, ["r_net", "y_during"], weight_col="weight", repeats=1)
        for yc, lab in [
            ("r_net", "WEIGHTED to all eligible games: revenue proxy, net"),
            ("y_during", "WEIGHTED to all eligible games: reviews, during"),
        ]:
            for est in ["ATT", "ATE"]:
                rows.append(M.result_row(lab, *r[yc][est], long_df, est))

    # ---- 4b. ratio (multiplicative) effects: no "+1", so small counts are not compressed ----
    # The log(1 + x) outcomes compress effects toward zero for low-volume games; the ratio
    # estimator compares mean counts directly. It is a ratio of means (busier games weigh
    # more), so it is also reported within each size band, where volumes are similar.
    if "lvl_during" in main_df:
        for df, cols in [
            (
                main_df,
                [
                    ("lvl_during", "Reviews, during"),
                    ("lvl_r_during", "Revenue proxy, during"),
                ],
            ),
            (
                long_df,
                [("lvl_net", "Reviews, net"), ("lvl_r_net", "Revenue proxy, net")],
            ),
        ]:
            for col, lab in cols:
                sub = df.dropna(subset=[col]).reset_index(drop=True)
                rows.append(
                    M.result_row(
                        f"RATIO: {lab}",
                        *M.ratio_att(sub, col, repeats=1),
                        sub,
                        "ATT (ratio)",
                    )
                )
        if "band" in long_df and long_df["band"].notna().all():
            for band in [
                b
                for b in ["small", "mid", "large", "very_large"]
                if (long_df["band"] == b).any()
            ]:
                sub = (
                    long_df[long_df["band"] == band]
                    .dropna(subset=["lvl_r_net"])
                    .reset_index(drop=True)
                )
                if sub["discounted"].sum() >= 30 and (~sub["discounted"]).sum() >= 30:
                    for col, lab in [
                        ("lvl_during", "reviews, during"),
                        ("lvl_r_during", "revenue proxy, during"),
                        ("lvl_r_net", "revenue proxy, net"),
                    ]:
                        rows.append(
                            M.result_row(
                                f"RATIO, size band {band}: {lab}",
                                *M.ratio_att(sub, col, repeats=1),
                                sub,
                                "ATT (ratio)",
                            )
                        )
        # the log version restricted to busier games, where the "+1" matters little
        busy = long_df[long_df["base_rate"] >= 1.0].reset_index(drop=True)
        if busy["discounted"].sum() >= 50 and (~busy["discounted"]).sum() >= 50:
            r, _, _ = M.aipw(busy, ["y_during", "r_during", "r_net"], repeats=1)
            for yc in ["y_during", "r_during", "r_net"]:
                rows.append(
                    M.result_row(
                        f"Busier games only (>= 1 review/day before): {config.MAIN_OUTCOMES[yc]}",
                        *r[yc]["ATT"],
                        busy,
                        "ATT",
                    )
                )
        # overlap-trimmed ratio
        t_lo, t_hi = config.OVERLAP_TRIM
        trim = main_df[(e_main >= t_lo) & (e_main <= t_hi)].reset_index(drop=True)
        if trim["discounted"].sum() >= 50 and (~trim["discounted"]).sum() >= 50:
            rows.append(
                M.result_row(
                    "RATIO, overlap-trimmed: revenue proxy, during",
                    *M.ratio_att(trim, "lvl_r_during", repeats=1),
                    trim,
                    "ATT (ratio)",
                )
            )

    # ---- 4c. Winter Sales that followed Autumn Sales within ~3 weeks (2023, 2024) ----
    # Their baseline window overlaps the end of the Autumn Sale; check without them.
    no_w = long_df[
        ~long_df["sale"].isin(["Winter Sale 2023", "Winter Sale 2024"])
    ].reset_index(drop=True)
    r, _, _ = M.aipw(no_w, ["y_net", "r_net"], repeats=1)
    rows.append(
        M.result_row(
            "Without Winter 2023 and 2024: reviews, net",
            *r["y_net"]["ATT"],
            no_w,
            "ATT",
        )
    )
    rows.append(
        M.result_row(
            "Without Winter 2023 and 2024: revenue proxy, net",
            *r["r_net"]["ATT"],
            no_w,
            "ATT",
        )
    )
    sub = main_df[
        ~main_df["sale"].isin(["Winter Sale 2023", "Winter Sale 2024"])
    ].reset_index(drop=True)
    r, _, _ = M.aipw(sub, ["y_during"], repeats=1)
    rows.append(
        M.result_row(
            "Without Winter 2023 and 2024: reviews, during",
            *r["y_during"]["ATT"],
            sub,
            "ATT",
        )
    )

    # ---- 5. robustness (single repetition each) ----
    trimmed = main_df[inside].reset_index(drop=True)
    if trimmed["discounted"].sum() >= 50 and (~trimmed["discounted"]).sum() >= 50:
        r, _, _ = M.aipw(trimmed, ["y_during", "r_during"], repeats=1)
        for yc in ["y_during", "r_during"]:
            rows.append(
                M.result_row(
                    f"Overlap-trimmed: {config.MAIN_OUTCOMES[yc]}",
                    *r[yc]["ATE"],
                    trimmed,
                    "ATE",
                )
            )
    for lab, seasons in [
        ("Spring and Autumn sales only", ["spring", "autumn"]),
        ("Summer and Winter sales only", ["summer", "winter"]),
    ]:
        sub = main_df[main_df["season"].isin(seasons)].reset_index(drop=True)
        rows.append(
            M.result_row(
                f"{lab}: reviews, during",
                *M.plr(sub, "y_during", "discounted", repeats=1),
                sub,
                "PLR",
            )
        )
    strict = main_df[~main_df["pre_discount_strict"]].reset_index(drop=True)
    rows.append(
        M.result_row(
            "Stricter exclusion (no discount in 30 days before): reviews, during",
            *M.plr(strict, "y_during", "discounted", repeats=1),
            strict,
            "PLR",
        )
    )
    depth = main_df.assign(cut_frac=main_df["cut"].fillna(0) / 100)
    th, se = M.plr(depth, "y_during", "cut_frac", repeats=1)
    rows.append(
        M.result_row(
            "Discount depth: reviews, per 10 points of discount",
            th / 10,
            se / 10,
            depth,
            "PLR",
        )
    )

    # ---- 6. pre-specified heterogeneity (confirmatory), on ATE scores ----
    het = []
    g_all = main_df["app_id"].to_numpy()
    for yc in ["y_during", "r_during"]:
        for dim, labels in groups_for(main_df).items():
            for lab in labels.cat.categories:
                m = (labels == lab).to_numpy()
                est = phi_main[yc][m].mean()
                rows.append(
                    M.result_row(
                        f"Heterogeneity ({config.MAIN_OUTCOMES[yc]}) - {dim}: {lab}",
                        est,
                        M.cluster_se(phi_main[yc][m] - est, g_all[m], m.sum()),
                        main_df[m],
                        "ATE",
                    )
                )
            t = group_tests(phi_main[yc], labels, g_all)
            het.append({"outcome": yc, "dimension": dim, **t})
    het = pd.DataFrame(het)
    het["p_holm"] = np.concatenate(
        [holm(h["p_value"].to_numpy()) for _, h in het.groupby("outcome", sort=False)]
    )

    # ---- save and print ----
    pd.DataFrame(rows).to_csv(config.OUTPUT / "dml_results.csv", index=False)
    pd.DataFrame(diag).to_csv(config.OUTPUT / "nuisance_fit.csv", index=False)
    het.to_csv(config.OUTPUT / "heterogeneity_tests.csv", index=False)
    pd.set_option("display.width", 240, "display.max_colwidth", 90)
    show(rows)
    print("\nHow well the nuisance models predict out of sample (first repetition):")
    print(pd.DataFrame(diag).round(3).to_string(index=False))
    print(
        "\nCaution: group effects by size band and pre-event review rate on the log scale are"
    )
    print(
        "partly driven by compression of small counts; compare the RATIO rows by size band."
    )
    print("\nDo the group effects differ? (Holm-adjusted within each outcome)")
    print(het.round(3).to_string(index=False))
    print(
        f"\nDone in {(time.time() - t0) / 60:.0f} minutes. Wrote dml_results.csv, nuisance_fit.csv, "
        f"heterogeneity_tests.csv and dml_overlap.png in {config.OUTPUT}"
    )


if __name__ == "__main__":
    main()
