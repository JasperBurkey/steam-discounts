"""Step 7: double machine learning estimate of the effect of discounting during a sale event.

Unit: one game in one seasonal sale event (analysis sample from 04b).
Treatment D: the game was discounted during the event.
Outcome Y: change in log(1 + reviews, as a 14-day-equivalent total) during the sale vs. the
           pre-event baseline (days -21 to -8).
Controls X (all measured before the event): review rate and trend, % positive, game age,
regular price, days since last discount, plus event and season indicators.

Two estimators, both with 5-fold cross-fitting where folds split by GAME, so no game's
own data is used to predict itself (Chernozhukov et al. 2018; Athey and Imbens 2019, sec. 6.2):

  PLR   partially linear model (Robinson-style residual-on-residual regression)
  AIPW  doubly robust average treatment effect, using the influence function in
        Athey and Imbens' Equation 3; its scores also give the group (heterogeneity) effects

Standard errors are clustered by game. Effects are on the log scale: exp(effect) - 1 is
roughly the % change in daily reviews.

Identifying assumption: after conditioning on X, whether a game discounted is unrelated
to how its reviews would otherwise have moved (unconfoundedness). The pre-week placebo and
the overlap check below are evidence about it, not proof.

  python 07_dml.py
Also: the net effect over a longer window (review lag and pulled-forward demand), an
overlap-trimmed estimate, and formal tests of whether the group effects differ.

Outputs: output/dml_results.csv, output/heterogeneity_tests.csv, output/dml_overlap.png
"""

import matplotlib

matplotlib.use("Agg")
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold

FOLDS = 5
CLIP = 0.02  # propensity scores are kept within [0.02, 0.98]


def reg():
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=0,
    )


def clf():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=0,
    )


def features(df, extra=()):
    X = pd.DataFrame(index=df.index)
    X["log_pre28"] = np.log1p(df["pre_rate_28"])
    X["log_pre60"] = np.log1p(df["pre_rate_60"])
    X["pre_trend"] = df["pre_trend"]
    X["pct_pos_60"] = df["pct_pos_60"].fillna(df["pct_pos_60"].median())
    X["pct_pos_missing"] = df["pct_pos_60"].isna().astype(int)
    X["log_age"] = np.log1p(df["age_days"].clip(lower=0))
    X["log_price"] = np.log(df["regular_usd"].clip(lower=0.5))
    X["log_days_since_disc"] = np.log1p(
        df["days_since_discount"].fillna(2000).clip(upper=2000)
    )
    X["never_discounted"] = df["never_discounted_before"].astype(int)
    for c in extra:
        X[c] = df[c].astype(float)
    X = X.join(pd.get_dummies(df["sale"], prefix="ev", dtype=float))
    return X.to_numpy(dtype=float)


def crossfit(X, y, groups, model, binary=False):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(FOLDS).split(X, y, groups):
        m = model().fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if binary else m.predict(X[te])
    return pred


def cluster_se(psi, groups, denom):
    """Cluster-robust standard error for a mean-zero score psi."""
    sums = pd.Series(psi).groupby(np.asarray(groups)).sum().to_numpy()
    return np.sqrt((sums**2).sum()) / denom


def plr(df, y_col, d_col, extra=()):
    X, y, d, g = (
        features(df, extra),
        df[y_col].to_numpy(float),
        df[d_col].to_numpy(float),
        df["app_id"].to_numpy(),
    )
    y_res = y - crossfit(X, y, g, reg)
    d_res = d - crossfit(X, d, g, reg)
    theta = (d_res * y_res).sum() / (d_res**2).sum()
    psi = (y_res - theta * d_res) * d_res
    se = cluster_se(psi, g, (d_res**2).sum())
    return theta, se


def aipw(df, y_col, extra=()):
    X, y, d, g = (
        features(df, extra),
        df[y_col].to_numpy(float),
        df["discounted"].to_numpy(int),
        df["app_id"].to_numpy(),
    )
    e = np.clip(crossfit(X, d, g, clf, binary=True), CLIP, 1 - CLIP)
    mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(FOLDS).split(X, y, g):
        t1, t0 = tr[d[tr] == 1], tr[d[tr] == 0]
        mu1[te] = reg().fit(X[t1], y[t1]).predict(X[te])
        mu0[te] = reg().fit(X[t0], y[t0]).predict(X[te])
    phi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    ate = phi.mean()
    se = cluster_se(phi - ate, g, len(phi))
    # effect for games that DID discount (ATT), doubly robust
    p = d.mean()
    att = (d * (y - mu0) - (1 - d) * e / (1 - e) * (y - mu0)).sum() / d.sum()
    psi_att = (d * (y - mu0 - att) - (1 - d) * e / (1 - e) * (y - mu0)) / p
    se_att = cluster_se(psi_att, g, len(psi_att))
    return ate, se, phi, e, att, se_att


def group_tests(phi, labels, clusters):
    """Do the group effects differ? Joint Wald test that all groups are equal, plus the gap
    between the largest and smallest group, with game-clustered covariance. Games can appear
    in different groups in different events, so the group estimates are not independent."""
    cats = list(labels.cat.categories)
    taus, cols = [], []
    for c in cats:
        m = (labels == c).to_numpy()
        tau = phi[m].mean()
        taus.append(tau)
        cols.append(np.where(m, (phi - tau) / m.sum(), 0.0))
    taus = np.array(taus)
    S = (
        pd.DataFrame(np.column_stack(cols))
        .groupby(np.asarray(clusters))
        .sum()
        .to_numpy()
    )
    V = S.T @ S
    k = len(cats)
    C = np.hstack([np.ones((k - 1, 1)), -np.eye(k - 1)])  # tau_1 - tau_j = 0 for all j
    diff = C @ taus
    wald = float(diff @ np.linalg.solve(C @ V @ C.T, diff))
    p = float(stats.chi2.sf(wald, k - 1))
    i, j = int(np.argmax(taus)), int(np.argmin(taus))
    gap = taus[i] - taus[j]
    gap_se = float(np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j]))
    return {
        "largest": cats[i],
        "smallest": cats[j],
        "gap": gap,
        "gap_se": gap_se,
        "wald": wald,
        "df": k - 1,
        "p_value": p,
    }


def holm(pvals):
    """Holm adjustment for testing several dimensions at once."""
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def row(label, est, se, n, n_games):
    return {
        "specification": label,
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "pct_change": np.expm1(est),
        "n": n,
        "games": n_games,
    }


def main():
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    main_df = (
        ev[ev["analysis_sample"]]
        .dropna(subset=["y_during", "y_after", "y_pre_week", "pre_trend"])
        .reset_index(drop=True)
    )
    print(
        f"Analysis sample: {len(main_df):,} game-events, {main_df['app_id'].nunique()} games, "
        f"{main_df['discounted'].mean():.0%} discounted\n"
    )
    out = []

    # ---- main estimates ----
    th, se = plr(main_df, "y_during", "discounted")
    out.append(
        row(
            "PLR: discounted -> reviews during sale (MAIN)",
            th,
            se,
            len(main_df),
            main_df["app_id"].nunique(),
        )
    )
    ate, se_a, phi, e, att, se_att = aipw(main_df, "y_during")
    out.append(
        row(
            "AIPW: same, doubly robust (average over all games)",
            ate,
            se_a,
            len(main_df),
            main_df["app_id"].nunique(),
        )
    )
    out.append(
        row(
            "AIPW: effect for games that discounted (ATT)",
            att,
            se_att,
            len(main_df),
            main_df["app_id"].nunique(),
        )
    )

    # ---- overlap ----
    d = main_df["discounted"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(e[d == 1], bins=40, alpha=0.6, label="discounted", density=True)
    ax.hist(e[d == 0], bins=40, alpha=0.6, label="not discounted", density=True)
    ax.set_xlabel("Estimated probability of discounting (propensity score)")
    ax.set_title("Overlap: both groups should span similar ranges")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "dml_overlap.png", dpi=150)
    extreme = ((e <= CLIP + 1e-9) | (e >= 1 - CLIP - 1e-9)).mean()
    print(
        f"Overlap: {extreme:.1%} of propensity scores at the clipping bounds "
        f"(high values mean discounters and non-discounters are hard to compare)"
    )
    print(
        f"Comparison group: {(d == 0).sum():,} non-discounted game-events "
        f"from {main_df.loc[d == 0, 'app_id'].nunique()} games\n"
    )

    # ---- checks and robustness ----
    for label, y_col in [
        (
            "Placebo: days -7..-1 before sale (should be ~0, or negative if buyers wait)",
            "y_pre_week",
        ),
        (
            "After sale: 14 days after it ends (negative = pulled-forward demand)",
            "y_after",
        ),
    ]:
        th, se = plr(main_df, y_col, "discounted")
        out.append(row(label, th, se, len(main_df), main_df["app_id"].nunique()))

    long_df = (
        main_df[main_df["coverage_long"]]
        .dropna(subset=["y_net", "y_wk3_4", "y_wk5_6"])
        .reset_index(drop=True)
    )
    for label, y_col in [
        (
            f"NET effect: total reviews from {config.CUM_PRE_DAYS} days before to "
            f"{config.CUM_POST_DAYS} days after the sale ends",
            "y_net",
        ),
        ("Weeks 3-4 after the sale ends (negative = pulled-forward demand)", "y_wk3_4"),
        ("Weeks 5-6 after the sale ends (negative = pulled-forward demand)", "y_wk5_6"),
    ]:
        th, se = plr(long_df, y_col, "discounted")
        out.append(row(label, th, se, len(long_df), long_df["app_id"].nunique()))

    # ---- overlap-trimmed: only game-events whose propensity is inside the trim range ----
    t_lo, t_hi = config.OVERLAP_TRIM
    trimmed = main_df[(e >= t_lo) & (e <= t_hi)].reset_index(drop=True)
    n_t, n_c = int(trimmed["discounted"].sum()), int((~trimmed["discounted"]).sum())
    if n_t >= 50 and n_c >= 50:
        th, se = plr(trimmed, "y_during", "discounted")
        out.append(
            row(
                f"Overlap-trimmed (propensity {t_lo}-{t_hi}; {n_t} discounted, {n_c} not): PLR",
                th,
                se,
                len(trimmed),
                trimmed["app_id"].nunique(),
            )
        )
        ate_t, se_t, *_ = aipw(trimmed, "y_during")
        out.append(
            row(
                "Overlap-trimmed: AIPW",
                ate_t,
                se_t,
                len(trimmed),
                trimmed["app_id"].nunique(),
            )
        )
    else:
        print(
            f"Overlap-trimmed sample too small ({n_t} discounted, {n_c} not); skipped."
        )

    for label, seasons in [
        ("Spring and Autumn sales only", ["spring", "autumn"]),
        ("Summer and Winter sales only", ["summer", "winter"]),
    ]:
        sub = main_df[main_df["season"].isin(seasons)].reset_index(drop=True)
        th, se = plr(sub, "y_during", "discounted")
        out.append(row(label, th, se, len(sub), sub["app_id"].nunique()))

    strict = main_df[~main_df["pre_discount_strict"]].reset_index(drop=True)
    th, se = plr(strict, "y_during", "discounted")
    out.append(
        row(
            f"Stricter exclusion: no discount in the {config.PRE_DISCOUNT_DAYS_STRICT} days before",
            th,
            se,
            len(strict),
            strict["app_id"].nunique(),
        )
    )

    with_pre = (
        ev[ev["coverage_ok"] & ev["price_known"] & ~ev["too_new"] & ~ev["free"]]
        .dropna(subset=["y_during", "pre_trend"])
        .reset_index(drop=True)
    )
    th, se = plr(with_pre, "y_during", "discounted", extra=("pre_discount",))
    out.append(
        row(
            f"Including games discounted in the {config.PRE_DISCOUNT_DAYS} days before (flag as control)",
            th,
            se,
            len(with_pre),
            with_pre["app_id"].nunique(),
        )
    )

    disc = main_df.assign(cut_frac=main_df["cut"].fillna(0) / 100)
    th, se = plr(disc, "y_during", "cut_frac")
    out.append(
        row(
            "Discount depth: effect of each 10 points of discount",
            th / 10,
            se / 10,
            len(disc),
            disc["app_id"].nunique(),
        )
    )

    # ---- heterogeneity: AIPW scores averaged within pre-specified groups ----
    groups = {
        "Game age": pd.qcut(
            main_df["age_days"],
            3,
            labels=["newest third", "middle third", "oldest third"],
        ),
        "Regular price": pd.cut(
            main_df["regular_usd"],
            [0, 9.99, 19.99, np.inf],
            labels=["under $10", "$10-$19.99", "$20+"],
        ),
        "Pre-event review rate": pd.qcut(
            main_df["pre_rate_28"].rank(method="first"),
            3,
            labels=["lowest third", "middle third", "highest third"],
        ),
    }
    g_all = main_df["app_id"].to_numpy()
    for dim, labels in groups.items():
        for lab in labels.cat.categories:
            m = (labels == lab).to_numpy()
            est = phi[m].mean()
            se = cluster_se(phi[m] - est, g_all[m], m.sum())
            out.append(
                row(
                    f"Heterogeneity - {dim}: {lab}",
                    est,
                    se,
                    int(m.sum()),
                    len(set(g_all[m])),
                )
            )

    tests = []
    for dim, labels in groups.items():
        t = group_tests(phi, labels, g_all)
        t["dimension"] = dim
        tests.append(t)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_value"].to_numpy())
    tests.to_csv(config.OUTPUT / "heterogeneity_tests.csv", index=False)

    res = pd.DataFrame(out)
    res.to_csv(config.OUTPUT / "dml_results.csv", index=False)
    show = res.copy()
    show["estimate"] = show.apply(
        lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
    )
    show["95% CI"] = show.apply(
        lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1
    )
    show["~% change"] = show["pct_change"].map(lambda v: f"{v:+.0%}")
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(
        show[
            ["specification", "estimate", "95% CI", "~% change", "n", "games"]
        ].to_string(index=False)
    )
    print(
        "\nDo the group effects differ? (joint test that all groups in a dimension are equal;"
    )
    print("Holm-adjusted p-values account for testing three dimensions)")
    for _, t in tests.iterrows():
        print(
            f"  {t['dimension']:<22} largest - smallest = {t['gap']:+.3f} ({t['gap_se']:.3f})  "
            f"[{t['largest']} vs {t['smallest']}]  p = {t['p_value']:.3f}, Holm p = {t['p_holm']:.3f}"
        )
    print(
        f"\nWrote {config.OUTPUT / 'dml_results.csv'}, heterogeneity_tests.csv and dml_overlap.png"
    )


if __name__ == "__main__":
    main()
