"""Double machine learning estimates of the effect of discounting during a sale event.

Partially linear model (PLR) and doubly robust AIPW, both cross-fitted with folds split
by game (Chernozhukov et al. 2018; Athey and Imbens 2019, sec. 6.2). Standard errors are
clustered by game. Effects are on a log scale: exp(effect) - 1 is roughly the % change.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold

from auxiliary import settings


def _reg():
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=0,
    )


def _clf():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=0,
    )


def features(df, extra=()):
    """Controls, all measured before the event, plus event indicators."""
    x = pd.DataFrame(index=df.index)
    x["log_pre28"] = np.log1p(df["pre_rate_28"])
    x["log_pre60"] = np.log1p(df["pre_rate_60"])
    x["pre_trend"] = df["pre_trend"]
    x["pct_pos_60"] = df["pct_pos_60"].fillna(df["pct_pos_60"].median())
    x["pct_pos_missing"] = df["pct_pos_60"].isna().astype(int)
    x["log_age"] = np.log1p(df["age_days"].clip(lower=0))
    x["log_price"] = np.log(df["regular_usd"].clip(lower=0.5))
    days = df["days_since_discount"].fillna(2000).clip(upper=2000)
    x["log_days_since_disc"] = np.log1p(days)
    x["never_discounted"] = df["never_discounted_before"].astype(int)
    for c in extra:
        x[c] = df[c].astype(float)
    x = x.join(pd.get_dummies(df["sale"], prefix="ev", dtype=float))
    return x.to_numpy(dtype=float)


def _crossfit(x, y, groups, model, binary=False):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(settings.FOLDS).split(x, y, groups):
        m = model().fit(x[tr], y[tr])
        pred[te] = m.predict_proba(x[te])[:, 1] if binary else m.predict(x[te])
    return pred


def cluster_se(psi, groups, denom):
    """Cluster-robust standard error for a mean-zero score."""
    sums = pd.Series(psi).groupby(np.asarray(groups)).sum().to_numpy()
    return np.sqrt((sums**2).sum()) / denom


def plr(df, y_col, d_col, extra=()):
    """Partially linear model: effect of d_col on y_col. Returns (estimate, se)."""
    x = features(df, extra)
    y, d = df[y_col].to_numpy(float), df[d_col].to_numpy(float)
    g = df["app_id"].to_numpy()
    y_res = y - _crossfit(x, y, g, _reg)
    d_res = d - _crossfit(x, d, g, _reg)
    theta = (d_res * y_res).sum() / (d_res**2).sum()
    psi = (y_res - theta * d_res) * d_res
    return theta, cluster_se(psi, g, (d_res**2).sum())


def aipw(df, y_col):
    """Doubly robust ATE and ATT for `discounted`. Returns a dict with scores."""
    x = features(df)
    y, d = df[y_col].to_numpy(float), df["discounted"].to_numpy(int)
    g = df["app_id"].to_numpy()
    e = np.clip(_crossfit(x, d, g, _clf, binary=True), settings.CLIP, 1 - settings.CLIP)
    mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(settings.FOLDS).split(x, y, g):
        t1, t0 = tr[d[tr] == 1], tr[d[tr] == 0]
        mu1[te] = _reg().fit(x[t1], y[t1]).predict(x[te])
        mu0[te] = _reg().fit(x[t0], y[t0]).predict(x[te])
    phi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    ate = phi.mean()
    att = (d * (y - mu0) - (1 - d) * e / (1 - e) * (y - mu0)).sum() / d.sum()
    psi_att = (d * (y - mu0 - att) - (1 - d) * e / (1 - e) * (y - mu0)) / d.mean()
    return {
        "ate": ate,
        "se_ate": cluster_se(phi - ate, g, len(phi)),
        "att": att,
        "se_att": cluster_se(psi_att, g, len(psi_att)),
        "phi": phi,
        "propensity": e,
    }


def result_row(label, est, se, df):
    return {
        "specification": label,
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "pct_change": np.expm1(est),
        "n": len(df),
        "games": df["app_id"].nunique(),
    }


def group_tests(phi, labels, clusters):
    """Joint Wald test that all groups in a dimension have the same effect, plus the gap
    between the largest and smallest group, with game-clustered covariance."""
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
    c_mat = np.hstack([np.ones((k - 1, 1)), -np.eye(k - 1)])
    diff = c_mat @ taus
    wald = float(diff @ np.linalg.solve(c_mat @ v @ c_mat.T, diff))
    i, j = int(np.argmax(taus)), int(np.argmin(taus))
    return {
        "largest": cats[i],
        "smallest": cats[j],
        "gap": taus[i] - taus[j],
        "gap_se": float(np.sqrt(v[i, i] + v[j, j] - 2 * v[i, j])),
        "p_value": float(stats.chi2.sf(wald, k - 1)),
    }


def holm(pvals):
    """Holm adjustment for testing several dimensions at once."""
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def heterogeneity_groups(df):
    """The three pre-specified dimensions."""
    return {
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


def main_estimates(main_df):
    """Main effect three ways. Returns (table, aipw result)."""
    rows = [result_row("PLR (main)", *plr(main_df, "y_during", "discounted"), main_df)]
    a = aipw(main_df, "y_during")
    rows.append(result_row("AIPW, all games (ATE)", a["ate"], a["se_ate"], main_df))
    rows.append(
        result_row("AIPW, games that discounted (ATT)", a["att"], a["se_att"], main_df)
    )
    return pd.DataFrame(rows), a


def timing_estimates(main_df):
    """Before, during, after and net: anticipation, persistence, pulled-forward demand."""
    rows = []
    for label, y in [
        ("Week before the sale (placebo / waiting)", "y_pre_week"),
        ("During the sale", "y_during"),
        ("14 days after the sale ends", "y_after"),
    ]:
        rows.append(result_row(label, *plr(main_df, y, "discounted"), main_df))
    long_df = main_df[main_df["coverage_long"]].dropna(
        subset=["y_net", "y_wk3_4", "y_wk5_6"]
    )
    long_df = long_df.reset_index(drop=True)
    for label, y in [
        ("Weeks 3-4 after the sale ends", "y_wk3_4"),
        ("Weeks 5-6 after the sale ends", "y_wk5_6"),
        ("NET: 7 days before to 4 weeks after", "y_net"),
    ]:
        rows.append(result_row(label, *plr(long_df, y, "discounted"), long_df))
    return pd.DataFrame(rows)


def robustness(events, main_df, propensity):
    """Alternative samples and specifications for the effect during the sale."""
    rows = []
    lo, hi = settings.OVERLAP_TRIM
    trimmed = main_df[(propensity >= lo) & (propensity <= hi)].reset_index(drop=True)
    if trimmed["discounted"].sum() >= 50 and (~trimmed["discounted"]).sum() >= 50:
        rows.append(
            result_row(
                f"Overlap-trimmed sample (propensity {lo}-{hi}), PLR",
                *plr(trimmed, "y_during", "discounted"),
                trimmed,
            )
        )
    for label, seasons in [
        ("Spring and Autumn sales only", ["spring", "autumn"]),
        ("Summer and Winter sales only", ["summer", "winter"]),
    ]:
        sub = main_df[main_df["season"].isin(seasons)].reset_index(drop=True)
        rows.append(result_row(label, *plr(sub, "y_during", "discounted"), sub))
    strict = main_df[~main_df["pre_discount_strict"]].reset_index(drop=True)
    rows.append(
        result_row(
            "Stricter exclusion: no discount in the 30 days before",
            *plr(strict, "y_during", "discounted"),
            strict,
        )
    )
    base = (
        events["coverage_ok"]
        & events["price_known"]
        & ~events["too_new"]
        & ~events["free"]
    )
    with_pre = (
        events[base].dropna(subset=["y_during", "pre_trend"]).reset_index(drop=True)
    )
    rows.append(
        result_row(
            "Including games discounted just before (as a control)",
            *plr(with_pre, "y_during", "discounted", extra=("pre_discount",)),
            with_pre,
        )
    )
    depth = main_df.assign(cut_frac=main_df["cut"].fillna(0) / 100)
    th, se = plr(depth, "y_during", "cut_frac")
    rows.append(
        result_row("Discount depth: per 10 points of discount", th / 10, se / 10, depth)
    )
    return pd.DataFrame(rows)


def heterogeneity(main_df, phi):
    """Group effects from the AIPW scores, and tests of whether groups differ."""
    groups = heterogeneity_groups(main_df)
    g_all = main_df["app_id"].to_numpy()
    rows, tests = [], []
    for dim, labels in groups.items():
        for lab in labels.cat.categories:
            m = (labels == lab).to_numpy()
            est = phi[m].mean()
            se = cluster_se(phi[m] - est, g_all[m], m.sum())
            rows.append(result_row(f"{dim}: {lab}", est, se, main_df[m]))
        t = group_tests(phi, labels, g_all)
        t["dimension"] = dim
        tests.append(t)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_value"].to_numpy())
    cols = ["dimension", "largest", "smallest", "gap", "gap_se", "p_value", "p_holm"]
    return pd.DataFrame(rows), tests[cols]


def tidy(table):
    """Readable version of a results table."""
    t = table.copy()
    t["estimate (se)"] = t.apply(
        lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
    )
    t["95% CI"] = t.apply(
        lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1
    )
    t["~% change"] = t["pct_change"].map(lambda v: f"{v:+.0%}")
    return t[["specification", "estimate (se)", "95% CI", "~% change", "n", "games"]]
