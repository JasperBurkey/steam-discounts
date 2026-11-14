"""Estimation core shared by 07_dml.py and later batches (heterogeneity, policy, spillovers).

Follows Athey and Imbens (2019):
  * Ensemble of learners (sec. 2.7): LASSO, random forest and gradient boosting, combined
    with non-negative weights chosen on out-of-fold predictions (a "super learner").
  * Cross-fitting with folds split by GAME (sec. 6.2), repeated with different random
    splits; estimates are the median across repetitions, and standard errors include the
    spread across splits (Chernozhukov et al. 2018).
  * Doubly robust AIPW scores (sec. 6.1, Eq. 3) for the ATT, ATE and ATU, and the
    partially linear model (PLR).
  * Approximate residual balancing for the ATT (Athey, Imbens and Wager; sec. 6.1), for
    the many-weak-confounders, extreme-propensity case. This version balances covariate
    means with an L2 penalty instead of the original sup-norm, and says so in its label.
Standard errors are clustered by game throughout.
"""

import warnings

import config
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------- controls
BASE_FEATURES = [
    # review history before the event
    "pre_rate_28",
    "pre_rate_60",
    "pre_trend",
    "pct_pos_60",
    # game and price
    "age_days",
    "regular_usd",
    # discount history before the event (price histories)
    "days_since_discount",
    "never_discounted_before",
    "disc_starts_365",
    "mean_cut_365",
    "share_days_disc_365",
    "min_ratio_365",
    "max_cut_ever",
    # behavior and response in earlier sale events
    "prev_discounted",
    "prev_cut",
    "prev_y_during",
    "prev_share_joined",
    # publisher habits in the previous event
    "pub_prev_share",
    "pub_n_games",
]
LOG_FEATURES = {
    "pre_rate_28",
    "pre_rate_60",
    "age_days",
    "regular_usd",
    "days_since_discount",
    "pub_n_games",
}


def features(df, extra=(), names=False):
    """Numeric control matrix: pre-event features, missing-value indicators, event and
    season indicators (and size band when present). Columns are standardized, since LASSO
    needs comparable scales and trees are unaffected."""
    x = pd.DataFrame(index=df.index)
    for c in list(BASE_FEATURES) + list(extra):
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if c in LOG_FEATURES:
            v = np.log1p(v.clip(lower=0))
        if v.isna().any():
            x[c + "_missing"] = v.isna().astype(float)
            v = v.fillna(v.median() if v.notna().any() else 0.0)
        x[c] = v
    x = x.join(pd.get_dummies(df["sale"], prefix="ev", dtype=float))
    if "band" in df and df["band"].notna().all():
        x = x.join(pd.get_dummies(df["band"], prefix="band", dtype=float))
    x = x.loc[:, x.std() > 0]
    arr = ((x - x.mean()) / x.std()).to_numpy(dtype=float)
    return (arr, list(x.columns)) if names else arr


# ---------------------------------------------------------------- learners
def _regressors():
    if config.LEARNER == "boost":
        return {
            "boost": lambda: HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                random_state=0,
            )
        }
    return {
        "lasso": lambda: LassoCV(cv=3, max_iter=5000, random_state=0),
        "forest": lambda: RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=25,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        ),
        "boost": lambda: HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=0,
        ),
    }


def _classifiers():
    if config.LEARNER == "boost":
        return {
            "boost": lambda: HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                random_state=0,
            )
        }
    return {
        "logit": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "forest": lambda: RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=25,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        ),
        "boost": lambda: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=0,
        ),
    }


def _count_regressors():
    """For count outcomes: gradient boosting with a Poisson loss (non-negative predictions,
    multiplicative structure). LASSO and forests with squared loss are poorly suited here."""
    return {
        "boost_poisson": lambda: HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=0,
        )
    }


def game_folds(groups, seed):
    """Random assignment of GAMES to folds (all of a game's events share a fold)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % config.FOLDS))
    return np.array([fold_of[g] for g in groups])


def crossfit(
    x, y, folds, binary=False, w=None, mask=None, diag=None, name="", count=False
):
    """Out-of-fold predictions from the ensemble. `mask` restricts TRAINING rows (e.g.
    controls only, for the untreated outcome model); predictions are made for every row.
    Ensemble weights are non-negative and sum to one, chosen on out-of-fold predictions."""
    learners = (
        _classifiers() if binary else (_count_regressors() if count else _regressors())
    )
    train = np.ones(len(y), bool) if mask is None else mask
    oof = np.zeros((len(y), len(learners)))
    for k in range(config.FOLDS):
        tr, te = (folds != k) & train, folds == k
        for j, make in enumerate(learners.values()):
            m = make().fit(x[tr], y[tr], sample_weight=None if w is None else w[tr])
            oof[te, j] = m.predict_proba(x[te])[:, 1] if binary else m.predict(x[te])
    fit_rows = train
    if oof.shape[1] == 1:
        weights = np.array([1.0])
    else:
        sw = np.ones(len(y)) if w is None else w
        a = oof[fit_rows] * np.sqrt(sw[fit_rows])[:, None]
        b = y[fit_rows] * np.sqrt(sw[fit_rows])
        weights, _ = nnls(a, b)
        weights = (
            weights / weights.sum()
            if weights.sum() > 0
            else np.full(oof.shape[1], 1 / oof.shape[1])
        )
    pred = oof @ weights
    if diag is not None:
        yy, pp = y[fit_rows], pred[fit_rows]
        entry = {
            "model": name,
            "rows": int(fit_rows.sum()),
            **{f"weight_{k}": round(float(v), 3) for k, v in zip(learners, weights)},
        }
        if binary:
            pc = np.clip(pp, 1e-4, 1 - 1e-4)
            entry.update(auc=roc_auc_score(yy, pc), log_loss=log_loss(yy, pc))
        else:
            entry.update(r2_out_of_sample=1 - np.mean((yy - pp) ** 2) / np.var(yy))
        diag.append(entry)
    return pred


# ---------------------------------------------------------------- inference helpers
def cluster_se(psi, groups, denom):
    """Cluster-robust standard error for a mean-zero score, clustered by game."""
    sums = pd.Series(psi).groupby(np.asarray(groups)).sum().to_numpy()
    return float(np.sqrt((sums**2).sum()) / denom)


def median_combine(ests, ses):
    """Median across cross-fitting repetitions; the SE includes the spread across splits."""
    ests, ses = np.asarray(ests), np.asarray(ses)
    m = float(np.median(ests))
    return m, float(np.median(np.sqrt(ses**2 + (ests - m) ** 2)))


def _w(df, weight_col):
    return np.ones(len(df)) if weight_col is None else df[weight_col].to_numpy(float)


# ---------------------------------------------------------------- estimators
def plr(df, y_col, d_col, extra=(), weight_col=None, repeats=None, diag=None):
    """Partially linear model (residual on residual). Returns (estimate, se)."""
    x = features(df, extra)
    y, d, g, w = (
        df[y_col].to_numpy(float),
        df[d_col].to_numpy(float),
        df["app_id"].to_numpy(),
        _w(df, weight_col),
    )
    ests, ses = [], []
    for r in range(repeats or config.CF_REPEATS):
        f = game_folds(g, config.SEED + r)
        sw = None if weight_col is None else w
        yr = y - crossfit(
            x, y, f, w=sw, diag=diag if r == 0 else None, name=f"PLR outcome: {y_col}"
        )
        dr = d - crossfit(
            x, d, f, w=sw, diag=diag if r == 0 else None, name=f"PLR treatment: {d_col}"
        )
        th = (w * dr * yr).sum() / (w * dr**2).sum()
        ests.append(th)
        ses.append(cluster_se(w * (yr - th * dr) * dr, g, (w * dr**2).sum()))
    return median_combine(ests, ses)


def aipw(df, y_cols, weight_col=None, repeats=None, diag=None):
    """Doubly robust ATT, ATE and ATU for `discounted`, for several outcomes at once (the
    propensity model is shared). Returns {outcome: {estimand: (est, se)}}, plus the
    averaged ATE scores and propensities for downstream use (heterogeneity, policy)."""
    x = features(df)
    d, g, w = (
        df["discounted"].to_numpy(int),
        df["app_id"].to_numpy(),
        _w(df, weight_col),
    )
    sw = None if weight_col is None else w
    store = {y: {"ATT": [], "ATE": [], "ATU": []} for y in y_cols}
    phi_sum = {y: np.zeros(len(df)) for y in y_cols}
    e_sum = np.zeros(len(df))
    n_rep = repeats or config.CF_REPEATS
    for r in range(n_rep):
        f = game_folds(g, config.SEED + r)
        e = np.clip(
            crossfit(
                x,
                d.astype(float),
                f,
                binary=True,
                w=sw,
                diag=diag if r == 0 else None,
                name="Propensity (discounted)",
            ),
            config.CLIP,
            1 - config.CLIP,
        )
        e_sum += e
        for yc in y_cols:
            y = df[yc].to_numpy(float)
            mu1 = crossfit(
                x,
                y,
                f,
                w=sw,
                mask=d == 1,
                diag=diag if r == 0 else None,
                name=f"Outcome if discounted: {yc}",
            )
            mu0 = crossfit(
                x,
                y,
                f,
                w=sw,
                mask=d == 0,
                diag=diag if r == 0 else None,
                name=f"Outcome if not: {yc}",
            )
            phi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
            phi_sum[yc] += phi / n_rep
            ate = (w * phi).sum() / w.sum()
            store[yc]["ATE"].append((ate, cluster_se(w * (phi - ate), g, w.sum())))
            att_num = w * (d * (y - mu0) - (1 - d) * e / (1 - e) * (y - mu0))
            att = att_num.sum() / (w * d).sum()
            psi = w * (d * (y - mu0 - att) - (1 - d) * e / (1 - e) * (y - mu0))
            store[yc]["ATT"].append((att, cluster_se(psi, g, (w * d).sum())))
            atu_num = w * ((1 - d) * (mu1 - y) + d * (1 - e) / e * (y - mu1))
            atu = atu_num.sum() / (w * (1 - d)).sum()
            psi = w * ((1 - d) * (mu1 - y - atu) + d * (1 - e) / e * (y - mu1))
            store[yc]["ATU"].append((atu, cluster_se(psi, g, (w * (1 - d)).sum())))
    out = {
        yc: {k: median_combine(*zip(*v)) for k, v in est.items()}
        for yc, est in store.items()
    }
    return out, phi_sum, e_sum / n_rep


def _project_simplex(v):
    """Euclidean projection onto {g >= 0, sum g = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def balancing_weights(xc, target, zeta=0.5, iters=3000):
    """Weights on controls: min zeta*||g||^2 + (1-zeta)*||xc'g - target||^2, g on the simplex."""
    n = xc.shape[0]
    g = np.full(n, 1.0 / n)
    lip = 2 * zeta + 2 * (1 - zeta) * np.linalg.norm(xc, 2) ** 2
    z, t = g.copy(), 1.0
    for _ in range(iters):  # accelerated projected gradient
        grad = 2 * zeta * z + 2 * (1 - zeta) * xc @ (xc.T @ z - target)
        g_new = _project_simplex(z - grad / lip)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = g_new + (t - 1) / t_new * (g_new - g)
        g, t = g_new, t_new
    return g


def residual_balancing_att(df, y_col, weight_col=None, diag=None):
    """Approximate residual balancing for the ATT (L2 variant): treated outcomes minus the
    untreated-outcome model, corrected by a balancing-weighted average of control residuals."""
    x = features(df)
    d, g, w = (
        df["discounted"].to_numpy(int),
        df["app_id"].to_numpy(),
        _w(df, weight_col),
    )
    y = df[y_col].to_numpy(float)
    f = game_folds(g, config.SEED)
    mu0 = crossfit(
        x,
        y,
        f,
        w=None if weight_col is None else w,
        mask=d == 0,
        diag=diag,
        name=f"Balancing: outcome if not discounted: {y_col}",
    )
    t, c = d == 1, d == 0
    target = (w[t, None] * x[t]).sum(0) / w[t].sum()
    gam = balancing_weights(x[c], target)
    r = y - mu0
    a = w[t] / w[t].sum()
    att = (a * r[t]).sum() - (gam * r[c]).sum()
    psi = np.zeros(len(df))
    psi[t] = a * (r[t] - (a * r[t]).sum())
    psi[c] = -gam * (r[c] - (gam * r[c]).sum())
    se = cluster_se(psi, g, 1.0)
    imbalance = float(np.abs(x[c].T @ gam - target).max())
    return att, se, imbalance, float(1 / (gam**2).sum())


def crossfit_multiclass(x, labels, folds, w=None, diag=None, name=""):
    """Out-of-fold class probabilities (n x K) for labels 0..K-1 from the classifier
    ensemble, with non-negative ensemble weights chosen on out-of-fold predictions
    (multi-class Brier score)."""
    learners = _classifiers()
    k = int(labels.max()) + 1
    oof = np.zeros((len(labels), k, len(learners)))
    for f in range(config.FOLDS):
        tr, te = folds != f, folds == f
        for j, make in enumerate(learners.values()):
            m = make().fit(
                x[tr], labels[tr], sample_weight=None if w is None else w[tr]
            )
            p = np.zeros((te.sum(), k))
            p[:, m.classes_.astype(int)] = m.predict_proba(x[te])
            oof[te, :, j] = p
    onehot = np.eye(k)[labels]
    if len(learners) == 1:
        weights = np.array([1.0])
    else:
        a = oof.reshape(-1, len(learners))
        weights, _ = nnls(a, onehot.reshape(-1))
        weights = (
            weights / weights.sum()
            if weights.sum() > 0
            else np.full(len(learners), 1 / len(learners))
        )
    probs = oof @ weights
    probs = probs / probs.sum(1, keepdims=True)
    if diag is not None:
        diag.append(
            {
                "model": name,
                "rows": len(labels),
                **{
                    f"weight_{n}": round(float(v), 3) for n, v in zip(learners, weights)
                },
                "brier": float(((probs - onehot) ** 2).sum(1).mean()),
            }
        )
    return probs


def multiarm(
    df,
    y_col,
    arm_col,
    arms,
    ref,
    weight_col=None,
    repeats=None,
    diag=None,
    clip=0.01,
    return_att=False,
):
    """Doubly robust effects of several treatment levels (e.g. discount-depth groups) versus a
    reference level: E[Y(a)] - E[Y(ref)] for every arm a, averaged over the rows of df.
    Generalized propensity from the classifier ensemble; outcome models per arm; scores
    phi_a = mu_a + 1{D=a}(Y - mu_a)/e_a. Returns ({arm: (est, se)}, mean potential outcomes,
    min propensity share below clip, scores matrix)."""
    x = features(df)
    g, w = df["app_id"].to_numpy(), _w(df, weight_col)
    y = df[y_col].to_numpy(float)
    lab = df[arm_col].map({a: i for i, a in enumerate(arms)}).to_numpy()
    sw = None if weight_col is None else w
    est = {a: [] for a in arms if a != ref}
    att = {a: [] for a in arms if a != ref}  # effect for the rows that chose arm a
    phi_avg = np.zeros((len(df), len(arms)))
    low = []
    n_rep = repeats or config.CF_REPEATS
    for r in range(n_rep):
        f = game_folds(g, config.SEED + r)
        e = crossfit_multiclass(
            x,
            lab,
            f,
            w=sw,
            diag=diag if r == 0 else None,
            name=f"Depth propensity ({y_col})",
        )
        low.append(float((e[np.arange(len(lab)), lab] < clip).mean()))
        e = np.clip(e, clip, 1.0)
        phi = np.zeros((len(df), len(arms)))
        mus = np.zeros((len(df), len(arms)))
        for k, a in enumerate(arms):
            mu = crossfit(
                x,
                y,
                f,
                w=sw,
                mask=lab == k,
                diag=diag if r == 0 else None,
                name=f"Outcome if {a}: {y_col}",
            )
            mus[:, k] = mu
            phi[:, k] = mu + (lab == k) * (y - mu) / e[:, k]
        i_r = arms.index(ref)
        for k, a in enumerate(arms):
            if a == ref:
                continue
            da, dr = (lab == k).astype(float), (lab == i_r).astype(float)
            res = y - mus[:, i_r]
            n_a = (w * da).sum()
            tau = (
                (w * da * res).sum() - (w * dr * e[:, k] / e[:, i_r] * res).sum()
            ) / n_a
            psi = w * (da * (res - tau) - dr * e[:, k] / e[:, i_r] * res)
            att[a].append((tau, cluster_se(psi, g, n_a)))
        phi_avg += phi / n_rep
        i_ref = arms.index(ref)
        for k, a in enumerate(arms):
            if a == ref:
                continue
            dlt = phi[:, k] - phi[:, i_ref]
            m = (w * dlt).sum() / w.sum()
            est[a].append((m, cluster_se(w * (dlt - m), g, w.sum())))
    out = {a: median_combine(*zip(*v)) for a, v in est.items()}
    means = (w[:, None] * phi_avg).sum(0) / w.sum()
    if return_att:
        return (
            out,
            dict(zip(arms, means)),
            float(np.median(low)),
            phi_avg,
            {a: median_combine(*zip(*v)) for a, v in att.items()},
        )
    return out, dict(zip(arms, means)), float(np.median(low)), phi_avg


def ratio_att(df, level_col, weight_col=None, repeats=None, diag=None):
    """Doubly robust RATIO effect for games that discounted, on a count outcome:

        log( E[Y(1) | discounted] / E[Y(0) | discounted] )

    Y is a 14-day-equivalent count (reviews, or revenue-proxy units). The numerator is the
    observed mean for discounted game-events; the denominator is the AIPW estimate of their
    mean without the discount. No "+1" is involved, so small counts are not compressed.
    It is a ratio of MEANS, so busier games carry more weight (within a size band the
    weighting is much more even). Standard errors by the delta method, clustered by game."""
    x = features(df)
    d, g, w = (
        df["discounted"].to_numpy(int),
        df["app_id"].to_numpy(),
        _w(df, weight_col),
    )
    y = df[level_col].to_numpy(float)
    sw = None if weight_col is None else w
    ests, ses = [], []
    for r in range(repeats or config.CF_REPEATS):
        f = game_folds(g, config.SEED + r)
        e = np.clip(
            crossfit(
                x,
                d.astype(float),
                f,
                binary=True,
                w=sw,
                diag=diag if r == 0 else None,
                name="Propensity (ratio estimator)",
            ),
            config.CLIP,
            1 - config.CLIP,
        )
        mu0 = np.maximum(
            crossfit(
                x,
                y,
                f,
                w=sw,
                mask=d == 0,
                count=True,
                diag=diag if r == 0 else None,
                name=f"Ratio: mean count if not discounted: {level_col}",
            ),
            0.0,
        )
        nt = (w * d).sum()
        a = (w * d * y).sum() / nt
        b = ((w * d * mu0).sum() + (w * (1 - d) * e / (1 - e) * (y - mu0)).sum()) / nt
        if a <= 0 or b <= 0:
            continue
        psi_a = w * d * (y - a) / nt
        psi_b = (w * d * (mu0 - b) + w * (1 - d) * e / (1 - e) * (y - mu0)) / nt
        ests.append(np.log(a) - np.log(b))
        ses.append(cluster_se(psi_a / a - psi_b / b, g, 1.0))
    if not ests:
        return float("nan"), float("nan")
    return median_combine(ests, ses)


def result_row(label, est, se, df, estimand=""):
    return {
        "specification": label,
        "estimand": estimand,
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "pct_change": np.expm1(est),
        "n": len(df),
        "games": df["app_id"].nunique(),
    }
