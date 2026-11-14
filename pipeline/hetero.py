"""Heterogeneous effects (Athey and Imbens 2019, sec. 6.3).

CATE proxy: an HONEST forest on doubly robust scores. Each tree draws a subsample of games,
learns its splits on one half and estimates leaf effects (mean score) on the other half,
the honesty principle of causal forests (Wager and Athey 2018; Athey, Tibshirani and Wager
2019). Built from scikit-learn trees, so nothing extra to install.

Inference: generic machine-learning inference (Chernozhukov, Demirer, Duflo and
Fernandez-Val 2018), which the paper recommends for summarizing forests. Repeatedly split
the GAMES in half; fit the forest on one half; on the other half compute
  BLP    regress the scores on (1, S - mean S): beta1 = average effect, beta2 > 0 means the
         forest's predicted differences are real (heterogeneity detected)
  GATES  average effect within quintiles of the predicted effect S
  CLAN   average characteristics of the most and least affected quintiles
and report medians across splits, with p-values = 2 x median p (their adjustment).
Valid whatever the forest's quality: a poor forest just finds less heterogeneity.
"""

import config
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor


class HonestForest:
    def __init__(
        self, n_trees=300, min_leaf=50, max_features=0.33, subsample=0.5, seed=0
    ):
        self.n_trees, self.min_leaf, self.max_features = n_trees, min_leaf, max_features
        self.subsample, self.seed = subsample, seed

    def fit(self, x, phi, groups):
        rng = np.random.default_rng(self.seed)
        games = np.unique(groups)
        self.trees, imp = [], np.zeros(x.shape[1])
        for t in range(self.n_trees):
            pick = rng.choice(games, int(len(games) * self.subsample), replace=False)
            half = rng.permutation(pick)
            g_split, g_est = set(half[: len(half) // 2]), set(half[len(half) // 2 :])
            i_s = np.fromiter((g in g_split for g in groups), bool, len(groups))
            i_e = np.fromiter((g in g_est for g in groups), bool, len(groups))
            tree = DecisionTreeRegressor(
                min_samples_leaf=self.min_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(1e9)),
            ).fit(x[i_s], phi[i_s])
            leaf_e = tree.apply(x[i_e])
            vals = pd.Series(phi[i_e]).groupby(leaf_e).mean()  # honest leaf effects
            self.trees.append((tree, vals.to_dict(), float(phi[i_e].mean())))
            imp += tree.feature_importances_
        self.importance_ = imp / self.n_trees
        return self

    def predict(self, x):
        out = np.zeros(len(x))
        for tree, vals, fallback in self.trees:
            leaves = tree.apply(x)
            out += np.array([vals.get(lf, fallback) for lf in leaves])
        return out / len(self.trees)


def _cluster_ols(y, X, groups):
    """OLS with standard errors clustered by game."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    s = pd.DataFrame(X * u[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
    v = xtx_inv @ (s.T @ s) @ xtx_inv
    return beta, np.sqrt(np.diag(v))


def _mean_diff(a, b, ga, gb):
    """Difference in means of a and b (two groups of rows), clustered by game."""
    d = a.mean() - b.mean()
    sa = pd.Series(a - a.mean()).groupby(ga).sum().to_numpy() / len(a)
    sb = pd.Series(b - b.mean()).groupby(gb).sum().to_numpy() / len(b)
    return d, float(np.sqrt((sa**2).sum() + (sb**2).sum()))


def generic_ml(x, phi, groups, clan_df, n_splits=None, n_trees=None, seed=0):
    """Repeated game-level splits: forest on one half, BLP/GATES/CLAN on the other."""
    n_splits = n_splits or config.HET_SPLITS
    n_trees = n_trees or config.HET_TREES
    rng = np.random.default_rng(seed)
    games = np.unique(groups)
    blp, gates, clan, imps = [], [], [], []
    for s in range(n_splits):
        half = set(rng.choice(games, len(games) // 2, replace=False))
        a = np.fromiter((g in half for g in groups), bool, len(groups))
        b = ~a
        forest = HonestForest(n_trees=n_trees, seed=seed + s).fit(
            x[a], phi[a], groups[a]
        )
        imps.append(forest.importance_)
        sb = forest.predict(x[b])
        yb, gb = phi[b], groups[b]
        (b1, b2), (se1, se2) = _cluster_ols(
            yb, np.column_stack([np.ones(b.sum()), sb - sb.mean()]), gb
        )
        blp.append(
            {
                "split": s,
                "ate": b1,
                "ate_se": se1,
                "het_loading": b2,
                "het_se": se2,
                "het_p": float(stats.norm.sf(b2 / se2)) if se2 > 0 else np.nan,
            }
        )  # one-sided: beta2 > 0
        q = pd.qcut(pd.Series(sb).rank(method="first"), 5, labels=False).to_numpy()
        for k in range(5):
            m = q == k
            est = yb[m].mean()
            se = float(
                np.sqrt((pd.Series(yb[m] - est).groupby(gb[m]).sum() ** 2).sum())
                / m.sum()
            )
            gates.append({"split": s, "quintile": k + 1, "effect": est, "se": se})
        d, se = _mean_diff(yb[q == 4], yb[q == 0], gb[q == 4], gb[q == 0])
        gates.append({"split": s, "quintile": "top - bottom", "effect": d, "se": se})
        cb = clan_df[b].reset_index(drop=True)
        for col in cb.columns:
            v = pd.to_numeric(cb[col], errors="coerce").to_numpy(float)
            top, bot = v[q == 4], v[q == 0]
            ok_t, ok_b = ~np.isnan(top), ~np.isnan(bot)
            if ok_t.sum() < 10 or ok_b.sum() < 10:
                continue
            dd, sse = _mean_diff(
                top[ok_t], bot[ok_b], gb[q == 4][ok_t], gb[q == 0][ok_b]
            )
            clan.append(
                {
                    "split": s,
                    "characteristic": col,
                    "most_affected": top[ok_t].mean(),
                    "least_affected": bot[ok_b].mean(),
                    "difference": dd,
                    "se": sse,
                }
            )
    return _summarize(
        pd.DataFrame(blp), pd.DataFrame(gates), pd.DataFrame(clan)
    ), np.mean(imps, axis=0)


def _summarize(blp, gates, clan):
    """Medians across splits; p-values = 2 x median p (Chernozhukov et al.), capped at 1."""

    def two_sided_p(est, se):
        return 2 * stats.norm.sf(np.abs(est / se))

    b = {
        "ate": blp["ate"].median(),
        "ate_se": blp["ate_se"].median(),
        "het_loading": blp["het_loading"].median(),
        "het_se": blp["het_se"].median(),
        "het_p_adjusted": min(1.0, 2 * blp["het_p"].median()),
        "splits": len(blp),
    }
    g = (
        gates.assign(p=lambda t: two_sided_p(t["effect"], t["se"]))
        .groupby("quintile", sort=False)
        .agg(effect=("effect", "median"), se=("se", "median"), p=("p", "median"))
    )
    g["p_adjusted"] = (2 * g["p"]).clip(upper=1)
    c = (
        clan.assign(p=lambda t: two_sided_p(t["difference"], t["se"]))
        .groupby("characteristic", sort=False)
        .agg(
            most_affected=("most_affected", "median"),
            least_affected=("least_affected", "median"),
            difference=("difference", "median"),
            se=("se", "median"),
            p=("p", "median"),
        )
    )
    c["p_adjusted"] = (2 * c["p"]).clip(upper=1)
    return b, g.drop(columns="p"), c.drop(columns="p")
