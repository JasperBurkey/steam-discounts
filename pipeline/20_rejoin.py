"""Step 20: do publishers rejoin sales because of the spike or the net gain?

Among games that discounted in a sale, regress whether they discount in the next seasonal
sale on what the sale did for them, measured two ways:
    r_during   revenue proxy during the sale (the visible spike)
    r_net      revenue proxy over the whole net window (pre-week, sale, 4 weeks after)
Both are changes against the game's own baseline. Holding r_net fixed, a positive weight on
r_during means a game whose gain was more concentrated in the sale window (same total,
bigger spike) is more likely to come back: publishers respond to the spike. A publisher
judging sales by their net effect would put no weight on r_during once r_net is held fixed.

Specifications (linear probability, SE clustered by game, regressors standardized):
  sale FE + controls; sale FE + game FE (within-game: a better-than-usual spike);
  window components (pre-week, during, weeks 1-2 after, weeks 3-4 after) instead of the
  two summaries; busier games only (less noise in the review proxy).
Also: next sale's depth among games that rejoin.

These are realized outcomes, not causal effects, and reviews are a noisy stand-in for what
the publisher sees. Noise is larger in the shorter sale window, which biases the weight on
r_during toward zero, so a positive weight is, if anything, understated.

  python 20_rejoin.py
Outputs: results/rejoin/
"""

import config
import numpy as np
import pandas as pd

OUT = config.ROOT.parent / "results" / "rejoin"
CONTROLS = [
    "pre_rate_60",
    "pre_trend",
    "pct_pos_60",
    "cut",
    "log_price",
    "log_age",
    "disc_starts_365",
    "share_days_disc_365",
    "prev_share_joined",
    "pub_prev_share",
]


def demean(X, groups, iters=50, tol=1e-10):
    """Remove fixed effects for each grouping by alternating projections."""
    X = X.copy()
    for _ in range(iters):
        old = X.copy()
        for g in groups:
            X -= pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
        if np.max(np.abs(X - old)) < tol:
            break
    return X


def ols(df, y, xs, fe, cluster="app_id"):
    cols = [y] + xs
    M = df[cols].to_numpy(float)
    M = demean(M, [df[f].to_numpy() for f in fe])
    Y, X = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    u = Y - X @ b
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in df.groupby(cluster).indices.items():
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G = df[cluster].nunique()
    n, k = X.shape
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (n - 1) / (n - k)
    return pd.DataFrame({"coef": b, "se": np.sqrt(np.diag(V))}, index=xs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(config.EVENTS_PARQUET).sort_values(["app_id", "event_start"])
    g = ev.groupby("app_id")
    ev["next_disc"] = g["discounted"].shift(-1)
    ev["next_cut"] = g["cut"].shift(-1)
    ev["next_known"] = g["price_known"].shift(-1)
    ev["log_price"] = np.log(ev["regular_usd"])
    ev["log_age"] = np.log1p(ev["age_days"])
    s = ev[
        (ev["discounted"] == 1)
        & ev["price_known"]
        & (ev["next_known"] == True)  # noqa: E712
        & ~ev["free"]
        & ~ev["too_new"]
    ].dropna(
        subset=["r_during", "r_net", "y_pre_week", "r_after", "y_wk3_4", "next_disc"]
        + CONTROLS
    )
    s = s.assign(next_disc=s["next_disc"].astype(float)).reset_index(drop=True)
    summ = ["r_during", "r_net"]
    comp = ["y_pre_week", "r_during", "r_after", "y_wk3_4"]
    for c in set(summ + comp + CONTROLS):
        s[c + "_z"] = (s[c] - s[c].mean()) / s[c].std()
    z = lambda cs: [c + "_z" for c in cs]  # noqa: E731
    print(
        f"{len(s):,} discounted game-events with a next sale; {s['app_id'].nunique():,} games; "
        f"{s['next_disc'].mean():.1%} rejoin. corr(r_during, r_net) = {s['r_during'].corr(s['r_net']):.2f}"
    )
    busy = s[s["base_rate"] >= config.BUSY_MIN]
    specs = [
        ("spike vs net", "all", s, z(summ), ["sale"]),
        ("spike vs net", "all, game FE", s, z(summ), ["sale", "app_id"]),
        ("spike only", "all, game FE", s, z(["r_during"]), ["sale", "app_id"]),
        ("net only", "all, game FE", s, z(["r_net"]), ["sale", "app_id"]),
        ("spike vs net", "busier", busy, z(summ), ["sale"]),
        ("spike vs net", "busier, game FE", busy, z(summ), ["sale", "app_id"]),
        ("windows", "all", s, z(comp), ["sale"]),
        ("windows", "all, game FE", s, z(comp), ["sale", "app_id"]),
    ]
    rows = []
    for name, sample, d, xs, fe in specs:
        ctrl = z(CONTROLS) if "app_id" not in fe else z(["pre_rate_60", "pre_trend", "cut"])
        r = ols(d, "next_disc", xs + ctrl, fe)
        for v in xs:
            rows.append(
                {
                    "outcome": "joins next sale",
                    "spec": name,
                    "sample": sample,
                    "term": v.removesuffix("_z"),
                    "coef": r.loc[v, "coef"],
                    "se": r.loc[v, "se"],
                    "n": len(d),
                }
            )
    j = s[s["next_disc"] == 1].dropna(subset=["next_cut"])
    for sample, fe in [("rejoiners", ["sale"]), ("rejoiners, game FE", ["sale", "app_id"])]:
        ctrl = z(CONTROLS) if "app_id" not in fe else z(["pre_rate_60", "pre_trend", "cut"])
        r = ols(j, "next_cut", z(summ) + ctrl, fe)
        for v in z(summ):
            rows.append(
                {
                    "outcome": "next sale's cut",
                    "spec": "spike vs net",
                    "sample": sample,
                    "term": v.removesuffix("_z"),
                    "coef": r.loc[v, "coef"],
                    "se": r.loc[v, "se"],
                    "n": len(j),
                }
            )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rejoin.csv", index=False)
    res["coef (se)"] = res.apply(lambda x: f"{x['coef']:+.4f} ({x['se']:.4f})", axis=1)
    pd.set_option("display.width", 200)
    print(res[["outcome", "spec", "sample", "term", "coef (se)", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
