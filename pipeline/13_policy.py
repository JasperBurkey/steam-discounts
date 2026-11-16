"""Step 13 (batch 9): policy learning. Which games should discount, and how deep?

Doubly robust scores Gamma[i, a] for each depth group a (none, 1-33%, 34-60%, 61%+) come
from the multi-arm estimator of step 9: they estimate what game-event i would get under
each choice. A policy assigns one depth to each game-event from features a publisher knows
before the sale; its value is the mean of Gamma at the assigned depth (Athey and Wager
2021; multi-action trees: Zhou, Athey and Wager 2023).

Policy class: exhaustive depth-1 and depth-2 trees over POLICY_FEATURES (interpretable,
known before the sale), every leaf at least MIN_LEAF of the rows. Fixed before any result:
  objectives   net revenue proxy (r_net) and its lag-adjusted version (r_net_lag), which
               bracket the review-lag problem; a rule counts as robust only if it beats the
               status quo under BOTH. Reviews during the sale are reported as a side effect.
  evaluation   cross-fitted over FOLDS game folds: trees learned on the other folds, value
               measured on held-out games, SEs clustered by game.
  baselines    status quo (the depth actually chosen), never discount, each fixed depth,
               the best fixed depth (chosen on the training folds).
Overlap: a policy may only assign a depth that similar games sometimes choose (propensity
>= LOW_PROPENSITY); where it would not, the game keeps the depth it actually chose. Without
this, trees recommend choices almost no comparable game made, and their value is
extrapolation.
Values are log changes relative to each game's own pre-sale baseline; a gain of 0.05 is
roughly +5%. Effects are for one game changing its choice with rivals' choices held fixed
(no evidence of spillovers: batch 8), not for all publishers changing at once.

  python 13_policy.py
Outputs (output/policy/): policy_values.csv, policy_trees.txt, policy_leaves.csv,
  policy_assignments.csv, overlap.csv
"""

import time

import config
import dml_core as M
import numpy as np
import pandas as pd

OUT = config.OUTPUT / "policy"
ARMS = config.DEPTH_ARMS  # ["none", "1-33%", "34-60%", "61%+"]
OBJECTIVES = {
    "r_net": "Revenue proxy, net",
    "r_net_lag": "Revenue proxy, net, lag-adjusted",
    "r_cal_during": "Chart-calibrated revenue, during (sensitivity)",
}
# Steam's revenue charts weigh a log-unit price cut at 0.072 / 0.534 = 0.135 of a log-unit
# review change (rank validation, batch 10). The review proxy weighs them equally, so it
# likely overstates what discounting costs. This sensitivity objective uses the charts' weight
# (during the sale only; from big games and imprecise, so an illustration).
CHART_PRICE_WEIGHT = 0.072 / 0.534
SIDE = {"y_during": "Reviews, during"}
POLICY_FEATURES = {
    "pre_rate_28": "reviews/day before the sale",
    "age_days": "age (days)",
    "regular_usd": "regular price (USD)",
    "pct_pos_60": "% positive, last 60 days",
    "share_days_disc_365": "share of last year on discount",
    "mean_cut_365": "mean discount depth, last year",
    "days_since_discount": "days since last discount",
    "prev_cut": "discount in the previous sale",
}
QUANTILES = np.linspace(0.05, 0.95, 19)
MIN_LEAF = 0.05
FOLDS = 5
LOW_PROPENSITY = 0.05


# ---------------------------------------------------------------- trees
def policy_matrix(df):
    """Raw policy features (median-filled), for readable thresholds."""
    x = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce") for c in POLICY_FEATURES}
    )
    return x.fillna(x.median()).to_numpy(float)


def thresholds(x):
    return [np.unique(np.quantile(x[:, j], QUANTILES)) for j in range(x.shape[1])]


def best_leaf(gamma):
    s = gamma.sum(0)
    return float(s.max()), int(s.argmax())


def best_depth1(x, gamma, thr, min_n):
    """Best single split (or no split) maximizing the summed score. Returns (value, node)."""
    best = (*best_leaf(gamma),)
    node = {"leaf": best[1]}
    value = best[0]
    total = gamma.sum(0)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        xs = x[order, j]
        cum = np.vstack([np.zeros(gamma.shape[1]), np.cumsum(gamma[order], 0)])
        for t in thr[j]:
            k = int(np.searchsorted(xs, t, side="right"))  # left: x <= t
            if k < min_n or len(xs) - k < min_n:
                continue
            left, right = cum[k], total - cum[k]
            v = left.max() + right.max()
            if v > value + 1e-12:
                value = v
                node = {
                    "feature": j,
                    "threshold": float(t),
                    "left": {"leaf": int(left.argmax())},
                    "right": {"leaf": int(right.argmax())},
                }
    return value, node


def best_depth2(x, gamma, thr, min_n):
    value, node = best_depth1(x, gamma, thr, min_n)
    for j in range(x.shape[1]):
        for t in thr[j]:
            m = x[:, j] <= t
            if m.sum() < 2 * min_n or (~m).sum() < 2 * min_n:
                continue
            vl, nl = best_depth1(x[m], gamma[m], thr, min_n)
            vr, nr = best_depth1(x[~m], gamma[~m], thr, min_n)
            if vl + vr > value + 1e-12:
                value = vl + vr
                node = {"feature": j, "threshold": float(t), "left": nl, "right": nr}
    return value, node


def fit_tree(x, gamma, depth):
    thr = thresholds(x)
    min_n = int(np.ceil(MIN_LEAF * len(x)))
    if depth == 1:
        return best_depth1(x, gamma, thr, min_n)[1]
    return best_depth2(x, gamma, thr, min_n)[1]


def predict(node, x):
    if "leaf" in node:
        return np.full(len(x), node["leaf"])
    m = x[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(x), int)
    out[m] = predict(node["left"], x[m])
    out[~m] = predict(node["right"], x[~m])
    return out


def describe(node, indent=""):
    names = list(POLICY_FEATURES.values())
    if "leaf" in node:
        return f"{indent}-> {ARMS[node['leaf']]}\n"
    f, t = names[node["feature"]], node["threshold"]
    return (
        f"{indent}if {f} <= {t:.4g}:\n"
        + describe(node["left"], indent + "    ")
        + f"{indent}else:\n"
        + describe(node["right"], indent + "    ")
    )


def leaf_ids(node, x, prefix=""):
    if "leaf" in node:
        return np.full(len(x), prefix or "all", dtype=object)
    m = x[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(x), dtype=object)
    out[m] = leaf_ids(node["left"], x[m], prefix + "L")
    out[~m] = leaf_ids(node["right"], x[~m], prefix + "R")
    return out


# ---------------------------------------------------------------- evaluation
def value_rows(gamma, assign, groups, label, objective, sample, status_quo):
    """Mean score under a policy, and its gain over the status quo (clustered SEs)."""
    v = gamma[np.arange(len(gamma)), assign]
    d = v - status_quo
    n = len(v)
    return {
        "sample": sample,
        "objective": objective,
        "policy": label,
        "value": v.mean(),
        "value_se": M.cluster_se(v - v.mean(), groups, n),
        "gain_vs_status_quo": d.mean(),
        "gain_se": M.cluster_se(d - d.mean(), groups, n),
        **{f"share_{a}": float((assign == k).mean()) for k, a in enumerate(ARMS)},
    }


def crossfit_policies(x, gamma, groups, labels):
    """Out-of-fold assignments for each learned policy."""
    folds = M.game_folds(groups, config.SEED + 100)
    out = {
        k: np.empty(len(x), int)
        for k in ("best fixed depth", "tree, 1 split", "tree, 2 levels")
    }
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        out["best fixed depth"][te] = best_leaf(gamma[tr])[1]
        out["tree, 1 split"][te] = predict(fit_tree(x[tr], gamma[tr], 1), x[te])
        out["tree, 2 levels"][te] = predict(fit_tree(x[tr], gamma[tr], 2), x[te])
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(config.EVENTS_PARQUET)
    ev["r_cal_during"] = ev["y_during"] + CHART_PRICE_WEIGHT * np.log(
        ev["price_ratio_sale"]
    )
    base = ev[ev["analysis_sample"]].dropna(
        subset=[
            "y_during",
            "r_net",
            "r_net_lag",
            "r_cal_during",
            "pre_trend",
            "depth_bin",
        ]
    )
    samples = {
        "all game-events": base.reset_index(drop=True),
        "busier games": base[base["base_rate"] >= config.BUSY_MIN].reset_index(
            drop=True
        ),
    }
    values, leaves, assigns, overlap = [], [], [], []
    trees_txt = []
    for sname, df in samples.items():
        g = df["app_id"].to_numpy()
        lab = df["depth_bin"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
        xp = policy_matrix(df)
        e = np.clip(
            M.crossfit_multiclass(M.features(df), lab, M.game_folds(g, config.SEED)),
            1e-6,
            1,
        )
        print(
            f"\n=== {sname}: {len(df):,} game-events, {len(np.unique(g)):,} games",
            flush=True,
        )
        scores = {}
        for yc in list(OBJECTIVES) + list(SIDE):
            scores[yc] = M.multiarm(df, yc, "depth_bin", ARMS, "none")[3]
            print(
                f"  scores for {yc} done  [{(time.time() - t0) / 60:.0f} min]",
                flush=True,
            )
        n = np.arange(len(df))
        ok = e >= LOW_PROPENSITY
        ok[n, lab] = True  # the depth actually chosen is always allowed

        def effective(a):
            """Assigned depth where supported; otherwise the depth actually chosen."""
            return np.where(ok[n, a], a, lab)

        for yc, olab in OBJECTIVES.items():
            raw = scores[yc]
            sq = raw[n, lab]
            gamma = np.where(
                ok, raw, sq[:, None]
            )  # unsupported choices score as status quo
            pol = {
                "status quo": lab,
                **{
                    f"always {a}" if a != "none" else "never discount": np.full(
                        len(df), k
                    )
                    for k, a in enumerate(ARMS)
                },
                **crossfit_policies(xp, gamma, g, lab),
            }
            for pname, a0 in pol.items():
                a = effective(a0)
                r = value_rows(gamma, a, g, pname, olab, sname, sq)
                r["share_kept_for_overlap"] = float((a != a0).mean())
                # side effect on reviews during the sale, same assignment
                side = scores["y_during"]
                sv = side[np.arange(len(df)), a] - side[np.arange(len(df)), lab]
                r["reviews_vs_status_quo"] = sv.mean()
                r["reviews_se"] = M.cluster_se(sv - sv.mean(), g, len(sv))
                values.append(r)
                overlap.append(
                    {
                        "sample": sname,
                        "objective": olab,
                        "policy": pname,
                        "share_unsupported_before_fallback": float((~ok[n, a0]).mean()),
                    }
                )
            # the tree on all data, for interpretation
            tree = fit_tree(xp, gamma, 2)
            trees_txt.append(f"[{sname}] objective: {olab}\n{describe(tree)}")
            ids = leaf_ids(tree, xp)
            assigns.append(
                pd.DataFrame(
                    {
                        "app_id": g,
                        "sale": df["sale"],
                        "sample": sname,
                        "objective": olab,
                        "leaf": ids,
                        "assigned": [ARMS[k] for k in effective(predict(tree, xp))],
                        "chosen": df["depth_bin"].to_numpy(),
                    }
                )
            )
            for leaf in sorted(set(ids)):
                m = ids == leaf
                row = {
                    "sample": sname,
                    "objective": olab,
                    "leaf": leaf,
                    "n": int(m.sum()),
                    "games": int(len(np.unique(g[m]))),
                    "assigned": ARMS[int(predict(tree, xp[m][:1])[0])],
                }
                k0 = int(predict(tree, xp[m][:1])[0])
                row["share_can_follow"] = float(ok[n[m], k0].mean())
                for k, a in enumerate(ARMS[1:], 1):
                    d = (
                        raw[m, k] - raw[m, 0]
                    )  # raw scores: the estimated effect in this leaf
                    row[f"{a} vs none"] = d.mean()
                    row[f"{a} vs none se"] = M.cluster_se(d - d.mean(), g[m], m.sum())
                    row[f"share chose {a}"] = float((lab[m] == k).mean())
                row["share chose none"] = float((lab[m] == 0).mean())
                leaves.append(row)
        print(f"  policies done  [{(time.time() - t0) / 60:.0f} min]", flush=True)

    values = pd.DataFrame(values)
    values.to_csv(OUT / "policy_values.csv", index=False)
    pd.DataFrame(leaves).to_csv(OUT / "policy_leaves.csv", index=False)
    pd.concat(assigns).to_csv(OUT / "policy_assignments.csv", index=False)
    pd.DataFrame(overlap).to_csv(OUT / "overlap.csv", index=False)
    (OUT / "policy_trees.txt").write_text("\n".join(trees_txt))

    pd.set_option("display.width", 220)
    show = values.copy()
    show["gain (se)"] = show.apply(
        lambda r: f"{r['gain_vs_status_quo']:+.3f} ({r['gain_se']:.3f})", axis=1
    )
    show["reviews (se)"] = show.apply(
        lambda r: f"{r['reviews_vs_status_quo']:+.3f} ({r['reviews_se']:.3f})", axis=1
    )
    cols = (
        ["sample", "objective", "policy", "gain (se)", "reviews (se)"]
        + [f"share_{a}" for a in ARMS]
        + ["share_kept_for_overlap"]
    )
    print("\nPolicy value vs. the status quo (held-out games for learned policies):")
    print(show[cols].round(2).to_string(index=False))
    print("\nTrees fitted on all data:\n" + "\n".join(trees_txt))
    print(pd.DataFrame(leaves).round(3).to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes. Outputs in {OUT}")


if __name__ == "__main__":
    main()
