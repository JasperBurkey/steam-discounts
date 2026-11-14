"""The notebook's analyses, run with the pipeline's own estimation code (pipeline/dml_core.py
and pipeline/hetero.py), so the notebook and the pipeline cannot drift apart.

Settings come from environment variables read by pipeline/config.py:
  LEARNER     "ensemble" (LASSO + random forest + boosting; the thesis results) or "boost"
  CF_REPEATS  cross-fitting repetitions (3 for the thesis results)
  HET_SPLITS, HET_TREES  sample splits and trees for the heterogeneity analysis (25, 200)
GitHub Actions uses quick settings (boost, 1, 5, 50) to keep the run short; the README says
how to reproduce the full results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import config  # noqa: E402
import dml_core as M  # noqa: E402
from hetero import generic_ml  # noqa: E402

OUTCOMES = {
    "y_during": "Reviews, during the sale",
    "r_during": "Revenue proxy, during the sale",
    "y_net": "Reviews, net (7 days before to 4 weeks after)",
    "r_net": "Revenue proxy, net (headline)",
}


def settings_note():
    full = (
        config.LEARNER == "ensemble"
        and config.CF_REPEATS >= 3
        and config.HET_SPLITS >= 25
    )
    kind = (
        "full thesis settings"
        if full
        else "QUICK settings (estimates close to, not identical with, the thesis)"
    )
    return (
        f"Learner: {config.LEARNER}; cross-fitting repetitions: {config.CF_REPEATS}; "
        f"heterogeneity splits/trees: {config.HET_SPLITS}/{config.HET_TREES} -> {kind}"
    )


def _row(label, estimand, est, se, df):
    return {
        "result": label,
        "estimand": estimand,
        "estimate": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "n": len(df),
        "games": df["app_id"].nunique(),
    }


def tidy(table):
    """Readable version: log estimate (se), approximate % change and 95% CI in %."""
    t = table.copy()
    t["estimate (se)"] = t.apply(
        lambda r: f"{r['estimate']:+.3f} ({r['se']:.3f})", axis=1
    )
    t["~% change"] = t["estimate"].map(lambda v: f"{np.expm1(v):+.0%}")
    t["95% CI, % change"] = t.apply(
        lambda r: f"{np.expm1(r['ci_low']):+.0%} to {np.expm1(r['ci_high']):+.0%}",
        axis=1,
    )
    cols = ["result", "estimand", "estimate (se)", "~% change", "95% CI, % change"]
    return t[cols + ["n", "games"]]


def main_estimates(main_df, long_df):
    """ATT (headline), ATE, ATU, partially linear model and residual balancing, per
    outcome. Returns (table, propensity scores for main_df)."""
    rows, e_main = [], None
    for df, outs in [
        (main_df, ["y_during", "r_during"]),
        (long_df, ["y_net", "r_net"]),
    ]:
        res, _, e = M.aipw(df, outs)
        if e_main is None:
            e_main = e
        for yc in outs:
            for est in ["ATT", "ATE", "ATU"]:
                rows.append(_row(OUTCOMES[yc], est, *res[yc][est], df))
            rows.append(_row(OUTCOMES[yc], "PLR", *M.plr(df, yc, "discounted"), df))
            att_b, se_b, _, _ = M.residual_balancing_att(df, yc)
            rows.append(_row(OUTCOMES[yc], "ATT, residual balancing", att_b, se_b, df))
    return pd.DataFrame(rows), e_main


def robustness(main_df, long_df, e_main):
    rows = []
    busy = long_df[long_df["base_rate"] >= config.BUSY_MIN].reset_index(drop=True)
    r, _, _ = M.aipw(busy, ["y_during", "r_net"], repeats=1)
    rows.append(
        _row(
            "Busier games (>= 1 review/day): reviews, during",
            "ATT",
            *r["y_during"]["ATT"],
            busy,
        )
    )
    rows.append(
        _row("Busier games: revenue proxy, net", "ATT", *r["r_net"]["ATT"], busy)
    )
    for col, lab in [
        ("lvl_during", "Ratio of mean counts: reviews, during"),
        ("lvl_r_net", "Ratio of mean counts: revenue proxy, net"),
    ]:
        sub = long_df.dropna(subset=[col]).reset_index(drop=True)
        rows.append(_row(lab, "ATT (ratio)", *M.ratio_att(sub, col, repeats=1), sub))
    winters = ["Winter Sale 2023", "Winter Sale 2024"]
    no_w = long_df[~long_df["sale"].isin(winters)].reset_index(drop=True)
    r, _, _ = M.aipw(no_w, ["r_net"], repeats=1)
    rows.append(
        _row(
            "Without Winter 2023 and 2024: revenue proxy, net",
            "ATT",
            *r["r_net"]["ATT"],
            no_w,
        )
    )
    lo, hi = config.OVERLAP_TRIM
    trim = main_df[(e_main >= lo) & (e_main <= hi)].reset_index(drop=True)
    r, _, _ = M.aipw(trim, ["y_during"], repeats=1)
    rows.append(
        _row(
            f"Overlap-trimmed (propensity {lo}-{hi}): reviews, during",
            "ATE",
            *r["y_during"]["ATE"],
            trim,
        )
    )
    strict = main_df[~main_df["pre_discount_strict"]].reset_index(drop=True)
    rows.append(
        _row(
            "Stricter exclusion (no discount in 30 days before): reviews, during",
            "PLR",
            *M.plr(strict, "y_during", "discounted", repeats=1),
            strict,
        )
    )
    return pd.DataFrame(rows)


def by_size(long_df):
    """Within each size band: ratio estimates (no compression) and log-scale ATT."""
    rows = []
    for band in ["small", "mid", "large", "very_large"]:
        sub = long_df[long_df["band"] == band].reset_index(drop=True)
        if sub["discounted"].sum() < 30 or (~sub["discounted"]).sum() < 30:
            continue
        rows.append(
            _row(
                f"{band}: reviews, during",
                "ATT (ratio)",
                *M.ratio_att(sub, "lvl_during", repeats=1),
                sub,
            )
        )
        r, _, _ = M.aipw(sub, ["y_during", "r_net"], repeats=1)
        rows.append(
            _row(f"{band}: reviews, during", "ATT (log)", *r["y_during"]["ATT"], sub)
        )
        rows.append(
            _row(f"{band}: revenue proxy, net", "ATT (log)", *r["r_net"]["ATT"], sub)
        )
    return pd.DataFrame(rows)


CLAN_COLS = [
    "age_days",
    "regular_usd",
    "pre_rate_28",
    "pre_trend",
    "days_since_discount",
    "disc_starts_365",
    "mean_cut_365",
    "cut",
    "pct_pos_60",
]


def heterogeneity(main_df):
    """Honest forest + BLP/GATES/CLAN on busier games, overall and in the overlap region."""
    df = main_df[main_df["base_rate"] >= config.BUSY_MIN]
    df = df.dropna(subset=["r_net"]).reset_index(drop=True)
    _, phi, e = M.aipw(df, ["y_during", "r_net"])
    x = M.features(df)
    g = df["app_id"].to_numpy()
    clan_df = df[[c for c in CLAN_COLS if c in df]].assign(propensity=e)
    lo, hi = config.OVERLAP_TRIM
    inside = (e >= lo) & (e <= hi)
    blp, gates, clan = [], [], []
    for yc in ["y_during", "r_net"]:
        for sample, m in [
            ("busier games", np.ones(len(df), bool)),
            (f"overlap region ({lo}-{hi})", inside),
        ]:
            (b, gt, cl), _ = generic_ml(
                x[m], phi[yc][m], g[m], clan_df[m].reset_index(drop=True)
            )
            blp.append(
                {
                    "sample": sample,
                    "outcome": OUTCOMES[yc],
                    "game-events": int(m.sum()),
                    **b,
                }
            )
            gates.append(gt.reset_index().assign(sample=sample, outcome=OUTCOMES[yc]))
            clan.append(cl.reset_index().assign(sample=sample, outcome=OUTCOMES[yc]))
    cols = [
        "sample",
        "outcome",
        "game-events",
        "ate",
        "het_loading",
        "het_se",
        "het_p_adjusted",
    ]
    return (
        pd.DataFrame(blp)[cols].round(3),
        pd.concat(gates, ignore_index=True),
        pd.concat(clan, ignore_index=True),
    )


def depth(main_df):
    """Each depth versus not discounting: average over all game-events (ATE) and for the
    game-events that chose that depth (ATT); net revenue with and without the
    review-lag adjustment."""
    df = main_df.dropna(subset=["r_net", "r_net_lag", "depth_bin"]).reset_index(
        drop=True
    )
    rows = []
    for yc, lab in [
        ("y_during", "Reviews, during"),
        ("r_net", "Revenue proxy, net"),
        ("r_net_lag", "Revenue proxy, net, lag-adjusted"),
    ]:
        eff, _, _, _, att = M.multiarm(
            df, yc, "depth_bin", config.DEPTH_ARMS, "none", return_att=True
        )
        for a in config.DEPTH_ARMS[1:]:
            rows.append(
                {
                    "outcome": lab,
                    "depth": a,
                    "ATE (all games)": f"{np.expm1(eff[a][0]):+.0%}",
                    "ATT (games choosing this depth)": f"{np.expm1(att[a][0]):+.0%}",
                    "ATT se (log)": round(att[a][1], 3),
                }
            )
    return pd.DataFrame(rows)
